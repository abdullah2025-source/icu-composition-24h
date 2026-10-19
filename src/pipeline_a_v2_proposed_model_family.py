#!/usr/bin/env python3
"""
Pipeline A V2 — Proposed Model Family (Fixed Screening)
========================================================

Six complete architecture variants using the SAME frozen V2 data contract,
patient-level splits, text embeddings, target masks, five-task loss, validation
selection criterion, subject-clustered evaluation, and test reservation.

Versions:
  Full-A — Current proposed Transformer + gate
  Full-B — Deeper/wider temporal Transformer + stronger temporal pooling
  Full-C — Structured temporal encoder + stronger multimodal text fusion
  Full-D — Task-specific representations / adapters
  Full-E — Gated residual multimodal fusion + task-specific adapters
  Full-F — Local temporal convolution + global Transformer

This file is intentionally a FAMILY rather than six separate scripts.
Run `--mode family-fixed` to train/evaluate all versions under the same
first-run fixed protocol. Test data are never used for checkpoint selection.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import optuna
except ImportError:
    optuna = None

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except Exception as exc:  # pragma: no cover
    torch = None
    nn = None
    F = None
    DataLoader = None
    Dataset = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


# ============================================================================
# 0. PATHS / CONSTANTS
# ============================================================================
PROJECT_ROOT = Path("/home/abdullah2025/fyp_project")
V2_DIR = PROJECT_ROOT / "outputs" / "pipeline_a_v2"
RESULTS_ROOT = PROJECT_ROOT / "outputs" / "proposed_model"
FAMILY_RESULTS_ROOT = PROJECT_ROOT / "outputs" / "proposed_model_family"

ALIGNMENT_PATH = V2_DIR / "a5_alignment_index_24h.csv"
STRUCTURED_PATH = V2_DIR / "structured_24h_processed.npy"
DEMOGRAPHICS_PATH = V2_DIR / "demographics_24h_processed.npy"
INPUT_METADATA_PATH = V2_DIR / "final_model_input_metadata_24h.csv"
INPUT_DEFINITION_PATH = V2_DIR / "final_model_input_definition_24h.json"
OUTCOMES_PATH = V2_DIR / "targets_24h" / "outcomes_24h_primary_cohort.csv"
TEXT_EMBEDDING_PATH = V2_DIR / "clinical_note_embeddings.parquet"

SEED = 20260923
EXPECTED_N = 74_572
EXPECTED_TRAIN = 52_268
EXPECTED_VAL = 7_417
EXPECTED_TEST = 14_887
EXPECTED_STRUCTURED_SHAPE = (EXPECTED_N, 24, 110)
EXPECTED_DEMO_SHAPE = (EXPECTED_N, 4)
EXPECTED_TEXT_DIM = 768
EXPECTED_TEXT_PRESENT = 46_968

BINARY_TASKS = (
    "mortality_24h",
    "ventilation_onset_24h",
    "vasopressor_onset_24h",
    "readmission_30d_24h",
)
PRIMARY_TASKS = BINARY_TASKS + ("los_hours_24h",)
REPORT_TASKS = PRIMARY_TASKS + ("deterioration_24h",)
VALID_COL = {
    "mortality_24h": "mortality_24h_valid",
    "ventilation_onset_24h": "ventilation_onset_24h_valid",
    "vasopressor_onset_24h": "vasopressor_onset_24h_valid",
    "readmission_30d_24h": "readmission_30d_24h_valid",
    "los_hours_24h": "los_hours_24h_valid",
    "deterioration_24h": "deterioration_24h_valid",
}

FIRST_RUN = {
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "batch_size": 256,
    "dropout": 0.15,
    "scheduler": "cosine",
    "grad_clip": 1.0,
    "patience": 8,
    "max_epochs": 50,
    "seed": SEED,
    "amp": "BF16",
}

# Benchmark reference prediction locations for post-test comparison.
BENCHMARK_TUNED_PATHS = {
    "00": PROJECT_ROOT / "outputs/baseline_family_01_tabular/tuned/00_XGBoost/final_selected_configuration/test_predictions.csv",
    "01": PROJECT_ROOT / "outputs/baseline_family_01_tabular/tuned/01_Logistic_Regression/final_selected_configuration/test_predictions.csv",
    "02": PROJECT_ROOT / "outputs/baseline_family_01_tabular/tuned/02_Regularized_Logistic_Regression/final_selected_configuration/test_predictions.csv",
    "03": PROJECT_ROOT / "outputs/baseline_family_01_tabular/tuned/03_Random_Forest/final_selected_configuration/test_predictions.csv",
    "04": PROJECT_ROOT / "outputs/baseline_family_01_tabular/tuned/04_Extra_Trees/final_selected_configuration/test_predictions.csv",
    "05": PROJECT_ROOT / "outputs/baseline_family_01_tabular/tuned/05_LightGBM/final_selected_configuration/test_predictions.csv",
    "06": PROJECT_ROOT / "outputs/baseline_family_02_rnn/tuned/06_GRU/final_selected_configuration/test_predictions.csv",
    "07": PROJECT_ROOT / "outputs/baseline_family_02_rnn/tuned/07_LSTM/final_selected_configuration/test_predictions.csv",
    "08": PROJECT_ROOT / "outputs/baseline_family_02_rnn/tuned/08_BiGRU/final_selected_configuration/test_predictions.csv",
    "09": PROJECT_ROOT / "outputs/baseline_family_02_rnn/tuned/09_GRU-D/final_selected_configuration/test_predictions.csv",
    "10": PROJECT_ROOT / "outputs/baseline_family_03_tcn_10_12/tuned/10_TCN/final_selected_configuration/test_predictions.csv",
    "11": PROJECT_ROOT / "outputs/baseline_family_03_tcn_10_12/tuned/11_ModernTCN/final_selected_configuration/test_predictions.csv",
    "12": PROJECT_ROOT / "outputs/baseline_family_03_tcn_10_12/tuned/12_CNN_+_GRU/final_selected_configuration/test_predictions.csv",
    "13": PROJECT_ROOT / "outputs/baseline_family_04_clinical_sequence_13_16/tuned/13_RETAIN/final_selected_configuration/test_predictions.csv",
    "14": PROJECT_ROOT / "outputs/baseline_family_04_clinical_sequence_13_16/tuned/14_AdaCare/final_selected_configuration/test_predictions.csv",
    "15": PROJECT_ROOT / "outputs/baseline_family_04_clinical_sequence_13_16/tuned/15_StageNet/final_selected_configuration/test_predictions.csv",
    "16": PROJECT_ROOT / "outputs/baseline_family_04_clinical_sequence_13_16/tuned/16_ConCare/final_selected_configuration/test_predictions.csv",
    "17": PROJECT_ROOT / "outputs/baseline_family_05_transformer_modern_17_23/tuned/17_Transformer/final_selected_configuration/test_predictions.csv",
    "18": PROJECT_ROOT / "outputs/baseline_family_05_transformer_modern_17_23/tuned/18_Transformer_+_Conv/final_selected_configuration/test_predictions.csv",
    "19": PROJECT_ROOT / "outputs/baseline_family_05_transformer_modern_17_23/tuned/19_Transformer_+_LSTM/final_selected_configuration/test_predictions.csv",
    "20": PROJECT_ROOT / "outputs/baseline_family_05_transformer_modern_17_23/tuned/20_TFT/final_selected_configuration/test_predictions.csv",
    "21": PROJECT_ROOT / "outputs/baseline_family_05_transformer_modern_17_23/tuned/21_PatchTST/final_selected_configuration/test_predictions.csv",
    "22": PROJECT_ROOT / "outputs/baseline_family_05_transformer_modern_17_23/tuned/22_TimesNet/final_selected_configuration/test_predictions.csv",
    "23": PROJECT_ROOT / "outputs/baseline_family_05_transformer_modern_17_23/tuned/23_iTransformer/final_selected_configuration/test_predictions.csv",
    "24": PROJECT_ROOT / "outputs/baseline_family_06_state_space_24_27/tuned/24_S4D/final_selected_configuration/test_predictions.csv",
    "25": PROJECT_ROOT / "outputs/baseline_family_06_state_space_24_27/tuned/25_Mamba-1/final_selected_configuration/test_predictions.csv",
    "26": PROJECT_ROOT / "outputs/baseline_family_06_state_space_24_27/tuned/26_Mamba-2/final_selected_configuration/test_predictions.csv",
    "27": PROJECT_ROOT / "outputs/baseline_family_06_state_space_24_27/tuned/27_Mamba-3/final_selected_configuration/test_predictions.csv",
    "28": PROJECT_ROOT / "outputs/baseline_family_07_advanced_multimodal_28_32/tuned/28_Hybrid_Mamba-Transformer/final_selected_configuration/test_predictions.csv",
    "29": PROJECT_ROOT / "outputs/baseline_family_07_advanced_multimodal_28_32/tuned/29_Multimodal_Progression/final_selected_configuration/test_predictions.csv",
    "30": PROJECT_ROOT / "outputs/baseline_family_07_advanced_multimodal_28_32/tuned/30_Cross-Attention_Multimodal/final_selected_configuration/test_predictions.csv",
    "31": PROJECT_ROOT / "outputs/baseline_family_07_advanced_multimodal_28_32/tuned/31_Gated_Fusion_Cross-Attn/final_selected_configuration/test_predictions.csv",
    "32": PROJECT_ROOT / "outputs/baseline_family_07_advanced_multimodal_28_32/tuned/32_MoE_Modality_Fusion/final_selected_configuration/test_predictions.csv",
}

BENCHMARK_NAMES = {
    "00": "XGBoost", "01": "Logistic Regression", "02": "Regularized Logistic Regression",
    "03": "Random Forest", "04": "Extra Trees", "05": "LightGBM",
    "06": "GRU", "07": "LSTM", "08": "BiGRU", "09": "GRU-D",
    "10": "TCN", "11": "ModernTCN", "12": "CNN+GRU", "13": "RETAIN",
    "14": "AdaCare", "15": "StageNet", "16": "ConCare", "17": "Transformer",
    "18": "Transformer+Conv", "19": "Transformer+LSTM", "20": "TFT", "21": "PatchTST",
    "22": "TimesNet", "23": "iTransformer", "24": "S4D", "25": "Mamba-1",
    "26": "Mamba-2", "27": "Mamba-3", "28": "Hybrid Mamba-Transformer",
    "29": "Multimodal Progression", "30": "Cross-Attention Multimodal",
    "31": "Gated Fusion/Cross-Attn", "32": "MoE Modality Fusion",
}

# Fixed candidate comparisons emphasized in the prior audit.
DEFAULT_CANDIDATE_REFS = ["32", "00", "05", "31", "13", "24"]


@dataclass
class FrozenData:
    structured: np.ndarray
    demographics: np.ndarray
    text: np.ndarray
    text_missing: np.ndarray
    stay_id: np.ndarray
    subject_id: np.ndarray
    split: np.ndarray
    outcomes: pd.DataFrame
    targets: Dict[str, np.ndarray]
    valid_arrays: Dict[str, np.ndarray]
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


@dataclass
class AblationSpec:
    name: str
    use_text: bool = True
    use_text_missing_flag: bool = True
    use_demographics: bool = True
    use_attention_pool: bool = True
    use_last_hour: bool = True
    use_gate: bool = True
    use_uncertainty_weighting: bool = True
    use_los_mixture: bool = True


ABLATIONS = [
    AblationSpec("full"),
    AblationSpec("no_text", use_text=False),
    AblationSpec("no_text_missing_flag", use_text_missing_flag=False),
    AblationSpec("no_demographics", use_demographics=False),
    AblationSpec("no_attention_pool", use_attention_pool=False),
    AblationSpec("no_last_hour", use_last_hour=False),
    AblationSpec("no_gate", use_gate=False),
    AblationSpec("equal_task_weights", use_uncertainty_weighting=False),
    AblationSpec("single_los_expert", use_los_mixture=False),
]


# ============================================================================
# 1. REPRODUCIBILITY / RUNTIME
# ============================================================================
def stable_seed(label: str) -> int:
    """Deterministic 32-bit seed derived from the global seed and a label."""
    digest = hashlib.sha256(f"{SEED}|{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32 - 1)


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


def require_torch() -> None:
    if _TORCH_IMPORT_ERROR is not None:
        raise RuntimeError(f"PyTorch import failed: {_TORCH_IMPORT_ERROR}")


def require_optuna() -> None:
    if optuna is None:
        raise RuntimeError("Optuna is required for --mode tuned.")


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required frozen V2 artifact not found: {path}")


def get_device() -> torch.device:
    require_torch()
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def validate_gpu_bf16(device: torch.device, context: str = "training") -> None:
    require_torch()
    if device.type != "cuda":
        raise RuntimeError(
            f"Proposed-model {context} requires a CUDA/ROCm GPU with BF16 support; got {device}."
        )
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Proposed model requires BF16-capable GPU training; torch.cuda.is_bf16_supported() is False.")


def autocast_context(device: torch.device):
    enabled = device.type == "cuda"
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=enabled)


# ============================================================================
# 2. FROZEN V2 DATA LOADING
# ============================================================================
def load_text_embeddings(stay_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    require_file(TEXT_EMBEDDING_PATH)
    emb = pd.read_parquet(TEXT_EMBEDDING_PATH)
    required = {"stay_id", "text_embedding"}
    missing = sorted(required - set(emb.columns))
    if missing:
        raise KeyError(f"Text parquet missing columns: {missing}")
    if emb["stay_id"].duplicated().any():
        raise ValueError("clinical_note_embeddings.parquet contains duplicate stay_id rows.")

    row_of = pd.Series(np.arange(len(stay_ids), dtype=np.int64), index=stay_ids)
    present = emb[emb["stay_id"].isin(row_of.index)].copy()
    text = np.zeros((len(stay_ids), EXPECTED_TEXT_DIM), dtype=np.float32)
    text_missing = np.ones(len(stay_ids), dtype=np.float32)
    if len(present):
        rows = row_of.loc[present["stay_id"]].to_numpy(dtype=np.int64)
        vectors = []
        for vec in present["text_embedding"].tolist():
            arr = np.asarray(vec, dtype=np.float32).reshape(-1)
            if arr.size != EXPECTED_TEXT_DIM:
                raise ValueError(f"Expected text embedding dim {EXPECTED_TEXT_DIM}, found {arr.size}.")
            vectors.append(arr)
        text[rows] = np.stack(vectors).astype(np.float32, copy=False)
        text_missing[rows] = 0.0

    matched = int((text_missing == 0).sum())
    if matched != EXPECTED_TEXT_PRESENT:
        raise ValueError(f"Expected {EXPECTED_TEXT_PRESENT} text-bearing stays; found {matched}.")
    if not np.isfinite(text).all():
        raise ValueError("Aligned text embeddings contain non-finite values.")
    return text, text_missing


def load_frozen_v2() -> FrozenData:
    for p in [
        ALIGNMENT_PATH,
        STRUCTURED_PATH,
        DEMOGRAPHICS_PATH,
        INPUT_METADATA_PATH,
        INPUT_DEFINITION_PATH,
        OUTCOMES_PATH,
        TEXT_EMBEDDING_PATH,
    ]:
        require_file(p)

    idx_df = pd.read_csv(ALIGNMENT_PATH)
    outcomes_df = pd.read_csv(OUTCOMES_PATH)
    required_idx = {"stay_id", "subject_id", "split_24h"}
    missing = sorted(required_idx - set(idx_df.columns))
    if missing:
        raise KeyError(f"Alignment file missing columns: {missing}")

    required_outcome = {"stay_id", "subject_id", *REPORT_TASKS, *VALID_COL.values()}
    missing = sorted(required_outcome - set(outcomes_df.columns))
    if missing:
        raise KeyError(f"Outcomes file missing columns: {missing}")

    selected_outcome_cols = [
        "stay_id", "subject_id", *REPORT_TASKS,
        *[c for c in outcomes_df.columns if c.endswith("_valid")],
    ]
    outcomes_small = outcomes_df[list(dict.fromkeys(selected_outcome_cols))].copy()
    if outcomes_small["stay_id"].duplicated().any():
        raise ValueError("Frozen outcomes contain duplicate stay_id rows.")

    # A5 alignment already contains the frozen target columns because it is
    # derived from the A4 final index.  Merge only the frozen identity/split
    # fields with the authoritative target package to avoid duplicate target
    # columns (pandas would otherwise create *_x/*_y names).
    alignment_core_cols = ["stay_id", "subject_id", "split_24h"]
    alignment_core = idx_df[alignment_core_cols].copy()
    merged = alignment_core.merge(
        outcomes_small,
        on=["stay_id", "subject_id"],
        how="left",
        validate="one_to_one",
    )

    structured = np.load(STRUCTURED_PATH, mmap_mode="r")
    demographics = np.load(DEMOGRAPHICS_PATH, mmap_mode="r")
    if tuple(structured.shape) != EXPECTED_STRUCTURED_SHAPE:
        raise ValueError(f"Structured shape {structured.shape} != {EXPECTED_STRUCTURED_SHAPE}.")
    if tuple(demographics.shape) != EXPECTED_DEMO_SHAPE:
        raise ValueError(f"Demographics shape {demographics.shape} != {EXPECTED_DEMO_SHAPE}.")
    if len(merged) != EXPECTED_N:
        raise ValueError(f"Merged alignment/outcomes rows {len(merged)} != {EXPECTED_N}.")

    split = merged["split_24h"].astype(str).to_numpy()
    split_counts = pd.Series(split).value_counts().to_dict()
    expected_counts = {"train": EXPECTED_TRAIN, "val": EXPECTED_VAL, "test": EXPECTED_TEST}
    if split_counts != expected_counts:
        raise ValueError(f"Split counts {split_counts} != {expected_counts}.")

    train_idx = np.flatnonzero(split == "train")
    val_idx = np.flatnonzero(split == "val")
    test_idx = np.flatnonzero(split == "test")

    targets = {t: merged[t].to_numpy(dtype=np.float32) for t in REPORT_TASKS}
    valid_arrays = {
        t: (merged[VALID_COL[t]].to_numpy(dtype=np.int8) == 1)
        for t in REPORT_TASKS
    }

    text, text_missing = load_text_embeddings(merged["stay_id"].to_numpy())

    return FrozenData(
        structured=structured,
        demographics=demographics,
        text=text,
        text_missing=text_missing,
        stay_id=merged["stay_id"].to_numpy(),
        subject_id=merged["subject_id"].to_numpy(),
        split=split,
        outcomes=merged,
        targets=targets,
        valid_arrays=valid_arrays,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )


def contract_audit(data: FrozenData) -> Dict[str, Any]:
    checks: Dict[str, Any] = {
        "structured_shape": list(data.structured.shape),
        "demographics_shape": list(data.demographics.shape),
        "n_stays": int(len(data.stay_id)),
        "split_counts": {
            "train": int(len(data.train_idx)),
            "val": int(len(data.val_idx)),
            "test": int(len(data.test_idx)),
        },
        "text_present": int(np.sum(data.text_missing == 0)),
        "text_missing": int(np.sum(data.text_missing == 1)),
        "structured_shape_pass": tuple(data.structured.shape) == EXPECTED_STRUCTURED_SHAPE,
        "demographics_shape_pass": tuple(data.demographics.shape) == EXPECTED_DEMO_SHAPE,
        "split_pass": (
            len(data.train_idx) == EXPECTED_TRAIN
            and len(data.val_idx) == EXPECTED_VAL
            and len(data.test_idx) == EXPECTED_TEST
        ),
        "text_count_pass": int(np.sum(data.text_missing == 0)) == EXPECTED_TEXT_PRESENT,
    }

    # Patient-level isolation check.
    train_subj = set(data.subject_id[data.train_idx].tolist())
    val_subj = set(data.subject_id[data.val_idx].tolist())
    test_subj = set(data.subject_id[data.test_idx].tolist())
    checks["patient_overlap"] = {
        "train_val": len(train_subj & val_subj),
        "train_test": len(train_subj & test_subj),
        "val_test": len(val_subj & test_subj),
    }
    checks["patient_isolation_pass"] = all(v == 0 for v in checks["patient_overlap"].values())
    checks["all_pass"] = all(
        [checks["structured_shape_pass"], checks["demographics_shape_pass"], checks["split_pass"], checks["text_count_pass"], checks["patient_isolation_pass"]]
    )
    return checks


class FrozenDataset(Dataset):
    def __init__(self, data: FrozenData, indices: np.ndarray):
        self.data = data
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, pos: int):
        i = int(self.indices[pos])
        x = torch.from_numpy(np.array(self.data.structured[i], dtype=np.float32, copy=True))
        d = torch.from_numpy(np.array(self.data.demographics[i], dtype=np.float32, copy=True))
        text = torch.from_numpy(np.array(self.data.text[i], dtype=np.float32, copy=True))
        missing = torch.tensor(float(self.data.text_missing[i]), dtype=torch.float32)
        targets = {t: torch.tensor(float(self.data.targets[t][i]), dtype=torch.float32) for t in PRIMARY_TASKS}
        valid = {t: torch.tensor(int(self.data.valid_arrays[t][i]), dtype=torch.bool) for t in PRIMARY_TASKS}
        return x, d, text, missing, targets, valid


def make_loader(data: FrozenData, indices: np.ndarray, batch_size: int, shuffle: bool, device: torch.device) -> DataLoader:
    return DataLoader(
        FrozenDataset(data, indices),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )



# ============================================================================
# 3. MODEL FAMILY COMPONENTS
# ============================================================================
class AttentionPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dim, max(dim // 2, 32)),
            nn.Tanh(),
            nn.Linear(max(dim // 2, 32), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = self.score(x).float().squeeze(-1)
        weights = torch.softmax(scores, dim=-1).to(dtype=x.dtype)
        return (x * weights.unsqueeze(-1)).sum(dim=1)


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float, norm: bool = True):
        super().__init__()
        layers: List[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, out_dim)]
        if norm:
            layers.append(nn.LayerNorm(out_dim))
        layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BinaryTaskTower(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class LOSHead(nn.Module):
    """Single source of truth for the two-expert mortality-conditioned LOS head."""
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.survivor = nn.Sequential(
            nn.Linear(dim, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 1)
        )
        self.death = nn.Sequential(
            nn.Linear(dim, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 1)
        )

    def forward(self, h: torch.Tensor, p_mort: torch.Tensor) -> torch.Tensor:
        mu_surv = F.softplus(self.survivor(h).squeeze(-1))
        mu_death = F.softplus(self.death(h).squeeze(-1))
        return (1.0 - p_mort) * mu_surv + p_mort * mu_death


class LocalTemporalBlock(nn.Module):
    """Depthwise-separable residual convolution for short-horizon temporal structure."""
    def __init__(self, dim: int, dropout: float, kernel_size: int):
        super().__init__()
        pad = kernel_size // 2
        self.norm = nn.LayerNorm(dim)
        self.dw = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=pad, groups=dim)
        self.pw = nn.Conv1d(dim, dim, kernel_size=1)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.norm(x).transpose(1, 2)
        z = self.dw(z)
        z = self.pw(z).transpose(1, 2)
        return x + self.drop(self.act(z))


@dataclass(frozen=True)
class FamilySpec:
    key: str
    name: str
    main_idea: str
    tests: str


FAMILY_SPECS = {
    "A": FamilySpec(
        "A", "Full-A", "Current proposed Transformer + gate", "Current baseline"
    ),
    "B": FamilySpec(
        "B", "Full-B", "Deeper/wider temporal Transformer + stronger temporal pooling", "Structured representation"
    ),
    "C": FamilySpec(
        "C", "Full-C", "Structured temporal encoder + stronger multimodal text fusion", "Whether text is being underused"
    ),
    "D": FamilySpec(
        "D", "Full-D", "Task-specific representations / adapters rather than one shared head representation", "Whether multitask interference is hurting weaker tasks"
    ),
    "E": FamilySpec(
        "E", "Full-E", "Structured + text + demographics with gated residual fusion and task-specific adapters", "More flexible multimodal learning"
    ),
    "F": FamilySpec(
        "F", "Full-F", "Hybrid sequence architecture combining local temporal modeling + global attention", "Short-term + long-term temporal patterns"
    ),
}


class FamilyModel(nn.Module):
    """Six complete architecture variants sharing one frozen V2 data/evaluation contract."""
    def __init__(self, version: str, dropout: float = 0.15):
        super().__init__()
        version = str(version).upper()
        if version not in FAMILY_SPECS:
            raise ValueError(f"Unknown family version: {version}")
        self.version = version
        self.spec = FAMILY_SPECS[version]
        self.dropout = float(dropout)
        self.use_uncertainty_weighting = True

        # ------------------------------------------------------------------
        # Structured encoder
        # ------------------------------------------------------------------
        if version == "B":
            sdim = 384
            self.structured_proj = nn.Sequential(
                nn.Linear(110, sdim), nn.LayerNorm(sdim), nn.GELU(), nn.Dropout(dropout)
            )
            self.position = nn.Parameter(torch.zeros(24, sdim))
            nn.init.normal_(self.position, mean=0.0, std=0.02)
            enc = nn.TransformerEncoderLayer(
                d_model=sdim, nhead=6, dim_feedforward=768, dropout=dropout,
                activation="gelu", batch_first=True, norm_first=True,
            )
            self.temporal = nn.TransformerEncoder(enc, num_layers=3)
            self.temporal_norm = nn.LayerNorm(sdim)
            self.attn_pool = AttentionPool(sdim)
            self.struct_project = nn.Sequential(
                nn.Linear(sdim * 3, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout)
            )
            self.struct_dim = sdim
        else:
            sdim = 256
            self.structured_proj = nn.Sequential(
                nn.Linear(110, sdim), nn.LayerNorm(sdim), nn.GELU(), nn.Dropout(dropout)
            )
            self.position = nn.Parameter(torch.zeros(24, sdim))
            nn.init.normal_(self.position, mean=0.0, std=0.02)
            enc = nn.TransformerEncoderLayer(
                d_model=sdim, nhead=4, dim_feedforward=512, dropout=dropout,
                activation="gelu", batch_first=True, norm_first=True,
            )
            self.temporal = nn.TransformerEncoder(enc, num_layers=2)
            self.temporal_norm = nn.LayerNorm(sdim)
            self.attn_pool = AttentionPool(sdim)
            self.struct_project = nn.Sequential(
                nn.Linear(sdim * 2, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout)
            )
            self.struct_dim = sdim

        if version == "F":
            self.local_block_3 = LocalTemporalBlock(256, dropout, 3)
            self.local_block_5 = LocalTemporalBlock(256, dropout, 5)

        # ------------------------------------------------------------------
        # Text and demographics
        # ------------------------------------------------------------------
        if version == "C":
            self.text_proj = nn.Sequential(
                nn.Linear(768, 384), nn.LayerNorm(384), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(384, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout),
            )
        else:
            self.text_proj = nn.Sequential(
                nn.Linear(768, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(256, 256), nn.LayerNorm(256), nn.GELU(),
            )
        self.demo_proj = nn.Sequential(
            nn.Linear(4, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 256), nn.LayerNorm(256), nn.GELU(),
        )

        # ------------------------------------------------------------------
        # Multimodal fusion
        # ------------------------------------------------------------------
        if version == "C":
            # Stronger text interaction: pairwise product + absolute difference
            # + text-guided residual modulation. No sequence cross-attention.
            self.text_gate = nn.Sequential(
                nn.Linear(513, 256), nn.GELU(), nn.Dropout(dropout), nn.Linear(256, 256), nn.Sigmoid()
            )
            self.text_pair_fusion = nn.Sequential(
                nn.Linear(1024, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(),
            )
            self.fusion = nn.Sequential(
                nn.Linear(256 + 256 + 256 + 1, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(),
            )
        elif version == "E":
            # Gated residual fusion explicitly retains a residual multimodal path.
            self.gate = nn.Sequential(
                nn.Linear(769, 256), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(256, 256), nn.Sigmoid(),
            )
            self.residual_fusion = nn.Sequential(
                nn.Linear(769, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(),
            )
            self.fusion_norm = nn.LayerNorm(256)
        else:
            gate_in = 769
            self.gate = nn.Sequential(
                nn.Linear(gate_in, 256), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(256, 256), nn.Sigmoid(),
            )
            if version == "D":
                self.fusion = nn.Sequential(
                    nn.Linear(769 + 256, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(dropout),
                    nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(),
                )
            else:
                self.fusion = nn.Sequential(
                    nn.Linear(769 + 256, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(dropout),
                    nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(),
                )

        # ------------------------------------------------------------------
        # Task representations / heads
        # ------------------------------------------------------------------
        task_names = [
            "mortality_24h", "ventilation_onset_24h", "vasopressor_onset_24h",
            "readmission_30d_24h", "los_hours_24h"
        ]
        self.task_adapters = nn.ModuleDict()
        if version in {"D", "E"}:
            for task in task_names:
                self.task_adapters[task] = nn.Sequential(
                    nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout),
                    nn.Linear(128, 256), nn.LayerNorm(256), nn.GELU(),
                )

        self.mortality = BinaryTaskTower(256, dropout)
        self.ventilation = BinaryTaskTower(256, dropout)
        self.vasopressor = BinaryTaskTower(256, dropout)
        self.readmission = BinaryTaskTower(256, dropout)
        self.los_head = LOSHead(256, dropout)
        self.log_vars = nn.Parameter(torch.zeros(5))

    def _encode_structured(self, x: torch.Tensor) -> torch.Tensor:
        h = self.structured_proj(x)
        if self.version == "F":
            h = self.local_block_3(h)
            h = self.local_block_5(h)
        h = h + self.position[:h.shape[1]].to(dtype=h.dtype, device=h.device)
        h = self.temporal(h)
        h = self.temporal_norm(h)
        h_attn = self.attn_pool(h)
        h_last = h[:, -1, :]
        if self.version == "B":
            h_mean = h.mean(dim=1)
            h_struct = self.struct_project(torch.cat([h_attn, h_mean, h_last], dim=-1))
        else:
            h_struct = self.struct_project(torch.cat([h_attn, h_last], dim=-1))
        return h_struct

    def encode(self, x, demographics, text, text_missing):
        s = self._encode_structured(x)
        m = text_missing.unsqueeze(-1).to(dtype=s.dtype)
        t = self.text_proj(text)
        # Missing note = exact zero after projection, including all projection biases.
        t = t * (1.0 - m)
        d = self.demo_proj(demographics)

        if self.version == "C":
            gate = self.text_gate(torch.cat([s, t, m], dim=-1))
            text_residual = gate * t
            pair = torch.cat([s, t, s * t, torch.abs(s - t)], dim=-1)
            pair_repr = self.text_pair_fusion(pair)
            h = self.fusion(torch.cat([s + text_residual + pair_repr, t, d, m], dim=-1))
        elif self.version == "E":
            g = self.gate(torch.cat([s, t, d, m], dim=-1))
            base = g * t + (1.0 - g) * s + d
            residual = self.residual_fusion(torch.cat([base, s * t, torch.abs(s - t), m], dim=-1))
            h = self.fusion_norm(base + residual)
        else:
            g = self.gate(torch.cat([s, t, d, m], dim=-1))
            base = g * t + (1.0 - g) * s + d
            h = self.fusion(torch.cat([base, s, t, d, m], dim=-1))
        return h, s, t, d

    def _task_repr(self, h: torch.Tensor, task: str) -> torch.Tensor:
        if task in self.task_adapters:
            return h + self.task_adapters[task](h)
        return h

    def forward(self, x, demographics, text, text_missing):
        h, _, _, _ = self.encode(x, demographics, text, text_missing)
        hm = self._task_repr(h, "mortality_24h")
        hv = self._task_repr(h, "ventilation_onset_24h")
        hp = self._task_repr(h, "vasopressor_onset_24h")
        hr = self._task_repr(h, "readmission_30d_24h")
        hl = self._task_repr(h, "los_hours_24h")

        p_mort_logit = self.mortality(hm)
        p_mort = torch.sigmoid(p_mort_logit)
        mu_los = self.los_head(hl, p_mort)
        return {
            "mortality_24h": p_mort_logit,
            "ventilation_onset_24h": self.ventilation(hv),
            "vasopressor_onset_24h": self.vasopressor(hp),
            "readmission_30d_24h": self.readmission(hr),
            "los_excess_log1p": mu_los,
        }

# ============================================================================
# 4. LOS / MULTITASK LOSS
# ============================================================================
def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not torch.any(mask):
        return values.new_zeros(())
    return values[mask].mean()


def compute_task_losses(outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], valid: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    losses: Dict[str, torch.Tensor] = {}
    for task in BINARY_TASKS:
        mask = valid[task]
        if torch.any(mask):
            losses[task] = F.binary_cross_entropy_with_logits(
                outputs[task][mask], targets[task][mask].float()
            )
        else:
            losses[task] = outputs[task].new_zeros(())

    los_mask = valid["los_hours_24h"]
    if torch.any(los_mask):
        target_shifted = torch.log1p(
            torch.clamp(targets["los_hours_24h"][los_mask].float() - 24.0, min=0.0)
        )
        losses["los_hours_24h"] = masked_mean(
            torch.abs(outputs["los_excess_log1p"][los_mask].float() - target_shifted),
            torch.ones_like(target_shifted, dtype=torch.bool),
        )
    else:
        losses["los_hours_24h"] = outputs["los_excess_log1p"].new_zeros(())
    return losses


def total_multitask_loss(model: nn.Module, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
    ordered = [
        losses["mortality_24h"],
        losses["ventilation_onset_24h"],
        losses["vasopressor_onset_24h"],
        losses["readmission_30d_24h"],
        losses["los_hours_24h"],
    ]
    stack = torch.stack(ordered)
    if model.log_vars is not None:
        return torch.sum(torch.exp(-model.log_vars) * stack + model.log_vars)
    return torch.sum(stack)


# ============================================================================
# 5. METRICS
# ============================================================================
def safe_auroc(y: np.ndarray, p: np.ndarray) -> float:
    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y.astype(int), p))


def safe_auprc(y: np.ndarray, p: np.ndarray) -> float:
    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y.astype(int), p))


def safe_brier(y: np.ndarray, p: np.ndarray) -> float:
    if len(y) == 0:
        return float("nan")
    return float(brier_score_loss(y.astype(int), p))


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    if len(y) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    n = len(y)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p <= hi if hi == 1.0 else p < hi)
        if not np.any(mask):
            continue
        ece += float(np.sum(mask) / n) * abs(float(np.mean(p[mask])) - float(np.mean(y[mask])))
    return float(ece)


def threshold_metrics(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    if len(y) == 0:
        return {"accuracy": np.nan, "precision": np.nan, "recall": np.nan, "f1": np.nan, "tn": np.nan, "fp": np.nan, "fn": np.nan, "tp": np.nan}
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y.astype(int), pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y.astype(int), pred)),
        "precision": float(precision_score(y.astype(int), pred, zero_division=0)),
        "recall": float(recall_score(y.astype(int), pred, zero_division=0)),
        "f1": float(f1_score(y.astype(int), pred, zero_division=0)),
        "tn": float(tn), "fp": float(fp), "fn": float(fn), "tp": float(tp),
    }


def metric_row(task: str, y: np.ndarray, p: np.ndarray) -> Dict[str, Any]:
    if task in BINARY_TASKS or task == "deterioration_24h":
        out = {
            "valid_n": int(len(y)),
            "prevalence": float(np.mean(y)) if len(y) else np.nan,
            "auprc": safe_auprc(y, p),
            "auroc": safe_auroc(y, p),
            "brier": safe_brier(y, p),
            "ece": expected_calibration_error(y, p),
        }
        out.update(threshold_metrics(y, p))
        return out

    return {
        "valid_n": int(len(y)),
        "mae": float(mean_absolute_error(y, p)) if len(y) else np.nan,
        "rmse": float(np.sqrt(mean_squared_error(y, p))) if len(y) else np.nan,
        "median_ae": float(median_absolute_error(y, p)) if len(y) else np.nan,
        "r2": float(r2_score(y, p)) if len(y) > 1 else np.nan,
    }


def bootstrap_metric_ci(
    y: np.ndarray,
    p: np.ndarray,
    task: str,
    reps: int,
    seed: int,
    subject_ids: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Metric point estimate + subject-clustered percentile bootstrap CI.

    When subject_ids are supplied, complete subject clusters are resampled so the
    confidence intervals use the same clustering unit as paired benchmark comparisons.
    """
    point = metric_row(task, y, p)
    metric_keys = [
        k for k in point
        if k in {"auprc", "auroc", "brier", "ece", "accuracy", "precision", "recall", "f1", "mae", "rmse", "median_ae", "r2"}
    ]
    values = {k: [] for k in metric_keys}
    n = len(y)
    if n == 0 or reps <= 0:
        return point

    if subject_ids is None:
        # Backward-compatible fallback: row-level bootstrap. Main model evaluation paths
        # always pass subject_ids and therefore use subject-clustered resampling.
        cluster_labels = np.arange(n, dtype=np.int64)
    else:
        subject_ids = np.asarray(subject_ids)
        if len(subject_ids) != n:
            raise ValueError("subject_ids must have the same length as y and p.")
        cluster_labels = np.asarray(subject_ids)

    unique_subjects = np.unique(cluster_labels)
    rows_by_subject = {s: np.flatnonzero(cluster_labels == s) for s in unique_subjects}
    rng = np.random.default_rng(seed)
    n_subjects = len(unique_subjects)

    for _ in range(reps):
        sampled_subjects = rng.choice(unique_subjects, size=n_subjects, replace=True)
        idx = np.concatenate([rows_by_subject[s] for s in sampled_subjects])
        m = metric_row(task, y[idx], p[idx])
        for k in metric_keys:
            value = m.get(k, np.nan)
            if np.isfinite(value):
                values[k].append(float(value))

    out = dict(point)
    out["bootstrap_unit"] = "subject_id" if subject_ids is not None else "row"
    out["bootstrap_subjects"] = int(n_subjects) if subject_ids is not None else int(n)
    for k in metric_keys:
        arr = np.asarray(values[k], dtype=float)
        out[f"{k}_ci95_low"] = float(np.quantile(arr, 0.025)) if len(arr) else np.nan
        out[f"{k}_ci95_high"] = float(np.quantile(arr, 0.975)) if len(arr) else np.nan
    return out


def calibration_bins(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    if len(y) == 0:
        return pd.DataFrame(columns=["bin", "n", "mean_prediction", "mean_observed", "abs_gap"])
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for b, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (p >= lo) & (p <= hi if hi == 1 else p < hi)
        if not np.any(mask):
            continue
        mp = float(np.mean(p[mask]))
        mo = float(np.mean(y[mask]))
        rows.append({"bin": b, "n": int(mask.sum()), "mean_prediction": mp, "mean_observed": mo, "abs_gap": abs(mp-mo)})
    return pd.DataFrame(rows)


def validation_selection_score(metric_df: pd.DataFrame) -> float:
    rows = metric_df[metric_df["task"].isin(BINARY_TASKS)].copy()
    if len(rows) != 4:
        return float("-inf")
    vals = rows["auprc"].to_numpy(dtype=float)
    return float(np.mean(vals)) if np.all(np.isfinite(vals)) else float("-inf")


# ============================================================================
# 6. PREDICTION / EVALUATION
# ============================================================================
@torch.no_grad()
def collect_predictions(model: nn.Module, data: FrozenData, indices: np.ndarray, device: torch.device, batch_size: int) -> Dict[str, np.ndarray]:
    model.eval()
    loader = make_loader(data, indices, batch_size, False, device)
    preds = {t: [] for t in PRIMARY_TASKS}
    for x, d, text, missing, _, _ in loader:
        x = x.to(device, non_blocking=True)
        d = d.to(device, non_blocking=True)
        text = text.to(device, non_blocking=True)
        missing = missing.to(device, non_blocking=True)
        with autocast_context(device):
            out = model(x, d, text, missing)
        for task in BINARY_TASKS:
            preds[task].append(torch.sigmoid(out[task]).float().cpu().numpy())
        preds["los_hours_24h"].append((24.0 + torch.expm1(out["los_excess_log1p"].float())).cpu().numpy())
    return {t: np.concatenate(preds[t]) for t in PRIMARY_TASKS}


def derive_deterioration(data: FrozenData, indices: np.ndarray, pred: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Use the frozen stored deterioration target/validity, but derive the model prediction from the three components.
    y = data.targets["deterioration_24h"][indices].astype(float)
    valid = data.valid_arrays["deterioration_24h"][indices].astype(bool)
    p = 1.0 - (
        (1.0 - pred["mortality_24h"])
        * (1.0 - pred["ventilation_onset_24h"])
        * (1.0 - pred["vasopressor_onset_24h"])
    )
    return y, valid, p


def evaluate_arrays(data: FrozenData, indices: np.ndarray, pred: Dict[str, np.ndarray], reps: int, split_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: List[Dict[str, Any]] = []
    cal_tables: List[pd.DataFrame] = []
    for task in PRIMARY_TASKS:
        y = data.targets[task][indices].astype(float)
        p = pred[task].astype(float)
        valid = data.valid_arrays[task][indices].astype(bool)
        keep = valid & np.isfinite(y) & np.isfinite(p)
        yy = y[keep]
        pp = p[keep]
        subjects = data.subject_id[indices][keep]
        row = bootstrap_metric_ci(
            yy, pp, task, reps=reps, seed=stable_seed(f"{split_name}|{task}"), subject_ids=subjects
        )
        metric_rows.append({"split": split_name, "task": task, **row})
        if task in BINARY_TASKS:
            cb = calibration_bins(yy, pp)
            if not cb.empty:
                cb.insert(0, "split", split_name)
                cb.insert(1, "task", task)
                cal_tables.append(cb)

    y_det, valid_det, p_det = derive_deterioration(data, indices, pred)
    keep = valid_det & np.isfinite(y_det) & np.isfinite(p_det)
    det_subjects = data.subject_id[indices][keep]
    row = bootstrap_metric_ci(
        y_det[keep],
        p_det[keep],
        "deterioration_24h",
        reps=reps,
        seed=stable_seed(f"{split_name}|deterioration"),
        subject_ids=det_subjects,
    )
    metric_rows.append({"split": split_name, "task": "deterioration_24h", **row})
    cb = calibration_bins(y_det[keep].astype(int), p_det[keep])
    if not cb.empty:
        cb.insert(0, "split", split_name)
        cb.insert(1, "task", "deterioration_24h")
        cal_tables.append(cb)

    return pd.DataFrame(metric_rows), pd.concat(cal_tables, ignore_index=True) if cal_tables else pd.DataFrame()


def prediction_table(data: FrozenData, indices: np.ndarray, pred: Dict[str, np.ndarray], split_name: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    text_available = (data.text_missing[indices] == 0).astype(int)
    for j, i in enumerate(indices):
        for task in PRIMARY_TASKS:
            rows.append({
                "stay_id": data.stay_id[i],
                "subject_id": data.subject_id[i],
                "split": split_name,
                "text_available": int(text_available[j]),
                "task": task,
                "target": float(data.targets[task][i]),
                "validity": int(data.valid_arrays[task][i]),
                "prediction": float(pred[task][j]),
            })
        det_y = data.targets["deterioration_24h"][i]
        det_v = data.valid_arrays["deterioration_24h"][i]
        det_p = 1.0 - (
            (1.0 - pred["mortality_24h"][j])
            * (1.0 - pred["ventilation_onset_24h"][j])
            * (1.0 - pred["vasopressor_onset_24h"][j])
        )
        rows.append({
            "stay_id": data.stay_id[i],
            "subject_id": data.subject_id[i],
            "split": split_name,
            "text_available": int(text_available[j]),
            "task": "deterioration_24h",
            "target": float(det_y),
            "validity": int(det_v),
            "prediction": float(det_p),
        })
    return pd.DataFrame(rows)


def text_stratified_stats(pred_df: pd.DataFrame, reps: int = 1000) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    test = pred_df[pred_df["split"] == "test"].copy()
    for availability, label in [(1, "text_available"), (0, "text_missing")]:
        sdf = test[test["text_available"] == availability]
        for task in REPORT_TASKS:
            tdf = sdf[sdf["task"] == task].copy()
            tdf = tdf[tdf["validity"].astype(bool)]
            tdf = tdf[np.isfinite(tdf["target"]) & np.isfinite(tdf["prediction"])]
            y = tdf["target"].to_numpy(float)
            p = tdf["prediction"].to_numpy(float)
            if task in BINARY_TASKS or task == "deterioration_24h":
                m = metric_row(task, y, p)
                rows.append({"group": label, "task": task, **m})
            else:
                m = metric_row(task, y, p)
                rows.append({"group": label, "task": task, **m})
    return pd.DataFrame(rows)


# ============================================================================
# 7. TRAINING
# ============================================================================
def build_optimizer(model: nn.Module, lr: float, weight_decay: float):
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def build_scheduler(optimizer, name: str, max_epochs: int, steps_per_epoch: int, lr: float):
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    if name == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            total_steps=max(1, max_epochs * steps_per_epoch),
            pct_start=0.1,
            anneal_strategy="cos",
            cycle_momentum=False,
        )
    raise ValueError(name)


def run_epoch_train(model: nn.Module, data: FrozenData, indices: np.ndarray, device: torch.device, params: Dict[str, Any]) -> float:
    model.train()
    loader = make_loader(data, indices, int(params["batch_size"]), True, device)
    optimizer = params["optimizer"]
    scheduler = params["scheduler_obj"]
    grad_clip = float(params["grad_clip"])
    running = 0.0
    seen = 0
    scheduler_name = params["scheduler"]

    for x, d, text, missing, targets_cpu, valid_cpu in loader:
        x = x.to(device, non_blocking=True)
        d = d.to(device, non_blocking=True)
        text = text.to(device, non_blocking=True)
        missing = missing.to(device, non_blocking=True)
        targets = {k: v.to(device, non_blocking=True) for k, v in targets_cpu.items()}
        valid = {k: v.to(device, non_blocking=True) for k, v in valid_cpu.items()}

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device):
            outputs = model(x, d, text, missing)
            losses = compute_task_losses(outputs, targets, valid)
            loss = total_multitask_loss(model, losses)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if scheduler_name == "onecycle":
            scheduler.step()

        bs = x.shape[0]
        running += float(loss.detach().cpu()) * bs
        seen += bs

    if scheduler_name == "cosine":
        scheduler.step()
    return running / max(seen, 1)


@torch.no_grad()
def run_validation(model: nn.Module, data: FrozenData, indices: np.ndarray, device: torch.device, params: Dict[str, Any], bootstrap_reps: int = 0) -> Tuple[float, float, pd.DataFrame]:
    pred = collect_predictions(model, data, indices, device, int(params["batch_size"]))
    metrics, _ = evaluate_arrays(data, indices, pred, reps=bootstrap_reps, split_name="val")
    score = validation_selection_score(metrics)
    los_row = metrics[metrics["task"] == "los_hours_24h"]
    los_mae = float(los_row["mae"].iloc[0]) if not los_row.empty else np.nan
    return score, los_mae, metrics



# ============================================================================
# 8. FAMILY METADATA / OUTPUTS
# ============================================================================
def first_run_params() -> Dict[str, Any]:
    return dict(FIRST_RUN)


def architecture_manifest(version: str, params: Dict[str, Any]) -> Dict[str, Any]:
    spec = FAMILY_SPECS[version]
    manifests = {
        "A": {
            "structured": "110->256; 2-layer Transformer; 4 heads; FFN 512; attention pool + exact last-hour",
            "text": "768->256->256",
            "fusion": "missingness-aware gated fusion + direct fusion",
            "task_representation": "shared 256 representation",
        },
        "B": {
            "structured": "110->384; 3-layer Transformer; 6 heads; FFN 768; attention + mean + exact last-hour",
            "text": "768->256->256",
            "fusion": "missingness-aware gated fusion + direct fusion",
            "task_representation": "shared 256 representation",
        },
        "C": {
            "structured": "110->256; 2-layer Transformer; 4 heads; FFN 512; attention pool + exact last-hour",
            "text": "768->384->256",
            "fusion": "text-gated residual + pairwise [s,t,s*t,|s-t|] fusion",
            "task_representation": "shared 256 representation",
        },
        "D": {
            "structured": "Full-A structured encoder",
            "text": "768->256->256",
            "fusion": "Full-A gated/direct fusion",
            "task_representation": "per-task residual adapters before each head/LOS expert",
        },
        "E": {
            "structured": "Full-A structured encoder",
            "text": "768->256->256",
            "fusion": "gated residual fusion using modality interaction features",
            "task_representation": "per-task residual adapters before each head/LOS expert",
        },
        "F": {
            "structured": "local depthwise Conv1d kernels 3+5 + 2-layer global Transformer",
            "text": "768->256->256",
            "fusion": "Full-A gated/direct fusion",
            "task_representation": "shared 256 representation",
        },
    }
    return {
        "family": "Pipeline A V2 Proposed Model Family",
        "version": spec.name,
        "version_key": version,
        "main_idea": spec.main_idea,
        "tests": spec.tests,
        "architecture_status": "fixed_screening_version",
        "input_contract": {
            "structured": [24, 110],
            "demographics": 4,
            "text_embedding": 768,
            "text_missing_flag": True,
        },
        "details": manifests[version],
        "dropout": float(params["dropout"]),
        "los": {
            "experts": "survivor + death",
            "gate": "mortality probability",
            "gradient": "no stop-gradient",
            "target": "log1p(max(los_hours_24h - 24, 0))",
            "loss": "masked L1 on shifted log1p target",
            "inverse": "24 + expm1(predicted_excess_log1p)",
        },
        "selection_metric": "mean validation AUPRC across mortality, ventilation, vasopressor, readmission",
        "secondary_screening_metrics": "mean validation/test AUROC across the same four binary tasks",
        "derived_deterioration": "evaluation-only noisy-OR",
        "multitask": "uncertainty-weighted five-task summed objective",
        "seed": SEED,
        "first_run_protocol": FIRST_RUN,
        "test_used_for_selection": False,
    }


def runtime_metadata(device: torch.device) -> Dict[str, Any]:
    return {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "hip_version": getattr(torch.version, "hip", None),
        "python": sys.version,
        "platform": platform.platform(),
        "amp_mode": "BF16 autocast" if device.type == "cuda" else "CPU not permitted for training",
    }


def model_parameter_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def save_contract_audit(data: FrozenData) -> Dict[str, Any]:
    audit = contract_audit(data)
    FAMILY_RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    (FAMILY_RESULTS_ROOT / "frozen_v2_contract_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    return audit


def save_family_outputs(
    data: FrozenData,
    model: nn.Module,
    version: str,
    indices_val: np.ndarray,
    indices_test: np.ndarray,
    device: torch.device,
    params: Dict[str, Any],
    run_dir: Path,
    best_score: float,
    best_epoch: int,
    training_seconds: float,
    bootstrap_reps: int,
) -> Dict[str, Any]:
    val_pred = collect_predictions(model, data, indices_val, device, int(params["batch_size"]))
    test_pred = collect_predictions(model, data, indices_test, device, int(params["batch_size"]))
    val_metrics, val_cal = evaluate_arrays(data, indices_val, val_pred, reps=bootstrap_reps, split_name="val")
    test_metrics, test_cal = evaluate_arrays(data, indices_test, test_pred, reps=bootstrap_reps, split_name="test")

    val_table = prediction_table(data, indices_val, val_pred, "val")
    test_table = prediction_table(data, indices_test, test_pred, "test")
    all_predictions = pd.concat([val_table, test_table], ignore_index=True)

    run_dir.mkdir(parents=True, exist_ok=True)
    all_predictions.to_csv(run_dir / "predictions.csv", index=False)
    test_table.to_csv(run_dir / "test_predictions.csv", index=False)
    metrics = pd.concat([val_metrics, test_metrics], ignore_index=True)
    metrics.to_csv(run_dir / "metrics.csv", index=False)
    pd.concat([val_cal, test_cal], ignore_index=True).to_csv(run_dir / "calibration_bins.csv", index=False)
    text_stratified_stats(all_predictions).to_csv(run_dir / "text_availability_stratified_metrics.csv", index=False)

    test_bin = test_metrics[test_metrics["task"].isin(BINARY_TASKS)]
    val_bin = val_metrics[val_metrics["task"].isin(BINARY_TASKS)]
    test_los = test_metrics[test_metrics["task"] == "los_hours_24h"]
    val_los = val_metrics[val_metrics["task"] == "los_hours_24h"]
    row = {
        "version": version,
        "model": FAMILY_SPECS[version].name,
        "validation_mean_auprc": float(val_bin["auprc"].mean()),
        "validation_mean_auroc": float(val_bin["auroc"].mean()),
        "test_mean_auprc": float(test_bin["auprc"].mean()),
        "test_mean_auroc": float(test_bin["auroc"].mean()),
        "validation_los_mae": float(val_los["mae"].iloc[0]) if not val_los.empty else np.nan,
        "test_los_mae": float(test_los["mae"].iloc[0]) if not test_los.empty else np.nan,
        "best_epoch": int(best_epoch),
        "validation_selection_score": float(best_score),
        "training_seconds": float(training_seconds),
        "parameter_count": model_parameter_count(model),
    }
    meta = {
        **runtime_metadata(device),
        "seed": SEED,
        "version": version,
        "version_name": FAMILY_SPECS[version].name,
        "main_idea": FAMILY_SPECS[version].main_idea,
        "tests": FAMILY_SPECS[version].tests,
        "mode": "family_fixed",
        "best_epoch": int(best_epoch),
        "validation_selection_score": float(best_score),
        "test_used_for_selection": False,
        "training_seconds": float(training_seconds),
        "parameter_count": model_parameter_count(model),
        "params": params,
        "input_paths": {
            "structured": str(STRUCTURED_PATH),
            "demographics": str(DEMOGRAPHICS_PATH),
            "text": str(TEXT_EMBEDDING_PATH),
            "alignment": str(ALIGNMENT_PATH),
            "outcomes": str(OUTCOMES_PATH),
        },
        "split_counts": {"train": EXPECTED_TRAIN, "val": EXPECTED_VAL, "test": EXPECTED_TEST},
        "text_present_count": EXPECTED_TEXT_PRESENT,
        "text_missing_count": EXPECTED_N - EXPECTED_TEXT_PRESENT,
        "architecture": architecture_manifest(version, params),
    }
    (run_dir / "metadata.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    (run_dir / "architecture_manifest.json").write_text(json.dumps(architecture_manifest(version, params), indent=2), encoding="utf-8")
    return row


def train_family_model(
    data: FrozenData,
    params: Dict[str, Any],
    device: torch.device,
    version: str,
    run_dir: Path,
) -> Tuple[nn.Module, float, int, pd.DataFrame, float]:
    set_seed(SEED)
    model = FamilyModel(version, dropout=float(params["dropout"])).to(device)
    optimizer = build_optimizer(model, float(params["lr"]), float(params["weight_decay"]))
    steps_per_epoch = math.ceil(len(data.train_idx) / int(params["batch_size"]))
    scheduler = build_scheduler(
        optimizer, str(params["scheduler"]), int(params["max_epochs"]), steps_per_epoch, float(params["lr"])
    )
    train_params = dict(params)
    train_params["optimizer"] = optimizer
    train_params["scheduler_obj"] = scheduler

    best_state = None
    best_score = float("-inf")
    best_epoch = 0
    no_improve = 0
    history_rows = []
    t_start = time.time()

    for epoch in range(1, int(params["max_epochs"]) + 1):
        t0 = time.time()
        train_loss = run_epoch_train(model, data, data.train_idx, device, train_params)
        val_score, val_los_mae, _ = run_validation(model, data, data.val_idx, device, params, bootstrap_reps=0)
        current_lr = float(optimizer.param_groups[0]["lr"])
        history_rows.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_mean_AUPRC": val_score,
            "val_LOS_MAE": val_los_mae,
            "lr": current_lr,
            "epoch_seconds": time.time() - t0,
        })
        print(
            f"[{FAMILY_SPECS[version].name}] epoch={epoch:02d} train_loss={train_loss:.6f} "
            f"val_mean_AUPRC={val_score:.6f} val_LOS_MAE={val_los_mae:.4f}", flush=True
        )
        if val_score > best_score:
            best_score = val_score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= int(params["patience"]):
            print(f"[{FAMILY_SPECS[version].name}] early stopping at epoch {epoch}.", flush=True)
            break

    if best_state is None:
        raise RuntimeError(f"{version}: no valid checkpoint selected")
    model.load_state_dict(best_state)
    history = pd.DataFrame(history_rows)
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "params": params,
            "version": version,
            "family_spec": asdict(FAMILY_SPECS[version]),
            "best_epoch": int(best_epoch),
            "best_val_selection_score": float(best_score),
        },
        run_dir / "checkpoint.pt",
    )
    history.to_csv(run_dir / "training_history.csv", index=False)
    return model, best_score, best_epoch, history, time.time() - t_start


def run_family_fixed(data: FrozenData, device: torch.device, bootstrap_reps: int, versions: Sequence[str]) -> pd.DataFrame:
    validate_gpu_bf16(device, "family fixed screening")
    params = first_run_params()
    root = FAMILY_RESULTS_ROOT / "fixed"
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for raw_version in versions:
        version = str(raw_version).upper()
        if version not in FAMILY_SPECS:
            raise ValueError(f"Unknown version {raw_version}; choose A B C D E F")
        print("=" * 100, flush=True)
        print(f"RUNNING {FAMILY_SPECS[version].name}: {FAMILY_SPECS[version].main_idea}", flush=True)
        print(f"TESTS: {FAMILY_SPECS[version].tests}", flush=True)
        print("=" * 100, flush=True)
        run_dir = root / version
        model, score, epoch, history, train_seconds = train_family_model(data, params, device, version, run_dir)
        row = save_family_outputs(
            data, model, version, data.val_idx, data.test_idx, device, params, run_dir,
            score, epoch, train_seconds, bootstrap_reps,
        )
        rows.append(row)
        print(json.dumps(row, indent=2), flush=True)
        del model
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            except Exception:
                pass

    summary = pd.DataFrame(rows).sort_values("test_mean_auroc", ascending=False)
    summary.to_csv(FAMILY_RESULTS_ROOT / "family_fixed_summary.csv", index=False)
    print("=" * 100)
    print("FAMILY FIXED SUMMARY — ranked by test mean AUROC")
    print(summary.to_string(index=False))
    print("=" * 100)
    return summary


# ============================================================================
# 11. MAIN
# ============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Pipeline A V2 Proposed Model Family — fixed architecture screening")
    p.add_argument("--mode", required=True, choices=["audit", "family-fixed", "fixed"], help="Audit or fixed screening run")
    p.add_argument("--bootstrap-reps", type=int, default=2000)
    p.add_argument("--versions", nargs="*", default=list(FAMILY_SPECS.keys()), help="Subset of A B C D E F")
    return p.parse_args()


def run_main():
    args = parse_args()
    if args.bootstrap_reps < 0:
        raise ValueError("--bootstrap-reps must be >= 0")
    require_torch()
    FAMILY_RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    data = load_frozen_v2()
    audit = save_contract_audit(data)
    print("=" * 90)
    print("PIPELINE A V2 — PROPOSED MODEL FAMILY")
    print("=" * 90)
    print(json.dumps(audit, indent=2))
    if args.mode == "audit":
        return

    if args.mode == "fixed":
        # Backward-compatible single-version screening; Full-A = current architecture.
        versions = args.versions[:1] if args.versions else ["A"]
    else:
        versions = args.versions

    device = get_device()
    print(f"device={device}")
    if device.type == "cuda":
        print(f"GPU={torch.cuda.get_device_name(0)}")
        print(f"BF16 supported={torch.cuda.is_bf16_supported()}")

    summary = run_family_fixed(data, device, args.bootstrap_reps, versions)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    run_main()
