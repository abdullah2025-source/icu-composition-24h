#!/usr/bin/env python3
"""
Pipeline A V2 — Final Proposed Model
====================================

Locked thesis contribution model:
    Structured 24h Transformer (2 x 256d, 4 heads, FFN 512)
    + attention pooling + last-hour state
    + frozen pooled BioClinicalBERT text (768d -> 256d)
    + demographics (4d -> 256d)
    + explicit text-missing indicator
    + missingness-aware learned gate
    + parallel direct fusion path
    + four binary task heads
    + mortality-conditioned LOS mixture
    + uncertainty-weighted five-task loss
    + derived noisy-OR deterioration

DELIBERATELY EXCLUDED:
    * Model 29 progression GRU / delta branch
    * Model 30 cross-attention
    * Model 31 cross-attention block
    * MoE routing from Model 32

Frozen V2 input contract:
    structured_24h_processed.npy      [74572,24,110]
    demographics_24h_processed.npy    [74572,4]
    clinical_note_embeddings.parquet  per-stay pooled 768d embeddings
    a5_alignment_index_24h.csv
    targets_24h/outcomes_24h_primary_cohort.csv

Primary validation selection:
    mean AUPRC across mortality, ventilation, vasopressor, readmission.

LOS convention:
    training target = log1p(max(los_hours_24h - 24, 0))
    prediction hours = 24 + expm1(predicted_excess_log1p)
    LOS loss = masked L1 on shifted log1p target.

First-run configuration:
    AdamW
    lr=3e-4
    weight_decay=1e-4
    batch=256
    dropout=0.15
    scheduler=cosine
    grad_clip=1.0
    patience=8
    max_epochs=50
    seed=20260923
    BF16 on CUDA/ROCm

Modes:
    audit
    fixed
    tuned
    ablations
    stats
    compare-benchmark
    all
    trial-worker (internal)

The tuned mode uses 30 Optuna trials with sequential isolated child workers
for ROCm allocator/context safety. Batch size remains fixed at 256 because it
is part of the first-run protocol requested for the proposed model.
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
# 3. MODEL COMPONENTS
# ============================================================================
class AttentionPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = self.score(x).float().squeeze(-1)
        weights = torch.softmax(scores, dim=-1).to(dtype=x.dtype)
        return (x * weights.unsqueeze(-1)).sum(dim=1)


class BinaryTaskTower(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class LOSHead(nn.Module):
    """Unified LOS head used by both the full model and the single-expert ablation."""
    def __init__(self, dim: int, dropout: float, use_mixture: bool = True):
        super().__init__()
        self.use_mixture = use_mixture
        self.survivor = nn.Sequential(
            nn.Linear(dim, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 1)
        )
        self.death = nn.Sequential(
            nn.Linear(dim, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 1)
        ) if use_mixture else None

    def forward(self, h: torch.Tensor, p_mort: torch.Tensor) -> torch.Tensor:
        mu_surv = F.softplus(self.survivor(h).squeeze(-1))
        if not self.use_mixture:
            return mu_surv
        mu_death = F.softplus(self.death(h).squeeze(-1))
        return (1.0 - p_mort) * mu_surv + p_mort * mu_death


class ProposedModel(nn.Module):
    def __init__(self, dropout: float = 0.15, ablation: AblationSpec = ABLATIONS[0]):
        super().__init__()
        self.ablation = ablation
        self.use_uncertainty_weighting = ablation.use_uncertainty_weighting

        # Structured branch.
        self.structured_proj = nn.Sequential(
            nn.Linear(110, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.position = nn.Parameter(torch.zeros(24, 256))
        nn.init.normal_(self.position, mean=0.0, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=4,
            dim_feedforward=512,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.temporal_norm = nn.LayerNorm(256)
        self.attn_pool = AttentionPool(256)
        self.struct_project = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Text branch.
        self.text_proj = nn.Sequential(
            nn.Linear(768, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.GELU(),
        )

        # Demographic branch.
        self.demo_proj = nn.Sequential(
            nn.Linear(4, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 256),
            nn.LayerNorm(256),
            nn.GELU(),
        )

        gate_in = 256 + 256 + 256 + (1 if ablation.use_text_missing_flag else 0)
        direct_in = gate_in
        self.gate = nn.Sequential(
            nn.Linear(gate_in, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.Sigmoid(),
        )

        fusion_in = 256 + direct_in
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
        )

        self.mortality = BinaryTaskTower(256, dropout)
        self.ventilation = BinaryTaskTower(256, dropout)
        self.vasopressor = BinaryTaskTower(256, dropout)
        self.readmission = BinaryTaskTower(256, dropout)
        self.los_head = LOSHead(256, dropout, use_mixture=ablation.use_los_mixture)

        # Learnable uncertainty parameters for the five trained tasks.
        if self.use_uncertainty_weighting:
            self.log_vars = nn.Parameter(torch.zeros(5))
        else:
            self.register_parameter("log_vars", None)

    def encode(self, x: torch.Tensor, demographics: torch.Tensor, text: torch.Tensor, text_missing: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Structured branch.
        h = self.structured_proj(x)
        h = h + self.position[:h.shape[1]].to(dtype=h.dtype, device=h.device)
        h = self.temporal(h)
        h = self.temporal_norm(h)

        if self.ablation.use_attention_pool:
            h_attn = self.attn_pool(h)
        else:
            h_attn = h.mean(dim=1)

        if self.ablation.use_last_hour:
            h_last = h[:, -1, :]
        else:
            h_last = torch.zeros_like(h_attn)

        h_struct = self.struct_project(torch.cat([h_attn, h_last], dim=-1))

        # Text branch.
        # Explicit missingness mask is applied AFTER the biased projection so a missing note
        # is represented as the exact zero vector required by the specification.
        m = text_missing.unsqueeze(-1).to(dtype=h_struct.dtype)
        if self.ablation.use_text:
            h_text = self.text_proj(text)
            h_text = h_text * (1.0 - m)
        else:
            h_text = torch.zeros((x.shape[0], 256), device=x.device, dtype=h_struct.dtype)


        # Demographic branch.
        if self.ablation.use_demographics:
            h_demo = self.demo_proj(demographics)
        else:
            h_demo = torch.zeros((x.shape[0], 256), device=x.device, dtype=h_struct.dtype)

        gate_parts = [h_struct, h_text, h_demo]
        direct_parts = [h_struct, h_text, h_demo]
        if self.ablation.use_text_missing_flag:
            gate_parts.append(m)
            direct_parts.append(m)
        gate_input = torch.cat(gate_parts, dim=-1)
        direct = torch.cat(direct_parts, dim=-1)

        if self.ablation.use_gate:
            g = self.gate(gate_input)
            h_gated = g * h_text + (1.0 - g) * h_struct + h_demo
        else:
            h_gated = h_text + h_struct + h_demo

        fused = self.fusion(torch.cat([h_gated, direct], dim=-1))
        return fused, h_struct, h_text, h_demo

    def forward(self, x: torch.Tensor, demographics: torch.Tensor, text: torch.Tensor, text_missing: torch.Tensor) -> Dict[str, torch.Tensor]:
        h, _, _, _ = self.encode(x, demographics, text, text_missing)
        p_mort_logit = self.mortality(h)
        p_mort = torch.sigmoid(p_mort_logit)

        mu_los = self.los_head(h, p_mort)

        return {
            "mortality_24h": p_mort_logit,
            "ventilation_onset_24h": self.ventilation(h),
            "vasopressor_onset_24h": self.vasopressor(h),
            "readmission_30d_24h": self.readmission(h),
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


def total_multitask_loss(model: ProposedModel, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
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
def collect_predictions(model: ProposedModel, data: FrozenData, indices: np.ndarray, device: torch.device, batch_size: int) -> Dict[str, np.ndarray]:
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


def run_epoch_train(model: ProposedModel, data: FrozenData, indices: np.ndarray, device: torch.device, params: Dict[str, Any]) -> float:
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
def run_validation(model: ProposedModel, data: FrozenData, indices: np.ndarray, device: torch.device, params: Dict[str, Any], bootstrap_reps: int = 0) -> Tuple[float, float, pd.DataFrame]:
    pred = collect_predictions(model, data, indices, device, int(params["batch_size"]))
    metrics, _ = evaluate_arrays(data, indices, pred, reps=bootstrap_reps, split_name="val")
    score = validation_selection_score(metrics)
    los_row = metrics[metrics["task"] == "los_hours_24h"]
    los_mae = float(los_row["mae"].iloc[0]) if not los_row.empty else np.nan
    return score, los_mae, metrics


def train_model(
    data: FrozenData,
    params: Dict[str, Any],
    device: torch.device,
    ablation: AblationSpec,
    run_dir: Optional[Path],
) -> Tuple[ProposedModel, float, int, pd.DataFrame]:
    set_seed(SEED)
    model = ProposedModel(dropout=float(params["dropout"]), ablation=ablation).to(device)
    optimizer = build_optimizer(model, float(params["lr"]), float(params["weight_decay"]))
    steps_per_epoch = math.ceil(len(data.train_idx) / int(params["batch_size"]))
    scheduler = build_scheduler(
        optimizer,
        str(params["scheduler"]),
        int(params["max_epochs"]),
        steps_per_epoch,
        float(params["lr"]),
    )
    train_params = dict(params)
    train_params["optimizer"] = optimizer
    train_params["scheduler_obj"] = scheduler

    best_state = None
    best_score = float("-inf")
    best_epoch = 0
    no_improve = 0
    history_rows: List[Dict[str, Any]] = []

    for epoch in range(1, int(params["max_epochs"]) + 1):
        t0 = time.time()
        train_loss = run_epoch_train(model, data, data.train_idx, device, train_params)
        val_score, val_los_mae, _ = run_validation(
            model, data, data.val_idx, device, params, bootstrap_reps=0
        )
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
            f"[{ablation.name}] epoch={epoch:02d} train_loss={train_loss:.6f} "
            f"val_mean_AUPRC={val_score:.6f} val_LOS_MAE={val_los_mae:.4f}",
            flush=True,
        )

        if val_score > best_score:
            best_score = val_score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= int(params["patience"]):
            print(f"[{ablation.name}] early stopping at epoch {epoch}.", flush=True)
            break

    if best_state is None:
        raise RuntimeError("No valid proposed-model checkpoint selected.")
    model.load_state_dict(best_state)

    history = pd.DataFrame(history_rows)
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "params": params,
                "ablation": asdict(ablation),
                "best_epoch": best_epoch,
                "best_val_selection_score": best_score,
            },
            run_dir / "checkpoint.pt",
        )
        history.to_csv(run_dir / "training_history.csv", index=False)
    return model, best_score, best_epoch, history


# ============================================================================
# 8. METADATA / OUTPUTS
# ============================================================================
def architecture_manifest(ablation: AblationSpec, params: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "model": "Pipeline A V2 Final Proposed Model",
        "architecture_status": "locked",
        "input_contract": {
            "structured": [24, 110],
            "demographics": 4,
            "text_embedding": 768,
            "text_missing_flag": True,
        },
        "structured_encoder": {
            "projection": "110->256",
            "transformer_layers": 2,
            "d_model": 256,
            "heads": 4,
            "ffn": 512,
            "dropout": params["dropout"],
            "learned_position": 24,
            "pooling": "attention_pool + last_hour",
            "last_hour_index": -1,
            "pool_projection": "512->256",
        },
        "text_encoder": "768->256->256, frozen BioClinicalBERT embedding input",
        "demographic_encoder": "4->64->256",
        "fusion": {
            "gate": "sigmoid MLP over structured + text + demographic + text_missing",
            "gated_formula": "g*text + (1-g)*structured + demographic",
            "direct_path": "[structured,text,demographic,text_missing]",
            "final_shared_dim": 256,
        },
        "task_towers": "256->128->1",
        "los": {
            "experts": "survivor + death",
            "gate": "mortality probability",
            "gradient": "no stop-gradient",
            "target": "log1p(max(LOS_hours-24,0))",
            "inverse": "24 + expm1(predicted_excess_log1p)",
            "loss": "masked L1 on shifted log1p target",
        },
        "deterioration": "evaluation-only noisy-OR of mortality, ventilation, vasopressor",
        "multitask": "uncertainty-weighted five-task objective unless ablated",
        "exclusions": ["Model29 progression GRU", "delta branch", "Model30/31 cross-attention", "Model32 MoE"],
        "ablation": asdict(ablation),
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


def save_run_outputs(
    data: FrozenData,
    model: ProposedModel,
    indices_val: np.ndarray,
    indices_test: np.ndarray,
    device: torch.device,
    params: Dict[str, Any],
    run_dir: Path,
    best_score: float,
    best_epoch: int,
    mode: str,
    ablation: AblationSpec,
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

    stratified = text_stratified_stats(all_predictions)
    stratified.to_csv(run_dir / "text_availability_stratified_metrics.csv", index=False)

    metadata = {
        **runtime_metadata(device),
        "seed": SEED,
        "mode": mode,
        "best_epoch": int(best_epoch),
        "validation_selection_score": float(best_score),
        "test_used_for_selection": False,
        "fixed_tuned_phase_separation": True,
        "n_optuna_trials_target": None,
        "n_optuna_trials_completed": None,
        "params": params,
        "first_run_protocol": FIRST_RUN,
        "input_paths": {
            "structured": str(STRUCTURED_PATH),
            "demographics": str(DEMOGRAPHICS_PATH),
            "text": str(TEXT_EMBEDDING_PATH),
            "alignment": str(ALIGNMENT_PATH),
            "outcomes": str(OUTCOMES_PATH),
        },
        "split_counts": {
            "train": EXPECTED_TRAIN,
            "val": EXPECTED_VAL,
            "test": EXPECTED_TEST,
        },
        "text_present_count": EXPECTED_TEXT_PRESENT,
        "text_missing_count": EXPECTED_N - EXPECTED_TEXT_PRESENT,
        "selection_metric": "mean validation AUPRC across mortality, ventilation, vasopressor, readmission",
        "los_training_transform": "log1p(max(los_hours_24h - 24, 0))",
        "los_loss": "masked L1 on shifted log1p target",
        "los_reporting_transform": "24 + expm1(predicted_excess_log1p)",
        "derived_deterioration": "evaluation-only noisy-OR",
        "architecture": architecture_manifest(ablation, params),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    (run_dir / "architecture_manifest.json").write_text(
        json.dumps(architecture_manifest(ablation, params), indent=2), encoding="utf-8"
    )

    return {
        "validation_selection_score": float(best_score),
        "best_epoch": int(best_epoch),
        "run_dir": str(run_dir),
    }


# ============================================================================
# 9. FIXED RUN
# ============================================================================
def first_run_params() -> Dict[str, Any]:
    return dict(FIRST_RUN)


def run_fixed(data: FrozenData, device: torch.device, bootstrap_reps: int) -> Dict[str, Any]:
    validate_gpu_bf16(device, "fixed run")
    run_dir = RESULTS_ROOT / "fixed" / "final_selected_configuration"
    params = first_run_params()
    ablation = ABLATIONS[0]
    t0 = time.time()
    model, score, epoch, history = train_model(data, params, device, ablation, run_dir)
    result = save_run_outputs(
        data, model, data.val_idx, data.test_idx, device, params, run_dir,
        score, epoch, "fixed", ablation, bootstrap_reps,
    )
    result["training_seconds"] = time.time() - t0
    meta_path = run_dir / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["training_seconds"] = result["training_seconds"]
    meta["first_run"] = True
    meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


# ============================================================================
# 10. OPTUNA TUNING (30 TRIALS)
# ============================================================================
def suggest_params(trial: "optuna.Trial") -> Dict[str, Any]:
    # Architecture dimensions are locked. Only training hyperparameters are tuned.
    return {
        "lr": trial.suggest_float("lr", 1e-4, 1e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 5e-4, log=True),
        "dropout": trial.suggest_float("dropout", 0.10, 0.25),
        "grad_clip": trial.suggest_float("grad_clip", 0.5, 1.5),
        "scheduler": trial.suggest_categorical("scheduler", ["cosine", "onecycle"]),
        "patience": trial.suggest_int("patience", 6, 10),
        "batch_size": 256,
        "max_epochs": 50,
        "seed": SEED,
        "amp": "BF16",
    }


def trial_worker(params_json: str, result_json: str) -> None:
    require_torch()
    params = json.loads(params_json)
    data = load_frozen_v2()
    device = get_device()
    validate_gpu_bf16(device, "Optuna trial worker")
    ablation = ABLATIONS[0]
    print(
        f"[TRIAL-WORKER] pid={os.getpid()} device={device} gpu={torch.cuda.get_device_name(0)} "
        f"params={params}", flush=True
    )
    model, score, epoch, _ = train_model(data, params, device, ablation, run_dir=None)
    payload = {"score": float(score), "best_epoch": int(epoch)}
    Path(result_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    del model
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        except Exception:
            pass


def run_tuned(data: FrozenData, device: torch.device, n_trials: int, bootstrap_reps: int) -> Dict[str, Any]:
    require_optuna()
    validate_gpu_bf16(device, "tuned run")
    tuned_root = RESULTS_ROOT / "tuned"
    tuned_root.mkdir(parents=True, exist_ok=True)
    tuning_start = time.time()
    study_path = tuned_root / "study.db"
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        study_name="pipeline_a_v2_proposed_model",
        storage=f"sqlite:///{study_path.as_posix()}",
        load_if_exists=True,
    )

    def n_complete(s):
        return sum(t.state == optuna.trial.TrialState.COMPLETE for t in s.trials)

    def objective(trial: "optuna.Trial") -> float:
        params = suggest_params(trial)
        trial_result = tuned_root / f".trial_{trial.number}_result.json"
        if trial_result.exists():
            trial_result.unlink()
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--mode", "trial-worker",
            "--trial-params-json", json.dumps(params, separators=(",", ":")),
            "--trial-result-json", str(trial_result),
        ]
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env, check=False)
        if proc.returncode != 0 or not trial_result.exists():
            raise RuntimeError(f"Trial worker failed: trial={trial.number} returncode={proc.returncode}")
        result = json.loads(trial_result.read_text(encoding="utf-8"))
        try:
            trial_result.unlink()
        except OSError:
            pass
        trial.set_user_attr("best_epoch", int(result["best_epoch"]))
        trial.set_user_attr("params", params)
        return float(result["score"])

    while n_complete(study) < int(n_trials):
        before = n_complete(study)
        study.optimize(objective, n_trials=1, gc_after_trial=False, catch=(RuntimeError, ValueError), show_progress_bar=False)
        after = n_complete(study)
        print(f"[OPTUNA] completed {after}/{n_trials} trials (records={len(study.trials)}; delta={after-before})", flush=True)
        if len(study.trials) > 2 * int(n_trials) + 5:
            raise RuntimeError("Optuna safety cap exceeded before reaching requested completed trials.")

    trial_rows = []
    for t in study.trials:
        trial_rows.append({
            "number": t.number,
            "state": str(t.state),
            "value": t.value,
            **t.params,
        })
    pd.DataFrame(trial_rows).to_csv(tuned_root / "optuna_trials.csv", index=False)
    (tuned_root / "study_storage.txt").write_text(str(study_path), encoding="utf-8")

    best = study.best_trial
    best_params = dict(best.user_attrs.get("params") or best.params)
    final_dir = tuned_root / "final_selected_configuration"
    final_dir.mkdir(parents=True, exist_ok=True)

    model, score, epoch, history = train_model(data, best_params, device, ABLATIONS[0], final_dir)
    result = save_run_outputs(
        data, model, data.val_idx, data.test_idx, device, best_params, final_dir,
        score, epoch, "tuned", ABLATIONS[0], bootstrap_reps,
    )
    result.update({
        "best_trial": int(best.number),
        "optuna_best_value": float(best.value),
        "n_optuna_trials_target": int(n_trials),
        "n_optuna_trials_completed": int(n_complete(study)),
        "tuning_seconds": time.time() - tuning_start,
    })

    (tuned_root / "best_params.json").write_text(json.dumps(best_params, indent=2), encoding="utf-8")
    meta_path = final_dir / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.update({
        "n_optuna_trials_target": int(n_trials),
        "n_optuna_trials_completed": int(n_complete(study)),
        "optuna_best_trial_number": int(best.number),
        "optuna_best_value": float(best.value),
        "tuning_seconds": result["tuning_seconds"],
        "test_used_for_selection": False,
        "tuning_note": "Architecture dimensions locked; Optuna tunes training hyperparameters only.",
    })
    meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    print(json.dumps(result, indent=2), flush=True)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


# ============================================================================
# 11. ABLATION RUNS
# ============================================================================
def run_ablations(data: FrozenData, device: torch.device, bootstrap_reps: int) -> pd.DataFrame:
    validate_gpu_bf16(device, "ablation run")
    root = RESULTS_ROOT / "ablations"
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    params = first_run_params()

    for ablation in ABLATIONS[1:]:
        print("=" * 80)
        print(f"ABLATION: {ablation.name}")
        print("=" * 80)
        run_dir = root / ablation.name
        model, score, epoch, _ = train_model(data, params, device, ablation, run_dir)
        result = save_run_outputs(
            data, model, data.val_idx, data.test_idx, device, params, run_dir,
            score, epoch, "ablation", ablation, bootstrap_reps,
        )
        test_metrics = pd.read_csv(run_dir / "metrics.csv")
        test_bin = test_metrics[(test_metrics["split"] == "test") & (test_metrics["task"].isin(BINARY_TASKS))]
        mean_auprc = float(test_bin["auprc"].mean())
        los = test_metrics[(test_metrics["split"] == "test") & (test_metrics["task"] == "los_hours_24h")]
        los_mae = float(los["mae"].iloc[0]) if not los.empty else np.nan
        rows.append({
            "ablation": ablation.name,
            "validation_selection_score": score,
            "best_epoch": epoch,
            "test_mean_binary_auprc": mean_auprc,
            "test_los_mae": los_mae,
        })
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = pd.DataFrame(rows)
    summary.to_csv(root / "ablation_summary.csv", index=False)
    return summary


# ============================================================================
# 12. STATS MODE ON SAVED PREDICTIONS
# ============================================================================
def stats_saved_predictions(pred_path: Path, out_dir: Path, bootstrap_reps: int) -> pd.DataFrame:
    df = pd.read_csv(pred_path)
    required = {"split", "task", "target", "validity", "prediction", "subject_id"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Prediction file missing columns: {missing}")

    rows = []
    cal_tables = []
    for split in sorted(df["split"].dropna().unique().tolist()):
        sdf = df[df["split"] == split]
        for task in REPORT_TASKS:
            tdf = sdf[sdf["task"] == task].copy()
            tdf = tdf[tdf["validity"].astype(bool)]
            tdf = tdf[np.isfinite(tdf["target"]) & np.isfinite(tdf["prediction"])]
            y = tdf["target"].to_numpy(float)
            p = tdf["prediction"].to_numpy(float)
            subjects = pd.to_numeric(tdf["subject_id"], errors="raise").to_numpy(np.int64)
            row = bootstrap_metric_ci(
                y,
                p,
                task,
                bootstrap_reps,
                stable_seed(f"stats|{split}|{task}"),
                subject_ids=subjects,
            )
            rows.append({"split": split, "task": task, **row})
            if task in BINARY_TASKS or task == "deterioration_24h":
                cb = calibration_bins(y.astype(int), p)
                if not cb.empty:
                    cb.insert(0, "split", split)
                    cb.insert(1, "task", task)
                    cal_tables.append(cb)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "statistical_evaluation.csv", index=False)
    pd.concat(cal_tables, ignore_index=True).to_csv(out_dir / "calibration_bins_stats.csv", index=False) if cal_tables else None
    stratified = text_stratified_stats(df)
    stratified.to_csv(out_dir / "text_availability_stratified_metrics.csv", index=False)
    return metrics


def run_stats(bootstrap_reps: int) -> None:
    configs = [
        RESULTS_ROOT / "fixed" / "final_selected_configuration" / "test_predictions.csv",
        RESULTS_ROOT / "tuned" / "final_selected_configuration" / "test_predictions.csv",
    ]
    ablation_root = RESULTS_ROOT / "ablations"
    configs.extend(sorted(ablation_root.glob("*/test_predictions.csv")))
    for pred in configs:
        if not pred.exists():
            print(f"[STATS] SKIP missing {pred}", flush=True)
            continue
        print(f"[STATS] {pred}", flush=True)
        stats_saved_predictions(pred, pred.parent, bootstrap_reps)


# ============================================================================
# 13. BENCHMARK PAIRED COMPARISONS + HOLM
# ============================================================================
def stable_seed(label: str) -> int:
    digest = hashlib.sha256((str(SEED) + "|" + label).encode()).hexdigest()
    return int(digest[:8], 16)


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    out = np.full_like(p, np.nan)
    finite_idx = np.flatnonzero(np.isfinite(p))
    if len(finite_idx) == 0:
        return out
    order = finite_idx[np.argsort(p[finite_idx])]
    m = len(order)
    running = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * p[idx]
        running = max(running, adj)
        out[idx] = min(1.0, running)
    return out


def load_test_task(path: Path, task: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "split" in df.columns:
        df = df[df["split"].astype(str).str.lower() == "test"].copy()
    df = df[df["task"] == task].copy()
    required = {"stay_id", "subject_id", "target", "prediction", "validity"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"{path} missing comparison columns: {missing}")
    df = df[df["validity"].astype(bool)].copy()
    df["stay_id"] = pd.to_numeric(df["stay_id"], errors="raise").astype(np.int64)
    df["subject_id"] = pd.to_numeric(df["subject_id"], errors="raise").astype(np.int64)
    return df[["stay_id", "subject_id", "target", "prediction"]]


def paired_bootstrap_compare(
    proposed: pd.DataFrame,
    baseline: pd.DataFrame,
    task: str,
    reps: int,
    seed: int,
) -> Dict[str, Any]:
    m = proposed.merge(
        baseline,
        on=["stay_id", "subject_id"],
        how="inner",
        suffixes=("_p", "_b"),
        validate="one_to_one",
    )
    if m.empty:
        return {"n_rows": 0, "n_subjects": 0, "delta": np.nan, "ci95_low": np.nan, "ci95_high": np.nan, "p_two_sided": np.nan}

    # Verify the truth labels agree within numeric tolerance.
    if not np.allclose(m["target_p"].to_numpy(float), m["target_b"].to_numpy(float), atol=1e-4, rtol=0.0, equal_nan=False):
        raise ValueError(f"Truth mismatch for paired comparison task={task}.")

    subjects = m["subject_id"].to_numpy(np.int64)
    unique_subjects = np.unique(subjects)
    subject_to_rows = {s: np.flatnonzero(subjects == s) for s in unique_subjects}
    y = m["target_p"].to_numpy(float)
    p_prop = m["prediction_p"].to_numpy(float)
    p_base = m["prediction_b"].to_numpy(float)

    if task in BINARY_TASKS:
        point_p = safe_auprc(y.astype(int), p_prop)
        point_b = safe_auprc(y.astype(int), p_base)
    else:
        point_p = mean_absolute_error(y, p_prop)
        point_b = mean_absolute_error(y, p_base)
    point_delta = point_p - point_b

    rng = np.random.default_rng(seed)
    deltas = []
    n_subjects = len(unique_subjects)
    for _ in range(reps):
        sampled = rng.choice(unique_subjects, size=n_subjects, replace=True)
        row_parts = [subject_to_rows[s] for s in sampled]
        idx = np.concatenate(row_parts)
        yy = y[idx]
        pp = p_prop[idx]
        pb = p_base[idx]
        if task in BINARY_TASKS:
            a = safe_auprc(yy.astype(int), pp)
            b = safe_auprc(yy.astype(int), pb)
        else:
            a = mean_absolute_error(yy, pp)
            b = mean_absolute_error(yy, pb)
        if np.isfinite(a) and np.isfinite(b):
            deltas.append(float(a - b))

    arr = np.asarray(deltas, dtype=float)
    # For metric differences, use a two-sided bootstrap sign probability.
    p_hi = float(np.mean(arr >= 0.0)) if len(arr) else np.nan
    p_lo = float(np.mean(arr <= 0.0)) if len(arr) else np.nan
    p_two = min(1.0, 2.0 * min(p_hi, p_lo)) if len(arr) else np.nan
    return {
        "n_rows": int(len(m)),
        "n_subjects": int(n_subjects),
        "proposed_metric": float(point_p),
        "baseline_metric": float(point_b),
        "delta_proposed_minus_baseline": float(point_delta),
        "ci95_low": float(np.quantile(arr, 0.025)) if len(arr) else np.nan,
        "ci95_high": float(np.quantile(arr, 0.975)) if len(arr) else np.nan,
        "p_two_sided": p_two,
    }


def compare_benchmark(proposed_prediction_path: Path, refs: Sequence[str], reps: int, compare_all: bool = False) -> pd.DataFrame:
    if compare_all:
        refs = sorted(BENCHMARK_TUNED_PATHS.keys())
    proposed_df = pd.read_csv(proposed_prediction_path)
    proposed_df = proposed_df[proposed_df["split"].astype(str).str.lower() == "test"].copy()

    rows = []
    for ref in refs:
        path = BENCHMARK_TUNED_PATHS.get(ref)
        if path is None or not path.exists():
            print(f"[COMPARE] SKIP {ref}: prediction file missing: {path}", flush=True)
            continue
        for task in BINARY_TASKS + ("los_hours_24h",):
            p = load_test_task(proposed_prediction_path, task)
            b = load_test_task(path, task)
            result = paired_bootstrap_compare(
                p, b, task, reps=reps, seed=stable_seed(f"compare|{ref}|{task}"),
            )
            rows.append({
                "reference_model_id": ref,
                "reference_model_name": BENCHMARK_NAMES.get(ref, ref),
                "task": task,
                **result,
            })

    comp = pd.DataFrame(rows)
    if not comp.empty:
        comp["holm_p_two_sided"] = np.nan
        for task in comp["task"].unique():
            mask = comp["task"] == task
            comp.loc[mask, "holm_p_two_sided"] = holm_adjust(comp.loc[mask, "p_two_sided"].to_numpy(float))
        comp["significant_holm_0_05"] = comp["holm_p_two_sided"] < 0.05

    out_dir = RESULTS_ROOT / "benchmark_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    comp.to_csv(out_dir / "paired_bootstrap_comparisons_holm.csv", index=False)
    return comp


# ============================================================================
# 14. AUDIT / MAIN
# ============================================================================
def save_contract_audit(data: FrozenData) -> Dict[str, Any]:
    audit = contract_audit(data)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULTS_ROOT / "frozen_v2_contract_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def parse_args():
    p = argparse.ArgumentParser(description="Pipeline A V2 Final Proposed Model")
    p.add_argument("--mode", required=True, choices=["audit", "fixed", "tuned", "ablations", "stats", "compare-benchmark", "all", "trial-worker"])
    p.add_argument("--n-trials", type=int, default=30)
    p.add_argument("--bootstrap-reps", type=int, default=2000)
    p.add_argument("--compare-reps", type=int, default=5000)
    p.add_argument("--compare-all-33", action="store_true")
    p.add_argument("--candidate-refs", nargs="*", default=DEFAULT_CANDIDATE_REFS)
    p.add_argument("--trial-params-json", default=None, help=argparse.SUPPRESS)
    p.add_argument("--trial-result-json", default=None, help=argparse.SUPPRESS)
    return p.parse_args()


def run_main():
    args = parse_args()
    if args.n_trials < 1:
        raise ValueError("--n-trials must be >= 1")
    if args.bootstrap_reps < 0 or args.compare_reps < 1:
        raise ValueError("Invalid bootstrap repetition count")

    require_torch()
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    if args.mode == "trial-worker":
        if not args.trial_params_json or not args.trial_result_json:
            raise ValueError("trial-worker requires --trial-params-json and --trial-result-json")
        trial_worker(args.trial_params_json, args.trial_result_json)
        return

    set_seed(SEED)
    data = load_frozen_v2()
    audit = save_contract_audit(data)
    print("=" * 90)
    print("PIPELINE A V2 — FINAL PROPOSED MODEL")
    print("=" * 90)
    print(json.dumps(audit, indent=2))

    if args.mode == "audit":
        return

    if args.mode == "stats":
        run_stats(args.bootstrap_reps)
        return

    if args.mode == "compare-benchmark":
        pred_path = RESULTS_ROOT / "tuned" / "final_selected_configuration" / "test_predictions.csv"
        if not pred_path.exists():
            pred_path = RESULTS_ROOT / "fixed" / "final_selected_configuration" / "test_predictions.csv"
        if not pred_path.exists():
            raise FileNotFoundError("No proposed-model prediction file found. Run fixed or tuned first.")
        refs = list(args.candidate_refs)
        comp = compare_benchmark(pred_path, refs, reps=args.compare_reps, compare_all=args.compare_all_33)
        print(comp.to_string(index=False))
        return

    device = get_device()
    print(f"device={device}")
    if device.type == "cuda":
        print(f"GPU={torch.cuda.get_device_name(0)}")
        print(f"BF16 supported={torch.cuda.is_bf16_supported()}")

    if args.mode == "fixed":
        run_fixed(data, device, args.bootstrap_reps)
        return

    if args.mode == "tuned":
        run_tuned(data, device, args.n_trials, args.bootstrap_reps)
        return

    if args.mode == "ablations":
        summary = run_ablations(data, device, args.bootstrap_reps)
        print(summary.to_string(index=False))
        return

    if args.mode == "all":
        run_fixed(data, device, args.bootstrap_reps)
        run_tuned(data, device, args.n_trials, args.bootstrap_reps)
        run_ablations(data, device, args.bootstrap_reps)
        run_stats(args.bootstrap_reps)
        pred_path = RESULTS_ROOT / "tuned" / "final_selected_configuration" / "test_predictions.csv"
        if pred_path.exists():
            compare_benchmark(pred_path, args.candidate_refs, reps=args.compare_reps, compare_all=args.compare_all_33)
        return

    raise ValueError(args.mode)


if __name__ == "__main__":
    run_main()
