#!/usr/bin/env python3
"""
Pipeline A V2 — Family 7: Advanced Multimodal Architectures (28–32)

Single reusable runner for:
    28 Hybrid Mamba-Transformer
    29 Multimodal Progression
    30 Cross-Attention Multimodal
    31 Gated Fusion Cross-Attn
    32 MoE Modality Fusion

Modes:
    --mode fixed        : run registered fixed baselines
    --mode tuned        : isolated Optuna tuning, then train selected configuration
    --mode stats        : evaluate existing fixed + tuned prediction tables
    --mode trial-worker : internal one-trial child process

ROCm / MI210 compatibility policy:
    * Model 28 uses a pure-PyTorch Mamba-style selective SSM subcomponent; no Triton or mamba_ssm dependency.
    * The explicit Mamba recurrence is evaluated in fp32 before returning to BF16.
    * Advanced attention/fusion/MoE tuning batches are capped at 256/512 for ROCm allocator stability.
    * Each Optuna trial runs in an isolated child process, so a HIP fault does not poison the parent study.

These are compact V2 adaptations of advanced multimodal architectures, not claims of
literal reproduction of every upstream implementation detail. Family 7 is the first
benchmark family that adds clinical text to the frozen structured+demographic inputs.
The external model contract is: structured [B,24,110] + demographics [B,4] +
clinical text [B,768] + text_missing [B] -> five multitask heads.

Frozen V2 contract:
    structured_24h_processed.npy      [74572, 24, 110]
    demographics_24h_processed.npy    [74572, 4]
    patient-level splits: 52268 / 7417 / 14887
    validation selection = mean AUPRC across mortality, ventilation, vasopressor, readmission
    LOS model-side transform = log1p(max(LOS_hours - 24, 0))

Family 7 multimodal extension:
    structured [B,24,110] + demographics [B,4] + clinical text embedding [B,768]
    + text_missing [B] derived from clinical_note_embeddings.parquet.

Shared benchmark seed: 20260923
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import platform
import random
import sys
import subprocess
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
    roc_auc_score,
)


# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path("/home/abdullah2025/fyp_project")
V2_DIR = PROJECT_ROOT / "outputs" / "pipeline_a_v2"

ALIGNMENT_PATH = V2_DIR / "a5_alignment_index_24h.csv"
STRUCTURED_PATH = V2_DIR / "structured_24h_processed.npy"
DEMOGRAPHICS_PATH = V2_DIR / "demographics_24h_processed.npy"
INPUT_METADATA_PATH = V2_DIR / "final_model_input_metadata_24h.csv"
INPUT_DEFINITION_PATH = V2_DIR / "final_model_input_definition_24h.json"
OUTCOMES_PATH = V2_DIR / "targets_24h" / "outcomes_24h_primary_cohort.csv"
TEXT_EMBEDDING_PATH = V2_DIR / "clinical_note_embeddings.parquet"
TEXT_EMBEDDING_DIM = 768
EXPECTED_TEXT_PRESENT = 46_968

RESULTS_ROOT = PROJECT_ROOT / "outputs" / "baseline_family_07_advanced_multimodal_28_32"

SEED = 20260923
BOOTSTRAP_REPS = 1000
EXPECTED_N = 74_572
EXPECTED_TRAIN = 52_268
EXPECTED_VAL = 7_417
EXPECTED_TEST = 14_887
EXPECTED_SHAPE = (EXPECTED_N, 24, 110)
BINARY_TASKS = (
    "mortality_24h",
    "ventilation_onset_24h",
    "vasopressor_onset_24h",
    "readmission_30d_24h",
)
PRIMARY_TASKS = BINARY_TASKS + ("los_hours_24h",)
VALID_COL = {
    "mortality_24h": "mortality_24h_valid",
    "ventilation_onset_24h": "ventilation_onset_24h_valid",
    "vasopressor_onset_24h": "vasopressor_onset_24h_valid",
    "readmission_30d_24h": "readmission_30d_24h_valid",
    "los_hours_24h": "los_hours_24h_valid",
}


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    name: str
    fixed: Dict[str, Any]
    search_space: Dict[str, Any]


MODEL_NAMES = {
    "28": "Hybrid Mamba-Transformer",
    "29": "Multimodal Progression",
    "30": "Cross-Attention Multimodal",
    "31": "Gated Fusion Cross-Attn",
    "32": "MoE Modality Fusion",
}


MODEL_CONFIGS: Dict[str, ModelConfig] = {
    "28": ModelConfig(
        model_id="28", name="Hybrid Mamba-Transformer",
        fixed={"d_model": 256, "d_state": 16, "d_conv": 4, "expand": 2, "mamba_layers": 2, "transformer_layers": 2, "heads": 8, "dropout": 0.15, "lr": 8e-4},
        search_space={"d_model": "int[128,384] step=8", "d_state": "categorical[8,16,32]", "expand": "categorical[2,4]", "mamba_layers": "int[1,3]", "transformer_layers": "int[1,4]", "heads": "categorical[4,8]", "dropout": "float[0.05,0.30]", "lr": "float[1e-4,2e-3] log"},
    ),
    "29": ModelConfig(
        model_id="29", name="Multimodal Progression",
        fixed={"d_model": 256, "progression_hidden": 256, "layers": 2, "dropout": 0.15, "lr": 8e-4},
        search_space={"d_model": "int[128,384] step=8", "progression_hidden": "int[128,384] step=8", "layers": "int[1,4]", "dropout": "float[0.05,0.30]", "lr": "float[1e-4,2e-3] log"},
    ),
    "30": ModelConfig(
        model_id="30", name="Cross-Attention Multimodal",
        fixed={"d_model": 256, "layers": 3, "heads": 8, "dropout": 0.15, "lr": 8e-4},
        search_space={"d_model": "int[128,384] step=8", "layers": "int[2,4]", "heads": "categorical[4,8]", "dropout": "float[0.05,0.30]", "lr": "float[1e-4,2e-3] log"},
    ),
    "31": ModelConfig(
        model_id="31", name="Gated Fusion Cross-Attn",
        fixed={"d_model": 256, "layers": 2, "heads": 8, "gate_hidden": 128, "dropout": 0.15, "lr": 8e-4},
        search_space={"d_model": "int[128,384] step=8", "layers": "int[1,3]", "heads": "categorical[4,8]", "gate_hidden": "int[64,256] step=32", "dropout": "float[0.05,0.30]", "lr": "float[1e-4,2e-3] log"},
    ),
    "32": ModelConfig(
        model_id="32", name="MoE Modality Fusion",
        fixed={"d_model": 256, "experts": 4, "top_k": 2, "expert_expand": 2, "load_balance_coef": 0.01, "dropout": 0.15, "lr": 8e-4},
        search_space={"d_model": "int[128,384] step=8", "experts": "categorical[2,4,6]", "top_k": "categorical[1,2]", "expert_expand": "categorical[2,4]", "load_balance_coef": "categorical[0.0,0.01,0.05]", "dropout": "float[0.05,0.30]", "lr": "float[1e-4,2e-3] log"},
    ),
}


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


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def require_torch() -> None:
    if _TORCH_IMPORT_ERROR is not None:
        raise RuntimeError(f"PyTorch import failed: {_TORCH_IMPORT_ERROR}")


def require_optuna() -> None:
    if optuna is None:
        raise RuntimeError("Optuna is required for --mode tuned. Install optuna in the active environment.")


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def get_device() -> torch.device:
    require_torch()
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def require_gpu_protocol(device: torch.device, context: str) -> None:
    require_torch()
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(
            f"Family 7 {context} requires an allocated CUDA/ROCm GPU; "
            f"got device={device}. CPU fallback is disabled for benchmark training/tuning."
        )
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            f"Family 7 {context} requires BF16-capable GPU training for protocol comparability; "
            "torch.cuda.is_bf16_supported() returned False."
        )


def amp_enabled(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    supported = bool(torch.cuda.is_bf16_supported())
    if not supported:
        raise RuntimeError("Family 7 requires BF16 AMP on the GPU for protocol comparability; torch.cuda.is_bf16_supported() returned False.")
    return True


def validity_mask(df: pd.DataFrame, task: str) -> np.ndarray:
    col = VALID_COL[task]
    if col not in df.columns:
        raise KeyError(f"Required frozen V2 validity column '{col}' is missing for '{task}'.")
    return df[col].to_numpy(dtype=np.int8) == 1


def load_text_embeddings(aligned_stay_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    require_file(TEXT_EMBEDDING_PATH)
    emb = pd.read_parquet(TEXT_EMBEDDING_PATH)
    required = {"stay_id", "text_embedding"}
    missing = sorted(required - set(emb.columns))
    if missing:
        raise KeyError(f"Clinical text embedding parquet missing columns: {missing}")
    if emb["stay_id"].duplicated().any():
        raise ValueError("Clinical text embeddings contain duplicate stay_id rows.")
    row_of = pd.Series(np.arange(len(aligned_stay_ids), dtype=np.int64), index=aligned_stay_ids)
    present = emb[emb["stay_id"].isin(row_of.index)].copy()
    rows = row_of.loc[present["stay_id"]].to_numpy(dtype=np.int64)
    text = np.zeros((len(aligned_stay_ids), TEXT_EMBEDDING_DIM), dtype=np.float32)
    text_missing = np.ones(len(aligned_stay_ids), dtype=np.float32)
    if len(present):
        vectors = []
        for vec in present["text_embedding"].tolist():
            arr = np.asarray(vec, dtype=np.float32).reshape(-1)
            if arr.shape[0] != TEXT_EMBEDDING_DIM:
                raise ValueError(f"Expected text embedding dimension {TEXT_EMBEDDING_DIM}, got {arr.shape[0]}")
            vectors.append(arr)
        text[rows] = np.stack(vectors).astype(np.float32, copy=False)
        text_missing[rows] = 0.0
    if int((text_missing == 0).sum()) != EXPECTED_TEXT_PRESENT:
        raise ValueError(
            f"Expected {EXPECTED_TEXT_PRESENT} matched clinical-text stays in the frozen cohort; "
            f"found {int((text_missing == 0).sum())}."
        )
    if not np.isfinite(text).all():
        raise ValueError("Clinical text embeddings contain non-finite values after alignment.")
    return text, text_missing


def load_frozen_v2() -> FrozenData:
    for path in (ALIGNMENT_PATH, STRUCTURED_PATH, DEMOGRAPHICS_PATH, INPUT_METADATA_PATH, INPUT_DEFINITION_PATH, OUTCOMES_PATH, TEXT_EMBEDDING_PATH):
        require_file(path)
    idx_df = pd.read_csv(ALIGNMENT_PATH)
    outcomes_df = pd.read_csv(OUTCOMES_PATH)
    required_alignment = {"stay_id", "subject_id", "split_24h"}
    missing_alignment = sorted(required_alignment - set(idx_df.columns))
    if missing_alignment:
        raise KeyError(f"Missing alignment columns: {missing_alignment}")
    required_outcomes = {"stay_id", "subject_id", *PRIMARY_TASKS, *VALID_COL.values()}
    missing_outcomes = sorted(required_outcomes - set(outcomes_df.columns))
    if missing_outcomes:
        raise KeyError(f"Missing outcome columns: {missing_outcomes}")
    out_cols = ["stay_id", "subject_id", *PRIMARY_TASKS, *[c for c in outcomes_df.columns if c.endswith("_valid")]]
    outcomes_small = outcomes_df[list(dict.fromkeys(out_cols))].copy()
    if outcomes_small["stay_id"].duplicated().any():
        raise ValueError("Frozen outcomes contain duplicate stay_id rows.")
    merged = idx_df.merge(
        outcomes_small,
        on=["stay_id", "subject_id"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_outcome"),
    )
    if merged[list(PRIMARY_TASKS)].isna().any().any():
        raise ValueError("Frozen alignment/outcomes join contains missing primary targets.")
    structured = np.load(STRUCTURED_PATH, mmap_mode="r")
    demographics = np.load(DEMOGRAPHICS_PATH, mmap_mode="r")
    if tuple(structured.shape) != EXPECTED_SHAPE:
        raise ValueError(f"Unexpected structured shape {structured.shape}; expected {EXPECTED_SHAPE}.")
    if tuple(demographics.shape) != (EXPECTED_N, 4):
        raise ValueError(f"Unexpected demographics shape {demographics.shape}.")
    mask_sample = np.asarray(structured[:min(2000, EXPECTED_N), :, 28:56])
    mask_finite = mask_sample[np.isfinite(mask_sample)]
    if mask_finite.size and not np.isin(np.unique(mask_finite), [0.0, 1.0]).all():
        raise ValueError("Frozen structured mask channels 28:56 must contain only 0/1 values.")
    if len(merged) != EXPECTED_N:
        raise ValueError(f"Unexpected row count {len(merged)} != {EXPECTED_N}.")
    split = merged["split_24h"].astype(str).to_numpy()
    expected_counts = {"train": EXPECTED_TRAIN, "val": EXPECTED_VAL, "test": EXPECTED_TEST}
    actual_counts = pd.Series(split).value_counts().to_dict()
    if actual_counts != expected_counts:
        raise ValueError(f"Frozen split counts differ: {actual_counts} != {expected_counts}.")
    train_idx = np.flatnonzero(split == "train")
    val_idx = np.flatnonzero(split == "val")
    test_idx = np.flatnonzero(split == "test")
    targets = {task: merged[task].to_numpy(dtype=np.float32) for task in PRIMARY_TASKS}
    valid_arrays = {task: validity_mask(merged, task) for task in PRIMARY_TASKS}
    text, text_missing = load_text_embeddings(merged["stay_id"].to_numpy())
    return FrozenData(
        structured=structured, demographics=demographics, text=text, text_missing=text_missing,
        stay_id=merged["stay_id"].to_numpy(),
        subject_id=merged["subject_id"].to_numpy(), split=split, outcomes=merged, targets=targets,
        valid_arrays=valid_arrays, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
    )


class FrozenSequenceDataset(Dataset):
    def __init__(self, data: FrozenData, indices: np.ndarray) -> None:
        self.structured = data.structured
        self.demographics = data.demographics
        self.text = data.text
        self.text_missing = data.text_missing
        self.targets = data.targets
        self.valid_arrays = data.valid_arrays
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, pos: int):
        i = int(self.indices[pos])
        x = np.array(self.structured[i], dtype=np.float32, copy=True)
        d = np.array(self.demographics[i], dtype=np.float32, copy=True)
        text = np.array(self.text[i], dtype=np.float32, copy=True)
        text_missing = np.float32(self.text_missing[i])
        y = {task: np.float32(self.targets[task][i]) for task in PRIMARY_TASKS}
        valid = {task: np.int8(self.valid_arrays[task][i]) for task in PRIMARY_TASKS}
        return torch.from_numpy(x), torch.from_numpy(d), torch.from_numpy(text), torch.tensor(text_missing, dtype=torch.float32), y, valid


def make_loader(data: FrozenData, indices: np.ndarray, batch_size: int, shuffle: bool, pin_memory: bool) -> DataLoader:
    return DataLoader(FrozenSequenceDataset(data, indices), batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=pin_memory, drop_last=False)

# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------
class MultitaskHead(nn.Module):
    def __init__(self, representation_dim: int, dropout: float) -> None:
        super().__init__()
        head_dim = max(64, representation_dim // 2)
        self.trunk = nn.Sequential(
            nn.Linear(representation_dim, head_dim),
            nn.LayerNorm(head_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.mortality = nn.Linear(head_dim, 1)
        self.ventilation = nn.Linear(head_dim, 1)
        self.vasopressor = nn.Linear(head_dim, 1)
        self.readmission = nn.Linear(head_dim, 1)
        self.los_raw = nn.Linear(head_dim, 1)

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.trunk(h)
        return {
            "mortality_24h": self.mortality(z).squeeze(-1),
            "ventilation_onset_24h": self.ventilation(z).squeeze(-1),
            "vasopressor_onset_24h": self.vasopressor(z).squeeze(-1),
            "readmission_30d_24h": self.readmission(z).squeeze(-1),
            "los_excess_log1p": F.softplus(self.los_raw(z).squeeze(-1)),
        }


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x32 = x.float()
        rms = torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x32 * rms * self.weight.float()).to(dtype=x.dtype)


def split_modalities(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return x[..., :28], x[..., 28:56], x[..., 56:84], x[..., 84:110]


class PureMambaBlock(nn.Module):
    """ROCm-safe Mamba-style selective SSM used by Model 28.

    Pure PyTorch scan over T=24; no Triton or mamba_ssm dependency.
    The recurrence is explicitly evaluated in fp32 for BF16/ROCm stability.
    """
    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int, dropout: float) -> None:
        super().__init__()
        self.d_inner = d_model * expand
        self.d_state = d_state
        self.dt_rank = max(8, min(32, d_model // 16))
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner)
        self.conv = nn.Conv1d(self.d_inner, self.d_inner, d_conv, groups=self.d_inner, padding=d_conv - 1)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner)
        a_init = torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)).unsqueeze(0).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(a_init)
        dt_init = torch.exp(torch.empty(self.d_inner).uniform_(math.log(1e-3), math.log(1e-1)))
        inv_dt = dt_init + torch.log(-torch.expm1(-dt_init))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model)
        self.norm = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        z = self.norm(x)
        uv = self.in_proj(z)
        u, gate = uv.chunk(2, dim=-1)
        u = self.conv(u.transpose(1,2))[..., :z.shape[1]].transpose(1,2)
        u = F.silu(u)
        dt_raw, B_t, C_t = torch.split(self.x_proj(u), [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt_raw.float())).clamp(1e-4, 1.0)
        A = -torch.exp(self.A_log.float()).unsqueeze(0)
        state = torch.zeros((x.shape[0], self.d_inner, self.d_state), device=x.device, dtype=torch.float32)
        u32, B32, C32 = u.float(), B_t.float(), C_t.float()
        ys = []
        for ti in range(x.shape[1]):
            dtt = dt[:, ti].unsqueeze(-1)
            state = torch.exp(dtt * A) * state + dtt * u32[:, ti].unsqueeze(-1) * B32[:, ti].unsqueeze(1)
            ys.append((state * C32[:, ti].unsqueeze(1)).sum(-1) + self.D.float().view(1,-1) * u32[:, ti])
        y = torch.stack(ys, dim=1).to(dtype=x.dtype)
        y = y * torch.sigmoid(gate)
        return residual + self.dropout(self.out_proj(y))


class ModalityProjection(nn.Module):
    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.value = nn.Sequential(nn.Linear(28, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout))
        self.mask = nn.Sequential(nn.Linear(28, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout))
        self.time = nn.Sequential(nn.Linear(28, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout))
        self.trend = nn.Sequential(nn.Linear(26, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        v,m,t,tr = split_modalities(x)
        return self.value(v), self.mask(m), self.time(t), self.trend(tr)


class TextProjection(nn.Module):
    """Project 768-d clinical-note embeddings and an explicit missing-text flag into d_model."""
    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(TEXT_EMBEDDING_DIM + 1),
            nn.Linear(TEXT_EMBEDDING_DIM + 1, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, text: torch.Tensor, text_missing: torch.Tensor) -> torch.Tensor:
        tm = text_missing.float().unsqueeze(-1)
        return self.net(torch.cat([text, tm], dim=-1))


class TemporalAttentionPool(nn.Module):
    """Order-aware attention pooling with a learned hourly positional embedding."""
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.pos = nn.Parameter(torch.randn(24, d_model) * 0.02)
        self.score = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x + self.pos[:x.shape[1]]
        scores = self.score(z).float().squeeze(-1)
        weights = torch.softmax(scores, dim=1).to(dtype=x.dtype)
        return (z * weights.unsqueeze(-1)).sum(dim=1)


class TransformerStack(nn.Module):
    def __init__(self, d_model: int, layers: int, heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % heads:
            raise ValueError(f"d_model={d_model} must be divisible by heads={heads}")
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=heads, dim_feedforward=4*d_model, dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.encoder(x))


class HybridMambaTransformer(nn.Module):
    def __init__(self, p: Dict[str, Any]) -> None:
        super().__init__()
        d = int(p["d_model"]); drop = float(p["dropout"])
        self.input_proj = nn.Sequential(nn.Linear(110, d), nn.LayerNorm(d), nn.GELU())
        self.pos = nn.Parameter(torch.randn(24, d) * 0.02)
        self.mamba = nn.ModuleList([PureMambaBlock(d, int(p["d_state"]), int(p["d_conv"]), int(p["expand"]), drop) for _ in range(int(p["mamba_layers"]))])
        self.transformer = TransformerStack(d, int(p["transformer_layers"]), int(p["heads"]), drop)
        self.cross_gate = nn.Sequential(nn.Linear(2*d, d), nn.GELU(), nn.Linear(d, d), nn.Sigmoid())
        self.text = TextProjection(d, drop)
        self.demo = nn.Sequential(nn.Linear(2*d + 5, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(drop))
        self.head = MultitaskHead(d, drop)

    def forward(self, x: torch.Tensor, demographics: torch.Tensor, text: torch.Tensor, text_missing: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.input_proj(x) + self.pos[:x.shape[1]]
        for block in self.mamba: z = block(z)
        a = z
        b = self.transformer(z)
        g = self.cross_gate(torch.cat([a,b], dim=-1))
        h_struct = (g*a + (1-g)*b).mean(dim=1)
        text_token = self.text(text, text_missing)
        fused = self.demo(torch.cat([h_struct, text_token, demographics, text_missing.unsqueeze(-1)], dim=-1))
        return self.head(fused)


class MultimodalProgression(nn.Module):
    def __init__(self, p: Dict[str, Any]) -> None:
        super().__init__()
        d = int(p["d_model"]); h = int(p["progression_hidden"]); drop = float(p["dropout"])
        self.mod = ModalityProjection(d, drop)
        self.mix = nn.Sequential(nn.Linear(4*d, d), nn.LayerNorm(d), nn.GELU())
        self.gru = nn.GRU(d, h, num_layers=int(p["layers"]), batch_first=True, dropout=drop if int(p["layers"])>1 else 0.0)
        self.progression = nn.Sequential(nn.Linear(2*h, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(drop))
        self.text = TextProjection(d, drop)
        self.demo = nn.Sequential(nn.Linear(2*d + 5, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(drop))
        self.head = MultitaskHead(d, drop)

    def forward(self, x: torch.Tensor, demographics: torch.Tensor, text: torch.Tensor, text_missing: torch.Tensor) -> Dict[str, torch.Tensor]:
        mods = self.mod(x)
        z = self.mix(torch.cat(mods, dim=-1))
        seq, _ = self.gru(z)
        delta = seq[:,1:] - seq[:,:-1]
        cur = seq[:, -1]
        prog = delta.abs().mean(1) if delta.shape[1] else torch.zeros_like(cur)
        h = self.progression(torch.cat([cur, prog], dim=-1))
        text_token = self.text(text, text_missing)
        fused = self.demo(torch.cat([h, text_token, demographics, text_missing.unsqueeze(-1)], dim=-1))
        return self.head(fused)


class CrossAttentionMultimodal(nn.Module):
    def __init__(self, p: Dict[str, Any]) -> None:
        super().__init__()
        d = int(p["d_model"]); drop = float(p["dropout"]); heads = int(p["heads"])
        self.mod = ModalityProjection(d, drop)
        self.temporal = nn.Sequential(nn.Linear(110, d), nn.LayerNorm(d), nn.GELU())
        self.pos = nn.Parameter(torch.randn(24, d) * 0.02)
        self.temporal_stack = TransformerStack(d, int(p["layers"]), heads, drop)
        self.cross = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.demo_proj = nn.Linear(4, d)
        self.text = TextProjection(d, drop)
        self.fuse = nn.Sequential(nn.Linear(2*d, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(drop))
        self.head = MultitaskHead(d, drop)

    def forward(self, x: torch.Tensor, demographics: torch.Tensor, text: torch.Tensor, text_missing: torch.Tensor) -> Dict[str, torch.Tensor]:
        v,m,t,tr = self.mod(x)
        temporal = self.temporal_stack(self.temporal(x) + self.pos[:x.shape[1]])
        q = temporal.mean(1, keepdim=True)
        modality_tokens = torch.stack([v.mean(1), m.mean(1), t.mean(1), tr.mean(1)], dim=1)
        text_token = self.text(text, text_missing).unsqueeze(1)
        demo_token = self.demo_proj(demographics).unsqueeze(1)
        kv = torch.cat([modality_tokens, text_token, demo_token], dim=1)
        key_padding_mask = torch.zeros((x.shape[0], 6), dtype=torch.bool, device=x.device)
        key_padding_mask[:, 4] = text_missing.bool()
        attn, _ = self.cross(q, kv, kv, key_padding_mask=key_padding_mask, need_weights=False)
        h = self.fuse(torch.cat([q.squeeze(1), attn.squeeze(1)], dim=-1))
        return self.head(h)


class GatedFusionCrossAttn(nn.Module):
    def __init__(self, p: Dict[str, Any]) -> None:
        super().__init__()
        d = int(p["d_model"]); drop = float(p["dropout"]); heads = int(p["heads"])
        self.mod = ModalityProjection(d, drop)
        self.temporal = TransformerStack(d, int(p["layers"]), heads, drop)
        self.pos = nn.Parameter(torch.randn(24, d) * 0.02)
        self.mod_attn = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.temp_attn = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.raw_proj = nn.Sequential(nn.Linear(110, d), nn.LayerNorm(d), nn.GELU())
        self.demo_proj = nn.Linear(4, d)
        self.text = TextProjection(d, drop)
        gh = int(p["gate_hidden"])
        self.gate = nn.Sequential(nn.Linear(3*d + 1, gh), nn.GELU(), nn.Linear(gh, 4), nn.Softmax(dim=-1))
        self.fuse = nn.Sequential(nn.Linear(d, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(drop))
        self.head = MultitaskHead(d, drop)

    def forward(self, x: torch.Tensor, demographics: torch.Tensor, text: torch.Tensor, text_missing: torch.Tensor) -> Dict[str, torch.Tensor]:
        v,m,t,tr = self.mod(x)
        mods = torch.stack([v.mean(1), m.mean(1), t.mean(1), tr.mean(1)], dim=1)
        temporal = self.temporal(self.raw_proj(x) + self.pos[:x.shape[1]])
        tq = temporal.mean(1, keepdim=True)
        m2t, _ = self.mod_attn(tq, mods, mods, need_weights=False)
        t2m, _ = self.temp_attn(mods.mean(1, keepdim=True), temporal, temporal, need_weights=False)
        text_token = self.text(text, text_missing)
        dem = self.demo_proj(demographics)
        stacked = torch.stack([tq.squeeze(1), m2t.squeeze(1), t2m.squeeze(1), text_token], dim=1)
        gate_input = torch.cat([stacked.mean(1), dem, text_token, text_missing.unsqueeze(-1)], dim=-1)
        g = self.gate(gate_input)
        fused = (stacked * g.unsqueeze(-1)).sum(1) + dem
        return self.head(self.fuse(fused))


class MoEModalityFusion(nn.Module):
    def __init__(self, p: Dict[str, Any]) -> None:
        super().__init__()
        d = int(p["d_model"]); drop = float(p["dropout"]); n_exp = int(p["experts"]); top_k = int(p["top_k"]); exp = int(p["expert_expand"])
        if top_k > n_exp: raise ValueError("top_k cannot exceed experts")
        self.top_k = top_k
        self.load_balance_coef = float(p.get("load_balance_coef", 0.01))
        self.mod = ModalityProjection(d, drop)
        self.pool = TemporalAttentionPool(d)
        self.demo_proj = nn.Linear(4, d)
        self.text = TextProjection(d, drop)
        self.router = nn.Sequential(nn.Linear(6*d, d), nn.GELU(), nn.Dropout(drop), nn.Linear(d, n_exp))
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(6*d, exp*d), nn.GELU(), nn.Dropout(drop), nn.Linear(exp*d, d)) for _ in range(n_exp)])
        self.out = nn.Sequential(nn.Linear(d, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(drop))
        self.head = MultitaskHead(d, drop)
        self.last_aux_loss = torch.tensor(0.0)

    def forward(self, x: torch.Tensor, demographics: torch.Tensor, text: torch.Tensor, text_missing: torch.Tensor) -> Dict[str, torch.Tensor]:
        v,m,t,tr = self.mod(x)
        v,m,t,tr = self.pool(v), self.pool(m), self.pool(t), self.pool(tr)
        dem = self.demo_proj(demographics)
        text_token = self.text(text, text_missing)
        tokens = torch.cat([v,m,t,tr,text_token,dem], dim=-1)
        logits = self.router(tokens)
        probs = torch.softmax(logits.float(), dim=-1)
        k = min(self.top_k, probs.shape[-1])
        top_p, idx = torch.topk(probs, k=k, dim=-1)
        weights = top_p if k == 1 else top_p / top_p.sum(-1, keepdim=True).clamp_min(1e-8)
        expert_out = torch.stack([e(tokens) for e in self.experts], dim=1)
        gather_idx = idx.unsqueeze(-1).expand(-1, -1, expert_out.shape[-1])
        selected = torch.gather(expert_out, 1, gather_idx)
        fused = (selected * weights.unsqueeze(-1).to(dtype=selected.dtype)).sum(1)
        load = torch.zeros(probs.shape[-1], device=probs.device, dtype=probs.dtype)
        load.scatter_add_(0, idx.reshape(-1), torch.ones(idx.numel(), device=probs.device, dtype=probs.dtype))
        load = load / float(max(1, idx.numel()))
        importance = probs.mean(dim=0)
        self.last_aux_loss = probs.shape[-1] * torch.sum(importance * load)
        return self.head(self.out(fused))


def build_model(model_id: str, params: Dict[str, Any]) -> nn.Module:
    require_torch()
    if model_id == "28": return HybridMambaTransformer(params)
    if model_id == "29": return MultimodalProgression(params)
    if model_id == "30": return CrossAttentionMultimodal(params)
    if model_id == "31": return GatedFusionCrossAttn(params)
    if model_id == "32": return MoEModalityFusion(params)
    raise ValueError(f"Unknown Family 7 model_id: {model_id}")


# ---------------------------------------------------------------------------
# Loss / evaluation
# ---------------------------------------------------------------------------
def multitask_loss(outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], valid: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    losses: Dict[str, torch.Tensor] = {}
    total = torch.zeros((), device=next(iter(outputs.values())).device)
    for task in BINARY_TASKS:
        mask = valid[task]
        if torch.any(mask):
            losses[task] = F.binary_cross_entropy_with_logits(outputs[task][mask], targets[task][mask].float())
            total = total + losses[task]
        else:
            losses[task] = total.new_zeros(())
    los_mask = valid["los_hours_24h"]
    if torch.any(los_mask):
        target_excess = torch.log1p(torch.clamp(targets["los_hours_24h"][los_mask].float() - 24.0, min=0.0))
        losses["los_hours_24h"] = F.smooth_l1_loss(outputs["los_excess_log1p"][los_mask].float(), target_excess)
        total = total + losses["los_hours_24h"]
    else:
        losses["los_hours_24h"] = total.new_zeros(())
    return total / 5.0, losses


@torch.no_grad()
def collect_predictions(model: nn.Module, loader: DataLoader, device: torch.device, use_amp: bool) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    model.eval()
    preds = {task: [] for task in PRIMARY_TASKS}
    valid_all = {task: [] for task in PRIMARY_TASKS}
    for x, demographics, text, text_missing, y_dict, valid_dict in loader:
        x = x.to(device, non_blocking=True)
        demographics = demographics.to(device, non_blocking=True)
        text = text.to(device, non_blocking=True)
        text_missing = text_missing.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            out = model(x, demographics, text, text_missing)
        for task in BINARY_TASKS:
            preds[task].append(torch.sigmoid(out[task]).float().cpu().numpy())
        preds["los_hours_24h"].append((torch.expm1(out["los_excess_log1p"].float()) + 24.0).cpu().numpy())
        for task in PRIMARY_TASKS:
            valid_all[task].append(valid_dict[task].numpy().astype(np.int8))
    return (
        {task: np.concatenate(vals) for task, vals in preds.items()},
        {task: np.concatenate(vals) for task, vals in valid_all.items()},
    )


def safe_auroc(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y); p = np.asarray(p)
    if len(np.unique(y)) < 2: return float("nan")
    return float(roc_auc_score(y, p))


def safe_auprc(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y); p = np.asarray(p)
    if len(np.unique(y)) < 2: return float("nan")
    return float(average_precision_score(y, p))


def safe_brier(y: np.ndarray, p: np.ndarray) -> float:
    if len(y) == 0: return float("nan")
    return float(brier_score_loss(y, p))


def evaluate_predictions(data: FrozenData, pred: Dict[str, np.ndarray], valid: Dict[str, np.ndarray], indices: np.ndarray, split_name: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for task in PRIMARY_TASKS:
        y = data.targets[task][indices].astype(float, copy=False)
        p = pred[task].astype(float, copy=False)
        v = valid[task].astype(bool, copy=False)
        keep = v & np.isfinite(y) & np.isfinite(p)
        if task in BINARY_TASKS:
            yy = y[keep].astype(int); pp = p[keep]
            rows.append({"split": split_name, "task": task, "valid_n": int(len(yy)), "auroc": safe_auroc(yy, pp), "auprc": safe_auprc(yy, pp), "brier": safe_brier(yy, pp), "prevalence": float(np.mean(yy)) if len(yy) else float("nan")})
        else:
            yy = y[keep]; pp = p[keep]
            rows.append({"split": split_name, "task": task, "valid_n": int(len(yy)), "mae": float(mean_absolute_error(yy, pp)) if len(yy) else float("nan"), "rmse": float(np.sqrt(mean_squared_error(yy, pp))) if len(yy) else float("nan"), "median_ae": float(median_absolute_error(yy, pp)) if len(yy) else float("nan"), "r2": float(r2_score(yy, pp)) if len(yy) > 1 else float("nan")})
    comp_valid = valid["mortality_24h"].astype(bool) & valid["ventilation_onset_24h"].astype(bool) & valid["vasopressor_onset_24h"].astype(bool)
    ym = data.targets["mortality_24h"][indices].astype(float); yv = data.targets["ventilation_onset_24h"][indices].astype(float); yp = data.targets["vasopressor_onset_24h"][indices].astype(float)
    y_det = 1 - (1 - ym) * (1 - yv) * (1 - yp)
    p_det = 1 - (1 - pred["mortality_24h"]) * (1 - pred["ventilation_onset_24h"]) * (1 - pred["vasopressor_onset_24h"])
    keep = comp_valid & np.isfinite(y_det) & np.isfinite(p_det)
    rows.append({"split": split_name, "task": "deterioration_24h", "valid_n": int(np.sum(keep)), "auroc": safe_auroc(y_det[keep].astype(int), p_det[keep]), "auprc": safe_auprc(y_det[keep].astype(int), p_det[keep]), "brier": safe_brier(y_det[keep].astype(int), p_det[keep]), "prevalence": float(np.mean(y_det[keep])) if np.any(keep) else float("nan")})
    return pd.DataFrame(rows)


def validation_selection_score(metrics: pd.DataFrame) -> float:
    rows = metrics[metrics["task"].isin(BINARY_TASKS)].copy()
    if set(rows["task"]) != set(BINARY_TASKS) or len(rows) != 4:
        return float("-inf")
    vals = rows["auprc"].to_numpy(dtype=float)
    if not np.all(np.isfinite(vals)): return float("-inf")
    return float(np.mean(vals))

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def make_optimizer(model: nn.Module, lr: float, weight_decay: float):
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def make_scheduler(
    optimizer,
    scheduler_name: str,
    max_epochs: int,
    steps_per_epoch: int,
    max_lr: float,
):
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    if scheduler_name == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=max(1, steps_per_epoch * max_epochs),
            pct_start=0.1,
            anneal_strategy="cos",
            cycle_momentum=False,
        )
    raise ValueError(f"Unknown scheduler: {scheduler_name}")


def train_one(
    model_id: str,
    params: Dict[str, Any],
    data: FrozenData,
    device: torch.device,
    run_dir: Optional[Path] = None,
) -> Tuple[nn.Module, float, int, pd.DataFrame]:
    set_seed(SEED)
    model = build_model(model_id, params).to(device)

    batch_size = int(params.get("batch_size", 512))
    max_epochs = int(params.get("max_epochs", 50))
    patience = int(params.get("patience", 7))
    lr = float(params["lr"])
    weight_decay = float(params.get("weight_decay", 1e-4))
    grad_clip = float(params.get("grad_clip", 1.0))
    scheduler_name = str(params.get("scheduler", "cosine"))

    train_loader = make_loader(data, data.train_idx, batch_size, True, device.type == "cuda")
    val_loader = make_loader(data, data.val_idx, batch_size, False, device.type == "cuda")

    optimizer = make_optimizer(model, lr, weight_decay)
    scheduler = make_scheduler(
        optimizer, scheduler_name, max_epochs, len(train_loader), lr
    )
    use_amp = amp_enabled(device)

    best_score = float("-inf")
    best_epoch = -1
    best_state = None
    history: List[Dict[str, Any]] = []
    no_improve = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        seen = 0

        for x, demographics, text, text_missing, y_dict, valid_dict in train_loader:
            x = x.to(device, non_blocking=True)
            demographics = demographics.to(device, non_blocking=True)
            text = text.to(device, non_blocking=True)
            text_missing = text_missing.to(device, non_blocking=True)
            targets = {task: y_dict[task].to(device, non_blocking=True) for task in PRIMARY_TASKS}
            valids = {
                task: valid_dict[task].to(device, non_blocking=True).bool()
                for task in PRIMARY_TASKS
            }

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                outputs = model(x, demographics, text, text_missing)
                loss, _ = multitask_loss(outputs, targets, valids)
                aux = getattr(model, "last_aux_loss", None)
                if aux is not None:
                    loss = loss + float(params.get("load_balance_coef", 0.0)) * aux
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            if scheduler_name == "onecycle":
                scheduler.step()

            bs = x.shape[0]
            running_loss += float(loss.detach().cpu().item()) * bs
            seen += bs

        if scheduler_name == "cosine":
            scheduler.step()

        train_loss = running_loss / max(seen, 1)
        val_pred, val_valid = collect_predictions(model, val_loader, device, use_amp)
        val_metrics = evaluate_predictions(data, val_pred, val_valid, data.val_idx, "val")
        score = validation_selection_score(val_metrics)
        los_row = val_metrics[val_metrics["task"] == "los_hours_24h"]
        los_mae = float(los_row["mae"].iloc[0]) if not los_row.empty else float("nan")

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_mean_AUPRC": score,
                "val_LOS_MAE": los_mae,
                "lr": optimizer.param_groups[0]["lr"],
                "epoch_seconds": time.time() - epoch_start,
            }
        )
        print(
            f"[{model_id} {MODEL_NAMES[model_id]}] epoch={epoch:02d} "
            f"train_loss={train_loss:.5f} "
            f"val_mean_AUPRC={score:.5f} val_LOS_MAE={los_mae:.3f}",
            flush=True,
        )

        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"[{model_id}] early stopping at epoch {epoch}.", flush=True)
            break

    if best_state is None:
        raise RuntimeError(f"No valid checkpoint selected for Family 7 model {model_id}.")

    model.load_state_dict(best_state)

    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_id": model_id,
                "model_name": MODEL_NAMES[model_id],
                "params": params,
                "best_epoch": best_epoch,
                "best_val_selection_score": best_score,
                "state_dict": model.state_dict(),
            },
            run_dir / "checkpoint.pt",
        )
        pd.DataFrame(history).to_csv(run_dir / "training_history.csv", index=False)

    return model, best_score, best_epoch, pd.DataFrame(history)


@torch.no_grad()
def predict_and_write(
    model: nn.Module,
    data: FrozenData,
    device: torch.device,
    indices: np.ndarray,
    split_name: str,
    run_dir: Path,
    batch_size: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    loader = make_loader(data, indices, batch_size, False, device.type == "cuda")
    pred, valid = collect_predictions(model, loader, device, amp_enabled(device))
    metric_df = evaluate_predictions(data, pred, valid, indices, split_name)

    rows: List[Dict[str, Any]] = []
    for task in PRIMARY_TASKS:
        y = data.targets[task][indices].astype(float, copy=False)
        p = pred[task].astype(float, copy=False)
        v = valid[task].astype(int, copy=False)
        for j, original_idx in enumerate(indices):
            rows.append(
                {
                    "stay_id": data.stay_id[original_idx],
                    "subject_id": data.subject_id[original_idx],
                    "split": split_name,
                    "task": task,
                    "target": float(y[j]),
                    "validity": int(v[j]),
                    "prediction": float(p[j]),
                }
            )

    comp_valid = (
        valid["mortality_24h"].astype(bool)
        & valid["ventilation_onset_24h"].astype(bool)
        & valid["vasopressor_onset_24h"].astype(bool)
    )
    ym_raw = data.targets["mortality_24h"][indices].astype(float, copy=False)
    yv_raw = data.targets["ventilation_onset_24h"][indices].astype(float, copy=False)
    yp_raw = data.targets["vasopressor_onset_24h"][indices].astype(float, copy=False)
    y_det = np.full(len(indices), np.nan, dtype=float)
    valid_det = comp_valid & np.isfinite(ym_raw) & np.isfinite(yv_raw) & np.isfinite(yp_raw)
    y_det[valid_det] = (1 - (1 - ym_raw[valid_det]) * (1 - yv_raw[valid_det]) * (1 - yp_raw[valid_det]))
    p_det = (
        1
        - (1 - pred["mortality_24h"])
        * (1 - pred["ventilation_onset_24h"])
        * (1 - pred["vasopressor_onset_24h"])
    )
    for j, original_idx in enumerate(indices):
        rows.append(
            {
                "stay_id": data.stay_id[original_idx],
                "subject_id": data.subject_id[original_idx],
                "split": split_name,
                "task": "deterioration_24h",
                "target": float(y_det[j]),
                "validity": int(comp_valid[j]),
                "prediction": float(p_det[j]),
            }
        )

    return metric_df, pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fixed / tuned phases
# ---------------------------------------------------------------------------
def fixed_params(model_id: str) -> Dict[str, Any]:
    return {**MODEL_CONFIGS[model_id].fixed, "batch_size": 512, "max_epochs": 50, "patience": 7, "weight_decay": 1e-4, "grad_clip": 1.0, "scheduler": "cosine"}


def architecture_notes_for_model(model_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
    if model_id == "28":
        return {"type": "Hybrid Mamba-Transformer", "mamba": "pure PyTorch selective SSM; explicit fp32 scan over 24 hours with S4D-real A-log and Mamba-style dt initialization", "transformer": "pre-norm GELU Transformer encoder with learned hourly positions", "text": "768-d clinical-note embedding with explicit missing-text flag", "fusion": "learned gate between Mamba and Transformer structured streams, followed by text + demographic fusion", "d_model": params["d_model"], "d_state": params["d_state"], "mamba_layers": params["mamba_layers"], "transformer_layers": params["transformer_layers"], "heads": params["heads"]}
    if model_id == "29":
        return {"type": "Multimodal Progression", "structured_streams": "separate value/mask/time-since/trend projections", "progression": "GRU temporal encoder plus absolute first-difference progression branch anchored to the last hidden state", "text": "768-d clinical-note embedding with explicit missing-text flag", "fusion": "current-state + progression-state + clinical text + demographics", "d_model": params["d_model"], "progression_hidden": params["progression_hidden"], "layers": params["layers"]}
    if model_id == "30":
        return {"type": "Cross-Attention Multimodal", "structured_streams": "four structured channel-group tokens", "text": "768-d clinical-note token with missing-text key masking", "attention": "temporal clinical query attends to structured modality tokens, text token, and demographic token", "temporal_encoder": "Transformer encoder over 24 hourly observations with learned positions", "d_model": params["d_model"], "layers": params["layers"], "heads": params["heads"]}
    if model_id == "31":
        return {"type": "Gated Fusion / Cross-Attention", "structured_streams": "temporal Transformer branch and four structured summary tokens", "text": "clinical text is an explicit fourth gated branch; missing-text scalar is fed directly into the gate", "attention": "bidirectional cross-attention between temporal and structured summaries", "fusion": "softmax learned four-way gate with demographic and text-missing interaction", "d_model": params["d_model"], "layers": params["layers"], "heads": params["heads"], "gate_hidden": params["gate_hidden"]}
    if model_id == "32":
        return {"type": "MoE Modality Fusion", "experts": params["experts"], "top_k": params["top_k"], "routing": "sample-level sparse top-k router over four structured summaries + clinical text + demographics", "expert": "independent feed-forward multimodal experts", "router_gradient_fix": "full softmax over all experts followed by top-k probability selection; top_k=1 retains non-zero router gradient", "load_balancing": "Switch-style importance/load auxiliary loss", "fidelity_note": "compact V2 multimodal MoE adaptation with order-aware temporal attention pooling", "d_model": params["d_model"]}
    raise KeyError(model_id)


def run_fixed_model(model_id: str, data: FrozenData, device: torch.device) -> Dict[str, Any]:
    cfg = MODEL_CONFIGS[model_id]
    out_dir = RESULTS_ROOT / "fixed" / f"{model_id}_{cfg.name.replace(' ', '_').replace('/', '_')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    params = fixed_params(model_id)
    start = time.time()
    model, score, best_epoch, history = train_one(model_id, params, data, device, run_dir=out_dir)
    val_metrics, val_preds = predict_and_write(model, data, device, data.val_idx, "val", out_dir, params["batch_size"])
    test_metrics, test_preds = predict_and_write(model, data, device, data.test_idx, "test", out_dir, params["batch_size"])
    pd.concat([val_metrics, test_metrics], ignore_index=True).to_csv(out_dir / "metrics.csv", index=False)
    pd.concat([val_preds, test_preds], ignore_index=True).to_csv(out_dir / "predictions.csv", index=False)
    pd.concat([val_preds, test_preds], ignore_index=True).to_csv(out_dir / "test_predictions.csv", index=False)
    arch_notes = architecture_notes_for_model(model_id, params)
    metadata = base_metadata(model_id, params, data, device, "fixed", score, best_epoch, arch_notes)
    metadata["training_seconds"] = time.time() - start
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    result = {"model_id": model_id, "model_name": cfg.name, "mode": "fixed", "validation_selection_score": score, "best_epoch": best_epoch, "output_dir": str(out_dir)}
    print(json.dumps(result, indent=2), flush=True)
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return result


def suggest_model_params(model_id: str, trial: "optuna.Trial") -> Dict[str, Any]:
    if model_id == "28":
        model_params = {"d_model": trial.suggest_int("d_model",128,384,step=8), "d_state": trial.suggest_categorical("d_state",[8,16,32]), "d_conv": trial.suggest_categorical("d_conv",[3,4,5]), "expand": trial.suggest_categorical("expand",[2,4]), "mamba_layers": trial.suggest_int("mamba_layers",1,3), "transformer_layers": trial.suggest_int("transformer_layers",1,4), "heads": trial.suggest_categorical("heads",[4,8]), "dropout": trial.suggest_float("dropout",0.05,0.30), "lr": trial.suggest_float("lr",1e-4,2e-3,log=True)}
    elif model_id == "29":
        model_params = {"d_model": trial.suggest_int("d_model",128,384,step=8), "progression_hidden": trial.suggest_int("progression_hidden",128,384,step=8), "layers": trial.suggest_int("layers",1,4), "dropout": trial.suggest_float("dropout",0.05,0.30), "lr": trial.suggest_float("lr",1e-4,2e-3,log=True)}
    elif model_id == "30":
        model_params = {"d_model": trial.suggest_int("d_model",128,384,step=8), "layers": trial.suggest_int("layers",2,4), "heads": trial.suggest_categorical("heads",[4,8]), "dropout": trial.suggest_float("dropout",0.05,0.30), "lr": trial.suggest_float("lr",1e-4,2e-3,log=True)}
    elif model_id == "31":
        model_params = {"d_model": trial.suggest_int("d_model",128,384,step=8), "layers": trial.suggest_int("layers",1,3), "heads": trial.suggest_categorical("heads",[4,8]), "gate_hidden": trial.suggest_int("gate_hidden",64,256,step=32), "dropout": trial.suggest_float("dropout",0.05,0.30), "lr": trial.suggest_float("lr",1e-4,2e-3,log=True)}
    elif model_id == "32":
        model_params = {"d_model": trial.suggest_int("d_model",128,384,step=8), "experts": trial.suggest_categorical("experts",[2,4,6]), "top_k": trial.suggest_categorical("top_k",[1,2]), "expert_expand": trial.suggest_categorical("expert_expand",[2,4]), "load_balance_coef": trial.suggest_categorical("load_balance_coef",[0.0,0.01,0.05]), "dropout": trial.suggest_float("dropout",0.05,0.30), "lr": trial.suggest_float("lr",1e-4,2e-3,log=True)}
    else: raise ValueError(model_id)
    # Advanced multimodal models are capped at 256/512 to reduce ROCm attention/MoE allocator pressure.
    batch_choices = [256, 512]
    model_params.update({"batch_size": trial.suggest_categorical("batch_size", batch_choices), "max_epochs": 50, "patience": trial.suggest_int("patience",5,10), "weight_decay": trial.suggest_float("weight_decay",1e-5,1e-3,log=True), "grad_clip": trial.suggest_float("grad_clip",0.5,2.0), "scheduler": trial.suggest_categorical("scheduler",["cosine","onecycle"])})
    d_model = int(model_params["d_model"])
    if "heads" in model_params and d_model % int(model_params["heads"]) != 0: raise ValueError("d_model must be divisible by heads")
    if model_id == "32" and int(model_params["top_k"]) > int(model_params["experts"]): raise ValueError("top_k cannot exceed experts")
    return model_params


def run_optuna_trial_worker(model_id: str, params_json: str, result_json: str) -> None:
    require_torch()
    params = json.loads(params_json)
    validate_model_contract(model_id, params=params)
    set_seed(SEED)
    device = get_device()
    require_gpu_protocol(device, f"trial-worker model {model_id}")
    data = load_frozen_v2()
    print(f"[TRIAL-WORKER] family=07 model={model_id} pid={os.getpid()} device={device} gpu={torch.cuda.get_device_name(0)} amp=BF16 params={params}", flush=True)
    model, score, best_epoch, _ = train_one(model_id, params, data, device, run_dir=None)
    result_path = Path(result_json); result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps({"score": float(score), "best_epoch": int(best_epoch)}, indent=2), encoding="utf-8")
    del model; gc.collect()
    if torch.cuda.is_available():
        try: torch.cuda.synchronize(); torch.cuda.empty_cache()
        except Exception: pass


def run_tuned_model(model_id: str, data: FrozenData, device: torch.device, n_trials: int) -> Dict[str, Any]:
    require_optuna()
    require_gpu_protocol(device, f"tuned model {model_id}")
    cfg = MODEL_CONFIGS[model_id]
    tuned_root = RESULTS_ROOT / "tuned" / f"{model_id}_{cfg.name.replace(' ', '_').replace('/', '_')}"
    tuned_root.mkdir(parents=True, exist_ok=True)
    start = time.time(); sampler = optuna.samplers.TPESampler(seed=SEED)
    # Model 29 previously had a CPU final configuration; isolate the corrected GPU-only rerun
    # in a fresh study so the prior CPU study cannot contaminate GPU benchmark selection.
    if model_id == "29":
        study_path = tuned_root / "study_gpu.db"
        study_name = "family07_29_gpu"
    else:
        study_path = tuned_root / "study.db"
        study_name = f"family07_{model_id}"
    storage_url = f"sqlite:///{study_path.as_posix()}"
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name=study_name, storage=storage_url, load_if_exists=True)
    def n_done(s):
        return sum(t.state == optuna.trial.TrialState.COMPLETE for t in s.trials)

    print(
        f"[OPTUNA] model={model_id} complete={n_done(study)} total_records={len(study.trials)} "
        f"failed={sum(t.state == optuna.trial.TrialState.FAIL for t in study.trials)} "
        f"running={sum(t.state == optuna.trial.TrialState.RUNNING for t in study.trials)} target={n_trials}",
        flush=True,
    )

    def objective(trial: "optuna.Trial") -> float:
        params = suggest_model_params(model_id, trial); validate_model_contract(model_id, params=params)
        trial_result = tuned_root / f".trial_{trial.number}_result.json"
        if trial_result.exists(): trial_result.unlink()
        cmd = [sys.executable, str(Path(__file__).resolve()), "--model", model_id, "--mode", "trial-worker", "--trial-params-json", json.dumps(params, separators=(",", ":")), "--trial-result-json", str(trial_result)]
        env = os.environ.copy(); env.setdefault("PYTHONUNBUFFERED", "1")
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env, check=False)
        if proc.returncode != 0 or not trial_result.exists():
            raise RuntimeError(f"Isolated trial worker failed for Family 7 model {model_id}, trial {trial.number}, returncode={proc.returncode}")
        result = json.loads(trial_result.read_text(encoding="utf-8"))
        try: trial_result.unlink()
        except OSError: pass
        trial.set_user_attr("best_epoch", int(result["best_epoch"])); trial.set_user_attr("params", params)
        return float(result["score"])

    max_records = max(2 * int(n_trials), int(n_trials) + 1)
    while n_done(study) < int(n_trials) and len(study.trials) < max_records:
        study.optimize(objective, n_trials=1, gc_after_trial=False, catch=(RuntimeError, ValueError), show_progress_bar=False)
    pd.DataFrame([{ "number": t.number, "state": str(t.state), "value": t.value, **t.params } for t in study.trials]).to_csv(tuned_root / "optuna_trials.csv", index=False)
    (tuned_root / "study_storage.txt").write_text(str(study_path), encoding="utf-8")
    successful_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    completed_count = len(successful_trials)
    if completed_count < int(n_trials):
        print(f"[OPTUNA] WARNING: reached record safety cap with {completed_count}/{n_trials} completed trials for model {model_id}.", flush=True)
    if not successful_trials: raise RuntimeError(f"No successful Optuna trials completed for Family 7 model {model_id}.")
    best_trial = study.best_trial; best_trial_user_params = best_trial.user_attrs.get("params", {})
    final_params = dict(best_trial_user_params) if best_trial_user_params else dict(best_trial.params)
    validate_model_contract(model_id, params=final_params)
    require_gpu_protocol(device, f"final selected retrain model {model_id}")
    final_dir = tuned_root / "final_selected_configuration"; final_dir.mkdir(parents=True, exist_ok=True)
    model, score, best_epoch, history = train_one(model_id, final_params, data, device, run_dir=final_dir)
    val_metrics, val_preds = predict_and_write(model, data, device, data.val_idx, "val", final_dir, final_params["batch_size"])
    test_metrics, test_preds = predict_and_write(model, data, device, data.test_idx, "test", final_dir, final_params["batch_size"])
    pd.concat([val_metrics, test_metrics], ignore_index=True).to_csv(final_dir / "metrics.csv", index=False)
    pd.concat([val_preds, test_preds], ignore_index=True).to_csv(final_dir / "predictions.csv", index=False)
    pd.concat([val_preds, test_preds], ignore_index=True).to_csv(final_dir / "test_predictions.csv", index=False)
    arch_notes = {**architecture_notes_for_model(model_id, final_params), "canonical_search_space": cfg.search_space, "selected_model_params": {k:v for k,v in final_params.items() if k not in {"batch_size","max_epochs","patience","weight_decay","grad_clip","scheduler"}}, "global_selected_params": {k: final_params[k] for k in ["batch_size","patience","weight_decay","grad_clip","scheduler"]}, "test_used_for_selection": False}
    metadata = base_metadata(model_id, final_params, data, device, "tuned", score, best_epoch, arch_notes)
    metadata.update({"n_optuna_trials_target": n_trials, "n_optuna_trials_completed": completed_count, "optuna_best_trial_number": int(study.best_trial.number), "optuna_best_value": float(study.best_value), "tuning_seconds": time.time()-start})
    (final_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    (tuned_root / "best_params.json").write_text(json.dumps(final_params, indent=2, default=str), encoding="utf-8")
    result = {"model_id": model_id, "model_name": cfg.name, "mode": "tuned", "validation_selection_score": score, "best_epoch": best_epoch, "output_dir": str(final_dir), "best_trial": int(study.best_trial.number)}
    print(json.dumps(result, indent=2), flush=True)
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return result


# ---------------------------------------------------------------------------
# Shared metadata helper
# ---------------------------------------------------------------------------
def base_metadata(model_id: str, params: Dict[str, Any], data: FrozenData, device: torch.device, mode: str, score: float, best_epoch: int, architecture_notes: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "family": "07_Advanced_Multimodal_28_32",
        "model_id": model_id,
        "model_name": MODEL_NAMES[model_id],
        "mode": mode,
        "seed": SEED,
        "device": str(device),
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "hip_version": getattr(torch.version, "hip", None),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "compute_protocol": "GPU/BF16 required for fixed/tuned/trial-worker benchmark training; CPU fallback disabled.",
        "advanced_rocm_policy": "Pure PyTorch Mamba subcomponent in Model 28; no Triton dependency; explicit fp32 SSM recurrence; attention/MoE batch search capped at 256/512 for ROCm allocator stability.",
        "amp_mode": "BF16 autocast" if amp_enabled(device) else "FP32 (CPU)",
        "input_shape": list(EXPECTED_SHAPE),
        "text_embedding_dim": TEXT_EMBEDDING_DIM,
        "text_present_count": int(np.sum(data.text_missing == 0)),
        "text_missing_count": int(np.sum(data.text_missing == 1)),
        "text_embedding_path": str(TEXT_EMBEDDING_PATH),
        "split_counts": {"train": int(len(data.train_idx)), "val": int(len(data.val_idx)), "test": int(len(data.test_idx))},
        "registered_fixed_params": MODEL_CONFIGS[model_id].fixed,
        "run_params": params,
        "selection_metric": "Mean validation AUPRC across mortality, ventilation, vasopressor, readmission",
        "validation_selection_score": score,
        "best_epoch": best_epoch,
        "test_used_for_selection": False,
        "fixed_tuned_phase_separation": True,
        "optuna_initialization_policy": "Shared SEED=20260923 across HP trials; trial-to-trial variation comes from sampled hyperparameters only.",
        "optuna_trial_execution": "Each Optuna trial runs in an isolated child Python process to prevent ROCm/HIP allocator/context accumulation across trials.",
        "los_training_transform": "log1p(max(los_hours_24h - 24.0, 0))",
        "los_reporting_transform": "expm1(predicted_excess_log1p) + 24.0 hours",
        "derived_deterioration": "Evaluation-only noisy-OR of mortality, ventilation, vasopressor",
        "optuna_storage": "SQLite study.db with load_if_exists=True; --n-trials is the target total trial count for resumable tuning.",
        "mamba_scan_policy": "Model 28 uses pure-PyTorch explicit Mamba scan over 24 steps with S4D-real A-log initialization and Mamba-style dt bias initialization; Models 29–32 use standard PyTorch recurrent/attention/fusion operators.",
        "validity_schema": "All primary task validity columns are mandatory; missing validity schema raises an error.",
        "multimodal_input_contract": "Structured 24x110 + demographics 4 + aligned clinical-note embedding 768 + explicit text-missing flag.",
        "target_schema": "Frozen targets are consumed exactly as stored; invalid labels are excluded through validity masks.",
        "dataset_memory_safety": "Memory-mapped arrays are copied before torch.from_numpy to avoid non-writable tensor warnings.",
        "architecture_notes": architecture_notes,
    }

# ---------------------------------------------------------------------------
# Statistical evaluation of existing predictions
# ---------------------------------------------------------------------------
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
        ece += np.sum(mask) / n * abs(np.mean(p[mask]) - np.mean(y[mask]))
    return float(ece)


def metric_from_arrays(task: str, y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    if task in BINARY_TASKS or task == "deterioration_24h":
        return {
            "auroc": safe_auroc(y.astype(int), p),
            "auprc": safe_auprc(y.astype(int), p),
            "brier": safe_brier(y.astype(int), p),
            "ece": expected_calibration_error(y.astype(int), p),
        }
    return {
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "median_ae": float(median_absolute_error(y, p)),
        "r2": float(r2_score(y, p)) if len(y) > 1 else float("nan"),
    }


def bootstrap_ci(
    task: str,
    y: np.ndarray,
    p: np.ndarray,
    reps: int,
    seed: int,
) -> Dict[str, Any]:
    n = len(y)
    if n == 0:
        return {"n": 0}
    point = metric_from_arrays(task, y, p)
    rng = np.random.default_rng(seed)
    values: Dict[str, List[float]] = {k: [] for k in point}
    for _ in range(reps):
        idx = rng.integers(0, n, size=n)
        yy = y[idx]
        pp = p[idx]
        m = metric_from_arrays(task, yy, pp)
        for key, val in m.items():
            if np.isfinite(val):
                values[key].append(float(val))
    out: Dict[str, Any] = {"n": int(n)}
    for key, point_val in point.items():
        arr = np.asarray(values[key], dtype=float)
        out[key] = float(point_val)
        if len(arr):
            out[f"{key}_ci95_low"] = float(np.quantile(arr, 0.025))
            out[f"{key}_ci95_high"] = float(np.quantile(arr, 0.975))
        else:
            out[f"{key}_ci95_low"] = float("nan")
            out[f"{key}_ci95_high"] = float("nan")
    return out


def read_prediction_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Prediction table not found: {path}")
    df = pd.read_csv(path)
    required = {"split", "task", "target", "validity", "prediction"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Prediction table {path} missing columns: {missing}")
    return df


def stats_for_prediction_table(
    path: Path,
    model_id: str,
    mode: str,
) -> pd.DataFrame:
    pred_df = read_prediction_table(path)
    rows: List[Dict[str, Any]] = []
    for split_name in ("val", "test"):
        sdf = pred_df[pred_df["split"] == split_name]
        for task in list(PRIMARY_TASKS) + ["deterioration_24h"]:
            tdf = sdf[sdf["task"] == task].copy()
            tdf = tdf[tdf["validity"].astype(bool)]
            tdf = tdf[np.isfinite(tdf["target"]) & np.isfinite(tdf["prediction"])]
            y = tdf["target"].to_numpy(dtype=float)
            p = tdf["prediction"].to_numpy(dtype=float)
            result = bootstrap_ci(
                task,
                y,
                p,
                reps=BOOTSTRAP_REPS,
                seed=SEED + int.from_bytes(f"{model_id}|{mode}|{split_name}|{task}".encode("utf-8"), "little") % 10_000,
            )
            rows.append(
                {
                    "model_id": model_id,
                    "model_name": MODEL_NAMES[model_id],
                    "mode": mode,
                    "split": split_name,
                    "task": task,
                    **result,
                }
            )
    return pd.DataFrame(rows)


def evaluate_stats_mode() -> None:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    summaries: List[pd.DataFrame] = []

    targets = []
    for model_id, cfg in MODEL_CONFIGS.items():
        targets.append(
            (
                model_id,
                "fixed",
                RESULTS_ROOT / "fixed" / f"{model_id}_{cfg.name.replace(' ', '_')}" / "test_predictions.csv",
            )
        )
        targets.append(
            (
                model_id,
                "tuned",
                RESULTS_ROOT / "tuned" / f"{model_id}_{cfg.name.replace(' ', '_')}" / "final_selected_configuration" / "test_predictions.csv",
            )
        )

    for model_id, mode, path in targets:
        print(f"[STATS] {mode} / {model_id}_{MODEL_NAMES[model_id]}", flush=True)
        stats_df = stats_for_prediction_table(path, model_id, mode)
        stats_df.to_csv(path.parent / "statistical_evaluation.csv", index=False)
        summaries.append(stats_df)

    final = pd.concat(summaries, ignore_index=True)
    final.to_csv(RESULTS_ROOT / "family07_statistical_evaluation.csv", index=False)
    print(f"[STATS] complete -> {RESULTS_ROOT / 'family07_statistical_evaluation.csv'}", flush=True)


# ---------------------------------------------------------------------------
# Runner / CLI
# ---------------------------------------------------------------------------
def validate_model_contract(model_id: str, params: Optional[Dict[str, Any]] = None) -> None:
    if model_id not in MODEL_CONFIGS: raise ValueError(f"Unknown Family 7 model_id: {model_id}")
    p = dict(params) if params is not None else fixed_params(model_id)
    if "heads" in p and int(p["d_model"]) % int(p["heads"]) != 0:
        raise ValueError(f"d_model={p['d_model']} must be divisible by heads={p['heads']}")
    if model_id == "32" and int(p["top_k"]) > int(p["experts"]):
        raise ValueError("Model 32 top_k cannot exceed experts")
    model = build_model(model_id, p)
    del model; gc.collect()


def validate_model_contracts() -> None:
    for model_id in MODEL_CONFIGS: validate_model_contract(model_id)
    print("[CONTRACT] Family 7 model contracts validated: 28–32", flush=True)


def run_all_models(model_ids: List[str], mode: str, n_trials: int) -> None:
    require_torch(); set_seed(SEED); device = get_device()
    print("=" * 80)
    print("PIPELINE A V2 — FAMILY 7 ADVANCED MULTIMODAL 28–32")
    print("=" * 80)
    print(f"Device       : {device}"); print(f"PyTorch      : {torch.__version__}"); print(f"HIP          : {getattr(torch.version, 'hip', None)}")
    if device.type == "cuda": print(f"GPU          : {torch.cuda.get_device_name(0)}")
    print("ROCm policy  : Model 28 uses pure-PyTorch Mamba; tuned batches capped at 256/512", flush=True)
    print(f"AMP mode     : {'BF16 autocast' if amp_enabled(device) else 'FP32'}", flush=True)
    data = load_frozen_v2(); print(f"Loaded structured={EXPECTED_SHAPE}; train={len(data.train_idx)}, val={len(data.val_idx)}, test={len(data.test_idx)}", flush=True)
    if mode != "trial-worker": validate_model_contracts()
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    results=[]
    for model_id in model_ids:
        if mode == "fixed": result = run_fixed_model(model_id, data, device)
        elif mode == "tuned": result = run_tuned_model(model_id, data, device, n_trials)
        else: raise ValueError(mode)
        results.append(result); pd.DataFrame(results).to_csv(RESULTS_ROOT / f"family07_{mode}_summary.csv", index=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline A V2 Family 7: Advanced Multimodal Architectures (28–32)")
    parser.add_argument("--model", default="all", choices=["all", *MODEL_CONFIGS.keys()])
    parser.add_argument("--mode", required=True, choices=["fixed", "tuned", "stats", "trial-worker"])
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--trial-params-json", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--trial-result-json", default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_trials < 1: raise ValueError("--n-trials must be >= 1")
    if args.mode == "stats":
        evaluate_stats_mode(); return
    if args.mode == "trial-worker":
        if args.model == "all": raise ValueError("--trial-worker requires a concrete --model id")
        if not args.trial_params_json or not args.trial_result_json: raise ValueError("--trial-worker requires --trial-params-json and --trial-result-json")
        run_optuna_trial_worker(args.model, args.trial_params_json, args.trial_result_json); return
    model_ids = list(MODEL_CONFIGS.keys()) if args.model == "all" else [args.model]
    run_all_models(model_ids, args.mode, args.n_trials)


if __name__ == "__main__":
    main()
