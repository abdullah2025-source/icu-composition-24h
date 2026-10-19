#!/usr/bin/env python3
"""
Pipeline A V2 — Family 5: Transformer / Modern Time-Series Architectures (17–23)

Single reusable runner for:
    17 Transformer
    18 Transformer + Conv
    19 Transformer + LSTM
    20 Temporal Fusion Transformer (TFT-style)
    21 PatchTST
    22 TimesNet-inspired multi-period temporal network
    23 iTransformer-style variable-token Transformer

Modes:
    --mode fixed   : run registered fixed baselines
    --mode tuned   : isolated Optuna tuning, then train selected configuration
    --mode stats   : evaluate existing fixed + tuned prediction tables
    --mode trial-worker : internal one-trial child process

Frozen V2 contract:
    structured_24h_processed.npy      [74572, 24, 110]
    demographics_24h_processed.npy    [74572, 4]
    targets/outcomes from the frozen V2 package

Primary trained tasks:
    mortality_24h
    ventilation_onset_24h
    vasopressor_onset_24h
    readmission_30d_24h
    los_hours_24h

Deterioration is evaluation-only noisy-OR and never enters training loss.

LOS model-side transform:
    train  = log1p(max(los_hours_24h - 24, 0))
    report = expm1(predicted_excess_log1p) + 24 hours

Family 5 architecture adaptations use the frozen 24x110 continuous V2 tensor.
They preserve the defining temporal/variable-token ideas of the named models;
these are compact V2 adaptations, not legacy V1 input reproductions.

Shared benchmark seed:
    20260923

Important robustness choices carried forward from Families 2–4:
    * strict frozen-shape/split/target/validity checks
    * precomputed NumPy targets and validity arrays
    * writable copies before torch.from_numpy
    * test never used for model selection
    * validation selection = mean of four binary AUPRCs
    * safe AUROC/AUPRC/Brier helpers in stats
    * canonical test_predictions.csv plus predictions.csv
    * isolated child process for every Optuna trial to isolate ROCm/HIP context
    * BF16, AdamW, num_workers=0, global tuning ranges fixed by V2 contract
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

RESULTS_ROOT = PROJECT_ROOT / "outputs" / "baseline_family_05_transformer_modern_17_23"

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

LOS_LANDMARK_HOURS = 24.0

@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    name: str
    fixed: Dict[str, Any]
    search_space: Dict[str, Any]


MODEL_NAMES = {
    "17": "Transformer",
    "18": "Transformer + Conv",
    "19": "Transformer + LSTM",
    "20": "TFT",
    "21": "PatchTST",
    "22": "TimesNet",
    "23": "iTransformer",
}


MODEL_CONFIGS: Dict[str, ModelConfig] = {
    "17": ModelConfig(
        model_id="17",
        name="Transformer",
        fixed={
            "d_model": 256,
            "n_heads": 8,
            "layers": 4,
            "dropout": 0.15,
            "lr": 8e-4,
        },
        search_space={
            "d_model": "int[128,384] step=8",
            "n_heads": "categorical[4,8]",
            "layers": "int[2,6]",
            "dropout": "float[0.05,0.30]",
            "lr": "float[1e-4,2e-3] log",
        },
    ),
    "18": ModelConfig(
        model_id="18",
        name="Transformer + Conv",
        fixed={
            "d_model": 256,
            "n_heads": 8,
            "layers": 4,
            "dropout": 0.15,
            "lr": 8e-4,
            "conv_kernels": [3, 5],
        },
        search_space={
            "architecture": "same Transformer backbone; canonical table does not numerically define kernel search bounds",
        },
    ),
    "19": ModelConfig(
        model_id="19",
        name="Transformer + LSTM",
        fixed={
            "d_model": 256,
            "n_heads": 8,
            "transformer_layers": 3,
            "lstm_hidden": 256,
            "dropout": 0.15,
            "lr": 8e-4,
        },
        search_space={
            "transformer_layers": "int[2,4]",
            "lstm_hidden": "int[128,384]",
        },
    ),
    "20": ModelConfig(
        model_id="20",
        name="TFT",
        fixed={
            "hidden": 160,
            "heads": 4,
            "dropout": 0.20,
            "lr": 1e-3,
        },
        search_space={
            "hidden": "int[96,256] step=8",
            "heads": "categorical[2,4,8]",
        },
    ),
    "21": ModelConfig(
        model_id="21",
        name="PatchTST",
        fixed={
            "patch_len": 8,
            "stride": 4,
            "d_model": 256,
            "layers": 3,
            "heads": 8,
            "dropout": 0.15,
            "lr": 8e-4,
        },
        search_space={
            "patch_len": "int[4,12]",
            "d_model": "int[128,384] step=8",
            "layers": "int[2,5]",
        },
    ),
    "22": ModelConfig(
        model_id="22",
        name="TimesNet",
        fixed={
            "d_model": 64,
            "d_ff": 128,
            "top_k": 3,
            "dropout": 0.10,
            "lr": 1e-3,
        },
        search_space={
            "d_model": "int[32,128]",
            "top_k": "int[2,5]",
        },
    ),
    "23": ModelConfig(
        model_id="23",
        name="iTransformer",
        fixed={
            "d_model": 256,
            "n_heads": 8,
            "layers": 3,
            "d_ff": 512,
            "dropout": 0.15,
            "lr": 8e-4,
        },
        search_space={
            "d_model": "int[128,384] step=8",
            "layers": "int[2,5]",
        },
    ),
}


# ---------------------------------------------------------------------------
# Reproducibility / runtime helpers
# ---------------------------------------------------------------------------
def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def require_torch() -> None:
    if torch is None:
        raise ImportError("PyTorch is required for Family 5.") from _TORCH_IMPORT_ERROR


def require_optuna() -> None:
    if optuna is None:
        raise ImportError(
            "Optuna is required for tuned mode. Install it in mamba_env."
        )


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required frozen V2 artifact not found: {path}")


def get_device() -> torch.device:
    require_torch()
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def amp_enabled(device: torch.device) -> bool:
    return device.type == "cuda"


# ---------------------------------------------------------------------------
# Frozen V2 data loading
# ---------------------------------------------------------------------------
@dataclass
class FrozenData:
    structured: np.ndarray
    demographics: np.ndarray
    stay_id: np.ndarray
    subject_id: np.ndarray
    split: np.ndarray
    outcomes: pd.DataFrame
    targets: Dict[str, np.ndarray]
    valid_arrays: Dict[str, np.ndarray]
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def validity_mask(df: pd.DataFrame, task: str) -> np.ndarray:
    col = VALID_COL[task]
    if col not in df.columns:
        raise KeyError(
            f"Required frozen V2 validity column '{col}' is missing for '{task}'."
        )
    return df[col].to_numpy(dtype=np.int8) == 1


def load_frozen_v2() -> FrozenData:
    for path in (
        ALIGNMENT_PATH,
        STRUCTURED_PATH,
        DEMOGRAPHICS_PATH,
        INPUT_METADATA_PATH,
        INPUT_DEFINITION_PATH,
        OUTCOMES_PATH,
    ):
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

    out_cols = [
        "stay_id",
        "subject_id",
        *PRIMARY_TASKS,
        *[c for c in outcomes_df.columns if c.endswith("_valid")],
    ]
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

    return FrozenData(
        structured=structured,
        demographics=demographics,
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


class FrozenSequenceDataset(Dataset):
    def __init__(self, data: FrozenData, indices: np.ndarray) -> None:
        self.structured = data.structured
        self.demographics = data.demographics
        self.targets = data.targets
        self.valid_arrays = data.valid_arrays
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, pos: int):
        i = int(self.indices[pos])
        x = np.array(self.structured[i], dtype=np.float32, copy=True)
        d = np.array(self.demographics[i], dtype=np.float32, copy=True)
        y = {task: np.float32(self.targets[task][i]) for task in PRIMARY_TASKS}
        valid = {task: np.int8(self.valid_arrays[task][i]) for task in PRIMARY_TASKS}
        return torch.from_numpy(x), torch.from_numpy(d), y, valid


def make_loader(
    data: FrozenData,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        FrozenSequenceDataset(data, indices),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=pin_memory,
        drop_last=False,
    )


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



class TransformerBackbone(nn.Module):
    def __init__(self, input_dim: int, d_model: int, n_heads: int, layers: int, dropout: float):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.positional = nn.Parameter(torch.randn(24, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.input_proj(x)
        z = z + self.positional[: z.shape[1]].unsqueeze(0)
        z = self.encoder(z)
        return self.dropout(z)


class TransformerBaseline(nn.Module):
    """Compact Transformer encoder with learned temporal positions."""
    def __init__(self, d_model: int, n_heads: int, layers: int, dropout: float):
        super().__init__()
        self.backbone = TransformerBackbone(110, d_model, n_heads, layers, dropout)
        self.demo = nn.Sequential(
            nn.Linear(4, max(32, d_model // 4)),
            nn.LayerNorm(max(32, d_model // 4)),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        demo_dim = max(32, d_model // 4)
        self.fusion = nn.Sequential(
            nn.Linear(d_model + demo_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.heads = MultitaskHead(d_model, dropout)

    def forward(self, x: torch.Tensor, demographics: torch.Tensor):
        z = self.backbone(x)
        temporal = 0.5 * (z.mean(dim=1) + z[:, -1])
        demo = self.demo(demographics)
        fused = self.fusion(torch.cat([temporal, demo], dim=-1))
        return self.heads(fused)


class TransformerConvBaseline(nn.Module):
    """Transformer preceded by parallel local temporal Conv1D kernels 3 and 5."""
    def __init__(self, d_model: int, n_heads: int, layers: int, dropout: float, kernels=(3, 5)):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.input_proj = nn.Sequential(
            nn.Linear(110, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.positional = nn.Parameter(torch.randn(24, d_model) * 0.02)
        self.conv3 = nn.Conv1d(d_model, d_model, kernels[0], padding=kernels[0] // 2, groups=d_model)
        self.conv5 = nn.Conv1d(d_model, d_model, kernels[1], padding=kernels[1] // 2, groups=d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        demo_dim = max(32, d_model // 4)
        self.demo = nn.Sequential(
            nn.Linear(4, demo_dim), nn.LayerNorm(demo_dim), nn.GELU(), nn.Dropout(dropout)
        )
        self.fusion = nn.Sequential(
            nn.Linear(d_model + demo_dim, d_model),
            nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout)
        )
        self.heads = MultitaskHead(d_model, dropout)

    def forward(self, x: torch.Tensor, demographics: torch.Tensor):
        z = self.input_proj(x)
        conv_in = z.transpose(1, 2)
        local = F.gelu(self.conv3(conv_in)) + F.gelu(self.conv5(conv_in))
        z = self.norm(z + local.transpose(1, 2))
        z = z + self.positional[: z.shape[1]].unsqueeze(0)
        z = self.dropout(self.encoder(z))
        temporal = 0.5 * (z.mean(dim=1) + z[:, -1])
        demo = self.demo(demographics)
        fused = self.fusion(torch.cat([temporal, demo], dim=-1))
        return self.heads(fused)


class TransformerLSTMBaseline(nn.Module):
    """Transformer temporal encoder followed by recurrent LSTM aggregation."""
    def __init__(self, d_model: int, n_heads: int, transformer_layers: int, lstm_hidden: int, dropout: float):
        super().__init__()
        self.backbone = TransformerBackbone(110, d_model, n_heads, transformer_layers, dropout)
        self.lstm = nn.LSTM(d_model, lstm_hidden, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.demo = nn.Sequential(
            nn.Linear(4, max(32, lstm_hidden // 4)),
            nn.LayerNorm(max(32, lstm_hidden // 4)), nn.GELU(), nn.Dropout(dropout)
        )
        demo_dim = max(32, lstm_hidden // 4)
        self.fusion = nn.Sequential(
            nn.Linear(lstm_hidden + demo_dim, lstm_hidden),
            nn.LayerNorm(lstm_hidden), nn.GELU(), nn.Dropout(dropout)
        )
        self.heads = MultitaskHead(lstm_hidden, dropout)

    def forward(self, x: torch.Tensor, demographics: torch.Tensor):
        z = self.backbone(x)
        r, _ = self.lstm(z)
        h = self.dropout(0.5 * (r[:, -1] + r.mean(dim=1)))
        demo = self.demo(demographics)
        fused = self.fusion(torch.cat([h, demo], dim=-1))
        return self.heads(fused)


class TFTBaseline(nn.Module):
    """Compact TFT-style V2 adaptation: variable selection, gated recurrence, temporal attention, static context."""
    def __init__(self, hidden: int, heads: int, dropout: float):
        super().__init__()
        if hidden % heads != 0:
            raise ValueError(f"TFT hidden={hidden} must be divisible by heads={heads}")
        self.hidden = hidden
        self.variable_selector = nn.Sequential(
            nn.Linear(110, 110),
            nn.Sigmoid(),
        )
        self.input_proj = nn.Sequential(
            nn.Linear(110, hidden), nn.LayerNorm(hidden), nn.GELU()
        )
        self.static_proj = nn.Sequential(
            nn.Linear(4, hidden), nn.LayerNorm(hidden), nn.GELU()
        )
        self.gru = nn.GRU(hidden, hidden, num_layers=1, batch_first=True)
        self.gate_x = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
        self.attn = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, 4 * hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * hidden, hidden)
        )
        self.norm2 = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)
        self.heads = MultitaskHead(hidden, dropout)

    def forward(self, x: torch.Tensor, demographics: torch.Tensor):
        w = self.variable_selector(x)
        selected = x * w
        z = self.input_proj(selected)
        static = self.static_proj(demographics).unsqueeze(1)
        r, _ = self.gru(z)
        static_rep = static.expand(-1, z.shape[1], -1)
        gate = self.gate_x(torch.cat([r, static_rep], dim=-1))
        gated = gate * r + (1.0 - gate) * z
        attn_out, _ = self.attn(gated, gated, gated, need_weights=False)
        z = self.norm1(gated + self.dropout(attn_out))
        z = self.norm2(z + self.dropout(self.ffn(z)))
        h = 0.5 * (z.mean(dim=1) + z[:, -1])
        return self.heads(h)


class PatchTSTBaseline(nn.Module):
    """PatchTST-style channel-independent temporal patch Transformer."""
    def __init__(self, patch_len: int, stride: int, d_model: int, layers: int, heads: int, dropout: float):
        super().__init__()
        if d_model % heads != 0:
            raise ValueError(f"PatchTST d_model={d_model} must be divisible by heads={heads}")
        if patch_len < 1 or patch_len > 24:
            raise ValueError(f"PatchTST patch_len must be in [1,24], got {patch_len}")
        if stride < 1:
            raise ValueError(f"PatchTST stride must be >=1, got {stride}")
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.num_patches = 1 + (24 - patch_len) // stride
        self.patch_embed = nn.Linear(patch_len, d_model)
        # Dynamic positional table derived from the frozen 24-hour window and
        # current registered patch length/stride; no unexplained magic 8.
        self.patch_pos = nn.Parameter(torch.randn(self.num_patches, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=heads, dim_feedforward=4*d_model,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        demo_dim = max(32, d_model // 4)
        self.demo = nn.Sequential(
            nn.Linear(4, demo_dim), nn.LayerNorm(demo_dim), nn.GELU(), nn.Dropout(dropout)
        )
        self.fusion = nn.Sequential(
            nn.Linear(d_model + demo_dim, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout)
        )
        self.heads = MultitaskHead(d_model, dropout)

    def forward(self, x: torch.Tensor, demographics: torch.Tensor):
        # [B,T,C] -> channel-independent patches [B*C,N,P]
        patches = x.transpose(1, 2).unfold(-1, self.patch_len, self.stride)
        b, c, n, p = patches.shape
        if n != self.patch_pos.shape[0]:
            raise RuntimeError(
                f"PatchTST patch-count mismatch: runtime={n}, registered={self.patch_pos.shape[0]}"
            )
        z = self.patch_embed(patches.reshape(b*c, n, p))
        z = z + self.patch_pos[:n].unsqueeze(0)
        z = self.dropout(self.encoder(z))
        z = self.norm(z).mean(dim=1).reshape(b, c, self.d_model)
        h = z.mean(dim=1)
        demo = self.demo(demographics)
        fused = self.fusion(torch.cat([h, demo], dim=-1))
        return self.heads(fused)


class TimesNetBaseline(nn.Module):
    """Compact TimesNet-inspired multi-period 2D temporal transformation."""
    CANDIDATE_PERIODS = (2, 3, 4, 6, 8, 12)

    def __init__(self, d_model: int, d_ff: int, top_k: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.top_k = top_k
        self.input_proj = nn.Sequential(nn.Linear(110, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.period_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d_model, d_ff, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Dropout2d(dropout),
                nn.Conv2d(d_ff, d_model, kernel_size=3, padding=1),
            )
            for _ in self.CANDIDATE_PERIODS
        ])
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        demo_dim = max(32, d_model // 2)
        self.demo = nn.Sequential(nn.Linear(4, demo_dim), nn.LayerNorm(demo_dim), nn.GELU(), nn.Dropout(dropout))
        self.fusion = nn.Sequential(
            nn.Linear(d_model + demo_dim, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout)
        )
        self.heads = MultitaskHead(d_model, dropout)

    @staticmethod
    def _spectral_energy(x: torch.Tensor, period: int) -> torch.Tensor:
        xf = torch.fft.rfft(x.float(), dim=1)
        freq = max(1, min(xf.shape[1] - 1, round(x.shape[1] / period)))
        return xf[:, freq, :].abs().mean(dim=-1)

    def _period_block(self, z: torch.Tensor, period: int, block: nn.Module) -> torch.Tensor:
        b, t, d = z.shape
        rows = (t + period - 1) // period
        padded = F.pad(z, (0, 0, 0, rows * period - t))
        grid = padded.reshape(b, rows, period, d).permute(0, 3, 1, 2)
        out = block(grid)
        out = out.permute(0, 2, 3, 1).reshape(b, rows * period, d)[:, :t]
        return out

    def forward(self, x: torch.Tensor, demographics: torch.Tensor):
        z = self.input_proj(x)
        energies = torch.stack([self._spectral_energy(z, p) for p in self.CANDIDATE_PERIODS], dim=1)
        k = min(self.top_k, len(self.CANDIDATE_PERIODS))
        top_vals, top_idx = torch.topk(energies, k=k, dim=1)
        weights = torch.softmax(top_vals, dim=1)
        period_outputs = [self._period_block(z, p, block) for p, block in zip(self.CANDIDATE_PERIODS, self.period_blocks)]
        stacked = torch.stack(period_outputs, dim=1)  # [B,P,T,D]
        gather_idx = top_idx[:, :, None, None].expand(-1, -1, stacked.shape[2], stacked.shape[3])
        selected = torch.gather(stacked, 1, gather_idx)
        mix = (selected * weights[:, :, None, None]).sum(dim=1)
        z = self.norm(z + self.dropout(mix))
        h = 0.5 * (z.mean(dim=1) + z[:, -1])
        demo = self.demo(demographics)
        fused = self.fusion(torch.cat([h, demo], dim=-1))
        return self.heads(fused)


class ITransformerBaseline(nn.Module):
    """iTransformer-style variable-token Transformer: each variable is a token over time."""
    def __init__(self, d_model: int, n_heads: int, layers: int, d_ff: int, dropout: float):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"iTransformer d_model={d_model} must be divisible by n_heads={n_heads}")
        self.variable_embed = nn.Linear(24, d_model)
        self.var_pos = nn.Parameter(torch.randn(110, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        demo_dim = max(32, d_model // 4)
        self.demo = nn.Sequential(
            nn.Linear(4, demo_dim), nn.LayerNorm(demo_dim), nn.GELU(), nn.Dropout(dropout)
        )
        self.fusion = nn.Sequential(
            nn.Linear(d_model + demo_dim, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout)
        )
        self.heads = MultitaskHead(d_model, dropout)

    def forward(self, x: torch.Tensor, demographics: torch.Tensor):
        # Each of 110 variables becomes one token represented by its full 24h history.
        tokens = self.variable_embed(x.transpose(1, 2))
        tokens = tokens + self.var_pos.unsqueeze(0)
        tokens = self.dropout(self.encoder(tokens))
        h = self.norm(tokens.mean(dim=1))
        demo = self.demo(demographics)
        fused = self.fusion(torch.cat([h, demo], dim=-1))
        return self.heads(fused)


def build_model(model_id: str, params: Dict[str, Any]) -> nn.Module:
    if model_id == "17":
        return TransformerBaseline(
            d_model=int(params["d_model"]),
            n_heads=int(params["n_heads"]),
            layers=int(params["layers"]),
            dropout=float(params["dropout"]),
        )
    if model_id == "18":
        return TransformerConvBaseline(
            d_model=int(params["d_model"]),
            n_heads=int(params["n_heads"]),
            layers=int(params["layers"]),
            dropout=float(params["dropout"]),
            kernels=tuple(params.get("conv_kernels", [3, 5])),
        )
    if model_id == "19":
        return TransformerLSTMBaseline(
            d_model=int(params["d_model"]),
            n_heads=int(params["n_heads"]),
            transformer_layers=int(params["transformer_layers"]),
            lstm_hidden=int(params["lstm_hidden"]),
            dropout=float(params["dropout"]),
        )
    if model_id == "20":
        return TFTBaseline(
            hidden=int(params["hidden"]),
            heads=int(params["heads"]),
            dropout=float(params["dropout"]),
        )
    if model_id == "21":
        return PatchTSTBaseline(
            patch_len=int(params["patch_len"]),
            stride=int(params.get("stride", 4)),
            d_model=int(params["d_model"]),
            layers=int(params["layers"]),
            heads=int(params["heads"]),
            dropout=float(params["dropout"]),
        )
    if model_id == "22":
        return TimesNetBaseline(
            d_model=int(params["d_model"]),
            d_ff=int(params["d_ff"]),
            top_k=int(params["top_k"]),
            dropout=float(params["dropout"]),
        )
    if model_id == "23":
        return ITransformerBaseline(
            d_model=int(params["d_model"]),
            n_heads=int(params["n_heads"]),
            layers=int(params["layers"]),
            d_ff=int(params["d_ff"]),
            dropout=float(params["dropout"]),
        )
    raise ValueError(f"Unsupported Family 5 model id: {model_id}")

def masked_bce_loss(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    if int(valid.sum()) == 0:
        return logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(logits[valid], target[valid])


def los_loss(log_pred: torch.Tensor, target_hours: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    if int(valid.sum()) == 0:
        return log_pred.sum() * 0.0
    excess = torch.clamp(target_hours[valid] - LOS_LANDMARK_HOURS, min=0.0)
    target_log = torch.log1p(excess)
    return F.smooth_l1_loss(log_pred[valid], target_log)


def multitask_loss(
    outputs: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    valids: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    parts = {
        task: masked_bce_loss(outputs[task], targets[task], valids[task])
        for task in BINARY_TASKS
    }
    parts["los_hours_24h"] = los_loss(
        outputs["los_excess_log1p"],
        targets["los_hours_24h"],
        valids["los_hours_24h"],
    )
    total = sum(parts.values())
    return total, {k: float(v.detach().cpu().item()) for k, v in parts.items()}


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
):
    model.eval()
    pred_records = {task: [] for task in PRIMARY_TASKS}
    valid_records = {task: [] for task in PRIMARY_TASKS}

    for x, demographics, _, valid_dict in loader:
        x = x.to(device, non_blocking=True)
        demographics = demographics.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            outputs = model(x, demographics)

        for task in BINARY_TASKS:
            pred_records[task].append(torch.sigmoid(outputs[task]).float().cpu().numpy())
            valid_records[task].append(valid_dict[task].numpy())

        pred_records["los_hours_24h"].append(
            (torch.expm1(outputs["los_excess_log1p"]) + LOS_LANDMARK_HOURS).float().cpu().numpy()
        )
        valid_records["los_hours_24h"].append(valid_dict["los_hours_24h"].numpy())

    return (
        {task: np.concatenate(pred_records[task]) for task in PRIMARY_TASKS},
        {task: np.concatenate(valid_records[task]) for task in PRIMARY_TASKS},
    )


def safe_auroc(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def safe_auprc(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, p))


def safe_brier(y: np.ndarray, p: np.ndarray) -> float:
    if len(y) == 0:
        return float("nan")
    return float(brier_score_loss(y.astype(int), p))


def evaluate_predictions(
    data: FrozenData,
    pred: Dict[str, np.ndarray],
    valid: Dict[str, np.ndarray],
    indices: np.ndarray,
    split_name: str,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for task in PRIMARY_TASKS:
        y = data.targets[task][indices].astype(float, copy=False)
        p = pred[task].astype(float, copy=False)
        v = valid[task].astype(bool, copy=False)
        keep = v & np.isfinite(y) & np.isfinite(p)
        yt = y[keep]
        pp = p[keep]

        row: Dict[str, Any] = {
            "split": split_name,
            "task": task,
            "valid_n": int(len(yt)),
        }
        if task in BINARY_TASKS:
            row.update(
                {
                    "auroc": safe_auroc(yt.astype(int), pp),
                    "auprc": safe_auprc(yt.astype(int), pp),
                    "brier": safe_brier(yt.astype(int), pp),
                    "prevalence": float(np.mean(yt)) if len(yt) else float("nan"),
                }
            )
        else:
            row.update(
                {
                    "mae": float(mean_absolute_error(yt, pp)) if len(yt) else float("nan"),
                    "rmse": float(np.sqrt(mean_squared_error(yt, pp))) if len(yt) else float("nan"),
                    "median_ae": float(median_absolute_error(yt, pp)) if len(yt) else float("nan"),
                    "r2": float(r2_score(yt, pp)) if len(yt) > 1 else float("nan"),
                    "prevalence": float("nan"),
                }
            )
        rows.append(row)

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
    keep = comp_valid & np.isfinite(y_det) & np.isfinite(p_det)
    rows.append(
        {
            "split": split_name,
            "task": "deterioration_24h",
            "valid_n": int(np.sum(keep)),
            "auroc": safe_auroc(y_det[keep], p_det[keep]),
            "auprc": safe_auprc(y_det[keep], p_det[keep]),
            "brier": safe_brier(y_det[keep], p_det[keep]),
            "prevalence": float(np.mean(y_det[keep])) if np.any(keep) else float("nan"),
        }
    )
    return pd.DataFrame(rows)


def validation_selection_score(metrics: pd.DataFrame) -> float:
    rows = metrics[metrics["task"].isin(BINARY_TASKS)].copy()
    if set(rows["task"]) != set(BINARY_TASKS) or len(rows) != 4:
        return float("-inf")
    values = rows["auprc"].to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        return float("-inf")
    return float(np.mean(values))


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

        for x, demographics, y_dict, valid_dict in train_loader:
            x = x.to(device, non_blocking=True)
            demographics = demographics.to(device, non_blocking=True)
            targets = {task: y_dict[task].to(device, non_blocking=True) for task in PRIMARY_TASKS}
            valids = {
                task: valid_dict[task].to(device, non_blocking=True).bool()
                for task in PRIMARY_TASKS
            }

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                outputs = model(x, demographics)
                loss, _ = multitask_loss(outputs, targets, valids)
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
        raise RuntimeError(f"No valid checkpoint selected for Family 5 model {model_id}.")

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


def base_metadata(
    model_id: str,
    params: Dict[str, Any],
    data: FrozenData,
    device: torch.device,
    mode: str,
    score: float,
    best_epoch: int,
    architecture_notes: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "family": "05_Transformer_Modern",
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
        "input_shape": list(EXPECTED_SHAPE),
        "split_counts": {
            "train": int(len(data.train_idx)),
            "val": int(len(data.val_idx)),
            "test": int(len(data.test_idx)),
        },
        "registered_fixed_params": MODEL_CONFIGS[model_id].fixed,
        "run_params": params,
        "selection_metric": "Mean validation AUPRC across mortality, ventilation, vasopressor, readmission",
        "validation_selection_score": score,
        "best_epoch": best_epoch,
        "test_used_for_selection": False,
        "fixed_tuned_phase_separation": True,
        "optuna_initialization_policy": "Shared SEED=20260923 across HP trials; trial-to-trial variation comes from sampled hyperparameters only.",
        "optuna_trial_execution": "Each Optuna trial runs in an isolated child Python process to prevent ROCm/CUDA allocator/context accumulation across trials.",
        "los_training_transform": "log1p(max(los_hours_24h - 24.0, 0))",
        "los_reporting_transform": "expm1(predicted_excess_log1p) + 24.0 hours",
        "derived_deterioration": "Evaluation-only noisy-OR of mortality, ventilation, vasopressor",
        "optuna_storage": "SQLite study.db with load_if_exists=True; --n-trials is the target total trial count for resumable tuning.",
        "patchtst_rocm_safety": "Model 21 tuning batch choices capped at 256/512 because channel-independent patching expands effective attention batch to B*110.",
        "architecture_notes": architecture_notes,
        "validity_schema": "All primary task validity columns are mandatory; missing validity schema raises an error.",
        "target_schema": "Frozen targets are consumed exactly as stored; -1 invalid labels are excluded through validity masks.",
        "dataset_memory_safety": "Memory-mapped arrays are copied before torch.from_numpy to avoid non-writable tensor warnings.",
    }


# ---------------------------------------------------------------------------
# Fixed / tuned phases
# ---------------------------------------------------------------------------
def fixed_params(model_id: str) -> Dict[str, Any]:
    return {
        **MODEL_CONFIGS[model_id].fixed,
        "batch_size": 512,
        "max_epochs": 50,
        "patience": 7,
        "weight_decay": 1e-4,
        "grad_clip": 1.0,
        "scheduler": "cosine",
    }



def architecture_notes_for_model(model_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
    # IMPORTANT: use if/elif instead of a dict literal. A dict literal evaluates
    # every branch eagerly, so model 17/18/20/21/22/23 could fail on parameters
    # that only exist for model 19 (e.g. transformer_layers/lstm_hidden).
    if model_id == "17":
        return {
            "type": "Temporal Transformer encoder with learned hourly positional embeddings",
            "d_model": params["d_model"],
            "n_heads": params["n_heads"],
            "layers": params["layers"],
        }
    elif model_id == "18":
        return {
            "type": "Transformer + local depthwise Conv1D kernels",
            "d_model": params["d_model"],
            "n_heads": params["n_heads"],
            "layers": params["layers"],
            "conv_kernels": params.get("conv_kernels", [3, 5]),
        }
    elif model_id == "19":
        return {
            "type": "Transformer encoder followed by LSTM aggregation",
            "d_model": params["d_model"],
            "n_heads": params["n_heads"],
            "transformer_layers": params["transformer_layers"],
            "lstm_hidden": params["lstm_hidden"],
        }
    elif model_id == "20":
        return {
            "type": "Compact TFT-style variable selection + gated temporal attention",
            "hidden": params["hidden"],
            "heads": params["heads"],
        }
    elif model_id == "21":
        return {
            "type": "PatchTST-style channel-independent temporal patch Transformer",
            "patch_len": params["patch_len"],
            "stride": params["stride"],
            "d_model": params["d_model"],
            "layers": params["layers"],
            "heads": params["heads"],
        }
    elif model_id == "22":
        return {
            "type": "TimesNet-inspired multi-period 2D temporal transformation",
            "d_model": params["d_model"],
            "d_ff": params["d_ff"],
            "top_k": params["top_k"],
            "candidate_periods": list(TimesNetBaseline.CANDIDATE_PERIODS),
        }
    elif model_id == "23":
        return {
            "type": "iTransformer-style variable-token Transformer",
            "d_model": params["d_model"],
            "n_heads": params["n_heads"],
            "layers": params["layers"],
            "d_ff": params["d_ff"],
            "tokenization": "110 variables are tokens represented from their full 24-hour histories",
        }
    raise KeyError(f"Unknown Family 5 model_id: {model_id}")

def run_fixed_model(model_id: str, data: FrozenData, device: torch.device) -> Dict[str, Any]:
    cfg = MODEL_CONFIGS[model_id]
    out_dir = RESULTS_ROOT / "fixed" / f"{model_id}_{cfg.name.replace(' ', '_')}"
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

    result = {
        "model_id": model_id,
        "model_name": cfg.name,
        "mode": "fixed",
        "validation_selection_score": score,
        "best_epoch": best_epoch,
        "output_dir": str(out_dir),
    }
    print(json.dumps(result, indent=2), flush=True)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def suggest_model_params(model_id: str, trial: "optuna.Trial") -> Dict[str, Any]:
    # Model-specific ranges exactly follow the canonical Family 5 table.
    if model_id == "17":
        model_params = {
            "d_model": trial.suggest_int("d_model", 128, 384, step=8),
            "n_heads": trial.suggest_categorical("n_heads", [4, 8]),
            "layers": trial.suggest_int("layers", 2, 6),
            "dropout": trial.suggest_float("dropout", 0.05, 0.30),
            "lr": trial.suggest_float("lr", 1e-4, 2e-3, log=True),
        }
    elif model_id == "18":
        model_params = {
            "d_model": 256,
            "n_heads": 8,
            "layers": 4,
            "dropout": 0.15,
            "lr": 8e-4,
            "conv_kernels": [3, 5],
        }
    elif model_id == "19":
        model_params = {
            "d_model": 256,
            "n_heads": 8,
            "transformer_layers": trial.suggest_int("transformer_layers", 2, 4),
            "lstm_hidden": trial.suggest_int("lstm_hidden", 128, 384, step=8),
            "dropout": 0.15,
            "lr": 8e-4,
        }
    elif model_id == "20":
        model_params = {
            "hidden": trial.suggest_int("hidden", 96, 256, step=8),
            "heads": trial.suggest_categorical("heads", [2, 4, 8]),
            "dropout": 0.20,
            "lr": 1e-3,
        }
    elif model_id == "21":
        model_params = {
            "patch_len": trial.suggest_int("patch_len", 4, 12),
            "stride": 4,
            "d_model": trial.suggest_int("d_model", 128, 384, step=8),
            "layers": trial.suggest_int("layers", 2, 5),
            "heads": 8,
            "dropout": 0.15,
            "lr": 8e-4,
        }
    elif model_id == "22":
        model_params = {
            "d_model": trial.suggest_int("d_model", 32, 128),
            "d_ff": 128,
            "top_k": trial.suggest_int("top_k", 2, 5),
            "dropout": 0.10,
            "lr": 1e-3,
        }
    elif model_id == "23":
        model_params = {
            "d_model": trial.suggest_int("d_model", 128, 384, step=8),
            "n_heads": 8,
            "layers": trial.suggest_int("layers", 2, 5),
            "d_ff": 512,
            "dropout": 0.15,
            "lr": 8e-4,
        }
    else:
        raise ValueError(model_id)

    # Global V2 tuning ranges, with an explicit PatchTST safety cap.
    # PatchTST folds the channel axis into the effective Transformer batch
    # (B*110); 1024 would create 112,640 concurrent patch sequences. Keep
    # the registered 256/512 options for model 21 to reduce ROCm kernel stress.
    batch_choices = [256, 512] if model_id == "21" else [256, 512, 1024]
    model_params.update(
        {
            "batch_size": trial.suggest_categorical("batch_size", batch_choices),
            "max_epochs": 50,
            "patience": trial.suggest_int("patience", 5, 10),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
            "grad_clip": trial.suggest_float("grad_clip", 0.5, 2.0),
            "scheduler": trial.suggest_categorical("scheduler", ["cosine", "onecycle"]),
        }
    )
    return model_params


def run_optuna_trial_worker(
    model_id: str,
    params_json: str,
    result_json: str,
) -> None:
    """Run exactly one Optuna trial in an isolated child process.

    This prevents CUDA/ROCm allocator/context state from accumulating across
    consecutive trials in the parent Optuna process.
    """
    require_torch()
    # Trial workers are already isolated per Optuna trial. Do not rebuild all
    # seven Family 5 models here; validate only the architecture used by this
    # trial to reduce ROCm startup/allocation pressure and trial overhead.
    validate_model_contract(model_id)
    set_seed(SEED)
    device = get_device()
    params = json.loads(params_json)
    data = load_frozen_v2()

    print(
        f"[TRIAL-WORKER] model={model_id} pid={os.getpid()} "
        f"device={device} params={params}",
        flush=True,
    )

    model, score, best_epoch, _ = train_one(
        model_id,
        params,
        data,
        device,
        run_dir=None,
    )

    result_path = Path(result_json)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(
            {
                "score": float(score),
                "best_epoch": int(best_epoch),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Best-effort cleanup before the child exits. The decisive isolation is the
    # process exit itself, which releases the HIP/CUDA context.
    del model
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        except Exception:
            pass


def run_tuned_model(model_id: str, data: FrozenData, device: torch.device, n_trials: int) -> Dict[str, Any]:
    require_optuna()
    cfg = MODEL_CONFIGS[model_id]
    tuned_root = RESULTS_ROOT / "tuned" / f"{model_id}_{cfg.name.replace(' ', '_')}"
    tuned_root.mkdir(parents=True, exist_ok=True)
    start = time.time()

    sampler = optuna.samplers.TPESampler(seed=SEED)
    study_path = tuned_root / "study.db"
    storage_url = f"sqlite:///{study_path.as_posix()}"
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=f"family05_{model_id}",
        storage=storage_url,
        load_if_exists=True,
    )

    # Treat --n-trials as the total target trial count for this model.
    # This makes rerunning the same command resume an interrupted study
    # instead of starting another N trials from scratch.
    existing_trial_count = len(study.trials)
    remaining_trials = max(0, int(n_trials) - existing_trial_count)
    if existing_trial_count:
        print(
            f"[OPTUNA] model={model_id} persistent study loaded: "
            f"existing_trials={existing_trial_count}, remaining={remaining_trials}",
            flush=True,
        )

    def objective(trial: "optuna.Trial") -> float:
        # Shared initialization seed across Optuna trials by design; only HPs vary.
        params = suggest_model_params(model_id, trial)

        trial_result = tuned_root / f".trial_{trial.number}_result.json"
        if trial_result.exists():
            trial_result.unlink()

        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--model",
            model_id,
            "--mode",
            "trial-worker",
            "--trial-params-json",
            json.dumps(params, separators=(",", ":")),
            "--trial-result-json",
            str(trial_result),
        ]

        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")

        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            check=False,
        )

        if proc.returncode != 0 or not trial_result.exists():
            raise RuntimeError(
                f"Isolated trial worker failed for model {model_id}, "
                f"trial {trial.number}, returncode={proc.returncode}"
            )

        result = json.loads(trial_result.read_text(encoding="utf-8"))
        try:
            trial_result.unlink()
        except OSError:
            pass

        trial.set_user_attr("best_epoch", int(result["best_epoch"]))
        trial.set_user_attr("params", params)
        return float(result["score"])

    if remaining_trials > 0:
        study.optimize(
            objective,
            n_trials=remaining_trials,
            gc_after_trial=False,
            catch=(RuntimeError,),
            show_progress_bar=False,
        )
    else:
        print(
            f"[OPTUNA] model={model_id} already has {existing_trial_count} trials; "
            f"target={n_trials}. Reusing persisted study.",
            flush=True,
        )

    pd.DataFrame(
        [
            {
                "number": t.number,
                "state": str(t.state),
                "value": t.value,
                **t.params,
            }
            for t in study.trials
        ]
    ).to_csv(tuned_root / "optuna_trials.csv", index=False)
    (tuned_root / "study_storage.txt").write_text(str(study_path), encoding="utf-8")

    successful_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not successful_trials:
        raise RuntimeError(f"No successful Optuna trials completed for model {model_id}.")

    best_params = dict(study.best_trial.params)
    best_trial_user_params = study.best_trial.user_attrs.get("params", {})
    # user_attrs keeps the complete global training parameters; params from the
    # trial itself are the canonical values that need to be reproduced.
    final_params = dict(best_trial_user_params) if best_trial_user_params else dict(best_params)

    final_dir = tuned_root / "final_selected_configuration"
    final_dir.mkdir(parents=True, exist_ok=True)
    (final_dir / "runtime_batch_size.txt").write_text(str(final_params["batch_size"]), encoding="utf-8")

    model, score, best_epoch, history = train_one(model_id, final_params, data, device, run_dir=final_dir)
    val_metrics, val_preds = predict_and_write(model, data, device, data.val_idx, "val", final_dir, final_params["batch_size"])
    test_metrics, test_preds = predict_and_write(model, data, device, data.test_idx, "test", final_dir, final_params["batch_size"])

    pd.concat([val_metrics, test_metrics], ignore_index=True).to_csv(final_dir / "metrics.csv", index=False)
    pd.concat([val_preds, test_preds], ignore_index=True).to_csv(final_dir / "predictions.csv", index=False)
    pd.concat([val_preds, test_preds], ignore_index=True).to_csv(final_dir / "test_predictions.csv", index=False)

    arch_notes = {
        **architecture_notes_for_model(model_id, final_params),
        "canonical_search_space": cfg.search_space,
        "selected_model_params": {k: v for k, v in final_params.items() if k not in {"batch_size", "max_epochs", "patience", "weight_decay", "grad_clip", "scheduler"}},
        "global_selected_params": {k: final_params[k] for k in ["batch_size", "patience", "weight_decay", "grad_clip", "scheduler"]},
        "test_used_for_selection": False,
    }

    metadata = base_metadata(model_id, final_params, data, device, "tuned", score, best_epoch, arch_notes)
    metadata.update(
        {
            "n_optuna_trials": n_trials,
            "optuna_best_trial_number": int(study.best_trial.number),
            "optuna_best_value": float(study.best_value),
            "tuning_seconds": time.time() - start,
        }
    )
    (final_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    (tuned_root / "best_params.json").write_text(json.dumps(final_params, indent=2, default=str), encoding="utf-8")

    result = {
        "model_id": model_id,
        "model_name": cfg.name,
        "mode": "tuned",
        "validation_selection_score": score,
        "best_epoch": best_epoch,
        "output_dir": str(final_dir),
        "best_trial": int(study.best_trial.number),
    }
    print(json.dumps(result, indent=2), flush=True)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


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
    final.to_csv(RESULTS_ROOT / "family05_statistical_evaluation.csv", index=False)
    print(f"[STATS] complete -> {RESULTS_ROOT / 'family05_statistical_evaluation.csv'}", flush=True)


# ---------------------------------------------------------------------------
# Runner / CLI
# ---------------------------------------------------------------------------
def validate_model_contract(model_id: str) -> None:
    """Validate and instantiate only one registered Family 5 architecture."""
    if model_id not in MODEL_CONFIGS:
        raise ValueError(f"Unknown Family 5 model_id: {model_id}")

    cfg = MODEL_CONFIGS[model_id]
    params = dict(cfg.fixed)
    if model_id in {"17", "21", "23"} and int(params["d_model"]) % int(
        params["n_heads" if model_id != "21" else "heads"]
    ) != 0:
        raise ValueError(
            f"Model {model_id} has invalid attention dimensions: "
            f"d_model={params['d_model']} heads={params['n_heads' if model_id != '21' else 'heads']}"
        )
    if model_id == "20" and int(params["hidden"]) % int(params["heads"]) != 0:
        raise ValueError(
            f"Model 20 has invalid attention dimensions: hidden={params['hidden']} heads={params['heads']}"
        )

    model = build_model(model_id, params)
    del model
    gc.collect()


def validate_model_contracts() -> None:
    """Instantiate every registered fixed model and validate structural constraints."""
    for model_id in MODEL_CONFIGS:
        validate_model_contract(model_id)
    print("[CONTRACT] Family 5 model contracts validated: 17–23", flush=True)


def run_all_models(model_ids: List[str], mode: str, n_trials: int) -> None:
    require_torch()
    set_seed(SEED)
    device = get_device()

    print("=" * 80)
    print("PIPELINE A V2 — FAMILY 5 TRANSFORMER / MODERN TIME-SERIES 17–23")
    print("=" * 80)
    print(f"Device       : {device}")
    print(f"PyTorch      : {torch.__version__}")
    print(f"HIP          : {getattr(torch.version, 'hip', None)}")
    if device.type == "cuda":
        print(f"GPU          : {torch.cuda.get_device_name(0)}")

    data = load_frozen_v2()
    print(
        f"Loaded structured={EXPECTED_SHAPE}; train={len(data.train_idx)}, "
        f"val={len(data.val_idx)}, test={len(data.test_idx)}",
        flush=True,
    )

    results = []
    for model_id in model_ids:
        if mode == "fixed":
            result = run_fixed_model(model_id, data, device)
        elif mode == "tuned":
            result = run_tuned_model(model_id, data, device, n_trials)
        else:
            raise ValueError(mode)
        results.append(result)
        pd.DataFrame(results).to_csv(RESULTS_ROOT / f"family05_{mode}_summary.csv", index=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline A V2 Family 5: Transformer / Modern Time-Series (17–23)")
    parser.add_argument("--model", default="all", choices=["all", *MODEL_CONFIGS.keys()])
    parser.add_argument("--mode", required=True, choices=["fixed", "tuned", "stats", "trial-worker"])
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--trial-params-json", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--trial-result-json", default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_trials < 1:
        raise ValueError("--n-trials must be >= 1")

    if args.mode == "trial-worker":
        if args.model == "all":
            raise ValueError("--trial-worker requires a concrete --model id.")
        if not args.trial_params_json or not args.trial_result_json:
            raise ValueError(
                "--trial-worker requires --trial-params-json and --trial-result-json."
            )
        run_optuna_trial_worker(
            args.model,
            args.trial_params_json,
            args.trial_result_json,
        )
        return

    if args.mode == "stats":
        evaluate_stats_mode()
        return

    model_ids = list(MODEL_CONFIGS.keys()) if args.model == "all" else [args.model]
    run_all_models(model_ids, args.mode, args.n_trials)


if __name__ == "__main__":
    main()
