#!/usr/bin/env python3
"""
Pipeline A V2 — Family 6: State-Space / Selective SSM Architectures (24–27)

Single reusable runner for:
    24 S4D
    25 Mamba-1
    26 Mamba-2
    27 Mamba-3

Modes:
    --mode fixed        : run registered fixed baselines
    --mode tuned        : isolated Optuna tuning, then train selected configuration
    --mode stats        : evaluate existing fixed + tuned prediction tables
    --mode trial-worker : internal one-trial child process

ROCm / MI210 compatibility policy:
    * Pure PyTorch state-space implementations are the default.
    * No top-level import of mamba_ssm.
    * Mamba-3 is implemented pure PyTorch only on this server.
    * Mamba-1/2 may soft-import mamba_ssm only when FYP_MAMBA_PURE=0; any import
      or constructor failure automatically falls back to the pure PyTorch block.
    * Selective SSM scans run explicitly over T=24 in PyTorch rather than Triton/fused scans.
    * Tuning batches for Models 25–27 are capped at 256/512 because recurrent state
      tensors grow with d_model * d_state * effective batch.

These are compact V2 adaptations of the named state-space families, not claims of
literal reproduction of every upstream implementation detail. The external model
contract remains: structured [B,24,110] + demographics [B,4] -> five multitask heads.

Frozen V2 contract:
    structured_24h_processed.npy      [74572, 24, 110]
    demographics_24h_processed.npy    [74572, 4]
    patient-level splits: 52268 / 7417 / 14887
    validation selection = mean AUPRC across mortality, ventilation, vasopressor, readmission
    LOS model-side transform = log1p(max(LOS_hours - 24, 0))

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

RESULTS_ROOT = PROJECT_ROOT / "outputs" / "baseline_family_06_state_space_24_27"

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
    "24": "S4D",
    "25": "Mamba-1",
    "26": "Mamba-2",
    "27": "Mamba-3",
}


MODEL_CONFIGS: Dict[str, ModelConfig] = {
    "24": ModelConfig(
        model_id="24",
        name="S4D",
        fixed={
            "d_model": 256,
            "state_size": 16,
            "layers": 2,
            "dropout": 0.10,
            "lr": 1e-3,
        },
        search_space={
            "d_model": "int[128,384] step=8",
            "state_size": "categorical[8,16,32]",
            "layers": "int[1,4]",
            "dropout": "float[0.05,0.30]",
            "lr": "float[1e-4,2e-3] log",
        },
    ),
    "25": ModelConfig(
        model_id="25",
        name="Mamba-1",
        fixed={
            "d_model": 256,
            "d_state": 16,
            "d_conv": 4,
            "expand": 2,
            "layers": 2,
            "dropout": 0.15,
            "lr": 8e-4,
        },
        search_space={
            "d_model": "int[128,384] step=8",
            "d_state": "categorical[8,16,32]",
            "d_conv": "categorical[3,4,5]",
            "expand": "categorical[2,4]",
            "layers": "int[1,4]",
            "dropout": "float[0.05,0.30]",
            "lr": "float[1e-4,2e-3] log",
        },
    ),
    "26": ModelConfig(
        model_id="26",
        name="Mamba-2",
        fixed={
            "d_model": 256,
            "d_state": 16,
            "d_conv": 4,
            "expand": 2,
            "heads": 8,
            "layers": 2,
            "dropout": 0.15,
            "lr": 8e-4,
        },
        search_space={
            "d_model": "int[128,384] step=8",
            "d_state": "categorical[8,16,32]",
            "expand": "categorical[2,4]",
            "heads": "categorical[4,8]",
            "layers": "int[1,4]",
            "dropout": "float[0.05,0.30]",
            "lr": "float[1e-4,2e-3] log",
        },
    ),
    "27": ModelConfig(
        model_id="27",
        name="Mamba-3",
        fixed={
            "d_model": 256,
            "d_state": 16,
            "d_conv": 4,
            "expand": 2,
            "layers": 2,
            "dropout": 0.15,
            "lr": 8e-4,
            "pure_pytorch": True,
        },
        search_space={
            "d_model": "int[128,384] step=8",
            "d_state": "categorical[8,16,32]",
            "expand": "categorical[2,4]",
            "layers": "int[1,4]",
            "dropout": "float[0.05,0.30]",
            "lr": "float[1e-4,2e-3] log",
            "implementation": "pure PyTorch only on MI210/ROCm",
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
        raise ImportError("PyTorch is required for Family 6.") from _TORCH_IMPORT_ERROR


def require_optuna() -> None:
    if optuna is None:
        raise ImportError("Optuna is required for tuned mode. Install it in mamba_env.")


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required frozen V2 artifact not found: {path}")


def get_device() -> torch.device:
    require_torch()
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def amp_enabled(device: torch.device) -> bool:
    return device.type == "cuda"


def mamba_pure_default() -> bool:
    return os.environ.get("FYP_MAMBA_PURE", "1") == "1"


def rocm_visible_message() -> str:
    return (
        f"FYP_MAMBA_PURE={os.environ.get('FYP_MAMBA_PURE', '1')} "
        f"HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES', '<unset>')} "
        f"ROCR_VISIBLE_DEVICES={os.environ.get('ROCR_VISIBLE_DEVICES', '<unset>')}"
    )


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
        raise KeyError(f"Required frozen V2 validity column '{col}' is missing for '{task}'.")
    return df[col].to_numpy(dtype=np.int8) == 1


def load_frozen_v2() -> FrozenData:
    for path in (ALIGNMENT_PATH, STRUCTURED_PATH, DEMOGRAPHICS_PATH, INPUT_METADATA_PATH, INPUT_DEFINITION_PATH, OUTCOMES_PATH):
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
        structured=structured, demographics=demographics, stay_id=merged["stay_id"].to_numpy(),
        subject_id=merged["subject_id"].to_numpy(), split=split, outcomes=merged, targets=targets,
        valid_arrays=valid_arrays, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
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


class S4DLayer(nn.Module):
    """ROCm-safe real-valued diagonal SSM inspired by S4D.

    The short T=24 sequence is scanned explicitly in PyTorch. No FFT/Triton/fused kernel
    dependency is used. The implementation intentionally stays real-valued for BF16/ROCm stability.
    """
    def __init__(self, d_model: int, state_size: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.state_size = state_size
        init = torch.arange(1, state_size + 1, dtype=torch.float32).log()
        self.A_log = nn.Parameter(init.unsqueeze(0).repeat(d_model, 1))
        self.B = nn.Parameter(torch.randn(d_model, state_size) * 0.05)
        self.C = nn.Parameter(torch.randn(d_model, state_size) * 0.05)
        self.D = nn.Parameter(torch.ones(d_model))
        self.dt_log = nn.Parameter(torch.full((d_model,), -2.0))
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        x32 = x.float()
        A = -F.softplus(self.A_log).unsqueeze(0)  # [1,D,N]
        B = self.B.float().unsqueeze(0)
        C = self.C.float().unsqueeze(0)
        D = self.D.float().view(1, d)
        dt = F.softplus(self.dt_log.float()).view(1, d, 1).clamp(1e-4, 1.0)
        state = torch.zeros((b, d, self.state_size), device=x.device, dtype=torch.float32)
        ys: List[torch.Tensor] = []
        for ti in range(t):
            xt = x32[:, ti, :]
            decay = torch.exp(dt * A)
            state = decay * state + dt * B * xt.unsqueeze(-1)
            yt = (C * state).sum(dim=-1) + D * xt
            ys.append(yt)
        y = torch.stack(ys, dim=1).to(dtype=x.dtype)
        return self.dropout(self.norm(y))


class S4DBackbone(nn.Module):
    def __init__(self, input_dim: int, d_model: int, state_size: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(input_dim, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.positional = nn.Parameter(torch.randn(24, d_model) * 0.02)
        self.layers = nn.ModuleList([S4DLayer(d_model, state_size, dropout) for _ in range(layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.input_proj(x) + self.positional[: x.shape[1]]
        for layer in self.layers:
            z = z + layer(z)
        return self.dropout(z)


class PureMambaSelectiveBlock(nn.Module):
    """Mamba-1-style selective SSM implemented entirely in PyTorch."""
    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand
        self.dt_rank = max(8, min(32, d_model // 16))
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner)
        self.conv = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv, groups=self.d_inner, bias=True, padding=d_conv - 1)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner)
        self.A_log = nn.Parameter(torch.zeros(self.d_inner, d_state))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model)
        self.norm = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        uv = self.in_proj(x)
        u, gate = uv.chunk(2, dim=-1)
        u = self.conv(u.transpose(1, 2))[..., : x.shape[1]].transpose(1, 2)
        u = F.silu(u)
        x_db = self.x_proj(u)
        dt_raw, B_t, C_t = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt_raw.float())).clamp(1e-4, 1.0)
        A = -torch.exp(self.A_log.float()).unsqueeze(0)
        state = torch.zeros((x.shape[0], self.d_inner, self.d_state), device=x.device, dtype=torch.float32)
        ys: List[torch.Tensor] = []
        u32 = u.float()
        B_t = B_t.float()
        C_t = C_t.float()
        for ti in range(x.shape[1]):
            dt_t = dt[:, ti, :].unsqueeze(-1)
            inp = u32[:, ti, :].unsqueeze(-1)
            decay = torch.exp(dt_t * A)
            state = decay * state + dt_t * inp * B_t[:, ti, :].unsqueeze(1)
            y_t = (state * C_t[:, ti, :].unsqueeze(1)).sum(dim=-1) + self.D.float().view(1, -1) * u32[:, ti, :]
            ys.append(y_t)
        y = torch.stack(ys, dim=1).to(dtype=x.dtype)
        y = y * torch.sigmoid(gate)
        y = self.out_proj(y)
        return residual + self.dropout(y)


class Mamba2PureBlock(nn.Module):
    """Mamba-2-style multi-head selective SSM adaptation in pure PyTorch."""
    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand
        if self.d_inner % heads != 0:
            raise ValueError(f"Mamba-2 d_inner={self.d_inner} must be divisible by heads={heads}")
        self.heads = heads
        self.head_dim = self.d_inner // heads
        self.d_state = d_state
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner)
        self.conv = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv, groups=self.d_inner, padding=d_conv - 1)
        self.dt_proj = nn.Linear(self.head_dim, 1)
        self.B_proj = nn.Linear(self.head_dim, d_state, bias=False)
        self.C_proj = nn.Linear(self.head_dim, d_state, bias=False)
        self.A_log = nn.Parameter(torch.zeros(heads, d_state))
        self.D = nn.Parameter(torch.ones(heads, self.head_dim))
        self.out_proj = nn.Linear(self.d_inner, d_model)
        self.norm = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        uv = self.in_proj(x)
        u, gate = uv.chunk(2, dim=-1)
        u = self.conv(u.transpose(1, 2))[..., : x.shape[1]].transpose(1, 2)
        u = F.silu(u).view(x.shape[0], x.shape[1], self.heads, self.head_dim)
        u32 = u.float()
        dt = F.softplus(self.dt_proj(u32).squeeze(-1)).clamp(1e-4, 1.0)
        B_t = self.B_proj(u32).float()
        C_t = self.C_proj(u32).float()
        A = -torch.exp(self.A_log.float()).view(1, self.heads, self.d_state)
        # Preserve an independent SSM state for every head_dim channel:
        # [batch, heads, head_dim, d_state]. Mamba-2 shares A within a head,
        # but does not average head_dim channels into one scalar trajectory.
        state = torch.zeros(
            (x.shape[0], self.heads, self.head_dim, self.d_state),
            device=x.device, dtype=torch.float32,
        )
        ys: List[torch.Tensor] = []
        for ti in range(x.shape[1]):
            dtt = dt[:, ti, :].unsqueeze(-1)                    # [B,H,1]
            decay = torch.exp(dtt * A).unsqueeze(-2)             # [B,H,1,N]
            inp = u32[:, ti].unsqueeze(-1)                      # [B,H,Dh,1]
            B_step = B_t[:, ti].unsqueeze(-2)                  # [B,H,1,N]
            C_step = C_t[:, ti].unsqueeze(-2)                  # [B,H,1,N]
            state = decay * state + dtt.unsqueeze(-2) * inp * B_step
            y_t = (state * C_step).sum(dim=-1)                 # [B,H,Dh]
            y_t = y_t + self.D.float().unsqueeze(0) * u32[:, ti]
            ys.append(y_t)
        y = torch.stack(ys, dim=1).to(dtype=x.dtype).reshape(x.shape[0], x.shape[1], self.d_inner)
        y = y * torch.sigmoid(gate)
        y = self.out_proj(y)
        return residual + self.dropout(y)


class PureMamba3Block(nn.Module):
    """ROCm-safe Mamba-3-style selective SSM adaptation.

    This is intentionally pure PyTorch. The scan adds a learned state correction term
    and retains per-step selective dt/B/C generation while avoiding Triton/fused kernels.
    """
    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand
        self.d_state = d_state
        self.dt_rank = max(8, min(32, d_model // 16))
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner)
        self.conv = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv, groups=self.d_inner, padding=d_conv - 1)
        self.param_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner)
        self.A_log = nn.Parameter(torch.zeros(self.d_inner, d_state))
        self.beta = nn.Parameter(torch.ones(self.d_inner))
        self.gamma = nn.Parameter(torch.zeros(self.d_inner))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model)
        self.norm = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        uv = self.in_proj(x)
        u, gate = uv.chunk(2, dim=-1)
        u = self.conv(u.transpose(1, 2))[..., : x.shape[1]].transpose(1, 2)
        u = F.silu(u)
        params = self.param_proj(u)
        dt_raw, B_t, C_t = torch.split(params, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt_raw.float())).clamp(1e-4, 1.0)
        A = -torch.exp(self.A_log.float()).unsqueeze(0)
        state = torch.zeros((x.shape[0], self.d_inner, self.d_state), device=x.device, dtype=torch.float32)
        prev_u = torch.zeros((x.shape[0], self.d_inner), device=x.device, dtype=torch.float32)
        ys: List[torch.Tensor] = []
        u32 = u.float(); B_t = B_t.float(); C_t = C_t.float()
        for ti in range(x.shape[1]):
            ut = u32[:, ti, :]
            dtt = dt[:, ti, :].unsqueeze(-1)
            decay = torch.exp(dtt * A)
            correction = (ut - prev_u).unsqueeze(-1) * self.gamma.float().view(1, -1, 1)
            state = decay * state + dtt * ut.unsqueeze(-1) * B_t[:, ti, :].unsqueeze(1) + correction
            y_t = (state * C_t[:, ti, :].unsqueeze(1)).sum(dim=-1) + self.D.float().view(1, -1) * ut
            y_t = y_t + self.beta.float().view(1, -1) * (ut - prev_u)
            ys.append(y_t)
            prev_u = ut
        y = torch.stack(ys, dim=1).to(dtype=x.dtype)
        y = y * torch.sigmoid(gate)
        y = self.out_proj(y)
        return residual + self.dropout(y)


class ExternalMambaAdapter(nn.Module):
    """Best-effort adapter for optional CUDA-centric mamba_ssm installs.

    This path is disabled by default on ROCm/MI210. Import errors or constructor
    incompatibilities automatically fall back to the pure PyTorch implementations.
    """
    def __init__(self, kind: str, d_model: int, d_state: int, d_conv: int, expand: int, heads: int = 8) -> None:
        super().__init__()
        self.kind = kind
        try:
            from mamba_ssm import Mamba  # type: ignore
            Mamba2 = getattr(__import__("mamba_ssm", fromlist=["Mamba2"]), "Mamba2", None)
            cls = Mamba if kind == "mamba1" else Mamba2
            if cls is None:
                raise ImportError("mamba_ssm.Mamba2 is unavailable")
            if kind == "mamba1":
                self.block = cls(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            else:
                try:
                    self.block = cls(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
                except TypeError:
                    self.block = cls(d_model=d_model, d_state=d_state, d_conv=d_conv, headdim=max(16, d_model // heads), expand=expand)
        except Exception as exc:
            raise RuntimeError(f"Optional mamba_ssm path unavailable: {exc}") from exc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


def build_mamba_block(kind: str, d_model: int, d_state: int, d_conv: int, expand: int, dropout: float, heads: int = 8) -> nn.Module:
    # ROCm/MI210: pure PyTorch is the default and preferred path.
    use_pure = mamba_pure_default() or kind == "mamba3"
    if not use_pure and kind in {"mamba1", "mamba2"}:
        try:
            print(f"[MAMBA] attempting optional mamba_ssm for {kind} because FYP_MAMBA_PURE=0", flush=True)
            return ExternalMambaAdapter(kind, d_model, d_state, d_conv, expand, heads=heads)
        except Exception as exc:
            print(f"[WARN] mamba_ssm unavailable/incompatible ({exc}); using pure PyTorch {kind}", flush=True)
    if kind == "mamba1":
        return PureMambaSelectiveBlock(d_model, d_state, d_conv, expand, dropout)
    if kind == "mamba2":
        return Mamba2PureBlock(d_model, d_state, d_conv, expand, heads, dropout)
    if kind == "mamba3":
        return PureMamba3Block(d_model, d_state, d_conv, expand, dropout)
    raise ValueError(kind)


class StateSpaceStack(nn.Module):
    def __init__(self, model_id: str, params: Dict[str, Any]) -> None:
        super().__init__()
        self.model_id = model_id
        d_model = int(params["d_model"])
        dropout = float(params["dropout"])
        layers = int(params["layers"])
        self.input_proj = nn.Sequential(nn.Linear(110, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.positional = nn.Parameter(torch.randn(24, d_model) * 0.02)
        blocks: List[nn.Module] = []
        for _ in range(layers):
            if model_id == "24":
                blocks.append(S4DLayer(d_model, int(params["state_size"]), dropout))
            elif model_id == "25":
                blocks.append(build_mamba_block("mamba1", d_model, int(params["d_state"]), int(params["d_conv"]), int(params["expand"]), dropout))
            elif model_id == "26":
                blocks.append(build_mamba_block("mamba2", d_model, int(params["d_state"]), int(params["d_conv"]), int(params["expand"]), dropout, heads=int(params["heads"])))
            elif model_id == "27":
                blocks.append(build_mamba_block("mamba3", d_model, int(params["d_state"]), int(params["d_conv"]), int(params["expand"]), dropout))
            else:
                raise ValueError(model_id)
        self.blocks = nn.ModuleList(blocks)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.input_proj(x) + self.positional[: x.shape[1]]
        for block in self.blocks:
            z = block(z)
        return self.final_norm(z)


class StateSpaceBaseline(nn.Module):
    def __init__(self, model_id: str, params: Dict[str, Any]) -> None:
        super().__init__()
        self.backbone = StateSpaceStack(model_id, params)
        d_model = int(params["d_model"])
        dropout = float(params["dropout"])
        self.demo_proj = nn.Sequential(nn.Linear(d_model + 4, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout))
        self.head = MultitaskHead(d_model, dropout)

    def forward(self, x: torch.Tensor, demographics: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.backbone(x).mean(dim=1)
        z = self.demo_proj(torch.cat([z, demographics], dim=-1))
        return self.head(z)


class S4DBaseline(StateSpaceBaseline):
    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__("24", params)


class Mamba1Baseline(StateSpaceBaseline):
    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__("25", params)


class Mamba2Baseline(StateSpaceBaseline):
    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__("26", params)


class Mamba3Baseline(StateSpaceBaseline):
    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__("27", params)


def build_model(model_id: str, params: Dict[str, Any]) -> nn.Module:
    require_torch()
    if model_id == "24":
        return S4DBaseline(params)
    if model_id == "25":
        return Mamba1Baseline(params)
    if model_id == "26":
        return Mamba2Baseline(params)
    if model_id == "27":
        return Mamba3Baseline(params)
    raise ValueError(f"Unknown Family 6 model_id: {model_id}")


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
    for x, demographics, y_dict, valid_dict in loader:
        x = x.to(device, non_blocking=True)
        demographics = demographics.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            out = model(x, demographics)
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
        raise RuntimeError(f"No valid checkpoint selected for Family 6 model {model_id}.")

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
    if model_id == "24":
        return {
            "type": "S4D-style real diagonal state-space temporal block",
            "d_model": params["d_model"], "state_size": params["state_size"], "layers": params["layers"],
            "scan": "explicit PyTorch time loop over 24 hours; real-valued for ROCm/BF16 stability",
        }
    if model_id == "25":
        return {
            "type": "Mamba-1-style selective SSM adaptation",
            "d_model": params["d_model"], "d_state": params["d_state"], "d_conv": params["d_conv"], "expand": params["expand"], "layers": params["layers"],
            "implementation": "pure PyTorch by default; optional mamba_ssm only when FYP_MAMBA_PURE=0",
            "scan": "explicit PyTorch time loop; no Triton fused scan in default path",
        }
    if model_id == "26":
        return {
            "type": "Mamba-2-style multi-head selective SSM adaptation",
            "d_model": params["d_model"], "d_state": params["d_state"], "d_conv": params["d_conv"], "expand": params["expand"], "heads": params["heads"], "layers": params["layers"],
            "implementation": "pure PyTorch by default; optional mamba_ssm only when FYP_MAMBA_PURE=0",
            "scan": "explicit PyTorch time loop; no Triton fused scan in default path",
        }
    if model_id == "27":
        return {
            "type": "Mamba-3-style selective SSM adaptation",
            "d_model": params["d_model"], "d_state": params["d_state"], "d_conv": params["d_conv"], "expand": params["expand"], "layers": params["layers"],
            "implementation": "pure PyTorch only; ROCm/MI210-safe path; mamba_ssm/Triton intentionally not used",
            "scan": "explicit PyTorch time loop over 24 hours with learned state correction",
        }
    raise KeyError(f"Unknown Family 6 model_id: {model_id}")


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
    result = {"model_id": model_id, "model_name": cfg.name, "mode": "fixed", "validation_selection_score": score, "best_epoch": best_epoch, "output_dir": str(out_dir)}
    print(json.dumps(result, indent=2), flush=True)
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return result


def suggest_model_params(model_id: str, trial: "optuna.Trial") -> Dict[str, Any]:
    if model_id == "24":
        model_params = {
            "d_model": trial.suggest_int("d_model", 128, 384, step=8),
            "state_size": trial.suggest_categorical("state_size", [8, 16, 32]),
            "layers": trial.suggest_int("layers", 1, 4),
            "dropout": trial.suggest_float("dropout", 0.05, 0.30),
            "lr": trial.suggest_float("lr", 1e-4, 2e-3, log=True),
        }
    elif model_id == "25":
        model_params = {
            "d_model": trial.suggest_int("d_model", 128, 384, step=8),
            "d_state": trial.suggest_categorical("d_state", [8, 16, 32]),
            "d_conv": trial.suggest_categorical("d_conv", [3, 4, 5]),
            "expand": trial.suggest_categorical("expand", [2, 4]),
            "layers": trial.suggest_int("layers", 1, 4),
            "dropout": trial.suggest_float("dropout", 0.05, 0.30),
            "lr": trial.suggest_float("lr", 1e-4, 2e-3, log=True),
        }
    elif model_id == "26":
        # Keep heads modest and enforce d_model*expand divisibility in the validator.
        model_params = {
            "d_model": trial.suggest_int("d_model", 128, 384, step=8),
            "d_state": trial.suggest_categorical("d_state", [8, 16, 32]),
            "d_conv": 4,
            "expand": trial.suggest_categorical("expand", [2, 4]),
            "heads": trial.suggest_categorical("heads", [4, 8]),
            "layers": trial.suggest_int("layers", 1, 4),
            "dropout": trial.suggest_float("dropout", 0.05, 0.30),
            "lr": trial.suggest_float("lr", 1e-4, 2e-3, log=True),
        }
    elif model_id == "27":
        model_params = {
            "d_model": trial.suggest_int("d_model", 128, 384, step=8),
            "d_state": trial.suggest_categorical("d_state", [8, 16, 32]),
            "d_conv": 4,
            "expand": trial.suggest_categorical("expand", [2, 4]),
            "layers": trial.suggest_int("layers", 1, 4),
            "dropout": trial.suggest_float("dropout", 0.05, 0.30),
            "lr": trial.suggest_float("lr", 1e-4, 2e-3, log=True),
            "pure_pytorch": True,
        }
    else:
        raise ValueError(model_id)

    # Global V2 search, with a safety cap for recurrent state-space models.
    batch_choices = [256, 512] if model_id in {"25", "26", "27"} else [256, 512, 1024]
    model_params.update({
        "batch_size": trial.suggest_categorical("batch_size", batch_choices),
        "max_epochs": 50,
        "patience": trial.suggest_int("patience", 5, 10),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
        "grad_clip": trial.suggest_float("grad_clip", 0.5, 2.0),
        "scheduler": trial.suggest_categorical("scheduler", ["cosine", "onecycle"]),
    })
    if model_id == "26" and (int(model_params["d_model"]) * int(model_params["expand"])) % int(model_params["heads"]) != 0:
        # Keep the trial valid without wasting a worker on an impossible shape.
        raise ValueError("Invalid Mamba-2 trial: d_model*expand must be divisible by heads")
    return model_params


def run_optuna_trial_worker(model_id: str, params_json: str, result_json: str) -> None:
    require_torch()
    # Trial workers are already isolated; validate only the current architecture,
    # never all four Family 6 models. This avoids repeated allocations on ROCm.
    params = json.loads(params_json)
    validate_model_contract(model_id, params=params)
    set_seed(SEED)
    device = get_device()
    data = load_frozen_v2()
    print(f"[TRIAL-WORKER] family=06 model={model_id} pid={os.getpid()} device={device} {rocm_visible_message()} params={params}", flush=True)
    model, score, best_epoch, _ = train_one(model_id, params, data, device, run_dir=None)
    result_path = Path(result_json); result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps({"score": float(score), "best_epoch": int(best_epoch)}, indent=2), encoding="utf-8")
    del model; gc.collect()
    if torch.cuda.is_available():
        try: torch.cuda.synchronize(); torch.cuda.empty_cache()
        except Exception: pass


def run_tuned_model(model_id: str, data: FrozenData, device: torch.device, n_trials: int) -> Dict[str, Any]:
    require_optuna()
    cfg = MODEL_CONFIGS[model_id]
    tuned_root = RESULTS_ROOT / "tuned" / f"{model_id}_{cfg.name.replace(' ', '_')}"
    tuned_root.mkdir(parents=True, exist_ok=True)
    start = time.time()
    sampler = optuna.samplers.TPESampler(seed=SEED)
    study_path = tuned_root / "study.db"
    storage_url = f"sqlite:///{study_path.as_posix()}"
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name=f"family06_{model_id}", storage=storage_url, load_if_exists=True)
    existing_trial_count = len(study.trials)
    remaining_trials = max(0, int(n_trials) - existing_trial_count)
    if existing_trial_count:
        print(f"[OPTUNA] model={model_id} persistent study loaded: existing_trials={existing_trial_count}, remaining={remaining_trials}", flush=True)

    def objective(trial: "optuna.Trial") -> float:
        params = suggest_model_params(model_id, trial)
        validate_model_contract(model_id, params=params)
        trial_result = tuned_root / f".trial_{trial.number}_result.json"
        if trial_result.exists(): trial_result.unlink()
        cmd = [sys.executable, str(Path(__file__).resolve()), "--model", model_id, "--mode", "trial-worker", "--trial-params-json", json.dumps(params, separators=(",", ":")), "--trial-result-json", str(trial_result)]
        env = os.environ.copy(); env.setdefault("PYTHONUNBUFFERED", "1")
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env, check=False)
        if proc.returncode != 0 or not trial_result.exists():
            raise RuntimeError(f"Isolated trial worker failed for Family 6 model {model_id}, trial {trial.number}, returncode={proc.returncode}")
        result = json.loads(trial_result.read_text(encoding="utf-8"))
        try: trial_result.unlink()
        except OSError: pass
        trial.set_user_attr("best_epoch", int(result["best_epoch"]))
        trial.set_user_attr("params", params)
        return float(result["score"])

    if remaining_trials > 0:
        study.optimize(objective, n_trials=remaining_trials, gc_after_trial=False, catch=(RuntimeError, ValueError), show_progress_bar=False)
    else:
        print(f"[OPTUNA] model={model_id} already has {existing_trial_count} trials; target={n_trials}. Reusing persisted study.", flush=True)

    pd.DataFrame([{ "number": t.number, "state": str(t.state), "value": t.value, **t.params } for t in study.trials]).to_csv(tuned_root / "optuna_trials.csv", index=False)
    (tuned_root / "study_storage.txt").write_text(str(study_path), encoding="utf-8")
    successful_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not successful_trials: raise RuntimeError(f"No successful Optuna trials completed for Family 6 model {model_id}.")
    best_trial = study.best_trial
    best_trial_user_params = best_trial.user_attrs.get("params", {})
    final_params = dict(best_trial_user_params) if best_trial_user_params else dict(best_trial.params)
    validate_model_contract(model_id, params=final_params)
    final_dir = tuned_root / "final_selected_configuration"; final_dir.mkdir(parents=True, exist_ok=True)
    model, score, best_epoch, history = train_one(model_id, final_params, data, device, run_dir=final_dir)
    val_metrics, val_preds = predict_and_write(model, data, device, data.val_idx, "val", final_dir, final_params["batch_size"])
    test_metrics, test_preds = predict_and_write(model, data, device, data.test_idx, "test", final_dir, final_params["batch_size"])
    pd.concat([val_metrics, test_metrics], ignore_index=True).to_csv(final_dir / "metrics.csv", index=False)
    pd.concat([val_preds, test_preds], ignore_index=True).to_csv(final_dir / "predictions.csv", index=False)
    pd.concat([val_preds, test_preds], ignore_index=True).to_csv(final_dir / "test_predictions.csv", index=False)
    arch_notes = {**architecture_notes_for_model(model_id, final_params), "canonical_search_space": cfg.search_space, "selected_model_params": {k:v for k,v in final_params.items() if k not in {"batch_size","max_epochs","patience","weight_decay","grad_clip","scheduler"}}, "global_selected_params": {k: final_params[k] for k in ["batch_size","patience","weight_decay","grad_clip","scheduler"]}, "test_used_for_selection": False}
    metadata = base_metadata(model_id, final_params, data, device, "tuned", score, best_epoch, arch_notes)
    metadata.update({"n_optuna_trials": n_trials, "optuna_best_trial_number": int(study.best_trial.number), "optuna_best_value": float(study.best_value), "tuning_seconds": time.time()-start})
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
        "family": "06_State_Space_24_27",
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
        "mamba_pure_default": mamba_pure_default(),
        "rocm_runtime_flags": rocm_visible_message(),
        "input_shape": list(EXPECTED_SHAPE),
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
        "mamba_rocm_policy": "Pure PyTorch default (FYP_MAMBA_PURE=1); Model 27 is pure PyTorch only. Optional mamba_ssm path is soft-imported only when explicitly requested for Models 25/26.",
        "mamba_scan_policy": "Explicit PyTorch time-loop scan over 24 steps; no Triton/fused scan in the default ROCm-safe path.",
        "validity_schema": "All primary task validity columns are mandatory; missing validity schema raises an error.",
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
    final.to_csv(RESULTS_ROOT / "family06_statistical_evaluation.csv", index=False)
    print(f"[STATS] complete -> {RESULTS_ROOT / 'family06_statistical_evaluation.csv'}", flush=True)


# ---------------------------------------------------------------------------
# Runner / CLI
# ---------------------------------------------------------------------------
def validate_model_contract(model_id: str, params: Optional[Dict[str, Any]] = None) -> None:
    if model_id not in MODEL_CONFIGS:
        raise ValueError(f"Unknown Family 6 model_id: {model_id}")
    p = dict(params) if params is not None else fixed_params(model_id)
    if model_id == "24":
        if int(p["d_model"]) < 1 or int(p["state_size"]) < 1: raise ValueError("Invalid S4D dimensions")
    elif model_id == "25":
        if int(p["d_model"]) < 1 or int(p["d_state"]) < 1 or int(p["d_conv"]) < 2 or int(p["expand"]) < 1: raise ValueError("Invalid Mamba-1 dimensions")
    elif model_id == "26":
        d_inner = int(p["d_model"]) * int(p["expand"])
        if d_inner % int(p["heads"]) != 0: raise ValueError(f"Mamba-2 d_inner={d_inner} must be divisible by heads={p['heads']}")
    elif model_id == "27":
        if not bool(p.get("pure_pytorch", True)): raise ValueError("Mamba-3 is required to use pure PyTorch implementation")
    model = build_model(model_id, p)
    del model; gc.collect()


def validate_model_contracts() -> None:
    for model_id in MODEL_CONFIGS:
        validate_model_contract(model_id)
    print("[CONTRACT] Family 6 model contracts validated: 24–27", flush=True)


def run_all_models(model_ids: List[str], mode: str, n_trials: int) -> None:
    require_torch(); set_seed(SEED); device = get_device()
    print("=" * 80)
    print("PIPELINE A V2 — FAMILY 6 STATE-SPACE / SELECTIVE SSM 24–27")
    print("=" * 80)
    print(f"Device       : {device}")
    print(f"PyTorch      : {torch.__version__}")
    print(f"HIP          : {getattr(torch.version, 'hip', None)}")
    if device.type == "cuda": print(f"GPU          : {torch.cuda.get_device_name(0)}")
    print(f"Mamba policy : {rocm_visible_message()} | pure_default={mamba_pure_default()}")
    data = load_frozen_v2()
    print(f"Loaded structured={EXPECTED_SHAPE}; train={len(data.train_idx)}, val={len(data.val_idx)}, test={len(data.test_idx)}", flush=True)
    if mode != "trial-worker": validate_model_contracts()
    results = []
    for model_id in model_ids:
        if mode == "fixed":
            result = run_fixed_model(model_id, data, device)
        elif mode == "tuned":
            result = run_tuned_model(model_id, data, device, n_trials)
        else:
            raise ValueError(mode)
        results.append(result)
        pd.DataFrame(results).to_csv(RESULTS_ROOT / f"family06_{mode}_summary.csv", index=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline A V2 Family 6: State-Space / Selective SSM (24–27)")
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
