#!/usr/bin/env python3
"""
Pipeline A V2 — Family 3: TCN / ModernTCN / CNN+GRU (10–12)

Single reusable runner for:
    10 TCN
    11 ModernTCN
    12 CNN + GRU

Modes:
    --mode fixed   : run registered fixed baselines
    --mode tuned   : Optuna tuning, then train the selected configuration
    --mode stats   : evaluate existing fixed + tuned prediction tables

Frozen V2 contract:
    structured_24h_processed.npy      [74572, 24, 110]
    demographics_24h_processed.npy    [74572, 4]
    targets_24h/ outcomes_24h_primary_cohort.csv
    a5_alignment_index_24h.csv

Primary trained tasks:
    mortality_24h
    ventilation_onset_24h
    vasopressor_onset_24h
    readmission_30d_24h
    los_hours_24h

Deterioration is derived only at evaluation time and never enters training loss.
Ventilation / vasopressor validity masks are respected; -1 is invalid, not class 0.

LOS model-side transform:
    target coordinate = log1p(max(los_hours_24h - 24, 0))
    reporting coordinate = expm1(pred) + 24 hours
The frozen target artifact itself is never changed.

Shared benchmark seed:
    20260923

Architecture notes from the canonical V2 model table:
    10 TCN:
        channels=[64,64,64,64], kernel=3, dropout=0.2, lr=1e-3
        tuning: channels 32–128, kernel 3/5/7, levels 3–6
    11 ModernTCN:
        d_model=256, lr=1e-3
        tuning: d_model 128–384, expansion 1–4, dropout 0.1–0.3
        implementation uses a compact ModernTCN-style multi-scale depthwise
        temporal backbone adapted to the locked 24 x 110 V2 sequence.
    12 CNN + GRU:
        CNN channels=128, kernels=3+5, GRU hidden=256, dropout=0.2
        tuning: CNN channels 64–256, GRU hidden 128–512

Global V2 training contract applied here:
    batch size = 512
    max epochs = 50
    patience = 7
    AdamW
    weight decay = 1e-4
    gradient clip = 1.0
    scheduler = cosine (fixed) / cosine or OneCycle (tuned)
    BF16 autocast on CUDA/ROCm
    num_workers = 0
    validation selection = mean of four primary binary validation AUPRCs
    test split is never used for configuration selection
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

RESULTS_ROOT = PROJECT_ROOT / "outputs" / "baseline_family_03_tcn_10_12"

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

MODEL_NAMES = {
    "10": "TCN",
    "11": "ModernTCN",
    "12": "CNN_GRU",
}


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    name: str
    fixed: Dict[str, Any]
    search_space: Dict[str, Any]


MODEL_CONFIGS: Dict[str, ModelConfig] = {
    "10": ModelConfig(
        model_id="10",
        name="TCN",
        fixed={
            "channels": 64,
            "kernel_size": 3,
            "levels": 4,
            "dropout": 0.20,
            "lr": 1e-3,
        },
        search_space={
            "channels": "int[32,128]",
            "kernel_size": "categorical[3,5,7]",
            "levels": "int[3,6]",
        },
    ),
    "11": ModelConfig(
        model_id="11",
        name="ModernTCN",
        fixed={
            "d_model": 256,
            "expansion": 2,
            "dropout": 0.10,
            "lr": 1e-3,
        },
        search_space={
            "d_model": "int[128,384]",
            "expansion": "int[1,4]",
            "dropout": "float[0.10,0.30]",
        },
    ),
    "12": ModelConfig(
        model_id="12",
        name="CNN + GRU",
        fixed={
            "cnn_channels": 128,
            "kernels": [3, 5],
            "gru_hidden": 256,
            "dropout": 0.20,
            "lr": 1e-3,
        },
        search_space={
            "cnn_channels": "int[64,256]",
            "gru_hidden": "int[128,512]",
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
        raise ImportError("PyTorch is required for Family 3.") from _TORCH_IMPORT_ERROR


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


class ResidualTCNBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
        self.padding = padding

    def _chomp(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.padding] if self.padding > 0 else x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        z = self._chomp(self.conv1(x))
        z = F.gelu(self.norm1(z))
        z = self.dropout(z)
        z = self._chomp(self.conv2(z))
        z = F.gelu(self.norm2(z))
        z = self.dropout(z)
        return F.gelu(z + residual)


class TCNBaseline(nn.Module):
    def __init__(self, channels: int, kernel_size: int, levels: int, dropout: float) -> None:
        super().__init__()
        self.input_proj = nn.Conv1d(110, channels, kernel_size=1)
        self.blocks = nn.ModuleList(
            [
                ResidualTCNBlock(
                    channels=channels,
                    kernel_size=kernel_size,
                    dilation=2 ** i,
                    dropout=dropout,
                )
                for i in range(levels)
            ]
        )
        self.demo = nn.Sequential(
            nn.Linear(4, channels),
            nn.LayerNorm(channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fusion = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.LayerNorm(channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.heads = MultitaskHead(channels, dropout)

    def forward(self, x: torch.Tensor, demographics: torch.Tensor):
        # V2 contract is [B, 24, 110]; temporal axis becomes L in [B,C,L].
        z = x.transpose(1, 2)
        z = self.input_proj(z)
        for block in self.blocks:
            z = block(z)
        h_time = z[:, :, -1]
        h_mean = z.mean(dim=-1)
        h = 0.5 * (h_time + h_mean)
        d = self.demo(demographics)
        fused = self.fusion(torch.cat([h, d], dim=-1))
        return self.heads(fused)


class ModernTCNBlock(nn.Module):
    """Compact ModernTCN-style block: large/small depthwise temporal mixing + FFN."""

    def __init__(self, d_model: int, expansion: int, dropout: float, downsample: bool) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.large_dw = nn.Conv1d(d_model, d_model, kernel_size=13, padding=6, groups=d_model)
        self.small_dw = nn.Conv1d(d_model, d_model, kernel_size=5, padding=2, groups=d_model)
        self.mix = nn.Conv1d(d_model, d_model, kernel_size=1)
        ff_dim = d_model * expansion
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)
        self.downsample = (
            nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1)
            if downsample
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x [B,L,D]
        residual = x
        z = self.norm(x)
        zc = z.transpose(1, 2)
        mixed = self.large_dw(zc) + self.small_dw(zc)
        mixed = self.mix(mixed).transpose(1, 2)
        z = residual + self.dropout(mixed)
        z = z + self.ffn(self.norm(z))
        if self.downsample is not None:
            z = self.downsample(z.transpose(1, 2)).transpose(1, 2)
        return z


class ModernTCNBaseline(nn.Module):
    def __init__(self, d_model: int, expansion: int, dropout: float) -> None:
        super().__init__()
        self.input_proj = nn.Linear(110, d_model)
        self.pos = nn.Parameter(torch.zeros(1, 24, d_model))
        nn.init.normal_(self.pos, mean=0.0, std=0.02)
        # Canonical ModernTCN work documented a compact two-stage backbone;
        # stage depths and multi-scale kernels are kept fixed while d_model,
        # expansion and dropout are tuned according to the V2 table.
        self.blocks = nn.ModuleList(
            [
                ModernTCNBlock(d_model, expansion, dropout, downsample=False),
                ModernTCNBlock(d_model, expansion, dropout, downsample=True),
                ModernTCNBlock(d_model, expansion, dropout, downsample=False),
                ModernTCNBlock(d_model, expansion, dropout, downsample=False),
            ]
        )
        self.demo = nn.Sequential(
            nn.Linear(4, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fusion = nn.Sequential(
            nn.Linear(d_model + 64, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.heads = MultitaskHead(256, dropout)

    def forward(self, x: torch.Tensor, demographics: torch.Tensor):
        z = self.input_proj(x) + self.pos[:, : x.shape[1]]
        for block in self.blocks:
            z = block(z)
        h = z.mean(dim=1)
        d = self.demo(demographics)
        fused = self.fusion(torch.cat([h, d], dim=-1))
        return self.heads(fused)


class CNNGRUBaseline(nn.Module):
    def __init__(self, cnn_channels: int, gru_hidden: int, dropout: float) -> None:
        super().__init__()
        self.conv3 = nn.Conv1d(110, cnn_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(110, cnn_channels, kernel_size=5, padding=2)
        self.norm = nn.BatchNorm1d(cnn_channels * 2)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            input_size=cnn_channels * 2,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
        )
        self.demo = nn.Sequential(
            nn.Linear(4, gru_hidden),
            nn.LayerNorm(gru_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fusion = nn.Sequential(
            nn.Linear(gru_hidden * 2, gru_hidden),
            nn.LayerNorm(gru_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.heads = MultitaskHead(gru_hidden, dropout)

    def forward(self, x: torch.Tensor, demographics: torch.Tensor):
        z = x.transpose(1, 2)
        z3 = F.gelu(self.conv3(z))
        z5 = F.gelu(self.conv5(z))
        z = torch.cat([z3, z5], dim=1)
        z = self.dropout(F.gelu(self.norm(z)))
        z = z.transpose(1, 2)
        out, h = self.gru(z)
        h = h[-1]
        d = self.demo(demographics)
        fused = self.fusion(torch.cat([h, d], dim=-1))
        return self.heads(fused)


def build_model(model_id: str, params: Dict[str, Any]) -> nn.Module:
    if model_id == "10":
        return TCNBaseline(
            channels=int(params["channels"]),
            kernel_size=int(params["kernel_size"]),
            levels=int(params["levels"]),
            dropout=float(params["dropout"]),
        )
    if model_id == "11":
        return ModernTCNBaseline(
            d_model=int(params["d_model"]),
            expansion=int(params["expansion"]),
            dropout=float(params["dropout"]),
        )
    if model_id == "12":
        return CNNGRUBaseline(
            cnn_channels=int(params["cnn_channels"]),
            gru_hidden=int(params["gru_hidden"]),
            dropout=float(params["dropout"]),
        )
    raise ValueError(f"Unsupported Family 3 model id: {model_id}")


# ---------------------------------------------------------------------------
# Loss / predictions / metrics
# ---------------------------------------------------------------------------
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
    ym = data.targets["mortality_24h"][indices].astype(int, copy=False)
    yv = data.targets["ventilation_onset_24h"][indices].astype(int, copy=False)
    yp = data.targets["vasopressor_onset_24h"][indices].astype(int, copy=False)
    y_det = (1 - (1 - ym) * (1 - yv) * (1 - yp)).astype(int)
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
        raise RuntimeError(f"No valid checkpoint selected for Family 3 model {model_id}.")

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
    ym = data.targets["mortality_24h"][indices].astype(int, copy=False)
    yv = data.targets["ventilation_onset_24h"][indices].astype(int, copy=False)
    yp = data.targets["vasopressor_onset_24h"][indices].astype(int, copy=False)
    y_det = (1 - (1 - ym) * (1 - yv) * (1 - yp)).astype(int)
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
        "family": "03_TCN_Temporal_Hybrid",
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
        "los_training_transform": "log1p(max(los_hours_24h - 24.0, 0))",
        "los_reporting_transform": "expm1(predicted_excess_log1p) + 24.0 hours",
        "derived_deterioration": "Evaluation-only noisy-OR of mortality, ventilation, vasopressor",
        "architecture_notes": architecture_notes,
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

    if model_id == "10":
        arch_notes = {
            "type": "Residual dilated temporal convolutional network",
            "channels": params["channels"],
            "kernel": params["kernel_size"],
            "levels": params["levels"],
            "temporal_axis": "24-hour sequence",
            "input_permutation": "[B,24,110] -> [B,110,24]",
        }
    elif model_id == "11":
        arch_notes = {
            "type": "Compact ModernTCN-style depthwise multi-scale temporal backbone",
            "d_model": params["d_model"],
            "expansion": params["expansion"],
            "large_kernel": 13,
            "small_kernel": 5,
            "stage_downsample": "one stride-2 stage",
            "adaptation": "110-channel V2 sequence projected to d_model before temporal blocks",
        }
    elif model_id == "12":
        arch_notes = {
            "type": "Parallel Conv1D kernels 3/5 followed by 1-layer GRU",
            "cnn_channels": params["cnn_channels"],
            "kernels": params["kernels"],
            "gru_hidden": params["gru_hidden"],
        }
    else:
        arch_notes = {}

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
    if model_id == "10":
        model_params = {
            "channels": trial.suggest_int("channels", 32, 128),
            "kernel_size": trial.suggest_categorical("kernel_size", [3, 5, 7]),
            "levels": trial.suggest_int("levels", 3, 6),
            "dropout": trial.suggest_float("dropout", 0.10, 0.30),
            "lr": trial.suggest_float("lr", 3e-4, 2e-3, log=True),
        }
    elif model_id == "11":
        model_params = {
            "d_model": trial.suggest_int("d_model", 128, 384),
            "expansion": trial.suggest_int("expansion", 1, 4),
            "dropout": trial.suggest_float("dropout", 0.10, 0.30),
            "lr": trial.suggest_float("lr", 3e-4, 2e-3, log=True),
        }
    elif model_id == "12":
        model_params = {
            "cnn_channels": trial.suggest_int("cnn_channels", 64, 256),
            "gru_hidden": trial.suggest_int("gru_hidden", 128, 512),
            "dropout": trial.suggest_float("dropout", 0.10, 0.30),
            "lr": trial.suggest_float("lr", 3e-4, 2e-3, log=True),
        }
    else:
        raise ValueError(model_id)

    model_params.update(
        {
            "batch_size": trial.suggest_categorical("batch_size", [256, 512, 1024]),
            "max_epochs": 50,
            "patience": trial.suggest_int("patience", 5, 10),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
            "grad_clip": trial.suggest_float("grad_clip", 0.5, 2.0),
            "scheduler": trial.suggest_categorical("scheduler", ["cosine", "onecycle"]),
        }
    )
    return model_params


def run_tuned_model(model_id: str, data: FrozenData, device: torch.device, n_trials: int) -> Dict[str, Any]:
    require_optuna()
    cfg = MODEL_CONFIGS[model_id]
    tuned_root = RESULTS_ROOT / "tuned" / f"{model_id}_{cfg.name.replace(' ', '_')}"
    tuned_root.mkdir(parents=True, exist_ok=True)
    start = time.time()

    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name=f"family03_{model_id}")

    def objective(trial: "optuna.Trial") -> float:
        set_seed(SEED + trial.number)
        params = suggest_model_params(model_id, trial)
        try:
            model, score, best_epoch, _ = train_one(model_id, params, data, device, run_dir=None)
            trial.set_user_attr("best_epoch", int(best_epoch))
            trial.set_user_attr("params", params)
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return float(score)
        except Exception:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise

    study.optimize(objective, n_trials=n_trials, gc_after_trial=True, show_progress_bar=False)

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

    arch_notes = {
        "model_family": cfg.name,
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
                RESULTS_ROOT / "fixed" / f"{model_id}_{cfg.name.replace(' ', '_')}" / "predictions.csv",
            )
        )
        targets.append(
            (
                model_id,
                "tuned",
                RESULTS_ROOT / "tuned" / f"{model_id}_{cfg.name.replace(' ', '_')}" / "final_selected_configuration" / "predictions.csv",
            )
        )

    for model_id, mode, path in targets:
        print(f"[STATS] {mode} / {model_id}_{MODEL_NAMES[model_id]}", flush=True)
        stats_df = stats_for_prediction_table(path, model_id, mode)
        stats_df.to_csv(path.parent / "statistical_evaluation.csv", index=False)
        summaries.append(stats_df)

    final = pd.concat(summaries, ignore_index=True)
    final.to_csv(RESULTS_ROOT / "family03_statistical_evaluation.csv", index=False)
    print(f"[STATS] complete -> {RESULTS_ROOT / 'family03_statistical_evaluation.csv'}", flush=True)


# ---------------------------------------------------------------------------
# Runner / CLI
# ---------------------------------------------------------------------------
def run_all_models(model_ids: List[str], mode: str, n_trials: int) -> None:
    require_torch()
    set_seed(SEED)
    device = get_device()

    print("=" * 80)
    print("PIPELINE A V2 — FAMILY 3 TCN / TEMPORAL HYBRID 10–12")
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
        pd.DataFrame(results).to_csv(RESULTS_ROOT / f"family03_{mode}_summary.csv", index=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline A V2 Family 3: TCN, ModernTCN, CNN+GRU")
    parser.add_argument("--model", default="all", choices=["all", *MODEL_CONFIGS.keys()])
    parser.add_argument("--mode", required=True, choices=["fixed", "tuned", "stats"])
    parser.add_argument("--n-trials", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_trials < 1:
        raise ValueError("--n-trials must be >= 1")
    if args.mode == "stats":
        evaluate_stats_mode()
        return
    model_ids = list(MODEL_CONFIGS.keys()) if args.model == "all" else [args.model]
    run_all_models(model_ids, args.mode, args.n_trials)


if __name__ == "__main__":
    main()
