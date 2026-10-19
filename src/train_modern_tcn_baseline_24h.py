#!/usr/bin/env python3
# ============================================================
# PIPELINE A — ModernTCN BASELINE / 24H PREDICTION
# Same cohort, inputs, targets, splits, and evaluation protocol
# used by the completed Pipeline A baselines.
#
# ModernTCN backbone:
#   Based on the official ModernTCN classification implementation:
#     - patch/stem convolution
#     - hierarchical stages with large-kernel depthwise conv
#       + reparameterizable small-kernel branch
#     - variable-wise FFN and cross-variable FFN
#     - residual block structure
#
# FYP adaptation:
#   - canonical EHR tensor: [N, 24 hours, 44 features]
#   - ModernTCN input is [B, 44 variables, 24 time steps]
#   - demographics [B, 4] are fused after temporal encoding
#   - each task is trained as an independent single-task model
#     exactly like the existing deep-learning baselines
#   - RevIN is disabled because the canonical structured inputs are
#     already preprocessed/standardized and are not raw time-series
#     forecasting inputs
#   - classification uses BCEWithLogitsLoss + train-set pos_weight
#   - LOS uses MSELoss
#   - BF16 autocast is used on CUDA/ROCm when supported
#
# IMPORTANT:
#   This is a fixed baseline experiment (#11 in the locked 33-model
#   tracker). Hyperparameter tuning is NOT performed here.
# ============================================================

from __future__ import annotations

import gc
import json
import math
import random
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    recall_score,
    r2_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")

# ============================================================
# 1. REPRODUCIBILITY / CONFIG
# ============================================================

SEED = 2026

BASE_DIR = Path("/home/abdullah2025/fyp_project/outputs/pipeline_a")
INPUT_DIR = BASE_DIR / "model_inputs_24h"
TARGET_DIR = BASE_DIR / "targets_24h" / "component_targets"
OUTPUT_DIR = BASE_DIR / "modern_tcn_baseline_24h"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STRUCTURED_PATH = INPUT_DIR / "structured_24h_processed.npy"
DEMOGRAPHICS_PATH = INPUT_DIR / "demographics_24h_processed.npy"
INDEX_PATH = INPUT_DIR / "structured_24h_index.csv"
TARGET_PATH = TARGET_DIR / "five_task_component_targets_24h.npz"

SEQ_LEN = 24
INPUT_DIM = 44
DEMOG_DIM = 4

# ------------------------------------------------------------
# ModernTCN fixed baseline configuration
# ------------------------------------------------------------
PATCH_SIZE = 3
PATCH_STRIDE = 1
DOWNSAMPLE_RATIO = 2
FFN_RATIO = 2
NUM_BLOCKS = [2, 2]
LARGE_KERNELS = [13, 9]
SMALL_KERNELS = [5, 5]
DIMS = [64, 128]
BACKBONE_DROPOUT = 0.10
HEAD_DROPOUT = 0.10
CLASS_DROPOUT = 0.10
DROPOUT = 0.10
SMALL_KERNEL_MERGED = False
USE_MULTI_SCALE = False
USE_REVIN = False

# EHR/demographic adaptation head.
TEMPORAL_HEAD_HIDDEN = 256
DEMOG_HIDDEN = 64
FUSION_HIDDEN = 256
FUSION_OUT = 128

# Optimization protocol shared with completed deep-learning baselines.
BATCH_SIZE = 512
MAX_EPOCHS = 30
PATIENCE = 5
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
NUM_WORKERS = 0

CLASSIFICATION_TASKS = {
    "mortality_24h": "mortality_24h",
    "readmission_30d_24h": "readmission_30d_24h",
    "deterioration_24h": "deterioration_24h",
}
REGRESSION_TASK = "los_hours_24h"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def bf16_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        return bool(torch.cuda.is_bf16_supported())
    except Exception:
        return True


USE_BF16 = bf16_available()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)

if torch.cuda.is_available():
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

print("=" * 78)
print("MODERNTCN BASELINE — PIPELINE A / 24H")
print("=" * 78)
print(f"Device             : {DEVICE}")
if torch.cuda.is_available():
    try:
        print(f"GPU                : {torch.cuda.get_device_name(0)}")
    except Exception:
        print("GPU                : CUDA/ROCm device")
print(f"PyTorch             : {torch.__version__}")
print(f"BF16 autocast       : {USE_BF16}")
print(f"Output directory    : {OUTPUT_DIR}")
print()
print("ModernTCN fixed config:")
print(f"  patch_size        : {PATCH_SIZE}")
print(f"  patch_stride      : {PATCH_STRIDE}")
print(f"  downsample_ratio  : {DOWNSAMPLE_RATIO}")
print(f"  ffn_ratio         : {FFN_RATIO}")
print(f"  num_blocks        : {NUM_BLOCKS}")
print(f"  large_kernels     : {LARGE_KERNELS}")
print(f"  small_kernels     : {SMALL_KERNELS}")
print(f"  dims              : {DIMS}")
print(f"  backbone dropout  : {BACKBONE_DROPOUT}")
print(f"  head dropout      : {HEAD_DROPOUT}")
print(f"  class dropout     : {CLASS_DROPOUT}")
print(f"  RevIN             : {USE_REVIN}")
print(f"  batch size        : {BATCH_SIZE}")
print()

# ============================================================
# 2. LOAD + VALIDATE CANONICAL INPUTS
# ============================================================

print("Loading canonical inputs...")

X = np.load(STRUCTURED_PATH)
D = np.load(DEMOGRAPHICS_PATH)
index_df = pd.read_csv(INDEX_PATH)
targets = np.load(TARGET_PATH)

print(f"Structured shape    : {X.shape}")
print(f"Demographics shape  : {D.shape}")
print(f"Index shape         : {index_df.shape}")
print(f"Target stay IDs     : {targets['stay_id'].shape}")

if X.ndim != 3 or X.shape[1:] != (SEQ_LEN, INPUT_DIM):
    raise ValueError(
        f"Expected structured shape (N,{SEQ_LEN},{INPUT_DIM}), got {X.shape}"
    )

if D.ndim != 2 or D.shape[1] != DEMOG_DIM:
    raise ValueError(
        f"Expected demographics shape (N,{DEMOG_DIM}), got {D.shape}"
    )

if len(index_df) != len(X) or len(D) != len(X):
    raise ValueError(
        "Structured, demographics, and index lengths do not match."
    )

required_index_cols = {"stay_id", "split_24h"}
missing_index_cols = sorted(required_index_cols - set(index_df.columns))
if missing_index_cols:
    raise ValueError(
        "Canonical index is missing required columns: "
        + ", ".join(missing_index_cols)
    )

index_stay_ids = pd.to_numeric(index_df["stay_id"], errors="coerce").to_numpy()
target_stay_ids = np.asarray(targets["stay_id"])

if np.isnan(index_stay_ids).any():
    raise ValueError("Index contains invalid stay_id values.")

if not np.array_equal(
    index_stay_ids.astype(np.int64),
    target_stay_ids.astype(np.int64),
):
    raise ValueError(
        "Canonical stay_id alignment FAILED between index and targets."
    )

print("✅ Canonical stay_id alignment passed.")

X = np.asarray(X, dtype=np.float32)
D = np.asarray(D, dtype=np.float32)

if not np.isfinite(X).all():
    raise ValueError("Structured input contains NaN/Inf values.")
if not np.isfinite(D).all():
    raise ValueError("Demographics input contains NaN/Inf values.")

# Explicit writable copies.
X = np.array(X, dtype=np.float32, copy=True)
D = np.array(D, dtype=np.float32, copy=True)
print("✅ Inputs are finite and writable.")

# ============================================================
# 3. CANONICAL SPLITS
# ============================================================

split_values = index_df["split_24h"].astype(str).str.lower().to_numpy()
train_idx = np.flatnonzero(split_values == "train")
val_idx = np.flatnonzero(split_values == "val")
test_idx = np.flatnonzero(split_values == "test")

if min(len(train_idx), len(val_idx), len(test_idx)) == 0:
    raise ValueError("One or more canonical splits are empty.")

print(
    f"Splits              : train={len(train_idx):,} "
    f"val={len(val_idx):,} test={len(test_idx):,}"
)

expected_splits = {"train": 52268, "val": 7417, "test": 14887}
actual_splits = {
    "train": len(train_idx),
    "val": len(val_idx),
    "test": len(test_idx),
}
if actual_splits != expected_splits:
    raise ValueError(
        f"Canonical split counts changed. Expected {expected_splits}, "
        f"got {actual_splits}."
    )

print("✅ Canonical split counts verified.")

# ============================================================
# 4. DATASET
# ============================================================


class ArrayDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        d: np.ndarray,
        y: np.ndarray,
        indices: np.ndarray,
    ) -> None:
        self.x = torch.from_numpy(np.array(x[indices], dtype=np.float32, copy=True))
        self.d = torch.from_numpy(np.array(d[indices], dtype=np.float32, copy=True))
        self.y = torch.from_numpy(np.array(y[indices], dtype=np.float32, copy=True))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int):
        return self.x[i], self.d[i], self.y[i]


def make_loader(
    x: np.ndarray,
    d: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    shuffle: bool,
) -> DataLoader:
    dataset = ArrayDataset(x, d, y, indices)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
        drop_last=False,
    )

# ============================================================
# 5. MODERNTCN BACKBONE
# ============================================================


class ReparamLargeKernelConv(nn.Module):
    """Official ModernTCN-style large-kernel + small-kernel branch."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        groups: int,
        small_kernel: int | None,
        small_kernel_merged: bool = False,
    ) -> None:
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError("ModernTCN large kernels must be odd for same-padding.")
        if small_kernel is not None and small_kernel > kernel_size:
            raise ValueError("small_kernel cannot exceed kernel_size.")
        if small_kernel is not None and small_kernel % 2 == 0:
            raise ValueError("ModernTCN small kernels must be odd for same-padding.")

        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        padding = kernel_size // 2

        if small_kernel_merged:
            self.lkb_reparam = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=True,
            )
        else:
            self.lkb_origin = nn.Sequential(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    groups=groups,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels),
            )
            self.small_conv = None
            if small_kernel is not None:
                self.small_conv = nn.Sequential(
                    nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=small_kernel,
                        stride=stride,
                        padding=small_kernel // 2,
                        groups=groups,
                        bias=False,
                    ),
                    nn.BatchNorm1d(out_channels),
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "lkb_reparam"):
            out = self.lkb_reparam(x)
        else:
            out = self.lkb_origin(x)
            if self.small_conv is not None:
                out = out + self.small_conv(x)
        return out

    @staticmethod
    def _pad_small_kernel(
        kernel: torch.Tensor,
        left: int,
        right: int,
    ) -> torch.Tensor:
        if left == 0 and right == 0:
            return kernel
        return F.pad(kernel, (left, right))

    @staticmethod
    def _fuse_conv_bn(
        conv: nn.Conv1d,
        bn: nn.BatchNorm1d,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        weight = conv.weight
        mean = bn.running_mean
        var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = torch.sqrt(var + eps)
        scale = (gamma / std).reshape(-1, 1, 1)
        fused_weight = weight * scale
        fused_bias = beta - mean * gamma / std
        return fused_weight, fused_bias

    def get_equivalent_kernel_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if hasattr(self, "lkb_reparam"):
            return self.lkb_reparam.weight, self.lkb_reparam.bias

        large_conv = self.lkb_origin[0]
        large_bn = self.lkb_origin[1]
        eq_k, eq_b = self._fuse_conv_bn(large_conv, large_bn)

        if self.small_conv is not None:
            small_conv = self.small_conv[0]
            small_bn = self.small_conv[1]
            small_k, small_b = self._fuse_conv_bn(small_conv, small_bn)
            pad = (self.kernel_size - self.small_kernel) // 2
            eq_k = eq_k + self._pad_small_kernel(small_k, pad, pad)
            eq_b = eq_b + small_b

        return eq_k, eq_b

    def merge_kernel(self) -> None:
        if hasattr(self, "lkb_reparam"):
            return

        eq_k, eq_b = self.get_equivalent_kernel_bias()
        large_conv = self.lkb_origin[0]

        reparam = nn.Conv1d(
            in_channels=large_conv.in_channels,
            out_channels=large_conv.out_channels,
            kernel_size=large_conv.kernel_size,
            stride=large_conv.stride,
            padding=large_conv.padding,
            dilation=large_conv.dilation,
            groups=large_conv.groups,
            bias=True,
        )
        reparam.weight.data.copy_(eq_k)
        reparam.bias.data.copy_(eq_b)
        self.lkb_reparam = reparam
        del self.lkb_origin
        if self.small_conv is not None:
            del self.small_conv
            self.small_conv = None


class ModernTCNBlock(nn.Module):
    """ModernTCN block adapted directly from the official classification backbone."""

    def __init__(
        self,
        large_size: int,
        small_size: int,
        dmodel: int,
        dff: int,
        nvars: int,
        drop: float,
        small_kernel_merged: bool = False,
    ) -> None:
        super().__init__()

        # Large-kernel temporal mixing is depthwise across variable-channel pairs.
        self.dw = ReparamLargeKernelConv(
            in_channels=nvars * dmodel,
            out_channels=nvars * dmodel,
            kernel_size=large_size,
            stride=1,
            groups=nvars * dmodel,
            small_kernel=small_size,
            small_kernel_merged=small_kernel_merged,
        )

        self.norm = nn.BatchNorm1d(dmodel)

        # Variable-wise FFN: each variable's d_model channels are processed independently.
        self.ffn1pw1 = nn.Conv1d(
            nvars * dmodel,
            nvars * dff,
            kernel_size=1,
            groups=nvars,
        )
        self.ffn1pw2 = nn.Conv1d(
            nvars * dff,
            nvars * dmodel,
            kernel_size=1,
            groups=nvars,
        )
        self.ffn1act = nn.GELU()
        self.ffn1drop1 = nn.Dropout(drop)
        self.ffn1drop2 = nn.Dropout(drop)

        # Cross-variable FFN: each channel mixes information across variables.
        self.ffn2pw1 = nn.Conv1d(
            nvars * dmodel,
            nvars * dff,
            kernel_size=1,
            groups=dmodel,
        )
        self.ffn2pw2 = nn.Conv1d(
            nvars * dff,
            nvars * dmodel,
            kernel_size=1,
            groups=dmodel,
        )
        self.ffn2act = nn.GELU()
        self.ffn2drop1 = nn.Dropout(drop)
        self.ffn2drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, M, D, N]
        residual = x
        B, M, D, N = x.shape

        x = x.reshape(B, M * D, N)
        x = self.dw(x)
        x = x.reshape(B, M, D, N)

        # Per-variable channel normalization.
        x = x.reshape(B * M, D, N)
        x = self.norm(x)
        x = x.reshape(B, M, D, N)

        # Variable-wise FFN.
        x = x.reshape(B, M * D, N)
        x = self.ffn1drop1(self.ffn1pw1(x))
        x = self.ffn1act(x)
        x = self.ffn1drop2(self.ffn1pw2(x))
        x = x.reshape(B, M, D, N)

        # Cross-variable FFN.
        x = x.permute(0, 2, 1, 3).reshape(B, D * M, N)
        x = self.ffn2drop1(self.ffn2pw1(x))
        x = self.ffn2act(x)
        x = self.ffn2drop2(self.ffn2pw2(x))
        x = x.reshape(B, D, M, N).permute(0, 2, 1, 3)

        return residual + x


class ModernTCNStage(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        large_size: int,
        small_size: int,
        dmodel: int,
        nvars: int,
        ffn_ratio: int,
        drop: float,
        small_kernel_merged: bool,
    ) -> None:
        super().__init__()
        dff = dmodel * ffn_ratio
        self.blocks = nn.ModuleList(
            [
                ModernTCNBlock(
                    large_size=large_size,
                    small_size=small_size,
                    dmodel=dmodel,
                    dff=dff,
                    nvars=nvars,
                    drop=drop,
                    small_kernel_merged=small_kernel_merged,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class ModernTCNBackbone(nn.Module):
    """ModernTCN backbone for [B, variables, time]."""

    def __init__(
        self,
        nvars: int,
        seq_len: int,
    ) -> None:
        super().__init__()

        if not (
            len(NUM_BLOCKS)
            == len(LARGE_KERNELS)
            == len(SMALL_KERNELS)
            == len(DIMS)
        ):
            raise ValueError("ModernTCN stage config lengths do not match.")

        self.nvars = nvars
        self.seq_len = seq_len
        self.patch_size = PATCH_SIZE
        self.patch_stride = PATCH_STRIDE
        self.downsample_ratio = DOWNSAMPLE_RATIO
        self.num_stage = len(NUM_BLOCKS)

        self.downsample_layers = nn.ModuleList()
        self.stages = nn.ModuleList()

        # Official ModernTCN-style stem.
        self.downsample_layers.append(
            nn.Sequential(
                nn.Conv1d(
                    1,
                    DIMS[0],
                    kernel_size=PATCH_SIZE,
                    stride=PATCH_STRIDE,
                ),
                nn.BatchNorm1d(DIMS[0]),
            )
        )

        # Hierarchical downsampling between stages.
        for i in range(self.num_stage - 1):
            self.downsample_layers.append(
                nn.Sequential(
                    nn.BatchNorm1d(DIMS[i]),
                    nn.Conv1d(
                        DIMS[i],
                        DIMS[i + 1],
                        kernel_size=DOWNSAMPLE_RATIO,
                        stride=DOWNSAMPLE_RATIO,
                    ),
                )
            )

        for i in range(self.num_stage):
            self.stages.append(
                ModernTCNStage(
                    num_blocks=NUM_BLOCKS[i],
                    large_size=LARGE_KERNELS[i],
                    small_size=SMALL_KERNELS[i],
                    dmodel=DIMS[i],
                    nvars=nvars,
                    ffn_ratio=FFN_RATIO,
                    drop=BACKBONE_DROPOUT,
                    small_kernel_merged=SMALL_KERNEL_MERGED,
                )
            )

    @staticmethod
    def _repeat_last_pad(x: torch.Tensor, pad_len: int) -> torch.Tensor:
        if pad_len <= 0:
            return x
        return torch.cat([x, x[:, :, -1:].repeat(1, 1, pad_len)], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: [B, M, L]
        B, M, L = x.shape
        if M != self.nvars:
            raise ValueError(
                f"Expected {self.nvars} variables, got {M}."
            )

        x = x.unsqueeze(2)  # [B, M, 1, L]

        for i in range(self.num_stage):
            B, M, D, N = x.shape
            x = x.reshape(B * M, D, N)

            if i == 0:
                # Match official ModernTCN stem padding behavior.
                if self.patch_size != self.patch_stride:
                    pad_len = self.patch_size - self.patch_stride
                    x = self._repeat_last_pad(x, pad_len)
            else:
                if N % self.downsample_ratio != 0:
                    pad_len = self.downsample_ratio - (N % self.downsample_ratio)
                    x = self._repeat_last_pad(x, pad_len)

            x = self.downsample_layers[i](x)
            _, D_new, N_new = x.shape
            x = x.reshape(B, M, D_new, N_new)
            x = self.stages[i](x)

        return x


class ModernTCNMultiTaskModel(nn.Module):
    """ModernTCN temporal backbone + demographics fusion for one task."""

    def __init__(self, regression: bool) -> None:
        super().__init__()
        self.regression = regression

        self.backbone = ModernTCNBackbone(
            nvars=INPUT_DIM,
            seq_len=SEQ_LEN,
        )

        final_dim = DIMS[-1]
        self.class_dropout = nn.Dropout(CLASS_DROPOUT)

        # Mean-pool final ModernTCN representations over time, retaining all variables.
        # This keeps the head tractable while preserving variable-specific features.
        pooled_dim = INPUT_DIM * final_dim
        self.temporal_projection = nn.Sequential(
            nn.Linear(pooled_dim, TEMPORAL_HEAD_HIDDEN),
            nn.GELU(),
            nn.Dropout(HEAD_DROPOUT),
        )

        self.demographics = nn.Sequential(
            nn.Linear(DEMOG_DIM, DEMOG_HIDDEN),
            nn.GELU(),
            nn.Dropout(DROPOUT),
        )

        self.fusion = nn.Sequential(
            nn.Linear(TEMPORAL_HEAD_HIDDEN + DEMOG_HIDDEN, FUSION_HIDDEN),
            nn.GELU(),
            nn.Dropout(HEAD_DROPOUT),
            nn.Linear(FUSION_HIDDEN, FUSION_OUT),
            nn.GELU(),
        )
        self.output = nn.Linear(FUSION_OUT, 1)

    def forward(self, x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        # Canonical x arrives as [B, 24, 44]; ModernTCN uses [B, 44, 24].
        x = x.transpose(1, 2)
        z = self.backbone(x)  # [B, M, D, N]
        z = F.gelu(z)
        z = self.class_dropout(z)
        z = z.mean(dim=-1)  # [B, M, D]
        z = z.reshape(z.shape[0], -1)
        z = self.temporal_projection(z)

        demog = self.demographics(d)
        fused = self.fusion(torch.cat([z, demog], dim=1))
        out = self.output(fused).squeeze(1)
        return out


# ============================================================
# 6. METRICS
# ============================================================


def safe_roc_auc(y_true: np.ndarray, probs: np.ndarray) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, probs))
    except Exception:
        return float("nan")


def safe_auprc(y_true: np.ndarray, probs: np.ndarray) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(average_precision_score(y_true, probs))
    except Exception:
        return float("nan")


def binary_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    y_true = y_true.astype(int)
    preds = (probs >= threshold).astype(int)

    tp = int(((y_true == 1) & (preds == 1)).sum())
    tn = int(((y_true == 0) & (preds == 0)).sum())
    fp = int(((y_true == 0) & (preds == 1)).sum())
    fn = int(((y_true == 1) & (preds == 0)).sum())

    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        "auroc": safe_roc_auc(y_true, probs),
        "auprc": safe_auprc(y_true, probs),
        "brier": float(brier_score_loss(y_true, probs)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(
            (2 * tp) / (2 * tp + fp + fn)
            if (2 * tp + fp + fn)
            else 0.0
        ),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
        "prevalence": float(y_true.mean()),
    }


def regression_metrics(
    y_true: np.ndarray,
    pred: np.ndarray,
) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, pred))),
        "r2": float(r2_score(y_true, pred)) if len(y_true) >= 2 else float("nan"),
        "median_absolute_error": float(median_absolute_error(y_true, pred)),
    }

# ============================================================
# 7. EVALUATION
# ============================================================


def autocast_context():
    if DEVICE.type == "cuda" and USE_BF16:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type=DEVICE.type, enabled=False)


def evaluate_classification(
    model: nn.Module,
    loader: DataLoader,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    all_probs: List[np.ndarray] = []
    all_y: List[np.ndarray] = []

    with torch.no_grad():
        for xb, db, yb in loader:
            xb = xb.to(DEVICE, non_blocking=True)
            db = db.to(DEVICE, non_blocking=True)

            with autocast_context():
                logits = model(xb, db)

            probs = torch.sigmoid(logits)
            all_probs.append(probs.float().cpu().numpy())
            all_y.append(yb.numpy())

    probs = np.concatenate(all_probs)
    y_true = np.concatenate(all_y).astype(np.int64)
    return binary_metrics(y_true, probs), y_true, probs


def evaluate_regression(
    model: nn.Module,
    loader: DataLoader,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    all_pred: List[np.ndarray] = []
    all_y: List[np.ndarray] = []

    with torch.no_grad():
        for xb, db, yb in loader:
            xb = xb.to(DEVICE, non_blocking=True)
            db = db.to(DEVICE, non_blocking=True)

            with autocast_context():
                pred = model(xb, db)

            all_pred.append(pred.float().cpu().numpy())
            all_y.append(yb.numpy())

    pred = np.concatenate(all_pred)
    y_true = np.concatenate(all_y)
    return regression_metrics(y_true, pred), y_true, pred

# ============================================================
# 8. TARGET PREPARATION
# ============================================================


def get_target_and_valid(task: str) -> Tuple[np.ndarray, np.ndarray]:
    if task not in targets:
        raise KeyError(f"Target '{task}' not found in {TARGET_PATH}")

    valid_key = f"{task}_valid"

    value = np.asarray(targets[task])
    if valid_key in targets:
        valid = np.asarray(targets[valid_key])
    else:
        valid = np.ones(len(value), dtype=bool)

    if len(value) != len(X) or len(valid) != len(X):
        raise ValueError(
            f"Target '{task}' length mismatch: target={len(value)}, valid={len(valid)}, X={len(X)}"
        )

    return value, valid

# ============================================================
# 9. TRAIN ONE TASK
# ============================================================


def train_task(
    task_name: str,
    y_full: np.ndarray,
    valid_full: np.ndarray,
    is_regression: bool,
) -> Dict[str, Any]:
    valid = np.asarray(valid_full).astype(bool)
    finite_target = np.isfinite(y_full)
    usable = valid & finite_target

    task_train_idx = train_idx[usable[train_idx]]
    task_val_idx = val_idx[usable[val_idx]]
    task_test_idx = test_idx[usable[test_idx]]

    if (
        len(task_train_idx) == 0
        or len(task_val_idx) == 0
        or len(task_test_idx) == 0
    ):
        raise ValueError(f"{task_name}: one split has zero valid samples.")

    y = np.asarray(y_full, dtype=np.float32)

    print("\n" + "=" * 78)
    print(f"TRAINING: {task_name}")
    print("=" * 78)
    print(
        f"Valid samples      : train={len(task_train_idx):,} "
        f"val={len(task_val_idx):,} test={len(task_test_idx):,}"
    )

    if not is_regression:
        train_rate = float(y[task_train_idx].mean())
        val_rate = float(y[task_val_idx].mean())
        test_rate = float(y[task_test_idx].mean())
        print(
            f"Positive rate      : train={train_rate:.4f} "
            f"val={val_rate:.4f} test={test_rate:.4f}"
        )

    train_loader = make_loader(X, D, y, task_train_idx, shuffle=True)
    val_loader = make_loader(X, D, y, task_val_idx, shuffle=False)
    test_loader = make_loader(X, D, y, task_test_idx, shuffle=False)

    model = ModernTCNMultiTaskModel(regression=is_regression).to(DEVICE)

    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"Trainable params    : {trainable_params:,}")

    # Verify forward pass before training.
    model.eval()
    with torch.no_grad():
        xb0, db0, _ = next(iter(train_loader))
        xb0 = xb0[:2].to(DEVICE)
        db0 = db0[:2].to(DEVICE)
        with autocast_context():
            smoke_out = model(xb0, db0)
    if smoke_out.shape != (2,):
        raise RuntimeError(
            f"Unexpected ModernTCN forward output shape: {tuple(smoke_out.shape)}"
        )
    if not torch.isfinite(smoke_out.float()).all():
        raise RuntimeError("ModernTCN smoke output contains NaN/Inf.")
    print("✅ ModernTCN forward smoke test passed.")
    del xb0, db0, smoke_out
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    if is_regression:
        criterion = nn.MSELoss()
        print("Loss                : MSELoss")
    else:
        positives = float((y[task_train_idx] == 1).sum())
        negatives = float((y[task_train_idx] == 0).sum())
        if positives <= 0:
            raise ValueError(f"{task_name}: no positive samples in training split.")

        pos_weight_value = negatives / positives
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(
                pos_weight_value,
                device=DEVICE,
                dtype=torch.float32,
            )
        )
        print(f"BCE pos_weight      : {pos_weight_value:.6f}")

    checkpoint_path = OUTPUT_DIR / f"best_{task_name}.pt"
    history_path = OUTPUT_DIR / f"{task_name}_training_history.csv"
    pred_path = OUTPUT_DIR / f"{task_name}_test_predictions.csv"

    best_epoch = 1
    best_monitor = float("inf") if is_regression else -float("inf")
    patience_counter = 0
    history: List[Dict[str, Any]] = []
    start_time = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        batch_losses: List[float] = []
        epoch_start = time.time()

        for xb, db, yb in train_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            db = db.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast_context():
                pred = model(xb, db)
                loss = criterion(pred, yb)

            loss.backward()
            if GRAD_CLIP is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            loss_value = float(loss.detach().float().cpu())
            if not math.isfinite(loss_value):
                raise RuntimeError(
                    f"{task_name}: non-finite training loss at epoch {epoch}."
                )
            batch_losses.append(loss_value)

        train_loss = float(np.mean(batch_losses))

        if is_regression:
            val_metrics, _, _ = evaluate_regression(model, val_loader)
            monitor_value = val_metrics["mae"]
            current_lr = optimizer.param_groups[0]["lr"]
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_mae": val_metrics["mae"],
                    "val_rmse": val_metrics["rmse"],
                    "val_r2": val_metrics["r2"],
                    "val_median_absolute_error": val_metrics[
                        "median_absolute_error"
                    ],
                    "learning_rate": current_lr,
                    "epoch_seconds": time.time() - epoch_start,
                }
            )
            print(
                f"Epoch {epoch:02d}: train_loss={train_loss:.5f} "
                f"val_MAE={val_metrics['mae']:.5f} "
                f"val_RMSE={val_metrics['rmse']:.5f} "
                f"val_R2={val_metrics['r2']:.5f} "
                f"time={time.time()-epoch_start:.1f}s"
            )
            improved = np.isfinite(monitor_value) and monitor_value < best_monitor
        else:
            val_metrics, _, _ = evaluate_classification(model, val_loader)
            monitor_value = val_metrics["auroc"]
            current_lr = optimizer.param_groups[0]["lr"]
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_auroc": val_metrics["auroc"],
                    "val_auprc": val_metrics["auprc"],
                    "val_brier": val_metrics["brier"],
                    "val_precision": val_metrics["precision"],
                    "val_recall": val_metrics["recall"],
                    "val_f1": val_metrics["f1"],
                    "learning_rate": current_lr,
                    "epoch_seconds": time.time() - epoch_start,
                }
            )
            print(
                f"Epoch {epoch:02d}: train_loss={train_loss:.5f} "
                f"val_AUROC={val_metrics['auroc']:.5f} "
                f"val_AUPRC={val_metrics['auprc']:.5f} "
                f"val_Brier={val_metrics['brier']:.5f} "
                f"time={time.time()-epoch_start:.1f}s"
            )
            improved = np.isfinite(monitor_value) and monitor_value > best_monitor

        if improved:
            best_monitor = float(monitor_value)
            best_epoch = epoch
            patience_counter = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "task_name": task_name,
                    "best_epoch": best_epoch,
                    "best_monitor": best_monitor,
                    "trainable_params": trainable_params,
                    "config": {
                        "seq_len": SEQ_LEN,
                        "input_dim": INPUT_DIM,
                        "demog_dim": DEMOG_DIM,
                        "patch_size": PATCH_SIZE,
                        "patch_stride": PATCH_STRIDE,
                        "downsample_ratio": DOWNSAMPLE_RATIO,
                        "ffn_ratio": FFN_RATIO,
                        "num_blocks": NUM_BLOCKS,
                        "large_kernels": LARGE_KERNELS,
                        "small_kernels": SMALL_KERNELS,
                        "dims": DIMS,
                        "backbone_dropout": BACKBONE_DROPOUT,
                        "head_dropout": HEAD_DROPOUT,
                        "class_dropout": CLASS_DROPOUT,
                        "small_kernel_merged": SMALL_KERNEL_MERGED,
                        "use_multi_scale": USE_MULTI_SCALE,
                        "use_revin": USE_REVIN,
                        "temporal_head_hidden": TEMPORAL_HEAD_HIDDEN,
                        "demog_hidden": DEMOG_HIDDEN,
                        "fusion_hidden": FUSION_HIDDEN,
                        "fusion_out": FUSION_OUT,
                    },
                },
                checkpoint_path,
            )
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch}.")
            break

        del val_metrics
        gc.collect()

    pd.DataFrame(history).to_csv(history_path, index=False)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    if is_regression:
        val_metrics, val_y, val_pred = evaluate_regression(model, val_loader)
        test_metrics, test_y, test_pred = evaluate_regression(model, test_loader)

        prediction_df = pd.DataFrame(
            {
                "y_true": test_y,
                "prediction": test_pred,
            }
        )
        prediction_df.to_csv(pred_path, index=False)

        result = {
            "best_epoch": int(best_epoch),
            "best_monitor": float(best_monitor),
            "val": val_metrics,
            "test": test_metrics,
            "training_minutes": (time.time() - start_time) / 60.0,
            "n_train": int(len(task_train_idx)),
            "n_val": int(len(task_val_idx)),
            "n_test": int(len(task_test_idx)),
            "trainable_params": int(trainable_params),
        }

    else:
        val_metrics, val_y, val_probs = evaluate_classification(model, val_loader)
        test_metrics, test_y, test_probs = evaluate_classification(model, test_loader)

        prediction_df = pd.DataFrame(
            {
                "y_true": test_y,
                "probability": test_probs,
                "prediction_0p5": (test_probs >= 0.5).astype(np.int8),
            }
        )
        prediction_df.to_csv(pred_path, index=False)

        result = {
            "best_epoch": int(best_epoch),
            "best_monitor": float(best_monitor),
            "val": val_metrics,
            "test": test_metrics,
            "training_minutes": (time.time() - start_time) / 60.0,
            "n_train": int(len(task_train_idx)),
            "n_val": int(len(task_val_idx)),
            "n_test": int(len(task_test_idx)),
            "trainable_params": int(trainable_params),
        }

    print("\nValidation metrics:")
    for k, v in result["val"].items():
        print(f"  {k}: {v}")

    print("\nTEST metrics:")
    for k, v in result["test"].items():
        print(f"  {k}: {v}")

    print(f"\nBest epoch        : {best_epoch}")
    print(f"Training time     : {result['training_minutes']:.2f} minutes")

    del model, train_loader, val_loader, test_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result

# ============================================================
# 10. TRAIN ALL FOUR TASKS
# ============================================================

all_results: Dict[str, Any] = {}

for task_name in CLASSIFICATION_TASKS.values():
    y, valid = get_target_and_valid(task_name)
    all_results[task_name] = train_task(
        task_name=task_name,
        y_full=y,
        valid_full=valid,
        is_regression=False,
    )

los_y, los_valid = get_target_and_valid(REGRESSION_TASK)
all_results[REGRESSION_TASK] = train_task(
    task_name=REGRESSION_TASK,
    y_full=los_y,
    valid_full=los_valid,
    is_regression=True,
)

# ============================================================
# 11. FINAL SUMMARY TABLE
# ============================================================

summary_rows: List[Dict[str, Any]] = []

for task_name, result in all_results.items():
    if task_name in CLASSIFICATION_TASKS.values():
        val = result["val"]
        test = result["test"]
        summary_rows.append(
            {
                "target": task_name,
                "best_epoch": result["best_epoch"],
                "val_auroc": val["auroc"],
                "val_auprc": val["auprc"],
                "val_brier": val["brier"],
                "test_auroc": test["auroc"],
                "test_auprc": test["auprc"],
                "test_brier": test["brier"],
                "test_precision": test["precision"],
                "test_recall": test["recall"],
                "test_f1": test["f1"],
                "test_sensitivity": test["sensitivity"],
                "test_specificity": test["specificity"],
                "test_prevalence": test["prevalence"],
                "val_mae": np.nan,
                "val_rmse": np.nan,
                "val_r2": np.nan,
                "test_mae": np.nan,
                "test_rmse": np.nan,
                "test_r2": np.nan,
                "test_median_absolute_error": np.nan,
            }
        )
    else:
        val = result["val"]
        test = result["test"]
        summary_rows.append(
            {
                "target": task_name,
                "best_epoch": result["best_epoch"],
                "val_auroc": np.nan,
                "val_auprc": np.nan,
                "val_brier": np.nan,
                "test_auroc": np.nan,
                "test_auprc": np.nan,
                "test_brier": np.nan,
                "test_precision": np.nan,
                "test_recall": np.nan,
                "test_f1": np.nan,
                "test_sensitivity": np.nan,
                "test_specificity": np.nan,
                "test_prevalence": np.nan,
                "val_mae": val["mae"],
                "val_rmse": val["rmse"],
                "val_r2": val["r2"],
                "test_mae": test["mae"],
                "test_rmse": test["rmse"],
                "test_r2": test["r2"],
                "test_median_absolute_error": test["median_absolute_error"],
            }
        )

summary_df = pd.DataFrame(summary_rows)
summary_path = OUTPUT_DIR / "metrics.csv"
summary_df.to_csv(summary_path, index=False)

# ============================================================
# 12. METADATA
# ============================================================

# Derive final-stage sequence length exactly from the backbone math.
def stage_lengths(seq_len: int) -> List[int]:
    length = seq_len
    out: List[int] = []
    # Stem: explicit right padding of patch_size - patch_stride.
    stem_padded = length + max(PATCH_SIZE - PATCH_STRIDE, 0)
    stem_out = (stem_padded - PATCH_SIZE) // PATCH_STRIDE + 1
    length = stem_out
    out.append(length)
    for _ in range(1, len(NUM_BLOCKS)):
        padded = length + ((DOWNSAMPLE_RATIO - (length % DOWNSAMPLE_RATIO)) % DOWNSAMPLE_RATIO)
        length = (padded - DOWNSAMPLE_RATIO) // DOWNSAMPLE_RATIO + 1
        out.append(length)
    return out

stage_length_values = stage_lengths(SEQ_LEN)

metadata = {
    "model": "ModernTCN",
    "model_id": 11,
    "experiment": "fixed_baseline",
    "seed": SEED,
    "device": str(DEVICE),
    "torch_version": torch.__version__,
    "bf16_autocast": bool(USE_BF16),
    "paths": {
        "structured": str(STRUCTURED_PATH),
        "demographics": str(DEMOGRAPHICS_PATH),
        "index": str(INDEX_PATH),
        "targets": str(TARGET_PATH),
        "output": str(OUTPUT_DIR),
    },
    "cohort": {
        "n_total": int(len(X)),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "split_column": "split_24h",
    },
    "architecture": {
        "input": [SEQ_LEN, INPUT_DIM],
        "modern_tcn_source_family": "official ModernTCN classification backbone",
        "patch_size": PATCH_SIZE,
        "patch_stride": PATCH_STRIDE,
        "downsample_ratio": DOWNSAMPLE_RATIO,
        "ffn_ratio": FFN_RATIO,
        "num_blocks": NUM_BLOCKS,
        "large_kernels": LARGE_KERNELS,
        "small_kernels": SMALL_KERNELS,
        "dims": DIMS,
        "stage_output_lengths": stage_length_values,
        "backbone_dropout": BACKBONE_DROPOUT,
        "head_dropout": HEAD_DROPOUT,
        "class_dropout": CLASS_DROPOUT,
        "small_kernel_merged": SMALL_KERNEL_MERGED,
        "use_multi_scale": USE_MULTI_SCALE,
        "use_revin": USE_REVIN,
        "temporal_pooling": "mean_over_final_stage_time_dimension",
        "temporal_head_hidden": TEMPORAL_HEAD_HIDDEN,
        "demographics": DEMOG_DIM,
        "demog_hidden": DEMOG_HIDDEN,
        "fusion_hidden": FUSION_HIDDEN,
        "fusion_out": FUSION_OUT,
        "head_adaptation": (
            "ModernTCN final-stage variable representations are time-pooled, "
            "projected, then fused with 4 demographic features."
        ),
    },
    "optimization": {
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "grad_clip": GRAD_CLIP,
        "num_workers": NUM_WORKERS,
        "classification_loss": "BCEWithLogitsLoss + train pos_weight",
        "regression_loss": "MSELoss",
        "selection_metric_classification": "validation AUROC",
        "selection_metric_regression": "validation MAE",
        "tuning": False,
    },
    "results": all_results,
}

with open(OUTPUT_DIR / "metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, allow_nan=True)

print("\n" + "=" * 78)
print("MODERNTCN BASELINE COMPLETE")
print("=" * 78)
print(f"Metrics             : {summary_path}")
print(f"Output directory    : {OUTPUT_DIR}")
print("All four tasks completed with the fixed ModernTCN configuration.")
