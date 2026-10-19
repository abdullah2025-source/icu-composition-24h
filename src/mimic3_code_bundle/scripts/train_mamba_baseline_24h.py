#!/usr/bin/env python3
# ============================================================
# PIPELINE A — MAMBA-1 BASELINE / 24H PREDICTION
# Same cohort, inputs, targets, splits, and evaluation protocol
# used by the completed LightGBM / LSTM / BiGRU / TCN V2 / CNN+GRU baselines.
#
# Mamba-1 configuration:
#   Input projection      : 44 -> 256
#   Model dimension       : 256
#   Mamba blocks          : 6
#   SSM state             : 16
#   Local conv width      : 4
#   Expansion             : 2
#   Block normalization   : Pre-LayerNorm
#   Pooling               : final token
#   Demographics          : 4 -> 64
#   Fusion MLP            : 320 -> 256 -> 128
#
# Notes:
# - Uses the canonical structured_24h_processed.npy and
#   demographics_24h_processed.npy.
# - Uses split_24h from structured_24h_index.csv.
# - Uses canonical five_task_component_targets_24h.npz.
# - Classification tasks use BCEWithLogitsLoss with train-set
#   positive weighting.
# - LOS uses MSELoss.
# - BF16 autocast is enabled on CUDA/ROCm when supported.
# - Each task is trained as an independent single-task model,
#   matching the existing baseline protocol.
# ============================================================

from __future__ import annotations

import gc
import json
import math
import random
import time
import warnings
from pathlib import Path
from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

try:
    import importlib.metadata as importlib_metadata
    MAMBA_SSM_VERSION = importlib_metadata.version("mamba-ssm")
except Exception:
    MAMBA_SSM_VERSION = "not-installed"
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
OUTPUT_DIR = BASE_DIR / "mamba_baseline_24h"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STRUCTURED_PATH = INPUT_DIR / "structured_24h_processed.npy"
DEMOGRAPHICS_PATH = INPUT_DIR / "demographics_24h_processed.npy"
INDEX_PATH = INPUT_DIR / "structured_24h_index.csv"
TARGET_PATH = TARGET_DIR / "five_task_component_targets_24h.npz"

SEQ_LEN = 24
INPUT_DIM = 44
DEMOG_DIM = 4

# Transformer baseline: deliberately non-trivial / medium-sized.
D_MODEL = 256
N_LAYERS = 6
D_STATE = 16
D_CONV = 4
EXPAND = 2
DROPOUT = 0.10
USE_MAMBA_FAST_PATH = True

# Fusion / demographics.
DEMOG_HIDDEN = 64
FUSION_HIDDEN = 256
FUSION_OUT = 128

# Optimization.
BATCH_SIZE = 512
MAX_EPOCHS = 30
PATIENCE = 5
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
NUM_WORKERS = 0  # conservative and reliable on MI210 / ROCm

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
    # On recent ROCm PyTorch builds this is generally available on MI210.
    try:
        supported = torch.cuda.is_bf16_supported()
        return bool(supported)
    except Exception:
        # Conservative fallback for ROCm versions where the helper is missing.
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

print("=" * 72)
print("MAMBA-1 BASELINE — PIPELINE A / 24H")
print("=" * 72)
print(f"Device            : {DEVICE}")
if torch.cuda.is_available():
    try:
        print(f"GPU               : {torch.cuda.get_device_name(0)}")
    except Exception:
        print("GPU               : CUDA/ROCm device")
print(f"BF16 autocast     : {USE_BF16}")
print(f"mamba-ssm version  : {MAMBA_SSM_VERSION}")
print(f"Output directory  : {OUTPUT_DIR}")
print()
print("Mamba-1 config:")
print(f"  d_model         : {D_MODEL}")
print(f"  n_layers        : {N_LAYERS}")
print(f"  d_state         : {D_STATE}")
print(f"  d_conv          : {D_CONV}")
print(f"  expand          : {EXPAND}")
print(f"  dropout         : {DROPOUT}")
print(f"  batch size      : {BATCH_SIZE}")

# ============================================================
# 2. LOAD + VALIDATE CANONICAL INPUTS
# ============================================================

print("\nLoading canonical inputs...")

X = np.load(STRUCTURED_PATH)
D = np.load(DEMOGRAPHICS_PATH)
index_df = pd.read_csv(INDEX_PATH)
targets = np.load(TARGET_PATH)

print(f"Structured shape   : {X.shape}")
print(f"Demographics shape : {D.shape}")
print(f"Index shape        : {index_df.shape}")
print(f"Target stay IDs    : {targets['stay_id'].shape}")

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

# IMPORTANT: canonical index uses split_24h, not split.
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

if not np.array_equal(index_stay_ids.astype(np.int64), target_stay_ids.astype(np.int64)):
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

# Explicit copies so torch never receives read-only numpy buffers.
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
    raise ValueError(
        "One or more canonical splits are empty."
    )

print(
    f"Splits             : train={len(train_idx):,} "
    f"val={len(val_idx):,} test={len(test_idx):,}"
)

# Keep an explicit expected-size validation to catch accidental split changes.
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
        self.x = torch.from_numpy(
            np.array(x[indices], dtype=np.float32, copy=True)
        )
        self.d = torch.from_numpy(
            np.array(d[indices], dtype=np.float32, copy=True)
        )
        self.y = torch.from_numpy(
            np.array(y[indices], dtype=np.float32, copy=True)
        )

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
# 5. MODEL
# ============================================================

class MambaBlock(nn.Module):
    """Pre-LN residual Mamba-1 block."""

    def __init__(
        self,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        dropout: float,
    ) -> None:
        super().__init__()
        try:
            from mamba_ssm import Mamba
        except Exception as exc:
            raise ImportError(
                "mamba_ssm is required. Install it in the FYP environment "
                "before running this script."
            ) from exc

        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            use_fast_path=USE_MAMBA_FAST_PATH,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.mamba(self.norm(x)))


class MambaBaseline(nn.Module):
    """
    Input:
        x: [B, 24, 44]
        d: [B, 4]

    Architecture:
        44 -> 256 input projection
        6 x pre-LN Mamba-1 blocks
        causal last-token sequence representation
        demographics MLP
        fusion MLP
        one task head
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        d_model: int = D_MODEL,
        n_layers: int = N_LAYERS,
        d_state: int = D_STATE,
        d_conv: int = D_CONV,
        expand: int = EXPAND,
        dropout: float = DROPOUT,
        demog_dim: int = DEMOG_DIM,
        demog_hidden: int = DEMOG_HIDDEN,
        fusion_hidden: int = FUSION_HIDDEN,
        fusion_out: int = FUSION_OUT,
        regression: bool = False,
    ) -> None:
        super().__init__()

        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        self.blocks = nn.ModuleList(
            [
                MambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)

        self.demographics = nn.Sequential(
            nn.Linear(demog_dim, demog_hidden),
            nn.LayerNorm(demog_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(demog_hidden, demog_hidden),
            nn.GELU(),
        )

        self.fusion = nn.Sequential(
            nn.Linear(d_model + demog_hidden, fusion_hidden),
            nn.LayerNorm(fusion_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, fusion_out),
            nn.GELU(),
        )

        self.head = nn.Linear(fusion_out, 1)
        self.regression = regression

    def forward(self, x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)

        # Mamba scans left-to-right; the final token is the summary
        # representation of the complete 24-hour history.
        seq_repr = x[:, -1, :]
        demog_repr = self.demographics(d)

        fused = torch.cat([seq_repr, demog_repr], dim=-1)
        fused = self.fusion(fused)

        out = self.head(fused).squeeze(-1)
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
        "precision": float(
            precision_score(y_true, preds, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, preds, zero_division=0)
        ),
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
        "rmse": float(
            math.sqrt(mean_squared_error(y_true, pred))
        ),
        "r2": float(r2_score(y_true, pred))
        if len(y_true) >= 2
        else float("nan"),
        "median_absolute_error": float(
            median_absolute_error(y_true, pred)
        ),
    }

# ============================================================
# 7. EVALUATION
# ============================================================

def autocast_context():
    if DEVICE.type == "cuda" and USE_BF16:
        return torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        )
    return torch.autocast(
        device_type=DEVICE.type,
        enabled=False,
    )

def evaluate_classification(
    model: nn.Module,
    loader: DataLoader,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:

    model.eval()
    all_probs = []
    all_y = []

    with torch.no_grad():
        for xb, db, yb in loader:
            xb = xb.to(DEVICE, non_blocking=True)
            db = db.to(DEVICE, non_blocking=True)

            with autocast_context():
                logits = model(xb, db)

            probs = torch.sigmoid(logits)

            all_probs.append(
                probs.float().cpu().numpy()
            )
            all_y.append(
                yb.numpy()
            )

    probs = np.concatenate(all_probs)
    y_true = np.concatenate(all_y).astype(np.int64)

    return (
        binary_metrics(y_true, probs),
        y_true,
        probs,
    )

def evaluate_regression(
    model: nn.Module,
    loader: DataLoader,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:

    model.eval()
    all_pred = []
    all_y = []

    with torch.no_grad():
        for xb, db, yb in loader:
            xb = xb.to(DEVICE, non_blocking=True)
            db = db.to(DEVICE, non_blocking=True)

            with autocast_context():
                pred = model(xb, db)

            all_pred.append(
                pred.float().cpu().numpy()
            )
            all_y.append(
                yb.numpy()
            )

    pred = np.concatenate(all_pred)
    y_true = np.concatenate(all_y)

    return (
        regression_metrics(y_true, pred),
        y_true,
        pred,
    )

# ============================================================
# 8. TARGET PREPARATION
# ============================================================

def get_target_and_valid(
    task: str,
) -> Tuple[np.ndarray, np.ndarray]:

    if task not in targets:
        raise KeyError(
            f"Target '{task}' not found in {TARGET_PATH}"
        )

    valid_key = f"{task}_valid"

    if valid_key not in targets:
        raise KeyError(
            f"Validity mask '{valid_key}' not found in {TARGET_PATH}"
        )

    value = np.asarray(
        targets[task]
    ).reshape(-1)

    valid = np.asarray(
        targets[valid_key]
    ).reshape(-1)

    if len(value) != len(X):
        raise ValueError(
            f"{task}: target length mismatch with canonical cohort."
        )

    if len(valid) != len(X):
        raise ValueError(
            f"{task}: validity-mask length mismatch with canonical cohort."
        )

    return (
        value.astype(np.float32),
        valid.astype(bool),
    )

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
        raise ValueError(
            f"{task_name}: one split has zero valid samples."
        )

    y = np.asarray(
        y_full,
        dtype=np.float32,
    )

    print("\n" + "=" * 72)
    print(f"TRAINING: {task_name}")
    print("=" * 72)

    print(
        f"Valid samples     : train={len(task_train_idx):,} "
        f"val={len(task_val_idx):,} "
        f"test={len(task_test_idx):,}"
    )

    if not is_regression:
        train_rate = float(y[task_train_idx].mean())
        val_rate = float(y[task_val_idx].mean())
        test_rate = float(y[task_test_idx].mean())

        print(
            f"Positive rate     : train={train_rate:.4f} "
            f"val={val_rate:.4f} test={test_rate:.4f}"
        )

    # Datasets / loaders are task-specific because valid masks
    # can theoretically differ by target.
    train_loader = make_loader(
        X,
        D,
        y,
        task_train_idx,
        shuffle=True,
    )

    val_loader = make_loader(
        X,
        D,
        y,
        task_val_idx,
        shuffle=False,
    )

    test_loader = make_loader(
        X,
        D,
        y,
        task_test_idx,
        shuffle=False,
    )

    model = MambaBaseline(
        regression=is_regression
    ).to(DEVICE)

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Trainable params   : {trainable_params:,}"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    if is_regression:
        criterion = nn.MSELoss()
        print("Loss               : MSELoss")
    else:
        positives = float(
            (y[task_train_idx] == 1).sum()
        )
        negatives = float(
            (y[task_train_idx] == 0).sum()
        )

        if positives <= 0:
            raise ValueError(
                f"{task_name}: no positive samples in training split."
            )

        pos_weight_value = (
            negatives / positives
        )

        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(
                pos_weight_value,
                device=DEVICE,
                dtype=torch.float32,
            )
        )

        print(
            f"BCE pos_weight     : {pos_weight_value:.6f}"
        )

    checkpoint_path = (
        OUTPUT_DIR /
        f"best_{task_name}.pt"
    )

    history_path = (
        OUTPUT_DIR /
        f"{task_name}_training_history.csv"
    )

    pred_path = (
        OUTPUT_DIR /
        f"{task_name}_test_predictions.csv"
    )

    best_epoch = 1
    best_monitor = (
        float("inf")
        if is_regression
        else -float("inf")
    )
    patience_counter = 0
    history = []

    start_time = time.time()

    for epoch in range(
        1,
        MAX_EPOCHS + 1
    ):

        model.train()

        batch_losses = []
        epoch_start = time.time()

        for xb, db, yb in train_loader:

            xb = xb.to(
                DEVICE,
                non_blocking=True,
            )
            db = db.to(
                DEVICE,
                non_blocking=True,
            )
            yb = yb.to(
                DEVICE,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with autocast_context():
                pred = model(
                    xb,
                    db,
                )
                loss = criterion(
                    pred,
                    yb,
                )

            loss.backward()

            if GRAD_CLIP is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    GRAD_CLIP,
                )

            optimizer.step()

            batch_losses.append(
                float(loss.detach().float().cpu())
            )

        train_loss = float(
            np.mean(batch_losses)
        )

        if is_regression:
            val_metrics, _, _ = evaluate_regression(
                model,
                val_loader,
            )
            monitor_value = val_metrics["mae"]

            current_lr = optimizer.param_groups[0]["lr"]

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": float("nan"),
                    "val_mae": val_metrics["mae"],
                    "val_rmse": val_metrics["rmse"],
                    "val_r2": val_metrics["r2"],
                    "val_median_absolute_error": (
                        val_metrics["median_absolute_error"]
                    ),
                    "learning_rate": current_lr,
                    "epoch_seconds": time.time() - epoch_start,
                }
            )

            print(
                f"Epoch {epoch:02d}: "
                f"train_loss={train_loss:.5f} "
                f"val_MAE={val_metrics['mae']:.5f} "
                f"val_RMSE={val_metrics['rmse']:.5f} "
                f"val_R2={val_metrics['r2']:.5f} "
                f"time={time.time()-epoch_start:.1f}s"
            )

            improved = (
                monitor_value <
                best_monitor
            )

        else:
            val_metrics, _, _ = evaluate_classification(
                model,
                val_loader,
            )
            monitor_value = val_metrics["auroc"]

            current_lr = optimizer.param_groups[0]["lr"]

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": float("nan"),
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
                f"Epoch {epoch:02d}: "
                f"train_loss={train_loss:.5f} "
                f"val_AUROC={val_metrics['auroc']:.5f} "
                f"val_AUPRC={val_metrics['auprc']:.5f} "
                f"val_Brier={val_metrics['brier']:.5f} "
                f"time={time.time()-epoch_start:.1f}s"
            )

            improved = (
                np.isfinite(monitor_value)
                and monitor_value >
                best_monitor
            )

        if improved:
            best_monitor = monitor_value
            best_epoch = epoch
            patience_counter = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "task_name": task_name,
                    "best_epoch": best_epoch,
                    "best_monitor": best_monitor,
                    "config": {
                        "seq_len": SEQ_LEN,
                        "input_dim": INPUT_DIM,
                        "d_model": D_MODEL,
                        "n_layers": N_LAYERS,
                        "d_state": D_STATE,
                        "d_conv": D_CONV,
                        "expand": EXPAND,
                        "dropout": DROPOUT,
                        "demog_dim": DEMOG_DIM,
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
            print(
                f"Early stopping at epoch {epoch}."
            )
            break

        del val_metrics
        gc.collect()

    pd.DataFrame(history).to_csv(
        history_path,
        index=False,
    )

    # Load best checkpoint before final evaluation.
    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    if is_regression:
        val_metrics, val_y, val_pred = evaluate_regression(
            model,
            val_loader,
        )
        test_metrics, test_y, test_pred = evaluate_regression(
            model,
            test_loader,
        )

        prediction_df = pd.DataFrame(
            {
                "y_true": test_y,
                "prediction": test_pred,
            }
        )

        prediction_df.to_csv(
            pred_path,
            index=False,
        )

        result = {
            "best_epoch": int(best_epoch),
            "best_monitor": float(best_monitor),
            "val": val_metrics,
            "test": test_metrics,
            "training_minutes": (
                time.time() - start_time
            ) / 60.0,
            "n_train": int(len(task_train_idx)),
            "n_val": int(len(task_val_idx)),
            "n_test": int(len(task_test_idx)),
            "trainable_params": int(trainable_params),
        }

        print("\nValidation metrics:")
        for k, v in val_metrics.items():
            print(f"  {k}: {v}")

        print("\nTEST metrics:")
        for k, v in test_metrics.items():
            print(f"  {k}: {v}")

        print(
            f"\nBest epoch       : {best_epoch}"
        )
        print(
            f"Training time    : {result['training_minutes']:.2f} minutes"
        )

    else:
        val_metrics, val_y, val_probs = evaluate_classification(
            model,
            val_loader,
        )
        test_metrics, test_y, test_probs = evaluate_classification(
            model,
            test_loader,
        )

        prediction_df = pd.DataFrame(
            {
                "y_true": test_y,
                "probability": test_probs,
                "prediction_0p5": (
                    test_probs >= 0.5
                ).astype(np.int8),
            }
        )

        prediction_df.to_csv(
            pred_path,
            index=False,
        )

        result = {
            "best_epoch": int(best_epoch),
            "best_monitor": float(best_monitor),
            "val": val_metrics,
            "test": test_metrics,
            "training_minutes": (
                time.time() - start_time
            ) / 60.0,
            "n_train": int(len(task_train_idx)),
            "n_val": int(len(task_val_idx)),
            "n_test": int(len(task_test_idx)),
            "trainable_params": int(trainable_params),
        }

        print("\nValidation metrics:")
        for k, v in val_metrics.items():
            print(f"  {k}: {v}")

        print("\nTEST metrics:")
        for k, v in test_metrics.items():
            print(f"  {k}: {v}")

        print(
            f"\nBest epoch       : {best_epoch}"
        )
        print(
            f"Training time    : {result['training_minutes']:.2f} minutes"
        )

    del model
    del train_loader
    del val_loader
    del test_loader
    gc.collect()

    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

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

los_y, los_valid = get_target_and_valid(
    REGRESSION_TASK
)

all_results[REGRESSION_TASK] = train_task(
    task_name=REGRESSION_TASK,
    y_full=los_y,
    valid_full=los_valid,
    is_regression=True,
)

# ============================================================
# 11. FINAL SUMMARY TABLE
# ============================================================

rows = []

for task_name, result in all_results.items():

    if task_name in CLASSIFICATION_TASKS.values():

        val = result["val"]
        test = result["test"]

        row_val = {
            "target": task_name,
            "split": "val",
            "best_epoch": result["best_epoch"],
            "val_auroc": val["auroc"],
            "val_auprc": val["auprc"],
            "val_brier": val["brier"],
            "val_precision": val["precision"],
            "val_recall": val["recall"],
            "val_f1": val["f1"],
            "val_sensitivity": val["sensitivity"],
            "val_specificity": val["specificity"],
            "val_tn": val["tn"],
            "val_fp": val["fp"],
            "val_fn": val["fn"],
            "val_tp": val["tp"],
            "val_prevalence": val["prevalence"],
            "test_auroc": np.nan,
            "test_auprc": np.nan,
            "test_brier": np.nan,
            "test_precision": np.nan,
            "test_recall": np.nan,
            "test_f1": np.nan,
            "test_sensitivity": np.nan,
            "test_specificity": np.nan,
            "test_tn": np.nan,
            "test_fp": np.nan,
            "test_fn": np.nan,
            "test_tp": np.nan,
            "test_prevalence": np.nan,
            "val_mae": np.nan,
            "val_rmse": np.nan,
            "val_r2": np.nan,
            "val_median_absolute_error": np.nan,
            "test_mae": np.nan,
            "test_rmse": np.nan,
            "test_r2": np.nan,
            "test_median_absolute_error": np.nan,
        }
        rows.append(row_val)

        row_test = {
            "target": task_name,
            "split": "test",
            "best_epoch": result["best_epoch"],
            "val_auroc": np.nan,
            "val_auprc": np.nan,
            "val_brier": np.nan,
            "val_precision": np.nan,
            "val_recall": np.nan,
            "val_f1": np.nan,
            "val_sensitivity": np.nan,
            "val_specificity": np.nan,
            "val_tn": np.nan,
            "val_fp": np.nan,
            "val_fn": np.nan,
            "val_tp": np.nan,
            "val_prevalence": np.nan,
            "test_auroc": test["auroc"],
            "test_auprc": test["auprc"],
            "test_brier": test["brier"],
            "test_precision": test["precision"],
            "test_recall": test["recall"],
            "test_f1": test["f1"],
            "test_sensitivity": test["sensitivity"],
            "test_specificity": test["specificity"],
            "test_tn": test["tn"],
            "test_fp": test["fp"],
            "test_fn": test["fn"],
            "test_tp": test["tp"],
            "test_prevalence": test["prevalence"],
            "val_mae": np.nan,
            "val_rmse": np.nan,
            "val_r2": np.nan,
            "val_median_absolute_error": np.nan,
            "test_mae": np.nan,
            "test_rmse": np.nan,
            "test_r2": np.nan,
            "test_median_absolute_error": np.nan,
        }
        rows.append(row_test)

    else:

        val = result["val"]
        test = result["test"]

        row_val = {
            "target": task_name,
            "split": "val",
            "best_epoch": result["best_epoch"],
            "val_auroc": np.nan,
            "val_auprc": np.nan,
            "val_brier": np.nan,
            "val_precision": np.nan,
            "val_recall": np.nan,
            "val_f1": np.nan,
            "val_sensitivity": np.nan,
            "val_specificity": np.nan,
            "val_tn": np.nan,
            "val_fp": np.nan,
            "val_fn": np.nan,
            "val_tp": np.nan,
            "val_prevalence": np.nan,
            "test_auroc": np.nan,
            "test_auprc": np.nan,
            "test_brier": np.nan,
            "test_precision": np.nan,
            "test_recall": np.nan,
            "test_f1": np.nan,
            "test_sensitivity": np.nan,
            "test_specificity": np.nan,
            "test_tn": np.nan,
            "test_fp": np.nan,
            "test_fn": np.nan,
            "test_tp": np.nan,
            "test_prevalence": np.nan,
            "val_mae": val["mae"],
            "val_rmse": val["rmse"],
            "val_r2": val["r2"],
            "val_median_absolute_error": val["median_absolute_error"],
            "test_mae": np.nan,
            "test_rmse": np.nan,
            "test_r2": np.nan,
            "test_median_absolute_error": np.nan,
        }
        rows.append(row_val)

        row_test = {
            "target": task_name,
            "split": "test",
            "best_epoch": result["best_epoch"],
            "val_auroc": np.nan,
            "val_auprc": np.nan,
            "val_brier": np.nan,
            "val_precision": np.nan,
            "val_recall": np.nan,
            "val_f1": np.nan,
            "val_sensitivity": np.nan,
            "val_specificity": np.nan,
            "val_tn": np.nan,
            "val_fp": np.nan,
            "val_fn": np.nan,
            "val_tp": np.nan,
            "val_prevalence": np.nan,
            "test_auroc": np.nan,
            "test_auprc": np.nan,
            "test_brier": np.nan,
            "test_precision": np.nan,
            "test_recall": np.nan,
            "test_f1": np.nan,
            "test_sensitivity": np.nan,
            "test_specificity": np.nan,
            "test_tn": np.nan,
            "test_fp": np.nan,
            "test_fn": np.nan,
            "test_tp": np.nan,
            "test_prevalence": np.nan,
            "val_mae": np.nan,
            "val_rmse": np.nan,
            "val_r2": np.nan,
            "val_median_absolute_error": np.nan,
            "test_mae": test["mae"],
            "test_rmse": test["rmse"],
            "test_r2": test["r2"],
            "test_median_absolute_error": test["median_absolute_error"],
        }
        rows.append(row_test)

summary_df = pd.DataFrame(rows)

summary_path = OUTPUT_DIR / "metrics.csv"
summary_df.to_csv(
    summary_path,
    index=False,
)

metadata = {
    "model": "Mamba-1",
    "seed": SEED,
    "device": str(DEVICE),
    "torch_version": torch.__version__,
    "mamba_ssm_version": MAMBA_SSM_VERSION,
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
        "d_model": D_MODEL,
        "n_layers": N_LAYERS,
        "d_state": D_STATE,
        "d_conv": D_CONV,
        "expand": EXPAND,
        "dropout": DROPOUT,
        "pooling": "last_token",
        "demographics": DEMOG_DIM,
        "demog_hidden": DEMOG_HIDDEN,
        "fusion_hidden": FUSION_HIDDEN,
        "fusion_out": FUSION_OUT,
        "activation": "silu_inside_mamba__gelu_input_fusion",
        "normalization": "pre_layer_norm",
        "mamba_fast_path": bool(USE_MAMBA_FAST_PATH),
    },
    "optimization": {
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "grad_clip": GRAD_CLIP,
        "num_workers": NUM_WORKERS,
        "classification_loss": (
            "BCEWithLogitsLoss + train pos_weight"
        ),
        "regression_loss": "MSELoss",
    },
    "results": all_results,
}

with open(
    OUTPUT_DIR / "metadata.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        metadata,
        f,
        indent=2,
        allow_nan=True,
    )

print("\n" + "=" * 72)
print("MAMBA-1 BASELINE COMPLETE")
print("=" * 72)
print(summary_df.to_string(index=False))
print()
print(f"Results saved to: {OUTPUT_DIR}")
print("Files:")
print("  metrics.csv")
print("  metadata.json")
print("  *_training_history.csv")
print("  best_*.pt")
print("  *_test_predictions.csv")
print("=" * 72)
