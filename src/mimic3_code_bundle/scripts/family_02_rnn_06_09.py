
#!/usr/bin/env python3
"""
Pipeline A V2 — Family 2: RNN / Missingness-Aware Baselines (06–09)

Models
------
06 GRU
07 LSTM
08 BiGRU
09 GRU-D

Fixed settings
--------------
06 GRU     hidden=256, layers=2, dropout=0.20, lr=1e-3
07 LSTM    same as GRU
08 BiGRU   hidden=256, layers=2, dropout=0.20, lr=1e-3
09 GRU-D   hidden=256, dropout=0.15, lr=1e-3

Optuna spaces
-------------
06–08: hidden 128–512, layers 1–3, dropout 0.10–0.35,
       lr 3e-4–2e-3
09:    hidden 128–512, dropout 0.10–0.30, lr 5e-4–2e-3

Global V2 training defaults
---------------------------
batch_size=512, max_epochs=50, patience=7, AdamW,
weight_decay=1e-4, grad_clip=1.0, cosine scheduler,
BF16 when supported, num_workers=0.

Global tuned ranges
-------------------
batch_size: 256 / 512 / 1024
patience: 5–10
weight_decay: 1e-5–1e-3
grad_clip: 0.5–2.0
scheduler: cosine / OneCycle

V2 contract
-----------
- Structured input is the frozen [N,24,110] tensor.
- No new preprocessing, scaling, imputation, or split.
- Frozen split_24h: 52,268 / 7,417 / 14,887.
- Ventilation/vasopressor targets use their explicit validity masks.
- -1 is invalid, never a negative class.
- Deterioration is derived only; never a trained primary head.
- Checkpoint/Optuna selection = mean validation AUPRC over the
  four binary primary tasks.
- LOS is trained as log1p(excess LOS beyond the fixed 24-hour landmark)
  and reported back on the original total-hour scale by inverse-transforming
  and adding 24 hours. The frozen target remains total LOS; only the
  model-side regression transform is centered at the landmark.
- Demographics are consumed as a small separate branch and fused
  with the temporal representation.

GRU-D
------
GRU-D explicitly consumes:
    values      = channels 0:28
    masks       = channels 28:56
    time-since  = channels 56:84
and concatenates the frozen trend channels 84:110 as auxiliary
temporal inputs.

GRU-D input-decay fidelity:
    input decay uses a diagonal/elementwise mapping from each variable's
    own time-since channel, matching the Che et al. GRU-D formulation.
    Hidden-state decay remains a dense mapping from all time-since channels
    to the hidden state, as in the reference model.

The frozen V2 tensor is already forward-filled upstream. GRU-D therefore
uses the explicit V2 masks/time-since channels to learn an additional
decay mechanism on the frozen model-ready representation; it does not
replace or reopen preprocessing.

Statistical evaluation
-----------------------
The same file also provides --mode stats, which post-hoc evaluates
the existing fixed and tuned test predictions with:
- bootstrap 95% CIs (1,000 resamples)
- AUROC, AUPRC, Brier
- MAE, RMSE, MedianAE, R2
- ECE for binary tasks

No retraining occurs in stats mode.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import platform
import random
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Optional imports with clear errors later
# ---------------------------------------------------------------------
try:
    import optuna
except ImportError:
    optuna = None

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:
    torch = None
    nn = None
    F = None
    DataLoader = None
    Dataset = None
    _TORCH_IMPORT_ERROR = exc

from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
    roc_auc_score,
)


# ---------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------
PROJECT_ROOT = Path("/home/abdullah2025/fyp_project")
V2_DIR = PROJECT_ROOT / "outputs" / "pipeline_a_v2"

ALIGNMENT_PATH = V2_DIR / "a5_alignment_index_24h.csv"
STRUCTURED_PATH = V2_DIR / "structured_24h_processed.npy"
DEMOGRAPHICS_PATH = V2_DIR / "demographics_24h_processed.npy"
INPUT_METADATA_PATH = V2_DIR / "final_model_input_metadata_24h.csv"
INPUT_DEFINITION_PATH = V2_DIR / "final_model_input_definition_24h.json"
OUTCOMES_PATH = V2_DIR / "targets_24h" / "outcomes_24h_primary_cohort.csv"

RESULTS_ROOT = PROJECT_ROOT / "outputs" / "baseline_family_02_rnn"

SEED = 20260923  # Shared benchmark seed with finalized Family 1 harness.

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

MODEL_NAMES = {
    "06": "GRU",
    "07": "LSTM",
    "08": "BiGRU",
    "09": "GRU-D",
}


# ---------------------------------------------------------------------
# Registered model configurations
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    name: str
    fixed: Dict[str, Any]
    search_space: Dict[str, Any]


MODEL_CONFIGS: Dict[str, ModelConfig] = {
    "06": ModelConfig(
        "06",
        "GRU",
        {
            "hidden": 256,
            "layers": 2,
            "dropout": 0.20,
            "lr": 1e-3,
        },
        {
            "hidden": "int[128,512]",
            "layers": "int[1,3]",
            "dropout": "float[0.10,0.35]",
            "lr": "logfloat[3e-4,2e-3]",
        },
    ),
    "07": ModelConfig(
        "07",
        "LSTM",
        {
            "hidden": 256,
            "layers": 2,
            "dropout": 0.20,
            "lr": 1e-3,
        },
        {
            "hidden": "int[128,512]",
            "layers": "int[1,3]",
            "dropout": "float[0.10,0.35]",
            "lr": "logfloat[3e-4,2e-3]",
        },
    ),
    "08": ModelConfig(
        "08",
        "BiGRU",
        {
            "hidden": 256,
            "layers": 2,
            "dropout": 0.20,
            "lr": 1e-3,
        },
        {
            "hidden": "int[128,512]",
            "layers": "int[1,3]",
            "dropout": "float[0.10,0.35]",
            "lr": "logfloat[3e-4,2e-3]",
        },
    ),
    "09": ModelConfig(
        "09",
        "GRU-D",
        {
            "hidden": 256,
            "dropout": 0.15,
            "lr": 1e-3,
        },
        {
            "hidden": "int[128,512]",
            "dropout": "float[0.10,0.30]",
            "lr": "logfloat[5e-4,2e-3]",
        },
    ),
}


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)

    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def require_torch() -> None:
    if torch is None:
        raise ImportError(
            "PyTorch is required for Family 2."
        ) from _TORCH_IMPORT_ERROR


def require_optuna() -> None:
    if optuna is None:
        raise ImportError(
            "Optuna is required for tuned mode. "
            "Install it in /home/abdullah2025/fyp_project/mamba_env."
        )


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required frozen V2 artifact not found: {path}"
        )


def get_device() -> torch.device:
    require_torch()
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def amp_enabled(device: torch.device) -> bool:
    return device.type == "cuda"


# ---------------------------------------------------------------------
# Frozen-data loading
# ---------------------------------------------------------------------
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
    """Return the explicit frozen V2 validity mask for one primary task.

    V2 requires an explicit *_valid column for every primary target.
    Missing schema is a hard error; it must never silently become
    "all rows valid".
    """
    if task not in VALID_COL:
        raise KeyError(
            f"Unknown primary task '{task}'."
        )

    col = VALID_COL[task]

    if col not in df.columns:
        raise KeyError(
            f"Required frozen V2 validity column '{col}' "
            f"is missing for task '{task}'."
        )

    return (
        df[col]
        .to_numpy(dtype=np.int8)
        == 1
    )


def load_frozen_v2() -> FrozenData:
    for p in (
        ALIGNMENT_PATH,
        STRUCTURED_PATH,
        DEMOGRAPHICS_PATH,
        INPUT_METADATA_PATH,
        INPUT_DEFINITION_PATH,
        OUTCOMES_PATH,
    ):
        require_file(p)

    idx_df = pd.read_csv(ALIGNMENT_PATH)
    outcomes_df = pd.read_csv(OUTCOMES_PATH)

    required_alignment = {
        "stay_id",
        "subject_id",
        "split_24h",
    }
    missing_alignment = sorted(
        required_alignment
        - set(idx_df.columns)
    )
    if missing_alignment:
        raise KeyError(
            f"Missing alignment columns: {missing_alignment}"
        )

    required_outcomes = {
        "stay_id",
        "subject_id",
        *PRIMARY_TASKS,
        *VALID_COL.values(),
    }
    missing_outcomes = sorted(
        required_outcomes
        - set(outcomes_df.columns)
    )
    if missing_outcomes:
        raise KeyError(
            f"Missing outcome columns: {missing_outcomes}"
        )

    join_cols = ["stay_id", "subject_id"]
    out_cols = (
        join_cols
        + list(PRIMARY_TASKS)
        + [
            c
            for c in outcomes_df.columns
            if c.endswith("_valid")
        ]
    )

    outcomes_small = outcomes_df[
        list(dict.fromkeys(out_cols))
    ].copy()

    if outcomes_small["stay_id"].duplicated().any():
        raise ValueError(
            "Frozen outcomes contain duplicate stay_id rows."
        )

    merged = idx_df.merge(
        outcomes_small,
        on=join_cols,
        how="left",
        validate="one_to_one",
        suffixes=("", "_outcome"),
    )

    if merged[list(PRIMARY_TASKS)].isna().any().any():
        raise ValueError(
            "Frozen alignment/outcomes join contains missing primary targets."
        )

    structured = np.load(
        STRUCTURED_PATH,
        mmap_mode="r",
    )
    demographics = np.load(
        DEMOGRAPHICS_PATH,
        mmap_mode="r",
    )

    if tuple(structured.shape) != EXPECTED_SHAPE:
        raise ValueError(
            f"Unexpected structured shape {structured.shape}; "
            f"expected {EXPECTED_SHAPE}."
        )

    if tuple(demographics.shape) != (
        EXPECTED_N,
        4,
    ):
        raise ValueError(
            f"Unexpected demographics shape {demographics.shape}."
        )

    if len(merged) != EXPECTED_N:
        raise ValueError(
            f"Unexpected row count {len(merged)} != {EXPECTED_N}."
        )

    split = (
        merged["split_24h"]
        .astype(str)
        .to_numpy()
    )

    counts = (
        pd.Series(split)
        .value_counts()
        .to_dict()
    )

    expected = {
        "train": EXPECTED_TRAIN,
        "val": EXPECTED_VAL,
        "test": EXPECTED_TEST,
    }

    if counts != expected:
        raise ValueError(
            f"Frozen split counts differ from contract: "
            f"{counts} != {expected}."
        )

    train_idx = np.flatnonzero(
        split == "train"
    )
    val_idx = np.flatnonzero(
        split == "val"
    )
    test_idx = np.flatnonzero(
        split == "test"
    )

    # Precompute immutable numpy target/validity arrays once.
    # This avoids repeated pandas row slicing inside __getitem__ for every
    # sample, task and epoch.
    targets = {
        task: merged[task].to_numpy(dtype=np.float32)
        for task in PRIMARY_TASKS
    }
    valid_arrays = {
        task: validity_mask(merged, task)
        for task in PRIMARY_TASKS
    }

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


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------
class FrozenSequenceDataset(Dataset):
    def __init__(
        self,
        data: FrozenData,
        indices: np.ndarray,
    ) -> None:
        self.structured = data.structured
        self.demographics = data.demographics
        self.targets = data.targets
        self.valid_arrays = data.valid_arrays
        self.indices = np.asarray(
            indices,
            dtype=np.int64,
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, pos: int):
        i = int(self.indices[pos])

        # np.load(..., mmap_mode="r") is intentionally used for the frozen
        # arrays. Copy each sample before torch.from_numpy so PyTorch never
        # receives a non-writable NumPy view.
        x = np.array(
            self.structured[i],
            dtype=np.float32,
            copy=True,
        )

        d = np.array(
            self.demographics[i],
            dtype=np.float32,
            copy=True,
        )

        # Fast numpy indexing: no pandas iloc/column lookup inside training.
        y = {
            task: np.float32(
                self.targets[task][i]
            )
            for task in PRIMARY_TASKS
        }

        valid = {
            task: np.int8(
                self.valid_arrays[task][i]
            )
            for task in PRIMARY_TASKS
        }

        return (
            torch.from_numpy(x),
            torch.from_numpy(d),
            y,
            valid,
        )


def make_loader(
    data: FrozenData,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        FrozenSequenceDataset(
            data,
            indices,
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=pin_memory,
        drop_last=False,
    )


# ---------------------------------------------------------------------
# RNN architectures
# ---------------------------------------------------------------------
class MultitaskHead(nn.Module):
    def __init__(
        self,
        representation_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()

        head_dim = max(
            64,
            representation_dim // 2,
        )

        self.trunk = nn.Sequential(
            nn.Linear(
                representation_dim,
                head_dim,
            ),
            nn.LayerNorm(head_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.mortality = nn.Linear(
            head_dim,
            1,
        )
        self.ventilation = nn.Linear(
            head_dim,
            1,
        )
        self.vasopressor = nn.Linear(
            head_dim,
            1,
        )
        self.readmission = nn.Linear(
            head_dim,
            1,
        )
        self.los_raw = nn.Linear(
            head_dim,
            1,
        )

    def forward(self, h: torch.Tensor):
        z = self.trunk(h)

        return {
            "mortality_24h": self.mortality(z).squeeze(-1),
            "ventilation_onset_24h": self.ventilation(z).squeeze(-1),
            "vasopressor_onset_24h": self.vasopressor(z).squeeze(-1),
            "readmission_30d_24h": self.readmission(z).squeeze(-1),
            "los_excess_log1p": F.softplus(
                self.los_raw(z).squeeze(-1)
            ),
        }


class RNNBaseline(nn.Module):
    def __init__(
        self,
        model_id: str,
        hidden: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.model_id = model_id
        self.hidden = hidden
        self.layers = layers
        self.dropout = dropout

        bidirectional = (
            model_id == "08"
        )

        rnn_cls = {
            "06": nn.GRU,
            "07": nn.LSTM,
            "08": nn.GRU,
        }.get(model_id)

        if rnn_cls is None:
            raise ValueError(
                f"RNNBaseline does not handle model_id={model_id}; use GRUDBaseline."
            )

        self.encoder = rnn_cls(
            input_size=110,
            hidden_size=hidden,
            num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )

        rnn_dim = (
            hidden * 2
            if bidirectional
            else hidden
        )

        self.demo = nn.Sequential(
            nn.Linear(4, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.fusion = nn.Sequential(
            nn.Linear(
                rnn_dim + hidden,
                hidden,
            ),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.heads = MultitaskHead(
            representation_dim=hidden,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        demographics: torch.Tensor,
    ):
        outputs, hidden = self.encoder(x)

        if self.model_id == "07":
            h = hidden[0]
        else:
            h = hidden

        if self.encoder.bidirectional:
            h = torch.cat(
                [h[-2], h[-1]],
                dim=-1,
            )
        else:
            h = h[-1]

        d = self.demo(demographics)
        fused = self.fusion(
            torch.cat([h, d], dim=-1)
        )

        return self.heads(fused)


class GRUDBaseline(nn.Module):
    """
    GRU-D style baseline using the exact V2 channel semantics:
        values      0:28
        masks       28:56
        time-since  56:84
        trends      84:110

    The 28 value features are already V2-forward-filled/fallback-filled.
    GRU-D adds a learned decay mechanism driven by the explicit original
    observation state and time-since channels; it does not refit any
    preprocessing statistics.
    """

    def __init__(
        self,
        hidden: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.hidden = hidden
        self.dropout = dropout

        # Che et al. GRU-D: input decay uses a diagonal mapping,
        # so each variable's decay depends only on its own delta channel.
        self.x_decay_weight = nn.Parameter(
            torch.zeros(28)
        )
        self.x_decay_bias = nn.Parameter(
            torch.zeros(28)
        )
        self.h_decay = nn.Linear(
            28,
            hidden,
            bias=True,
        )

        # Learnable model parameter, NOT a recomputed train-set statistic.
        # V2 preprocessing is already frozen upstream.
        self.x_mean = nn.Parameter(
            torch.zeros(28)
        )

        # x_hat (28) + masks (28) + trends (26)
        self.cell = nn.GRUCell(
            input_size=82,
            hidden_size=hidden,
        )

        self.out_dropout = nn.Dropout(
            dropout
        )

        self.demo = nn.Sequential(
            nn.Linear(4, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.fusion = nn.Sequential(
            nn.Linear(
                hidden * 2,
                hidden,
            ),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.heads = MultitaskHead(
            representation_dim=hidden,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        demographics: torch.Tensor,
    ):
        values = x[:, :, 0:28]
        masks = x[:, :, 28:56]
        delta = x[:, :, 56:84]
        trends = x[:, :, 84:110]

        batch = x.shape[0]
        h = torch.zeros(
            batch,
            self.hidden,
            device=x.device,
            dtype=x.dtype,
        )

        x_prev = values[:, 0]

        for t in range(x.shape[1]):
            xt = values[:, t]
            mt = masks[:, t]
            dt = delta[:, t]

            # Elementwise equivalent of a diagonal W_gamma_x.
            gamma_x = torch.exp(
                -F.relu(
                    dt * self.x_decay_weight
                    + self.x_decay_bias
                )
            )

            x_hat = (
                mt * xt
                + (1.0 - mt)
                * (
                    gamma_x * x_prev
                    + (1.0 - gamma_x)
                    * self.x_mean
                )
            )

            gamma_h = torch.exp(
                -F.relu(
                    self.h_decay(dt)
                )
            )

            h = gamma_h * h

            recurrent_input = torch.cat(
                [x_hat, mt, trends[:, t]],
                dim=-1,
            )

            h = self.cell(
                recurrent_input,
                h,
            )

            x_prev = xt

        h = self.out_dropout(h)

        d = self.demo(demographics)

        fused = self.fusion(
            torch.cat([h, d], dim=-1)
        )

        return self.heads(fused)


def build_model(
    model_id: str,
    params: Dict[str, Any],
) -> nn.Module:
    if model_id in {"06", "07", "08"}:
        return RNNBaseline(
            model_id=model_id,
            hidden=int(params["hidden"]),
            layers=int(params["layers"]),
            dropout=float(params["dropout"]),
        )

    if model_id == "09":
        return GRUDBaseline(
            hidden=int(params["hidden"]),
            dropout=float(params["dropout"]),
        )

    raise ValueError(
        f"Unsupported Family 2 model_id={model_id}"
    )


# ---------------------------------------------------------------------
# Loss / prediction / metrics
# ---------------------------------------------------------------------
def masked_bce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    if int(valid.sum()) == 0:
        return logits.sum() * 0.0

    return F.binary_cross_entropy_with_logits(
        logits[valid],
        target[valid],
    )


LOS_LANDMARK_HOURS = 24.0


def los_loss(
    log_pred: torch.Tensor,
    target_hours: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    if int(valid.sum()) == 0:
        return log_pred.sum() * 0.0

    # Frozen V2 LOS target is total LOS in hours with a verified minimum
    # of 24h in the primary cohort. The benchmark target itself is NOT
    # changed; only the model-side regression coordinate system is shifted.
    excess_hours = torch.clamp(
        target_hours[valid] - LOS_LANDMARK_HOURS,
        min=0.0,
    )

    target_log = torch.log1p(
        excess_hours
    )

    return F.smooth_l1_loss(
        log_pred[valid],
        target_log,
    )


def multitask_loss(
    outputs: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    valids: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    parts = {}

    for task in BINARY_TASKS:
        parts[task] = masked_bce_loss(
            outputs[task],
            targets[task],
            valids[task],
        )

    parts["los_hours_24h"] = los_loss(
        outputs["los_excess_log1p"],
        targets["los_hours_24h"],
        valids["los_hours_24h"],
    )

    total = sum(parts.values())

    return total, {
        k: float(v.detach().cpu().item())
        for k, v in parts.items()
    }


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
):
    model.eval()

    records = {
        task: []
        for task in PRIMARY_TASKS
    }
    valid_records = {
        task: []
        for task in PRIMARY_TASKS
    }

    for x, demographics, y_dict, valid_dict in loader:
        x = x.to(
            device,
            non_blocking=True,
        )
        demographics = demographics.to(
            device,
            non_blocking=True,
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=use_amp,
        ):
            outputs = model(
                x,
                demographics,
            )

        for task in BINARY_TASKS:
            records[task].append(
                torch.sigmoid(
                    outputs[task]
                )
                .float()
                .cpu()
                .numpy()
            )
            valid_records[task].append(
                valid_dict[task]
                .numpy()
            )

        records["los_hours_24h"].append(
            (
                torch.expm1(
                    outputs["los_excess_log1p"]
                )
                + LOS_LANDMARK_HOURS
            )
            .float()
            .cpu()
            .numpy()
        )
        valid_records["los_hours_24h"].append(
            valid_dict["los_hours_24h"]
            .numpy()
        )

    return {
        task: np.concatenate(records[task])
        for task in PRIMARY_TASKS
    }, {
        task: np.concatenate(valid_records[task])
        for task in PRIMARY_TASKS
    }


def safe_auroc(y, p):
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def safe_auprc(y, p):
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, p))


def evaluate_predictions(
    data: FrozenData,
    pred: Dict[str, np.ndarray],
    valid: Dict[str, np.ndarray],
    indices: np.ndarray,
    split_name: str,
) -> pd.DataFrame:
    rows = []

    for task in PRIMARY_TASKS:
        y = data.targets[task][indices].astype(
            float,
            copy=False,
        )
        p = pred[task]
        v = valid[task].astype(bool)

        keep = (
            v
            & np.isfinite(y)
            & np.isfinite(p)
        )

        yt = y[keep]
        pp = p[keep]

        row = {
            "split": split_name,
            "task": task,
            "valid_n": int(len(yt)),
        }

        if task in BINARY_TASKS:
            row.update(
                {
                    "auroc": safe_auroc(
                        yt.astype(int),
                        pp,
                    ),
                    "auprc": safe_auprc(
                        yt.astype(int),
                        pp,
                    ),
                    "brier": float(
                        brier_score_loss(
                            yt.astype(int),
                            pp,
                        )
                    ),
                    "prevalence": float(
                        np.mean(yt)
                    ),
                }
            )
        else:
            row.update(
                {
                    "mae": float(
                        mean_absolute_error(
                            yt,
                            pp,
                        )
                    ),
                    "rmse": float(
                        np.sqrt(
                            mean_squared_error(
                                yt,
                                pp,
                            )
                        )
                    ),
                    "median_ae": float(
                        median_absolute_error(
                            yt,
                            pp,
                        )
                    ),
                    "r2": float(
                        r2_score(
                            yt,
                            pp,
                        )
                    )
                    if len(yt) > 1
                    else float("nan"),
                    "prevalence": float("nan"),
                }
            )

        rows.append(row)

    # Derived deterioration: evaluation-only.
    if set(BINARY_TASKS[:3]).issubset(
        set(pred)
    ):
        comp_valid = (
            valid["mortality_24h"].astype(bool)
            & valid["ventilation_onset_24h"].astype(bool)
            & valid["vasopressor_onset_24h"].astype(bool)
        )

        y_mort = data.targets[
            "mortality_24h"
        ][indices].astype(int, copy=False)
        y_vent = data.targets[
            "ventilation_onset_24h"
        ][indices].astype(int, copy=False)
        y_vaso = data.targets[
            "vasopressor_onset_24h"
        ][indices].astype(int, copy=False)

        y_det = (
            1
            - (1 - y_mort)
            * (1 - y_vent)
            * (1 - y_vaso)
        ).astype(int)

        p_det = (
            1
            - (1 - pred["mortality_24h"])
            * (1 - pred["ventilation_onset_24h"])
            * (1 - pred["vasopressor_onset_24h"])
        )

        keep = (
            comp_valid
            & np.isfinite(y_det)
            & np.isfinite(p_det)
        )

        yt = y_det[keep]
        pp = p_det[keep]

        rows.append(
            {
                "split": split_name,
                "task": "deterioration_24h",
                "valid_n": int(len(yt)),
                "auroc": safe_auroc(
                    yt,
                    pp,
                ),
                "auprc": safe_auprc(
                    yt,
                    pp,
                ),
                "brier": float(
                    brier_score_loss(
                        yt,
                        pp,
                    )
                ),
                "prevalence": float(
                    np.mean(yt)
                ),
            }
        )

    return pd.DataFrame(rows)


def validation_selection_score(
    metrics: pd.DataFrame,
) -> float:
    rows = metrics[
        metrics["task"].isin(BINARY_TASKS)
        & metrics["task"].isin(
            BINARY_TASKS[:4]
        )
    ].copy()

    # Only the four primary binary tasks enter selection.
    rows = rows[
        rows["task"].isin(
            BINARY_TASKS[:4]
        )
    ]

    if len(rows) != 4:
        return float("-inf")

    values = rows["auprc"].to_numpy(
        dtype=float
    )

    if not np.all(
        np.isfinite(values)
    ):
        return float("-inf")

    return float(
        np.mean(values)
    )


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------
def make_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
):
    return torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )


def make_scheduler(
    optimizer,
    scheduler_name: str,
    max_epochs: int,
    steps_per_epoch: int,
    max_lr: float,
):
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max_epochs,
        )

    if scheduler_name == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=max(
                1,
                steps_per_epoch * max_epochs,
            ),
            pct_start=0.1,
            anneal_strategy="cos",
            cycle_momentum=False,
        )

    raise ValueError(
        f"Unknown scheduler: {scheduler_name}"
    )


def train_one(
    model_id: str,
    params: Dict[str, Any],
    data: FrozenData,
    device: torch.device,
    mode: str,
    run_dir: Optional[Path] = None,
    trial_number: Optional[int] = None,
):
    model = build_model(
        model_id,
        params,
    ).to(device)

    batch_size = int(
        params.get("batch_size", 512)
    )
    max_epochs = int(
        params.get("max_epochs", 50)
    )
    patience = int(
        params.get("patience", 7)
    )
    lr = float(
        params["lr"]
    )
    weight_decay = float(
        params.get(
            "weight_decay",
            1e-4,
        )
    )
    grad_clip = float(
        params.get(
            "grad_clip",
            1.0,
        )
    )
    scheduler_name = params.get(
        "scheduler",
        "cosine",
    )

    train_loader = make_loader(
        data,
        data.train_idx,
        batch_size,
        True,
        device.type == "cuda",
    )
    val_loader = make_loader(
        data,
        data.val_idx,
        batch_size,
        False,
        device.type == "cuda",
    )

    optimizer = make_optimizer(
        model,
        lr,
        weight_decay,
    )

    scheduler = make_scheduler(
        optimizer,
        scheduler_name,
        max_epochs,
        len(train_loader),
        lr,
    )

    use_amp = amp_enabled(device)

    best_score = float("-inf")
    best_epoch = -1
    best_state = None
    best_val_metrics = None
    history_rows = []
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        seen = 0

        for x, demographics, y_dict, valid_dict in train_loader:
            x = x.to(
                device,
                non_blocking=True,
            )
            demographics = demographics.to(
                device,
                non_blocking=True,
            )

            targets = {
                task: y_dict[task].to(
                    device,
                    non_blocking=True,
                )
                for task in PRIMARY_TASKS
            }
            valids = {
                task: valid_dict[task].to(
                    device,
                    non_blocking=True,
                ).bool()
                for task in PRIMARY_TASKS
            }

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=use_amp,
            ):
                outputs = model(
                    x,
                    demographics,
                )
                loss, _ = multitask_loss(
                    outputs,
                    targets,
                    valids,
                )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                grad_clip,
            )
            optimizer.step()

            if (
                scheduler is not None
                and scheduler_name == "onecycle"
            ):
                scheduler.step()

            bs = x.shape[0]
            running_loss += (
                float(loss.detach().cpu().item())
                * bs
            )
            seen += bs

        if (
            scheduler is not None
            and scheduler_name == "cosine"
        ):
            scheduler.step()

        train_loss = (
            running_loss / max(seen, 1)
        )

        val_pred, val_valid = collect_predictions(
            model,
            val_loader,
            device,
            use_amp,
        )

        val_metrics = evaluate_predictions(
            data,
            val_pred,
            val_valid,
            data.val_idx,
            "val",
        )

        selection = validation_selection_score(
            val_metrics
        )

        los_row = val_metrics[
            val_metrics["task"]
            == "los_hours_24h"
        ]

        val_los_mae = (
            float(
                los_row["mae"].iloc[0]
            )
            if not los_row.empty
            else float("nan")
        )

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_selection_score": selection,
                "val_los_mae": val_los_mae,
                "lr": optimizer.param_groups[0]["lr"],
                "epoch_seconds": time.time() - epoch_start,
            }
        )

        print(
            f"[{model_id} {MODEL_NAMES[model_id]}] "
            f"epoch={epoch:02d} "
            f"train_loss={train_loss:.5f} "
            f"val_mean_AUPRC={selection:.5f} "
            f"val_LOS_MAE={val_los_mae:.3f}",
            flush=True,
        )

        if selection > best_score:
            best_score = selection
            best_epoch = epoch
            best_state = copy.deepcopy(
                model.state_dict()
            )
            best_val_metrics = val_metrics.copy()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(
                f"[{model_id}] early stopping at epoch {epoch}.",
                flush=True,
            )
            break

    if best_state is None:
        raise RuntimeError(
            "No valid checkpoint was selected."
        )

    model.load_state_dict(
        best_state
    )

    if run_dir is not None:
        run_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

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

        pd.DataFrame(
            history_rows
        ).to_csv(
            run_dir / "training_history.csv",
            index=False,
        )

        if best_val_metrics is not None:
            best_val_metrics.to_csv(
                run_dir / "best_validation_metrics.csv",
                index=False,
            )

    return (
        model,
        best_score,
        best_epoch,
        pd.DataFrame(history_rows),
    )


# ---------------------------------------------------------------------
# Prediction table writer
# ---------------------------------------------------------------------
@torch.no_grad()
def predict_and_write(
    model: nn.Module,
    data: FrozenData,
    device: torch.device,
    indices: np.ndarray,
    split_name: str,
    run_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    loader = make_loader(
        data,
        indices,
        int(
            run_dir.joinpath(
                "runtime_batch_size.txt"
            ).read_text()
        )
        if run_dir.joinpath(
            "runtime_batch_size.txt"
        ).exists()
        else 512,
        False,
        device.type == "cuda",
    )

    pred, valid = collect_predictions(
        model,
        loader,
        device,
        amp_enabled(device),
    )

    metric_df = evaluate_predictions(
        data,
        pred,
        valid,
        indices,
        split_name,
    )

    rows = []

    for task in PRIMARY_TASKS:
        y = data.targets[task][indices].astype(
            float,
            copy=False,
        )
        p = pred[task]
        v = valid[task].astype(int)

        for j, original_idx in enumerate(
            indices
        ):
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

    # Derived deterioration predictions.
    comp_valid = (
        valid["mortality_24h"].astype(bool)
        & valid["ventilation_onset_24h"].astype(bool)
        & valid["vasopressor_onset_24h"].astype(bool)
    )

    y_mort = data.outcomes[
        "mortality_24h"
    ].to_numpy(dtype=int)[indices]
    y_vent = data.outcomes[
        "ventilation_onset_24h"
    ].to_numpy(dtype=int)[indices]
    y_vaso = data.outcomes[
        "vasopressor_onset_24h"
    ].to_numpy(dtype=int)[indices]

    y_det = (
        1
        - (1 - y_mort)
        * (1 - y_vent)
        * (1 - y_vaso)
    ).astype(int)

    p_det = (
        1
        - (1 - pred["mortality_24h"])
        * (1 - pred["ventilation_onset_24h"])
        * (1 - pred["vasopressor_onset_24h"])
    )

    for j, original_idx in enumerate(
        indices
    ):
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

    return (
        metric_df,
        pd.DataFrame(rows),
    )


# ---------------------------------------------------------------------
# Fixed phase
# ---------------------------------------------------------------------
def run_fixed_model(
    model_id: str,
    data: FrozenData,
    device: torch.device,
):
    cfg = MODEL_CONFIGS[model_id]

    out_dir = (
        RESULTS_ROOT
        / "fixed"
        / f"{model_id}_{cfg.name.replace(' ', '_')}"
    )
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    params = {
        **cfg.fixed,
        "batch_size": 512,
        "max_epochs": 50,
        "patience": 7,
        "weight_decay": 1e-4,
        "grad_clip": 1.0,
        "scheduler": "cosine",
    }

    # Runtime batch-size contract for prediction writing.
    (
        out_dir / "runtime_batch_size.txt"
    ).write_text(
        "512",
        encoding="utf-8",
    )

    start = time.time()

    model, score, best_epoch, _ = train_one(
        model_id=model_id,
        params=params,
        data=data,
        device=device,
        mode="fixed",
        run_dir=out_dir,
    )

    val_metrics, val_preds = predict_and_write(
        model,
        data,
        device,
        data.val_idx,
        "val",
        out_dir,
    )

    test_metrics, test_preds = predict_and_write(
        model,
        data,
        device,
        data.test_idx,
        "test",
        out_dir,
    )

    all_metrics = pd.concat(
        [val_metrics, test_metrics],
        ignore_index=True,
    )

    all_preds = pd.concat(
        [val_preds, test_preds],
        ignore_index=True,
    )

    all_metrics.to_csv(
        out_dir / "metrics.csv",
        index=False,
    )
    all_preds.to_csv(
        out_dir / "test_predictions.csv",
        index=False,
    )

    metadata = {
        "model_id": model_id,
        "model_name": cfg.name,
        "mode": "fixed",
        "seed": SEED,
        "device": str(device),
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "hip_version": getattr(
            torch.version,
            "hip",
            None,
        ),
        "gpu_name": (
            torch.cuda.get_device_name(0)
            if device.type == "cuda"
            else None
        ),
        "input_shape": list(EXPECTED_SHAPE),
        "split_counts": {
            "train": int(len(data.train_idx)),
            "val": int(len(data.val_idx)),
            "test": int(len(data.test_idx)),
        },
        "fixed_registered_params": cfg.fixed,
        "global_training_params": params,
        "selection_metric": (
            "Mean validation AUPRC across the four primary binary tasks"
        ),
        "validation_selection_score": score,
        "best_epoch": best_epoch,
        "los_training_transform": "log1p(los_hours_24h - 24.0)",
        "los_reporting_transform": "expm1(excess) + 24.0 hours",
        "grud_channel_contract": {
            "values": "0:28",
            "masks": "28:56",
            "time_since": "56:84",
            "trends": "84:110 auxiliary",
        },
        "grud_reference_fidelity": {
            "input_decay": "diagonal/elementwise",
            "hidden_decay": "dense from 28 delta channels to hidden state",
            "learnable_x_mean": True,
            "reference_note": (
                "Input decay follows the diagonal W_gamma_x constraint "
                "described in the Che et al. GRU-D formulation."
            ),
        },
        "grud_preprocessing_context": (
            "Frozen V2 values are already forward-filled upstream; "
            "GRU-D consumes the frozen values together with explicit "
            "observation masks and time-since channels."
        ),
        "total_runtime_seconds": time.time() - start,
        "test_used_for_selection": False,
        "fixed_tuned_phase_separation": True,
    }

    (out_dir / "metadata.json").write_text(
        json.dumps(
            metadata,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    return {
        "model_id": model_id,
        "model_name": cfg.name,
        "mode": "fixed",
        "validation_selection_score": score,
        "best_epoch": best_epoch,
        "output_dir": str(out_dir),
    }


# ---------------------------------------------------------------------
# Tuned phase
# ---------------------------------------------------------------------
def tuned_params_from_trial(
    model_id: str,
    trial: "optuna.Trial",
) -> Dict[str, Any]:
    params = {
        "hidden": trial.suggest_int(
            "hidden",
            128,
            512,
        ),
        "dropout": trial.suggest_float(
            "dropout",
            0.10,
            0.35 if model_id != "09" else 0.30,
        ),
        "lr": trial.suggest_float(
            "lr",
            3e-4 if model_id != "09" else 5e-4,
            2e-3,
            log=True,
        ),
        "batch_size": trial.suggest_categorical(
            "batch_size",
            [256, 512, 1024],
        ),
        "patience": trial.suggest_int(
            "patience",
            5,
            10,
        ),
        "weight_decay": trial.suggest_float(
            "weight_decay",
            1e-5,
            1e-3,
            log=True,
        ),
        "grad_clip": trial.suggest_float(
            "grad_clip",
            0.5,
            2.0,
        ),
        "scheduler": trial.suggest_categorical(
            "scheduler",
            ["cosine", "onecycle"],
        ),
        "max_epochs": 50,
    }

    if model_id != "09":
        params["layers"] = trial.suggest_int(
            "layers",
            1,
            3,
        )
    else:
        params["layers"] = 1

    return params


def run_tuned_model(
    model_id: str,
    data: FrozenData,
    device: torch.device,
    n_trials: int,
):
    require_optuna()

    cfg = MODEL_CONFIGS[model_id]

    study_dir = (
        RESULTS_ROOT
        / "tuned"
        / f"{model_id}_{cfg.name.replace(' ', '_')}"
    )
    study_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    trial_rows = []

    def objective(
        trial: "optuna.Trial",
    ) -> float:
        params = tuned_params_from_trial(
            model_id,
            trial,
        )

        # Do not persist trial checkpoints.
        model, score, best_epoch, history = train_one(
            model_id=model_id,
            params=params,
            data=data,
            device=device,
            mode="tuned",
            run_dir=None,
            trial_number=trial.number,
        )

        trial.set_user_attr(
            "best_epoch",
            int(best_epoch),
        )

        trial.set_user_attr(
            "validation_mean_binary_auprc",
            float(score),
        )

        trial_rows.append(
            {
                "trial_number": trial.number,
                "value": float(score),
                "best_epoch": int(best_epoch),
                "params": json.dumps(
                    params,
                    sort_keys=True,
                ),
            }
        )

        # Explicit cleanup between trials.
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        return float(score)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=SEED
        ),
        study_name=(
            f"pipeline_a_v2_family02_{model_id}"
        ),
    )

    study.optimize(
        objective,
        n_trials=n_trials,
    )

    pd.DataFrame(
        trial_rows
    ).to_csv(
        study_dir / "optuna_trials.csv",
        index=False,
    )

    best_params = {
        **MODEL_CONFIGS[model_id].fixed,
    }

    best_trial_params = study.best_trial.params

    best_params.update(
        {
            "hidden": int(
                best_trial_params["hidden"]
            ),
            "dropout": float(
                best_trial_params["dropout"]
            ),
            "lr": float(
                best_trial_params["lr"]
            ),
            "batch_size": int(
                best_trial_params["batch_size"]
            ),
            "patience": int(
                best_trial_params["patience"]
            ),
            "weight_decay": float(
                best_trial_params["weight_decay"]
            ),
            "grad_clip": float(
                best_trial_params["grad_clip"]
            ),
            "scheduler": best_trial_params[
                "scheduler"
            ],
            "max_epochs": 50,
            "layers": (
                int(
                    best_trial_params["layers"]
                )
                if model_id != "09"
                else 1
            ),
        }
    )

    (study_dir / "best_params.json").write_text(
        json.dumps(
            best_params,
            indent=2,
        ),
        encoding="utf-8",
    )

    final_dir = (
        study_dir
        / "final_selected_configuration"
    )
    final_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        final_dir / "runtime_batch_size.txt"
    ).write_text(
        str(
            best_params["batch_size"]
        ),
        encoding="utf-8",
    )

    start = time.time()

    model, score, best_epoch, _ = train_one(
        model_id=model_id,
        params=best_params,
        data=data,
        device=device,
        mode="tuned",
        run_dir=final_dir,
    )

    val_metrics, val_preds = predict_and_write(
        model,
        data,
        device,
        data.val_idx,
        "val",
        final_dir,
    )

    test_metrics, test_preds = predict_and_write(
        model,
        data,
        device,
        data.test_idx,
        "test",
        final_dir,
    )

    all_metrics = pd.concat(
        [val_metrics, test_metrics],
        ignore_index=True,
    )

    all_preds = pd.concat(
        [val_preds, test_preds],
        ignore_index=True,
    )

    all_metrics.to_csv(
        final_dir / "metrics.csv",
        index=False,
    )
    all_preds.to_csv(
        final_dir / "test_predictions.csv",
        index=False,
    )

    metadata = {
        "model_id": model_id,
        "model_name": cfg.name,
        "mode": "tuned",
        "seed": SEED,
        "device": str(device),
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "hip_version": getattr(
            torch.version,
            "hip",
            None,
        ),
        "gpu_name": (
            torch.cuda.get_device_name(0)
            if device.type == "cuda"
            else None
        ),
        "input_shape": list(EXPECTED_SHAPE),
        "split_counts": {
            "train": int(len(data.train_idx)),
            "val": int(len(data.val_idx)),
            "test": int(len(data.test_idx)),
        },
        "fixed_registered_params": cfg.fixed,
        "registered_optuna_search_space": cfg.search_space,
        "best_optuna_params": best_params,
        "n_trials": int(n_trials),
        "best_trial_number": int(
            study.best_trial.number
        ),
        "best_trial_validation_score": float(
            study.best_value
        ),
        "selection_metric": (
            "Mean validation AUPRC across the four primary binary tasks"
        ),
        "final_validation_selection_score": float(
            score
        ),
        "best_epoch": int(best_epoch),
        "los_training_transform": "log1p(los_hours_24h - 24.0)",
        "los_reporting_transform": "expm1(excess) + 24.0 hours",
        "grud_channel_contract": {
            "values": "0:28",
            "masks": "28:56",
            "time_since": "56:84",
            "trends": "84:110 auxiliary",
        },
        "grud_reference_fidelity": {
            "input_decay": "diagonal/elementwise",
            "hidden_decay": "dense from 28 delta channels to hidden state",
            "learnable_x_mean": True,
            "reference_note": (
                "Input decay follows the diagonal W_gamma_x constraint "
                "described in the Che et al. GRU-D formulation."
            ),
        },
        "grud_preprocessing_context": (
            "Frozen V2 values are already forward-filled upstream; "
            "GRU-D consumes the frozen values together with explicit "
            "observation masks and time-since channels."
        ),
        "total_runtime_seconds": time.time() - start,
        "test_used_for_selection": False,
        "fixed_tuned_phase_separation": True,
    }

    (final_dir / "metadata.json").write_text(
        json.dumps(
            metadata,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "model_id": model_id,
        "model_name": cfg.name,
        "mode": "tuned",
        "validation_selection_score": float(
            score
        ),
        "best_trial_number": int(
            study.best_trial.number
        ),
        "output_dir": str(final_dir),
    }


# ---------------------------------------------------------------------
# Statistical post-hoc evaluation
# ---------------------------------------------------------------------
def safe_auroc(y, p):
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def safe_auprc(y, p):
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, p))


def percentile_ci(
    vals: np.ndarray,
    alpha: float = 0.05,
):
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return float("nan"), float("nan")
    return (
        float(
            np.percentile(
                vals,
                100 * alpha / 2,
            )
        ),
        float(
            np.percentile(
                vals,
                100 * (1 - alpha / 2),
            )
        ),
    )


def bootstrap_classification(
    y,
    p,
    n_boot=1000,
    seed=SEED,
):
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)

    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask]
    p = p[mask]

    n = len(y)

    if n < 10:
        return {}

    rng = np.random.default_rng(seed)

    auroc = np.empty(n_boot)
    auprc = np.empty(n_boot)
    brier = np.empty(n_boot)

    for i in range(n_boot):
        idx = rng.integers(
            0,
            n,
            size=n,
        )

        auroc[i] = safe_auroc(
            y[idx],
            p[idx],
        )
        auprc[i] = safe_auprc(
            y[idx],
            p[idx],
        )
        brier[i] = brier_score_loss(
            y[idx],
            p[idx],
        )

    alo, ahi = percentile_ci(auroc)
    plo, phi = percentile_ci(auprc)
    blo, bhi = percentile_ci(brier)

    return {
        "auroc": safe_auroc(y, p),
        "auroc_ci_low": alo,
        "auroc_ci_high": ahi,
        "auprc": safe_auprc(y, p),
        "auprc_ci_low": plo,
        "auprc_ci_high": phi,
        "brier": float(
            brier_score_loss(y, p)
        ),
        "brier_ci_low": blo,
        "brier_ci_high": bhi,
        "n": int(n),
        "prevalence": float(
            np.mean(y)
        ),
    }


def bootstrap_regression(
    y,
    p,
    n_boot=1000,
    seed=SEED,
):
    y = np.asarray(y).astype(float)
    p = np.asarray(p).astype(float)

    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask]
    p = p[mask]

    n = len(y)

    if n < 10:
        return {}

    rng = np.random.default_rng(seed)

    maes = np.empty(n_boot)
    rmses = np.empty(n_boot)
    medians = np.empty(n_boot)
    r2s = np.empty(n_boot)

    for i in range(n_boot):
        idx = rng.integers(
            0,
            n,
            size=n,
        )
        maes[i] = mean_absolute_error(
            y[idx],
            p[idx],
        )
        rmses[i] = np.sqrt(
            mean_squared_error(
                y[idx],
                p[idx],
            )
        )
        medians[i] = median_absolute_error(
            y[idx],
            p[idx],
        )
        r2s[i] = r2_score(
            y[idx],
            p[idx],
        )

    mlo, mhi = percentile_ci(
        maes
    )
    rlo, rhi = percentile_ci(
        rmses
    )
    dlo, dhi = percentile_ci(
        medians
    )
    r2lo, r2hi = percentile_ci(
        r2s
    )

    return {
        "mae": float(
            mean_absolute_error(y, p)
        ),
        "mae_ci_low": mlo,
        "mae_ci_high": mhi,
        "rmse": float(
            np.sqrt(
                mean_squared_error(
                    y,
                    p,
                )
            )
        ),
        "rmse_ci_low": rlo,
        "rmse_ci_high": rhi,
        "median_ae": float(
            median_absolute_error(
                y,
                p,
            )
        ),
        "median_ae_ci_low": dlo,
        "median_ae_ci_high": dhi,
        "r2": float(
            r2_score(y, p)
        ),
        "r2_ci_low": r2lo,
        "r2_ci_high": r2hi,
        "n": int(n),
    }


def expected_calibration_error(
    y,
    p,
    n_bins=10,
):
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)

    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask]
    p = p[mask]

    if len(y) == 0:
        return float("nan")

    edges = np.linspace(
        0,
        1,
        n_bins + 1,
    )

    ece = 0.0

    for i in range(n_bins):
        lo = edges[i]
        hi = edges[i + 1]

        if i < n_bins - 1:
            in_bin = (
                (p >= lo)
                & (p < hi)
            )
        else:
            in_bin = (
                (p >= lo)
                & (p <= hi)
            )

        if not np.any(in_bin):
            continue

        conf = float(
            np.mean(p[in_bin])
        )
        acc = float(
            np.mean(y[in_bin])
        )

        ece += (
            float(np.mean(in_bin))
            * abs(acc - conf)
        )

    return float(ece)


def evaluate_stats_mode() -> Path:
    rows = []

    for mode in (
        "fixed",
        "tuned",
    ):
        mode_dir = (
            RESULTS_ROOT
            / mode
        )

        if not mode_dir.exists():
            print(
                f"[STATS SKIP] {mode_dir} does not exist."
            )
            continue

        model_dirs = sorted(
            [
                p
                for p in mode_dir.iterdir()
                if p.is_dir()
            ]
        )

        for model_dir in model_dirs:
            if mode == "tuned":
                pred_path = (
                    model_dir
                    / "final_selected_configuration"
                    / "test_predictions.csv"
                )
                model_name = model_dir.name
            else:
                pred_path = (
                    model_dir
                    / "test_predictions.csv"
                )
                model_name = model_dir.name

            if not pred_path.exists():
                print(
                    f"[STATS SKIP] Missing: {pred_path}"
                )
                continue

            print(
                f"[STATS] {mode} / {model_name}"
            )

            df = pd.read_csv(
                pred_path
            )

            df = df[
                df["split"] == "test"
            ].copy()

            model_id = model_name.split("_")[0]

            for task, g in df.groupby(
                "task"
            ):
                if "validity" in g.columns:
                    g = g[
                        g["validity"] == 1
                    ].copy()

                y = g["target"].to_numpy()
                p = g["prediction"].to_numpy()

                row = {
                    "mode": mode,
                    "model_id": model_id,
                    "model_name": model_name,
                    "task": task,
                }

                if task in (
                    BINARY_TASKS
                    + ("deterioration_24h",)
                ):
                    row.update(
                        bootstrap_classification(
                            y,
                            p,
                            n_boot=1000,
                        )
                    )
                    row["ece"] = (
                        expected_calibration_error(
                            y,
                            p,
                        )
                    )
                else:
                    row.update(
                        bootstrap_regression(
                            y,
                            p,
                            n_boot=1000,
                        )
                    )
                    row["ece"] = float(
                        "nan"
                    )

                rows.append(row)

    if not rows:
        raise RuntimeError(
            "No Family 2 fixed/tuned test predictions were found."
        )

    result = pd.DataFrame(rows)

    out_dir = (
        RESULTS_ROOT / "stats"
    )
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    out_path = (
        out_dir
        / "family02_all_metrics_with_stats.csv"
    )

    result.to_csv(
        out_path,
        index=False,
    )

    print(
        f"Saved statistical results -> {out_path}"
    )

    return out_path


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def run_all_models(
    model_ids: Iterable[str],
    mode: str,
    n_trials: int,
):
    require_torch()
    set_seed(SEED)

    device = get_device()

    print(
        "=" * 80
    )
    print(
        "PIPELINE A V2 — FAMILY 2 RNN 06–09"
    )
    print(
        "=" * 80
    )
    print(
        f"Device       : {device}"
    )
    print(
        f"PyTorch      : {torch.__version__}"
    )
    print(
        f"HIP          : {getattr(torch.version, 'hip', None)}"
    )

    if device.type == "cuda":
        print(
            f"GPU          : {torch.cuda.get_device_name(0)}"
        )

    data = load_frozen_v2()

    print(
        f"Loaded structured={EXPECTED_SHAPE}; "
        f"train={len(data.train_idx)}, "
        f"val={len(data.val_idx)}, "
        f"test={len(data.test_idx)}",
        flush=True,
    )

    summaries = []

    for model_id in model_ids:
        if mode == "fixed":
            result = run_fixed_model(
                model_id,
                data,
                device,
            )
        elif mode == "tuned":
            result = run_tuned_model(
                model_id,
                data,
                device,
                n_trials,
            )
        else:
            raise ValueError(mode)

        summaries.append(result)

        RESULTS_ROOT.mkdir(
            parents=True,
            exist_ok=True,
        )

        pd.DataFrame(
            summaries
        ).to_csv(
            RESULTS_ROOT
            / f"family02_{mode}_summary.csv",
            index=False,
        )

        print(
            json.dumps(
                result,
                indent=2,
                default=str,
            ),
            flush=True,
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Pipeline A V2 Family 2: "
            "GRU, LSTM, BiGRU, GRU-D"
        )
    )

    parser.add_argument(
        "--model",
        default="all",
        choices=[
            "all",
            *MODEL_CONFIGS.keys(),
        ],
    )

    parser.add_argument(
        "--mode",
        required=True,
        choices=[
            "fixed",
            "tuned",
            "stats",
        ],
    )

    parser.add_argument(
        "--n-trials",
        type=int,
        default=30,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.n_trials < 1:
        raise ValueError(
            "--n-trials must be >= 1."
        )

    if args.mode == "stats":
        evaluate_stats_mode()
        return

    if args.model == "all":
        model_ids = list(
            MODEL_CONFIGS.keys()
        )
    else:
        model_ids = [args.model]

    run_all_models(
        model_ids=model_ids,
        mode=args.mode,
        n_trials=args.n_trials,
    )


if __name__ == "__main__":
    main()
