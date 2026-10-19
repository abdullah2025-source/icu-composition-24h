#!/usr/bin/env python3
"""
Pipeline A V2 — Full-X NextGen: Multi-Scale Task-Conditioned Hierarchical MoE

Single monolithic architecture for the frozen Pipeline A V2 cohort.

Design:
  1) Four structured streams (values / masks / time-since / trends), each projected
     to d_model and pooled at three temporal scales: full-24h attention, first-6h
     mean, last-6h mean -> 12 structured tokens.
  2) Frozen 768-d clinical text -> one token, exact-zero after projection when
     text is missing, plus a learned missing-modality token.
  3) Demographics -> one token.
  4) Explicit value-text interaction token: [v_full * t, |v_full - t|].
  5) 16 tokens are flattened -> 512-d shared patient representation.
  6) Shared sparse MoE: 6 experts, top-2 soft routing.
  7) Task-specific sparse MoE banks: 2 experts/task, top-1 after a 3-epoch
     top-2 warm-up; task routers are conditioned on [shared_input; shared_out].
  8) Five task heads: mortality, ventilation, vasopressor, readmission, LOS.
  9) LOS is mortality-conditioned survivor/death mixture.
 10) Loss = uncertainty-weighted sum of the five masked task losses + sparse
     load-balance auxiliary loss.

Frozen model-selection rule:
  mean validation AUPRC across mortality, ventilation, vasopressor, readmission.

Test data are reserved for final evaluation and never used for checkpoint selection.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import optuna
except ImportError:
    optuna = None

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
    roc_auc_score,
)

PROJECT_ROOT = Path("/home/abdullah2025/fyp_project")
V2_DIR = PROJECT_ROOT / "outputs" / "pipeline_a_v2"
RESULTS_ROOT = PROJECT_ROOT / "outputs" / "baseline_family_07_advanced_multimodal_28_32"
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

MODEL_ID = "33"
MODEL_NAME = "Full-X"
FIXED_PARAMS = {
    "d_model": 256,
    "n_shared_experts": 6,
    "n_task_experts": 2,
    "top_k_shared": 2,
    "top_k_task": 1,
    "task_warmup_top_k": 2,
    "task_warmup_epochs": 3,
    "expert_expand": 2,
    "load_balance_coef": 0.01,
    "dropout": 0.15,
    "lr": 6e-4,
    "batch_size": 256,
    "max_epochs": 50,
    "patience": 8,
    "weight_decay": 1e-4,
    "grad_clip": 1.0,
    "scheduler": "cosine",
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
    targets: Dict[str, np.ndarray]
    valid_arrays: Dict[str, np.ndarray]
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def stable_seed(label: str) -> int:
    digest = hashlib.sha256(f"{SEED}|{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32 - 1)


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def validate_gpu(device: torch.device) -> None:
    if device.type != "cuda":
        raise RuntimeError(f"Full-X requires CUDA/ROCm GPU training; got {device}")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Full-X requires BF16-capable GPU; torch.cuda.is_bf16_supported() is False")


def autocast_context(device: torch.device):
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda"))


# ---------------------------------------------------------------------------
# Frozen V2 data
# ---------------------------------------------------------------------------
def load_text_embeddings(stay_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    require_file(TEXT_EMBEDDING_PATH)
    emb = pd.read_parquet(TEXT_EMBEDDING_PATH)
    required = {"stay_id", "text_embedding"}
    missing = required - set(emb.columns)
    if missing:
        raise KeyError(f"Text parquet missing columns: {sorted(missing)}")
    if emb["stay_id"].duplicated().any():
        raise ValueError("Duplicate stay_id in text embeddings")
    row_of = pd.Series(np.arange(len(stay_ids), dtype=np.int64), index=stay_ids)
    present = emb[emb["stay_id"].isin(row_of.index)].copy()
    text = np.zeros((len(stay_ids), EXPECTED_TEXT_DIM), dtype=np.float32)
    missing_flag = np.ones(len(stay_ids), dtype=np.float32)
    if len(present):
        rows = row_of.loc[present["stay_id"]].to_numpy(dtype=np.int64)
        vectors = []
        for vec in present["text_embedding"].tolist():
            arr = np.asarray(vec, dtype=np.float32).reshape(-1)
            if arr.size != EXPECTED_TEXT_DIM:
                raise ValueError(f"Expected text dim {EXPECTED_TEXT_DIM}, got {arr.size}")
            vectors.append(arr)
        text[rows] = np.stack(vectors)
        missing_flag[rows] = 0.0
    if int((missing_flag == 0).sum()) != EXPECTED_TEXT_PRESENT:
        raise ValueError(
            f"Expected {EXPECTED_TEXT_PRESENT} text-bearing stays, found {int((missing_flag == 0).sum())}"
        )
    if not np.isfinite(text).all():
        raise ValueError("Non-finite text embedding values")
    return text, missing_flag


def load_frozen_v2() -> FrozenData:
    for p in (
        ALIGNMENT_PATH,
        STRUCTURED_PATH,
        DEMOGRAPHICS_PATH,
        INPUT_METADATA_PATH,
        INPUT_DEFINITION_PATH,
        OUTCOMES_PATH,
        TEXT_EMBEDDING_PATH,
    ):
        require_file(p)

    idx_df = pd.read_csv(ALIGNMENT_PATH)
    out_df = pd.read_csv(OUTCOMES_PATH)
    required_idx = {"stay_id", "subject_id", "split_24h"}
    missing_idx = required_idx - set(idx_df.columns)
    if missing_idx:
        raise KeyError(f"Alignment missing: {sorted(missing_idx)}")
    required_out = {"stay_id", "subject_id", *REPORT_TASKS, *VALID_COL.values()}
    missing_out = required_out - set(out_df.columns)
    if missing_out:
        raise KeyError(f"Outcomes missing: {sorted(missing_out)}")

    keep_cols = ["stay_id", "subject_id", *REPORT_TASKS]
    keep_cols += [c for c in out_df.columns if c.endswith("_valid")]
    out_small = out_df[list(dict.fromkeys(keep_cols))].copy()
    if out_small["stay_id"].duplicated().any():
        raise ValueError("Duplicate stay_id in outcomes")

    alignment_core = idx_df[["stay_id", "subject_id", "split_24h"]].copy()
    merged = alignment_core.merge(
        out_small,
        on=["stay_id", "subject_id"],
        how="left",
        validate="one_to_one",
    )

    structured = np.load(STRUCTURED_PATH, mmap_mode="r")
    demographics = np.load(DEMOGRAPHICS_PATH, mmap_mode="r")
    if tuple(structured.shape) != EXPECTED_STRUCTURED_SHAPE:
        raise ValueError(f"Structured shape {structured.shape} != {EXPECTED_STRUCTURED_SHAPE}")
    if tuple(demographics.shape) != EXPECTED_DEMO_SHAPE:
        raise ValueError(f"Demographics shape {demographics.shape} != {EXPECTED_DEMO_SHAPE}")
    if len(merged) != EXPECTED_N:
        raise ValueError(f"Rows {len(merged)} != {EXPECTED_N}")

    split = merged["split_24h"].astype(str).to_numpy()
    expected_split = {"train": EXPECTED_TRAIN, "val": EXPECTED_VAL, "test": EXPECTED_TEST}
    actual_split = pd.Series(split).value_counts().to_dict()
    if actual_split != expected_split:
        raise ValueError(f"Split counts {actual_split} != {expected_split}")

    train_idx = np.flatnonzero(split == "train")
    val_idx = np.flatnonzero(split == "val")
    test_idx = np.flatnonzero(split == "test")
    targets = {t: merged[t].to_numpy(dtype=np.float32) for t in PRIMARY_TASKS}
    valid_arrays = {
        t: merged[VALID_COL[t]].to_numpy(dtype=np.int8).astype(bool) for t in PRIMARY_TASKS
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
        targets=targets,
        valid_arrays=valid_arrays,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )


class FrozenDataset(Dataset):
    def __init__(self, data: FrozenData, indices: np.ndarray):
        self.data = data
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, pos: int):
        i = int(self.indices[pos])
        x = torch.from_numpy(np.array(self.data.structured[i], dtype=np.float32, copy=True))
        d = torch.from_numpy(np.array(self.data.demographics[i], dtype=np.float32, copy=True))
        txt = torch.from_numpy(np.array(self.data.text[i], dtype=np.float32, copy=True))
        missing = torch.tensor(float(self.data.text_missing[i]), dtype=torch.float32)
        y = {t: torch.tensor(float(self.data.targets[t][i]), dtype=torch.float32) for t in PRIMARY_TASKS}
        v = {t: torch.tensor(bool(self.data.valid_arrays[t][i]), dtype=torch.bool) for t in PRIMARY_TASKS}
        return x, d, txt, missing, y, v


def make_loader(data: FrozenData, indices: np.ndarray, batch_size: int, shuffle: bool, device: torch.device):
    return DataLoader(
        FrozenDataset(data, indices),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )


# ---------------------------------------------------------------------------
# Full-X model
# ---------------------------------------------------------------------------
def split_modalities(x: torch.Tensor):
    return x[..., :28], x[..., 28:56], x[..., 56:84], x[..., 84:110]


class ModalityProjector(nn.Module):
    def __init__(self, d: int, dropout: float):
        super().__init__()
        self.value = nn.Sequential(nn.Linear(28, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(dropout))
        self.mask = nn.Sequential(nn.Linear(28, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(dropout))
        self.time = nn.Sequential(nn.Linear(28, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(dropout))
        self.trend = nn.Sequential(nn.Linear(26, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(dropout))

    def forward(self, x):
        v, m, t, tr = split_modalities(x)
        return self.value(v), self.mask(m), self.time(t), self.trend(tr)


class MultiScalePool(nn.Module):
    """Full-24h learned attention + fixed first-6h mean + last-6h mean."""
    def __init__(self, d: int):
        super().__init__()
        self.pos = nn.Parameter(torch.randn(24, d) * 0.02)
        self.score = nn.Sequential(nn.Linear(d, max(32, d // 2)), nn.Tanh(), nn.Linear(max(32, d // 2), 1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = x + self.pos[:x.shape[1]].to(dtype=x.dtype, device=x.device)
        scores = self.score(z).float().squeeze(-1)
        weights = torch.softmax(scores, dim=-1).to(dtype=z.dtype)
        full = (z * weights.unsqueeze(-1)).sum(dim=1)
        first = x[:, :6].mean(dim=1)
        last = x[:, -6:].mean(dim=1)
        return full, first, last


class FullXText(nn.Module):
    def __init__(self, d: int, dropout: float):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(EXPECTED_TEXT_DIM, d),
            nn.LayerNorm(d),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.missing_embed = nn.Parameter(torch.zeros(d))
        nn.init.normal_(self.missing_embed, mean=0.0, std=0.02)

    def forward(self, text: torch.Tensor, missing: torch.Tensor):
        t = self.proj(text)
        m = missing.unsqueeze(-1).to(dtype=t.dtype)
        t = t * (1.0 - m)  # exact-zero text token when note is absent
        missing_token = self.missing_embed.unsqueeze(0).to(dtype=t.dtype) * m
        return t, missing_token


class ExpertMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class SparseMoE(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, n_experts: int, top_k: int, hidden_dim: int, dropout: float):
        super().__init__()
        if top_k > n_experts:
            raise ValueError("top_k cannot exceed n_experts")
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_experts = n_experts
        self.top_k = top_k
        self.router = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, n_experts),
        )
        self.experts = nn.ModuleList([
            ExpertMLP(in_dim, out_dim, hidden_dim, dropout) for _ in range(n_experts)
        ])
        self.last_aux_loss = torch.tensor(0.0)

    def forward(self, x: torch.Tensor, top_k: Optional[int] = None):
        k = int(top_k if top_k is not None else self.top_k)
        k = max(1, min(k, self.n_experts))
        logits = self.router(x)
        probs = torch.softmax(logits.float(), dim=-1)
        top_probs, top_idx = torch.topk(probs, k=k, dim=-1)
        if k > 1:
            top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        # Compute all experts for stable, transparent first-run implementation.
        expert_out = torch.stack([expert(x) for expert in self.experts], dim=1)  # [B,E,D]
        gather_idx = top_idx.unsqueeze(-1).expand(-1, -1, self.out_dim)
        selected = torch.gather(expert_out, 1, gather_idx)
        out = (selected * top_probs.unsqueeze(-1).to(dtype=selected.dtype)).sum(dim=1)

        load = torch.zeros(self.n_experts, device=x.device, dtype=probs.dtype)
        load.scatter_add_(0, top_idx.reshape(-1), torch.ones(top_idx.numel(), device=x.device, dtype=probs.dtype))
        load = load / float(max(1, top_idx.numel()))
        importance = probs.mean(dim=0)
        self.last_aux_loss = self.n_experts * torch.sum(importance * load)
        return out, self.last_aux_loss


class LOSHead(nn.Module):
    def __init__(self, d: int, dropout: float):
        super().__init__()
        self.survivor = nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 1))
        self.death = nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 1))

    def forward(self, h: torch.Tensor, p_mort: torch.Tensor):
        mu_surv = F.softplus(self.survivor(h).squeeze(-1))
        mu_death = F.softplus(self.death(h).squeeze(-1))
        return (1.0 - p_mort) * mu_surv + p_mort * mu_death


class TaskBinaryHead(nn.Module):
    def __init__(self, d: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class TaskConditionedHierarchicalMoE(nn.Module):
    def __init__(self, p: Dict[str, Any]):
        super().__init__()
        d = int(p["d_model"])
        drop = float(p["dropout"])
        self.d = d
        self.current_epoch = 0
        self.top_k_task = int(p["top_k_task"])
        self.task_warmup_top_k = int(p["task_warmup_top_k"])
        self.task_warmup_epochs = int(p["task_warmup_epochs"])
        self.n_tasks = len(PRIMARY_TASKS)

        self.modality = ModalityProjector(d, drop)
        self.pool = MultiScalePool(d)
        self.text = FullXText(d, drop)
        self.demo = nn.Sequential(nn.Linear(4, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(drop))
        self.interaction = nn.Sequential(
            nn.Linear(2 * d, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(drop)
        )

        # 12 structured + text + demo + interaction + missing = 16 tokens.
        token_dim = 16 * d
        self.shared_input = nn.Sequential(
            nn.Linear(token_dim, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.GELU(),
        )

        self.shared_moe = SparseMoE(
            in_dim=512,
            out_dim=d,
            n_experts=int(p["n_shared_experts"]),
            top_k=int(p["top_k_shared"]),
            hidden_dim=int(p["expert_expand"]) * d,
            dropout=drop,
        )

        task_input_dim = 512 + d
        self.task_moes = nn.ModuleDict({
            task: SparseMoE(
                in_dim=task_input_dim,
                out_dim=d,
                n_experts=int(p["n_task_experts"]),
                top_k=int(p["top_k_task"]),
                hidden_dim=int(p["expert_expand"]) * d,
                dropout=drop,
            )
            for task in PRIMARY_TASKS
        })
        self.task_norms = nn.ModuleDict({task: nn.LayerNorm(d) for task in PRIMARY_TASKS})

        self.mortality_head = TaskBinaryHead(d, drop)
        self.ventilation_head = TaskBinaryHead(d, drop)
        self.vasopressor_head = TaskBinaryHead(d, drop)
        self.readmission_head = TaskBinaryHead(d, drop)
        self.los_head = LOSHead(d, drop)
        self.log_vars = nn.Parameter(torch.zeros(5))
        self.last_aux_loss = torch.tensor(0.0)
        self.last_shared_aux = torch.tensor(0.0)
        self.last_task_aux = torch.tensor(0.0)

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    def active_task_top_k(self) -> int:
        if self.current_epoch <= self.task_warmup_epochs:
            return self.task_warmup_top_k
        return self.top_k_task

    def _structured_tokens(self, x: torch.Tensor):
        projected = self.modality(x)
        tokens = []
        pooled_full = []
        for stream in projected:
            full, first6, last6 = self.pool(stream)
            pooled_full.append(full)
            tokens.extend([full, first6, last6])
        return tokens, pooled_full[0]

    def forward(self, x, demographics, text, text_missing):
        structured_tokens, v_full = self._structured_tokens(x)
        t_text, missing_token = self.text(text, text_missing)
        d_token = self.demo(demographics)
        interaction_token = self.interaction(
            torch.cat([v_full * t_text, torch.abs(v_full - t_text)], dim=-1)
        )

        tokens = torch.cat(
            [*structured_tokens, t_text, d_token, interaction_token, missing_token], dim=-1
        )
        shared_input = self.shared_input(tokens)
        shared_out, shared_aux = self.shared_moe(shared_input, top_k=int(self.shared_moe.top_k))
        route_input = torch.cat([shared_input, shared_out], dim=-1)
        task_reps = {}
        task_aux = []
        top_k_task = self.active_task_top_k()
        for task in PRIMARY_TASKS:
            out, aux = self.task_moes[task](route_input, top_k=top_k_task)
            task_reps[task] = self.task_norms[task](out + shared_out)
            task_aux.append(aux)

        n_shared = max(1, int(self.shared_moe.n_experts))
        n_task = max(1, int(self.task_moes[PRIMARY_TASKS[0]].n_experts))
        self.last_shared_aux = shared_aux / float(n_shared)
        self.last_task_aux = torch.stack(task_aux).mean() / float(n_task)
        self.last_aux_loss = self.last_shared_aux + self.last_task_aux

        mort_logit = self.mortality_head(task_reps["mortality_24h"])
        p_mort = torch.sigmoid(mort_logit)
        return {
            "mortality_24h": mort_logit,
            "ventilation_onset_24h": self.ventilation_head(task_reps["ventilation_onset_24h"]),
            "vasopressor_onset_24h": self.vasopressor_head(task_reps["vasopressor_onset_24h"]),
            "readmission_30d_24h": self.readmission_head(task_reps["readmission_30d_24h"]),
            "los_excess_log1p": self.los_head(task_reps["los_hours_24h"], p_mort),
        }


# ---------------------------------------------------------------------------
# Losses / metrics
# ---------------------------------------------------------------------------
def task_losses(model: TaskConditionedHierarchicalMoE, outputs, targets, valid):
    losses = {}
    for task in BINARY_TASKS:
        m = valid[task]
        losses[task] = (
            F.binary_cross_entropy_with_logits(outputs[task][m], targets[task][m].float())
            if torch.any(m) else outputs[task].sum() * 0.0
        )
    m = valid["los_hours_24h"]
    if torch.any(m):
        target_log = torch.log1p(torch.clamp(targets["los_hours_24h"][m].float() - 24.0, min=0.0))
        losses["los_hours_24h"] = torch.abs(outputs["los_excess_log1p"][m].float() - target_log).mean()
    else:
        losses["los_hours_24h"] = outputs["los_excess_log1p"].sum() * 0.0
    ordered = torch.stack([losses[t] for t in PRIMARY_TASKS])
    total = torch.sum(torch.exp(-model.log_vars) * ordered + model.log_vars)
    return total, losses


def safe_auprc(y, p):
    if len(y) == 0 or len(np.unique(y)) < 2:
        return np.nan
    return float(average_precision_score(y.astype(int), p))


def safe_auroc(y, p):
    if len(y) == 0 or len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y.astype(int), p))


def evaluate_predictions(data: FrozenData, indices: np.ndarray, pred: Dict[str, np.ndarray], split: str):
    rows = []
    for task in PRIMARY_TASKS:
        y = data.targets[task][indices].astype(float)
        p = pred[task].astype(float)
        valid = data.valid_arrays[task][indices]
        keep = valid & np.isfinite(y) & np.isfinite(p)
        yy, pp = y[keep], p[keep]
        if task in BINARY_TASKS:
            rows.append({
                "split": split, "task": task, "valid_n": int(len(yy)),
                "auprc": safe_auprc(yy, pp), "auroc": safe_auroc(yy, pp),
                "brier": float(brier_score_loss(yy.astype(int), pp)) if len(yy) else np.nan,
                "prevalence": float(np.mean(yy)) if len(yy) else np.nan,
            })
        else:
            rows.append({
                "split": split, "task": task, "valid_n": int(len(yy)),
                "mae": float(mean_absolute_error(yy, pp)) if len(yy) else np.nan,
                "rmse": float(np.sqrt(mean_squared_error(yy, pp))) if len(yy) else np.nan,
                "median_ae": float(median_absolute_error(yy, pp)) if len(yy) else np.nan,
                "r2": float(r2_score(yy, pp)) if len(yy) > 1 else np.nan,
            })

    y_m = data.targets["mortality_24h"][indices].astype(float)
    y_v = data.targets["ventilation_onset_24h"][indices].astype(float)
    y_p = data.targets["vasopressor_onset_24h"][indices].astype(float)
    valid_det = (
        data.valid_arrays["mortality_24h"][indices]
        & data.valid_arrays["ventilation_onset_24h"][indices]
        & data.valid_arrays["vasopressor_onset_24h"][indices]
    )
    y_det = 1.0 - (1.0 - y_m) * (1.0 - y_v) * (1.0 - y_p)
    p_det = 1.0 - (
        (1.0 - pred["mortality_24h"])
        * (1.0 - pred["ventilation_onset_24h"])
        * (1.0 - pred["vasopressor_onset_24h"])
    )
    keep = valid_det & np.isfinite(y_det) & np.isfinite(p_det)
    rows.append({
        "split": split, "task": "deterioration_24h", "valid_n": int(np.sum(keep)),
        "auprc": safe_auprc(y_det[keep], p_det[keep]), "auroc": safe_auroc(y_det[keep], p_det[keep]),
        "brier": float(brier_score_loss(y_det[keep].astype(int), p_det[keep])) if np.any(keep) else np.nan,
        "prevalence": float(np.mean(y_det[keep])) if np.any(keep) else np.nan,
    })
    return pd.DataFrame(rows)


def validation_score(metrics: pd.DataFrame) -> float:
    sub = metrics[metrics["task"].isin(BINARY_TASKS)]
    if len(sub) != 4:
        return float("-inf")
    vals = sub["auprc"].to_numpy(dtype=float)
    return float(np.mean(vals)) if np.all(np.isfinite(vals)) else float("-inf")


@torch.no_grad()
def predict(model, data, indices, device, batch_size):
    model.eval()
    loader = make_loader(data, indices, batch_size, False, device)
    out = {t: [] for t in PRIMARY_TASKS}
    for x, d, txt, missing, _, _ in loader:
        x = x.to(device, non_blocking=True)
        d = d.to(device, non_blocking=True)
        txt = txt.to(device, non_blocking=True)
        missing = missing.to(device, non_blocking=True)
        with autocast_context(device):
            outputs = model(x, d, txt, missing)
        for task in BINARY_TASKS:
            out[task].append(torch.sigmoid(outputs[task]).float().cpu().numpy())
        out["los_hours_24h"].append((24.0 + torch.expm1(outputs["los_excess_log1p"].float())).cpu().numpy())
    return {t: np.concatenate(out[t]) for t in PRIMARY_TASKS}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_model(data: FrozenData, device: torch.device, params: Dict[str, Any], run_dir: Optional[Path] = None):
    set_seed(SEED)
    model = TaskConditionedHierarchicalMoE(params).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(params["lr"]), weight_decay=float(params["weight_decay"]))
    steps = math.ceil(len(data.train_idx) / int(params["batch_size"]))
    if params["scheduler"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(params["max_epochs"]))
    elif params["scheduler"] == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=float(params["lr"]), total_steps=max(1, steps * int(params["max_epochs"])),
            pct_start=0.1, anneal_strategy="cos", cycle_momentum=False
        )
    else:
        raise ValueError(params["scheduler"])

    best_state = None
    best_score = -np.inf
    best_epoch = 0
    no_improve = 0
    history = []
    use_amp = device.type == "cuda"

    for epoch in range(1, int(params["max_epochs"]) + 1):
        model.set_epoch(epoch)
        model.train()
        start = time.time()
        train_loader = make_loader(data, data.train_idx, int(params["batch_size"]), True, device)
        running = 0.0
        running_shared_aux = 0.0
        running_task_aux = 0.0
        seen = 0
        for x, d, txt, missing, y_cpu, v_cpu in train_loader:
            x = x.to(device, non_blocking=True)
            d = d.to(device, non_blocking=True)
            txt = txt.to(device, non_blocking=True)
            missing = missing.to(device, non_blocking=True)
            y = {t: y_cpu[t].to(device, non_blocking=True) for t in PRIMARY_TASKS}
            valid = {t: v_cpu[t].to(device, non_blocking=True).bool() for t in PRIMARY_TASKS}
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device):
                outputs = model(x, d, txt, missing)
                loss, _ = task_losses(model, outputs, y, valid)
                loss = loss + float(params["load_balance_coef"]) * model.last_aux_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(params["grad_clip"]))
            optimizer.step()
            if params["scheduler"] == "onecycle":
                scheduler.step()
            bs = x.shape[0]
            running += float(loss.detach().cpu()) * bs
            running_shared_aux += float(model.last_shared_aux.detach().cpu()) * bs
            running_task_aux += float(model.last_task_aux.detach().cpu()) * bs
            seen += bs
        if params["scheduler"] == "cosine":
            scheduler.step()

        val_pred = predict(model, data, data.val_idx, device, int(params["batch_size"]))
        val_metrics = evaluate_predictions(data, data.val_idx, val_pred, "val")
        score = validation_score(val_metrics)
        los = val_metrics[val_metrics["task"] == "los_hours_24h"]
        los_mae = float(los["mae"].iloc[0]) if not los.empty else np.nan
        history.append({
            "epoch": epoch,
            "train_loss": running / max(seen, 1),
            "val_mean_AUPRC": score,
            "val_LOS_MAE": los_mae,
            "task_top_k": model.active_task_top_k(),
            "shared_aux_mean": running_shared_aux / max(seen, 1),
            "task_aux_mean": running_task_aux / max(seen, 1),
            "epoch_aux_mean": (running_shared_aux + running_task_aux) / max(seen, 1),
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_seconds": time.time() - start,
        })
        print(
            f"[Full-X] epoch={epoch:02d} topk_task={model.active_task_top_k()} "
            f"train_loss={history[-1]['train_loss']:.6f} val_mean_AUPRC={score:.6f} "
            f"val_LOS_MAE={los_mae:.3f} aux={history[-1]['epoch_aux_mean']:.4f}",
            flush=True,
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= int(params["patience"]):
            print(f"[Full-X] early stopping at epoch {epoch}", flush=True)
            break

    if best_state is None:
        raise RuntimeError("Full-X produced no valid checkpoint")
    model.load_state_dict(best_state)
    model.set_epoch(best_epoch)
    history_df = pd.DataFrame(history)
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model": MODEL_NAME,
            "model_id": MODEL_ID,
            "params": params,
            "best_epoch": best_epoch,
            "best_val_selection_score": best_score,
            "state_dict": model.state_dict(),
        }, run_dir / "checkpoint.pt")
        history_df.to_csv(run_dir / "training_history.csv", index=False)
    return model, best_score, best_epoch, history_df


# ---------------------------------------------------------------------------
# Outputs / bootstrap
# ---------------------------------------------------------------------------
def subject_bootstrap_ci(data: FrozenData, indices, pred, task: str, reps: int = 2000):
    y = data.targets[task][indices].astype(float)
    p = pred[task].astype(float)
    valid = data.valid_arrays[task][indices]
    keep = valid & np.isfinite(y) & np.isfinite(p)
    y, p = y[keep], p[keep]
    subjects = data.subject_id[indices][keep]
    if len(y) == 0:
        return {}
    unique = np.unique(subjects)
    rows_by_subj = {s: np.flatnonzero(subjects == s) for s in unique}
    rng = np.random.default_rng(stable_seed(task))
    auprc_vals, auroc_vals = [], []
    mae_vals = []
    for _ in range(int(reps)):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([rows_by_subj[s] for s in sampled])
        if task in BINARY_TASKS:
            a = safe_auprc(y[idx], p[idx]); r = safe_auroc(y[idx], p[idx])
            if np.isfinite(a): auprc_vals.append(a)
            if np.isfinite(r): auroc_vals.append(r)
        else:
            mae_vals.append(mean_absolute_error(y[idx], p[idx]))
    if task in BINARY_TASKS:
        point = {"auprc": safe_auprc(y, p), "auroc": safe_auroc(y, p)}
        if auprc_vals:
            point["auprc_ci95_low"] = float(np.quantile(auprc_vals, .025)); point["auprc_ci95_high"] = float(np.quantile(auprc_vals, .975))
        if auroc_vals:
            point["auroc_ci95_low"] = float(np.quantile(auroc_vals, .025)); point["auroc_ci95_high"] = float(np.quantile(auroc_vals, .975))
        point["bootstrap_unit"] = "subject_id"; point["bootstrap_subjects"] = len(unique)
        return point
    point = {"mae": float(mean_absolute_error(y, p))}
    if mae_vals:
        point["mae_ci95_low"] = float(np.quantile(mae_vals, .025)); point["mae_ci95_high"] = float(np.quantile(mae_vals, .975))
    point["bootstrap_unit"] = "subject_id"; point["bootstrap_subjects"] = len(unique)
    return point


def save_outputs(data: FrozenData, model: nn.Module, device: torch.device, params: Dict[str, Any], score: float, best_epoch: int, run_dir: Path, bootstrap_reps: int):
    val_pred = predict(model, data, data.val_idx, device, int(params["batch_size"]))
    test_pred = predict(model, data, data.test_idx, device, int(params["batch_size"]))
    val_metrics = evaluate_predictions(data, data.val_idx, val_pred, "val")
    test_metrics = evaluate_predictions(data, data.test_idx, test_pred, "test")
    metrics = pd.concat([val_metrics, test_metrics], ignore_index=True)
    metrics.to_csv(run_dir / "metrics.csv", index=False)

    rows = []
    for split, indices, pred in (("val", data.val_idx, val_pred), ("test", data.test_idx, test_pred)):
        for task in PRIMARY_TASKS:
            y = data.targets[task][indices]
            p = pred[task]
            v = data.valid_arrays[task][indices]
            for j, original_idx in enumerate(indices):
                rows.append({
                    "stay_id": data.stay_id[original_idx],
                    "subject_id": data.subject_id[original_idx],
                    "split": split,
                    "task": task,
                    "target": float(y[j]),
                    "validity": int(v[j]),
                    "prediction": float(p[j]),
                })
    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(run_dir / "predictions.csv", index=False)
    pred_df[pred_df["split"] == "test"].to_csv(run_dir / "test_predictions.csv", index=False)

    ci_rows = []
    for split, indices, pred in (("val", data.val_idx, val_pred), ("test", data.test_idx, test_pred)):
        for task in PRIMARY_TASKS:
            ci = subject_bootstrap_ci(data, indices, pred, task, reps=bootstrap_reps)
            ci_rows.append({"split": split, "task": task, **ci})
    pd.DataFrame(ci_rows).to_csv(run_dir / "subject_clustered_bootstrap_cis.csv", index=False)

    test_bin = test_metrics[test_metrics["task"].isin(BINARY_TASKS)]
    val_bin = val_metrics[val_metrics["task"].isin(BINARY_TASKS)]
    los_v = val_metrics[val_metrics["task"] == "los_hours_24h"]
    los_t = test_metrics[test_metrics["task"] == "los_hours_24h"]
    summary = {
        "version": "X",
        "model": MODEL_NAME,
        "validation_mean_auprc": float(val_bin["auprc"].mean()),
        "validation_mean_auroc": float(val_bin["auroc"].mean()),
        "test_mean_auprc": float(test_bin["auprc"].mean()),
        "test_mean_auroc": float(test_bin["auroc"].mean()),
        "validation_los_mae": float(los_v["mae"].iloc[0]),
        "test_los_mae": float(los_t["mae"].iloc[0]),
        "best_epoch": int(best_epoch),
        "validation_selection_score": float(score),
        "parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "params": params,
        "test_used_for_selection": False,
    }
    (run_dir / "final_result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (run_dir / "architecture_manifest.json").write_text(json.dumps({
        "model": MODEL_NAME,
        "model_id": MODEL_ID,
        "architecture": "TaskConditionedHierarchicalMoE",
        "structured": "four modality projections x three windows (attention full-24h, mean first-6h, mean last-6h)",
        "tokens": "12 structured + text + demographics + value-text interaction + learned missing-text token = 16",
        "interaction": "[v_full * t_text, abs(v_full - t_text)]",
        "shared_bank": "6 experts, top-2 soft routing",
        "task_banks": "2 experts per primary task, task-conditioned on [shared_input; shared_out]; top-2 for first 3 epochs then top-1",
        "heads": "independent binary towers + mortality-conditioned survivor/death LOS head",
        "loss": "uncertainty-weighted sum of five masked losses + load-balance auxiliary loss",
        "selection": "mean validation AUPRC across four binary tasks",
        "frozen_contract": [74572, 24, 110],
    }, indent=2), encoding="utf-8")
    return summary


# ---------------------------------------------------------------------------
# Optuna
# ---------------------------------------------------------------------------
def suggest_params(trial):
    p = {
        "d_model": trial.suggest_categorical("d_model", [192, 224, 256, 288, 320, 352, 384]),
        "n_shared_experts": trial.suggest_categorical("n_shared_experts", [4, 6, 8]),
        "n_task_experts": trial.suggest_categorical("n_task_experts", [2, 3]),
        "top_k_shared": trial.suggest_categorical("top_k_shared", [1, 2, 3]),
        "top_k_task": trial.suggest_categorical("top_k_task", [1, 2]),
        "expert_expand": trial.suggest_categorical("expert_expand", [2, 4]),
        "load_balance_coef": trial.suggest_categorical("load_balance_coef", [0.0, 0.01, 0.05]),
        "dropout": trial.suggest_float("dropout", 0.05, 0.30),
        "lr": trial.suggest_float("lr", 1e-4, 2e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [256, 512]),
        "max_epochs": 50,
        "patience": trial.suggest_int("patience", 5, 10),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
        "grad_clip": trial.suggest_float("grad_clip", 0.5, 2.0),
        "scheduler": trial.suggest_categorical("scheduler", ["cosine", "onecycle"]),
        "task_warmup_top_k": 2,
        "task_warmup_epochs": 3,
    }
    if p["top_k_shared"] > p["n_shared_experts"]:
        raise optuna.TrialPruned()
    if p["top_k_task"] > p["n_task_experts"]:
        raise optuna.TrialPruned()
    return p


def run_trial_worker(params_json: str, result_json: str):
    params = json.loads(params_json)
    set_seed(SEED)
    device = get_device(); validate_gpu(device)
    data = load_frozen_v2()
    print(f"[Full-X trial-worker] GPU={torch.cuda.get_device_name(0)} params={params}", flush=True)
    model, score, epoch, _ = train_model(data, device, params, run_dir=None)
    Path(result_json).write_text(json.dumps({"score": float(score), "best_epoch": int(epoch)}), encoding="utf-8")
    del model; gc.collect(); torch.cuda.empty_cache()


def run_tuned(n_trials: int, bootstrap_reps: int):
    if optuna is None:
        raise RuntimeError("Optuna is not installed")
    root = RESULTS_ROOT / "tuned" / "33_Full-X"
    root.mkdir(parents=True, exist_ok=True)
    study_path = root / "study.db"
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        storage=f"sqlite:///{study_path.as_posix()}",
        study_name="full_x_nextgen",
        load_if_exists=True,
    )
    def completed():
        return sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)

    def objective(trial):
        params = suggest_params(trial)
        result_path = root / f".trial_{trial.number}_result.json"
        if result_path.exists(): result_path.unlink()
        cmd = [
            sys.executable, str(Path(__file__).resolve()),
            "--mode", "trial-worker",
            "--trial-params-json", json.dumps(params, separators=(",", ":")),
            "--trial-result-json", str(result_path),
        ]
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=os.environ.copy(), check=False)
        if proc.returncode != 0 or not result_path.exists():
            raise RuntimeError(f"Full-X trial worker failed: returncode={proc.returncode}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result_path.unlink(missing_ok=True)
        trial.set_user_attr("params", params)
        trial.set_user_attr("best_epoch", int(result["best_epoch"]))
        return float(result["score"])

    while completed() < int(n_trials):
        study.optimize(objective, n_trials=1, catch=(RuntimeError, ValueError), show_progress_bar=False)

    trials_df = pd.DataFrame([
        {"number": t.number, "state": str(t.state), "value": t.value, **t.params}
        for t in study.trials
    ])
    trials_df.to_csv(root / "optuna_trials.csv", index=False)
    best = study.best_trial
    params = dict(best.user_attrs.get("params") or best.params)
    params.update({"task_warmup_top_k": 2, "task_warmup_epochs": 3, "max_epochs": 50})
    set_seed(SEED); device = get_device(); validate_gpu(device); data = load_frozen_v2()
    final_dir = root / "final_selected_configuration"
    model, score, epoch, _ = train_model(data, device, params, run_dir=final_dir)
    summary = save_outputs(data, model, device, params, score, epoch, final_dir, bootstrap_reps)
    summary.update({"mode": "tuned", "n_optuna_trials_completed": completed(), "optuna_best_trial_number": int(best.number), "optuna_best_value": float(best.value)})
    (root / "best_params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
    (root / "final_result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def audit_data(data: FrozenData):
    train_subj = set(data.subject_id[data.train_idx].tolist())
    val_subj = set(data.subject_id[data.val_idx].tolist())
    test_subj = set(data.subject_id[data.test_idx].tolist())
    audit = {
        "structured_shape": list(data.structured.shape),
        "demographics_shape": list(data.demographics.shape),
        "n_stays": len(data.stay_id),
        "split_counts": {"train": len(data.train_idx), "val": len(data.val_idx), "test": len(data.test_idx)},
        "text_present": int(np.sum(data.text_missing == 0)),
        "text_missing": int(np.sum(data.text_missing == 1)),
        "patient_overlap": {
            "train_val": len(train_subj & val_subj),
            "train_test": len(train_subj & test_subj),
            "val_test": len(val_subj & test_subj),
        },
    }
    audit["all_pass"] = (
        tuple(data.structured.shape) == EXPECTED_STRUCTURED_SHAPE
        and tuple(data.demographics.shape) == EXPECTED_DEMO_SHAPE
        and len(data.train_idx) == EXPECTED_TRAIN
        and len(data.val_idx) == EXPECTED_VAL
        and len(data.test_idx) == EXPECTED_TEST
        and int(np.sum(data.text_missing == 0)) == EXPECTED_TEXT_PRESENT
        and all(v == 0 for v in audit["patient_overlap"].values())
    )
    print(json.dumps(audit, indent=2), flush=True)
    if not audit["all_pass"]:
        raise RuntimeError("Frozen V2 contract audit failed")


def smoke_test():
    p = dict(FIXED_PARAMS)
    model = TaskConditionedHierarchicalMoE(p)
    n = sum(v.numel() for v in model.parameters() if v.requires_grad)
    x = torch.randn(2, 24, 110)
    d = torch.randn(2, 4)
    txt = torch.zeros(2, 768)
    missing = torch.tensor([0.0, 1.0])
    out = model(x, d, txt, missing)
    expected_outputs = set(BINARY_TASKS) | {"los_excess_log1p"}
    assert set(out.keys()) == expected_outputs, out.keys()
    assert all(tuple(v.shape) == (2,) for v in out.values()), {k: tuple(v.shape) for k, v in out.items()}
    print(json.dumps({"parameter_count": n, "output_shapes": {k: list(v.shape) for k, v in out.items()}, "smoke_pass": True}, indent=2))


def run_fixed(bootstrap_reps: int):
    set_seed(SEED)
    device = get_device(); validate_gpu(device)
    data = load_frozen_v2(); audit_data(data)
    root = RESULTS_ROOT / "fixed" / "33_Full-X"
    root.mkdir(parents=True, exist_ok=True)
    start = time.time()
    model, score, epoch, history = train_model(data, device, dict(FIXED_PARAMS), run_dir=root)
    summary = save_outputs(data, model, device, dict(FIXED_PARAMS), score, epoch, root, bootstrap_reps)
    summary["training_seconds"] = time.time() - start
    summary["mode"] = "fixed"
    summary["gpu"] = torch.cuda.get_device_name(0)
    summary["device"] = str(device)
    summary["bf16"] = True
    summary["test_used_for_selection"] = False
    (root / "final_result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("=" * 100)
    print("FULL-X FIXED FINAL RESULT")
    print("=" * 100)
    print(json.dumps(summary, indent=2), flush=True)
    del model; gc.collect(); torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["audit", "smoke", "fixed", "tuned", "trial-worker"])
    parser.add_argument("--model", default="33")
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--trial-params-json", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--trial-result-json", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.model != "33" and args.mode not in {"audit", "smoke"}:
        raise ValueError("This runner is for model 33 / Full-X")
    if args.mode == "smoke":
        smoke_test(); return
    if args.mode == "audit":
        data = load_frozen_v2(); audit_data(data); return
    if args.mode == "trial-worker":
        if not args.trial_params_json or not args.trial_result_json:
            raise ValueError("trial-worker requires trial params/result paths")
        run_trial_worker(args.trial_params_json, args.trial_result_json); return
    if args.mode == "fixed":
        run_fixed(args.bootstrap_reps); return
    if args.mode == "tuned":
        run_tuned(args.n_trials, args.bootstrap_reps); return


if __name__ == "__main__":
    main()
