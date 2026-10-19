#!/usr/bin/env python3
"""
Complete Statistical Evaluation Layer for the FYP
=================================================

Purpose
-------
Build a new, auditable statistical-evaluation layer from the EXISTING prediction
CSV artifacts. No model is retrained, refit, retuned, or regenerated.

Scope
-----
* Families 01-07 (models 00-32)
* Locked Z2_tuned ensemble (MIMIC validation/test)
* Locked Z2_tuned external eICU validation when the canonical eICU artifacts
  and label file are present

Binary metrics
--------------
* AUROC, AUPRC, Brier, ECE
* Calibration intercept and calibration slope
* Sensitivity, specificity, precision/PPV, NPV, F1
* TP, TN, FP, FN
* 95% CIs for threshold metrics via row bootstrap
* Calibration/reliability curve data
* ROC curve data
* Precision-recall curve data

Comparative inference
---------------------
Reference model: Model 32 (MoE Modality Fusion)
* Paired DeLong test for AUROC
* McNemar exact test for paired thresholded classifications
* Paired bootstrap deltas + percentile 95% CI for AUPRC, Brier, F1,
  sensitivity, specificity, precision, and NPV
* Two-sided bootstrap sign p-value for paired bootstrap deltas
* Benjamini-Hochberg FDR q-values within split/task/test_type groups

Threshold protocol
------------------
* Binary threshold is selected ONLY on the validation split.
* Objective: maximize validation F1.
* Tie-break: choose the smallest threshold among tied maxima.
* The locked threshold is then applied unchanged to test and eICU external data.

LOS metrics
-----------
* MAE, RMSE, Median AE, P90 AE, mean error, R2
* 95% percentile bootstrap CIs

Calibration
-----------
* 10 uniform bins for reliability curves/ECE.
* Calibration intercept/slope are logistic calibration parameters using
  logit(probability) as the sole predictor; CIs are Wald 95% CIs.

Notes
-----
* Existing family files use `target`, `validity`, and `prediction`.
* Z2_tuned prediction files additionally contain `probability_mean` for binary
  tasks; this is used for probability-based statistics, not rank-space `prediction`.
* The script preserves the native Z2 probability semantics.
* eICU labels with invalid binary values (-1) are excluded.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path("/home/abdullah2025/fyp_project")
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "complete_statistical_evaluation"
SEED = 20261003
BOOTSTRAP_REPS = 1000
PAIRWISE_BOOTSTRAP_REPS = 1000
ECE_BINS = 10
REFERENCE_MODEL_ID = "32"
INVALID_BINARY_LABEL = -1

BINARY_TASKS = (
    "mortality_24h",
    "ventilation_onset_24h",
    "vasopressor_onset_24h",
    "readmission_30d_24h",
    "deterioration_24h",
)
LOS_TASK = "los_hours_24h"
ALL_TASKS = BINARY_TASKS + (LOS_TASK,)

FAMILIES = {
    "Family 01 — Tabular 00–05": "baseline_family_01_tabular",
    "Family 02 — RNN 06–09": "baseline_family_02_rnn",
    "Family 03 — TCN 10–12": "baseline_family_03_tcn_10_12",
    "Family 04 — Clinical Sequence 13–16": "baseline_family_04_clinical_sequence_13_16",
    "Family 05 — Transformer Modern 17–23": "baseline_family_05_transformer_modern_17_23",
    "Family 06 — State Space 24–27": "baseline_family_06_state_space_24_27",
    "Family 07 — Advanced Multimodal 28–32": "baseline_family_07_advanced_multimodal_28_32",
}

BINARY_COMPARE_METRICS = (
    "auroc",
    "auprc",
    "brier",
    "f1",
    "sensitivity",
    "specificity",
    "precision",
    "npv",
)

BASE_REQUIRED = {"stay_id", "subject_id", "split", "task", "target", "validity", "prediction"}
Z2_REQUIRED = {"stay_id", "subject_id", "target", "validity", "task", "prediction"}


# -----------------------------------------------------------------------------
# Utility
# -----------------------------------------------------------------------------
def safe_float(x) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else float("nan")
    except Exception:
        return float("nan")


def sigmoid(x):
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-8, 1 - 1e-8)
    return np.log(p / (1.0 - p))


def model_seed_value(model_id: str) -> int:
    """Stable numeric seed component for numeric IDs and named models."""
    sid = str(model_id)
    if sid.isdigit():
        return int(sid)
    return int(sum((i + 1) * ord(ch) for i, ch in enumerate(sid)))


def model_name_from_path(path: Path) -> Tuple[str, str]:
    for part in reversed(path.parts):
        m = re.match(r"^(\d{2})_(.+)$", part)
        if m:
            return m.group(1), m.group(2)
    return "", path.parent.name


def family_for_model(mid: str) -> str:
    i = model_seed_value(mid)
    if i <= 5:
        return "Family 01 — Tabular 00–05"
    if i <= 9:
        return "Family 02 — RNN 06–09"
    if i <= 12:
        return "Family 03 — TCN 10–12"
    if i <= 16:
        return "Family 04 — Clinical Sequence 13–16"
    if i <= 23:
        return "Family 05 — Transformer Modern 17–23"
    if i <= 27:
        return "Family 06 — State Space 24–27"
    return "Family 07 — Advanced Multimodal 28–32"


def probability_series(df: pd.DataFrame, task: str) -> np.ndarray:
    """Return probability-scale binary predictions using native Z2 probability_mean when present."""
    if task in BINARY_TASKS:
        if "probability_mean" in df.columns:
            return pd.to_numeric(df["probability_mean"], errors="coerce").to_numpy(float)
        return pd.to_numeric(df["prediction"], errors="coerce").to_numpy(float)
    return pd.to_numeric(df["prediction"], errors="coerce").to_numpy(float)


def discrimination_series(df: pd.DataFrame, task: str) -> np.ndarray:
    """Native discrimination score. For Z2 this is the native rank-space prediction."""
    return pd.to_numeric(df["prediction"], errors="coerce").to_numpy(float)


def bootstrap_core_binary_ci(
    y: np.ndarray,
    p_disc: np.ndarray,
    p_prob: np.ndarray,
    reps: int,
    rng: np.random.Generator,
) -> Dict[str, Tuple[float, float]]:
    """Percentile bootstrap CIs for AUROC, AUPRC, Brier and ECE."""
    out = {m: [] for m in ("auroc", "auprc", "brier", "ece")}
    n = len(y)
    for _ in range(reps):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        pd = p_disc[idx]
        pp = p_prob[idx]
        if len(np.unique(yy)) < 2:
            continue
        out["auroc"].append(float(roc_auc_score(yy, pd)))
        out["auprc"].append(float(average_precision_score(yy, pd)))
        out["brier"].append(float(brier_score_loss(yy, pp)))
        out["ece"].append(float(ece_score(yy, pp)))
    return {
        k: (float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975)))
        if v else (np.nan, np.nan)
        for k, v in out.items()
    }


def valid_binary(df: pd.DataFrame, task: str, prob: np.ndarray) -> np.ndarray:
    y = pd.to_numeric(df["target"], errors="coerce").to_numpy(float)
    valid = df["validity"].astype(bool).to_numpy()
    valid &= np.isfinite(y) & np.isfinite(prob)
    valid &= np.isin(y, [0.0, 1.0])
    return valid


def valid_los(df: pd.DataFrame, pred: np.ndarray) -> np.ndarray:
    y = pd.to_numeric(df["target"], errors="coerce").to_numpy(float)
    valid = df["validity"].astype(bool).to_numpy()
    valid &= np.isfinite(y) & np.isfinite(pred)
    return valid


def ece_score(y: np.ndarray, p: np.ndarray, bins: int = ECE_BINS) -> float:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 0, 1)
    if len(y) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    n = len(y)
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi if i < bins - 1 else p <= hi)
        if not np.any(mask):
            continue
        total += (mask.sum() / n) * abs(y[mask].mean() - p[mask].mean())
    return float(total)


def threshold_at_max_f1(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2 or len(y) == 0:
        return 0.5
    precision, recall, thresholds = precision_recall_curve(y, p)
    if thresholds.size == 0:
        return 0.5
    f1 = (2 * precision[:-1] * recall[:-1]) / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    best = np.nanmax(f1)
    candidates = thresholds[np.isclose(f1, best, rtol=1e-10, atol=1e-12)]
    return float(np.min(candidates)) if candidates.size else 0.5


def threshold_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> Dict[str, float]:
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    ppv = tp / (tp + fp) if tp + fp else np.nan
    npv = tn / (tn + fn) if tn + fn else np.nan
    f1 = 2 * ppv * sens / (ppv + sens) if np.isfinite(ppv) and np.isfinite(sens) and ppv + sens else np.nan
    return {
        "threshold": threshold,
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "sensitivity": float(sens), "specificity": float(spec),
        "precision": float(ppv), "npv": float(npv), "f1": float(f1),
    }


def calibration_params(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    z = logit(p)
    X = np.column_stack([np.ones(len(z)), z])
    y = np.asarray(y, dtype=float)
    if len(np.unique(y)) < 2:
        return {"calibration_intercept": np.nan, "calibration_slope": np.nan,
                "calibration_intercept_ci_low": np.nan, "calibration_intercept_ci_high": np.nan,
                "calibration_slope_ci_low": np.nan, "calibration_slope_ci_high": np.nan}

    def nll(beta):
        eta = X @ beta
        return float(np.sum(np.logaddexp(0, eta) - y * eta))

    def grad(beta):
        eta = X @ beta
        pr = 1.0 / (1.0 + np.exp(-np.clip(eta, -60, 60)))
        return X.T @ (pr - y)

    res = minimize(nll, np.array([0.0, 1.0]), jac=grad, method="BFGS")
    beta = np.asarray(res.x, dtype=float)
    eta = X @ beta
    pr = 1.0 / (1.0 + np.exp(-np.clip(eta, -60, 60)))
    W = pr * (1 - pr)
    H = X.T @ (X * W[:, None])
    try:
        cov = np.linalg.inv(H)
        se = np.sqrt(np.maximum(np.diag(cov), 0))
    except Exception:
        se = np.array([np.nan, np.nan])
    zcrit = stats.norm.ppf(0.975)
    return {
        "calibration_intercept": float(beta[0]),
        "calibration_slope": float(beta[1]),
        "calibration_intercept_ci_low": float(beta[0] - zcrit * se[0]) if np.isfinite(se[0]) else np.nan,
        "calibration_intercept_ci_high": float(beta[0] + zcrit * se[0]) if np.isfinite(se[0]) else np.nan,
        "calibration_slope_ci_low": float(beta[1] - zcrit * se[1]) if np.isfinite(se[1]) else np.nan,
        "calibration_slope_ci_high": float(beta[1] + zcrit * se[1]) if np.isfinite(se[1]) else np.nan,
    }


def bootstrap_threshold_ci(y: np.ndarray, p: np.ndarray, threshold: float, reps: int, rng: np.random.Generator) -> Dict[str, Tuple[float, float]]:
    n = len(y)
    out = {m: [] for m in ("sensitivity", "specificity", "precision", "npv", "f1")}
    pred = (p >= threshold).astype(np.int8)
    for _ in range(reps):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        pp = pred[idx]
        tn, fp, fn, tp = confusion_matrix(yy, pp, labels=[0, 1]).ravel()
        vals = {
            "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
            "specificity": tn / (tn + fp) if tn + fp else np.nan,
            "precision": tp / (tp + fp) if tp + fp else np.nan,
            "npv": tn / (tn + fn) if tn + fn else np.nan,
        }
        vals["f1"] = 2 * vals["precision"] * vals["sensitivity"] / (vals["precision"] + vals["sensitivity"]) if np.isfinite(vals["precision"]) and np.isfinite(vals["sensitivity"]) and vals["precision"] + vals["sensitivity"] else np.nan
        for k, v in vals.items():
            if np.isfinite(v):
                out[k].append(v)
    return {k: (float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))) if v else (np.nan, np.nan) for k, v in out.items()}


def bootstrap_los_ci(y: np.ndarray, pred: np.ndarray, reps: int, rng: np.random.Generator) -> Dict[str, Tuple[float, float]]:
    n = len(y)
    metrics = {m: [] for m in ("mae", "rmse", "median_ae", "p90_ae", "mean_error", "r2")}
    err = pred - y
    for _ in range(reps):
        idx = rng.integers(0, n, n)
        e = err[idx]
        yy = y[idx]
        pp = pred[idx]
        metrics["mae"].append(float(np.mean(np.abs(e))))
        metrics["rmse"].append(float(np.sqrt(np.mean(e * e))))
        metrics["median_ae"].append(float(np.median(np.abs(e))))
        metrics["p90_ae"].append(float(np.quantile(np.abs(e), 0.90)))
        metrics["mean_error"].append(float(np.mean(e)))
        metrics["r2"].append(float(r2_score(yy, pp)) if len(np.unique(yy)) > 1 else np.nan)
    return {k: (float(np.nanquantile(v, 0.025)), float(np.nanquantile(v, 0.975))) for k, v in metrics.items()}


# -----------------------------------------------------------------------------
# DeLong implementation (paired AUROC comparison)
# -----------------------------------------------------------------------------
def compute_midrank(x: np.ndarray) -> np.ndarray:
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(N, dtype=float)
    out[J] = T
    return out


def fast_delong(predictions_sorted_transposed: np.ndarray, label_1_count: int):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    k = predictions_sorted_transposed.shape[0]
    tx = np.empty((k, m))
    ty = np.empty((k, n))
    tz = np.empty((k, m + n))
    for r in range(k):
        tx[r, :] = compute_midrank(predictions_sorted_transposed[r, :m])
        ty[r, :] = compute_midrank(predictions_sorted_transposed[r, m:])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    if np.ndim(sx) == 0:
        sx = np.array([[float(sx)]])
    if np.ndim(sy) == 0:
        sy = np.array([[float(sy)]])
    delong_cov = sx / m + sy / n
    return aucs, delong_cov


def delong_pvalue(y: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    order = np.argsort(-y)
    y = y[order]
    preds = np.vstack([p1[order], p2[order]])
    m = int(y.sum())
    n = len(y) - m
    if m == 0 or n == 0:
        return np.nan
    aucs, cov = fast_delong(preds, m)
    diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return 0.0 if diff != 0 else 1.0
    z = abs(diff) / math.sqrt(var)
    return float(2 * stats.norm.sf(z))


def mcnemar_exact_pvalue(y: np.ndarray, pred1: np.ndarray, pred2: np.ndarray) -> float:
    pred1 = np.asarray(pred1).astype(int)
    pred2 = np.asarray(pred2).astype(int)
    e1 = pred1 != y
    e2 = pred2 != y
    b = int(np.sum(e1 & ~e2))
    c = int(np.sum(~e1 & e2))
    discordant = b + c
    if discordant == 0:
        return 1.0
    return float(stats.binomtest(min(b, c), n=discordant, p=0.5).pvalue)


# -----------------------------------------------------------------------------
# Paired bootstrap differences
# -----------------------------------------------------------------------------
def metric_value(y: np.ndarray, p: np.ndarray, metric: str, threshold: Optional[float] = None) -> float:
    if metric == "auroc":
        return float(roc_auc_score(y, p))
    if metric == "auprc":
        return float(average_precision_score(y, p))
    if metric == "brier":
        return float(brier_score_loss(y, p))
    if threshold is None:
        raise ValueError("Threshold required for threshold metric")
    return threshold_metrics(y, p, threshold)[metric]


def paired_bootstrap_delta(
    y: np.ndarray,
    p_model: np.ndarray,
    p_ref: np.ndarray,
    metric: str,
    threshold_model: Optional[float],
    threshold_ref: Optional[float],
    reps: int,
    rng: np.random.Generator,
) -> Tuple[float, float, float, float]:
    point = metric_value(y, p_model, metric, threshold_model) - metric_value(y, p_ref, metric, threshold_ref)
    n = len(y)
    deltas = []
    for _ in range(reps):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if metric in {"auroc", "auprc"} and len(np.unique(yy)) < 2:
            continue
        d = metric_value(yy, p_model[idx], metric, threshold_model) - metric_value(yy, p_ref[idx], metric, threshold_ref)
        if np.isfinite(d):
            deltas.append(d)
    if not deltas:
        return float(point), np.nan, np.nan, np.nan
    arr = np.asarray(deltas, dtype=float)
    ci_low, ci_high = np.quantile(arr, [0.025, 0.975])
    p_boot = 2 * min(np.mean(arr <= 0), np.mean(arr >= 0))
    return float(point), float(ci_low), float(ci_high), float(min(1.0, p_boot))


# -----------------------------------------------------------------------------
# Discovery / loading
# -----------------------------------------------------------------------------
def discover_base_models() -> Dict[str, Dict[str, object]]:
    result: Dict[str, Dict[str, object]] = {}
    for family_name, family_dir in FAMILIES.items():
        root = PROJECT_ROOT / "outputs" / family_dir
        if not root.exists():
            raise FileNotFoundError(f"Missing family directory: {root}")
        paths = sorted(root.glob("**/final_selected_configuration/test_predictions.csv"))
        for path in paths:
            mid, mname = model_name_from_path(path)
            if not mid:
                continue
            if mid in result and result[mid]["path"] != path:
                # Family 07 is the intended final location for 28-32.
                if model_seed_value(mid) >= 28 and family_dir == "baseline_family_07_advanced_multimodal_28_32":
                    result[mid] = {"family": family_name, "name": mname, "path": path}
                continue
            result[mid] = {"family": family_name, "name": mname, "path": path}
    missing = [f"{i:02d}" for i in range(33) if f"{i:02d}" not in result]
    if missing:
        raise FileNotFoundError("Missing model prediction files for: " + ", ".join(missing))
    return result


def load_model_df(path: Path, require_split: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = BASE_REQUIRED if require_split else Z2_REQUIRED
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return df


def discover_z2() -> Dict[str, Path]:
    zdir = PROJECT_ROOT / "outputs" / "model_z_family"
    paths = {
        "val": zdir / "Z2_tuned_validation_predictions.csv",
        "test": zdir / "Z2_tuned_test_predictions.csv",
    }
    for k, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing Z2_tuned {k} predictions: {p}")
    return paths


def derive_z2_deterioration_rows(df: pd.DataFrame, split: str) -> pd.DataFrame:
    """Reproduce native Model-Z deterioration as a 3-way probability-space noisy-OR.

    Native model_z_family.py defines deterioration from mortality, ventilation,
    and vasopressor only. It does not store deterioration as a standalone Z2
    task row, so the complete evaluator materializes the derived row here.
    """
    component_tasks = (
        "mortality_24h",
        "ventilation_onset_24h",
        "vasopressor_onset_24h",
    )
    parts = []
    for task in component_tasks:
        z = df[df["task"].astype(str).eq(task)].sort_values("stay_id").reset_index(drop=True).copy()
        if z.empty:
            raise ValueError(f"Z2 {split}: missing deterioration component task {task}")
        if "probability_mean" not in z.columns:
            raise ValueError(f"Z2 {split}: missing probability_mean for {task}")
        parts.append(z)

    base = parts[0]
    for task, z in zip(component_tasks[1:], parts[1:]):
        if not np.array_equal(base["stay_id"].to_numpy(), z["stay_id"].to_numpy()):
            raise ValueError(f"Z2 {split}: deterioration stay_id alignment failure for {task}")
        if not np.array_equal(base["subject_id"].to_numpy(), z["subject_id"].to_numpy()):
            raise ValueError(f"Z2 {split}: deterioration subject_id alignment failure for {task}")

    targets = np.stack([z["target"].to_numpy(float) for z in parts], axis=0)
    probs = np.stack([np.clip(z["probability_mean"].to_numpy(float), 0.0, 1.0) for z in parts], axis=0)
    validity = np.all(np.stack([z["validity"].astype(bool).to_numpy() for z in parts], axis=0), axis=0)

    y_det = 1.0 - np.prod(1.0 - targets, axis=0)
    p_det = 1.0 - np.prod(1.0 - probs, axis=0)

    out = pd.DataFrame({
        "stay_id": base["stay_id"].to_numpy(),
        "subject_id": base["subject_id"].to_numpy(),
        "target": y_det,
        "validity": validity,
        "task": "deterioration_24h",
        # There is no separate rank-space deterioration prediction in native Z2.
        # Native Model-Z evaluates deterioration from the probability-space noisy-OR.
        "prediction": p_det,
        "probability_mean": p_det,
        "split": split,
    })
    return out


# -----------------------------------------------------------------------------
# Output helpers
# -----------------------------------------------------------------------------
def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str))


def bh_fdr(pvals: pd.Series) -> pd.Series:
    vals = pvals.to_numpy(float)
    q = np.full(vals.shape, np.nan)
    keep = np.isfinite(vals)
    if not np.any(keep):
        return pd.Series(q, index=pvals.index)
    idx = np.where(keep)[0]
    pv = vals[keep]
    order = np.argsort(pv)
    ranked = pv[order]
    m = len(ranked)
    q_ranked = ranked * m / np.arange(1, m + 1)
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q[keep] = np.minimum(q_ranked[np.argsort(order)], 1.0)
    return pd.Series(q, index=pvals.index)


# -----------------------------------------------------------------------------
# Main evaluation
# -----------------------------------------------------------------------------
def evaluate_model(model_id: str, model_name: str, family: str, df_by_split: Dict[str, pd.DataFrame], rng_seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    threshold_rows = []
    curve_rows_roc = []
    curve_rows_pr = []
    curve_rows_cal = []
    los_rows = []

    locked_thresholds = {}
    for task in BINARY_TASKS:
        d = df_by_split["val"]
        sub = d[d["task"] == task].copy()
        p_prob = probability_series(sub, task)
        keep = valid_binary(sub, task, p_prob)
        y = pd.to_numeric(sub.loc[keep, "target"], errors="coerce").to_numpy(int)
        p_prob = p_prob[keep]
        threshold = threshold_at_max_f1(y, p_prob)
        locked_thresholds[task] = threshold
        val_tm = threshold_metrics(y, p_prob, threshold)
        threshold_rows.append({
            "family": family,
            "model_id": model_id,
            "model_name": model_name,
            "task": task,
            "threshold_source": "validation",
            "threshold": threshold,
            "validation_f1_at_locked_threshold": val_tm["f1"],
        })

    for split in ("val", "test"):
        d = df_by_split[split]
        for task in BINARY_TASKS:
            sub = d[d["task"] == task].copy()
            p_disc = discrimination_series(sub, task)
            p_prob = probability_series(sub, task)
            keep = valid_binary(sub, task, p_prob)
            keep &= np.isfinite(p_disc)
            y = pd.to_numeric(sub.loc[keep, "target"], errors="coerce").to_numpy(int)
            p_disc = p_disc[keep]
            p_prob = p_prob[keep]
            n = len(y)
            if n == 0 or len(np.unique(y)) < 2:
                continue

            threshold = locked_thresholds[task]
            tm = threshold_metrics(y, p_prob, threshold)
            cal = calibration_params(y, p_prob)
            row = {
                "family": family,
                "model_id": model_id,
                "model_name": model_name,
                "split": split,
                "task": task,
                "n": n,
                "events": int(y.sum()),
                "prevalence": float(y.mean()),
                "threshold": threshold,
                "auroc": float(roc_auc_score(y, p_disc)),
                "auprc": float(average_precision_score(y, p_disc)),
                "brier": float(brier_score_loss(y, p_prob)),
                "ece": float(ece_score(y, p_prob)),
                **tm,
                **cal,
            }

            seed = (
                rng_seed
                + model_seed_value(model_id) * 101
                + (0 if split == "val" else 100000)
                + list(BINARY_TASKS).index(task) * 1000
            )
            core_ci = bootstrap_core_binary_ci(
                y, p_disc, p_prob, BOOTSTRAP_REPS,
                np.random.default_rng(seed),
            )
            for metric, (lo, hi) in core_ci.items():
                row[f"{metric}_ci95_low"] = lo
                row[f"{metric}_ci95_high"] = hi

            threshold_ci = bootstrap_threshold_ci(
                y, p_prob, threshold, BOOTSTRAP_REPS,
                np.random.default_rng(seed + 2000000),
            )
            for metric, (lo, hi) in threshold_ci.items():
                row[f"{metric}_ci95_low"] = lo
                row[f"{metric}_ci95_high"] = hi

            summary_rows.append(row)

            fpr, tpr, roc_thr = roc_curve(y, p_disc)
            for i in range(len(fpr)):
                curve_rows_roc.append({
                    "family": family, "model_id": model_id, "model_name": model_name,
                    "split": split, "task": task,
                    "fpr": fpr[i], "tpr": tpr[i], "threshold": roc_thr[i],
                })

            precision, recall, pr_thr = precision_recall_curve(y, p_disc)
            for i in range(len(precision)):
                curve_rows_pr.append({
                    "family": family, "model_id": model_id, "model_name": model_name,
                    "split": split, "task": task,
                    "precision": precision[i], "recall": recall[i],
                    "threshold": pr_thr[i] if i < len(pr_thr) else np.nan,
                })

            prob_true, prob_pred = calibration_curve(
                y, p_prob, n_bins=ECE_BINS, strategy="uniform",
            )
            for i in range(len(prob_true)):
                curve_rows_cal.append({
                    "family": family, "model_id": model_id, "model_name": model_name,
                    "split": split, "task": task, "bin_index": i + 1,
                    "mean_predicted_probability": prob_pred[i],
                    "observed_event_rate": prob_true[i],
                })

        sub = d[d["task"] == LOS_TASK].copy()
        pred = probability_series(sub, LOS_TASK)
        keep = valid_los(sub, pred)
        y = pd.to_numeric(sub.loc[keep, "target"], errors="coerce").to_numpy(float)
        pred = pred[keep]
        if len(y) > 0:
            err = pred - y
            los = {
                "family": family, "model_id": model_id, "model_name": model_name,
                "split": split, "task": LOS_TASK, "n": len(y),
                "mae": float(np.mean(np.abs(err))),
                "rmse": float(np.sqrt(np.mean(err ** 2))),
                "median_ae": float(np.median(np.abs(err))),
                "p90_ae": float(np.quantile(np.abs(err), 0.90)),
                "mean_error": float(np.mean(err)),
                "r2": float(r2_score(y, pred)) if len(np.unique(y)) > 1 else np.nan,
            }
            ci = bootstrap_los_ci(
                y, pred, BOOTSTRAP_REPS,
                np.random.default_rng(
                    rng_seed + model_seed_value(model_id) * 73 + (0 if split == "val" else 200000)
                ),
            )
            for metric, (lo, hi) in ci.items():
                los[f"{metric}_ci95_low"] = lo
                los[f"{metric}_ci95_high"] = hi
            los_rows.append(los)

    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(threshold_rows),
        pd.DataFrame(curve_rows_roc),
        pd.DataFrame(los_rows),
    ), pd.DataFrame(curve_rows_pr), pd.DataFrame(curve_rows_cal), pd.DataFrame(threshold_rows)

def evaluate_reference_comparisons(all_predictions: Dict[str, Dict[str, pd.DataFrame]], thresholds: Dict[Tuple[str, str], float], seed: int) -> pd.DataFrame:
    rows = []
    ref = REFERENCE_MODEL_ID
    if ref not in all_predictions:
        raise KeyError("Reference Model 32 is missing")

    for split in ("val", "test"):
        for task in BINARY_TASKS:
            refdf = all_predictions[ref][split]
            rsub = refdf[refdf["task"] == task].copy()
            r_disc = discrimination_series(rsub, task)
            r_prob = probability_series(rsub, task)
            keep = valid_binary(rsub, task, r_prob) & np.isfinite(r_disc)
            rr = rsub.loc[keep, ["stay_id", "subject_id", "target"]].copy()
            rr["disc_ref"] = r_disc[keep]
            rr["prob_ref"] = r_prob[keep]

            for mid, parts in all_predictions.items():
                if mid == ref:
                    continue
                msub = parts[split]
                msub = msub[msub["task"] == task].copy()
                m_disc = discrimination_series(msub, task)
                m_prob = probability_series(msub, task)
                mkeep = valid_binary(msub, task, m_prob) & np.isfinite(m_disc)
                mm = msub.loc[mkeep, ["stay_id", "subject_id", "target"]].copy()
                mm["disc_model"] = m_disc[mkeep]
                mm["prob_model"] = m_prob[mkeep]

                z = rr.merge(mm, on=["stay_id", "subject_id"], suffixes=("_ref", "_model"))
                if z.empty:
                    continue

                y = z["target_ref"].to_numpy(int)
                disc_ref = z["disc_ref"].to_numpy(float)
                disc_model = z["disc_model"].to_numpy(float)
                prob_ref = z["prob_ref"].to_numpy(float)
                prob_model = z["prob_model"].to_numpy(float)
                t_ref = thresholds[(ref, task)]
                t_model = thresholds[(mid, task)]
                model_cls = (prob_model >= t_model).astype(int)
                ref_cls = (prob_ref >= t_ref).astype(int)

                # AUROC: native discrimination scores. CI from paired bootstrap; p-value from DeLong.
                auc_point, auc_lo, auc_hi, _ = paired_bootstrap_delta(
                    y, disc_model, disc_ref, "auroc", None, None,
                    PAIRWISE_BOOTSTRAP_REPS,
                    np.random.default_rng(
                        seed + model_seed_value(mid) * 1000
                        + (0 if split == "val" else 500000)
                        + list(BINARY_TASKS).index(task) * 10000 + 901
                    ),
                )
                p_delong = delong_pvalue(y, disc_model, disc_ref)
                rows.append({
                    "split": split, "task": task, "model_id": mid, "reference_model_id": ref,
                    "test_type": "paired_DeLong_AUROC", "metric": "auroc", "n": len(y),
                    "delta_model_minus_reference": auc_point,
                    "ci95_low": auc_lo, "ci95_high": auc_hi, "p_value": p_delong,
                })

                # McNemar on thresholded probability-scale classification.
                p_mc = mcnemar_exact_pvalue(y, model_cls, ref_cls)
                err_delta = float(np.mean(model_cls == y) - np.mean(ref_cls == y))
                rows.append({
                    "split": split, "task": task, "model_id": mid, "reference_model_id": ref,
                    "test_type": "McNemar_exact", "metric": "classification_error", "n": len(y),
                    "delta_model_minus_reference": err_delta,
                    "ci95_low": np.nan, "ci95_high": np.nan, "p_value": p_mc,
                })

                # Paired bootstrap for the remaining metrics. AUPRC uses native discrimination;
                # Brier and threshold metrics use native probability scale.
                specs = [
                    ("auprc", disc_model, disc_ref, None, None),
                    ("brier", prob_model, prob_ref, None, None),
                    ("f1", prob_model, prob_ref, t_model, t_ref),
                    ("sensitivity", prob_model, prob_ref, t_model, t_ref),
                    ("specificity", prob_model, prob_ref, t_model, t_ref),
                    ("precision", prob_model, prob_ref, t_model, t_ref),
                    ("npv", prob_model, prob_ref, t_model, t_ref),
                ]
                for metric, pm, pr, tm, tr in specs:
                    metric_seed = (
                        seed + model_seed_value(mid) * 1000
                        + (0 if split == "val" else 500000)
                        + list(BINARY_TASKS).index(task) * 10000
                        + len(metric)
                    )
                    point, lo, hi, p_boot = paired_bootstrap_delta(
                        y, pm, pr, metric, tm, tr,
                        PAIRWISE_BOOTSTRAP_REPS,
                        np.random.default_rng(metric_seed),
                    )
                    rows.append({
                        "split": split, "task": task, "model_id": mid, "reference_model_id": ref,
                        "test_type": "paired_bootstrap", "metric": metric, "n": len(y),
                        "delta_model_minus_reference": point,
                        "ci95_low": lo, "ci95_high": hi, "p_value": p_boot,
                    })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["q_value_bh_fdr"] = np.nan
    for _, idx in out.groupby(["split", "task", "test_type"], dropna=False).groups.items():
        out.loc[idx, "q_value_bh_fdr"] = bh_fdr(out.loc[idx, "p_value"]).to_numpy()
    return out


def evaluate_eicu(z2_val: pd.DataFrame, threshold_map: Dict[str, float], outdir: Path):
    pred_path = PROJECT_ROOT / "outputs" / "external_eicu_z2" / "z2_tuned_eicu_predictions_canonical.csv"
    label_path = PROJECT_ROOT / "outputs" / "external_validation_inventory" / "eicu_labels_aligned_24h.csv"
    if not pred_path.exists() or not label_path.exists():
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {"status": "eICU artifacts not found"}

    pred = pd.read_csv(pred_path)
    labels = pd.read_csv(label_path)
    id_col = "patientunitstayid"
    if id_col not in pred.columns or id_col not in labels.columns:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {"status": "eICU key column missing"}

    d = pred.merge(labels, on=id_col, how="inner", suffixes=("", "_label"))
    rows, curves_roc, curves_pr, curves_cal = [], [], [], []

    for task in BINARY_TASKS:
        label_col = next((c for c in (task, f"{task}_label") if c in d.columns), None)
        disc_col = f"{task}_prediction"
        prob_col = f"{task}_probability_mean"
        if label_col is None or disc_col not in d.columns or prob_col not in d.columns:
            continue

        y = pd.to_numeric(d[label_col], errors="coerce").to_numpy(float)
        p_disc = pd.to_numeric(d[disc_col], errors="coerce").to_numpy(float)
        p_prob = pd.to_numeric(d[prob_col], errors="coerce").to_numpy(float)
        keep = np.isfinite(y) & np.isfinite(p_disc) & np.isfinite(p_prob) & np.isin(y, [0,1]) & (y != INVALID_BINARY_LABEL)
        y = y[keep].astype(int); p_disc = p_disc[keep]; p_prob = p_prob[keep]
        if len(y) == 0 or len(np.unique(y)) < 2:
            continue

        threshold = threshold_map[task]
        tm = threshold_metrics(y, p_prob, threshold)
        cal = calibration_params(y, p_prob)
        core_ci = bootstrap_core_binary_ci(
            y, p_disc, p_prob, BOOTSTRAP_REPS,
            np.random.default_rng(SEED + 900000 + list(BINARY_TASKS).index(task)),
        )
        thr_ci = bootstrap_threshold_ci(
            y, p_prob, threshold, BOOTSTRAP_REPS,
            np.random.default_rng(SEED + 950000 + list(BINARY_TASKS).index(task)),
        )

        row = {
            "dataset": "eICU-CRD", "model": "Z2_tuned", "task": task,
            "n": len(y), "events": int(y.sum()), "prevalence": float(y.mean()),
            "auroc": float(roc_auc_score(y, p_disc)), "auprc": float(average_precision_score(y, p_disc)),
            "brier": float(brier_score_loss(y, p_prob)), "ece": float(ece_score(y, p_prob)),
            "threshold_locked_from_mimic_validation": threshold,
            "discrimination_source": "native Z2 prediction",
            "probability_source": "native Z2 probability_mean",
            **tm, **cal,
        }
        for metric, (lo, hi) in core_ci.items():
            row[f"{metric}_ci95_low"] = lo; row[f"{metric}_ci95_high"] = hi
        for metric, (lo, hi) in thr_ci.items():
            row[f"{metric}_ci95_low"] = lo; row[f"{metric}_ci95_high"] = hi
        rows.append(row)

        fpr, tpr, thr = roc_curve(y, p_disc)
        for i in range(len(fpr)):
            curves_roc.append({"dataset":"eICU-CRD","model":"Z2_tuned","task":task,"fpr":fpr[i],"tpr":tpr[i],"threshold":thr[i]})
        pr, rc, thr = precision_recall_curve(y, p_disc)
        for i in range(len(pr)):
            curves_pr.append({"dataset":"eICU-CRD","model":"Z2_tuned","task":task,"precision":pr[i],"recall":rc[i],"threshold":thr[i] if i < len(thr) else np.nan})
        pt, pp = calibration_curve(y, p_prob, n_bins=ECE_BINS, strategy="uniform")
        for i in range(len(pt)):
            curves_cal.append({"dataset":"eICU-CRD","model":"Z2_tuned","task":task,"bin_index":i+1,"mean_predicted_probability":pp[i],"observed_event_rate":pt[i]})

    los_rows = []
    label_col = LOS_TASK if LOS_TASK in d.columns else f"{LOS_TASK}_label" if f"{LOS_TASK}_label" in d.columns else None
    pred_col = "los_prediction"
    if label_col and pred_col in d.columns:
        y = pd.to_numeric(d[label_col], errors="coerce").to_numpy(float)
        pred_los = pd.to_numeric(d[pred_col], errors="coerce").to_numpy(float)
        keep = np.isfinite(y) & np.isfinite(pred_los)
        y = y[keep]; pred_los = pred_los[keep]
        if len(y):
            err = pred_los - y
            los = {"dataset":"eICU-CRD","model":"Z2_tuned","task":LOS_TASK,"n":len(y),"mae":float(np.mean(np.abs(err))),"rmse":float(np.sqrt(np.mean(err**2))),"median_ae":float(np.median(np.abs(err))),"p90_ae":float(np.quantile(np.abs(err),.9)),"mean_error":float(np.mean(err)),"r2":float(r2_score(y,pred_los)) if len(np.unique(y))>1 else np.nan}
            ci = bootstrap_los_ci(y,pred_los,BOOTSTRAP_REPS,np.random.default_rng(SEED+970000))
            for metric,(lo,hi) in ci.items():
                los[f"{metric}_ci95_low"] = lo; los[f"{metric}_ci95_high"] = hi
            los_rows.append(los)

    return (
        pd.DataFrame(rows), pd.DataFrame(los_rows), pd.DataFrame(curves_roc),
        pd.DataFrame(curves_pr), pd.DataFrame(curves_cal),
        {"status":"ok","prediction_file":str(pred_path),"label_file":str(label_path),
         "discrimination_source":"native Z2 prediction","probability_source":"native Z2 probability_mean"}
    )

def build_plots(curves_roc: pd.DataFrame, curves_pr: pd.DataFrame, curves_cal: pd.DataFrame, outdir: Path):
    import matplotlib.pyplot as plt
    plot_dir = outdir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    if curves_roc.empty:
        return
    # Family/task/split comparison figures. One figure per group.
    for (family, task, split), g in curves_roc.groupby(["family","task","split"]):
        fig, ax = plt.subplots(figsize=(7,6))
        for (mid,name), gm in g.groupby(["model_id","model_name"]):
            ax.plot(gm["fpr"], gm["tpr"], lw=1.2, label=f"{mid} {name}")
        ax.plot([0,1],[0,1], linestyle="--", linewidth=1)
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC — {family} — {task} — {split}")
        ax.legend(fontsize=7, loc="lower right")
        fig.tight_layout(); fig.savefig(plot_dir / f"roc_{re.sub(r'[^A-Za-z0-9]+','_',family)}_{task}_{split}.png", dpi=160); plt.close(fig)

    for (family, task, split), g in curves_pr.groupby(["family","task","split"]):
        fig, ax = plt.subplots(figsize=(7,6))
        for (mid,name), gm in curves_pr[curves_pr["family"].eq(family) & curves_pr["task"].eq(task) & curves_pr["split"].eq(split)].groupby(["model_id","model_name"]):
            ax.plot(gm["recall"], gm["precision"], lw=1.2, label=f"{mid} {name}")
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.set_title(f"Precision–Recall — {family} — {task} — {split}")
        ax.legend(fontsize=7, loc="lower left")
        fig.tight_layout(); fig.savefig(plot_dir / f"pr_{re.sub(r'[^A-Za-z0-9]+','_',family)}_{task}_{split}.png", dpi=160); plt.close(fig)

    for (family, task, split), g in curves_cal.groupby(["family","task","split"]):
        fig, ax = plt.subplots(figsize=(7,6))
        for (mid,name), gm in curves_cal[curves_cal["family"].eq(family) & curves_cal["task"].eq(task) & curves_cal["split"].eq(split)].groupby(["model_id","model_name"]):
            ax.plot(gm["mean_predicted_probability"], gm["observed_event_rate"], marker="o", lw=1.2, label=f"{mid} {name}")
        ax.plot([0,1],[0,1], linestyle="--", linewidth=1)
        ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed event rate")
        ax.set_title(f"Calibration — {family} — {task} — {split}")
        ax.legend(fontsize=7, loc="upper left")
        fig.tight_layout(); fig.savefig(plot_dir / f"calibration_{re.sub(r'[^A-Za-z0-9]+','_',family)}_{task}_{split}.png", dpi=160); plt.close(fig)


def main():
    global BOOTSTRAP_REPS, PAIRWISE_BOOTSTRAP_REPS, SEED
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS)
    parser.add_argument("--pairwise-bootstrap-reps", type=int, default=PAIRWISE_BOOTSTRAP_REPS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    BOOTSTRAP_REPS = int(args.bootstrap_reps)
    PAIRWISE_BOOTSTRAP_REPS = int(args.pairwise_bootstrap_reps)
    SEED = int(args.seed)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "plots").mkdir(exist_ok=True)

    print("[1/6] Discovering 33 audited base-model prediction files...", flush=True)
    base = discover_base_models()
    all_predictions: Dict[str, Dict[str, pd.DataFrame]] = {}
    model_meta = {}
    for mid, meta in sorted(base.items(), key=lambda x: int(x[0])):
        df = load_model_df(Path(meta["path"]))
        all_predictions[mid] = {
            "val": df[df["split"].eq("val")].copy(),
            "test": df[df["split"].eq("test")].copy(),
        }
        model_meta[mid] = meta

    print("[2/6] Loading locked Z2_tuned validation/test prediction files...", flush=True)
    z2_paths = discover_z2()
    z2_val = load_model_df(z2_paths["val"], require_split=False)
    z2_test = load_model_df(z2_paths["test"], require_split=False)
    # These are separate files, so assign split labels for downstream consistency.
    z2_val["split"] = "val"; z2_test["split"] = "test"

    # Native Model-Z does not serialize deterioration_24h as a standalone Z2 row.
    # Materialize the exact native 3-way probability-space noisy-OR here so the
    # complete evaluator covers deterioration for Z2 on MIMIC val/test as well.
    z2_val_det = derive_z2_deterioration_rows(z2_val, "val")
    z2_test_det = derive_z2_deterioration_rows(z2_test, "test")
    z2_val = pd.concat([z2_val, z2_val_det], ignore_index=True)
    z2_test = pd.concat([z2_test, z2_test_det], ignore_index=True)

    all_predictions["Z2_tuned"] = {"val": z2_val, "test": z2_test}
    model_meta["Z2_tuned"] = {"family":"Family Z", "name":"Z2_tuned", "path_val":str(z2_paths["val"]), "path_test":str(z2_paths["test"])}

    print("[3/6] Computing thresholds + complete single-model statistics...", flush=True)
    summary_frames=[]; los_frames=[]; roc_frames=[]; pr_frames=[]; cal_frames=[]; threshold_frames=[]
    threshold_map={}

    for mid, parts in all_predictions.items():
        family = "Family Z" if mid == "Z2_tuned" else model_meta[mid]["family"]
        name = "Z2_tuned" if mid == "Z2_tuned" else model_meta[mid]["name"]

        for task in BINARY_TASKS:
            d = parts["val"]
            sub = d[d["task"].eq(task)].copy()
            p_prob = probability_series(sub, task)
            keep = valid_binary(sub, task, p_prob)
            y = pd.to_numeric(sub.loc[keep,"target"], errors="coerce").to_numpy(int)
            p_prob = p_prob[keep]
            threshold_map[(mid,task)] = threshold_at_max_f1(y,p_prob)

        for split in ("val","test"):
            d = parts[split]
            for task in BINARY_TASKS:
                sub = d[d["task"].eq(task)].copy()
                p_disc = discrimination_series(sub, task)
                p_prob = probability_series(sub, task)
                keep = valid_binary(sub, task, p_prob) & np.isfinite(p_disc)
                y = pd.to_numeric(sub.loc[keep,"target"], errors="coerce").to_numpy(int)
                p_disc = p_disc[keep]; p_prob = p_prob[keep]
                if len(y)==0 or len(np.unique(y))<2: continue
                thr = threshold_map[(mid,task)]
                tm = threshold_metrics(y,p_prob,thr); cal = calibration_params(y,p_prob)
                row = {"family":family,"model_id":mid,"model_name":name,"split":split,"task":task,"n":len(y),"events":int(y.sum()),"prevalence":float(y.mean()),"threshold":thr,"auroc":float(roc_auc_score(y,p_disc)),"auprc":float(average_precision_score(y,p_disc)),"brier":float(brier_score_loss(y,p_prob)),"ece":float(ece_score(y,p_prob)),**tm,**cal}
                seed = SEED + model_seed_value(mid)*101 + (0 if split=="val" else 100000) + list(BINARY_TASKS).index(task)*1000
                core_ci = bootstrap_core_binary_ci(y,p_disc,p_prob,BOOTSTRAP_REPS,np.random.default_rng(seed))
                for metric,(lo,hi) in core_ci.items(): row[f"{metric}_ci95_low"]=lo; row[f"{metric}_ci95_high"]=hi
                thr_ci = bootstrap_threshold_ci(y,p_prob,thr,BOOTSTRAP_REPS,np.random.default_rng(seed+2000000))
                for metric,(lo,hi) in thr_ci.items(): row[f"{metric}_ci95_low"]=lo; row[f"{metric}_ci95_high"]=hi
                summary_frames.append(row)

                fpr,tpr,roc_thr=roc_curve(y,p_disc)
                roc_frames.extend([{**{"family":family,"model_id":mid,"model_name":name,"split":split,"task":task},"fpr":a,"tpr":b,"threshold":c} for a,b,c in zip(fpr,tpr,roc_thr)])
                pr,rc,pr_thr=precision_recall_curve(y,p_disc)
                pr_frames.extend([{**{"family":family,"model_id":mid,"model_name":name,"split":split,"task":task},"precision":a,"recall":b,"threshold":pr_thr[i] if i<len(pr_thr) else np.nan} for i,(a,b) in enumerate(zip(pr,rc))])
                pc,pp=calibration_curve(y,p_prob,n_bins=ECE_BINS,strategy="uniform")
                cal_frames.extend([{**{"family":family,"model_id":mid,"model_name":name,"split":split,"task":task,"bin_index":i+1},"mean_predicted_probability":pp[i],"observed_event_rate":pc[i]} for i in range(len(pc))])
                threshold_frames.append({"family":family,"model_id":mid,"model_name":name,"task":task,"locked_threshold":thr,"selection_rule":"validation max F1; smallest threshold on ties"})

            sub=d[d["task"].eq(LOS_TASK)].copy(); pred=probability_series(sub,LOS_TASK); keep=valid_los(sub,pred)
            y=pd.to_numeric(sub.loc[keep,"target"],errors="coerce").to_numpy(float); pred=pred[keep]
            if len(y):
                e=pred-y; los={"family":family,"model_id":mid,"model_name":name,"split":split,"task":LOS_TASK,"n":len(y),"mae":float(np.mean(np.abs(e))),"rmse":float(np.sqrt(np.mean(e**2))),"median_ae":float(np.median(np.abs(e))),"p90_ae":float(np.quantile(np.abs(e),.9)),"mean_error":float(np.mean(e)),"r2":float(r2_score(y,pred)) if len(np.unique(y))>1 else np.nan}
                ci=bootstrap_los_ci(y,pred,BOOTSTRAP_REPS,np.random.default_rng(SEED+model_seed_value(mid)*73+(0 if split=="val" else 200000)))
                for metric,(lo,hi) in ci.items(): los[f"{metric}_ci95_low"]=lo; los[f"{metric}_ci95_high"]=hi
                los_frames.append(los)

    print("[4/6] Running comparative inference vs Model 32...", flush=True)
    # Build a map without mutating global model file structure.
    comp_preds={mid:parts for mid,parts in all_predictions.items() if mid != "32"}
    all_predictions["32"] = all_predictions["32"]
    thresholds={}
    for mid,parts in all_predictions.items():
        for task in BINARY_TASKS:
            d=parts["val"]; sub=d[d["task"].eq(task)].copy(); p=probability_series(sub,task); keep=valid_binary(sub,task,p)
            thresholds[(mid,task)] = threshold_at_max_f1(pd.to_numeric(sub.loc[keep,"target"],errors="coerce").to_numpy(int), p[keep])
    comparisons=evaluate_reference_comparisons(all_predictions, thresholds, SEED)

    print("[5/6] Evaluating locked Z2_tuned on eICU (if artifacts are present)...", flush=True)
    eicu_binary, eicu_los, eicu_roc, eicu_pr, eicu_cal, eicu_meta = evaluate_eicu(z2_val, {t:thresholds[("Z2_tuned",t)] for t in BINARY_TASKS}, outdir)

    print("[6/6] Writing complete statistical package...", flush=True)
    pd.DataFrame(summary_frames).to_csv(outdir/"complete_binary_statistics.csv", index=False)
    pd.DataFrame(los_frames).to_csv(outdir/"complete_los_statistics.csv", index=False)
    pd.DataFrame(threshold_frames).to_csv(outdir/"locked_validation_thresholds.csv", index=False)
    pd.DataFrame(roc_frames).to_csv(outdir/"roc_curve_points.csv", index=False)
    pd.DataFrame(pr_frames).to_csv(outdir/"pr_curve_points.csv", index=False)
    pd.DataFrame(cal_frames).to_csv(outdir/"calibration_curve_points.csv", index=False)
    comparisons.to_csv(outdir/"paired_comparisons_vs_model32.csv", index=False)
    eicu_binary.to_csv(outdir/"eicu_z2_complete_binary_statistics.csv", index=False)
    eicu_los.to_csv(outdir/"eicu_z2_complete_los_statistics.csv", index=False)
    eicu_roc.to_csv(outdir/"eicu_z2_roc_curve_points.csv", index=False)
    eicu_pr.to_csv(outdir/"eicu_z2_pr_curve_points.csv", index=False)
    eicu_cal.to_csv(outdir/"eicu_z2_calibration_curve_points.csv", index=False)

    metadata={
        "created_by":"complete_statistical_evaluation.py",
        "seed":SEED,
        "bootstrap_reps":BOOTSTRAP_REPS,
        "pairwise_bootstrap_reps":PAIRWISE_BOOTSTRAP_REPS,
        "threshold_rule":"validation max F1; smallest threshold on ties",
        "reference_model":"32_MoE_Modality_Fusion",
        "evaluated_binary_tasks":list(BINARY_TASKS),
        "z2_mimic_deterioration_definition":"derived native Model-Z probability-space 3-way noisy-OR: 1-(1-mortality)*(1-ventilation)*(1-vasopressor)",
        "discrimination_source":"native prediction; Z2 MIMIC deterioration uses native derived noisy-OR probability because no standalone deterioration prediction row is serialized",
        "probability_source":"native probability_mean when present",
        "binary_metrics":["AUROC","AUPRC","Brier","ECE","calibration intercept","calibration slope","sensitivity","specificity","precision","NPV","F1","TP","TN","FP","FN"],
        "binary_metric_ci":["AUROC","AUPRC","Brier","ECE","sensitivity","specificity","precision","NPV","F1"],
        "comparison_tests":["paired DeLong AUROC","McNemar exact thresholded classification","paired bootstrap delta tests"],
        "multiple_testing":"Benjamini-Hochberg FDR q-values grouped by split/task/test_type",
        "calibration_bins":ECE_BINS,
        "los_metrics":["MAE","RMSE","Median AE","P90 AE","mean error","R2"],
        "eicu":eicu_meta,
    }
    write_json(outdir/"complete_statistical_evaluation_metadata.json",metadata)

    # A compact README is useful for thesis auditability.
    readme=f"""# Complete Statistical Evaluation — FYP\n\nGenerated from existing prediction artifacts only. No retraining, refitting, or retuning was performed.\n\n## Threshold rule\nBinary thresholds are selected on validation only by maximizing F1; the smallest threshold is used for ties. The threshold is frozen for test and eICU external evaluation.\n\n## Binary metrics\nAUROC, AUPRC, Brier, ECE, calibration intercept/slope, sensitivity, specificity, precision/PPV, NPV, F1, TP/TN/FP/FN. AUROC/AUPRC/Brier/ECE and threshold metrics have percentile bootstrap 95% CIs.\n\n## Curves\nROC, precision-recall, and 10-bin uniform reliability/calibration curve points are saved as CSV.\n\n## Comparative inference\nModel 32 is the fixed reference. AUROC uses paired DeLong. Thresholded classification uses exact McNemar. AUPRC/Brier/F1/sensitivity/specificity/precision/NPV use paired bootstrap deltas with percentile 95% CI and a two-sided bootstrap sign p-value. Benjamini-Hochberg FDR q-values are reported within split/task/test-type groups.\n\n## LOS\nMAE, RMSE, median absolute error, P90 absolute error, mean error, R2, with percentile bootstrap CIs.\n\n## eICU\nLocked Z2_tuned external evaluation is included when the canonical prediction and label artifacts are present. No external tuning/refit is performed.\n\nFiles:\n- complete_binary_statistics.csv\n- complete_los_statistics.csv\n- locked_validation_thresholds.csv\n- paired_comparisons_vs_model32.csv\n- roc_curve_points.csv\n- pr_curve_points.csv\n- calibration_curve_points.csv\n- eicu_z2_complete_binary_statistics.csv\n- eicu_z2_complete_los_statistics.csv\n- complete_statistical_evaluation_metadata.json\n"""
    (outdir/"README.md").write_text(readme)

    if not args.no_plots:
        try:
            build_plots(pd.DataFrame(roc_frames), pd.DataFrame(pr_frames), pd.DataFrame(cal_frames), outdir)
        except Exception as exc:
            (outdir/"plot_warning.txt").write_text(str(exc))

    print(f"DONE: {outdir}", flush=True)


if __name__ == "__main__":
    main()
