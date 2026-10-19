#!/usr/bin/env python3
"""
FINAL THESIS-GRADE CHI MODEL / EXPERIMENT EXTRACTOR
==================================================

Purpose
-------
Build a complete, reproducible documentation package from the existing CHI
output tree WITHOUT retraining, retuning, changing predictions, or using the
test set for selection.

This script is intentionally broader than the earlier extractors. It records:

1. Every prediction CSV anywhere under outputs/ for the model families.
2. Final/aggregate model results as a clean registry.
3. Fixed and tuned baseline models 00-33 as actually present.
4. Proposed model family A-F and all saved proposed-model artifacts.
5. Family Y aggregate models, including A1-A7, Y/C/U aggregates when present.
6. Family Z models including A2, A3, Z1, Z2, Z3 and fixed/tuned variants.
7. Z4 / Z3 seed summaries and every available seed cache.
8. Z2 Ultimate outputs and validation-only status.
9. Every seed/training cache with seed, checkpoint, history, metadata and params.
10. Every tuning/trial directory and trial-like CSV/JSON artifact.
11. Every JSON/config/metadata file associated with a model/seed/trial.
12. Full-file SHA-256 hashes for duplicate detection (no weak head fingerprint).
13. Validation/test metrics calculated from prediction files.
14. Probability-aware calibration metrics:
      - Brier/ECE/0.5 threshold metrics ONLY when probability_mean exists.
      - Rank/prediction scores are NOT treated as probabilities.
15. A master artifact inventory so the thesis can state exactly what was saved.

Important
---------
- No data preprocessing is performed.
- No training is performed.
- No model is selected by test performance.
- This script inventories and scores what is already present.
- If a previous run overwrote files, no extractor can recover the overwritten run.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
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

# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------

PROJECT_ROOT = Path("/home/abdullah2025/fyp_project")
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
DOC_ROOT = OUTPUT_ROOT / "documentation_final"

# We include these model-result families dynamically. Other pipeline/audit
# artifacts remain in the inventory, but are not treated as final models.
MODEL_ROOT_PATTERNS = (
    "baseline_family_*",
    "proposed_model",
    "proposed_model_family",
    "model_y_family",
    "model_y_v2_stacked_ensemble",
    "model_z_family",
    "model_z4_final",
    "model_z2_ultimate",
)

BINARY_TASKS = (
    "mortality_24h",
    "ventilation_onset_24h",
    "vasopressor_onset_24h",
    "readmission_30d_24h",
)
LOS_TASK = "los_hours_24h"
TASKS = BINARY_TASKS + (LOS_TASK,)

# Frozen benchmark stay counts. These are anomaly checks, not scoring rules.
EXPECTED_VAL_STAYS = 7_417
EXPECTED_TEST_STAYS = 14_887

# Rank/ensemble outputs must not be treated as calibrated probabilities unless
# a genuine probability_mean column is present. Ordinary baseline/proposed
# prediction columns are treated as probabilities only when all values are in
# [0, 1].
RANK_BLEND_NAME_PATTERNS = (
    "rank",
    "weighted_rank",
    "y0",
    "z1",
    "z2",
    "z3",
    "z4",
    "a3",
)

# File names that commonly contain parameters / provenance.
PARAM_FILENAMES = {
    "metadata.json",
    "config.json",
    "configuration.json",
    "params.json",
    "parameters.json",
    "hyperparameters.json",
    "best_params.json",
    "selected_config.json",
    "selected_configuration.json",
    "best_configuration.json",
    "args.json",
    "run_config.json",
    "trial_config.json",
    "settings.json",
}

HISTORY_NAMES = {
    "training_history.csv",
    "training_history.json",
    "history.csv",
    "history.json",
}

TRIAL_NAME_RE = re.compile(
    r"(?:^|[_\-/])(?:trial[_-]?|t)(\d+)(?:$|[_\-/])",
    flags=re.IGNORECASE,
)
SEED_RE = re.compile(r"seed[_-]?(\d+)", flags=re.IGNORECASE)


# ---------------------------------------------------------------------
# GENERIC HELPERS
# ---------------------------------------------------------------------

def safe_json_load(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            return json.load(f)
    except Exception:
        return None


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Full-file content hash. This replaces the old weak 200-line fingerprint."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def relative(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except Exception:
        return str(path)


def json_compact(value: Any, max_chars: int = 20000) -> str:
    try:
        s = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        s = repr(value)
    return s if len(s) <= max_chars else s[:max_chars] + "...<TRUNCATED>"


def natural_key(text: str):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", text)]


def parse_seed(text: str) -> str:
    m = SEED_RE.search(text)
    return m.group(1) if m else ""


def parse_trial(text: str) -> str:
    m = TRIAL_NAME_RE.search(text)
    return m.group(1) if m else ""


def is_model_root(path: Path) -> bool:
    try:
        rel = path.relative_to(OUTPUT_ROOT)
    except Exception:
        return False
    first = rel.parts[0] if rel.parts else ""
    if first.startswith("baseline_family_"):
        return True
    return first in {
        "proposed_model",
        "proposed_model_family",
        "model_y_family",
        "model_y_v2_stacked_ensemble",
        "model_z_family",
        "model_z4_final",
        "model_z2_ultimate",
    }


def looks_like_prediction_file(path: Path) -> bool:
    name = path.name.lower()
    if path.suffix.lower() != ".csv":
        return False
    return (
        "prediction" in name
        or name in {"predictions.csv", "test_predictions.csv", "validation_predictions.csv"}
    )


def looks_like_trial_artifact(path: Path) -> bool:
    s = str(path).lower()
    name = path.name.lower()
    return (
        "trial" in s
        or "tuning" in s
        or "optuna" in s
        or "search" in s
        or parse_trial(name) != ""
        or "best_params" in name
        or name in {"trial_results.csv", "tuning_results.csv", "search_results.csv"}
    )


# ---------------------------------------------------------------------
# PARAMETER / PROVENANCE EXTRACTION
# ---------------------------------------------------------------------

def collect_nearby_param_files(model_dir: Path) -> list[Path]:
    """
    Collect parameter/provenance files in the model directory and one/two
    parent levels. We do not recursively swallow the entire output tree.
    """
    candidates: set[Path] = set()

    roots = [model_dir]
    if model_dir.parent != model_dir:
        roots.append(model_dir.parent)
    if model_dir.parent.parent != model_dir.parent:
        roots.append(model_dir.parent.parent)

    for root in roots:
        if not root.exists():
            continue
        try:
            for p in root.iterdir():
                if p.is_file() and p.name.lower() in PARAM_FILENAMES:
                    candidates.add(p)
                elif p.is_file() and (
                    "param" in p.name.lower()
                    or "config" in p.name.lower()
                    or "metadata" in p.name.lower()
                ) and p.suffix.lower() in {".json", ".yaml", ".yml", ".txt"}:
                    candidates.add(p)
        except OSError:
            pass

    return sorted(candidates, key=lambda p: natural_key(str(p)))


def collect_history_files(model_dir: Path) -> list[Path]:
    roots = [model_dir]
    if model_dir.parent != model_dir:
        roots.append(model_dir.parent)

    found: set[Path] = set()
    for root in roots:
        for name in HISTORY_NAMES:
            p = root / name
            if p.exists():
                found.add(p)
    return sorted(found, key=lambda p: natural_key(str(p)))


def summarize_training_history(path: Path) -> dict[str, Any]:
    out = {
        "history_file": relative(path),
        "history_rows": "",
        "best_epoch": "",
        "best_val_metric": "",
        "history_columns": "",
    }

    try:
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        else:
            raw = safe_json_load(path)
            if isinstance(raw, dict):
                # Common formats: {"history": [...]}, {"epochs": [...]},
                # or {"metric_name": [...]}. Try to normalize.
                if isinstance(raw.get("history"), list):
                    df = pd.DataFrame(raw["history"])
                elif isinstance(raw.get("epochs"), list):
                    df = pd.DataFrame(raw["epochs"])
                else:
                    # If dict values are same-length lists, make a frame.
                    list_values = {
                        k: v for k, v in raw.items()
                        if isinstance(v, list)
                    }
                    df = pd.DataFrame(list_values) if list_values else pd.DataFrame()
            elif isinstance(raw, list):
                df = pd.DataFrame(raw)
            else:
                df = pd.DataFrame()

        out["history_rows"] = int(len(df))
        out["history_columns"] = json_compact(list(df.columns), 4000)

        if df.empty:
            return out

        epoch_col = next(
            (c for c in df.columns if str(c).lower() in {"epoch", "epochs", "step"}),
            None,
        )

        metric_candidates = [
            c for c in df.columns
            if any(
                token in str(c).lower()
                for token in (
                    "val_mean_binary_auprc",
                    "val_auprc",
                    "val_auc",
                    "val_loss",
                    "validation_loss",
                    "val_metric",
                    "selection",
                )
            )
        ]

        if metric_candidates:
            col = metric_candidates[0]
            vals = pd.to_numeric(df[col], errors="coerce")
            if vals.notna().any():
                # Losses are minimized; AUPRC/AUC/selection metrics maximized.
                c_lower = str(col).lower()
                if "loss" in c_lower:
                    idx = vals.idxmin()
                else:
                    idx = vals.idxmax()
                out["best_val_metric"] = float(vals.loc[idx])
                if epoch_col is not None:
                    try:
                        out["best_epoch"] = int(df.loc[idx, epoch_col])
                    except Exception:
                        out["best_epoch"] = str(df.loc[idx, epoch_col])

        return out
    except Exception as exc:
        out["history_error"] = str(exc)
        return out


def collect_parameters(model_dir: Path) -> tuple[list[str], str]:
    files = collect_nearby_param_files(model_dir)
    merged: dict[str, Any] = {}
    used = []

    for p in files:
        raw = safe_json_load(p)
        if raw is None:
            used.append(relative(p))
            continue
        merged[relative(p)] = raw
        used.append(relative(p))

    return used, json_compact(merged, max_chars=50000)


# ---------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------

def ece_score(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    if len(y) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    n = len(y)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p <= hi if hi == 1.0 else p < hi)
        if not np.any(m):
            continue
        total += (m.sum() / n) * abs(float(p[m].mean()) - float(y[m].mean()))
    return float(total)


def threshold_metrics(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    if len(y) == 0:
        return {
            "precision": np.nan, "recall": np.nan, "f1": np.nan,
            "accuracy": np.nan, "tn": np.nan, "fp": np.nan,
            "fn": np.nan, "tp": np.nan,
        }

    pred = (np.clip(p, 0.0, 1.0) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y.astype(int), pred, labels=[0, 1]).ravel()

    return {
        "precision": float(precision_score(y.astype(int), pred, zero_division=0)),
        "recall": float(recall_score(y.astype(int), pred, zero_division=0)),
        "f1": float(f1_score(y.astype(int), pred, zero_division=0)),
        "accuracy": float(accuracy_score(y.astype(int), pred)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def is_rank_blend_prediction(path: Path) -> bool:
    """Heuristic guard: rank/ensemble outputs are not probabilities by default."""
    text = str(path).lower()
    return any(token in text for token in RANK_BLEND_NAME_PATTERNS)


def find_named_sibling(path: Path, suffix_name: str) -> Path | None:
    """
    Find a sibling artifact by replacing _test/_validation or by using the
    same base name inside the same directory.
    """
    stem = path.stem
    base = re.sub(r"_(?:test|validation|val)_?(?:predictions|summary)$", "", stem, flags=re.I)

    candidates = [
        path.parent / suffix_name,
        path.parent / f"{base}_{suffix_name}",
        path.parent / f"{base}_{suffix_name.replace('.csv', '_predictions.csv')}",
        path.parent / f"{base}_{suffix_name.replace('.csv', '_summary.csv')}",
    ]

    # Explicit common pairings.
    if "test_predictions" in path.name.lower():
        candidates.extend([
            path.parent / "validation_predictions.csv",
            path.parent / f"{base}_validation_predictions.csv",
        ])
    if "test_summary" in path.name.lower():
        candidates.extend([
            path.parent / "Z3_seed_summary_validation.csv",
            path.parent / f"{base}_validation_summary.csv",
        ])

    for p in candidates:
        if p.exists() and p != path:
            return p
    return None


def score_named_summary_file(path: Path) -> dict[str, Any]:
    """
    Score already-computed task-summary tables such as
    Z3_seed_summary_test.csv. The table remains an audit artifact; no
    recalculation is invented when raw predictions are unavailable.
    """
    rec: dict[str, Any] = {
        "prediction_file": relative(path),
        "sha256": sha256_file(path),
        "summary_artifact": True,
    }

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        rec["read_error"] = str(exc)
        return rec

    rec["rows_total"] = int(len(df))
    rec["columns"] = json_compact(list(df.columns), 12000)
    rec["not_a_task_prediction_file"] = True
    if df.empty:
        return rec

    row = df.iloc[0]

    # Preserve all numeric summary columns.
    for col in df.columns:
        val = row[col]
        if pd.isna(val):
            continue
        if isinstance(val, (int, float, np.integer, np.floating)):
            rec[f"summary_{col}"] = float(val)

    # Try common task-level names.
    task_tokens = {
        "mortality_24h": ("mortality",),
        "ventilation_onset_24h": ("ventilation", "vent"),
        "vasopressor_onset_24h": ("vasopressor", "vaso"),
        "readmission_30d_24h": ("readmission", "readm"),
        "los_hours_24h": ("los",),
    }

    for task, tokens in task_tokens.items():
        relevant = [
            c for c in df.columns
            if any(tok in str(c).lower() for tok in tokens)
        ]
        if not relevant:
            continue

        if task in BINARY_TASKS:
            for metric in ("auprc", "auroc", "brier", "ece"):
                cols = [c for c in relevant if metric in str(c).lower()]
                if cols:
                    rec[f"test_{task}_{metric}"] = safe_float(row[cols[0]])
        else:
            cols = [c for c in relevant if "mae" in str(c).lower()]
            if cols:
                rec[f"test_{task}_mae"] = safe_float(row[cols[0]])

    ap_cols = [c for c in df.columns if "mean" in str(c).lower() and "auprc" in str(c).lower()]
    auc_cols = [c for c in df.columns if "mean" in str(c).lower() and "auroc" in str(c).lower()]
    if ap_cols:
        rec["test_mean_binary_auprc"] = safe_float(row[ap_cols[0]])
    if auc_cols:
        rec["test_mean_binary_auroc"] = safe_float(row[auc_cols[0]])

    return rec


def score_prediction_file(path: Path) -> dict[str, Any]:
    """
    Score one prediction file.

    A file may contain validation and test rows internally. We therefore
    calculate the two splits independently.

    Calibration rule:
      * Prefer probability_mean when sufficiently populated.
      * Otherwise ordinary probability-producing models use prediction itself
        only when it is a valid [0,1] probability.
      * Rank blends do NOT use prediction as probability.
    """
    rec: dict[str, Any] = {
        "prediction_file": relative(path),
        "sha256": sha256_file(path),
        "rank_blend_artifact": is_rank_blend_prediction(path),
    }

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        rec["read_error"] = str(exc)
        return rec

    rec["rows_total"] = int(len(df))
    rec["columns"] = json_compact(list(df.columns), 12000)

    required = {"task", "target", "prediction"}
    if not required.issubset(df.columns):
        rec["not_a_task_prediction_file"] = True
        return rec

    rec["not_a_task_prediction_file"] = False
    rec["has_probability_mean"] = "probability_mean" in df.columns
    rec["has_validity"] = "validity" in df.columns
    if "split" in df.columns:
        rec["split_values"] = json_compact(
            sorted(df["split"].astype(str).unique().tolist())
        )
    else:
        rec["split_values"] = "[]"

    probability_warnings: list[str] = []

    for split, expected_stays in (
        ("val", EXPECTED_VAL_STAYS),
        ("test", EXPECTED_TEST_STAYS),
    ):
        z = df
        if "split" in z.columns:
            z = z[z["split"].astype(str) == split]

        rec[f"{split}_rows_raw"] = int(len(z))

        if "stay_id" in z.columns:
            unique_stays = int(z["stay_id"].nunique(dropna=True))
            rec[f"{split}_unique_stays"] = unique_stays
            if unique_stays not in (0, expected_stays):
                rec[f"{split}_row_count_anomaly"] = True
                rec[f"{split}_row_count_anomaly_detail"] = (
                    f"unique_stays={unique_stays}, expected={expected_stays}"
                )

        for task in TASKS:
            zt = z[z["task"].astype(str) == task]
            if "validity" in zt.columns:
                zt = zt[zt["validity"].astype(bool)]

            if len(zt) == 0:
                rec[f"{split}_{task}_n"] = 0
                continue

            y = pd.to_numeric(zt["target"], errors="coerce").to_numpy(float)
            pred = pd.to_numeric(zt["prediction"], errors="coerce").to_numpy(float)

            probability = None
            if "probability_mean" in zt.columns:
                prob_arr = pd.to_numeric(
                    zt["probability_mean"], errors="coerce"
                ).to_numpy(float)
                finite_count = int(np.isfinite(prob_arr).sum())

                # Guard against a column that exists but is effectively empty.
                if finite_count >= max(10, int(0.5 * len(prob_arr))):
                    probability = prob_arr
                else:
                    probability_warnings.append(
                        f"{split}/{task}: probability_mean has only "
                        f"{finite_count}/{len(prob_arr)} finite values; treated as missing"
                    )

            keep = np.isfinite(y) & np.isfinite(pred)
            if probability is not None:
                keep &= np.isfinite(probability)

            y = y[keep]
            pred = pred[keep]
            if probability is not None:
                probability = np.clip(probability[keep], 0.0, 1.0)

            n = int(len(y))
            rec[f"{split}_{task}_n"] = n

            if task in BINARY_TASKS:
                if n == 0 or len(np.unique(y)) < 2:
                    for metric in (
                        "auprc", "auroc", "brier", "ece", "prevalence",
                        "precision", "recall", "f1", "accuracy",
                        "tn", "fp", "fn", "tp",
                    ):
                        rec[f"{split}_{task}_{metric}"] = np.nan
                    continue

                y_int = y.astype(int)

                # Ranking metrics always use the actual prediction score.
                ap = float(average_precision_score(y_int, pred))
                auc = float(roc_auc_score(y_int, pred))

                rec[f"{split}_{task}_auprc"] = ap
                rec[f"{split}_{task}_auroc"] = auc
                rec[f"{split}_{task}_prevalence"] = float(y_int.mean())

                # Calibration metrics use a probability only.
                cal = None
                cal_source = ""

                if probability is not None:
                    cal = probability
                    cal_source = "probability_mean"
                elif not rec["rank_blend_artifact"]:
                    if (
                        np.all(np.isfinite(pred))
                        and np.all((pred >= 0.0) & (pred <= 1.0))
                    ):
                        cal = np.clip(pred, 0.0, 1.0)
                        cal_source = "prediction_probability"
                    else:
                        cal_source = "unavailable_prediction_not_probability"
                        probability_warnings.append(
                            f"{split}/{task}: prediction is not a valid [0,1] probability"
                        )
                else:
                    cal_source = "unavailable_rank_blend_without_probability_mean"

                rec[f"{split}_{task}_calibration_source"] = cal_source

                if cal is not None:
                    rec[f"{split}_{task}_brier"] = float(
                        brier_score_loss(y_int, cal)
                    )
                    rec[f"{split}_{task}_ece"] = ece_score(y_int, cal)

                    for k, v in threshold_metrics(y_int, cal, 0.5).items():
                        rec[f"{split}_{task}_{k}"] = v
                else:
                    for metric in (
                        "brier", "ece", "precision", "recall",
                        "f1", "accuracy", "tn", "fp", "fn", "tp",
                    ):
                        rec[f"{split}_{task}_{metric}"] = np.nan

            else:
                if n == 0:
                    for metric in ("mae", "rmse", "median_ae", "r2"):
                        rec[f"{split}_{task}_{metric}"] = np.nan
                    continue

                rec[f"{split}_{task}_mae"] = float(
                    mean_absolute_error(y, pred)
                )
                rec[f"{split}_{task}_rmse"] = float(
                    np.sqrt(mean_squared_error(y, pred))
                )
                rec[f"{split}_{task}_median_ae"] = float(
                    median_absolute_error(y, pred)
                )
                rec[f"{split}_{task}_r2"] = (
                    float(r2_score(y, pred)) if n > 1 else np.nan
                )

    for split in ("val", "test"):
        aps = [rec.get(f"{split}_{task}_auprc", np.nan) for task in BINARY_TASKS]
        aucs = [rec.get(f"{split}_{task}_auroc", np.nan) for task in BINARY_TASKS]
        briers = [rec.get(f"{split}_{task}_brier", np.nan) for task in BINARY_TASKS]
        eces = [rec.get(f"{split}_{task}_ece", np.nan) for task in BINARY_TASKS]

        rec[f"{split}_mean_binary_auprc"] = (
            float(np.nanmean(aps)) if np.isfinite(aps).any() else np.nan
        )
        rec[f"{split}_mean_binary_auroc"] = (
            float(np.nanmean(aucs)) if np.isfinite(aucs).any() else np.nan
        )
        rec[f"{split}_mean_binary_brier"] = (
            float(np.nanmean(briers)) if np.isfinite(briers).any() else np.nan
        )
        rec[f"{split}_mean_binary_ece"] = (
            float(np.nanmean(eces)) if np.isfinite(eces).any() else np.nan
        )

    if probability_warnings:
        rec["probability_mean_warnings"] = json_compact(
            probability_warnings, 10000
        )

    return rec


# ---------------------------------------------------------------------
# ARTIFACT CLASSIFICATION
# ---------------------------------------------------------------------

def classify_prediction(path: Path) -> dict[str, str]:
    """
    Assign a documentation role without relying on a brittle hard-coded list.
    """
    rel = path.relative_to(OUTPUT_ROOT)
    parts = rel.parts
    text = str(rel).lower()

    family = parts[0] if parts else ""

    role = "prediction_artifact"
    stage = "unknown"

    if "seed_" in text or "seed" in text:
        role = "seed_prediction"
        stage = "seed"
    elif "trial" in text or re.search(r"/t\d+(?:/|_)", text):
        role = "trial_prediction"
        stage = "trial"
    elif "final_selected_configuration" in text:
        role = "final_model_prediction"
        stage = "tuned_final" if "/tuned/" in text else "final"
    elif "validation_predictions.csv" in path.name.lower():
        role = "validation_prediction"
        stage = "validation"
    elif "test_predictions.csv" in path.name.lower():
        role = "aggregate_or_test_prediction"
        stage = "aggregate_or_test"
    elif family in {
        "model_y_family",
        "model_y_v2_stacked_ensemble",
        "model_z_family",
        "model_z4_final",
        "model_z2_ultimate",
    } and path.parent == OUTPUT_ROOT / family:
        role = "family_aggregate_prediction"
        stage = "aggregate"

    # Baseline fixed directories may have prediction files without
    # "test_predictions" nomenclature.
    if family.startswith("baseline_family_") and "/fixed/" in text:
        if role == "prediction_artifact":
            role = "fixed_model_prediction"
            stage = "fixed"

    return {
        "family_root": family,
        "role": role,
        "stage": stage,
    }


def infer_model_name(path: Path) -> str:
    rel = path.relative_to(OUTPUT_ROOT)
    parts = list(rel.parts)

    # Family-level *_test_predictions.csv / *_validation_predictions.csv.
    stem = path.stem
    stem = re.sub(r"_(?:test|validation)_predictions$", "", stem, flags=re.I)
    if stem not in {"predictions", "test", "validation"}:
        if len(parts) <= 2:
            return stem

    # final_selected_configuration: model name is normally its grandparent.
    if "final_selected_configuration" in parts:
        idx = parts.index("final_selected_configuration")
        if idx > 0:
            return parts[idx - 1]

    # baseline fixed/tuned directory: direct model directory.
    if "fixed" in parts or "tuned" in parts:
        for key in ("fixed", "tuned"):
            if key in parts:
                idx = parts.index(key)
                if idx + 1 < len(parts):
                    return parts[idx + 1]

    # seed/trial directories.
    if any("seed" in p.lower() for p in parts):
        for p in reversed(parts[:-1]):
            if "seed" not in p.lower():
                return p

    if any("trial" in p.lower() for p in parts):
        for p in reversed(parts[:-1]):
            if "trial" not in p.lower():
                return p

    # Family aggregate filename.
    return stem


def infer_model_id(model_name: str) -> str:
    m = re.match(r"^(\d{2,3})[_-]", model_name)
    return m.group(1) if m else ""


def infer_seed_from_path(path: Path) -> str:
    return parse_seed(str(path))


def infer_trial_from_path(path: Path) -> str:
    return parse_trial(str(path))


# ---------------------------------------------------------------------
# INVENTORIES
# ---------------------------------------------------------------------

def inventory_all_artifacts() -> pd.DataFrame:
    rows = []

    for path in sorted(OUTPUT_ROOT.rglob("*"), key=lambda p: natural_key(str(p))):
        if not path.is_file():
            continue

        # Never inventory the documentation output we are currently creating.
        try:
            path.relative_to(DOC_ROOT)
            continue
        except ValueError:
            pass

        try:
            stat = path.stat()
        except OSError:
            continue

        suffix = path.suffix.lower()

        rows.append({
            "artifact_path": relative(path),
            "artifact_type": suffix.lstrip("."),
            "filename": path.name,
            "parent_directory": relative(path.parent),
            "size_bytes": int(stat.st_size),
            "modified_time": pd.Timestamp(stat.st_mtime, unit="s").isoformat(),
            "sha256": sha256_file(path),
            "is_prediction": bool(looks_like_prediction_file(path)),
            "is_param_or_config": bool(
                path.name.lower() in PARAM_FILENAMES
                or "config" in path.name.lower()
                or "param" in path.name.lower()
                or "metadata" in path.name.lower()
            ),
            "is_history": bool(path.name.lower() in HISTORY_NAMES),
            "is_trial_artifact": bool(looks_like_trial_artifact(path)),
            "is_checkpoint": suffix in {".pt", ".pth", ".ckpt"},
        })

    return pd.DataFrame(rows)


def merge_validation_into_record(
    test_rec: dict[str, Any],
    val_path: Path | None,
) -> dict[str, Any]:
    """Merge a separately stored validation prediction file into a test record."""
    if val_path is None:
        return test_rec

    val_rec = score_prediction_file(val_path)
    out = dict(test_rec)
    out["validation_prediction_file"] = relative(val_path)
    out["validation_sha256"] = val_rec.get("sha256", "")

    for k, v in val_rec.items():
        if k.startswith("val_"):
            out[k] = v
        elif k == "probability_mean_warnings":
            out["validation_probability_mean_warnings"] = v

    return out


def merge_summary_validation_into_record(
    test_rec: dict[str, Any],
    val_path: Path | None,
) -> dict[str, Any]:
    if val_path is None:
        return test_rec

    val_rec = score_named_summary_file(val_path)
    out = dict(test_rec)
    out["validation_summary_file"] = relative(val_path)
    out["validation_summary_sha256"] = val_rec.get("sha256", "")

    # Summary files are already metric tables; copy their summary columns.
    for k, v in val_rec.items():
        if str(k).startswith("summary_"):
            out[f"validation_{k}"] = v

    # If the summary includes explicit val_* metrics, preserve those too.
    for k, v in val_rec.items():
        if str(k).startswith("val_"):
            out[k] = v

    return out


def prediction_inventory() -> pd.DataFrame:
    rows = []
    seen_paths: set[Path] = set()

    all_csvs = sorted(
        list(OUTPUT_ROOT.rglob("*.csv")),
        key=lambda p: natural_key(str(p)),
    )

    for path in all_csvs:
        if path in seen_paths:
            continue

        try:
            if not is_model_root(path):
                continue

            lname = path.name.lower()

            # Standard task prediction files.
            if looks_like_prediction_file(path):
                cls = classify_prediction(path)
                scored = score_prediction_file(path)
                model_name = infer_model_name(path)

                rec = {
                    "family": cls["family_root"],
                    "model_name": model_name,
                    "model_id": infer_model_id(model_name),
                    "role": cls["role"],
                    "stage": cls["stage"],
                    "seed": infer_seed_from_path(path),
                    "trial": infer_trial_from_path(path),
                    **scored,
                }

                if "test_predictions" in lname:
                    vp = find_named_sibling(
                        path, "validation_predictions.csv"
                    )
                    if vp is not None:
                        rec = merge_validation_into_record(rec, vp)
                        seen_paths.add(vp)

                rows.append(rec)
                seen_paths.add(path)
                continue

            # Z3 seed summary files are important even though their filename
            # does not contain "prediction".
            if lname.startswith("z3_seed_summary_") and lname.endswith(".csv"):
                rec = score_named_summary_file(path)
                stem = path.stem
                model_name = "Z3_seed8_summary"
                if "validation" in stem.lower():
                    rec["role"] = "summary_metric_artifact_validation"
                    rec["stage"] = "summary_validation"
                else:
                    rec["role"] = "summary_metric_artifact_test"
                    rec["stage"] = "summary_test"
                    vp = find_named_sibling(
                        path, "Z3_seed_summary_validation.csv"
                    )
                    if vp is not None:
                        rec = merge_summary_validation_into_record(rec, vp)
                        seen_paths.add(vp)

                rec["family"] = path.relative_to(OUTPUT_ROOT).parts[0]
                rec["model_name"] = model_name
                rec["model_id"] = ""
                rec["seed"] = ""
                rec["trial"] = ""
                rows.append(rec)
                seen_paths.add(path)
        except Exception as exc:
            rows.append({
                "family": "",
                "model_name": "",
                "model_id": "",
                "role": "scoring_error",
                "stage": "",
                "prediction_file": relative(path),
                "error": str(exc),
            })
            seen_paths.add(path)

    return pd.DataFrame(rows)


def seed_inventory() -> pd.DataFrame:
    """
    One row per checkpoint-bearing seed/training cache anywhere under model
    families. These rows are documentation artifacts, not final performance
    rows unless predictions also exist.
    """
    rows = []

    candidate_dirs = set()
    for p in OUTPUT_ROOT.rglob("*"):
        if not p.is_dir():
            continue
        if any(
            (p / name).exists()
            for name in ("checkpoint.pt", "best_checkpoint.pt", "model.pt", "model.pth")
        ):
            candidate_dirs.add(p)

    for d in sorted(candidate_dirs, key=lambda p: natural_key(str(p))):
        if not is_model_root(d):
            continue

        params_files, params_json = collect_parameters(d)
        histories = collect_history_files(d)

        history_summaries = [
            summarize_training_history(h) for h in histories
        ]

        # checkpoint file summary
        checkpoints = [
            p for p in d.iterdir()
            if p.is_file() and p.suffix.lower() in {".pt", ".pth", ".ckpt"}
        ]

        metadata_files = [
            p for p in d.iterdir()
            if p.is_file() and p.name.lower() == "metadata.json"
        ]

        rows.append({
            "family": d.relative_to(OUTPUT_ROOT).parts[0],
            "seed": infer_seed_from_path(d),
            "trial": infer_trial_from_path(d),
            "seed_or_run_directory": relative(d),
            "has_checkpoint": bool(checkpoints),
            "checkpoint_files": json_compact([relative(x) for x in checkpoints], 4000),
            "has_metadata": bool(metadata_files),
            "metadata_files": json_compact([relative(x) for x in metadata_files], 4000),
            "history_files": json_compact([relative(x) for x in histories], 4000),
            "history_summaries": json_compact(history_summaries, 12000),
            "parameter_files": json_compact(params_files, 8000),
            "parameters_json": params_json,
            "has_predictions": any(
                looks_like_prediction_file(x) for x in d.glob("*.csv")
            ),
        })

    return pd.DataFrame(rows)


def trial_inventory() -> pd.DataFrame:
    rows = []

    trial_dirs: set[Path] = set()
    trial_files: set[Path] = set()

    for p in OUTPUT_ROOT.rglob("*"):
        if not is_model_root(p):
            continue
        s = str(p).lower()

        if p.is_dir() and (
            "trial" in s or "tuning" in s or "optuna" in s
        ):
            trial_dirs.add(p)

        if p.is_file() and (
            "trial" in p.name.lower()
            or "tuning" in p.name.lower()
            or "search" in p.name.lower()
            or "best_params" in p.name.lower()
        ):
            trial_files.add(p)

    for d in sorted(trial_dirs, key=lambda p: natural_key(str(p))):
        children = [x for x in d.rglob("*") if x.is_file()]
        params_files, params_json = collect_parameters(d)
        rows.append({
            "artifact_kind": "trial_directory",
            "family": d.relative_to(OUTPUT_ROOT).parts[0],
            "trial": infer_trial_from_path(d),
            "seed": infer_seed_from_path(d),
            "path": relative(d),
            "file_count": len(children),
            "parameter_files": json_compact(params_files, 8000),
            "parameters_json": params_json,
        })

    for p in sorted(trial_files, key=lambda p: natural_key(str(p))):
        if p in trial_dirs:
            continue
        rows.append({
            "artifact_kind": "trial_or_tuning_file",
            "family": p.relative_to(OUTPUT_ROOT).parts[0],
            "trial": infer_trial_from_path(p),
            "seed": infer_seed_from_path(p),
            "path": relative(p),
            "file_count": 1,
            "sha256": sha256_file(p),
            "parameters_json": "",
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# FINAL MODEL REGISTRY
# ---------------------------------------------------------------------

def choose_final_prediction_rows(preds: pd.DataFrame) -> pd.DataFrame:
    """
    Create the thesis-facing registry without collapsing identical prediction
    files across different experiment families.

    Exact byte-identical files can be legitimate aliases/copies (for example
    the Z4 copy of Z2_tuned). Those aliases are retained for auditability and
    labelled in the registry. Only seed/trial-only prediction rows are removed
    from the final model table.
    """
    if preds.empty:
        return preds.copy()

    excluded_roles = {
        "seed_prediction",
        "trial_prediction",
        "scoring_error",
    }
    z = preds[~preds["role"].isin(excluded_roles)].copy()

    if z.empty:
        return z

    priority = {
        "final_model_prediction": 100,
        "family_aggregate_prediction": 95,
        "aggregate_or_test_prediction": 90,
        "summary_metric_artifact_test": 80,
        "summary_metric_artifact_validation": 79,
        "fixed_model_prediction": 75,
        "validation_prediction": 70,
        "prediction_artifact": 50,
    }
    z["_priority"] = z["role"].map(priority).fillna(0)

    z["_has_test"] = z.get(
        "test_mean_binary_auprc",
        pd.Series(index=z.index, dtype=float),
    ).notna()
    z["_has_val"] = z.get(
        "val_mean_binary_auprc",
        pd.Series(index=z.index, dtype=float),
    ).notna()

    # Deterministic ordering, but DO NOT drop cross-family duplicates.
    z = z.sort_values(
        ["family", "model_name", "stage", "_priority", "_has_test", "_has_val"],
        ascending=[True, True, True, False, False, False],
        key=lambda s: s.astype(str).str.lower(),
        kind="mergesort",
    ).reset_index(drop=True)

    # Mark full-file duplicate aliases without deleting them.
    if "sha256" in z.columns:
        counts = z["sha256"].value_counts(dropna=False)
        z["duplicate_sha256_count"] = z["sha256"].map(counts).astype(int)

        canonical_by_hash = {}
        aliases = []
        for _, row in z.iterrows():
            h = row.get("sha256", "")
            path = row.get("prediction_file", "")
            if not h or pd.isna(h):
                aliases.append("")
                continue
            if h not in canonical_by_hash:
                canonical_by_hash[h] = path
                aliases.append("")
            else:
                aliases.append(canonical_by_hash[h])
        z["duplicate_alias_of"] = aliases

    return z.drop(
        columns=["_priority", "_has_test", "_has_val"],
        errors="ignore",
    )


def attach_model_metadata(final_df: pd.DataFrame) -> pd.DataFrame:
    if final_df.empty:
        return final_df

    rows = []

    for _, rec in final_df.iterrows():
        pred_path = PROJECT_ROOT / str(rec["prediction_file"])

        model_dir = pred_path.parent
        if "final_selected_configuration" in pred_path.parts:
            idx = pred_path.parts.index("final_selected_configuration")
            model_dir = PROJECT_ROOT.joinpath(*pred_path.parts[:idx + 1])
        elif pred_path.parent.exists():
            model_dir = pred_path.parent

        params_files, params_json = collect_parameters(model_dir)
        histories = collect_history_files(model_dir)
        hist = [summarize_training_history(h) for h in histories]

        row = dict(rec)
        row["parameter_files"] = json_compact(params_files, 8000)
        row["parameters_json"] = params_json
        row["training_history_files"] = json_compact(
            [relative(h) for h in histories], 8000
        )
        row["training_history_summary"] = json_compact(hist, 12000)

        # Keep related checkpoint/config/weight/recipe files visible for thesis
        # provenance. This is an inventory list, not a claim that each file is
        # used by the model.
        related = []
        for q in model_dir.iterdir() if model_dir.exists() else []:
            if not q.is_file():
                continue
            ql = q.name.lower()
            if (
                q.suffix.lower() in {".json", ".yaml", ".yml", ".txt"}
                and any(
                    tok in ql for tok in (
                        "config", "param", "metadata", "weight", "recipe",
                        "gate", "selection", "calibration", "temperature",
                        "bootstrap", "summary", "audit",
                    )
                )
            ):
                related.append(relative(q))
        row["related_provenance_files"] = json_compact(
            sorted(related), 12000
        )

        # Best epoch / validation metric from first informative history file.
        row["best_epoch"] = ""
        row["best_val_metric_from_history"] = ""
        for h in hist:
            if h.get("best_epoch", "") != "":
                row["best_epoch"] = h["best_epoch"]
                row["best_val_metric_from_history"] = h.get("best_val_metric", "")
                break

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# DUPLICATE AUDIT
# ---------------------------------------------------------------------

def duplicate_audit(inventory: pd.DataFrame) -> pd.DataFrame:
    if inventory.empty:
        return pd.DataFrame()

    z = inventory[inventory["artifact_type"].eq("csv")].copy()
    if z.empty:
        return z

    groups = (
        z.groupby("sha256", dropna=False)["artifact_path"]
        .agg(list)
        .reset_index()
    )
    groups["n_copies"] = groups["artifact_path"].map(len)
    groups = groups[groups["n_copies"] > 1].copy()

    rows = []
    for _, g in groups.iterrows():
        for path in g["artifact_path"]:
            rows.append({
                "sha256": g["sha256"],
                "n_copies": int(g["n_copies"]),
                "artifact_path": path,
                "duplicate_group": "; ".join(g["artifact_path"]),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# THESIS REPORT
# ---------------------------------------------------------------------

def safe_float(v):
    try:
        return float(v)
    except Exception:
        return np.nan


def write_long_format(final_df: pd.DataFrame, out_path: Path) -> None:
    """One row per model/family/stage/split/task for thesis tables."""
    rows = []
    for _, r in final_df.iterrows():
        for split in ("val", "test"):
            for task in TASKS:
                n = r.get(f"{split}_{task}_n")
                if pd.isna(n) or n == 0:
                    continue

                row = {
                    "family": r.get("family", ""),
                    "model_name": r.get("model_name", ""),
                    "model_id": r.get("model_id", ""),
                    "role": r.get("role", ""),
                    "stage": r.get("stage", ""),
                    "split": split,
                    "task": task,
                    "n": int(n),
                    "prediction_file": r.get("prediction_file", ""),
                    "validation_prediction_file": r.get(
                        "validation_prediction_file", ""
                    ),
                    "sha256": r.get("sha256", ""),
                }

                if task in BINARY_TASKS:
                    for metric in (
                        "auprc", "auroc", "brier", "ece", "prevalence",
                        "precision", "recall", "f1", "accuracy",
                    ):
                        row[metric] = r.get(f"{split}_{task}_{metric}")
                    row["calibration_source"] = r.get(
                        f"{split}_{task}_calibration_source", ""
                    )
                else:
                    for metric in ("mae", "rmse", "median_ae", "r2"):
                        row[metric] = r.get(f"{split}_{task}_{metric}")

                rows.append(row)

    pd.DataFrame(rows).to_csv(out_path, index=False)


def missing_items_report(
    preds: pd.DataFrame,
    seeds: pd.DataFrame,
    inventory: pd.DataFrame,
) -> list[str]:
    lines = [
        "## I. Missing / incomplete artifacts",
        "",
    ]

    val_only, test_only, neither = [], [], []
    for _, r in preds.iterrows():
        # Summary-only artifacts (e.g. Z3 seed summary) are documented
        # separately and should not be labelled "validation-only models".
        if r.get("summary_artifact", False) is True:
            continue

        has_val = pd.notna(r.get("val_mean_binary_auprc"))
        has_test = pd.notna(r.get("test_mean_binary_auprc"))

        if has_val and not has_test:
            val_only.append(r.get("prediction_file", ""))
        elif has_test and not has_val:
            test_only.append(r.get("prediction_file", ""))
        elif not has_val and not has_test:
            neither.append(r.get("prediction_file", ""))

    lines.extend([
        f"### Val-only artifacts ({len(val_only)})",
        "Validation-only experiments are not automatically failures; Z2 Ultimate is intentionally validation-only.",
    ])
    lines.extend(f"- `{p}`" for p in val_only[:50])
    if len(val_only) > 50:
        lines.append(f"- ... +{len(val_only) - 50} more")

    lines.extend([
        "",
        f"### Test-only artifacts ({len(test_only)})",
        "",
    ])
    lines.extend(f"- `{p}`" for p in test_only[:50])
    if len(test_only) > 50:
        lines.append(f"- ... +{len(test_only) - 50} more")

    lines.extend([
        "",
        f"### Unscorable prediction artifacts ({len(neither)})",
        "",
    ])
    lines.extend(f"- `{p}`" for p in neither[:50])
    if len(neither) > 50:
        lines.append(f"- ... +{len(neither) - 50} more")

    lines.extend([
        "",
        "### Checkpoint/seed caches without saved predictions",
        "These are training artifacts and are documented in all_seed_training_runs.csv,",
        "but are not treated as independent final thesis models when their aggregate exists separately.",
    ])

    if not seeds.empty:
        missing = seeds[seeds["has_predictions"] == False]
        for _, r in missing.iterrows():
            lines.append(
                f"- `{r.get('seed_or_run_directory','')}` "
                f"(seed={r.get('seed','')}, trial={r.get('trial','')})"
            )

    return lines


def load_bootstrap_stats() -> pd.DataFrame:
    """
    Load preserved bootstrap/statistics tables without inventing intervals.
    The exact all_models_bootstrap_stats.csv is preferred if present.
    """
    exact = list(OUTPUT_ROOT.rglob("all_models_bootstrap_stats.csv"))
    candidates = exact[:]

    if not candidates:
        for p in OUTPUT_ROOT.rglob("*.csv"):
            s = p.name.lower()
            if "bootstrap" in s and ("stat" in s or "ci" in s):
                candidates.append(p)

    frames = []
    for p in sorted(set(candidates), key=lambda x: natural_key(str(x))):
        try:
            df = pd.read_csv(p)
            if df.empty:
                continue
            df = df.copy()
            df.insert(0, "source_file", relative(p))
            frames.append(df)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def merge_bootstrap_stats_into_models(
    final_df: pd.DataFrame,
    bootstrap_df: pd.DataFrame,
) -> pd.DataFrame:
    if final_df.empty or bootstrap_df.empty:
        return final_df

    b = bootstrap_df.copy()

    # Determine the most useful model-identification column.
    model_col = next(
        (
            c for c in
            ["model_name", "model", "name", "candidate", "ensemble"]
            if c in b.columns
        ),
        None,
    )
    if model_col is None:
        return final_df

    b["_model_key"] = (
        b[model_col].astype(str).str.strip().str.lower()
    )

    f = final_df.copy()
    f["_model_key"] = f["model_name"].astype(str).str.strip().str.lower()

    # Attach all bootstrap/stat columns except the identifying column, with a
    # bootstrap_ prefix so they cannot overwrite core performance metrics.
    keep = [
        c for c in b.columns
        if c not in {model_col, "_model_key"}
    ]

    b2 = b[["_model_key"] + keep].copy()
    rename = {
        c: c if c == "source_file" else f"bootstrap_{c}"
        for c in keep
    }
    b2 = b2.rename(columns=rename)

    # If multiple rows per model exist, preserve all values in a semicolon-
    # separated JSON-like string rather than silently choosing one.
    agg = {}
    for c in b2.columns:
        if c == "_model_key":
            continue
        agg[c] = lambda s: json_compact(
            [x for x in s.tolist() if not pd.isna(x)],
            12000,
        )

    b2 = b2.groupby("_model_key", dropna=False).agg(agg).reset_index()
    f = f.merge(b2, on="_model_key", how="left")

    return f.drop(columns=["_model_key"], errors="ignore")


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Render a DataFrame as Markdown without requiring the optional tabulate package."""
    if df is None or df.empty:
        return ""

    cols = [str(c) for c in df.columns]

    def cell(value):
        if pd.isna(value):
            return ""
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, row in df.iterrows():
        lines.append(
            "| " + " | ".join(cell(row[c]) for c in df.columns) + " |"
        )
    return "\n".join(lines)


def write_markdown_report(
    final_df: pd.DataFrame,
    preds: pd.DataFrame,
    seeds: pd.DataFrame,
    trials: pd.DataFrame,
    inventory: pd.DataFrame,
    duplicates: pd.DataFrame,
):
    lines = [
        "# Final Thesis Model / Experiment Registry",
        "",
        "Generated from the preserved CHI output tree without retraining or retuning.",
        "",
        "## A. Counts",
        "",
        f"- Total preserved output files: {len(inventory):,}",
        f"- Prediction/summary artifacts discovered: {len(preds):,}",
        f"- Final/aggregate rows after full-SHA-256 alias collapse: {len(final_df):,}",
        f"- Seed/checkpoint directories discovered: {len(seeds):,}",
        f"- Trial/tuning artifacts discovered: {len(trials):,}",
        f"- Duplicate CSV content groups (full SHA-256): "
        f"{duplicates['sha256'].nunique() if not duplicates.empty else 0:,}",
        "",
        "## LOCKED PRIMARY MODEL",
        "",
        "**`family_z / Z2_tuned`** is the locked primary model.",
        "",
        "Selection rule: validation mean binary AUPRC.",
        "The held-out test set is evaluated only after model selection is frozen.",
        "A higher test score in the descriptive ranking is not used to replace the locked model.",
        "",
        "## A2. Top of the test ranking (documentation only)",
        "",
    ]

    if not final_df.empty and "test_mean_binary_auprc" in final_df.columns:
        ranked = final_df.dropna(
            subset=["test_mean_binary_auprc"]
        ).copy()

        if "duplicate_alias_of" in ranked.columns:
            ranked = ranked[
                ranked["duplicate_alias_of"].fillna("").astype(str).eq("")
            ]

        ranked = ranked.sort_values(
            "test_mean_binary_auprc", ascending=False
        )
        if not ranked.empty:
            cols = [
                "family", "model_name", "stage",
                "val_mean_binary_auprc", "test_mean_binary_auprc",
                "test_mean_binary_auroc", "test_los_hours_24h_mae",
            ]
            cols = [c for c in cols if c in ranked.columns]
            top = ranked.head(15)[cols].copy()
            for c in cols:
                if c not in {"family", "model_name", "stage"}:
                    top[c] = pd.to_numeric(
                        top[c], errors="coerce"
                    ).round(6)
            lines.append(dataframe_to_markdown(top))
            lines.append("")

    lines.extend([
        "## B. Final / aggregate model registry",
        "",
    ])

    if final_df.empty:
        lines.append("No final/aggregate prediction artifacts were detected.")
    else:
        cols = [
            "family", "model_name", "model_id", "role", "stage",
            "val_mean_binary_auprc", "val_mean_binary_auroc",
            "test_mean_binary_auprc", "test_mean_binary_auroc",
            "test_los_hours_24h_mae",
            "prediction_file",
            "validation_prediction_file",
            "parameter_files",
            "best_epoch",
            "sha256",
        ]
        cols = [c for c in cols if c in final_df.columns]
        show = final_df[cols].copy()
        for c in cols:
            if c not in {
                "family", "model_name", "model_id",
                "role", "stage", "prediction_file",
                "validation_prediction_file", "parameter_files",
                "best_epoch", "sha256",
            }:
                show[c] = pd.to_numeric(show[c], errors="coerce").round(6)
        lines.append(dataframe_to_markdown(show))
        lines.append("")

    lines.extend([
        "## C. Baseline model checklist",
        "",
        "All baseline-family directories actually present on disk were scanned.",
        "The final registry is dynamic; it does not rely on a manually maintained model list.",
        "",
    ])

    base = final_df[
        final_df["family"].astype(str).str.startswith("baseline_family_")
    ].copy() if not final_df.empty else pd.DataFrame()

    if not base.empty:
        cols = [
            "family", "model_name", "stage",
            "val_mean_binary_auprc", "test_mean_binary_auprc",
            "test_mean_binary_auroc", "prediction_file",
        ]
        cols = [c for c in cols if c in base.columns]
        lines.append(dataframe_to_markdown(base[cols]))
        lines.append("")

    lines.extend([
        "## D. Seed / training-cache registry",
        "",
        "Seed caches are documented separately from final aggregate models.",
        "A checkpoint-only seed is not counted as a standalone final thesis model.",
        "",
    ])
    if not seeds.empty:
        cols = [
            "family", "seed", "trial", "seed_or_run_directory",
            "has_checkpoint", "has_metadata", "has_predictions",
            "history_files", "parameter_files",
        ]
        cols = [c for c in cols if c in seeds.columns]
        lines.append(dataframe_to_markdown(seeds[cols]))
        lines.append("")

    lines.extend([
        "## E. Tuning / trial registry",
        "",
    ])
    if not trials.empty:
        lines.append(dataframe_to_markdown(trials))
    else:
        lines.append("No trial/tuning artifacts detected.")
    lines.append("")

    lines.extend([
        "## F. Calibration rule",
        "",
        "AUPRC/AUROC use the prediction score.",
        "Brier/ECE/0.5-threshold metrics use probability_mean when sufficiently populated.",
        "For ordinary probability-producing models, prediction is used for calibration only when it is a valid [0,1] probability.",
        "Rank blends are not treated as calibrated probabilities without probability_mean.",
        "",
        "## G. Selection statement",
        "",
        "The final primary model is identified by the locked validation selection rule,",
        "not by ranking the held-out test set.",
        "The primary model is family_z / Z2_tuned.",
        "",
        "## H. Important limitation",
        "",
        "This report inventories artifacts still present under outputs/.",
        "It cannot reconstruct a run whose files were overwritten or deleted.",
        "",
    ])

    lines.extend(missing_items_report(preds, seeds, inventory))
    lines.append("")

    (DOC_ROOT / "THESIS_FINAL_MODEL_REGISTRY.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    DOC_ROOT.mkdir(parents=True, exist_ok=True)

    print("=" * 110)
    print("FINAL THESIS-GRADE MODEL / EXPERIMENT EXTRACTION")
    print("=" * 110)
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Output root  : {OUTPUT_ROOT}")
    print(f"Docs root    : {DOC_ROOT}")
    print()

    # 1) Inventory every preserved artifact.
    print("[1/7] Building COMPLETE artifact inventory...")
    inventory = inventory_all_artifacts()
    inventory_path = DOC_ROOT / "complete_artifact_inventory.csv"
    inventory.to_csv(inventory_path, index=False)
    print(f"      files = {len(inventory):,}")

    # 2) Score every prediction and named summary artifact.
    print("[2/7] Scoring EVERY prediction/summary artifact...")
    preds = prediction_inventory()
    preds_path = DOC_ROOT / "all_prediction_artifacts_scored.csv"
    preds.to_csv(preds_path, index=False)
    print(f"      prediction/summary artifacts = {len(preds):,}")

    # 3) Extract every checkpoint/seed run and its parameters/history.
    print("[3/7] Extracting EVERY seed/checkpoint cache...")
    seeds = seed_inventory()
    seeds_path = DOC_ROOT / "all_seed_training_runs.csv"
    seeds.to_csv(seeds_path, index=False)
    print(f"      seed/checkpoint dirs = {len(seeds):,}")

    # 4) Extract every trial/tuning artifact.
    print("[4/7] Extracting EVERY trial/tuning artifact...")
    trials = trial_inventory()
    trials_path = DOC_ROOT / "all_tuning_trials_and_artifacts.csv"
    trials.to_csv(trials_path, index=False)
    print(f"      trial/tuning entries = {len(trials):,}")

    # 5) Full SHA-256 duplicate audit.
    print("[5/7] Detecting duplicate CSVs using FULL SHA-256...")
    duplicates = duplicate_audit(inventory)
    duplicates_path = DOC_ROOT / "duplicate_content_audit_sha256.csv"
    duplicates.to_csv(duplicates_path, index=False)
    print(
        f"      duplicate groups = "
        f"{duplicates['sha256'].nunique() if not duplicates.empty else 0:,}"
    )

    # 6) Build the final/aggregate thesis model registry.
    print("[6/7] Building thesis-facing FINAL MODEL registry...")
    final_df = choose_final_prediction_rows(preds)
    final_df = attach_model_metadata(final_df)

    # Merge preserved bootstrap confidence-interval/statistics tables when present.
    bootstrap_df = load_bootstrap_stats()
    if not bootstrap_df.empty:
        bootstrap_df.to_csv(
            DOC_ROOT / "all_bootstrap_and_stats_artifacts.csv",
            index=False,
        )
        final_df = merge_bootstrap_stats_into_models(
            final_df, bootstrap_df
        )
    else:
        pd.DataFrame().to_csv(
            DOC_ROOT / "all_bootstrap_and_stats_artifacts.csv",
            index=False,
        )

    # Stable ordering.
    # Do not return natural_key lists from pandas' key= hook: pandas tries to
    # factorize the returned values, and Python lists are unhashable.
    # String sorting is deterministic and robust for the thesis registry.
    if not final_df.empty:
        sort_cols = [c for c in ("family", "model_name", "stage") if c in final_df.columns]
        final_df = final_df.sort_values(
            sort_cols,
            key=lambda s: s.astype(str).str.lower(),
            kind="mergesort",
        ).reset_index(drop=True)

    final_path = DOC_ROOT / "thesis_model_results.csv"
    final_df.to_csv(final_path, index=False)

    # Per-task long format for thesis tables.
    write_long_format(
        final_df,
        DOC_ROOT / "thesis_model_results_long.csv",
    )

    # Locked primary extract.
    z2 = final_df[
        final_df["model_name"].astype(str).eq("Z2_tuned")
        & final_df["family"].astype(str).eq("model_z_family")
    ].copy()
    z2.to_csv(
        DOC_ROOT / "locked_primary_Z2_tuned.csv",
        index=False,
    )

    # 7) Human-readable thesis registry.
    print("[7/7] Writing thesis Markdown registry...")
    write_markdown_report(
        final_df=final_df,
        preds=preds,
        seeds=seeds,
        trials=trials,
        inventory=inventory,
        duplicates=duplicates,
    )

    print()
    print("=" * 110)
    print("FINAL EXTRACTION SUMMARY")
    print("=" * 110)
    print(f"All preserved artifacts        : {len(inventory):,}")
    print(f"Prediction/summary artifacts   : {len(preds):,}")
    print(f"Seed/checkpoint runs           : {len(seeds):,}")
    print(f"Trial/tuning artifacts         : {len(trials):,}")
    print(
        "Duplicate CSV SHA-256 groups  : "
        f"{duplicates['sha256'].nunique() if not duplicates.empty else 0:,}"
    )
    print(f"Final/aggregate registry rows  : {len(final_df):,}")
    print(
        "Bootstrap/stat artifact rows   : "
        f"{len(bootstrap_df):,}"
    )
    print()
    print("THESIS OUTPUT FILES")
    for p in (
        inventory_path,
        preds_path,
        seeds_path,
        trials_path,
        duplicates_path,
        DOC_ROOT / "all_bootstrap_and_stats_artifacts.csv",
        final_path,
        DOC_ROOT / "thesis_model_results_long.csv",
        DOC_ROOT / "locked_primary_Z2_tuned.csv",
        DOC_ROOT / "THESIS_FINAL_MODEL_REGISTRY.md",
    ):
        print(f"  {p}")

    print()
    print("DONE — no training, retraining, retuning, or test-based selection was performed.")


if __name__ == "__main__":
    main()
