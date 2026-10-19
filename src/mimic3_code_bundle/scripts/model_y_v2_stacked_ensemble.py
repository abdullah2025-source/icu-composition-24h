#!/usr/bin/env python3
"""
Model Y v2 — Level-2 Task-Specialized Stacked Ensemble
=======================================================

Purpose
-------
Use all 33 audited tuned models (IDs 00..32) as frozen base predictors and
train a task-specific meta-learner on the validation split only.

Design
------
1. Discovers exactly one tuned/final_selected_configuration/test_predictions.csv
   for each model ID 00..32.
2. Aligns every task by stay_id + subject_id.
3. Builds one meta-feature matrix per task from ALL 33 base-model predictions.
4. Adds optional low-dimensional cross-model consensus features:
      mean, std, min, max, rank spread
5. Uses subject-grouped 3-fold CV over the validation split to compare:
      - LogisticRegression (binary) / Ridge (LOS)
      - fixed-tree-budget XGBoost (binary) / XGBoost regressor (LOS)
6. Uses conservative task priors for meta-learner selection: XGBoost for
   mortality/ventilation; LogisticRegression for rarer vaso/readmission unless
   XGBoost beats it by >0.01 AUPRC; Ridge for LOS unless XGBoost beats it
   by >0.01 MAE.
7. Refits the selected meta-learner on the complete validation split.
8. Applies the frozen stacker exactly once to the test predictions.
9. Reports Model Y v2 vs tuned Model 32 with subject-clustered bootstrap CIs.
10. Calculates deterioration from probability-space mortality/ventilation/
    vasopressor predictions only. It never uses readmission and never uses
    rank scores as probabilities.

Important statistical note
--------------------------
The base models themselves were trained/tuned upstream using the project
validation split. Therefore grouped CV here protects the meta-learner from
ordinary row leakage, but it does NOT erase the upstream validation-based
checkpoint/hyperparameter-selection dependence. This script therefore calls
its CV score a "meta-learner OOF validation score", not an independent
external validation estimate.

Default run
-----------
python -u model_y_v2_stacked_ensemble.py --mode compare --bootstrap-reps 2000

Modes
-----
compare  : greedy Model Y0 + stacked Model Y2 + final bootstrap comparison
stacked  : stacked Model Y2 only
baseline : greedy Model Y0 only
"""
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path("/home/abdullah2025/fyp_project")
SEARCH_ROOT = PROJECT_ROOT / "outputs"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "model_y_v2_stacked_ensemble"

EXPECTED_N = 74_572
EXPECTED_TRAIN = 52_268
EXPECTED_VAL = 7_417
EXPECTED_TEST = 14_887

BINARY_TASKS = (
    "mortality_24h",
    "ventilation_onset_24h",
    "vasopressor_onset_24h",
    "readmission_30d_24h",
)
DETERIORATION_TASKS = BINARY_TASKS[:3]
LOS_TASK = "los_hours_24h"
PRIMARY_TASKS = BINARY_TASKS + (LOS_TASK,)
ALL_MODEL_IDS = [f"{i:02d}" for i in range(33)]
REFERENCE_MODEL = "32"
SEED = 20260923

MIN_AUPRC_GAIN = 0.001
MIN_MAE_GAIN = 0.05


@dataclass
class Candidate:
    model_id: str
    name: str
    path: Path
    df: pd.DataFrame


def safe_metric_binary(y: np.ndarray, p: np.ndarray) -> Tuple[float, float]:
    y = np.asarray(y)
    p = np.asarray(p)
    keep = np.isfinite(y) & np.isfinite(p)
    y = y[keep].astype(int)
    p = p[keep].astype(float)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    return float(average_precision_score(y, p)), float(roc_auc_score(y, p))


def spearman_abs(a: np.ndarray, b: np.ndarray) -> float:
    ra = rankdata(a, method="average")
    rb = rankdata(b, method="average")
    if np.std(ra) == 0.0 or np.std(rb) == 0.0:
        return 1.0
    return abs(float(np.corrcoef(ra, rb)[0, 1]))


def model_id_from_path(path: Path) -> Optional[str]:
    for part in reversed(path.parts):
        m = re.match(r"^(\d{2})_", part)
        if m:
            return m.group(1)
    return None


def model_name_from_path(path: Path, model_id: str) -> str:
    for part in reversed(path.parts):
        if part.startswith(model_id + "_"):
            return part[len(model_id) + 1:].replace("_", " ")
    return model_id


def discover_candidates(model_ids: Sequence[str]) -> Dict[str, Candidate]:
    found: Dict[str, List[Path]] = {mid: [] for mid in model_ids}
    for path in SEARCH_ROOT.glob("**/final_selected_configuration/test_predictions.csv"):
        mid = model_id_from_path(path)
        if mid in found:
            found[mid].append(path)

    result: Dict[str, Candidate] = {}
    missing = []
    ambiguous = []
    for mid in model_ids:
        paths = sorted(found[mid])
        if not paths:
            missing.append(mid)
            continue
        if len(paths) > 1:
            # Prefer the exact Family 7 corrected GPU result if available;
            # otherwise force explicit resolution rather than silently choosing.
            family7 = [p for p in paths if "baseline_family_07_advanced_multimodal_28_32" in str(p)]
            if len(family7) == 1:
                paths = family7
            else:
                ambiguous.append((mid, [str(p) for p in paths]))
                continue
        chosen = paths[0]
        df = pd.read_csv(chosen)
        required = {"stay_id", "subject_id", "split", "task", "target", "validity", "prediction"}
        missing_cols = required - set(df.columns)
        if missing_cols:
            raise ValueError(f"{chosen} missing required columns: {sorted(missing_cols)}")
        if df.duplicated(["stay_id", "split", "task"]).any():
            raise ValueError(f"Duplicate stay_id/split/task rows in {chosen}")
        for split, expected in (("val", EXPECTED_VAL), ("test", EXPECTED_TEST)):
            n = int(df.loc[df["split"] == split, "stay_id"].nunique())
            if n != expected:
                raise ValueError(f"Model {mid} {split} stays={n}, expected {expected}: {chosen}")
        result[mid] = Candidate(mid, model_name_from_path(chosen, mid), chosen, df)

    if missing:
        raise FileNotFoundError(f"Missing tuned final prediction files for models: {missing}")
    if ambiguous:
        raise RuntimeError("Ambiguous tuned final prediction files: " + repr(ambiguous))
    return result


def task_frame(candidate: Candidate, split: str, task: str) -> pd.DataFrame:
    z = candidate.df[(candidate.df["split"] == split) & (candidate.df["task"] == task)].copy()
    if z.empty:
        raise ValueError(f"No rows for model {candidate.model_id}, split={split}, task={task}")
    z = z.sort_values("stay_id").reset_index(drop=True)
    if z["stay_id"].duplicated().any():
        raise ValueError(f"Duplicate stay_id for {candidate.model_id}, {split}, {task}")
    return z


def align_all(
    candidates: Dict[str, Candidate],
    model_ids: Sequence[str],
    split: str,
    task: str,
) -> pd.DataFrame:
    first = task_frame(candidates[model_ids[0]], split, task)
    out = first[["stay_id", "subject_id", "target", "validity"]].copy()
    out[f"pred_{model_ids[0]}"] = first["prediction"].to_numpy(float)
    for mid in model_ids[1:]:
        z = task_frame(candidates[mid], split, task)
        if not np.array_equal(out["stay_id"].to_numpy(), z["stay_id"].to_numpy()):
            raise ValueError(f"stay_id alignment mismatch: {mid}, {split}, {task}")
        if not np.array_equal(out["subject_id"].to_numpy(), z["subject_id"].to_numpy()):
            raise ValueError(f"subject_id alignment mismatch: {mid}, {split}, {task}")
        out[f"pred_{mid}"] = z["prediction"].to_numpy(float)
    return out


def candidate_validation_scores(candidates: Dict[str, Candidate], model_ids: Sequence[str]) -> pd.DataFrame:
    rows = []
    for mid in model_ids:
        for task in PRIMARY_TASKS:
            z = task_frame(candidates[mid], "val", task)
            keep = z["validity"].astype(bool).to_numpy()
            keep &= np.isfinite(z["target"].to_numpy(float)) & np.isfinite(z["prediction"].to_numpy(float))
            y = z.loc[keep, "target"].to_numpy(float)
            p = z.loc[keep, "prediction"].to_numpy(float)
            if task in BINARY_TASKS:
                ap, auc = safe_metric_binary(y, p)
                rows.append({"model_id": mid, "model_name": candidates[mid].name, "task": task, "val_auprc": ap, "val_auroc": auc, "val_mae": np.nan})
            else:
                rows.append({"model_id": mid, "model_name": candidates[mid].name, "task": task, "val_auprc": np.nan, "val_auroc": np.nan, "val_mae": float(mean_absolute_error(y, p))})
    return pd.DataFrame(rows)


def rank_mean(preds: Sequence[np.ndarray]) -> np.ndarray:
    ranks = [rankdata(p, method="average") / float(len(p)) for p in preds]
    return np.mean(np.stack(ranks, axis=0), axis=0)


def raw_mean(preds: Sequence[np.ndarray]) -> np.ndarray:
    return np.mean(np.stack(preds, axis=0), axis=0)


def choose_binary_greedy(
    candidates: Dict[str, Candidate], scores: pd.DataFrame, model_ids: Sequence[str], task: str, max_members: int = 3
) -> dict:
    ranking = scores[scores["task"] == task].sort_values("val_auprc", ascending=False)
    aligned = align_all(candidates, model_ids, "val", task)
    target = aligned["target"].to_numpy(float)
    valid = aligned["validity"].astype(bool).to_numpy()

    def score(members: Sequence[str]) -> float:
        p = rank_mean([aligned[f"pred_{m}"].to_numpy(float) for m in members])
        return float(average_precision_score(target[valid].astype(int), p[valid]))

    selected = [str(ranking.iloc[0]["model_id"])]
    current = score(selected)
    for mid in ranking["model_id"].astype(str).tolist()[1:]:
        if len(selected) >= max_members:
            break
        candidate_score = score(selected + [mid])
        correlations = [spearman_abs(aligned.loc[valid, f"pred_{old}"].to_numpy(float), aligned.loc[valid, f"pred_{mid}"].to_numpy(float)) for old in selected]
        if candidate_score > current + MIN_AUPRC_GAIN or (candidate_score > current and correlations and min(correlations) < 0.90):
            selected.append(mid)
            current = candidate_score
    return {"task": task, "members": selected, "validation_auprc": current, "blend": "rank_mean"}


def choose_los_greedy(
    candidates: Dict[str, Candidate], scores: pd.DataFrame, model_ids: Sequence[str], max_members: int = 3
) -> dict:
    ranking = scores[scores["task"] == LOS_TASK].sort_values("val_mae", ascending=True)
    aligned = align_all(candidates, model_ids, "val", LOS_TASK)
    target = aligned["target"].to_numpy(float)
    valid = aligned["validity"].astype(bool).to_numpy()
    selected = [str(ranking.iloc[0]["model_id"])]
    current = float(mean_absolute_error(target[valid], aligned.loc[valid, f"pred_{selected[0]}"].to_numpy(float)))
    for mid in ranking["model_id"].astype(str).tolist()[1:]:
        if len(selected) >= max_members:
            break
        p = raw_mean([aligned.loc[valid, f"pred_{m}"].to_numpy(float) for m in selected + [mid]])
        candidate_mae = float(mean_absolute_error(target[valid], p))
        correlations = [spearman_abs(aligned.loc[valid, f"pred_{old}"].to_numpy(float), aligned.loc[valid, f"pred_{mid}"].to_numpy(float)) for old in selected]
        if candidate_mae < current - MIN_MAE_GAIN or (candidate_mae < current and correlations and min(correlations) < 0.90):
            selected.append(mid)
            current = candidate_mae
    return {"task": LOS_TASK, "members": selected, "validation_mae": current, "blend": "raw_mean_hours"}


def apply_greedy_recipe(candidates: Dict[str, Candidate], recipe: dict, split: str) -> pd.DataFrame:
    task = str(recipe["task"])
    members = list(recipe["members"])
    aligned = align_all(candidates, members, split, task)
    raw_preds = [aligned[f"pred_{m}"].to_numpy(float) for m in members]
    if task in BINARY_TASKS:
        prediction = rank_mean(raw_preds)
        probability_mean = raw_mean(raw_preds)
    else:
        prediction = raw_mean(raw_preds)
        probability_mean = None
    out = aligned[["stay_id", "subject_id", "target", "validity"]].copy()
    out["task"] = task
    out["prediction"] = prediction
    if probability_mean is not None:
        out["probability_mean"] = probability_mean
    return out


def make_meta_features(frame: pd.DataFrame, model_ids: Sequence[str], add_consensus: bool) -> Tuple[np.ndarray, List[str]]:
    feature_cols = [f"pred_{mid}" for mid in model_ids]
    X = frame[feature_cols].to_numpy(float)
    names = list(feature_cols)
    if add_consensus:
        consensus = np.column_stack([
            np.mean(X, axis=1),
            np.std(X, axis=1),
            np.min(X, axis=1),
            np.max(X, axis=1),
            np.ptp(X, axis=1),
        ])
        X = np.column_stack([X, consensus])
        names.extend(["consensus_mean", "consensus_std", "consensus_min", "consensus_max", "consensus_range"])
    return X, names


def make_binary_logistic() -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=0.5, max_iter=2000, solver="lbfgs")),
    ])


def make_los_ridge() -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", Ridge(alpha=5.0)),
    ])


def require_xgb():
    try:
        import xgboost as xgb  # noqa: F401
    except Exception as exc:
        raise RuntimeError(f"xgboost is required for stacked Model Y: {exc}") from exc


def make_xgb_classifier(seed: int = SEED):
    import xgboost as xgb
    return xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=5.0,
        reg_alpha=0.0,
        tree_method="hist",
        eval_metric="aucpr",
        n_jobs=8,
        random_state=seed,
    )


def make_xgb_regressor(seed: int = SEED):
    import xgboost as xgb
    return xgb.XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=5.0,
        reg_alpha=0.0,
        tree_method="hist",
        eval_metric="mae",
        n_jobs=8,
        random_state=seed,
    )


def grouped_oof_binary(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model_kind: str,
    n_splits: int,
    seed: int,
) -> Tuple[np.ndarray, float, List[int]]:
    gkf = GroupKFold(n_splits=n_splits)
    oof = np.full(len(y), np.nan, dtype=float)
    best_iters: List[int] = []
    for fold, (tr, va) in enumerate(gkf.split(X, y, groups)):
        if model_kind == "logistic":
            model = make_binary_logistic()
            model.fit(X[tr], y[tr].astype(int))
        else:
            model = make_xgb_classifier(seed + fold)
            model.fit(X[tr], y[tr].astype(int), verbose=False)
            best_iters.append(200)
        oof[va] = model.predict_proba(X[va])[:, 1]
    keep = np.isfinite(oof)
    score = float(average_precision_score(y[keep].astype(int), oof[keep]))
    return oof, score, best_iters


def grouped_oof_los(
    X: np.ndarray,
    y_raw: np.ndarray,
    groups: np.ndarray,
    model_kind: str,
    n_splits: int,
    seed: int,
) -> Tuple[np.ndarray, float, List[int]]:
    y_log = np.log1p(np.clip(y_raw - 24.0, 0, None))
    gkf = GroupKFold(n_splits=n_splits)
    oof = np.full(len(y_raw), np.nan, dtype=float)
    best_iters: List[int] = []
    for fold, (tr, va) in enumerate(gkf.split(X, y_log, groups)):
        if model_kind == "ridge":
            model = make_los_ridge()
            model.fit(X[tr], y_log[tr])
        else:
            model = make_xgb_regressor(seed + fold)
            model.fit(X[tr], y_log[tr], verbose=False)
            best_iters.append(200)
        pred_log = model.predict(X[va])
        oof[va] = np.expm1(pred_log) + 24.0
    keep = np.isfinite(oof)
    score = float(mean_absolute_error(y_raw[keep], oof[keep]))
    return oof, score, best_iters


def fit_final_binary(X: np.ndarray, y: np.ndarray, kind: str, seed: int):
    if kind == "logistic":
        model = make_binary_logistic()
        model.fit(X, y.astype(int))
        return model
    model = make_xgb_classifier(seed)
    model.fit(X, y.astype(int), verbose=False)
    return model


def fit_final_los(X: np.ndarray, y_raw: np.ndarray, kind: str, seed: int):
    if kind == "ridge":
        model = make_los_ridge()
        model.fit(X, np.log1p(np.clip(y_raw - 24.0, 0, None)))
        return model
    model = make_xgb_regressor(seed)
    model.fit(X, np.log1p(np.clip(y_raw - 24.0, 0, None)), verbose=False)
    return model


def stack_task(
    candidates: Dict[str, Candidate],
    model_ids: Sequence[str],
    task: str,
    n_splits: int,
    add_consensus: bool,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict, object]:
    val = align_all(candidates, model_ids, "val", task)
    test = align_all(candidates, model_ids, "test", task)

    val_keep = val["validity"].astype(bool).to_numpy()
    val_keep &= np.isfinite(val["target"].to_numpy(float))
    for mid in model_ids:
        val_keep &= np.isfinite(val[f"pred_{mid}"].to_numpy(float))
    if not val_keep.all():
        val_model = val.loc[val_keep].reset_index(drop=True)
    else:
        val_model = val.copy()

    X_val, feature_names = make_meta_features(val_model, model_ids, add_consensus)
    X_test, _ = make_meta_features(test, model_ids, add_consensus)
    y_val = val_model["target"].to_numpy(float)
    groups = val_model["subject_id"].to_numpy()

    if task in BINARY_TASKS:
        oof_log, score_log, _ = grouped_oof_binary(X_val, y_val, groups, "logistic", n_splits, seed)
        oof_xgb, score_xgb, iters = grouped_oof_binary(X_val, y_val, groups, "xgb", n_splits, seed + 100)
        results = pd.DataFrame({
            "row": np.arange(len(val_model)),
            "subject_id": val_model["subject_id"].to_numpy(),
            "logistic_oof": oof_log,
            "xgb_oof": oof_xgb,
        })
        xgb_gain = float(score_xgb - score_log)
        if task in ("vasopressor_onset_24h", "readmission_30d_24h"):
            selected_kind = "xgb" if xgb_gain > 0.01 else "logistic"
        else:
            selected_kind = "xgb" if score_xgb >= score_log else "logistic"
        if selected_kind == "xgb":
            final_model = fit_final_binary(X_val, y_val, "xgb", seed + 500)
            test_pred = final_model.predict_proba(X_test)[:, 1]
            selected_score = score_xgb
        else:
            final_model = fit_final_binary(X_val, y_val, "logistic", seed)
            test_pred = final_model.predict_proba(X_test)[:, 1]
            selected_score = score_log
        val_selected_oof = oof_xgb if selected_kind == "xgb" else oof_log
        val_out = val_model[["stay_id", "subject_id", "target", "validity"]].copy()
        val_out["task"] = task
        val_out["prediction"] = val_selected_oof
        val_out["probability_mean"] = val_selected_oof
        test_out = test[["stay_id", "subject_id", "target", "validity"]].copy()
        test_out["task"] = task
        test_out["prediction"] = test_pred
        test_out["probability_mean"] = test_pred
        info = {
            "task": task,
            "selected_meta_learner": selected_kind,
            "meta_feature_count": int(X_val.shape[1]),
            "meta_features": feature_names,
            "grouped_cv_folds": n_splits,
            "validation_oof_auprc_logistic": score_log,
            "validation_oof_auprc_xgb": score_xgb,
            "validation_oof_selected": float(selected_score),
            "xgb_n_estimators_fixed": 200,
            "meta_learner_selection_policy": {
                "mortality_24h": "xgb_if_score_is_better",
                "ventilation_onset_24h": "xgb_if_score_is_better",
                "vasopressor_onset_24h": "logistic_unless_xgb_gain_gt_0.01",
                "readmission_30d_24h": "logistic_unless_xgb_gain_gt_0.01",
            },
            "xgb_minus_logistic_auprc": float(score_xgb - score_log),
            "n_val": int(len(val_model)),
            "n_test": int(len(test)),
        }
        return val_out, test_out, info, final_model

    y_raw = y_val
    oof_ridge, score_ridge, _ = grouped_oof_los(X_val, y_raw, groups, "ridge", n_splits, seed)
    oof_xgb, score_xgb, iters = grouped_oof_los(X_val, y_raw, groups, "xgb", n_splits, seed + 100)
    ridge_gain = float(score_ridge - score_xgb)
    selected_kind = "xgb" if ridge_gain > 0.01 else "ridge"
    if selected_kind == "xgb":
        final_model = fit_final_los(X_val, y_raw, "xgb", seed + 500)
        test_pred = np.expm1(final_model.predict(X_test)) + 24.0
        selected_score = score_xgb
        val_selected_oof = oof_xgb
    else:
        final_model = fit_final_los(X_val, y_raw, "ridge", seed)
        test_pred = np.expm1(final_model.predict(X_test)) + 24.0
        selected_score = score_ridge
        val_selected_oof = oof_ridge
    val_out = val_model[["stay_id", "subject_id", "target", "validity"]].copy()
    val_out["task"] = task
    val_out["prediction"] = val_selected_oof
    test_out = test[["stay_id", "subject_id", "target", "validity"]].copy()
    test_out["task"] = task
    test_out["prediction"] = test_pred
    info = {
        "task": task,
        "selected_meta_learner": selected_kind,
        "meta_feature_count": int(X_val.shape[1]),
        "meta_features": feature_names,
        "grouped_cv_folds": n_splits,
        "validation_oof_mae_ridge": score_ridge,
        "validation_oof_mae_xgb": score_xgb,
        "validation_oof_selected": float(selected_score),
        "xgb_n_estimators_fixed": 200,
        "meta_learner_selection_policy": "ridge_unless_xgb_mae_gain_gt_0.01",
        "ridge_minus_xgb_mae": float(score_ridge - score_xgb),
        "n_val": int(len(val_model)),
        "n_test": int(len(test)),
    }
    return val_out, test_out, info, final_model


def evaluate(frame: pd.DataFrame) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for task in PRIMARY_TASKS:
        z = frame[(frame["task"] == task) & frame["validity"].astype(bool)].copy()
        y = z["target"].to_numpy(float)
        p = z["prediction"].to_numpy(float)
        keep = np.isfinite(y) & np.isfinite(p)
        y, p = y[keep], p[keep]
        if task in BINARY_TASKS:
            out[f"{task}_auprc"] = float(average_precision_score(y.astype(int), p))
            out[f"{task}_auroc"] = float(roc_auc_score(y.astype(int), p))
        else:
            out[f"{task}_mae"] = float(mean_absolute_error(y, p))
    out["mean_binary_auprc"] = float(np.mean([out[f"{t}_auprc"] for t in BINARY_TASKS]))
    out["mean_binary_auroc"] = float(np.mean([out[f"{t}_auroc"] for t in BINARY_TASKS]))
    return out


def deterioration_from_frame(frame: pd.DataFrame) -> Dict[str, float]:
    parts = []
    for task in DETERIORATION_TASKS:
        z = frame[frame["task"] == task].sort_values("stay_id")
        if "probability_mean" not in z.columns:
            z = z.copy()
            z["probability_mean"] = z["prediction"].to_numpy(float)
        parts.append(z[["stay_id", "target", "validity", "probability_mean"]])
    base = parts[0][["stay_id"]].copy()
    valid = np.ones(len(base), dtype=bool)
    targets = []
    probs = []
    for z in parts:
        if not np.array_equal(base["stay_id"].to_numpy(), z["stay_id"].to_numpy()):
            raise ValueError("Deterioration alignment failure")
        valid &= z["validity"].astype(bool).to_numpy()
        targets.append(z["target"].to_numpy(float))
        probs.append(np.clip(z["probability_mean"].to_numpy(float), 0.0, 1.0))
    label = 1.0 - np.prod(1.0 - np.stack(targets, axis=0), axis=0)
    pred = 1.0 - np.prod(1.0 - np.stack(probs, axis=0), axis=0)
    keep = valid & np.isfinite(label) & np.isfinite(pred)
    y = label[keep].astype(int)
    p = pred[keep]
    if len(y) == 0 or len(np.unique(y)) < 2:
        return {"auprc": float("nan"), "auroc": float("nan"), "n": int(keep.sum())}
    return {
        "auprc": float(average_precision_score(y, p)),
        "auroc": float(roc_auc_score(y, p)),
        "n": int(keep.sum()),
    }


def reference_frame(candidates: Dict[str, Candidate], split: str) -> pd.DataFrame:
    parts = []
    for task in PRIMARY_TASKS:
        z = task_frame(candidates[REFERENCE_MODEL], split, task)
        z = z[["stay_id", "subject_id", "target", "validity", "prediction"]].copy()
        z["task"] = task
        if task in BINARY_TASKS:
            z["probability_mean"] = z["prediction"].to_numpy(float)
        parts.append(z)
    return pd.concat(parts, ignore_index=True)


def subject_bootstrap(
    y_frame: pd.DataFrame,
    ref_frame: pd.DataFrame,
    task: str,
    reps: int,
    seed: int,
) -> Dict[str, float]:
    a = y_frame[y_frame["task"] == task].sort_values("stay_id")
    b = ref_frame[ref_frame["task"] == task].sort_values("stay_id")
    if not np.array_equal(a["stay_id"].to_numpy(), b["stay_id"].to_numpy()):
        raise ValueError(f"Bootstrap stay alignment failure: {task}")
    keep = a["validity"].astype(bool).to_numpy() & b["validity"].astype(bool).to_numpy()
    keep &= np.isfinite(a["target"].to_numpy(float)) & np.isfinite(a["prediction"].to_numpy(float)) & np.isfinite(b["prediction"].to_numpy(float))
    y = a.loc[keep, "target"].to_numpy(float)
    py = a.loc[keep, "prediction"].to_numpy(float)
    pr = b.loc[keep, "prediction"].to_numpy(float)
    subjects = a.loc[keep, "subject_id"].to_numpy()
    unique_subjects = np.unique(subjects)
    groups = {s: np.flatnonzero(subjects == s) for s in unique_subjects}
    rng = np.random.default_rng(seed)
    sampled_diffs = []
    for _ in range(reps):
        sampled = rng.choice(unique_subjects, size=len(unique_subjects), replace=True)
        idx = np.concatenate([groups[s] for s in sampled])
        if task in BINARY_TASKS:
            yy = y[idx].astype(int)
            if len(np.unique(yy)) < 2:
                continue
            sampled_diffs.append(average_precision_score(yy, py[idx]) - average_precision_score(yy, pr[idx]))
        else:
            sampled_diffs.append(mean_absolute_error(y[idx], py[idx]) - mean_absolute_error(y[idx], pr[idx]))
    diffs = np.asarray(sampled_diffs, dtype=float)
    if task in BINARY_TASKS:
        point = average_precision_score(y.astype(int), py) - average_precision_score(y.astype(int), pr)
        probability_better = float(np.mean(diffs > 0))
    else:
        point = mean_absolute_error(y, py) - mean_absolute_error(y, pr)
        probability_better = float(np.mean(diffs < 0))
    return {
        "task": task,
        "n": int(len(y)),
        "subjects": int(len(unique_subjects)),
        "point_delta_y_minus_model32": float(point),
        "ci95_low": float(np.quantile(diffs, 0.025)),
        "ci95_high": float(np.quantile(diffs, 0.975)),
        "prob_y_better": probability_better,
    }


def run_baseline(
    candidates: Dict[str, Candidate], model_ids: Sequence[str], max_members: int
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    scores = candidate_validation_scores(candidates, model_ids)
    recipes = {task: choose_binary_greedy(candidates, scores, model_ids, task, max_members) for task in BINARY_TASKS}
    recipes[LOS_TASK] = choose_los_greedy(candidates, scores, model_ids, max_members)
    val_parts = [apply_greedy_recipe(candidates, r, "val") for r in recipes.values()]
    test_parts = [apply_greedy_recipe(candidates, r, "test") for r in recipes.values()]
    val_frame = pd.concat(val_parts, ignore_index=True)
    test_frame = pd.concat(test_parts, ignore_index=True)
    info = {
        "type": "Y0_greedy_task_specialized_rank_ensemble",
        "recipes": recipes,
        "selection_metric": "validation AUPRC / validation MAE",
        "test_used_for_selection": False,
    }
    return val_frame, test_frame, info


def run_stacked(
    candidates: Dict[str, Candidate],
    model_ids: Sequence[str],
    n_splits: int,
    add_consensus: bool,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    require_xgb()
    val_parts = []
    test_parts = []
    task_info = {}
    for task in PRIMARY_TASKS:
        print(f"[STACK] {task}", flush=True)
        val_out, test_out, info, _ = stack_task(
            candidates, model_ids, task, n_splits, add_consensus, seed
        )
        val_parts.append(val_out)
        test_parts.append(test_out)
        task_info[task] = info
    val_frame = pd.concat(val_parts, ignore_index=True)
    test_frame = pd.concat(test_parts, ignore_index=True)
    info = {
        "type": "Y2_level2_task_specific_stack",
        "base_models": list(model_ids),
        "selection_split": "validation_grouped_oof",
        "test_used_for_selection": False,
        "grouped_cv_folds": n_splits,
        "consensus_features": add_consensus,
        "tasks": task_info,
        "upstream_validation_dependence": True,
    }
    return val_frame, test_frame, info


def compare_output(
    y_frames: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
    reference_test: pd.DataFrame,
    reps: int,
) -> pd.DataFrame:
    rows = []
    for name, (_, test_frame) in y_frames.items():
        for i, task in enumerate(PRIMARY_TASKS):
            r = subject_bootstrap(test_frame, reference_test, task, reps=reps, seed=SEED + i + 1000)
            r["model"] = name
            rows.append(r)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Model Y v2 level-2 stacking")
    parser.add_argument("--mode", choices=["baseline", "stacked", "compare"], default="compare")
    parser.add_argument("--candidates", default=",".join(ALL_MODEL_IDS))
    parser.add_argument("--max-members", type=int, default=3)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--no-consensus-features", action="store_true")
    args = parser.parse_args()

    model_ids = [x.strip() for x in args.candidates.split(",") if x.strip()]
    if REFERENCE_MODEL not in model_ids:
        raise ValueError("Model 32 must be included as the reference candidate")
    if len(model_ids) != len(set(model_ids)):
        raise ValueError("Duplicate candidate model IDs")
    if args.cv_folds < 2:
        raise ValueError("cv-folds must be >= 2")
    if args.max_members < 1 or args.max_members > 3:
        raise ValueError("max-members must be 1..3")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if args.mode in ("stacked", "compare"):
        require_xgb()
        import xgboost as xgb
        print(f"XGBoost version: {xgb.__version__}")
    candidates = discover_candidates(model_ids)

    print("=" * 100)
    print("MODEL Y v2 — LEVEL-2 TASK-SPECIALIZED STACKING")
    print("=" * 100)
    print(f"Base models: {len(model_ids)} | IDs: {model_ids[0]}..{model_ids[-1]}")
    print(f"CV folds: {args.cv_folds} | consensus features: {not args.no_consensus_features}")
    for mid in model_ids:
        print(f"{mid}: {candidates[mid].name} -> {candidates[mid].path}")

    scores = candidate_validation_scores(candidates, model_ids)
    scores.to_csv(OUTPUT_ROOT / "candidate_validation_scores.csv", index=False)

    reference_val = reference_frame(candidates, "val")
    reference_test = reference_frame(candidates, "test")
    baseline_val, baseline_test, baseline_info = run_baseline(candidates, model_ids, args.max_members)

    outputs: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]] = {}
    metadata: Dict[str, object] = {
        "frozen_contract": {
            "N": EXPECTED_N,
            "train": EXPECTED_TRAIN,
            "val": EXPECTED_VAL,
            "test": EXPECTED_TEST,
        },
        "candidate_model_ids": model_ids,
        "reference_model": REFERENCE_MODEL,
        "test_used_for_selection": False,
    }

    if args.mode in ("baseline", "compare"):
        outputs["ModelY0"] = (baseline_val, baseline_test)
        metadata["ModelY0"] = baseline_info

    if args.mode in ("stacked", "compare"):
        stacked_val, stacked_test, stacked_info = run_stacked(
            candidates,
            model_ids,
            args.cv_folds,
            add_consensus=not args.no_consensus_features,
            seed=SEED,
        )
        outputs["ModelY2"] = (stacked_val, stacked_test)
        metadata["ModelY2"] = stacked_info

    summary = {
        **metadata,
        "model32_validation_metrics": evaluate(reference_val),
        "model32_test_metrics": evaluate(reference_test),
        "models": {
            name: {
                "validation_metrics": evaluate(v),
                "test_metrics": evaluate(t),
                "test_deterioration": deterioration_from_frame(t),
            }
            for name, (v, t) in outputs.items()
        },
    }

    print("\n" + "=" * 100)
    print("VALIDATION / TEST SUMMARY")
    print("=" * 100)
    print(json.dumps(summary, indent=2, default=str))

    for name, (val_frame, test_frame) in outputs.items():
        val_frame.to_csv(OUTPUT_ROOT / f"{name}_validation_predictions.csv", index=False)
        test_frame.to_csv(OUTPUT_ROOT / f"{name}_test_predictions.csv", index=False)

    (OUTPUT_ROOT / "model_y_v2_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    boot_df = compare_output(outputs, reference_test, args.bootstrap_reps)
    boot_df.to_csv(OUTPUT_ROOT / "paired_bootstrap_vs_model32.csv", index=False)
    print("\nPAIRED SUBJECT-CLUSTERED BOOTSTRAP VS MODEL32")
    print(boot_df.to_string(index=False))

    # Per-model validation ranking for traceability.
    scores.to_csv(OUTPUT_ROOT / "candidate_validation_scores.csv", index=False)

    print(f"\nArtifacts: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
