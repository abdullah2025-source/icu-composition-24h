#!/usr/bin/env python3
"""
Model Z Family — Z1 + Z2 + Z3 in one executable
=================================================

Frozen-input rule
-----------------
NO new MIMIC preprocessing is performed.
This script consumes only:
  * the existing audited 33 model prediction artifacts (00..32),
  * the existing Pipeline-A-V2 structured_24h_processed.npy,
  * the existing Pipeline-A-V2 demographics_24h_processed.npy,
  * the existing frozen pooled clinical-note embedding parquet.

No new extraction, note pooling, labels, cohort construction, or temporal
preprocessing is created here.

Members
-------
Z1  Sparse weighted-rank super-ensemble
    - Fixed: K=5 weighted-rank recipe.
    - Tuned: grouped-CV chooses K from {5,6,7,8}; final weights are fit on
      the full validation split only.

Z2  Task-specific ensemble gate
    - Combines A3_rank_max5, A2_probability_mean_max3, and the corresponding
      Z1 member.
    - Fixed: predeclared convex weights (0.4 A3, 0.4 Z1, 0.2 A2).
    - Tuned: per-task convex grid search on the existing validation split.
      This is explicitly validation-tuned and therefore inherits upstream
      validation dependence.

Z3  Frozen-input multimodal refit + elite-expert gate
    - Uses the already processed structured tensor + demographics + pooled
      text embeddings.
    - Lightweight temporal Transformer + text projection + cross-attention
      + task-specific heads.
    - Fixed: pure multimodal model under one predeclared configuration.
    - Tuned: deterministic hyperparameter grid selected by validation mean
      binary AUPRC, then a validation-only task-specific convex gate mixes
      the tuned multimodal predictions with A3 and Z1 (the strongest existing
      tuned ensemble experts). Test is never used for selection.

Selection / evaluation
----------------------
Primary metric: mean of the four binary validation AUPRCs.
Deterioration: probability-space 3-way noisy-OR of mortality, ventilation,
and vasopressor only. Readmission is excluded.
LOS: log1p(max(LOS-24,0)) training target; reported on hour scale.

All final models are evaluated on the frozen test set exactly after selection.
Subject-clustered bootstrap comparisons are reported for Z1/Z2/Z3 fixed and
Z1/Z2/Z3 tuned versus A3 and Model32.

NOTE: the upstream 33 base models were trained/tuned using the project
validation split. Meta-level CV cannot erase that dependence.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score
from sklearn.model_selection import GroupKFold


PROJECT_ROOT = Path("/home/abdullah2025/fyp_project")
SEARCH_ROOT = PROJECT_ROOT / "outputs"
V2_DIR = PROJECT_ROOT / "outputs" / "pipeline_a_v2"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "model_z_family"
SEED = 20260923
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
TASKS = BINARY_TASKS + (LOS_TASK,)
MODEL_IDS = [f"{i:02d}" for i in range(33)]
REFERENCE_MODEL = "32"
REQUIRED_COLUMNS = {"stay_id", "subject_id", "split", "task", "target", "validity", "prediction"}


# =============================================================================
# Base prediction discovery / alignment
# =============================================================================

def model_id_from_path(path: Path) -> Optional[str]:
    for part in reversed(path.parts):
        m = re.match(r"^(\d{2})_", part)
        if m:
            return m.group(1)
    return None


def discover_candidates() -> Dict[str, pd.DataFrame]:
    found: Dict[str, List[Path]] = {mid: [] for mid in MODEL_IDS}
    for path in SEARCH_ROOT.glob("**/final_selected_configuration/test_predictions.csv"):
        mid = model_id_from_path(path)
        if mid in found:
            found[mid].append(path)

    candidates: Dict[str, pd.DataFrame] = {}
    problems = []
    for mid in MODEL_IDS:
        paths = sorted(found[mid])
        if len(paths) > 1:
            family7 = [p for p in paths if "baseline_family_07_advanced_multimodal_28_32" in str(p)]
            if len(family7) == 1:
                paths = family7
        if len(paths) != 1:
            problems.append(f"{mid}: {paths}")
            continue
        path = paths[0]
        df = pd.read_csv(path)
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        if df.duplicated(["stay_id", "split", "task"]).any():
            raise ValueError(f"Duplicate stay_id/split/task rows in {path}")
        for split, expected in (("val", EXPECTED_VAL), ("test", EXPECTED_TEST)):
            n = int(df.loc[df["split"] == split, "stay_id"].nunique())
            if n != expected:
                raise ValueError(f"Model {mid}: {split} stays={n}, expected={expected}: {path}")
        candidates[mid] = df
        print(f"[BASE] {mid}: {path}", flush=True)
    if problems:
        raise FileNotFoundError("Prediction discovery failed:\n" + "\n".join(problems))
    return candidates


def task_frame(candidates: Dict[str, pd.DataFrame], model_id: str, split: str, task: str) -> pd.DataFrame:
    z = candidates[model_id]
    out = z[(z["split"] == split) & (z["task"] == task)].copy()
    if out.empty:
        raise ValueError(f"No rows for model={model_id}, split={split}, task={task}")
    out = out.sort_values("stay_id").reset_index(drop=True)
    if out["stay_id"].duplicated().any():
        raise ValueError(f"Duplicate stay_id for model={model_id}, split={split}, task={task}")
    return out


def align(candidates: Dict[str, pd.DataFrame], model_ids: Sequence[str], split: str, task: str) -> pd.DataFrame:
    first = task_frame(candidates, model_ids[0], split, task)
    out = first[["stay_id", "subject_id", "target", "validity"]].copy()
    out[f"pred_{model_ids[0]}"] = first["prediction"].to_numpy(float)
    for mid in model_ids[1:]:
        z = task_frame(candidates, mid, split, task)
        if not np.array_equal(out["stay_id"].to_numpy(), z["stay_id"].to_numpy()):
            raise ValueError(f"stay_id alignment failure: {mid}, {split}, {task}")
        if not np.array_equal(out["subject_id"].to_numpy(), z["subject_id"].to_numpy()):
            raise ValueError(f"subject_id alignment failure: {mid}, {split}, {task}")
        out[f"pred_{mid}"] = z["prediction"].to_numpy(float)
    return out


def valid_mask(frame: pd.DataFrame, ids: Sequence[str]) -> np.ndarray:
    keep = frame["validity"].astype(bool).to_numpy()
    keep &= np.isfinite(frame["target"].to_numpy(float))
    for mid in ids:
        keep &= np.isfinite(frame[f"pred_{mid}"].to_numpy(float))
    return keep


# =============================================================================
# Metrics / ensemble utilities
# =============================================================================

def binary_ap(y: np.ndarray, p: np.ndarray) -> float:
    keep = np.isfinite(y) & np.isfinite(p)
    yy = np.asarray(y[keep]).astype(int)
    pp = np.asarray(p[keep]).astype(float)
    if len(yy) == 0 or len(np.unique(yy)) < 2:
        return float("nan")
    return float(average_precision_score(yy, pp))


def binary_auc(y: np.ndarray, p: np.ndarray) -> float:
    keep = np.isfinite(y) & np.isfinite(p)
    yy = np.asarray(y[keep]).astype(int)
    pp = np.asarray(p[keep]).astype(float)
    if len(yy) == 0 or len(np.unique(yy)) < 2:
        return float("nan")
    return float(roc_auc_score(yy, pp))


def metrics(frame: pd.DataFrame) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for task in TASKS:
        z = frame[(frame["task"] == task) & frame["validity"].astype(bool)].copy()
        y = z["target"].to_numpy(float)
        p = z["prediction"].to_numpy(float)
        keep = np.isfinite(y) & np.isfinite(p)
        y, p = y[keep], p[keep]
        if task in BINARY_TASKS:
            out[f"{task}_auprc"] = binary_ap(y, p)
            out[f"{task}_auroc"] = binary_auc(y, p)
        else:
            out[f"{task}_mae"] = float(mean_absolute_error(y, p))
    out["mean_binary_auprc"] = float(np.mean([out[f"{t}_auprc"] for t in BINARY_TASKS]))
    out["mean_binary_auroc"] = float(np.mean([out[f"{t}_auroc"] for t in BINARY_TASKS]))
    return out


def rank_matrix(frame: pd.DataFrame, ids: Sequence[str]) -> np.ndarray:
    n = len(frame)
    cols = []
    for mid in ids:
        cols.append(pd.Series(frame[f"pred_{mid}"].to_numpy(float)).rank(method="average").to_numpy(float) / float(n))
    return np.column_stack(cols)


def raw_matrix(frame: pd.DataFrame, ids: Sequence[str]) -> np.ndarray:
    return np.column_stack([frame[f"pred_{mid}"].to_numpy(float) for mid in ids])


def apply_recipe(candidates: Dict[str, pd.DataFrame], recipe: dict, split: str) -> pd.DataFrame:
    ids = list(recipe["members"])
    task = recipe["task"]
    frame = align(candidates, ids, split, task)
    w = np.asarray([recipe["weights"][m] for m in ids], dtype=float)
    w = w / w.sum()
    raw = raw_matrix(frame, ids)
    if task in BINARY_TASKS:
        if recipe.get("blend", "rank_mean") == "rank_mean":
            R = rank_matrix(frame, ids)
            pred = R @ w
        else:
            pred = raw @ w
        prob = np.clip(raw @ w, 0.0, 1.0)
    else:
        pred = raw @ w
        prob = None
    out = frame[["stay_id", "subject_id", "target", "validity"]].copy()
    out["task"] = task
    out["prediction"] = pred
    if prob is not None:
        out["probability_mean"] = prob
    return out


def ensemble_components(a3: pd.DataFrame, a2: pd.DataFrame, z1: pd.DataFrame, task: str, split: str) -> pd.DataFrame:
    frames = [x[x["task"] == task].sort_values("stay_id").reset_index(drop=True) for x in (a3, a2, z1)]
    base = frames[0][["stay_id", "subject_id", "target", "validity"]].copy()
    for i, f in enumerate(frames[1:], 1):
        if not np.array_equal(base["stay_id"].to_numpy(), f["stay_id"].to_numpy()):
            raise ValueError(f"Z2 component alignment failed for {task}/{split}")
    return pd.DataFrame({
        "stay_id": base["stay_id"].to_numpy(),
        "subject_id": base["subject_id"].to_numpy(),
        "target": base["target"].to_numpy(float),
        "validity": base["validity"].to_numpy(bool),
        "a3_pred": frames[0]["prediction"].to_numpy(float),
        "a2_pred": frames[1]["prediction"].to_numpy(float),
        "z1_pred": frames[2]["prediction"].to_numpy(float),
        "a3_prob": frames[0].get("probability_mean", frames[0]["prediction"]).to_numpy(float),
        "a2_prob": frames[1].get("probability_mean", frames[1]["prediction"]).to_numpy(float),
        "z1_prob": frames[2].get("probability_mean", frames[2]["prediction"]).to_numpy(float),
    })


def z2_apply_from_components(comp: pd.DataFrame, task: str, weights: dict) -> pd.DataFrame:
    a3 = comp["a3_pred"].to_numpy(float)
    a2 = comp["a2_pred"].to_numpy(float)
    z1 = comp["z1_pred"].to_numpy(float)
    if task in BINARY_TASKS:
        R = np.column_stack([
            pd.Series(a3).rank(method="average").to_numpy(float) / len(a3),
            pd.Series(a2).rank(method="average").to_numpy(float) / len(a2),
            pd.Series(z1).rank(method="average").to_numpy(float) / len(z1),
        ])
        pred = R @ np.asarray([weights["A3"], weights["A2"], weights["Z1"]], dtype=float)
        prob = np.column_stack([comp["a3_prob"], comp["a2_prob"], comp["z1_prob"]]) @ np.asarray([weights["A3"], weights["A2"], weights["Z1"]], dtype=float)
        prob = np.clip(prob, 0.0, 1.0)
    else:
        pred = np.column_stack([a3, a2, z1]) @ np.asarray([weights["A3"], weights["A2"], weights["Z1"]], dtype=float)
        prob = None
    out = comp[["stay_id", "subject_id", "target", "validity"]].copy()
    out["task"] = task
    out["prediction"] = pred
    if prob is not None:
        out["probability_mean"] = prob
    return out


def deterioration_from_frame(frame: pd.DataFrame) -> Dict[str, float]:
    parts = []
    for task in DETERIORATION_TASKS:
        z = frame[frame["task"] == task].sort_values("stay_id")
        if "probability_mean" not in z.columns:
            raise ValueError(f"Missing probability_mean for {task}")
        parts.append(z[["stay_id", "target", "validity", "probability_mean"]].copy())
    base = parts[0]["stay_id"].to_numpy()
    valid = np.ones(len(base), dtype=bool)
    targets, probs = [], []
    for z in parts:
        if not np.array_equal(base, z["stay_id"].to_numpy()):
            raise ValueError("Deterioration alignment failure")
        valid &= z["validity"].astype(bool).to_numpy()
        targets.append(z["target"].to_numpy(float))
        probs.append(np.clip(z["probability_mean"].to_numpy(float), 0.0, 1.0))
    y = 1.0 - np.prod(1.0 - np.stack(targets, axis=0), axis=0)
    p = 1.0 - np.prod(1.0 - np.stack(probs, axis=0), axis=0)
    keep = valid & np.isfinite(y) & np.isfinite(p)
    return {"auprc": binary_ap(y[keep], p[keep]), "auroc": binary_auc(y[keep], p[keep]), "n": int(keep.sum())}


def bootstrap_pair(left: pd.DataFrame, right: pd.DataFrame, task: str, reps: int, seed: int) -> dict:
    a = left[left["task"] == task].sort_values("stay_id")
    b = right[right["task"] == task].sort_values("stay_id")
    if not np.array_equal(a["stay_id"].to_numpy(), b["stay_id"].to_numpy()):
        raise ValueError(f"Bootstrap alignment failure for {task}")
    keep = a["validity"].astype(bool).to_numpy() & b["validity"].astype(bool).to_numpy()
    y = a.loc[keep, "target"].to_numpy(float)
    pa = a.loc[keep, "prediction"].to_numpy(float)
    pb = b.loc[keep, "prediction"].to_numpy(float)
    subjects = a.loc[keep, "subject_id"].to_numpy()
    unique_subjects = np.unique(subjects)
    groups = {s: np.flatnonzero(subjects == s) for s in unique_subjects}
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(reps):
        sampled = rng.choice(unique_subjects, size=len(unique_subjects), replace=True)
        idx = np.concatenate([groups[s] for s in sampled])
        if task in BINARY_TASKS:
            yy = y[idx].astype(int)
            if len(np.unique(yy)) < 2:
                continue
            diffs.append(binary_ap(yy, pa[idx]) - binary_ap(yy, pb[idx]))
        else:
            diffs.append(mean_absolute_error(y[idx], pa[idx]) - mean_absolute_error(y[idx], pb[idx]))
    arr = np.asarray(diffs, dtype=float)
    point = binary_ap(y.astype(int), pa) - binary_ap(y.astype(int), pb) if task in BINARY_TASKS else mean_absolute_error(y, pa) - mean_absolute_error(y, pb)
    prob = float(np.mean(arr > 0)) if task in BINARY_TASKS else float(np.mean(arr < 0))
    return {
        "task": task,
        "n": int(len(y)),
        "subjects": int(len(unique_subjects)),
        "point_delta_left_minus_right": float(point),
        "ci95_low": float(np.quantile(arr, 0.025)),
        "ci95_high": float(np.quantile(arr, 0.975)),
        "prob_left_better": prob,
    }


# =============================================================================
# A2 / A3 / Z1
# =============================================================================

def greedy_rank_recipe(candidates: Dict[str, pd.DataFrame], task: str, max_members: int, mode: str = "rank") -> dict:
    ids = MODEL_IDS
    frame = align(candidates, ids, "val", task)
    keep = valid_mask(frame, ids)
    y = frame["target"].to_numpy(float)
    R = rank_matrix(frame, ids)
    raw = raw_matrix(frame, ids)
    individual = []
    for j, mid in enumerate(ids):
        pred = R[keep, j] if mode == "rank" else raw[keep, j]
        metric = binary_ap(y[keep], pred) if task in BINARY_TASKS else -float(mean_absolute_error(y[keep], pred))
        individual.append((metric, mid))
    individual.sort(reverse=True)
    selected = [individual[0][1]]
    current = individual[0][0]
    for _, mid in individual[1:]:
        if len(selected) >= max_members:
            break
        idx = [ids.index(x) for x in selected + [mid]]
        pred = (np.mean(R[keep][:, idx], axis=1) if mode == "rank" else np.mean(raw[keep][:, idx], axis=1))
        metric = binary_ap(y[keep], pred) if task in BINARY_TASKS else -float(mean_absolute_error(y[keep], pred))
        if metric > current + (0.001 if task in BINARY_TASKS else 0.05) or metric > current:
            selected.append(mid)
            current = metric
    weights = {m: 1.0 / len(selected) for m in selected}
    return {"task": task, "members": selected, "weights": weights, "validation_score": float(current), "blend": "rank_mean" if mode == "rank" else "raw_mean"}


def build_reference_ensembles(candidates: Dict[str, pd.DataFrame]) -> Tuple[dict, dict, pd.DataFrame, pd.DataFrame]:
    a3_recipes, a2_recipes = {}, {}
    a3v, a3t, a2v, a2t = [], [], [], []
    for task in BINARY_TASKS:
        a3 = greedy_rank_recipe(candidates, task, 5, "rank")
        a2 = greedy_rank_recipe(candidates, task, 3, "raw")
        a3_recipes[task] = a3
        a2_recipes[task] = a2
        a3v.append(apply_recipe(candidates, a3, "val")); a3t.append(apply_recipe(candidates, a3, "test"))
        a2v.append(apply_recipe(candidates, a2, "val")); a2t.append(apply_recipe(candidates, a2, "test"))
    # LOS uses raw-hour mean; A3/A2 naming retained for parallel reporting.
    a3_los = greedy_rank_recipe(candidates, LOS_TASK, 5, "raw")
    a2_los = greedy_rank_recipe(candidates, LOS_TASK, 3, "raw")
    a3_recipes[LOS_TASK] = a3_los; a2_recipes[LOS_TASK] = a2_los
    a3v.append(apply_recipe(candidates, a3_los, "val")); a3t.append(apply_recipe(candidates, a3_los, "test"))
    a2v.append(apply_recipe(candidates, a2_los, "val")); a2t.append(apply_recipe(candidates, a2_los, "test"))
    return a3_recipes, a2_recipes, pd.concat(a3v, ignore_index=True), pd.concat(a3t, ignore_index=True), pd.concat(a2v, ignore_index=True), pd.concat(a2t, ignore_index=True)


def alpha_grid(step: float) -> List[float]:
    vals = np.arange(step, 0.800001, step)
    return [float(v) for v in vals]


def greedy_weighted_rank(frame: pd.DataFrame, max_members: int, alpha_step: float, min_gain: float) -> dict:
    ids = MODEL_IDS
    keep = valid_mask(frame, ids)
    y = frame["target"].to_numpy(float)
    R = rank_matrix(frame, ids)
    individual = [(binary_ap(y[keep], R[keep, j]), mid) for j, mid in enumerate(ids)]
    individual.sort(reverse=True)
    selected = [individual[0][1]]
    weights = {selected[0]: 1.0}
    cur = R[:, ids.index(selected[0])].copy()
    current_score = float(individual[0][0])
    for step in range(1, max_members):
        best = None
        for mid in ids:
            if mid in selected:
                continue
            c = R[:, ids.index(mid)]
            for alpha in alpha_grid(alpha_step):
                pred = (1 - alpha) * cur[keep] + alpha * c[keep]
                sc = binary_ap(y[keep], pred)
                if best is None or sc > best[0]:
                    best = (sc, mid, alpha)
        if best is None or best[0] < current_score + min_gain:
            break
        score, mid, alpha = best
        for m in list(weights):
            weights[m] *= (1 - alpha)
        weights[mid] = alpha
        selected.append(mid)
        cur = (1 - alpha) * cur + alpha * R[:, ids.index(mid)]
        current_score = float(score)
    weights = {m: float(w) for m, w in weights.items() if w > 1e-8}
    total = sum(weights.values())
    weights = {m: w / total for m, w in weights.items()}
    return {"members": list(weights.keys()), "weights": weights, "validation_score": current_score, "blend": "weighted_rank_mean"}


def z1_make(candidates: Dict[str, pd.DataFrame], tuned: bool, cv_folds: int, seed: int) -> Tuple[dict, pd.DataFrame, pd.DataFrame]:
    recipes = {}
    val_parts, test_parts = [], []
    cv_rows = []
    for task in BINARY_TASKS:
        if not tuned:
            fit_frame = align(candidates, MODEL_IDS, "val", task)
            fit_frame = fit_frame.loc[valid_mask(fit_frame, MODEL_IDS)].reset_index(drop=True)
            recipe = greedy_weighted_rank(fit_frame, 5, 0.10, 0.001)
            recipe["selection"] = "fixed_K5_alpha0.10_min_gain0.001"
        else:
            full = align(candidates, MODEL_IDS, "val", task)
            keep = valid_mask(full, MODEL_IDS)
            full = full.loc[keep].reset_index(drop=True)
            groups = full["subject_id"].to_numpy()
            y = full["target"].to_numpy(float)
            gkf = GroupKFold(n_splits=cv_folds)
            best_k, best_score = 5, -np.inf
            for k in (5, 6, 7, 8):
                oof = np.full(len(full), np.nan)
                for fold, (tr, va) in enumerate(gkf.split(full, y, groups)):
                    r = greedy_weighted_rank(full.iloc[tr].reset_index(drop=True), k, 0.05, 0.0005)
                    mids = r["members"]; w = np.array([r["weights"][m] for m in mids])
                    Rva = rank_matrix(full.iloc[va].reset_index(drop=True), mids)
                    oof[va] = Rva @ w
                sc = binary_ap(y, oof)
                cv_rows.append({"member":"Z1_tuned","task":task,"K":k,"oof_auprc":sc})
                print(f"[Z1-CV] {task} K={k}: {sc:.6f}", flush=True)
                if sc > best_score:
                    best_score, best_k = sc, k
            recipe = greedy_weighted_rank(full, best_k, 0.05, 0.0005)
            recipe["selection"] = "grouped_CV_K5_6_7_8_alpha0.05_min_gain0.0005"
            recipe["selected_k"] = int(best_k)
            recipe["cv_score"] = float(best_score)
        recipe["task"] = task
        recipes[task] = recipe
        val_parts.append(apply_recipe(candidates, recipe, "val"))
        test_parts.append(apply_recipe(candidates, recipe, "test"))
    # Preserve A3 LOS so Z1 is targeted at binary ranking.
    a3_los = greedy_rank_recipe(candidates, LOS_TASK, 5, "raw")
    recipes[LOS_TASK] = a3_los
    val_parts.append(apply_recipe(candidates, a3_los, "val")); test_parts.append(apply_recipe(candidates, a3_los, "test"))
    return recipes, pd.concat(val_parts, ignore_index=True), pd.concat(test_parts, ignore_index=True), pd.DataFrame(cv_rows)


# =============================================================================
# Z2 task-specific gate
# =============================================================================

def tune_z2_task(comp: pd.DataFrame, task: str, tuned: bool) -> dict:
    if not tuned:
        return {"A3": 0.4, "Z1": 0.4, "A2": 0.2}
    best = None
    grid = np.arange(0.0, 1.0001, 0.05)
    for a3 in grid:
        for z1 in grid:
            a2 = 1.0 - a3 - z1
            if a2 < -1e-9:
                continue
            w = {"A3": float(a3), "Z1": float(z1), "A2": float(max(0.0, a2))}
            pred = z2_apply_from_components(comp, task, w)["prediction"].to_numpy(float)
            keep = comp["validity"].astype(bool).to_numpy() & np.isfinite(comp["target"].to_numpy(float))
            y = comp["target"].to_numpy(float)[keep]; p = pred[keep]
            score = binary_ap(y, p) if task in BINARY_TASKS else -float(mean_absolute_error(y, p))
            if best is None or score > best[0]:
                best = (score, w)
    assert best is not None
    return best[1]


def build_z2(candidates: Dict[str, pd.DataFrame], a3v: pd.DataFrame, a3t: pd.DataFrame, a2v: pd.DataFrame, a2t: pd.DataFrame, z1v: pd.DataFrame, z1t: pd.DataFrame, tuned: bool) -> Tuple[dict, pd.DataFrame, pd.DataFrame]:
    recipes, vparts, tparts = {}, [], []
    for task in TASKS:
        cv = ensemble_components(a3v, a2v, z1v, task, "val")
        ct = ensemble_components(a3t, a2t, z1t, task, "test")
        w = tune_z2_task(cv, task, tuned)
        recipes[task] = {"task":task, "weights":w, "selection":"fixed_predeclared" if not tuned else "full_validation_grid_step0.05"}
        vparts.append(z2_apply_from_components(cv, task, w))
        tparts.append(z2_apply_from_components(ct, task, w))
        print(f"[Z2] {task} weights={w}", flush=True)
    return recipes, pd.concat(vparts, ignore_index=True), pd.concat(tparts, ignore_index=True)


def z3_gate_components(a3: pd.DataFrame, z1: pd.DataFrame, z3: pd.DataFrame, task: str) -> pd.DataFrame:
    frames = [
        a3[a3["task"] == task].sort_values("stay_id").reset_index(drop=True),
        z1[z1["task"] == task].sort_values("stay_id").reset_index(drop=True),
        z3[z3["task"] == task].sort_values("stay_id").reset_index(drop=True),
    ]
    base = frames[0][["stay_id", "subject_id", "target", "validity"]].copy()
    for idx, f in enumerate(frames[1:], start=1):
        if not np.array_equal(base["stay_id"].to_numpy(), f["stay_id"].to_numpy()):
            raise ValueError(f"Z3 gate stay_id alignment failed for {task}, component={idx}")
        if not np.array_equal(base["subject_id"].to_numpy(), f["subject_id"].to_numpy()):
            raise ValueError(f"Z3 gate subject_id alignment failed for {task}, component={idx}")
    return pd.DataFrame({
        "stay_id": base["stay_id"].to_numpy(),
        "subject_id": base["subject_id"].to_numpy(),
        "target": base["target"].to_numpy(float),
        "validity": base["validity"].to_numpy(bool),
        "a3_pred": frames[0]["prediction"].to_numpy(float),
        "z1_pred": frames[1]["prediction"].to_numpy(float),
        "z3_pred": frames[2]["prediction"].to_numpy(float),
        "a3_prob": frames[0].get("probability_mean", frames[0]["prediction"]).to_numpy(float),
        "z1_prob": frames[1].get("probability_mean", frames[1]["prediction"]).to_numpy(float),
        "z3_prob": frames[2].get("probability_mean", frames[2]["prediction"]).to_numpy(float),
    })


def apply_z3_gate(comp: pd.DataFrame, task: str, weights: Dict[str, float]) -> pd.DataFrame:
    names = ["A3", "Z1", "Z3"]
    w = np.asarray([weights[n] for n in names], dtype=float)
    if not np.isclose(w.sum(), 1.0):
        w = w / w.sum()
    pred_cols = [comp[f"{n.lower()}_pred" if n != "A3" else "a3_pred"] for n in names]
    # Rank-space combination is the primary binary decision score.
    if task in BINARY_TASKS:
        R = np.column_stack([
            pd.Series(comp["a3_pred"]).rank(method="average").to_numpy(float) / len(comp),
            pd.Series(comp["z1_pred"]).rank(method="average").to_numpy(float) / len(comp),
            pd.Series(comp["z3_pred"]).rank(method="average").to_numpy(float) / len(comp),
        ])
        pred = R @ w
        probs = np.column_stack([comp["a3_prob"], comp["z1_prob"], comp["z3_prob"]]) @ w
        probs = np.clip(probs, 0.0, 1.0)
    else:
        pred = np.column_stack([comp["a3_pred"], comp["z1_pred"], comp["z3_pred"]]) @ w
        probs = None
    out = comp[["stay_id", "subject_id", "target", "validity"]].copy()
    out["task"] = task
    out["prediction"] = pred
    if probs is not None:
        out["probability_mean"] = probs
    return out


def tune_z3_gate_task(a3: pd.DataFrame, z1: pd.DataFrame, z3: pd.DataFrame, task: str, tuned: bool, step: float = 0.05) -> Dict[str, float]:
    if not tuned:
        return {"A3": 0.50, "Z1": 0.00, "Z3": 0.50}
    comp = z3_gate_components(a3, z1, z3, task)
    keep = comp["validity"].astype(bool).to_numpy() & np.isfinite(comp["target"].to_numpy(float))
    best = None
    grid = np.arange(0.0, 1.0001, step)
    for a3w in grid:
        for z1w in grid:
            z3w = 1.0 - a3w - z1w
            if z3w < -1e-9:
                continue
            w = {"A3": float(a3w), "Z1": float(z1w), "Z3": float(max(0.0, z3w))}
            pred = apply_z3_gate(comp, task, w)["prediction"].to_numpy(float)
            y = comp["target"].to_numpy(float)[keep]
            p = pred[keep]
            score = binary_ap(y, p) if task in BINARY_TASKS else -float(mean_absolute_error(y, p))
            if best is None or score > best[0]:
                best = (score, w)
    assert best is not None
    return best[1]


def build_z3_gated(a3v: pd.DataFrame, a3t: pd.DataFrame, z1v: pd.DataFrame, z1t: pd.DataFrame, z3v: pd.DataFrame, z3t: pd.DataFrame, tuned: bool) -> Tuple[dict, pd.DataFrame, pd.DataFrame]:
    recipes, vparts, tparts = {}, [], []
    for task in TASKS:
        w = tune_z3_gate_task(a3v, z1v, z3v, task, tuned)
        cv = z3_gate_components(a3v, z1v, z3v, task)
        ct = z3_gate_components(a3t, z1t, z3t, task)
        recipes[task] = {
            "task": task,
            "weights": w,
            "selection": "fixed_predeclared_50pct_A3_50pct_Z3" if not tuned else "validation_grid_step0.05_A3_Z1_Z3",
        }
        vparts.append(apply_z3_gate(cv, task, w))
        tparts.append(apply_z3_gate(ct, task, w))
        print(f"[Z3-GATE] {task} weights={w}", flush=True)
    return recipes, pd.concat(vparts, ignore_index=True), pd.concat(tparts, ignore_index=True)


# =============================================================================
# Frozen-input Z3 multimodal model
# =============================================================================

def find_first(root: Path, patterns: Sequence[str]) -> Optional[Path]:
    for pattern in patterns:
        hits = sorted(root.glob(pattern))
        if hits:
            return hits[0]
    return None


def locate_z3_inputs(text_override: Optional[str]) -> Tuple[Path, Path, Path, Path]:
    structured = V2_DIR / "structured_24h_processed.npy"
    demographics = V2_DIR / "demographics_24h_processed.npy"
    alignment = V2_DIR / "a5_alignment_index_24h.csv"
    if not structured.exists() or not demographics.exists() or not alignment.exists():
        raise FileNotFoundError("Required frozen Pipeline-A-V2 structured/demographics/alignment artifacts are missing")
    text_path = Path(text_override) if text_override else (find_first(V2_DIR, ["**/clinical_note_embeddings.parquet", "**/*note*embedding*.parquet"]) or find_first(SEARCH_ROOT, ["**/clinical_note_embeddings.parquet", "**/*note*embedding*.parquet"]))
    if text_path is None or not text_path.exists():
        raise FileNotFoundError("Could not auto-discover the frozen clinical_note_embeddings parquet under outputs/pipeline_a_v2. Use --text-path.")
    return structured, demographics, alignment, text_path


def parse_embedding_column(series: pd.Series) -> np.ndarray:
    vals = []
    for x in series.tolist():
        if isinstance(x, np.ndarray):
            vals.append(x.astype(np.float32))
        elif isinstance(x, list):
            vals.append(np.asarray(x, dtype=np.float32))
        elif isinstance(x, tuple):
            vals.append(np.asarray(x, dtype=np.float32))
        elif isinstance(x, str):
            vals.append(np.asarray(ast.literal_eval(x), dtype=np.float32))
        else:
            raise ValueError("Unsupported embedding cell type")
    dims = {len(v) for v in vals}
    if dims != {768}:
        raise ValueError(f"Expected 768-d pooled text embeddings, found dimensions={sorted(dims)}")
    return np.vstack(vals).astype(np.float32)


def load_text_embeddings(text_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(text_path)
    if "stay_id" not in df.columns:
        raise ValueError(f"Text embeddings file has no stay_id column: {text_path}")
    if df["stay_id"].duplicated().any():
        raise ValueError("Frozen text embedding file contains duplicate stay_id rows; this script does not create new note pooling")
    embedding_col = None
    for candidate in ("embedding", "text_embedding", "embeddings"):
        if candidate in df.columns:
            embedding_col = candidate
            break
    if embedding_col is not None:
        emb = parse_embedding_column(df[embedding_col])
    else:
        numeric = [c for c in df.columns if c != "stay_id" and pd.api.types.is_numeric_dtype(df[c])]
        if len(numeric) != 768:
            preferred = [c for c in numeric if any(s in str(c).lower() for s in ("text_", "emb_", "embedding"))]
            if len(preferred) == 768:
                numeric = preferred
        if len(numeric) != 768:
            raise ValueError(f"Could not identify 768 numeric embedding columns in {text_path}; found {len(numeric)}")
        emb = df[numeric].to_numpy(np.float32)
    out = pd.DataFrame({"stay_id": df["stay_id"].to_numpy()})
    out["embedding"] = list(emb)
    return out


def load_z3_data(text_path: Path):
    import torch
    structured_path = V2_DIR / "structured_24h_processed.npy"
    demographics_path = V2_DIR / "demographics_24h_processed.npy"
    alignment_path = V2_DIR / "a5_alignment_index_24h.csv"
    structured = np.load(structured_path, mmap_mode="r")
    demographics = np.load(demographics_path, mmap_mode="r")
    alignment = pd.read_csv(alignment_path)
    if len(alignment) != EXPECTED_N or structured.shape[0] != EXPECTED_N or demographics.shape[0] != EXPECTED_N:
        raise ValueError(f"Frozen artifact size mismatch: alignment={len(alignment)}, structured={structured.shape}, demographics={demographics.shape}")
    if structured.shape[1:] != (24, 110):
        raise ValueError(f"Expected structured shape (N,24,110), found {structured.shape}")
    if demographics.shape[1:] != (4,):
        raise ValueError(f"Expected demographics shape (N,4), found {demographics.shape}")

    # Find outcomes from the already created Pipeline-A-V2 target table.
    outcome_candidates = [V2_DIR / "targets_24h" / "outcomes_24h_primary_cohort.csv"] + sorted((V2_DIR / "targets_24h").glob("*.csv"))
    outcomes = None
    for p in outcome_candidates:
        if not p.exists():
            continue
        q = pd.read_csv(p)
        if {"stay_id", "subject_id"}.issubset(q.columns) and all(t in q.columns or f"{t}_valid" in q.columns for t in TASKS if t != LOS_TASK):
            if "mortality_24h" in q.columns:
                outcomes = q
                break
    if outcomes is None:
        raise FileNotFoundError("Could not identify existing Pipeline-A-V2 outcomes_24h table")

    cohort = alignment.merge(outcomes, on=["stay_id", "subject_id"], how="left", validate="one_to_one", suffixes=("", "_outcome"))
    split_col = "split_24h" if "split_24h" in cohort.columns else "split"
    if split_col not in cohort.columns:
        raise ValueError("Frozen alignment lacks split/split_24h")
    cohort["split_24h"] = cohort[split_col].astype(str)
    for task in TASKS:
        if task not in cohort.columns:
            # tolerate one suffix level if present
            alt = f"{task}_outcome"
            if alt in cohort.columns:
                cohort[task] = cohort[alt]
            else:
                raise ValueError(f"Frozen outcomes missing {task}")
        valid_col = f"{task}_valid"
        if valid_col not in cohort.columns:
            alt = f"{valid_col}_outcome"
            if alt in cohort.columns:
                cohort[valid_col] = cohort[alt]
            else:
                # validity is part of the frozen target contract; do not invent it except finite target fallback
                cohort[valid_col] = np.isfinite(pd.to_numeric(cohort[task], errors="coerce")).astype(int)

    text = load_text_embeddings(text_path)
    text_map = {int(s): e for s, e in zip(text["stay_id"], text["embedding"])}
    emb = np.zeros((len(cohort), 768), dtype=np.float32)
    missing = np.ones(len(cohort), dtype=np.float32)
    for i, sid in enumerate(cohort["stay_id"].to_numpy()):
        e = text_map.get(int(sid))
        if e is not None:
            emb[i] = e
            missing[i] = 0.0
    text_input = np.column_stack([emb, missing]).astype(np.float32)
    # Replace only non-finite values in frozen numeric arrays at runtime; this is numerical safety,
    # not new preprocessing or feature engineering.
    return structured, np.asarray(demographics, dtype=np.float32), text_input, cohort


class Z3Dataset:
    def __init__(self, structured, demographics, text_input, cohort: pd.DataFrame, indices: np.ndarray):
        self.structured = structured
        self.demographics = demographics
        self.text_input = text_input
        self.cohort = cohort
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, j):
        i = int(self.indices[j])
        x = np.asarray(self.structured[i], dtype=np.float32)
        d = np.asarray(self.demographics[i], dtype=np.float32)
        t = np.asarray(self.text_input[i], dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
        t = np.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
        row = self.cohort.iloc[i]
        targets = np.array([float(row[t]) if pd.notna(row[t]) else -1.0 for t in TASKS], dtype=np.float32)
        valids = np.array([int(row[f"{t}_valid"]) if pd.notna(row[f"{t}_valid"]) else 0 for t in TASKS], dtype=np.float32)
        return x, d, t, targets, valids, i


def z3_configs() -> Tuple[dict, List[dict]]:
    fixed = {
        "name": "fixed",
        "d_model": 128,
        "n_heads": 4,
        "n_layers": 2,
        "dropout": 0.20,
        "lr": 2e-4,
        "weight_decay": 1e-4,
        "batch_size": 256,
        "max_epochs": 25,
        "patience": 6,
        "grad_clip": 1.0,
        "loss_mode": "bce",
    }
    tuned = [
        {**fixed, "name":"t1", "d_model":128, "n_layers":2, "dropout":0.10, "lr":2e-4, "loss_mode":"bce"},
        {**fixed, "name":"t2", "d_model":192, "n_layers":2, "dropout":0.20, "lr":1.5e-4, "loss_mode":"bce"},
        {**fixed, "name":"t3", "d_model":192, "n_layers":3, "dropout":0.15, "lr":2e-4, "loss_mode":"bce"},
        {**fixed, "name":"t4", "d_model":192, "n_layers":2, "dropout":0.20, "lr":2e-4, "loss_mode":"focal_rare"},
    ]
    return fixed, tuned


def set_seed(seed: int) -> None:
    import random
    random.seed(seed); np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_z3_model(cfg: dict):
    import torch
    import torch.nn as nn
    d = cfg["d_model"]
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.in_proj = nn.Linear(110, d)
            self.pos = nn.Parameter(torch.zeros(1, 24, d))
            layer = nn.TransformerEncoderLayer(
                d_model=d, nhead=cfg["n_heads"], dim_feedforward=4*d,
                dropout=cfg["dropout"], activation="gelu", batch_first=True, norm_first=True
            )
            self.temporal = nn.TransformerEncoder(layer, num_layers=cfg["n_layers"])
            self.temporal_norm = nn.LayerNorm(d)
            self.text = nn.Sequential(
                nn.Linear(769, d), nn.GELU(), nn.Dropout(cfg["dropout"]), nn.Linear(d, d), nn.LayerNorm(d)
            )
            self.demo = nn.Sequential(nn.Linear(4, 32), nn.GELU(), nn.Dropout(cfg["dropout"]), nn.Linear(32, 32), nn.LayerNorm(32))
            self.cross = nn.MultiheadAttention(d, num_heads=cfg["n_heads"], dropout=cfg["dropout"], batch_first=True)
            self.fuse = nn.Sequential(
                nn.Linear(3*d + 32, 2*d), nn.GELU(), nn.Dropout(cfg["dropout"]), nn.Linear(2*d, d), nn.LayerNorm(d)
            )
            self.binary_heads = nn.ModuleDict({t: nn.Linear(d,1) for t in BINARY_TASKS})
            self.los_head = nn.Linear(d,1)

        def forward(self, x, demo, text):
            temp = self.in_proj(x) + self.pos[:, :x.shape[1]]
            temp = self.temporal_norm(self.temporal(temp))
            temp_pool = temp.mean(dim=1)
            txt = self.text(text)
            q = txt.unsqueeze(1)
            cross, _ = self.cross(q, temp, temp, need_weights=False)
            cross = cross.squeeze(1)
            dem = self.demo(demo)
            fused = self.fuse(torch.cat([temp_pool, cross, txt, dem], dim=-1))
            out = {t: self.binary_heads[t](fused).squeeze(-1) for t in BINARY_TASKS}
            out[LOS_TASK] = self.los_head(fused).squeeze(-1)
            return out
    return Net()


def z3_loader(dataset, batch_size, shuffle):
    import torch
    from torch.utils.data import DataLoader
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=torch.cuda.is_available())


def z3_target_loss(outputs, targets, valids, cfg, train_prevalence=None):
    import torch
    import torch.nn.functional as F
    losses = []
    for j, task in enumerate(TASKS):
        mask = valids[:, j] > 0.5
        if not torch.any(mask):
            continue
        y = targets[:, j]
        if task in BINARY_TASKS:
            yy = y[mask]
            logits = outputs[task][mask]
            if cfg["loss_mode"] == "focal_rare" and task in ("vasopressor_onset_24h", "readmission_30d_24h"):
                bce = F.binary_cross_entropy_with_logits(logits, yy, reduction="none")
                p = torch.sigmoid(logits)
                pt = p * yy + (1.0-p) * (1.0-yy)
                loss = (0.25 * (1.0-pt).pow(2.0) * bce).mean()
            else:
                loss = F.binary_cross_entropy_with_logits(logits, yy)
        else:
            yy = torch.log1p(torch.clamp(y[mask] - 24.0, min=0.0))
            loss = F.smooth_l1_loss(outputs[task][mask], yy)
        losses.append(loss)
    return torch.stack(losses).mean()


def z3_predict(model, loader, device, cohort, indices):
    import torch
    model.eval()
    pred_store = {t: np.full(len(indices), np.nan, dtype=np.float32) for t in TASKS}
    true_store = {t: np.full(len(indices), np.nan, dtype=np.float32) for t in TASKS}
    valid_store = {t: np.zeros(len(indices), dtype=np.float32) for t in TASKS}
    local_pos = {int(gidx): pos for pos, gidx in enumerate(indices)}
    with torch.no_grad():
        for batch in loader:
            x,d,t,targets,valids,gidx = batch
            x=x.to(device, non_blocking=True); d=d.to(device, non_blocking=True); t=t.to(device, non_blocking=True)
            outputs = model(x,d,t)
            gidx_np = gidx.numpy()
            for task_i, task in enumerate(TASKS):
                pos = np.asarray([local_pos[int(g)] for g in gidx_np], dtype=np.int64)
                if task in BINARY_TASKS:
                    p = torch.sigmoid(outputs[task]).detach().cpu().numpy()
                else:
                    p = (torch.expm1(outputs[task]) + 24.0).detach().cpu().numpy()
                    p = np.clip(p, 0.0, None)
                pred_store[task][pos] = p.astype(np.float32)
                true_store[task][pos] = targets[:, task_i].numpy()
                valid_store[task][pos] = valids[:, task_i].numpy()
    rows=[]
    sub = cohort.iloc[indices]
    for task in TASKS:
        for i, gidx in enumerate(indices):
            row = {
                "stay_id": int(cohort.iloc[gidx]["stay_id"]),
                "subject_id": int(cohort.iloc[gidx]["subject_id"]),
                "split": str(cohort.iloc[gidx]["split_24h"]),
                "task": task,
                "target": float(true_store[task][i]),
                "validity": int(valid_store[task][i]),
                "prediction": float(pred_store[task][i]),
            }
            if task in BINARY_TASKS:
                row["probability_mean"] = float(pred_store[task][i])
            rows.append(row)
    return pd.DataFrame(rows)


def evaluate_z3(frame: pd.DataFrame) -> Dict[str,float]:
    return metrics(frame)


def train_z3_once(structured, demographics, text_input, cohort, cfg, seed, run_dir: Path):
    import torch
    from torch.cuda.amp import GradScaler, autocast
    set_seed(seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_idx = np.flatnonzero(cohort["split_24h"].to_numpy() == "train")
    val_idx = np.flatnonzero(cohort["split_24h"].to_numpy() == "val")
    test_idx = np.flatnonzero(cohort["split_24h"].to_numpy() == "test")
    if len(train_idx) != EXPECTED_TRAIN or len(val_idx) != EXPECTED_VAL or len(test_idx) != EXPECTED_TEST:
        raise ValueError(f"Z3 split sizes mismatch train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
    model = build_z3_model(cfg).to(device)
    train_ds = Z3Dataset(structured, demographics, text_input, cohort, train_idx)
    val_ds = Z3Dataset(structured, demographics, text_input, cohort, val_idx)
    test_ds = Z3Dataset(structured, demographics, text_input, cohort, test_idx)
    train_loader = z3_loader(train_ds, cfg["batch_size"], True)
    val_loader = z3_loader(val_ds, cfg["batch_size"], False)
    test_loader = z3_loader(test_ds, cfg["batch_size"], False)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["max_epochs"])
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = GradScaler(enabled=(device.type == "cuda" and not use_bf16))
    best = -np.inf; best_epoch=0; patience=0
    history=[]
    best_path = run_dir / "best_checkpoint.pt"

    for epoch in range(1, cfg["max_epochs"]+1):
        model.train(); losses=[]
        for batch in train_loader:
            x,d,t,targets,valids,_ = batch
            x=x.to(device, non_blocking=True); d=d.to(device, non_blocking=True); t=t.to(device, non_blocking=True)
            targets=targets.to(device); valids=valids.to(device)
            opt.zero_grad(set_to_none=True)
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    outputs=model(x,d,t); loss=z3_target_loss(outputs,targets,valids,cfg)
            else:
                outputs=model(x,d,t); loss=z3_target_loss(outputs,targets,valids,cfg)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite Z3 training loss at epoch {epoch}")
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(opt); scaler.update()
            losses.append(float(loss.detach().cpu()))
        sched.step()
        # validation prediction
        val_pred = z3_predict(model, val_loader, device, cohort, val_idx)
        val_m = metrics(val_pred)
        score = val_m["mean_binary_auprc"]
        rec={"epoch":epoch,"train_loss":float(np.mean(losses)),"val_mean_binary_auprc":float(score),"lr":float(opt.param_groups[0]["lr"])}
        history.append(rec)
        print(f"[Z3 {cfg['name']}] epoch={epoch:02d} train_loss={np.mean(losses):.5f} val_mean_AUPRC={score:.6f}", flush=True)
        if score > best + 1e-7:
            best=score; best_epoch=epoch; patience=0
            torch.save({"model_state":model.state_dict(),"config":cfg,"seed":seed,"best_epoch":best_epoch}, best_path)
        else:
            patience += 1
            if patience >= cfg["patience"]:
                break
    # restore best and predict val/test once more
    try:
        ckpt=torch.load(best_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt=torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    val_pred=z3_predict(model,val_loader,device,cohort,val_idx)
    test_pred=z3_predict(model,test_loader,device,cohort,test_idx)
    val_pred.to_csv(run_dir / "validation_predictions.csv", index=False)
    test_pred.to_csv(run_dir / "test_predictions.csv", index=False)
    (run_dir / "training_history.json").write_text(json.dumps(history, indent=2))
    meta={"config":cfg,"seed":int(seed),"device":str(device),"bf16":bool(use_bf16),"best_epoch":int(best_epoch),"validation_metrics":metrics(val_pred),"test_metrics":metrics(test_pred)}
    (run_dir / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    return val_pred,test_pred,meta


# =============================================================================
# Main
# =============================================================================

def build_args():
    p=argparse.ArgumentParser()
    p.add_argument("--cv-folds",type=int,default=3)
    p.add_argument("--z1-min-gain",type=float,default=0.0005)
    p.add_argument("--bootstrap-reps",type=int,default=2000)
    p.add_argument("--seed",type=int,default=SEED)
    p.add_argument("--text-path",type=str,default=None)
    p.add_argument("--z3-tuned-trials",type=int,default=4,choices=[1,2,3,4])
    p.add_argument("--skip-z3",action="store_true")
    return p.parse_args()


def save_model_outputs(name, val, test, outdir):
    val.to_csv(outdir / f"{name}_validation_predictions.csv", index=False)
    test.to_csv(outdir / f"{name}_test_predictions.csv", index=False)


def main():
    args=build_args()
    OUTPUT_ROOT.mkdir(parents=True,exist_ok=True)
    print("="*110)
    print("MODEL Z FAMILY — Z1 + Z2 + Z3")
    print("="*110)
    print("NO NEW PREPROCESSING: frozen prediction artifacts + existing Pipeline-A-V2 arrays/embeddings only")
    print(f"Seed={args.seed} | bootstrap={args.bootstrap_reps} | Z3 skipped={args.skip_z3}")

    candidates=discover_candidates()

    # ---- A3 / A2 references ----
    a3_recipes,a2_recipes,a3v,a3t,a2v,a2t=build_reference_ensembles(candidates)
    save_model_outputs("A3_rank_max5",a3v,a3t,OUTPUT_ROOT)
    save_model_outputs("A2_probability_mean_max3",a2v,a2t,OUTPUT_ROOT)
    print(f"A3 validation mean AUPRC={metrics(a3v)['mean_binary_auprc']:.6f}",flush=True)
    print(f"A2 validation mean AUPRC={metrics(a2v)['mean_binary_auprc']:.6f}",flush=True)

    # ---- Z1 fixed/tuned ----
    z1_fixed_rec,z1fv,z1ft,z1_fixed_cv=z1_make(candidates,False,args.cv_folds,args.seed)
    z1_tuned_rec,z1tv,z1tt,z1_tuned_cv=z1_make(candidates,True,args.cv_folds,args.seed)
    save_model_outputs("Z1_fixed",z1fv,z1ft,OUTPUT_ROOT)
    save_model_outputs("Z1_tuned",z1tv,z1tt,OUTPUT_ROOT)
    z1_fixed_cv.to_csv(OUTPUT_ROOT/"Z1_fixed_cv.csv",index=False); z1_tuned_cv.to_csv(OUTPUT_ROOT/"Z1_tuned_cv.csv",index=False)
    (OUTPUT_ROOT/"Z1_fixed_recipes.json").write_text(json.dumps(z1_fixed_rec,indent=2,default=str))
    (OUTPUT_ROOT/"Z1_tuned_recipes.json").write_text(json.dumps(z1_tuned_rec,indent=2,default=str))

    # ---- Z2 fixed/tuned ----
    z2_fixed_rec,z2fv,z2ft=build_z2(candidates,a3v,a3t,a2v,a2t,z1fv,z1ft,False)
    z2_tuned_rec,z2tv,z2tt=build_z2(candidates,a3v,a3t,a2v,a2t,z1tv,z1tt,True)
    save_model_outputs("Z2_fixed",z2fv,z2ft,OUTPUT_ROOT); save_model_outputs("Z2_tuned",z2tv,z2tt,OUTPUT_ROOT)
    (OUTPUT_ROOT/"Z2_fixed_recipes.json").write_text(json.dumps(z2_fixed_rec,indent=2,default=str))
    (OUTPUT_ROOT/"Z2_tuned_recipes.json").write_text(json.dumps(z2_tuned_rec,indent=2,default=str))

    outputs={
        "A3_rank_max5":(a3v,a3t),
        "A2_probability_mean_max3":(a2v,a2t),
        "Z1_fixed":(z1fv,z1ft),
        "Z1_tuned":(z1tv,z1tt),
        "Z2_fixed":(z2fv,z2ft),
        "Z2_tuned":(z2tv,z2tt),
    }

    # ---- Z3 fixed/tuned ----
    z3_trial_table=[]
    if not args.skip_z3:
        structured_path,demographics_path,alignment_path,text_path=locate_z3_inputs(args.text_path)
        print(f"[Z3] structured={structured_path}",flush=True)
        print(f"[Z3] demographics={demographics_path}",flush=True)
        print(f"[Z3] alignment={alignment_path}",flush=True)
        print(f"[Z3] frozen text embeddings={text_path}",flush=True)
        structured,demographics,text_input,cohort=load_z3_data(text_path)
        fixed_cfg,tuned_cfgs=z3_configs()
        fixed_cfg_dir=OUTPUT_ROOT/"Z3_fixed"; fixed_cfg_dir.mkdir(parents=True,exist_ok=True)
        z3fv,z3ft,z3meta=train_z3_once(structured,demographics,text_input,cohort,fixed_cfg,args.seed,fixed_cfg_dir)
        outputs["Z3_fixed"]=(z3fv,z3ft)
        save_model_outputs("Z3_fixed",z3fv,z3ft,OUTPUT_ROOT)

        selected=None
        for cfg in tuned_cfgs[:args.z3_tuned_trials]:
            run_dir=OUTPUT_ROOT/"Z3_tuned_trials"/cfg["name"]
            vv,tt,meta=train_z3_once(structured,demographics,text_input,cohort,cfg,args.seed+100+len(z3_trial_table),run_dir)
            z3_trial_table.append({"trial":cfg["name"],"validation_mean_AUPRC":meta["validation_metrics"]["mean_binary_auprc"],"best_epoch":meta["best_epoch"],"config":json.dumps(cfg)})
            print(f"[Z3-TUNE] {cfg['name']} validation mean AUPRC={meta['validation_metrics']['mean_binary_auprc']:.6f}",flush=True)
            if selected is None or meta["validation_metrics"]["mean_binary_auprc"] > selected["meta"]["validation_metrics"]["mean_binary_auprc"]:
                selected={"cfg":cfg,"val":vv,"test":tt,"meta":meta,"run_dir":run_dir}
        if selected is None:
            raise RuntimeError("Z3 tuning produced no trials")
        # Preserve the pure tuned multimodal result as an audit artifact.
        z3_raw_val=selected["val"]; z3_raw_test=selected["test"]
        save_model_outputs("Z3_tuned_raw_multimodal",z3_raw_val,z3_raw_test,OUTPUT_ROOT)
        (OUTPUT_ROOT/"Z3_tuned_selected.json").write_text(json.dumps(selected["meta"],indent=2,default=str))
        pd.DataFrame(z3_trial_table).to_csv(OUTPUT_ROOT/"Z3_tuned_trials.csv",index=False)

        # Final Z3 tuned member: validation-only gate between A3, tuned Z1, and
        # the selected multimodal model. This is the elite-expert upgrade.
        z3_gate_rec,z3tv,z3tt=build_z3_gated(a3v,a3t,z1tv,z1tt,z3_raw_val,z3_raw_test,True)
        outputs["Z3_tuned"]=(z3tv,z3tt)
        save_model_outputs("Z3_tuned",z3tv,z3tt,OUTPUT_ROOT)
        (OUTPUT_ROOT/"Z3_tuned_gate_recipes.json").write_text(json.dumps(z3_gate_rec,indent=2,default=str))

    # ---- Reference Model32 test frame ----
    ref_parts=[]
    for task in TASKS:
        z=task_frame(candidates,REFERENCE_MODEL,"test",task)
        ref_parts.append(z[["stay_id","subject_id","target","validity"]].assign(task=task,prediction=z["prediction"].to_numpy(float)))
    ref_test=pd.concat(ref_parts,ignore_index=True)

    # ---- Summary + stats ----
    comp_rows=[]; boot_rows=[]; det_rows=[]
    for name,(val,test) in outputs.items():
        vm=metrics(val); tm=metrics(test); dv=deterioration_from_frame(val); dt=deterioration_from_frame(test)
        comp_rows.append({
            "model":name,
            "validation_mean_auprc":vm["mean_binary_auprc"],
            "validation_mean_auroc":vm["mean_binary_auroc"],
            "test_mean_auprc":tm["mean_binary_auprc"],
            "test_mean_auroc":tm["mean_binary_auroc"],
            "test_los_mae":tm[f"{LOS_TASK}_mae"],
            "test_delta_vs_A3":tm["mean_binary_auprc"]-metrics(a3t)["mean_binary_auprc"],
            "test_delta_vs_Model32":tm["mean_binary_auprc"]-metrics(ref_test)["mean_binary_auprc"],
        })
        det_rows.append({"model":name,**{f"val_{k}":v for k,v in dv.items()},**{f"test_{k}":v for k,v in dt.items()}})
        for j,task in enumerate(TASKS):
            for refname,refdf in (("A3",a3t),("Model32",ref_test)):
                b=bootstrap_pair(test,refdf,task,args.bootstrap_reps,args.seed+1000+j+(0 if refname=="A3" else 100))
                b.update({"model":name,"reference":refname})
                boot_rows.append(b)

    comparison=pd.DataFrame(comp_rows).sort_values("validation_mean_auprc",ascending=False)
    comparison.to_csv(OUTPUT_ROOT/"family_comparison.csv",index=False)
    pd.DataFrame(det_rows).to_csv(OUTPUT_ROOT/"deterioration_comparison.csv",index=False)
    pd.DataFrame(boot_rows).to_csv(OUTPUT_ROOT/"paired_bootstrap_all.csv",index=False)
    (OUTPUT_ROOT/"A3_recipes.json").write_text(json.dumps(a3_recipes,indent=2,default=str))
    (OUTPUT_ROOT/"A2_recipes.json").write_text(json.dumps(a2_recipes,indent=2,default=str))

    summary={
        "model_family":"Z1_Z2_Z3",
        "no_new_preprocessing":True,
        "base_model_ids":MODEL_IDS,
        "reference":"A3_rank_max5",
        "primary_selection_metric":"validation_mean_binary_auprc",
        "upstream_validation_dependence":True,
        "metrics":comp_rows,
        "deterioration":det_rows,
        "z3_skipped":bool(args.skip_z3),
        "z3_trial_count":len(z3_trial_table),
        "z3_tuned_gate_uses": ["A3_rank_max5", "Z1_tuned", "selected_Z3_multimodal"],
        "a3_validation_mean_auprc":metrics(a3v)["mean_binary_auprc"],
    }
    (OUTPUT_ROOT/"family_summary.json").write_text(json.dumps(summary,indent=2,default=str))

    print("\n"+"="*110)
    print("MODEL Z FAMILY FINAL COMPARISON")
    print("="*110)
    print(comparison.to_string(index=False))
    print("\nArtifacts:",OUTPUT_ROOT)
    print("PRIMARY REFERENCE A3 validation mean AUPRC:",metrics(a3v)["mean_binary_auprc"])


if __name__ == "__main__":
    main()
