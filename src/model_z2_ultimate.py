
#!/usr/bin/env python3
"""
FINAL Z2 ULTIMATE — advanced frozen-prediction ensemble experiment
===================================================================

Purpose
-------
One final, CPU-only expansion of Z2 using ONLY frozen existing prediction
artifacts. No new MIMIC preprocessing and no new neural architecture.

Frozen inputs
-------------
- Existing tuned prediction artifacts for models 00..32.
- Existing Pipeline-A-V2 split/identity carried by prediction files.
- No test-based model selection.

Experiment
----------
Previous Z2:
    the actual selected Z2_tuned artifact from the completed Z-family run.
    Its task-specific validation-tuned weights are preserved exactly as the
    historical primary comparator; this script does not substitute the older
    fixed 0.4/0.2/0.4 mixture.

Ultimate Z2 adds fixed candidate experts:
    1 A3_rank_max5
    2 A2_probability_mean_max3
    3 Z1_tuned
    4 task_best_single
    5 task_best_pair
    6 error_complement
    7 all33_rank_mean
    8 top10_rank_mean
    9 two_regime_gate
   10 previous_Z2

For binary tasks:
    selection/evaluation score is weighted rank mean; raw probability mean is
    retained separately for deterioration/calibration.

For LOS:
    weighted raw-hour mean only; LOS never enters primary model selection.

Tuning
------
- 30 fixed candidate starting points on a 0.05 simplex grid.
- 5-fold subject-grouped nested cross-fitting on validation.
  Specialist selection, top-10 selection, gate fitting, and previous-Z2 fitting
  occur inside each training fold before scoring its held-out fold.
- Local pair-transfer coordinate optimization, bounded [0.05, 0.60].
- Primary Z2-ultimate selection metric = OOF mean of the four binary AUPRCs.
- 100 subject-bootstrap refits of the selected source weights.
- Final weights = average bootstrap weight vector.
- Full-validation score reported but NOT used instead of OOF selection.

Final decision
--------------
Ultimate Z2 replaces Previous Z2 only if:
    nested OOF mean binary AUPRC exceeds the actual historical Z2_tuned
    validation mean AUPRC by >= 0.0005.

Otherwise Previous Z2 remains the primary model.

Test
----
- Test is evaluated only AFTER the final decision is frozen.
- Test is never used to choose a trial, weight, source, or model.
- Subject-clustered bootstrap comparisons are reported against Previous Z2,
  A3, and Model32.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score
from sklearn.model_selection import GroupKFold


PROJECT_ROOT = Path("/home/abdullah2025/fyp_project")
SEARCH_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_OUTPUT = SEARCH_ROOT / "model_z2_ultimate"

SEED = 20260923
EXPECTED_VAL = 7417
EXPECTED_TEST = 14887

BINARY_TASKS = (
    "mortality_24h",
    "ventilation_onset_24h",
    "vasopressor_onset_24h",
    "readmission_30d_24h",
)
LOS_TASK = "los_hours_24h"
TASKS = BINARY_TASKS + (LOS_TASK,)
MODEL_IDS = [f"{i:02d}" for i in range(33)]
REFERENCE_MODEL = "32"

Z2_WEIGHT_MIN = 0.05
Z2_WEIGHT_MAX = 0.60
Z2_WEIGHT_STEP = 0.05
N_TRIALS_DEFAULT = 30
CV_FOLDS_DEFAULT = 5
BOOTSTRAP_WEIGHT_DEFAULT = 100
BOOTSTRAP_REPS_DEFAULT = 2000
MIN_OOF_GAIN = 0.0005

SOURCE_NAMES = [
    "A3_rank_max5",
    "A2_probability_mean_max3",
    "Z1_tuned",
    "task_best_single",
    "task_best_pair",
    "error_complement",
    "all33_rank_mean",
    "top10_rank_mean",
    "two_regime_gate",
    "previous_Z2",
]


def safe_ap(y, p):
    # Scalar binary target/prediction only. Flatten any pandas/sklearn singleton
    # dimension so average_precision_score always receives two 1-D arrays.
    y = np.asarray(y, dtype=float).reshape(-1)
    p = np.asarray(p, dtype=float).reshape(-1)
    if y.shape[0] != p.shape[0]:
        raise ValueError(f"safe_ap length mismatch: y={y.shape}, p={p.shape}")
    keep = np.isfinite(y) & np.isfinite(p)
    y = y[keep].astype(int)
    p = p[keep].astype(float)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, p))


def safe_auc(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    keep = np.isfinite(y) & np.isfinite(p)
    y = y[keep].astype(int)
    p = p[keep]
    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def rank01(x):
    x = np.asarray(x, float)
    if len(x) == 0:
        return x.copy()
    out = pd.Series(x).rank(method="average").to_numpy(float) / float(len(x))
    # Deterministic numeric safety.
    return np.nan_to_num(out, nan=0.5, posinf=1.0, neginf=0.0)


def task_frame(candidates, model_id, split, task):
    z = candidates[model_id]
    out = z[(z["split"] == split) & (z["task"] == task)].copy()
    if out.empty:
        raise ValueError(f"No rows for model={model_id}, split={split}, task={task}")
    out = out.sort_values("stay_id").reset_index(drop=True)
    if out["stay_id"].duplicated().any():
        raise ValueError(f"Duplicate stay_id: model={model_id}, split={split}, task={task}")
    return out


def model_id_from_path(path):
    for part in reversed(path.parts):
        m = re.match(r"^(\d{2})_", part)
        if m:
            return m.group(1)
    return None


def discover_candidates(search_root):
    found = {mid: [] for mid in MODEL_IDS}
    for path in search_root.glob("**/final_selected_configuration/test_predictions.csv"):
        mid = model_id_from_path(path)
        if mid in found:
            found[mid].append(path)

    candidates = {}
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
        required = {"stay_id", "subject_id", "split", "task", "target", "validity", "prediction"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")

        for split, expected in (("val", EXPECTED_VAL), ("test", EXPECTED_TEST)):
            n = int(df.loc[df["split"] == split, "stay_id"].nunique())
            if n != expected:
                raise ValueError(
                    f"Model {mid}: {split} stays={n}, expected={expected}: {path}"
                )

        candidates[mid] = df
        print(f"[BASE] {mid}: {path}", flush=True)

    if problems:
        raise FileNotFoundError("Prediction discovery failed:\n" + "\n".join(problems))
    return candidates


def align(candidates, model_ids, split, task):
    first = task_frame(candidates, model_ids[0], split, task)
    out = first[["stay_id", "subject_id", "target", "validity"]].copy()
    out[f"pred_{model_ids[0]}"] = first["prediction"].to_numpy(float)

    for mid in model_ids[1:]:
        z = task_frame(candidates, mid, split, task)
        if not np.array_equal(out["stay_id"].to_numpy(), z["stay_id"].to_numpy()):
            raise ValueError(f"stay_id alignment failure: {mid}/{split}/{task}")
        if not np.array_equal(out["subject_id"].to_numpy(), z["subject_id"].to_numpy()):
            raise ValueError(f"subject_id alignment failure: {mid}/{split}/{task}")
        out[f"pred_{mid}"] = z["prediction"].to_numpy(float)

    return out


def valid_mask(frame, ids):
    keep = frame["validity"].astype(bool).to_numpy()
    keep &= np.isfinite(frame["target"].to_numpy(float))
    for mid in ids:
        keep &= np.isfinite(frame[f"pred_{mid}"].to_numpy(float))
    return keep


def metrics(frame):
    out = {}
    for task in TASKS:
        z = frame[(frame["task"] == task) & frame["validity"].astype(bool)].copy()
        y = z["target"].to_numpy(float)
        p = z["prediction"].to_numpy(float)
        keep = np.isfinite(y) & np.isfinite(p)
        y, p = y[keep], p[keep]
        if task in BINARY_TASKS:
            out[f"{task}_auprc"] = safe_ap(y, p)
            out[f"{task}_auroc"] = safe_auc(y, p)
        else:
            out[f"{task}_mae"] = float(mean_absolute_error(y, p)) if len(y) else float("nan")
    out["mean_binary_auprc"] = float(np.nanmean([out[f"{t}_auprc"] for t in BINARY_TASKS]))
    out["mean_binary_auroc"] = float(np.nanmean([out[f"{t}_auroc"] for t in BINARY_TASKS]))
    return out


def weighted_rank_apply(frame, members, weights):
    cols = []
    for mid in members:
        cols.append(rank01(frame[f"pred_{mid}"].to_numpy(float)))
    R = np.column_stack(cols)
    w = np.asarray(weights, float)
    w = w / w.sum()
    return R @ w


def raw_apply(frame, members, weights):
    X = np.column_stack([frame[f"pred_{mid}"].to_numpy(float) for mid in members])
    w = np.asarray(weights, float)
    w = w / w.sum()
    return X @ w


def greedy_recipe(candidates, task, max_members, mode):
    frame = align(candidates, MODEL_IDS, "val", task)
    keep = valid_mask(frame, MODEL_IDS)
    y = frame["target"].to_numpy(float)
    R = np.column_stack([rank01(frame[f"pred_{m}"].to_numpy(float)) for m in MODEL_IDS])
    raw = np.column_stack([frame[f"pred_{m}"].to_numpy(float) for m in MODEL_IDS])

    scores = []
    for j, mid in enumerate(MODEL_IDS):
        pred = R[keep, j] if mode == "rank" else raw[keep, j]
        score = safe_ap(y[keep], pred) if task in BINARY_TASKS else -mean_absolute_error(y[keep], pred)
        scores.append((float(score), mid))
    scores.sort(reverse=True)

    selected = [scores[0][1]]
    current = scores[0][0]
    for _, mid in scores[1:]:
        if len(selected) >= max_members:
            break
        idx = [MODEL_IDS.index(m) for m in selected + [mid]]
        pred = (
            np.mean(R[keep][:, idx], axis=1)
            if mode == "rank"
            else np.mean(raw[keep][:, idx], axis=1)
        )
        score = safe_ap(y[keep], pred) if task in BINARY_TASKS else -mean_absolute_error(y[keep], pred)
        if score > current:
            selected.append(mid)
            current = float(score)

    weights = {m: 1.0 / len(selected) for m in selected}
    return {
        "task": task,
        "members": selected,
        "weights": weights,
        "validation_score": float(current),
        "blend": "rank_mean" if mode == "rank" else "raw_mean",
    }


def apply_recipe(candidates, recipe, split):
    task = recipe["task"]
    members = list(recipe["members"])
    frame = align(candidates, members, split, task)
    w = np.asarray([recipe["weights"][m] for m in members], float)

    if task in BINARY_TASKS:
        pred = weighted_rank_apply(frame, members, w)
        prob = np.clip(raw_apply(frame, members, w), 0.0, 1.0)
    else:
        pred = raw_apply(frame, members, w)
        prob = None

    out = frame[["stay_id", "subject_id", "target", "validity"]].copy()
    out["task"] = task
    out["prediction"] = pred
    if prob is not None:
        out["probability_mean"] = prob
    return out


def build_a3_a2(candidates):
    a3v, a3t, a2v, a2t = [], [], [], []
    recipes = {"A3": {}, "A2": {}}
    for task in TASKS:
        a3 = greedy_recipe(candidates, task, 5, "rank" if task in BINARY_TASKS else "raw")
        a2 = greedy_recipe(candidates, task, 3, "raw")
        recipes["A3"][task] = a3
        recipes["A2"][task] = a2
        a3v.append(apply_recipe(candidates, a3, "val"))
        a3t.append(apply_recipe(candidates, a3, "test"))
        a2v.append(apply_recipe(candidates, a2, "val"))
        a2t.append(apply_recipe(candidates, a2, "test"))
    return (
        recipes,
        pd.concat(a3v, ignore_index=True),
        pd.concat(a3t, ignore_index=True),
        pd.concat(a2v, ignore_index=True),
        pd.concat(a2t, ignore_index=True),
    )


def z1_tuned(candidates, cv_folds=5):
    recipes = {}
    val_parts, test_parts = [], []
    cv_rows = []

    for task in BINARY_TASKS:
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
                trf = full.iloc[tr].reset_index(drop=True)
                vaf = full.iloc[va].reset_index(drop=True)
                r = greedy_recipe_from_frame(trf, task, k)
                mids = r["members"]
                w = np.asarray([r["weights"][m] for m in mids], float)
                oof[va] = weighted_rank_apply(vaf, mids, w)
            sc = safe_ap(y, oof)
            cv_rows.append({"task": task, "K": k, "oof_auprc": sc})
            print(f"[Z1-CV] {task} K={k}: {sc:.6f}", flush=True)
            if sc > best_score:
                best_score, best_k = sc, k

        # Refit weighted-rank recipe on full validation.
        r = greedy_recipe_from_frame(full, task, best_k)
        r["selection"] = "grouped_CV_K5_6_7_8"
        r["selected_k"] = int(best_k)
        r["cv_score"] = float(best_score)
        recipes[task] = r

        val_parts.append(apply_recipe(candidates, r, "val"))
        test_parts.append(apply_recipe(candidates, r, "test"))

    # LOS is raw-hour mean and does not enter primary selection.
    r_los = greedy_recipe(candidates, LOS_TASK, 5, "raw")
    recipes[LOS_TASK] = r_los
    val_parts.append(apply_recipe(candidates, r_los, "val"))
    test_parts.append(apply_recipe(candidates, r_los, "test"))

    return (
        recipes,
        pd.concat(val_parts, ignore_index=True),
        pd.concat(test_parts, ignore_index=True),
        pd.DataFrame(cv_rows),
    )


def greedy_recipe_from_frame(frame, task, max_members):
    ids = MODEL_IDS
    y = frame["target"].to_numpy(float)
    R = np.column_stack([rank01(frame[f"pred_{m}"].to_numpy(float)) for m in ids])
    keep = np.isfinite(y)

    single = []
    for j, mid in enumerate(ids):
        sc = safe_ap(y[keep], R[keep, j])
        single.append((sc, mid))
    single.sort(reverse=True)

    selected = [single[0][1]]
    current = single[0][0]
    for _, mid in single[1:]:
        if len(selected) >= max_members:
            break
        idx = [ids.index(m) for m in selected + [mid]]
        pred = R[keep][:, idx].mean(axis=1)
        sc = safe_ap(y[keep], pred)
        if sc > current:
            selected.append(mid)
            current = sc

    w = {m: 1.0 / len(selected) for m in selected}
    return {
        "task": task,
        "members": selected,
        "weights": w,
        "validation_score": float(current),
        "blend": "rank_mean",
    }


def load_actual_z2_tuned() -> Tuple[pd.DataFrame, pd.DataFrame]:
    root = SEARCH_ROOT / "model_z_family"
    val_path = root / "Z2_tuned_validation_predictions.csv"
    test_path = root / "Z2_tuned_test_predictions.csv"
    if not val_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Actual selected Z2_tuned prediction artifacts not found. "
            f"Expected {val_path} and {test_path}."
        )
    val = pd.read_csv(val_path)
    test = pd.read_csv(test_path)
    required = {"stay_id", "subject_id", "task", "target", "validity", "prediction"}
    for path, df in ((val_path, val), (test_path, test)):
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"{path} missing columns: {missing}")
        if df.duplicated(["stay_id", "task"]).any():
            raise ValueError(f"Duplicate stay_id/task rows in actual Z2_tuned artifact: {path}")
    return val, test


def specialist_single_pair(candidates):
    single_rows = []
    pair_rows = []
    comp_rows = []

    for task in TASKS:
        frame = align(candidates, MODEL_IDS, "val", task)
        keep = valid_mask(frame, MODEL_IDS)
        y = frame["target"].to_numpy(float)

        # Single specialist.
        single_scores = []
        for mid in MODEL_IDS:
            p = frame[f"pred_{mid}"].to_numpy(float)
            if task in BINARY_TASKS:
                score = safe_ap(y[keep], p[keep])
            else:
                score = -mean_absolute_error(y[keep], p[keep])
            single_scores.append((float(score), mid))
        single_scores.sort(reverse=True)
        best_single = single_scores[0][1]

        # Best pair. For binaries use rank mean; LOS raw mean.
        best_pair = None
        for i, m1 in enumerate(MODEL_IDS):
            p1 = frame[f"pred_{m1}"].to_numpy(float)
            for m2 in MODEL_IDS[i + 1:]:
                p2 = frame[f"pred_{m2}"].to_numpy(float)
                if task in BINARY_TASKS:
                    pred = (rank01(p1) + rank01(p2)) / 2.0
                    score = safe_ap(y[keep], pred[keep])
                else:
                    pred = (p1 + p2) / 2.0
                    score = -mean_absolute_error(y[keep], pred[keep])
                item = (float(score), m1, m2)
                if best_pair is None or item[0] > best_pair[0]:
                    best_pair = item

        # Error-complement to A3: residual correlation on validation.
        a3_rec = greedy_recipe(candidates, task, 5, "rank" if task in BINARY_TASKS else "raw")
        a3_frame = apply_recipe(candidates, a3_rec, "val")
        a3_frame = a3_frame.sort_values("stay_id").reset_index(drop=True)
        a3p = a3_frame["prediction"].to_numpy(float)
        err_a3 = y - a3p

        corr_best = None
        for mid in MODEL_IDS:
            pm = frame[f"pred_{mid}"].to_numpy(float)
            resid = y - pm
            m = keep & np.isfinite(err_a3) & np.isfinite(resid)
            if m.sum() < 3:
                continue
            c = np.corrcoef(err_a3[m], resid[m])[0, 1]
            if not np.isfinite(c):
                c = 1.0
            item = (float(abs(c)), mid, float(c))
            if corr_best is None or item[0] < corr_best[0]:
                corr_best = item

        if corr_best is None:
            error_complement = best_single
        else:
            error_complement = corr_best[1]

        single_rows.append({
            "task": task,
            "model_id": best_single,
            "score": float(single_scores[0][0]),
        })
        pair_rows.append({
            "task": task,
            "model1": best_pair[1],
            "model2": best_pair[2],
            "score": float(best_pair[0]),
        })
        comp_rows.append({
            "task": task,
            "model_id": error_complement,
            "residual_abs_corr_with_A3": float(corr_best[0]) if corr_best else float("nan"),
            "residual_corr_with_A3": float(corr_best[2]) if corr_best else float("nan"),
        })

        print(
            f"[SPECIALIST] {task}: single={best_single}, "
            f"pair=({best_pair[1]},{best_pair[2]}), "
            f"error_complement={error_complement}",
            flush=True,
        )

    return pd.DataFrame(single_rows), pd.DataFrame(pair_rows), pd.DataFrame(comp_rows)


def build_specialist_frame(candidates, split, task, specialist_single, specialist_pair, complements):
    base = task_frame(candidates, "00", split, task)
    base = base[["stay_id", "subject_id", "target", "validity"]].copy()

    srow = specialist_single[ specialist_single["task"] == task ].iloc[0]
    prow = specialist_pair[ specialist_pair["task"] == task ].iloc[0]
    crow = complements[ complements["task"] == task ].iloc[0]

    m = str(srow["model_id"])
    p1 = str(prow["model1"])
    p2 = str(prow["model2"])
    ec = str(crow["model_id"])

    for mid in {m, p1, p2, ec}:
        z = task_frame(candidates, mid, split, task)
        if not np.array_equal(base["stay_id"].to_numpy(), z["stay_id"].to_numpy()):
            raise ValueError(f"Specialist alignment failure {mid}/{task}/{split}")

    pm = task_frame(candidates, m, split, task)["prediction"].to_numpy(float)
    pp1 = task_frame(candidates, p1, split, task)["prediction"].to_numpy(float)
    pp2 = task_frame(candidates, p2, split, task)["prediction"].to_numpy(float)
    pe = task_frame(candidates, ec, split, task)["prediction"].to_numpy(float)

    out = base.copy()
    if task in BINARY_TASKS:
        out["task_best_single"] = rank01(pm)
        out["task_best_single_prob"] = np.clip(pm, 0.0, 1.0)
        out["task_best_pair"] = (rank01(pp1) + rank01(pp2)) / 2.0
        out["task_best_pair_prob"] = np.clip((pp1 + pp2) / 2.0, 0.0, 1.0)
        out["error_complement"] = rank01(pe)
        out["error_complement_prob"] = np.clip(pe, 0.0, 1.0)
    else:
        out["task_best_single"] = pm
        out["task_best_single_prob"] = np.nan
        out["task_best_pair"] = (pp1 + pp2) / 2.0
        out["task_best_pair_prob"] = np.nan
        out["error_complement"] = pe
        out["error_complement_prob"] = np.nan
    return out


def build_consensus_frame(candidates, split, task):
    frame = align(candidates, MODEL_IDS, split, task)
    base = frame[["stay_id", "subject_id", "target", "validity"]].copy()
    R = np.column_stack([rank01(frame[f"pred_{m}"].to_numpy(float)) for m in MODEL_IDS])
    raw = np.column_stack([frame[f"pred_{m}"].to_numpy(float) for m in MODEL_IDS])

    top10 = []
    if split == "val":
        y = frame["target"].to_numpy(float)
        keep = valid_mask(frame, MODEL_IDS)
        scores = []
        for j, mid in enumerate(MODEL_IDS):
            scores.append((safe_ap(y[keep], raw[keep, j]) if task in BINARY_TASKS else -mean_absolute_error(y[keep], raw[keep, j]), mid))
        top10 = [m for _, m in sorted(scores, reverse=True)[:10]]
    else:
        # Read the already selected validation top-10 from a deterministic saved artifact
        # generated by the validation call below.
        raise RuntimeError("build_consensus_frame(test) requires top10 members from validation.")

    out = base.copy()
    out["all33_rank_mean"] = R.mean(axis=1)
    out["all33_prob_mean"] = np.clip(raw.mean(axis=1), 0.0, 1.0) if task in BINARY_TASKS else np.nan

    idx = [MODEL_IDS.index(m) for m in top10]
    out["top10_rank_mean"] = R[:, idx].mean(axis=1)
    out["top10_prob_mean"] = np.clip(raw[:, idx].mean(axis=1), 0.0, 1.0) if task in BINARY_TASKS else np.nan
    return out, top10


def build_consensus_with_fixed_top10(candidates, split, task, top10):
    frame = align(candidates, MODEL_IDS, split, task)
    base = frame[["stay_id", "subject_id", "target", "validity"]].copy()
    R = np.column_stack([rank01(frame[f"pred_{m}"].to_numpy(float)) for m in MODEL_IDS])
    raw = np.column_stack([frame[f"pred_{m}"].to_numpy(float) for m in MODEL_IDS])
    idx = [MODEL_IDS.index(m) for m in top10]

    out = base.copy()
    out["all33_rank_mean"] = R.mean(axis=1)
    out["all33_prob_mean"] = np.clip(raw.mean(axis=1), 0.0, 1.0) if task in BINARY_TASKS else np.nan
    out["top10_rank_mean"] = R[:, idx].mean(axis=1)
    out["top10_prob_mean"] = np.clip(raw[:, idx].mean(axis=1), 0.0, 1.0) if task in BINARY_TASKS else np.nan
    return out


def tune_two_regime_gate(candidates, task, specialist_single, specialist_pair, complements, z1_df, split="val"):
    # Disagreement is defined in rank space, avoiding LOS scale dependence.
    a3_rec = greedy_recipe(candidates, task, 5, "rank" if task in BINARY_TASKS else "raw")
    a3 = apply_recipe(candidates, a3_rec, split)
    z1 = z1_df
    z1 = z1[z1["task"] == task].sort_values("stay_id").reset_index(drop=True)
    sp = build_specialist_frame(candidates, split, task, specialist_single, specialist_pair, complements)

    a3p = a3["prediction"].to_numpy(float)
    z1p = z1["prediction"].to_numpy(float)
    spp = sp["task_best_single"].to_numpy(float)

    disagreement = np.std(np.column_stack([rank01(a3p), rank01(z1p), rank01(spp)]), axis=1)
    median_dis = float(np.nanmedian(disagreement))
    low = disagreement <= median_dis
    high = disagreement > median_dis

    y = a3["target"].to_numpy(float)
    valid = a3["validity"].astype(bool).to_numpy() & np.isfinite(y)
    best = None

    grid = np.arange(0.0, 1.0001, Z2_WEIGHT_STEP)
    for low_a3 in grid:
        low_z1 = 1.0 - low_a3
        for high_sp in grid:
            high_ec = 1.0 - high_sp

            if task in BINARY_TASKS:
                low_pred = low_a3 * rank01(a3p) + low_z1 * rank01(z1p)
                high_pred = high_sp * rank01(sp["task_best_single"].to_numpy(float)) + high_ec * rank01(sp["error_complement"].to_numpy(float))
            else:
                low_pred = low_a3 * a3p + low_z1 * z1p
                high_pred = high_sp * sp["task_best_single"].to_numpy(float) + high_ec * sp["error_complement"].to_numpy(float)

            pred = np.empty(len(y), float)
            pred[low] = low_pred[low]
            pred[high] = high_pred[high]
            keep = valid & np.isfinite(pred)
            if task in BINARY_TASKS:
                score = safe_ap(y[keep], pred[keep])
            else:
                score = -mean_absolute_error(y[keep], pred[keep])

            if best is None or score > best[0]:
                best = (float(score), {
                    "threshold": median_dis,
                    "low_weights": [float(low_a3), float(low_z1)],
                    "high_weights": [float(high_sp), float(high_ec)],
                })
    assert best is not None
    return best[1]


def apply_two_regime_gate(candidates, split, task, params, specialist_single, specialist_pair, complements, z1_df):
    a3_rec = greedy_recipe(candidates, task, 5, "rank" if task in BINARY_TASKS else "raw")
    a3 = apply_recipe(candidates, a3_rec, split)
    z1 = z1_df[z1_df["task"] == task].sort_values("stay_id").reset_index(drop=True)
    sp = build_specialist_frame(candidates, split, task, specialist_single, specialist_pair, complements)

    a3p = a3["prediction"].to_numpy(float)
    z1p = z1["prediction"].to_numpy(float)
    spp = sp["task_best_single"].to_numpy(float)
    ecp = sp["error_complement"].to_numpy(float)

    disagreement = np.std(np.column_stack([rank01(a3p), rank01(z1p), rank01(spp)]), axis=1)
    low = disagreement <= params["threshold"]
    high = disagreement > params["threshold"]

    pred = np.full(len(a3p), np.nan, dtype=float)
    if task in BINARY_TASKS:
        lowp = params["low_weights"][0] * rank01(a3p) + params["low_weights"][1] * rank01(z1p)
        highp = params["high_weights"][0] * rank01(spp) + params["high_weights"][1] * rank01(ecp)
        pred[low] = lowp[low]
        pred[high] = highp[high]
        a3prob = np.clip(a3.get("probability_mean", a3["prediction"]).to_numpy(float), 0, 1)
        z1prob = np.clip(z1.get("probability_mean", z1["prediction"]).to_numpy(float), 0, 1)
        spprob = np.clip(sp["task_best_single_prob"].to_numpy(float), 0, 1)
        ecprob = np.clip(sp["error_complement_prob"].to_numpy(float), 0, 1)
        prob = np.full(len(a3p), np.nan, dtype=float)
        prob[low] = params["low_weights"][0] * a3prob[low] + params["low_weights"][1] * z1prob[low]
        prob[high] = params["high_weights"][0] * spprob[high] + params["high_weights"][1] * ecprob[high]
    else:
        lowp = params["low_weights"][0] * a3p + params["low_weights"][1] * z1p
        highp = params["high_weights"][0] * spp + params["high_weights"][1] * ecp
        pred[low] = lowp[low]
        pred[high] = highp[high]
        prob = None

    bad = ~np.isfinite(pred)
    if bad.any():
        fb = np.where(np.isfinite(a3p), a3p, z1p)
        fb = np.where(np.isfinite(fb), fb, 0.0)
        pred[bad] = fb[bad]
        if prob is not None:
            prob[bad] = np.where(np.isfinite(prob[bad]), prob[bad], 0.0)

    out = a3[["stay_id", "subject_id", "target", "validity"]].copy()
    out["task"] = task
    out["prediction"] = pred
    if prob is not None:
        out["probability_mean"] = np.clip(prob, 0, 1)
    return out


def build_source_data(candidates, split, task, specialist_single, specialist_pair, complements, top10_by_task, gate_params, previous_z2, z1_df):
    base = task_frame(candidates, "00", split, task)[["stay_id", "subject_id", "target", "validity"]].copy()

    a3_rec = greedy_recipe(candidates, task, 5, "rank" if task in BINARY_TASKS else "raw")
    a3 = apply_recipe(candidates, a3_rec, split)
    a2_rec = greedy_recipe(candidates, task, 3, "raw")
    a2 = apply_recipe(candidates, a2_rec, split)

    z1 = z1_df[z1_df["task"] == task].sort_values("stay_id").reset_index(drop=True)

    sp = build_specialist_frame(candidates, split, task, specialist_single, specialist_pair, complements)
    consensus = build_consensus_with_fixed_top10(candidates, split, task, top10_by_task[task])
    gate = apply_two_regime_gate(candidates, split, task, gate_params[task], specialist_single, specialist_pair, complements, z1_df)
    prev = previous_z2[previous_z2["task"] == task].sort_values("stay_id").reset_index(drop=True)

    frames = [a3, a2, z1, sp, consensus, gate, prev]
    for f in frames:
        ff = f.sort_values("stay_id").reset_index(drop=True)
        if not np.array_equal(base["stay_id"].to_numpy(), ff["stay_id"].to_numpy()):
            raise ValueError(f"Source alignment failure {task}/{split}")
        if not np.array_equal(base["subject_id"].to_numpy(), ff["subject_id"].to_numpy()):
            raise ValueError(f"Source subject alignment failure {task}/{split}")

    score_cols = [
        a3["prediction"].to_numpy(float),
        a2["prediction"].to_numpy(float),
        z1["prediction"].to_numpy(float),
        sp["task_best_single"].to_numpy(float),
        sp["task_best_pair"].to_numpy(float),
        sp["error_complement"].to_numpy(float),
        consensus["all33_rank_mean"].to_numpy(float),
        consensus["top10_rank_mean"].to_numpy(float),
        gate["prediction"].to_numpy(float),
        prev["prediction"].to_numpy(float),
    ]
    prob_cols = [
        a3.get("probability_mean", a3["prediction"]).to_numpy(float),
        a2.get("probability_mean", a2["prediction"]).to_numpy(float),
        z1.get("probability_mean", z1["prediction"]).to_numpy(float),
        sp["task_best_single_prob"].to_numpy(float),
        sp["task_best_pair_prob"].to_numpy(float),
        sp["error_complement_prob"].to_numpy(float),
        consensus["all33_prob_mean"].to_numpy(float),
        consensus["top10_prob_mean"].to_numpy(float),
        gate.get("probability_mean", gate["prediction"]).to_numpy(float),
        prev.get("probability_mean", prev["prediction"]).to_numpy(float),
    ]
    S = np.column_stack(score_cols)
    P = np.column_stack(prob_cols)

    a3_score = np.where(np.isfinite(S[:, 0]), S[:, 0], 0.0)
    a3_prob = np.where(np.isfinite(P[:, 0]), P[:, 0], 0.0)
    for j in range(S.shape[1]):
        bad = ~np.isfinite(S[:, j])
        if bad.any():
            S[bad, j] = a3_score[bad]
    for j in range(P.shape[1]):
        bad = ~np.isfinite(P[:, j])
        if bad.any():
            P[bad, j] = a3_prob[bad]

    return base, S, P, SOURCE_NAMES.copy()




def aligned_val_frame(candidates, task):
    return align(candidates, MODEL_IDS, "val", task)


def apply_recipe_to_frame(frame, recipe):
    mids = list(recipe["members"])
    w = np.asarray([recipe["weights"][m] for m in mids], dtype=float)
    if recipe.get("blend") == "raw_mean":
        pred = np.column_stack([frame[f"pred_{m}"].to_numpy(float) for m in mids]) @ (w / w.sum())
    else:
        pred = np.column_stack([rank01(frame[f"pred_{m}"].to_numpy(float)) for m in mids]) @ (w / w.sum())
    return pred


def greedy_recipe_on_frame(frame, task, max_members, mode):
    y = frame["target"].to_numpy(float)
    keep = frame["validity"].astype(bool).to_numpy() & np.isfinite(y)
    if not np.any(keep):
        # No usable rows in this fold: return a deterministic fallback recipe.
        return {"members": [MODEL_IDS[0]], "weights": {MODEL_IDS[0]: 1.0},
                "blend": "rank_mean" if mode == "rank" else "raw_mean"}
    R = np.column_stack([rank01(frame[f"pred_{m}"].to_numpy(float)) for m in MODEL_IDS])
    raw = np.column_stack([frame[f"pred_{m}"].to_numpy(float) for m in MODEL_IDS])
    scores = []
    for j, mid in enumerate(MODEL_IDS):
        p = R[keep, j] if mode == "rank" else raw[keep, j]
        sc = safe_ap(y[keep], p) if task in BINARY_TASKS else -mean_absolute_error(y[keep], p)
        scores.append((float(sc), mid))
    scores.sort(reverse=True)
    selected = [scores[0][1]]
    current = scores[0][0]
    for _, mid in scores[1:]:
        if len(selected) >= max_members:
            break
        idx = [MODEL_IDS.index(m) for m in selected + [mid]]
        p = (R[keep][:, idx].mean(axis=1) if mode == "rank" else raw[keep][:, idx].mean(axis=1))
        sc = safe_ap(y[keep], p) if task in BINARY_TASKS else -mean_absolute_error(y[keep], p)
        if sc > current + 1e-12:
            selected.append(mid); current = float(sc)
    return {"members": selected, "weights": {m: 1.0/len(selected) for m in selected}, "blend": "rank_mean" if mode == "rank" else "raw_mean"}


def select_specialists_from_frame(frame, task):
    y = frame["target"].to_numpy(float)
    keep = frame["validity"].astype(bool).to_numpy() & np.isfinite(y)
    if not np.any(keep):
        return [MODEL_IDS[0], MODEL_IDS[1]], [MODEL_IDS[0], MODEL_IDS[1]], MODEL_IDS[0]
    single = []
    for mid in MODEL_IDS:
        p = frame[f"pred_{mid}"].to_numpy(float)
        sc = safe_ap(y[keep], p[keep]) if task in BINARY_TASKS else -mean_absolute_error(y[keep], p[keep])
        single.append((float(sc), mid))
    single.sort(reverse=True); best_single = single[0][1]
    best_pair = None
    for i, m1 in enumerate(MODEL_IDS):
        p1 = frame[f"pred_{m1}"].to_numpy(float)
        for m2 in MODEL_IDS[i+1:]:
            p2 = frame[f"pred_{m2}"].to_numpy(float)
            if task in BINARY_TASKS:
                pred = (rank01(p1) + rank01(p2))/2.0; sc = safe_ap(y[keep], pred[keep])
            else:
                pred = (p1+p2)/2.0; sc = -mean_absolute_error(y[keep], pred[keep])
            item=(float(sc),m1,m2)
            if best_pair is None or item[0] > best_pair[0]: best_pair=item
    a3 = greedy_recipe_on_frame(frame, task, 5, "rank" if task in BINARY_TASKS else "raw")
    a3p = apply_recipe_to_frame(frame, a3)
    err_a3 = y - a3p
    corr_best = None
    for mid in MODEL_IDS:
        resid = y - frame[f"pred_{mid}"].to_numpy(float)
        m = keep & np.isfinite(err_a3) & np.isfinite(resid)
        if m.sum() < 3: continue
        c = np.corrcoef(err_a3[m], resid[m])[0,1]
        if not np.isfinite(c): c=1.0
        item=(abs(float(c)), mid, float(c))
        if corr_best is None or item[0] < corr_best[0]: corr_best=item
    complement = corr_best[1] if corr_best else best_single
    return best_single, best_pair[1], best_pair[2], complement


def specialist_frame_from_ids(frame, task, ids):
    single, pair1, pair2, complement = ids
    base = frame[["stay_id","subject_id","target","validity"]].copy()
    p1=frame[f"pred_{single}"].to_numpy(float); pp1=frame[f"pred_{pair1}"].to_numpy(float); pp2=frame[f"pred_{pair2}"].to_numpy(float); pe=frame[f"pred_{complement}"].to_numpy(float)
    out=base.copy()
    if task in BINARY_TASKS:
        out["task_best_single"] = rank01(p1); out["task_best_single_prob"] = np.clip(p1,0,1)
        out["task_best_pair"] = (rank01(pp1)+rank01(pp2))/2.0; out["task_best_pair_prob"] = np.clip((pp1+pp2)/2.0,0,1)
        out["error_complement"] = rank01(pe); out["error_complement_prob"] = np.clip(pe,0,1)
    else:
        out["task_best_single"] = p1; out["task_best_single_prob"] = np.nan
        out["task_best_pair"] = (pp1+pp2)/2.0; out["task_best_pair_prob"] = np.nan
        out["error_complement"] = pe; out["error_complement_prob"] = np.nan
    return out


def fit_gate_params_on_frame(frame, task, z1_pred, specialist_ids):
    ids = specialist_ids
    sp = specialist_frame_from_ids(frame, task, ids)
    a3r = greedy_recipe_on_frame(frame, task, 5, "rank" if task in BINARY_TASKS else "raw")
    a3p = apply_recipe_to_frame(frame, a3r)
    z1p = np.asarray(z1_pred,float); spp=sp["task_best_single"].to_numpy(float); ecp=sp["error_complement"].to_numpy(float)
    dis=np.std(np.column_stack([rank01(a3p),rank01(z1p),rank01(spp)]),axis=1)
    threshold=float(np.nanmedian(dis))
    low=dis<=threshold; high=dis>threshold
    y=frame["target"].to_numpy(float); valid=frame["validity"].astype(bool).to_numpy() & np.isfinite(y)
    best=None; grid=np.arange(0.0,1.0001,Z2_WEIGHT_STEP)
    for la3 in grid:
        lz1=1.0-la3
        for hs in grid:
            he=1.0-hs
            lp=la3*(rank01(a3p) if task in BINARY_TASKS else a3p)+lz1*(rank01(z1p) if task in BINARY_TASKS else z1p)
            hp=hs*(rank01(spp) if task in BINARY_TASKS else spp)+he*(rank01(ecp) if task in BINARY_TASKS else ecp)
            pred=np.empty(len(y),float); pred[low]=lp[low]; pred[high]=hp[high]
            sc=score_prediction(task,y[valid & np.isfinite(pred)],pred[valid & np.isfinite(pred)])
            if best is None or sc>best[0]: best=(float(sc),{"threshold":threshold,"low_weights":[float(la3),float(lz1)],"high_weights":[float(hs),float(he)]})
    return best[1]


def apply_gate_to_frame(frame, task, z1_pred, specialist_ids, params):
    sp=specialist_frame_from_ids(frame,task,specialist_ids); a3r=greedy_recipe_on_frame(frame,task,5,"rank" if task in BINARY_TASKS else "raw"); a3p=apply_recipe_to_frame(frame,a3r); z1p=np.asarray(z1_pred,float); spp=sp["task_best_single"].to_numpy(float); ecp=sp["error_complement"].to_numpy(float)
    dis=np.std(np.column_stack([rank01(a3p),rank01(z1p),rank01(spp)]),axis=1); low=dis<=params["threshold"]; high=dis>params["threshold"]
    pred=np.full(len(frame),np.nan,float)
    if task in BINARY_TASKS:
        pred[low]=params["low_weights"][0]*rank01(a3p)[low]+params["low_weights"][1]*rank01(z1p)[low]
        pred[high]=params["high_weights"][0]*rank01(spp)[high]+params["high_weights"][1]*rank01(ecp)[high]
        prob=np.full(len(frame),np.nan,float)
        prob[low]=params["low_weights"][0]*np.clip(a3p,0,1)[low]+params["low_weights"][1]*np.clip(z1p,0,1)[low]
        prob[high]=params["high_weights"][0]*np.clip(sp["task_best_single_prob"].to_numpy(float),0,1)[high]+params["high_weights"][1]*np.clip(sp["error_complement_prob"].to_numpy(float),0,1)[high]
    else:
        pred[low]=params["low_weights"][0]*a3p[low]+params["low_weights"][1]*z1p[low]
        pred[high]=params["high_weights"][0]*spp[high]+params["high_weights"][1]*ecp[high]
        prob=None
    bad=~np.isfinite(pred)
    if bad.any():
        fb=np.where(np.isfinite(a3p),a3p,0.0); pred[bad]=fb[bad]
        if prob is not None: prob[bad]=np.where(np.isfinite(prob[bad]),prob[bad],0.0)
    out=frame[["stay_id","subject_id","target","validity"]].copy(); out["task"]=task; out["prediction"]=pred
    if prob is not None: out["probability_mean"]=np.clip(prob,0,1)
    return out


def top10_from_frame(frame, task):
    y=frame["target"].to_numpy(float)
    keep=frame["validity"].astype(bool).to_numpy() & np.isfinite(y)
    if not np.any(keep):
        return MODEL_IDS[:10]
    scores=[]
    for mid in MODEL_IDS:
        p=frame[f"pred_{mid}"].to_numpy(float)
        sc=safe_ap(y[keep],p[keep]) if task in BINARY_TASKS else -mean_absolute_error(y[keep],p[keep])
        scores.append((float(sc),mid))
    return [m for _,m in sorted(scores,reverse=True)[:10]]


def nested_previous_z2_weights(frame, task, z1_pred):
    a3=apply_recipe_to_frame(frame,greedy_recipe_on_frame(frame,task,5,"rank" if task in BINARY_TASKS else "raw"))
    a2=apply_recipe_to_frame(frame,greedy_recipe_on_frame(frame,task,3,"raw"))
    z1=np.asarray(z1_pred,float); y=frame["target"].to_numpy(float); valid=frame["validity"].astype(bool).to_numpy() & np.isfinite(y)
    best=None; grid=np.arange(0.0,1.0001,Z2_WEIGHT_STEP)
    for a3w in grid:
        for a2w in grid:
            z1w=1.0-a3w-a2w
            if z1w < -1e-9: continue
            pred=(np.column_stack([rank01(a3),rank01(a2),rank01(z1)])@np.array([a3w,a2w,z1w])) if task in BINARY_TASKS else (np.column_stack([a3,a2,z1])@np.array([a3w,a2w,z1w]))
            sc=score_prediction(task,y[valid],pred[valid])
            if best is None or sc>best[0]: best=(float(sc),np.array([a3w,a2w,z1w]))
    return best[1]


def nested_fold_source_pair(full_frame, tr_idx, ho_idx, task, z1_full_task):
    tr=full_frame.iloc[tr_idx].reset_index(drop=True); ho=full_frame.iloc[ho_idx].reset_index(drop=True)
    ids_val=full_frame["stay_id"].to_numpy()
    z1_map=dict(zip(z1_full_task["stay_id"].to_numpy(),z1_full_task["prediction"].to_numpy(float)))
    z1tr=np.asarray([z1_map[int(x)] for x in tr["stay_id"]],float); z1ho=np.asarray([z1_map[int(x)] for x in ho["stay_id"]],float)
    a3r=greedy_recipe_on_frame(tr,task,5,"rank" if task in BINARY_TASKS else "raw"); a2r=greedy_recipe_on_frame(tr,task,3,"raw")
    a3tr=apply_recipe_to_frame(tr,a3r); a3ho=apply_recipe_to_frame(ho,a3r); a2tr=apply_recipe_to_frame(tr,a2r); a2ho=apply_recipe_to_frame(ho,a2r)
    spec_ids=select_specialists_from_frame(tr,task)
    sptr=specialist_frame_from_ids(tr,task,spec_ids); spho=specialist_frame_from_ids(ho,task,spec_ids)
    top10=top10_from_frame(tr,task)
    def consensus(f):
        R=np.column_stack([rank01(f[f"pred_{m}"].to_numpy(float)) for m in MODEL_IDS]); raw=np.column_stack([f[f"pred_{m}"].to_numpy(float) for m in MODEL_IDS]); idx=[MODEL_IDS.index(m) for m in top10]
        return R.mean(axis=1),R[:,idx].mean(axis=1), (np.clip(raw.mean(axis=1),0,1) if task in BINARY_TASKS else np.full(len(f),np.nan)), (np.clip(raw[:,idx].mean(axis=1),0,1) if task in BINARY_TASKS else np.full(len(f),np.nan))
    alltr,toptr,allptr,topPtr=consensus(tr); allho,topho,allpho,topPho=consensus(ho)
    gate_params=fit_gate_params_on_frame(tr,task,z1tr,spec_ids)
    gtr=apply_gate_to_frame(tr,task,z1tr,spec_ids,gate_params); gho=apply_gate_to_frame(ho,task,z1ho,spec_ids,gate_params)
    prev_w=nested_previous_z2_weights(tr,task,z1tr)
    prev_tr=(np.column_stack([rank01(a3tr),rank01(a2tr),rank01(z1tr)])@prev_w if task in BINARY_TASKS else np.column_stack([a3tr,a2tr,z1tr])@prev_w)
    prev_ho=(np.column_stack([rank01(a3ho),rank01(a2ho),rank01(z1ho)])@prev_w if task in BINARY_TASKS else np.column_stack([a3ho,a2ho,z1ho])@prev_w)
    def pack(f,a3,a2,z1,sp,allr,topr,allp,topp,gate,prev):
        S=np.column_stack([a3,a2,z1,sp["task_best_single"].to_numpy(float),sp["task_best_pair"].to_numpy(float),sp["error_complement"].to_numpy(float),allr,topr,gate["prediction"].to_numpy(float),prev])
        P=np.column_stack([a3,a2,z1,sp["task_best_single_prob"].to_numpy(float),sp["task_best_pair_prob"].to_numpy(float),sp["error_complement_prob"].to_numpy(float),allp,topp,gate.get("probability_mean",gate["prediction"]).to_numpy(float),prev])
        a3c=np.where(np.isfinite(S[:,0]),S[:,0],0.0); ap=np.where(np.isfinite(P[:,0]),P[:,0],0.0)
        for j in range(S.shape[1]): S[~np.isfinite(S[:,j]),j]=a3c[~np.isfinite(S[:,j])]
        for j in range(P.shape[1]): P[~np.isfinite(P[:,j]),j]=ap[~np.isfinite(P[:,j])]
        b=f[["stay_id","subject_id","target","validity"]].copy(); b.attrs["task"]=task
        return b,S,P,SOURCE_NAMES.copy()
    return pack(tr,a3tr,a2tr,z1tr,sptr,alltr,toptr,allptr,topPtr,gtr,prev_tr), pack(ho,a3ho,a2ho,z1ho,spho,allho,topho,allpho,topPho,gho,prev_ho)
def feasible_grid_weights(n, rng):
    min_units = int(round(Z2_WEIGHT_MIN / Z2_WEIGHT_STEP))
    max_units = int(round(Z2_WEIGHT_MAX / Z2_WEIGHT_STEP))
    total_units = int(round(1.0 / Z2_WEIGHT_STEP))

    if n * min_units > total_units or n * max_units < total_units:
        raise ValueError(f"No feasible simplex for n={n}")

    rem = total_units - n * min_units
    extra = np.zeros(n, dtype=int)
    while rem > 0:
        eligible = np.flatnonzero(extra < (max_units - min_units))
        j = int(rng.choice(eligible))
        extra[j] += 1
        rem -= 1

    w = (min_units + extra).astype(float) * Z2_WEIGHT_STEP
    rng.shuffle(w)
    if not np.isclose(w.sum(), 1.0):
        raise AssertionError("Grid weight construction failed.")
    return w


def equal_weights(n):
    w = np.full(n, 1.0 / n, dtype=float)
    if np.any(w < Z2_WEIGHT_MIN - 1e-12) or np.any(w > Z2_WEIGHT_MAX + 1e-12):
        raise ValueError(f"Equal weights violate bounds for n={n}")
    return w


def score_prediction(task, y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    keep = np.isfinite(y) & np.isfinite(p)
    if keep.sum() == 0:
        return float("-inf")
    if task in BINARY_TASKS:
        return safe_ap(y[keep], p[keep])
    return -float(mean_absolute_error(y[keep], p[keep]))


def coordinate_optimize(X, y, valid, task, start, iterations=5):
    w = np.array(start, dtype=float)
    if not np.isclose(w.sum(), 1.0):
        w = w / w.sum()
    if np.any(w < Z2_WEIGHT_MIN - 1e-10) or np.any(w > Z2_WEIGHT_MAX + 1e-10):
        raise ValueError("Coordinate optimizer received infeasible start.")
    valid = np.asarray(valid, dtype=bool)
    if int(valid.sum()) == 0:
        return w, float("-inf")

    base_score = score_prediction(task, y[valid], X[valid] @ w)
    n = len(w)

    for _ in range(iterations):
        changed = False
        best_global = (base_score, w.copy())

        for donor in range(n):
            if w[donor] <= Z2_WEIGHT_MIN + 1e-12:
                continue
            for receiver in range(n):
                if receiver == donor:
                    continue
                if w[receiver] >= Z2_WEIGHT_MAX - 1e-12:
                    continue

                tw = w.copy()
                tw[donor] -= Z2_WEIGHT_STEP
                tw[receiver] += Z2_WEIGHT_STEP

                if np.any(tw < Z2_WEIGHT_MIN - 1e-10) or np.any(tw > Z2_WEIGHT_MAX + 1e-10):
                    continue
                if not np.isclose(tw.sum(), 1.0):
                    continue

                sc = score_prediction(task, y[valid], X[valid] @ tw)
                if sc > best_global[0] + 1e-12:
                    best_global = (sc, tw)

        if best_global[0] > base_score + 1e-12:
            base_score = float(best_global[0])
            w = best_global[1]
            changed = True

        if not changed:
            break

    return w, float(base_score)


def crossfit_trials(task, source_data, n_trials, seed, cv_folds=5, nested_folds=None):
    base, S, P, names = source_data
    y = base["target"].to_numpy(float)
    rng = np.random.default_rng(seed)
    starts = [equal_weights(S.shape[1])]
    while len(starts) < n_trials:
        starts.append(feasible_grid_weights(S.shape[1], rng))

    rows = []
    best = None
    if nested_folds is None:
        groups = base["subject_id"].to_numpy()
        gkf = GroupKFold(n_splits=cv_folds)
        nested_folds = []
        for tr, ho in gkf.split(S, y, groups=groups):
            nested_folds.append((
                (base.iloc[tr].reset_index(drop=True), S[tr], P[tr], names),
                (base.iloc[ho].reset_index(drop=True), S[ho], P[ho], names),
            ))

    for trial_no, start in enumerate(starts, 1):
        oof = np.full(len(base), np.nan, dtype=float)
        fold_weights = []; fold_scores=[]
        for fold_idx, (train_sd, hold_sd) in enumerate(nested_folds):
            trb,trS,trP,trnames=train_sd
            hob,hoS,hoP,honames=hold_sd
            trv=trb["validity"].astype(bool).to_numpy() & np.isfinite(trb["target"].to_numpy(float))
            hov=hob["validity"].astype(bool).to_numpy() & np.isfinite(hob["target"].to_numpy(float))
            if not np.any(trv):
                fold_weights.append(start.copy()); continue
            w,_=coordinate_optimize(trS,trb["target"].to_numpy(float),trv,task,start,iterations=5)
            fold_weights.append(w)
            pred_ho=hoS @ w
            oof[np.asarray(base["stay_id"].isin(hob["stay_id"]).to_numpy())]=pred_ho
            keep_ho=hov & np.isfinite(pred_ho)
            if np.any(keep_ho): fold_scores.append(score_prediction(task,hob["target"].to_numpy(float)[keep_ho],pred_ho[keep_ho]))
        keep=np.isfinite(y) & base["validity"].astype(bool).to_numpy() & np.isfinite(oof)
        oof_score=score_prediction(task,y[keep],oof[keep])
        meanw=np.mean(np.vstack(fold_weights),axis=0)
        row={"task":task,"trial":trial_no,"oof_score":float(oof_score),"metric":"AUPRC" if task in BINARY_TASKS else "negative_MAE","fold_score_mean":float(np.mean(fold_scores)) if fold_scores else float("nan"),"fold_score_std":float(np.std(fold_scores)) if fold_scores else float("nan")}
        row.update({names[i]:float(meanw[i]) for i in range(len(names))}); rows.append(row)
        if best is None or oof_score>best["oof_score"]:
            best={"oof_score":float(oof_score),"trial":int(trial_no),"mean_fold_weights":meanw.copy(),"oof":oof.copy()}
        print(f"[Z2-ULT] {task} trial={trial_no:02d}/{n_trials} OOF={oof_score:.6f}",flush=True)
    return best,pd.DataFrame(rows)


def build_nested_oof_folds(candidates, task, cv_folds, z1v):
    full=aligned_val_frame(candidates,task)
    y=full["target"].to_numpy(float); groups=full["subject_id"].to_numpy()
    gkf=GroupKFold(n_splits=cv_folds)
    folds=[]
    for fold,(tr,ho) in enumerate(gkf.split(full,y,groups=groups)):
        train_sd, hold_sd = nested_fold_source_pair(full,tr,ho,task,z1v[z1v["task"]==task].sort_values("stay_id").reset_index(drop=True))
        folds.append((train_sd,hold_sd))
        print(f"[NESTED-SOURCES] {task} fold={fold+1}/{cv_folds} train={len(tr)} holdout={len(ho)}",flush=True)
    return folds


def nested_previous_z2_oof(task, nested_folds):
    idx=SOURCE_NAMES.index("previous_Z2"); preds=[]; ys=[]; valids=[]
    for _,hold_sd in nested_folds:
        hb,hS,hP,names=hold_sd
        preds.append(hS[:,idx]); ys.append(hb["target"].to_numpy(float)); valids.append(hb["validity"].astype(bool).to_numpy())
    y=np.concatenate(ys); p=np.concatenate(preds); v=np.concatenate(valids)
    keep=v & np.isfinite(y) & np.isfinite(p)
    return score_prediction(task,y[keep],p[keep])


def bootstrap_average_weights(task, source_data, initial_weights, reps=100, seed=20260923):
    base, S, P, names = source_data
    y = base["target"].to_numpy(float)
    valid = base["validity"].astype(bool).to_numpy() & np.isfinite(y) & np.all(np.isfinite(S), axis=1)
    subjects = base["subject_id"].to_numpy()
    unique_subjects = np.unique(subjects)
    groups = {s: np.flatnonzero(subjects == s) for s in unique_subjects}

    rng = np.random.default_rng(seed)
    weights = []
    n = len(unique_subjects)

    for b in range(reps):
        sampled = rng.choice(unique_subjects, size=n, replace=True)
        idx = np.concatenate([groups[s] for s in sampled])
        bv = valid[idx]

        if not np.any(bv):
            continue

        w, _ = coordinate_optimize(
            S[idx],
            y[idx],
            bv,
            task,
            initial_weights,
            iterations=5,
        )
        weights.append(w)

    if not weights:
        return np.asarray(initial_weights, float), pd.DataFrame()

    W = np.vstack(weights)
    meanw = W.mean(axis=0)

    # Averaging feasible simplex points preserves the bounds and sum-to-one.
    meanw = np.clip(meanw, Z2_WEIGHT_MIN, Z2_WEIGHT_MAX)
    # Numerical correction while preserving feasibility; because every bootstrap
    # vector is feasible, clipping should only affect floating-point noise.
    meanw /= meanw.sum()

    return meanw, pd.DataFrame(W, columns=names)


def apply_final_sources(source_data, weights, split_name, source_names=SOURCE_NAMES):
    base, S, P, names = source_data
    w = np.asarray(weights, float)
    w = w / w.sum()

    pred = S @ w
    prob = P @ w

    out = base.copy()
    out["task"] = out.attrs.get("task", "")
    out["prediction"] = pred
    out["probability_mean"] = np.clip(prob, 0.0, 1.0)
    out["split"] = split_name
    return out


def deterioration_from_frame(frame):
    probs = []
    targets = []
    valid = None
    for task in BINARY_TASKS[:3]:
        z = frame[frame["task"] == task].sort_values("stay_id").reset_index(drop=True)
        if "probability_mean" not in z.columns:
            raise ValueError(f"Missing probability_mean for deterioration task {task}")
        if valid is None:
            valid = z["validity"].astype(bool).to_numpy()
        else:
            valid &= z["validity"].astype(bool).to_numpy()
        targets.append(z["target"].to_numpy(float))
        probs.append(np.clip(z["probability_mean"].to_numpy(float), 0, 1))

    y = 1.0 - np.prod(1.0 - np.vstack(targets), axis=0)
    p = 1.0 - np.prod(1.0 - np.vstack(probs), axis=0)
    keep = valid & np.isfinite(y) & np.isfinite(p)
    return {
        "n": int(keep.sum()),
        "auprc": safe_ap(y[keep], p[keep]),
        "auroc": safe_auc(y[keep], p[keep]),
    }


def bootstrap_pair(left, right, task, reps, seed):
    a = left[left["task"] == task].sort_values("stay_id").reset_index(drop=True)
    b = right[right["task"] == task].sort_values("stay_id").reset_index(drop=True)

    if not np.array_equal(a["stay_id"].to_numpy(), b["stay_id"].to_numpy()):
        raise ValueError(f"Bootstrap alignment failure {task}")
    if not np.array_equal(a["subject_id"].to_numpy(), b["subject_id"].to_numpy()):
        raise ValueError(f"Bootstrap subject alignment failure {task}")

    keep = a["validity"].astype(bool).to_numpy() & b["validity"].astype(bool).to_numpy()
    y = a.loc[keep, "target"].to_numpy(float)
    pa = a.loc[keep, "prediction"].to_numpy(float)
    pb = b.loc[keep, "prediction"].to_numpy(float)
    subjects = a.loc[keep, "subject_id"].to_numpy()

    us = np.unique(subjects)
    groups = {s: np.flatnonzero(subjects == s) for s in us}
    rng = np.random.default_rng(seed)
    diffs = []

    for _ in range(reps):
        sampled = rng.choice(us, size=len(us), replace=True)
        idx = np.concatenate([groups[s] for s in sampled])

        if task in BINARY_TASKS:
            yy = y[idx].astype(int)
            if len(np.unique(yy)) < 2:
                continue
            diffs.append(safe_ap(yy, pa[idx]) - safe_ap(yy, pb[idx]))
        else:
            diffs.append(mean_absolute_error(y[idx], pa[idx]) - mean_absolute_error(y[idx], pb[idx]))

    arr = np.asarray(diffs, float)
    if len(arr) == 0:
        return {"task": task, "n": int(len(y)), "point_delta_left_minus_right": float("nan"),
                "ci95_low": float("nan"), "ci95_high": float("nan"),
                "prob_left_better": float("nan")}

    point = (
        safe_ap(y.astype(int), pa) - safe_ap(y.astype(int), pb)
        if task in BINARY_TASKS
        else mean_absolute_error(y, pa) - mean_absolute_error(y, pb)
    )
    prob = float(np.mean(arr > 0)) if task in BINARY_TASKS else float(np.mean(arr < 0))
    return {
        "task": task,
        "n": int(len(y)),
        "subjects": int(len(us)),
        "point_delta_left_minus_right": float(point),
        "ci95_low": float(np.quantile(arr, 0.025)),
        "ci95_high": float(np.quantile(arr, 0.975)),
        "prob_left_better": prob,
    }


def fit_temperature(y, p):
    # Diagnostic only; ranking metrics are invariant to monotone temperature
    # changes, so calibration never affects selection.
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    keep = np.isfinite(y) & np.isfinite(p)
    y = y[keep].astype(int)
    p = np.clip(p[keep], 1e-6, 1 - 1e-6)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return 1.0

    z = np.log(p / (1 - p))
    try:
        from scipy.optimize import minimize_scalar
        from sklearn.metrics import log_loss

        def obj(logT):
            T = float(np.exp(logT))
            q = 1.0 / (1.0 + np.exp(-z / T))
            return log_loss(y, q, labels=[0, 1])

        res = minimize_scalar(obj, bounds=(-3, 3), method="bounded")
        return float(np.exp(res.x))
    except Exception:
        return 1.0


def apply_temperature(p, T):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p))
    return 1.0 / (1.0 + np.exp(-z / float(T)))


def ece(y, p, bins=10):
    y = np.asarray(y).astype(int)
    p = np.asarray(p, float)
    if len(y) == 0:
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    val = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if not m.any():
            continue
        val += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(val)


def calibration_report(frame):
    rows = []
    for task in BINARY_TASKS:
        z = frame[frame["task"] == task]
        valid = z["validity"].astype(bool).to_numpy()
        y = z["target"].to_numpy(float)
        p = z["probability_mean"].to_numpy(float)
        keep = valid & np.isfinite(y) & np.isfinite(p)
        T = fit_temperature(y[keep], p[keep])
        q = apply_temperature(p[keep], T)
        from sklearn.metrics import brier_score_loss

        rows.append({
            "task": task,
            "n": int(keep.sum()),
            "temperature": float(T),
            "brier_before": float(brier_score_loss(y[keep].astype(int), p[keep])) if keep.any() else float("nan"),
            "brier_after": float(brier_score_loss(y[keep].astype(int), q)) if keep.any() else float("nan"),
            "ece_before": ece(y[keep], p[keep]),
            "ece_after": ece(y[keep], q),
        })
    return pd.DataFrame(rows)


def frame_from_weighted_sources(source_data_by_task, weights_by_task, split_name):
    parts = []
    for task, source_data in source_data_by_task.items():
        base, S, P, names = source_data
        w = np.asarray(weights_by_task[task], float)
        w = w / w.sum()
        out = base.copy()
        out["task"] = task
        out["prediction"] = S @ w
        if task in BINARY_TASKS:
            out["probability_mean"] = np.clip(P @ w, 0.0, 1.0)
        out["split"] = split_name
        parts.append(out)
    return pd.concat(parts, ignore_index=True)


def top10_members(candidates, task):
    frame = align(candidates, MODEL_IDS, "val", task)
    keep = valid_mask(frame, MODEL_IDS)
    y = frame["target"].to_numpy(float)
    scores = []
    for mid in MODEL_IDS:
        p = frame[f"pred_{mid}"].to_numpy(float)
        sc = safe_ap(y[keep], p[keep]) if task in BINARY_TASKS else -mean_absolute_error(y[keep], p[keep])
        scores.append((float(sc), mid))
    return [m for _, m in sorted(scores, reverse=True)[:10]]


def build_all_sources(candidates, specialist_single, specialist_pair, complements, previous_z2, split):
    top10 = {task: top10_members(candidates, task) for task in TASKS}

    gate_params = {}
    for task in TASKS:
        # Reuse deterministic specialist definitions and cached Z1 predictions.
        gate_params[task] = tune_gate_fast(
            candidates, task, specialist_single, specialist_pair, complements
        )
        print(f"[GATE] {task}: {gate_params[task]}", flush=True)

    source_data = {}
    for task in TASKS:
        z1_for_split = load_cached_z1(candidates, split, task)
        sd = build_source_data(
            candidates, split, task,
            specialist_single, specialist_pair, complements,
            top10, gate_params, previous_z2,
            z1_for_split,
        )
        source_data[task] = sd
        # Attach task for downstream construction.
        sd[0].attrs["task"] = task
        print(f"[SOURCES] {split} {task}: n={len(sd[0])} sources={len(sd[3])}", flush=True)

    return source_data, top10, gate_params


def tune_gate_fast(candidates, task, specialist_single, specialist_pair, complements):
    # Uses only validation. This intentionally avoids test-informed gate fitting.
    a3 = apply_recipe(
        candidates,
        greedy_recipe(candidates, task, 5, "rank" if task in BINARY_TASKS else "raw"),
        "val",
    )
    # Build Z1 from existing output if present; otherwise recalc deterministically.
    z1_out = load_cached_z1(candidates, "val", task)
    sp = build_specialist_frame(candidates, "val", task, specialist_single, specialist_pair, complements)

    a3p = a3["prediction"].to_numpy(float)
    z1p = z1_out["prediction"].to_numpy(float)
    spp = sp["task_best_single"].to_numpy(float)
    ecp = sp["error_complement"].to_numpy(float)

    dis = np.std(np.column_stack([rank01(a3p), rank01(z1p), rank01(spp)]), axis=1)
    threshold = float(np.nanmedian(dis))
    low = dis <= threshold
    high = dis > threshold
    y = a3["target"].to_numpy(float)
    valid = a3["validity"].astype(bool).to_numpy() & np.isfinite(y)

    best = None
    grid = np.arange(0.0, 1.0001, Z2_WEIGHT_STEP)
    for la3 in grid:
        lz1 = 1.0 - la3
        for hs in grid:
            he = 1.0 - hs
            if task in BINARY_TASKS:
                lp = la3 * rank01(a3p) + lz1 * rank01(z1p)
                hp = hs * rank01(spp) + he * rank01(ecp)
            else:
                lp = la3 * a3p + lz1 * z1p
                hp = hs * spp + he * ecp

            pred = np.empty(len(y), float)
            pred[low] = lp[low]
            pred[high] = hp[high]
            keep = valid & np.isfinite(pred)
            sc = score_prediction(task, y[keep], pred[keep])
            if best is None or sc > best[0]:
                best = (sc, {
                    "threshold": threshold,
                    "low_weights": [float(la3), float(lz1)],
                    "high_weights": [float(hs), float(he)],
                })
    assert best is not None
    return best[1]


def apply_gate_from_components(candidates, split, task, params, specialist_single, specialist_pair, complements, z1_df):
    a3 = apply_recipe(
        candidates,
        greedy_recipe(candidates, task, 5, "rank" if task in BINARY_TASKS else "raw"),
        split,
    )
    z1 = z1_df[z1_df["task"] == task].sort_values("stay_id").reset_index(drop=True)
    sp = build_specialist_frame(candidates, split, task, specialist_single, specialist_pair, complements)

    a3p = a3["prediction"].to_numpy(float)
    z1p = z1["prediction"].to_numpy(float)
    spp = sp["task_best_single"].to_numpy(float)
    ecp = sp["error_complement"].to_numpy(float)

    dis = np.std(np.column_stack([rank01(a3p), rank01(z1p), rank01(spp)]), axis=1)
    low = dis <= params["threshold"]
    high = dis > params["threshold"]

    pred = np.full(len(a3p), np.nan, dtype=float)

    if task in BINARY_TASKS:
        lowp = params["low_weights"][0] * rank01(a3p) + params["low_weights"][1] * rank01(z1p)
        highp = params["high_weights"][0] * rank01(spp) + params["high_weights"][1] * rank01(ecp)
        pred[low] = lowp[low]
        pred[high] = highp[high]

        a3prob = np.clip(a3.get("probability_mean", a3["prediction"]).to_numpy(float), 0, 1)
        z1prob = np.clip(z1.get("probability_mean", z1["prediction"]).to_numpy(float), 0, 1)
        spprob = np.clip(sp["task_best_single_prob"].to_numpy(float), 0, 1)
        ecprob = np.clip(sp["error_complement_prob"].to_numpy(float), 0, 1)
        prob = np.full(len(a3p), np.nan, dtype=float)
        prob[low] = params["low_weights"][0] * a3prob[low] + params["low_weights"][1] * z1prob[low]
        prob[high] = params["high_weights"][0] * spprob[high] + params["high_weights"][1] * ecprob[high]
    else:
        lowp = params["low_weights"][0] * a3p + params["low_weights"][1] * z1p
        highp = params["high_weights"][0] * spp + params["high_weights"][1] * ecp
        pred[low] = lowp[low]
        pred[high] = highp[high]
        prob = None

    bad = ~np.isfinite(pred)
    if bad.any():
        fb = np.where(np.isfinite(a3p), a3p, z1p)
        fb = np.where(np.isfinite(fb), fb, 0.0)
        pred[bad] = fb[bad]
        if prob is not None:
            prob[bad] = np.where(np.isfinite(prob[bad]), prob[bad], 0.0)

    out = a3[["stay_id", "subject_id", "target", "validity"]].copy()
    out["task"] = task
    out["prediction"] = pred
    if prob is not None:
        out["probability_mean"] = np.clip(prob, 0, 1)
    return out


_Z1_CACHE = None


def load_cached_z1(candidates, split, task):
    global _Z1_CACHE
    if _Z1_CACHE is None:
        path = SEARCH_ROOT / "model_z_family" / "Z1_tuned_validation_predictions.csv"
        test_path = SEARCH_ROOT / "model_z_family" / "Z1_tuned_test_predictions.csv"
        if not path.exists() or not test_path.exists():
            raise FileNotFoundError(
                "Frozen Z1_tuned artifacts are required; refusing to recompute Z1 inside Z2 Ultimate. "
                f"Expected {path} and {test_path}."
            )
        _Z1_CACHE = {"val": pd.read_csv(path), "test": pd.read_csv(test_path)}
    if task not in set(_Z1_CACHE[split]["task"]):
        raise ValueError(f"Z1_tuned cache has no task={task} for split={split}")
    return _Z1_CACHE[split][_Z1_CACHE[split]["task"] == task].sort_values("stay_id").reset_index(drop=True)


def prepare_everything(candidates):
    recipes, a3v, a3t, a2v, a2t = build_a3_a2(candidates)
    print(f"A3 validation mean AUPRC={metrics(a3v)['mean_binary_auprc']:.6f}", flush=True)
    print(f"A2 validation mean AUPRC={metrics(a2v)['mean_binary_auprc']:.6f}", flush=True)

    # IMPORTANT: Z1_tuned is an already completed upstream experiment.
    # Reuse its exact frozen prediction artifacts rather than re-running Z1
    # tuning inside Z2 Ultimate. This prevents source drift and saves compute.
    z1root = SEARCH_ROOT / "model_z_family"
    z1v_path = z1root / "Z1_tuned_validation_predictions.csv"
    z1t_path = z1root / "Z1_tuned_test_predictions.csv"
    z1cv_path = z1root / "Z1_tuned_cv.csv"
    z1rec_path = z1root / "Z1_tuned_recipes.json"
    for path in (z1v_path, z1t_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Frozen Z1_tuned artifact missing: {path}. "
                "Run the canonical Z family once before Z2 Ultimate."
            )
    z1v = pd.read_csv(z1v_path)
    z1t = pd.read_csv(z1t_path)
    z1cv = pd.read_csv(z1cv_path) if z1cv_path.exists() else pd.DataFrame()
    z1rec = json.loads(z1rec_path.read_text()) if z1rec_path.exists() else {}
    print(
        f"Frozen Z1_tuned validation mean AUPRC={metrics(z1v)['mean_binary_auprc']:.6f}",
        flush=True,
    )

    prev_v, prev_t = load_actual_z2_tuned()
    print(f"Actual Z2_tuned validation mean AUPRC={metrics(prev_v)['mean_binary_auprc']:.6f}", flush=True)

    specialists_single, specialists_pair, specialists_comp = specialist_single_pair(candidates)
    return (
        recipes,
        a3v, a3t,
        a2v, a2t,
        z1rec, z1v, z1t, z1cv,
        prev_v, prev_t,
        specialists_single, specialists_pair, specialists_comp,
    )


def crossfit_baseline_z2(task, source_data, seed, cv_folds=5):
    # Previous Z2 weights [A3,A2,Z1] in the same source matrix.
    base, S, P, names = source_data
    keep = base["validity"].astype(bool).to_numpy() & np.isfinite(base["target"].to_numpy(float))
    groups = base["subject_id"].to_numpy()
    y = base["target"].to_numpy(float)
    idx = [names.index(x) for x in ("A3_rank_max5", "A2_probability_mean_max3", "Z1_tuned")]

    pred = np.full(len(base), np.nan)
    w = np.array([0.4, 0.2, 0.4])

    gkf = GroupKFold(n_splits=cv_folds)
    for tr, ho in gkf.split(S, y, groups=groups):
        pred[ho] = S[ho][:, idx] @ w

    return score_prediction(task, y[keep & np.isfinite(pred)], pred[keep & np.isfinite(pred)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv-folds", type=int, default=CV_FOLDS_DEFAULT)
    parser.add_argument("--z2-trials", type=int, default=N_TRIALS_DEFAULT)
    parser.add_argument("--bootstrap-weight-reps", type=int, default=BOOTSTRAP_WEIGHT_DEFAULT)
    parser.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS_DEFAULT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 110)
    print("FINAL Z2 ULTIMATE — ADVANCED FROZEN-PREDICTION SUPER-ENSEMBLE")
    print("=" * 110)
    print(
        f"trials={args.z2_trials} | grouped_CV={args.cv_folds} | "
        f"weight_bootstrap={args.bootstrap_weight_reps} | "
        f"stat_bootstrap={args.bootstrap_reps}"
    )
    print("TEST IS NOT USED FOR Z2 ULTIMATE SELECTION", flush=True)

    candidates = discover_candidates(SEARCH_ROOT)

    (
        recipes,
        a3v, a3t,
        a2v, a2t,
        z1rec, z1v, z1t, z1cv,
        prev_v, prev_t,
        specialists_single, specialists_pair, specialists_comp,
    ) = prepare_everything(candidates)

    # Fixed top-10 members selected on validation only.
    top10_by_task = {task: top10_members(candidates, task) for task in TASKS}

    # Gate parameters selected on validation only.
    gate_params = {}
    for task in TASKS:
        gate_params[task] = tune_gate_fast(
            candidates, task, specialists_single, specialists_pair, specialists_comp
        )

    # Build the complete source matrices for validation.
    # The test source matrices are deliberately NOT built until the validation
    # decision is frozen, preserving the holdout protocol.
    source_val = {}

    # Cache Z1 once, avoiding repeated recomputation inside source construction.
    global _Z1_CACHE
    _Z1_CACHE = {"val": z1v, "test": z1t}

    for task in TASKS:
        source_val[task] = build_source_data(
            candidates, "val", task,
            specialists_single, specialists_pair, specialists_comp,
            top10_by_task, gate_params, prev_v, z1v
        )
        source_val[task][0].attrs["task"] = task

    # Fixed advanced Z2 comparator: equal-weight blend of all 10 frozen sources.
    fixed_equal_rows = []
    fixed_equal_parts = []
    equal_w = equal_weights(len(SOURCE_NAMES))
    for task in TASKS:
        base, S, P, names = source_val[task]
        fixed_equal = base.copy()
        fixed_equal["task"] = task
        fixed_equal["prediction"] = S @ equal_w
        if task in BINARY_TASKS:
            fixed_equal["probability_mean"] = np.clip(P @ equal_w, 0.0, 1.0)
        fixed_equal["split"] = "val"
        fixed_equal_parts.append(fixed_equal)
        fixed_equal_rows.append({
            "task": task,
            "fixed_equal10_validation_score": score_prediction(
                task,
                base["target"].to_numpy(float)[base["validity"].astype(bool).to_numpy()],
                fixed_equal["prediction"].to_numpy(float)[base["validity"].astype(bool).to_numpy()],
            ),
        })
    fixed_equal_v = pd.concat(fixed_equal_parts, ignore_index=True)
    fixed_equal_v.to_csv(outdir / "Z2_ultimate_fixed_equal10_validation_predictions.csv", index=False)
    pd.DataFrame(fixed_equal_rows).to_csv(outdir / "Z2_ultimate_fixed_equal10_scores.csv", index=False)
    print(
        f"Z2 Ultimate fixed equal-10 validation mean AUPRC="
        f"{metrics(fixed_equal_v)['mean_binary_auprc']:.6f}",
        flush=True,
    )

    # Nested OOF source construction prevents fold-level source engineering leakage.
    # Nested OOF source construction: specialists, top-10, gate, and previous Z2
    # are refit inside each training fold. This prevents those validation-fitted
    # source definitions from contaminating their own OOF evaluation.
    # are fitted inside each training fold. Z1 remains the cached existing tuned expert.
    nested_folds_by_task = {}
    baseline_oof_rows = []
    for i, task in enumerate(TASKS):
        nested_folds = build_nested_oof_folds(candidates, task, args.cv_folds, z1v)
        nested_folds_by_task[task] = nested_folds
        nested_score = nested_previous_z2_oof(task, nested_folds)
        baseline_oof_rows.append({"task":task,"previous_z2_nested_oof_score":nested_score,"actual_z2_tuned_validation_mean_binary_auprc":float(metrics(prev_v)["mean_binary_auprc"]),"metric":"AUPRC" if task in BINARY_TASKS else "negative_MAE"})
        print(f"[BASELINE-Z2-NESTED-OOF] {task}: {nested_score:.6f}",flush=True)
    pd.DataFrame(baseline_oof_rows).to_csv(outdir / "previous_z2_oof_baseline.csv",index=False)

    # 30-trial ultimate Z2 tuning.
    best_by_task = {}
    trial_tables = []
    boot_weight_tables = []

    for i, task in enumerate(TASKS):
        best, table = crossfit_trials(
            task,
            source_val[task],
            args.z2_trials,
            args.seed + 1000 + i,
            cv_folds=args.cv_folds,
            nested_folds=nested_folds_by_task[task],
        )
        best_by_task[task] = best
        trial_tables.append(table)

        boot_w, boot_table = bootstrap_average_weights(
            task,
            source_val[task],
            best["mean_fold_weights"],
            reps=args.bootstrap_weight_reps,
            seed=args.seed + 5000 + i,
        )
        best["bootstrap_weights"] = boot_w
        boot_table.insert(0, "task", task)
        boot_table.insert(0, "bootstrap_rep", np.arange(1, len(boot_table) + 1))
        boot_weight_tables.append(boot_table)

        pd.DataFrame(
            [best],
        ).drop(columns=["oof"], errors="ignore").to_json(
            outdir / f"best_{task}.json",
            indent=2,
            default_handler=str,
        )

        print(
            f"[Z2-ULT-FINAL] {task}: "
            f"OOF={best['oof_score']:.6f} "
            f"selected_trial={best['trial']} "
            f"bootstrap_weights={dict(zip(SOURCE_NAMES, np.round(boot_w, 4)))}",
            flush=True,
        )

    trial_all = pd.concat(trial_tables, ignore_index=True)
    trial_all.to_csv(outdir / "Z2_ultimate_all_30_trials.csv", index=False)

    if boot_weight_tables:
        pd.concat(boot_weight_tables, ignore_index=True).to_csv(
            outdir / "Z2_ultimate_bootstrap_weights.csv",
            index=False,
        )

    # OOF primary score from selected task-wise models.
    ultimate_oof = np.mean([
        best_by_task[t]["oof_score"]
        for t in BINARY_TASKS
    ])
    previous_oof_binary = np.mean([
        r["previous_z2_nested_oof_score"]
        for r in baseline_oof_rows if r["task"] in BINARY_TASKS
    ])
    actual_z2_validation = float(metrics(prev_v)["mean_binary_auprc"])
    oof_delta = ultimate_oof - previous_oof_binary
    delta_vs_actual_z2 = ultimate_oof - actual_z2_validation

    # Conservative final rule: the nested OOF result must also clear the
    # already-selected historical Z2_tuned validation result by the predeclared
    # margin. This avoids replacing the real primary model based only on an
    # optimistic or non-comparable nested reference.
    becomes_primary = bool(delta_vs_actual_z2 >= MIN_OOF_GAIN)

    decision = {
        "ultimate_z2_oof_mean_binary_auprc": float(ultimate_oof),
        "previous_z2_nested_oof_mean_binary_auprc": float(previous_oof_binary),
        "actual_z2_tuned_validation_mean_binary_auprc": actual_z2_validation,
        "oof_delta_vs_nested_previous_z2": float(oof_delta),
        "oof_delta_vs_actual_z2_tuned": float(delta_vs_actual_z2),
        "minimum_gain_required": MIN_OOF_GAIN,
        "ultimate_z2_selected_as_primary": becomes_primary,
        "selection_metric": "grouped_5fold_OOF_mean_binary_AUPRC",
        "test_used_for_selection": False,
        "stop_after_z2_ultimate": True,
    }
    (outdir / "FINAL_Z2_ULTIMATE_DECISION.json").write_text(json.dumps(decision, indent=2))

    print("\n" + "=" * 110)
    print(f"Z2 ULTIMATE OOF mean binary AUPRC = {ultimate_oof:.6f}")
    print(f"PREVIOUS Z2 NESTED OOF mean binary AUPRC = {previous_oof_binary:.6f}")
    print(f"ACTUAL Z2_tuned validation mean AUPRC  = {actual_z2_validation:.6f}")
    print(f"OOF delta vs nested Z2              = {oof_delta:+.6f}")
    print(f"OOF delta vs actual Z2_tuned        = {delta_vs_actual_z2:+.6f}")
    print(f"ULTIMATE Z2 PRIMARY?                = {becomes_primary}")
    print("=" * 110, flush=True)

    # Build final validation frame using bootstrap-averaged weights.
    ultimate_v = frame_from_weighted_sources(
        source_val,
        {task: best_by_task[task]["bootstrap_weights"] for task in TASKS},
        "val",
    )
    ultimate_v.to_csv(outdir / "Z2_ultimate_validation_predictions.csv", index=False)

    # Primary selection is completely frozen before any test evaluation.
    selected_name = "Z2_ultimate" if becomes_primary else "Previous_Z2"
    selected_v = ultimate_v if becomes_primary else prev_v
    selected_metrics_val = metrics(selected_v)

    selected_metrics_test = None
    ultimate_test_metrics = None

    if becomes_primary:
        print("[TEST] validation selection frozen; evaluating Z2 Ultimate on test once", flush=True)

        source_test = {}
        for task in TASKS:
            source_test[task] = build_source_data(
                candidates, "test", task,
                specialists_single, specialists_pair, specialists_comp,
                top10_by_task, gate_params, prev_t, z1t
            )
            source_test[task][0].attrs["task"] = task

        ultimate_t = frame_from_weighted_sources(
            source_test,
            {task: best_by_task[task]["bootstrap_weights"] for task in TASKS},
            "test",
        )
        ultimate_t.to_csv(outdir / "Z2_ultimate_test_predictions.csv", index=False)
        selected_t = ultimate_t
        ultimate_test_metrics = metrics(ultimate_t)
        selected_metrics_test = ultimate_test_metrics

        # Reference Model32 test frame.
        ref_parts = []
        for task in TASKS:
            z = task_frame(candidates, REFERENCE_MODEL, "test", task)
            ref_parts.append(
                z[["stay_id", "subject_id", "target", "validity"]]
                .assign(task=task, prediction=z["prediction"].to_numpy(float))
            )
            if task in BINARY_TASKS:
                ref_parts[-1]["probability_mean"] = np.clip(z["prediction"].to_numpy(float), 0, 1)
        ref_test = pd.concat(ref_parts, ignore_index=True)

        references = {
            "Previous_Z2": prev_t,
            "A3": a3t,
            "Model32": ref_test,
        }
        test_stats = []
        for name, ref in references.items():
            for j, task in enumerate(TASKS):
                b = bootstrap_pair(
                    ultimate_t,
                    ref,
                    task,
                    args.bootstrap_reps,
                    args.seed + 7000 + j + 100 * list(references).index(name),
                )
                b.update({"comparison": "Z2_ultimate_vs_" + name})
                test_stats.append(b)
        pd.DataFrame(test_stats).to_csv(outdir / "Z2_ultimate_paired_bootstrap.csv", index=False)

        cal = calibration_report(ultimate_t)
        cal.to_csv(outdir / "Z2_ultimate_calibration.csv", index=False)
    else:
        print("[TEST] Z2 Ultimate did not meet the predeclared OOF gain; no new test evaluation.", flush=True)
        # Previous Z2 test results are existing frozen artifacts; used only as
        # the already-selected primary reference when Ultimate is rejected.
        selected_metrics_test = metrics(prev_t)

    summary = {
        "final_candidate": "Z2_ultimate",
        "selected_primary": selected_name,
        "selection_metric": "grouped_5fold_OOF_mean_binary_AUPRC",
        "ultimate_oof_mean_binary_auprc": float(ultimate_oof),
        "previous_z2_nested_oof_mean_binary_auprc": float(previous_oof_binary),
        "actual_z2_tuned_validation_mean_binary_auprc": float(metrics(prev_v)["mean_binary_auprc"]),
        "oof_delta_vs_nested_previous_z2": float(oof_delta),
        "oof_delta_vs_actual_z2_tuned": float(delta_vs_actual_z2),
        "minimum_gain_required": MIN_OOF_GAIN,
        "validation_metrics_selected_primary": selected_metrics_val,
        "test_metrics_selected_primary": selected_metrics_test,
        "ultimate_test_evaluated": bool(becomes_primary),
        "ultimate_test_metrics": ultimate_test_metrics,
        "test_used_for_selection": False,
        "no_new_preprocessing": True,
        "source_names": SOURCE_NAMES,
        "n_trials": int(args.z2_trials),
        "grouped_cv_folds": int(args.cv_folds),
        "bootstrap_weight_reps": int(args.bootstrap_weight_reps),
        "bootstrap_stat_reps": int(args.bootstrap_reps),
        "stop_after_z2_ultimate": True,
    }
    (outdir / "Z2_ultimate_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    selected_for_det_v = selected_v
    selected_for_det_t = ultimate_t if becomes_primary else prev_t
    det_v = deterioration_from_frame(selected_for_det_v)
    det_t = deterioration_from_frame(selected_for_det_t)
    pd.DataFrame([
        {"split": "val", **det_v},
        {"split": "test", **det_t},
    ]).to_csv(outdir / "Z2_ultimate_deterioration.csv", index=False)

    print("\n" + "=" * 110)
    print("FINAL Z2 ULTIMATE RESULT")
    print("=" * 110)
    print(json.dumps(summary, indent=2, default=str))
    print(f"\nArtifacts: {outdir}")
    print("FINAL Z2 ULTIMATE COMPLETE — no further architecture search.", flush=True)


if __name__ == "__main__":
    main()
