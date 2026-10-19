#!/usr/bin/env python3
"""
FINAL Z4 — Seed-stabilized, task-specialized, cross-fitted super-ensemble
===========================================================================

Purpose
-------
This is the final planned ensemble experiment for Pipeline A V2.
It adds only validation-driven ensemble/stability machinery to the frozen
prediction pool and frozen Pipeline-A-V2 inputs. It does NOT create a new
preprocessing pipeline.

Locked ingredients
------------------
1) Existing 33 model prediction artifacts (00..32).
2) A3_rank_max5, A2_probability_mean_max3, Z1_tuned, Z2_tuned.
3) Best already-selected Z3 architecture/configuration, trained with the
   eight predeclared seeds 20260923..20260930.
4) Z3 probability mean, Z3 rank mean, Z3 trimmed mean, and seed std diagnostics.
5) Validation-best task specialist single model and validation-best pair.
6) Validation-only residual error-complement model relative to A3.
7) Two-regime disagreement gate: low disagreement -> A3/Z1; high disagreement
   -> Z3 probability/rank summaries.
8) Ten-source final pool; convex weights on a 0.05 grid, [0.05, 0.60].
9) Thirty deterministic/randomized-start coordinate-search trials, each scored
   by subject-grouped 5-fold OOF validation.
10) Bootstrap-averaged final weights (B=100) using validation subjects only.
11) Optional U6 trigger: if initial Z4 OOF improves over Z2_tuned validation by
    >=0.001, train the second-best existing Z3 architecture with four seeds and
    rerun the predeclared Z4 search with that extra source.
12) Temperature scaling is diagnostic only; it never changes primary selection.
13) Test is evaluated only after all validation decisions are frozen.

Primary selection
-----------------
Mean of four binary validation AUPRCs.
LOS MAE is reported but does not enter the binary primary score.
Deterioration is evaluation-only probability-space noisy-OR of mortality,
ventilation and vasopressor.

No further architecture search is intended after this script.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score, log_loss, brier_score_loss

PROJECT_ROOT = Path("/home/abdullah2025/fyp_project")
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "model_z4_final"
BASE_SCRIPT = PROJECT_ROOT / "model_z_family.py"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
try:
    import model_z_family as zb
except Exception:
    spec = importlib.util.spec_from_file_location("model_z_family", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise
    zb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(zb)

BINARY_TASKS = tuple(zb.BINARY_TASKS)
LOS_TASK = zb.LOS_TASK
TASKS = tuple(zb.TASKS)
MODEL_IDS = list(zb.MODEL_IDS)
Z2_REFERENCE_MARGIN = 0.001
Z4_WEIGHT_MIN = 0.05
Z4_WEIGHT_MAX = 0.60
Z4_WEIGHT_STEP = 0.05
Z3_SEEDS = [20260923,20260924,20260925,20260926,20260927,20260928,20260929,20260930]
U6_SEEDS = [20260923,20260924,20260925,20260926]


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))


def safe_ap(y, p):
    keep = np.isfinite(y) & np.isfinite(p)
    y = np.asarray(y[keep]).astype(int)
    p = np.asarray(p[keep]).astype(float)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, p))


def safe_auc(y, p):
    keep = np.isfinite(y) & np.isfinite(p)
    y = np.asarray(y[keep]).astype(int)
    p = np.asarray(p[keep]).astype(float)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def rank01(x: np.ndarray) -> np.ndarray:
    return pd.Series(np.asarray(x, dtype=float)).rank(method="average").to_numpy(float) / len(x)


def bounded_grid_weights(n: int, rng: np.random.Generator) -> np.ndarray:
    """Exact 0.05-grid simplex point with [0.05,0.60] bounds."""
    min_units = int(round(Z4_WEIGHT_MIN / Z4_WEIGHT_STEP))
    max_units = int(round(Z4_WEIGHT_MAX / Z4_WEIGHT_STEP))
    total_units = int(round(1.0 / Z4_WEIGHT_STEP))
    rem = total_units - n * min_units
    if n * min_units > total_units or n * max_units < total_units:
        raise ValueError(f"No feasible simplex for n={n}")
    extra = np.zeros(n, dtype=int)
    remaining = rem
    while remaining > 0:
        eligible = np.flatnonzero(extra < (max_units - min_units))
        if len(eligible) == 0:
            raise RuntimeError("Failed to construct bounded simplex weights")
        j = int(rng.choice(eligible))
        extra[j] += 1
        remaining -= 1
    w = (min_units + extra).astype(float) * Z4_WEIGHT_STEP
    rng.shuffle(w)
    if not np.isclose(w.sum(), 1.0):
        raise AssertionError(f"Bad weight sum: {w}")
    return w


def feasible_equal_weights(n: int) -> np.ndarray:
    if n <= 0:
        raise ValueError("No sources")
    w = np.full(n, 1.0 / n, dtype=float)
    if np.any(w < Z4_WEIGHT_MIN - 1e-12) or np.any(w > Z4_WEIGHT_MAX + 1e-12):
        raise ValueError(f"Equal weights violate configured bounds for n={n}")
    return w


def candidate_pairs_for_task(candidates: Dict[str, pd.DataFrame], task: str):
    frame = zb.align(candidates, MODEL_IDS, "val", task)
    keep = zb.valid_mask(frame, MODEL_IDS)
    if not keep.any():
        raise ValueError(f"No valid validation rows for task {task}")
    y = frame["target"].to_numpy(float)
    R = zb.rank_matrix(frame, MODEL_IDS)
    raw = zb.raw_matrix(frame, MODEL_IDS)
    single = []
    for j, mid in enumerate(MODEL_IDS):
        if task in BINARY_TASKS:
            score = safe_ap(y[keep], R[keep, j])
        else:
            score = -mean_absolute_error(y[keep], raw[keep, j])
        single.append((score, mid))
    single.sort(reverse=True)

    best_single = single[0][1]
    best_pair = None
    best_pair_score = -np.inf
    for i in range(len(MODEL_IDS)):
        for j in range(i + 1, len(MODEL_IDS)):
            if task in BINARY_TASKS:
                p = 0.5 * (R[:, i] + R[:, j])
            else:
                p = 0.5 * (raw[:, i] + raw[:, j])
            score = safe_ap(y[keep], p[keep]) if task in BINARY_TASKS else -mean_absolute_error(y[keep], p[keep])
            if score > best_pair_score:
                best_pair_score = score
                best_pair = (MODEL_IDS[i], MODEL_IDS[j])
    if best_pair is None:
        raise RuntimeError(f"Unable to find pair for {task}")
    return best_single, best_pair, float(single[0][0]), float(best_pair_score)


def task_component(candidates: Dict[str,pd.DataFrame], model_ids: Sequence[str], split: str, task: str) -> pd.DataFrame:
    f = zb.align(candidates, list(model_ids), split, task)
    out = f[["stay_id","subject_id","target","validity"]].copy()
    for mid in model_ids:
        out[f"{mid}"] = f[f"pred_{mid}"].to_numpy(float)
    return out


def choose_error_complement(candidates: Dict[str,pd.DataFrame], a3: pd.DataFrame, task: str) -> str:
    f = zb.align(candidates, MODEL_IDS, "val", task)
    a3t = a3[a3["task"] == task].sort_values("stay_id")
    if not np.array_equal(f["stay_id"].to_numpy(), a3t["stay_id"].to_numpy()):
        raise ValueError(f"A3 alignment failed for error-complement {task}")
    keep = f["validity"].astype(bool).to_numpy() & np.isfinite(f["target"].to_numpy(float))
    y = f["target"].to_numpy(float)[keep]
    a3r = a3t["prediction"].to_numpy(float)[keep]
    ra = y - a3r
    vals = []
    for mid in MODEL_IDS:
        p = f[f"pred_{mid}"].to_numpy(float)[keep]
        rb = y - p
        if len(rb) < 3 or np.std(ra) == 0 or np.std(rb) == 0:
            corr = 1.0
        else:
            corr = float(np.corrcoef(ra, rb)[0,1])
        vals.append((abs(corr), corr, mid))
    vals.sort(key=lambda x: (x[0], x[2]))
    return vals[0][2]


def specialists(candidates, a3v):
    out = {}
    for task in TASKS:
        single, pair, single_score, pair_score = candidate_pairs_for_task(candidates, task)
        complement = choose_error_complement(candidates, a3v, task)
        out[task] = {
            "task_best_single": single,
            "task_best_single_validation_score": single_score,
            "task_best_pair": list(pair),
            "task_best_pair_validation_score": pair_score,
            "error_complement": complement,
        }
    return out


def align_frames(frames: Sequence[pd.DataFrame], task: str) -> pd.DataFrame:
    fs = [x[x["task"] == task].sort_values("stay_id").reset_index(drop=True) for x in frames]
    base = fs[0][["stay_id","subject_id","target","validity"]].copy()
    for f in fs[1:]:
        if not np.array_equal(base["stay_id"].to_numpy(), f["stay_id"].to_numpy()):
            raise ValueError(f"Alignment failure for {task}")
        if not np.array_equal(base["subject_id"].to_numpy(), f["subject_id"].to_numpy()):
            raise ValueError(f"Subject alignment failure for {task}")
    return base


def frame_pred(df: pd.DataFrame, task: str) -> np.ndarray:
    return df[df["task"] == task].sort_values("stay_id")["prediction"].to_numpy(float)


def frame_prob(df: pd.DataFrame, task: str) -> np.ndarray:
    z = df[df["task"] == task].sort_values("stay_id")
    if task in BINARY_TASKS and "probability_mean" in z.columns:
        return z["probability_mean"].to_numpy(float)
    return z["prediction"].to_numpy(float)


def build_seed_summary(seed_frames_v, seed_frames_t):
    def one(frames):
        rows=[]
        for task in TASKS:
            zs = [f[f["task"] == task].sort_values("stay_id").reset_index(drop=True) for f in frames]
            base = zs[0][["stay_id","subject_id","target","validity"]].copy()
            vals = np.vstack([z["prediction"].to_numpy(float) for z in zs])
            if task in BINARY_TASKS:
                probs = np.vstack([frame_prob(f, task) for f in frames])
                prob_mean = np.nanmean(probs, axis=0)
                rank_stack = np.vstack([rank01(v) for v in vals])
                rank_mean = np.nanmean(rank_stack, axis=0)
                sorted_vals = np.sort(probs, axis=0)
                trim = np.nanmean(sorted_vals[1:-1], axis=0) if probs.shape[0] >= 3 else prob_mean
                pred = rank_mean
                prob = np.clip(prob_mean,0.0,1.0)
            else:
                pred = np.nanmean(vals, axis=0)
                prob = None
                rank_mean = pred
                trim = pred
                probs = vals
            out = base.copy()
            out["task"] = task
            out["prediction"] = pred
            if task in BINARY_TASKS:
                out["probability_mean"] = prob
            out["seed_std"] = np.nanstd(vals, axis=0, ddof=1)
            out["rank_mean"] = rank_mean
            out["trim_mean"] = trim
            rows.append(out)
        return pd.concat(rows, ignore_index=True)
    return one(seed_frames_v), one(seed_frames_t)


def build_source_matrix(task, split, a3, a2, z1, z2, z3summary, candidates, spec, regime=None):
    # Returns common rows plus score/probability matrices in fixed source order.
    sources = [
        ("A3", a3, "prediction"),
        ("A2", a2, "prediction"),
        ("Z1_tuned", z1, "prediction"),
        ("Z3_prob_mean", z3summary, "probability_mean"),
        ("Z3_rank_mean", z3summary, "rank_mean"),
        ("Z3_trim_mean", z3summary, "trim_mean"),
    ]
    names = [x[0] for x in sources]
    frames = [x[1][x[1]["task"] == task].sort_values("stay_id").reset_index(drop=True) for x in sources]
    base = frames[0][["stay_id","subject_id","target","validity"]].copy()
    for f in frames[1:]:
        if not np.array_equal(base["stay_id"].to_numpy(), f["stay_id"].to_numpy()):
            raise ValueError(f"Source alignment failure: {task}/{split}")
    score_cols=[]; prob_cols=[]
    for name, f, col in sources:
        if col in f.columns:
            score = f[col].to_numpy(float)
        else:
            score = f["prediction"].to_numpy(float)
        if task in BINARY_TASKS:
            score = rank01(score)
            prob = frame_prob(f, task)
        else:
            prob = score
        score_cols.append(score); prob_cols.append(prob)

    # Task-best single.
    sb = spec[task]["task_best_single"]
    sf = task_component(candidates,[sb],split,task)
    score = sf[sb].to_numpy(float)
    score_cols.append(rank01(score) if task in BINARY_TASKS else score)
    prob_cols.append(np.clip(score,0,1))
    names.append("task_best_single")

    # Task-best pair.
    pair = spec[task]["task_best_pair"]
    pf = task_component(candidates,pair,split,task)
    p1 = pf[pair[0]].to_numpy(float); p2 = pf[pair[1]].to_numpy(float)
    if task in BINARY_TASKS:
        pair_score = 0.5*(rank01(p1)+rank01(p2))
        pair_prob = 0.5*(np.clip(p1,0,1)+np.clip(p2,0,1))
    else:
        pair_score = 0.5*(p1+p2); pair_prob=pair_score
    score_cols.append(pair_score); prob_cols.append(pair_prob); names.append("task_best_pair")

    # Error complement.
    ec = spec[task]["error_complement"]
    ef = task_component(candidates,[ec],split,task)
    ep = ef[ec].to_numpy(float)
    score_cols.append(rank01(ep) if task in BINARY_TASKS else ep)
    prob_cols.append(np.clip(ep,0,1)); names.append("error_complement")

    # Two-regime gate (prepared separately and appended as a single source).
    if regime is None:
        gate_score = score_cols[0].copy()
    else:
        gate_score = regime["prediction"]
    score_cols.append(gate_score)
    prob_cols.append(np.clip(gate_score,0,1))
    names.append("two_regime_gate")

    S = np.column_stack(score_cols)
    P = np.column_stack(prob_cols)
    return base, names, S, P


def fit_two_regime_gate(train_arrays, task, fixed_seed=20260923):
    """Fit regime threshold and two 2-source blends on a training subset."""
    base, a3, z1, z3p, z3r = train_arrays
    a3=np.asarray(a3,float); z1=np.asarray(z1,float); z3p=np.asarray(z3p,float); z3r=np.asarray(z3r,float)
    dis=np.std(np.column_stack([a3,z1,z3p]),axis=1)
    med=float(np.nanmedian(dis))
    low=dis<=med; high=~low
    y=base["target"].to_numpy(float); valid=base["validity"].astype(bool).to_numpy()
    rng=np.random.default_rng(fixed_seed)
    best_low=(float("-inf"), np.array([0.5,0.5]))
    best_high=(float("-inf"), np.array([0.5,0.5]))
    grid=np.arange(0,1.0001,0.05)
    for w in grid:
        if task in BINARY_TASKS:
            ph=w*rank01(a3)+(1-w)*rank01(z1)
            ph2=w*rank01(z3p)+(1-w)*rank01(z3r)
        else:
            ph=w*a3+(1-w)*z1
            ph2=w*z3p+(1-w)*z3r
        keep=valid&low
        if keep.any():
            sc=safe_ap(y[keep],ph[keep]) if task in BINARY_TASKS else -mean_absolute_error(y[keep],ph[keep])
            if sc>best_low[0]: best_low=(sc,np.array([w,1-w]))
        keep=valid&high
        if keep.any():
            sc=safe_ap(y[keep],ph2[keep]) if task in BINARY_TASKS else -mean_absolute_error(y[keep],ph2[keep])
            if sc>best_high[0]: best_high=(sc,np.array([w,1-w]))
    pred=np.empty(len(y),float)
    if task in BINARY_TASKS:
        low_pred=best_low[1][0]*rank01(a3)+best_low[1][1]*rank01(z1)
        high_pred=best_high[1][0]*rank01(z3p)+best_high[1][1]*rank01(z3r)
    else:
        low_pred=best_low[1][0]*a3+best_low[1][1]*z1
        high_pred=best_high[1][0]*z3p+best_high[1][1]*z3r
    pred[low]=low_pred[low]; pred[high]=high_pred[high]
    return {"median_disagreement":med,"low_weights":best_low[1].tolist(),"high_weights":best_high[1].tolist(),"prediction":pred}


def regime_gate_for_frames(a3, z1, z3summary, task, split):
    frames=[x[x["task"]==task].sort_values("stay_id").reset_index(drop=True) for x in (a3,z1,z3summary)]
    base=frames[0][["stay_id","subject_id","target","validity"]].copy()
    a=frames[0]["prediction"].to_numpy(float)
    z1p=frames[1]["prediction"].to_numpy(float)
    z3p=frames[2]["probability_mean"].to_numpy(float) if task in BINARY_TASKS else frames[2]["prediction"].to_numpy(float)
    z3r=frames[2]["rank_mean"].to_numpy(float) if task in BINARY_TASKS else frames[2]["prediction"].to_numpy(float)
    fit=fit_two_regime_gate((base,a,z1p,z3p,z3r),task)
    out=base.copy(); out["task"]=task; out["prediction"]=fit["prediction"]
    if task in BINARY_TASKS: out["probability_mean"]=np.clip(fit["prediction"],0,1)
    return out, fit


def gate_from_training(train_idx, full_frame_data, task):
    """Fit gate on subset indices and return a callable-like dict for holdout."""
    base,a,z1,z3p,z3r=full_frame_data
    sub=lambda x: x[train_idx]
    b=base.iloc[train_idx].reset_index(drop=True)
    pars=fit_two_regime_gate((b,sub(a),sub(z1),sub(z3p),sub(z3r)),task)
    return pars


def apply_gate_params(base, a3, z1, z3p, z3r, params, task):
    dis=np.std(np.column_stack([a3,z1,z3p]),axis=1)
    low=dis<=params["median_disagreement"]
    high=~low
    pred=np.full(len(base),np.nan,dtype=float)
    if task in BINARY_TASKS:
        lowp=params["low_weights"][0]*rank01(a3)+params["low_weights"][1]*rank01(z1)
        highp=params["high_weights"][0]*rank01(z3p)+params["high_weights"][1]*rank01(z3r)
    else:
        lowp=params["low_weights"][0]*a3+params["low_weights"][1]*z1
        highp=params["high_weights"][0]*z3p+params["high_weights"][1]*z3r
    pred[low]=lowp[low]; pred[high]=highp[high]
    # Rows where disagreement was non-finite (both low and high False) fall back to A3.
    unassigned=~np.isfinite(pred)
    if unassigned.any():
        fallback=a3.copy()
        fallback=np.where(np.isfinite(fallback),fallback,0.0)
        pred[unassigned]=fallback[unassigned]
    return pred


def build_gate_full(a3,z1,z3summary,task,split):
    return regime_gate_for_frames(a3,z1,z3summary,task,split)


def apply_gate_with_params(a3, z1, z3summary, task, params):
    frames=[x[x["task"]==task].sort_values("stay_id").reset_index(drop=True) for x in (a3,z1,z3summary)]
    base=frames[0][["stay_id","subject_id","target","validity"]].copy()
    a=frames[0]["prediction"].to_numpy(float)
    z1p=frames[1]["prediction"].to_numpy(float)
    z3p=frames[2]["probability_mean"].to_numpy(float) if task in BINARY_TASKS else frames[2]["prediction"].to_numpy(float)
    z3r=frames[2]["rank_mean"].to_numpy(float) if task in BINARY_TASKS else frames[2]["prediction"].to_numpy(float)
    pred=apply_gate_params(base,a,z1p,z3p,z3r,params,task)
    out=base.copy(); out["task"]=task; out["prediction"]=pred
    if task in BINARY_TASKS:
        out["probability_mean"]=np.clip(params["low_weights"][0]*np.clip(a,0,1)+params["low_weights"][1]*np.clip(z1p,0,1),0,1)
        # For high-regime cases, use the high-regime probability blend.
        dis=np.std(np.column_stack([a,z1p,z3p]),axis=1); high=dis>params["median_disagreement"];
        highprob=params["high_weights"][0]*np.clip(z3p,0,1)+params["high_weights"][1]*np.clip(z3r,0,1)
        lowprob=params["low_weights"][0]*np.clip(a,0,1)+params["low_weights"][1]*np.clip(z1p,0,1)
        out["probability_mean"]=np.where(high,np.clip(highprob,0,1),np.clip(lowprob,0,1))
    return out


def source_arrays_for_task(task, split, a3, a2, z1, z2, z3summary, candidates, spec, gate_df, extra=None):
    sources=["A3","A2","Z1_tuned","Z3_prob_mean","Z3_rank_mean","Z3_trim_mean","task_best_single","task_best_pair","error_complement","two_regime_gate"]
    frames=[a3,a2,z1,z3summary,z3summary,z3summary]
    fs=[f[f["task"]==task].sort_values("stay_id").reset_index(drop=True) for f in frames]
    base=fs[0][["stay_id","subject_id","target","validity"]].copy()
    score=[]; prob=[]
    for i,name in enumerate(sources[:6]):
        f=fs[i]
        if name=="Z3_prob_mean" and task in BINARY_TASKS:
            s=f["probability_mean"].to_numpy(float)
        elif name=="Z3_rank_mean" and task in BINARY_TASKS:
            s=f["rank_mean"].to_numpy(float)
        elif name=="Z3_trim_mean" and task in BINARY_TASKS:
            s=f["trim_mean"].to_numpy(float)
        else:
            s=f["prediction"].to_numpy(float)
        if task in BINARY_TASKS: sr=rank01(s); pr=frame_prob(f,task)
        else: sr=s; pr=s
        score.append(sr); prob.append(pr)
    sb=spec[task]["task_best_single"]; sf=task_component(candidates,[sb],split,task); p=sf[sb].to_numpy(float)
    score.append(rank01(p) if task in BINARY_TASKS else p); prob.append(np.clip(p,0,1))
    pair=spec[task]["task_best_pair"]; pf=task_component(candidates,pair,split,task); p1=pf[pair[0]].to_numpy(float); p2=pf[pair[1]].to_numpy(float)
    score.append(0.5*(rank01(p1)+rank01(p2)) if task in BINARY_TASKS else 0.5*(p1+p2)); prob.append(0.5*(np.clip(p1,0,1)+np.clip(p2,0,1)))
    ec=spec[task]["error_complement"]; ef=task_component(candidates,[ec],split,task); ep=ef[ec].to_numpy(float)
    score.append(rank01(ep) if task in BINARY_TASKS else ep); prob.append(np.clip(ep,0,1))
    gf=gate_df[gate_df["task"]==task].sort_values("stay_id").reset_index(drop=True); gp=gf["prediction"].to_numpy(float)
    score.append(gp); prob.append(np.clip(gp,0,1))
    S=np.column_stack(score)
    P=np.column_stack(prob)
    # Replace any residual non-finite values with a finite fallback (A3 column) so
    # a single pathological source cannot invalidate every row.
    a3_col=S[:,0].copy()
    a3_col=np.where(np.isfinite(a3_col),a3_col,0.0)
    for j in range(S.shape[1]):
        bad=~np.isfinite(S[:,j])
        if bad.any():
            S[bad,j]=a3_col[bad]
    for j in range(P.shape[1]):
        bad=~np.isfinite(P[:,j])
        if bad.any():
            P[bad,j]=0.0
    return base,S,P,sources


def score_prediction(task,y,p):
    y=np.asarray(y,float); p=np.asarray(p,float)
    keep=np.isfinite(y)&np.isfinite(p)
    if keep.sum()==0:
        return float("-inf")
    if task in BINARY_TASKS:
        return safe_ap(y[keep],p[keep])
    return -float(mean_absolute_error(y[keep],p[keep]))


def coordinate_optimize(X, y, valid, task, names, start, seed=0, iterations=2):
    """Grid-constrained coordinate ascent on a bounded simplex."""
    rng=np.random.default_rng(seed)
    w=np.array(start,dtype=float)
    step=Z4_WEIGHT_STEP
    min_u=int(round(Z4_WEIGHT_MIN/step)); max_u=int(round(Z4_WEIGHT_MAX/step)); total_u=int(round(1/step))
    n=len(w)
    # Project to exact grid if needed by generating a nearby valid point.
    if not (np.isclose(w.sum(),1.0) and np.all(w>=Z4_WEIGHT_MIN-1e-9) and np.all(w<=Z4_WEIGHT_MAX+1e-9)):
        w=bounded_grid_weights(n,rng)
    valid=np.asarray(valid,dtype=bool)
    if int(valid.sum())==0:
        return w, float("-inf")
    base_score=score_prediction(task,y[valid],X[valid]@w)
    for _ in range(iterations):
        changed=False
        order=rng.permutation(n)
        for j in order:
            candidates=[]
            for delta in (-step,0.0,step):
                nw=w.copy(); nw[j]+=delta
                if nw[j] < Z4_WEIGHT_MIN-1e-9 or nw[j] > Z4_WEIGHT_MAX+1e-9: continue
                # compensate by moving the same amount from another coordinate.
                if abs(delta)>1e-12:
                    others=[k for k in range(n) if k!=j and w[k]-delta>=Z4_WEIGHT_MIN-1e-9 and w[k]-delta<=Z4_WEIGHT_MAX+1e-9]
                    if not others: continue
                    # choose the donor that gives the best score.
                    for k in others:
                        tw=w.copy(); tw[j]+=delta; tw[k]-=delta
                        if not (np.isclose(tw.sum(),1.0) and np.all(tw>=Z4_WEIGHT_MIN-1e-9) and np.all(tw<=Z4_WEIGHT_MAX+1e-9)): continue
                        sc=score_prediction(task,y[valid],X[valid]@tw)
                        candidates.append((sc,tw))
                else:
                    candidates.append((base_score,nw))
            if candidates:
                sc,tw=max(candidates,key=lambda z:z[0])
                if sc>base_score+1e-12:
                    w=tw; base_score=sc; changed=True
        if not changed: break
    return w, float(base_score)


def generate_trials(n, count, seed):
    rng=np.random.default_rng(seed)
    out=[]; seen=set(); out.append(feasible_equal_weights(n)); seen.add(tuple(np.round(out[0],8)))
    while len(out)<count:
        w=bounded_grid_weights(n,rng); key=tuple(np.round(w,8))
        if key not in seen:
            seen.add(key); out.append(w)
    return out


def crossfit_z4(task, source_data, groups, n_trials, seed):
    base,S,P,names=source_data
    y=base["target"].to_numpy(float); valid=base["validity"].astype(bool).to_numpy()
    finite=np.all(np.isfinite(S),axis=1) & np.isfinite(y)
    valid &= finite
    dropped=int((~valid).sum())
    if dropped:
        bad_cols=~np.all(np.isfinite(S),axis=0)
        offenders=[names[i] for i in np.flatnonzero(bad_cols)]
        print(f"[Z4-DROP] task={task} dropped {dropped}/{len(base)} rows; "
              f"non-finite source columns: {offenders or ['y_target']}",flush=True)
    if not np.any(valid):
        raise RuntimeError(
            f"Z4 task={task}: every row was dropped by the finiteness filter. "
            f"Sources={names}. Inspect outputs for NaN in any source matrix."
        )
    g=base["subject_id"].to_numpy()
    splitter=GroupKFold(n_splits=5)
    trials=generate_trials(S.shape[1], n_trials, seed)
    rows=[]; best=None
    for trial_no, trial_start in enumerate(trials,1):
        oof=np.full(len(base),np.nan,float); fold_weights=[]
        for fold,(tr,ho) in enumerate(splitter.split(S, y, groups=g)):
            trv=valid[tr]; hov=valid[ho]
            w,fit_score=coordinate_optimize(S[tr], y[tr], trv, task, names, trial_start, seed=seed+trial_no*100+fold, iterations=2)
            fold_weights.append(w)
            oof[ho]=S[ho]@w
        keep=valid & np.isfinite(oof)
        score=score_prediction(task,y[keep],oof[keep])
        meanw=np.mean(np.vstack(fold_weights),axis=0)
        row={"task":task,"trial":trial_no,"oof_score":float(score),"metric":"AUPRC" if task in BINARY_TASKS else "negative_MAE"}
        row.update({names[i]:float(meanw[i]) for i in range(len(names))})
        rows.append(row)
        if best is None or score>best[0]: best=(float(score),trial_no,meanw,oof)
    if best is None: raise RuntimeError(f"No Z4 trial for {task}")
    return {"score":best[0],"trial":best[1],"weights":best[2],"oof":best[3]},pd.DataFrame(rows)


def bootstrap_average_weights(task, source_data, base_seed, best_weights, reps=100):
    base,S,P,names=source_data
    y=base["target"].to_numpy(float); valid=base["validity"].astype(bool).to_numpy(); groups=base["subject_id"].to_numpy()
    subjects=np.unique(groups[valid]); index_by={s:np.flatnonzero(groups==s) for s in subjects}; rng=np.random.default_rng(base_seed)
    ws=[]
    for b in range(reps):
        sampled=rng.choice(subjects,size=len(subjects),replace=True)
        idx=np.concatenate([index_by[s] for s in sampled])
        keep=np.zeros(len(base),dtype=bool); keep[idx]=True; keep &= valid
        start=best_weights
        w,_=coordinate_optimize(S,y,keep,task,names,start,seed=base_seed+10000+b,iterations=2)
        ws.append(w)
    avg=np.mean(np.vstack(ws),axis=0)
    return avg,np.vstack(ws)


def assemble_from_weights(base,S,P,names,weights,task):
    w=np.asarray(weights,float); w=w/w.sum()
    pred=S@w
    out=base[["stay_id","subject_id","target","validity"]].copy(); out["task"]=task; out["prediction"]=pred
    if task in BINARY_TASKS:
        out["probability_mean"]=np.clip(P@w,0,1)
    return out


def build_z4_selected(source_map, task_weights):
    return pd.concat([assemble_from_weights(*source_map[t],task_weights[t],task) for t in TASKS],ignore_index=True)


def fit_temperature(y,p):
    keep=np.isfinite(y)&np.isfinite(p)
    y=y[keep].astype(int); p=np.clip(p[keep].astype(float),1e-6,1-1e-6)
    if len(y)==0 or len(np.unique(y))<2: return 1.0
    logits=np.log(p/(1-p))
    try:
        from scipy.optimize import minimize_scalar
        def obj(logT):
            T=float(np.exp(logT)); q=1/(1+np.exp(-logits/T)); return log_loss(y,q,labels=[0,1])
        res=minimize_scalar(obj,bounds=(-3,3),method="bounded")
        return float(np.exp(res.x))
    except Exception:
        grid=np.exp(np.linspace(-2,2,81)); vals=[]
        for T in grid:
            q=1/(1+np.exp(-logits/T)); vals.append(log_loss(y,q,labels=[0,1]))
        return float(grid[int(np.argmin(vals))])


def temp_apply(p,T):
    p=np.clip(np.asarray(p,float),1e-6,1-1e-6); z=np.log(p/(1-p)); return 1/(1+np.exp(-z/float(T)))


def ece(y,p,bins=10):
    y=np.asarray(y).astype(int); p=np.asarray(p,float); out=0.0
    edges=np.linspace(0,1,bins+1)
    for lo,hi in zip(edges[:-1],edges[1:]):
        m=(p>=lo)&((p<hi) if hi<1 else (p<=hi))
        if not m.any(): continue
        out += m.mean()*abs(p[m].mean()-y[m].mean())
    return float(out)


def statistical_reports(selected, a3, z2, model32, bootstrap_reps, seed):
    rows=[]
    for name,ref in [("A3_rank_max5",a3),("Z2_tuned",z2),("Model32",model32)]:
        for j,task in enumerate(TASKS):
            try:
                cmp_seed=seed + 1000*(1 + ["A3_rank_max5","Z2_tuned","Model32"].index(name)) + j
                rows.append({"comparison":f"Z4_tuned_vs_{name}",**zb.bootstrap_pair(selected,ref,task,bootstrap_reps,cmp_seed)})
            except Exception as e:
                rows.append({"comparison":f"Z4_tuned_vs_{name}","task":task,"error":str(e)})
    return pd.DataFrame(rows)


def load_best_z3_config():
    selected=zb.OUTPUT_ROOT/"Z3_tuned_selected.json"
    if selected.exists():
        obj=json.loads(selected.read_text())
        cfg=obj.get("config")
        if isinstance(cfg,dict):
            return cfg
    trials=zb.OUTPUT_ROOT/"Z3_tuned_trials.csv"
    if trials.exists():
        df=pd.read_csv(trials)
        if len(df):
            row=df.sort_values("validation_mean_AUPRC",ascending=False).iloc[0]
            cfg=json.loads(row["config"]) if isinstance(row["config"],str) else row["config"]
            return cfg
    raise FileNotFoundError("Existing selected Z3 configuration not found. Run the existing Z3 family first.")


def load_second_z3_config():
    path=zb.OUTPUT_ROOT/"Z3_tuned_trials.csv"
    if not path.exists(): return None
    df=pd.read_csv(path)
    if len(df)<2: return None
    df=df.sort_values("validation_mean_AUPRC",ascending=False).reset_index(drop=True)
    return json.loads(df.iloc[1]["config"])


def train_z3_seed_ensemble(config, seeds, structured, demographics, text_input, cohort, out_root, label):
    val=[]; test=[]; metas=[]
    cfg=dict(config)
    for i,seed in enumerate(seeds):
        c=dict(cfg); c["name"]=f"{label}_seed{seed}"
        run=out_root/label/f"seed_{seed}"
        run.mkdir(parents=True,exist_ok=True)

        # Resume-safe cache: all eight seeds in the interrupted run completed
        # successfully before summary assembly crashed. Reuse only a complete
        # prediction/metadata triplet whose seed and config exactly match.
        vpath=run/"validation_predictions.csv"
        tpath=run/"test_predictions.csv"
        mpath=run/"metadata.json"
        reused=False
        if vpath.exists() and tpath.exists() and mpath.exists():
            try:
                cached_meta=json.loads(mpath.read_text())
                if int(cached_meta.get("seed", -1)) == int(seed) and cached_meta.get("config") == c:
                    vv=pd.read_csv(vpath)
                    tt=pd.read_csv(tpath)
                    meta=cached_meta
                    reused=True
                    print(f"[Z4-Z3-CACHE] reusing completed {label} seed={seed} val_mean_AUPRC={meta['validation_metrics']['mean_binary_auprc']:.6f}",flush=True)
            except Exception as e:
                print(f"[Z4-Z3-CACHE] ignoring incomplete/incompatible cache for seed={seed}: {e}",flush=True)

        if not reused:
            vv,tt,meta=zb.train_z3_once(structured,demographics,text_input,cohort,c,seed,run)
            print(f"[Z4-Z3-SEED] {label} seed={seed} val_mean_AUPRC={meta['validation_metrics']['mean_binary_auprc']:.6f}",flush=True)

        val.append(vv); test.append(tt); metas.append(meta)

    z3v, z3t = build_seed_summary(val, test)
    return z3v, z3t, metas


def main():
    ap=argparse.ArgumentParser(description="Final Z4 seed-stabilized super-ensemble")
    ap.add_argument("--cv-folds",type=int,default=5)
    ap.add_argument("--z4-trials",type=int,default=30)
    ap.add_argument("--bootstrap-weight-reps",type=int,default=100)
    ap.add_argument("--bootstrap-reps",type=int,default=2000)
    ap.add_argument("--seed",type=int,default=20260923)
    ap.add_argument("--text-path",default=None)
    ap.add_argument("--skip-u6",action="store_true")
    args=ap.parse_args()
    if args.cv_folds!=5: raise ValueError("Z4 protocol is locked to 5 subject-grouped folds")
    if args.z4_trials!=30: raise ValueError("Z4 protocol is locked to 30 trials")
    OUTPUT_ROOT.mkdir(parents=True,exist_ok=True)
    set_all_seeds(args.seed)
    print("="*110); print("FINAL Z4 — SEED-STABILIZED SUPER-ENSEMBLE"); print("="*110)
    print(f"Seeds={Z3_SEEDS} | trials={args.z4_trials} | grouped_CV=5 | weight_bootstrap={args.bootstrap_weight_reps}",flush=True)
    print("TEST IS NOT USED FOR Z4 SELECTION",flush=True)

    candidates=zb.discover_candidates()
    a3_rec,a2_rec,a3v,a3t,a2v,a2t=zb.build_reference_ensembles(candidates)
    z1f_rec,z1fv,z1ft,z1f_cv=zb.z1_make(candidates,False,5,args.seed)
    z1t_rec,z1tv,z1tt,z1t_cv=zb.z1_make(candidates,True,5,args.seed)
    z2f_rec,z2fv,z2ft=zb.build_z2(candidates,a3v,a3t,a2v,a2t,z1fv,z1ft,False)
    z2t_rec,z2tv,z2tt=zb.build_z2(candidates,a3v,a3t,a2v,a2t,z1tv,z1tt,True)

    for name,v,t in [("A3_rank_max5",a3v,a3t),("A2_probability_mean_max3",a2v,a2t),("Z1_tuned",z1tv,z1tt),("Z2_tuned",z2tv,z2tt)]:
        zb.save_model_outputs(name,v,t,OUTPUT_ROOT)
    print(f"A3 val={zb.metrics(a3v)['mean_binary_auprc']:.6f}")
    print(f"Z2 val={zb.metrics(z2tv)['mean_binary_auprc']:.6f}")

    # Frozen Z3 data.
    structured_path,demographics_path,alignment_path,text_path=zb.locate_z3_inputs(args.text_path)
    structured,demographics,text_input,cohort=zb.load_z3_data(text_path)
    best_cfg=load_best_z3_config()
    save_json(OUTPUT_ROOT/"locked_z3_config.json",best_cfg)
    z3v,z3t,z3meta=train_z3_seed_ensemble(best_cfg,Z3_SEEDS,structured,demographics,text_input,cohort,OUTPUT_ROOT,"Z3_seed8")
    z3v.to_csv(OUTPUT_ROOT/"Z3_seed_summary_validation.csv",index=False); z3t.to_csv(OUTPUT_ROOT/"Z3_seed_summary_test.csv",index=False)
    seed_diag=[]
    for task in TASKS:
        z=z3v[z3v["task"]==task]
        seed_diag.append({"task":task,"mean_seed_std":float(z["seed_std"].mean()),"median_seed_std":float(z["seed_std"].median()),"max_seed_std":float(z["seed_std"].max())})
    pd.DataFrame(seed_diag).to_csv(OUTPUT_ROOT/"Z3_seed_stability.csv",index=False)

    spec= specialists(candidates,a3v)
    save_json(OUTPUT_ROOT/"task_specialists_and_complement.json",spec)

    # Full-validation two-regime gates.
    gates_v={}; gates_t={}
    gate_meta={}
    for task in TASKS:
        gv,gpars=build_gate_full(a3v,z1tv,z3v,task,"val")
        gt=apply_gate_with_params(a3t,z1tt,z3t,task,gpars)
        gates_v[task]=gv; gates_t[task]=gt; gate_meta[task]=gpars
    gate_v=pd.concat(gates_v.values(),ignore_index=True); gate_t=pd.concat(gates_t.values(),ignore_index=True)
    gate_v.to_csv(OUTPUT_ROOT/"two_regime_gate_validation.csv",index=False); gate_t.to_csv(OUTPUT_ROOT/"two_regime_gate_test.csv",index=False)
    save_json(OUTPUT_ROOT/"two_regime_gate_params.json",gate_meta)

    # Build source data per task. Initial 10 sources.
    source_val={}; source_test={}
    for task in TASKS:
        source_val[task]=source_arrays_for_task(task,"val",a3v,a2v,z1tv,z2tv,z3v,candidates,spec,gates_v[task])
        source_test[task]=source_arrays_for_task(task,"test",a3t,a2t,z1tt,z2tt,z3t,candidates,spec,gates_t[task])

    # Fixed Z4 = equal 10-source convex pool.
    fixed_weights={t:feasible_equal_weights(source_val[t][1].shape[1]) for t in TASKS}
    z4_fixed=[]
    for task in TASKS:
        base,S,P,names=source_val[task]; z4_fixed.append(assemble_from_weights(base,S,P,names,fixed_weights[task],task))
    z4fv=pd.concat(z4_fixed,ignore_index=True)
    z4_fixed_t=[]
    for task in TASKS:
        base,S,P,names=source_test[task]; z4_fixed_t.append(assemble_from_weights(base,S,P,names,fixed_weights[task],task))
    z4ft=pd.concat(z4_fixed_t,ignore_index=True)
    zb.save_model_outputs("Z4_fixed",z4fv,z4ft,OUTPUT_ROOT)

    # Tuned initial Z4: 30 trials, 5-fold subject grouped OOF.
    tuned_meta={}; trial_tables=[]; selected_task_weights={}
    for i,task in enumerate(TASKS):
        # Group labels come from the exact source-frame row order.
        best,table=crossfit_z4(task,source_val[task],source_val[task][0]["subject_id"].to_numpy(),args.z4_trials,args.seed+100+i)
        avg_w, boot_ws=bootstrap_average_weights(task,source_val[task],args.seed+500+i,best["weights"],args.bootstrap_weight_reps)
        selected_task_weights[task]=avg_w
        table.to_csv(OUTPUT_ROOT/f"Z4_{task}_trials.csv",index=False)
        trial_tables.append(table)
        tuned_meta[task]={"oof_score":best["score"],"selected_trial":best["trial"],"prebootstrap_weights":best["weights"].tolist(),"bootstrap_average_weights":avg_w.tolist(),"bootstrap_std":boot_ws.std(axis=0).tolist()}
        print(f"[Z4] {task}: OOF={best['score']:.6f} selected_trial={best['trial']} weights={dict(zip(source_val[task][3],np.round(avg_w,4)))}",flush=True)
    z4v_parts=[assemble_from_weights(*source_val[t],selected_task_weights[t],t) for t in TASKS]
    z4v=pd.concat(z4v_parts,ignore_index=True)
    z4t_parts=[assemble_from_weights(*source_test[t],selected_task_weights[t],t) for t in TASKS]
    z4t=pd.concat(z4t_parts,ignore_index=True)
    save_json(OUTPUT_ROOT/"Z4_tuned_selection.json",{"test_used_for_selection":False,"tasks":tuned_meta})
    pd.concat(trial_tables,ignore_index=True).to_csv(OUTPUT_ROOT/"Z4_all_trial_weights.csv",index=False)
    zb.save_model_outputs("Z4_tuned",z4v,z4t,OUTPUT_ROOT)
    z4_oof_mean=float(np.mean([tuned_meta[t]["oof_score"] for t in BINARY_TASKS]))
    z2_val_mean=zb.metrics(z2tv)["mean_binary_auprc"]
    print(f"Z4 initial OOF mean binary AUPRC={z4_oof_mean:.6f} | Z2_tuned validation={z2_val_mean:.6f}",flush=True)

    # Optional U6, predeclared validation trigger only.
    u6_trigger=(z4_oof_mean >= z2_val_mean + Z2_REFERENCE_MARGIN) and (not args.skip_u6)
    save_json(OUTPUT_ROOT/"U6_trigger.json",{"triggered":bool(u6_trigger),"condition":"Z4 OOF >= Z2 validation + 0.001","z4_oof":z4_oof_mean,"z2_validation":z2_val_mean})
    final_label="Z4_tuned"
    final_v=z4v; final_t=z4t; final_weights=selected_task_weights
    if u6_trigger:
        second_cfg=load_second_z3_config()
        if second_cfg is not None:
            u6v,u6t,_=train_z3_seed_ensemble(second_cfg,U6_SEEDS,structured,demographics,text_input,cohort,OUTPUT_ROOT,"Z3_b_seed4")
            u6v.to_csv(OUTPUT_ROOT/"Z3_b_seed_summary_validation.csv",index=False); u6t.to_csv(OUTPUT_ROOT/"Z3_b_seed_summary_test.csv",index=False)
            source_val_u6={}; source_test_u6={}
            for task in TASKS:
                # Add the second architecture as an additional probability/rank/trim source.
                base,S,P,names=source_val[task]
                ub=u6v[u6v["task"]==task].sort_values("stay_id"); ut=u6t[u6t["task"]==task].sort_values("stay_id")
                if not np.array_equal(base["stay_id"].to_numpy(),ub["stay_id"].to_numpy()): raise ValueError("U6 alignment failure")
                if task in BINARY_TASKS:
                    S=np.column_stack([S,rank01(ub["probability_mean"].to_numpy(float))]); P=np.column_stack([P,ub["probability_mean"].to_numpy(float)])
                else:
                    S=np.column_stack([S,ub["prediction"].to_numpy(float)]); P=np.column_stack([P,ub["prediction"].to_numpy(float)])
                names=names+["Z3_b_prob_mean"]
                source_val_u6[task]=(base,S,P,names)
                bt,S2,P2,names2=source_test[task]
                if task in BINARY_TASKS:
                    S2=np.column_stack([S2,rank01(ut["probability_mean"].to_numpy(float))]); P2=np.column_stack([P2,ut["probability_mean"].to_numpy(float)])
                else:
                    S2=np.column_stack([S2,ut["prediction"].to_numpy(float)]); P2=np.column_stack([P2,ut["prediction"].to_numpy(float)])
                names2=names2+["Z3_b_prob_mean"]
                source_test_u6[task]=(bt,S2,P2,names2)
            u6_weights={}; u6_meta={}; u6_tables=[]
            for i,task in enumerate(TASKS):
                b,tab=crossfit_z4(task,source_val_u6[task],source_val_u6[task][0]["subject_id"].to_numpy(),args.z4_trials,args.seed+900+i)
                aw,bws=bootstrap_average_weights(task,source_val_u6[task],args.seed+950+i,b["weights"],args.bootstrap_weight_reps)
                u6_weights[task]=aw; u6_meta[task]={"oof_score":b["score"],"selected_trial":b["trial"],"bootstrap_average_weights":aw.tolist()}; u6_tables.append(tab)
            u6_mean=float(np.mean([u6_meta[t]["oof_score"] for t in BINARY_TASKS]))
            save_json(OUTPUT_ROOT/"Z4_U6_selection.json",{"test_used_for_selection":False,"mean_binary_oof":u6_mean,"tasks":u6_meta})
            pd.concat(u6_tables,ignore_index=True).to_csv(OUTPUT_ROOT/"Z4_U6_all_trial_weights.csv",index=False)
            print(f"[U6] expanded-pool OOF mean={u6_mean:.6f}",flush=True)
            if u6_mean>z4_oof_mean:
                final_label="Z4_tuned_U6"; final_weights=u6_weights
                final_v=pd.concat([assemble_from_weights(*source_val_u6[t],u6_weights[t],t) for t in TASKS],ignore_index=True)
                final_t=pd.concat([assemble_from_weights(*source_test_u6[t],u6_weights[t],t) for t in TASKS],ignore_index=True)
                zb.save_model_outputs(final_label,final_v,final_t,OUTPUT_ROOT)

    # Frozen final selection record.
    final_val_score=zb.metrics(final_v)["mean_binary_auprc"]
    selection={"selected_model":final_label,"validation_mean_binary_auprc":final_val_score,"test_used_for_selection":False,"u6_triggered":u6_trigger}
    save_json(OUTPUT_ROOT/"FINAL_Z4_SELECTION.json",selection)

    # Temperature calibration diagnostic: fit only on validation, report val/test.
    cal_rows=[]
    for task in BINARY_TASKS:
        zv=final_v[final_v["task"]==task].sort_values("stay_id"); zt=final_t[final_t["task"]==task].sort_values("stay_id")
        vv=zv["validity"].astype(bool).to_numpy(); tv=zt["validity"].astype(bool).to_numpy()
        vp=zv.get("probability_mean",zv["prediction"]).to_numpy(float); tp=zt.get("probability_mean",zt["prediction"]).to_numpy(float)
        vy=zv["target"].to_numpy(float); ty=zt["target"].to_numpy(float)
        vk=vv&np.isfinite(vy)&np.isfinite(vp); tk=tv&np.isfinite(ty)&np.isfinite(tp)
        if not vk.any():
            continue
        T=fit_temperature(vy[vk],vp[vk])
        vp2=temp_apply(vp[vk],T); tp2=temp_apply(tp[tk],T) if tk.any() else np.array([])
        cal_rows.append({"task":task,"temperature":T,"val_brier_before":brier_score_loss(vy[vk],vp[vk]),"val_brier_after":brier_score_loss(vy[vk],vp2),"val_ece_before":ece(vy[vk],vp[vk]),"val_ece_after":ece(vy[vk],vp2),"test_brier_before":brier_score_loss(ty[tk],tp[tk]) if tk.any() else np.nan,"test_brier_after":brier_score_loss(ty[tk],tp2) if tk.any() else np.nan,"test_ece_before":ece(ty[tk],tp[tk]) if tk.any() else np.nan,"test_ece_after":ece(ty[tk],tp2) if tk.any() else np.nan})
        if tk.any():
            cal_rows[-1]["test_auprc_before"]=safe_ap(ty[tk],tp[tk]); cal_rows[-1]["test_auprc_after"]=safe_ap(ty[tk],tp2); cal_rows[-1]["test_auroc_before"]=safe_auc(ty[tk],tp[tk]); cal_rows[-1]["test_auroc_after"]=safe_auc(ty[tk],tp2)
        else:
            cal_rows[-1]["test_auprc_before"]=np.nan; cal_rows[-1]["test_auprc_after"]=np.nan; cal_rows[-1]["test_auroc_before"]=np.nan; cal_rows[-1]["test_auroc_after"]=np.nan
    # Guarded lookup: a task with no valid validation rows defaults to T=1 if a test row exists.
    tv_map={r["task"]:r["temperature"] for r in cal_rows}
    for task in BINARY_TASKS:
        T=tv_map.get(task,1.0)  # guarded fallback for a future zero-valid-validation task
        if task not in tv_map:
            print(f"[CAL] {task}: no valid validation rows; test calibration uses T=1.0",flush=True)
    pd.DataFrame(cal_rows).to_csv(OUTPUT_ROOT/"calibration_diagnostics.csv",index=False)

    # Final statistics: test only now, after selection is frozen.
    print("[TEST] selection frozen; evaluating final Z4 once",flush=True)
    print(json.dumps(zb.metrics(final_t),indent=2))
    print("[TEST] Z4 vs references",flush=True)
    ref_parts=[]
    for task in TASKS:
        z=zb.task_frame(candidates,"32","test",task)
        ref_parts.append(z[["stay_id","subject_id","target","validity"]].assign(task=task,prediction=z["prediction"].to_numpy(float)))
    ref_test=pd.concat(ref_parts,ignore_index=True)
    stats=statistical_reports(final_t,a3t,z2tt,ref_test,args.bootstrap_reps,args.seed)
    stats.to_csv(OUTPUT_ROOT/"Z4_statistical_comparisons.csv",index=False)

    det=zb.deterioration_from_frame(final_t)
    save_json(OUTPUT_ROOT/"Z4_deterioration_test.json",det)
    summary={
        "final_model":final_label,
        "validation_mean_binary_auprc":float(final_val_score),
        "test_mean_binary_auprc":float(zb.metrics(final_t)["mean_binary_auprc"]),
        "test_mean_binary_auroc":float(zb.metrics(final_t)["mean_binary_auroc"]),
        "test_los_mae":float(zb.metrics(final_t)[f"{LOS_TASK}_mae"]),
        "z2_validation_mean_binary_auprc":float(z2_val_mean),
        "test_used_for_selection":False,
        "stop_after_z4":True,
    }
    save_json(OUTPUT_ROOT/"Z4_FINAL_SUMMARY.json",summary)
    print(json.dumps(summary,indent=2),flush=True)
    print("FINAL Z4 COMPLETE — stop architecture search.",flush=True)

if __name__=="__main__":
    main()
