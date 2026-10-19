#!/usr/bin/env python3
"""
MIMIC-III CareVue external validation, step 4 of 4: score the frozen models
and compute the same statistics as the MIMIC-IV master evaluation.

Nothing is trained, tuned, refitted or recalibrated here.

What it does
------------
1. Scores the 14 frozen members that the locked ensemble recipes use
   (00, 04, 05, 09, 11, 12, 20, 23, 24, 25, 28, 29, 31, 32). Every member is
   run through its own family script: build_model -> checkpoint -> the family's
   make_loader and collect_predictions (same sigmoid, same BF16 autocast, same
   expm1(.) + 24 h LOS transform). Tabular members use _tabularize_structured
   and the saved .joblib estimators.
2. Rebuilds A2, A3, Z1_tuned and Z2_tuned with the logic of model_z_family.py
   (apply_recipe and z2_apply_from_components) and the locked recipe files.
   Deterioration is the probability-space noisy-OR of mortality, ventilation
   and vasopressor, as in the MIMIC-IV evaluation.
3. Computes, with the functions of complete_statistical_evaluation.py itself:
   AUROC, AUPRC, Brier, ECE (10 bins), sensitivity / specificity / precision /
   NPV / F1 at the threshold locked on MIMIC-IV validation, calibration
   intercept and slope, LOS MAE / RMSE / median AE / P90 AE / mean error / R2,
   all with percentile bootstrap intervals, plus paired comparisons against
   Model 32 (DeLong, McNemar, paired bootstrap, BH-FDR).
   Two cohorts are reported: all stays, and stays with charted vitals.

Self-check (run this first)
---------------------------
    python score_mimic3_external.py --verify-on-mimic4

  a) rebuilds A2 / A3 / Z1 / Z2 from the saved member test predictions and
     compares them with the saved outputs/model_z_family/*_test_predictions.csv
  b) re-scores the MIMIC-IV test set from the checkpoints and compares every
     member with its saved test_predictions.csv

Run
---
    python score_mimic3_external.py                 # radiology text embeddings
    python score_mimic3_external.py --text none     # sensitivity: text missing

Paste back only  <out-dir>/mimic3_summary_for_paste.txt  (aggregate numbers).
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

BINARY = ("mortality_24h", "ventilation_onset_24h", "vasopressor_onset_24h", "readmission_30d_24h")
DET = "deterioration_24h"
DET_PARTS = BINARY[:3]
LOS = "los_hours_24h"
ALL_BINARY = BINARY + (DET,)
PRIMARY = BINARY + (LOS,)
PARENTS = ("A3", "A2", "Z1")                     # order used by z2_apply_from_components
RECIPE_FILES = {"A2": "A2_recipes.json", "A3": "A3_recipes.json", "Z1": "Z1_tuned_recipes.json", "Z2": "Z2_tuned_recipes.json"}
ENSEMBLE_FILES = {"A3": "A3_rank_max5", "A2": "A2_probability_mean_max3", "Z1": "Z1_tuned", "Z2_tuned": "Z2_tuned"}
FAMILY_MODULES = [(5, "family_01_tabular_00_05"), (9, "family_02_rnn_06_09"), (12, "family_03_tcn_10_12"),
                  (16, "family_04_clinical_sequence_13_16_patched"), (23, "family_05_transformer_modern_17_23"),
                  (27, "family_06_state_space_24_27"), (32, "family_07_advanced_multimodal_28_32")]
FAMILY07_DIR = "baseline_family_07_advanced_multimodal_28_32"
TEXT_MODELS = {"28", "29", "30", "31", "32"}
REFERENCE = "32"
EXTERNAL_SEED_OFFSET = 300_000


# ----------------------------------------------------------------------------
# Discovery
# ----------------------------------------------------------------------------
def module_name_for(mid: str) -> str:
    return next(name for hi, name in FAMILY_MODULES if int(mid) <= hi)


def discover_model_dirs(root: Path) -> dict[str, Path]:
    found: dict[str, list[Path]] = {}
    for d in (root / "outputs").glob("baseline_family_*/tuned/*/final_selected_configuration"):
        m = re.match(r"^(\d{2})_", d.parent.name)
        if m:
            found.setdefault(m.group(1), []).append(d)
    out = {}
    for mid, dirs in found.items():
        if len(dirs) > 1:
            f7 = [d for d in dirs if FAMILY07_DIR in str(d)]
            dirs = f7 if len(f7) == 1 else dirs
        if len(dirs) != 1:
            raise FileNotFoundError(f"Model {mid}: expected one final_selected_configuration, found {dirs}")
        out[mid] = dirs[0]
    return out


def load_recipes(root: Path) -> dict:
    z = root / "outputs" / "model_z_family"
    return {k: json.loads((z / f).read_text()) for k, f in RECIPE_FILES.items()}


def members_needed(recipes: dict) -> list[str]:
    ids = set()
    for parent in ("A2", "A3", "Z1"):
        for task in PRIMARY:
            ids.update(m for m, w in recipes[parent][task]["weights"].items() if w > 0)
    return sorted(ids)


# ----------------------------------------------------------------------------
# Member scoring (each family's own code path)
# ----------------------------------------------------------------------------
def load_checkpoint(path: Path):
    import torch
    try:
        return torch.load(path, map_location="cpu")
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def score_neural(mid: str, mdir: Path, X, D, T, TM, device) -> dict[str, np.ndarray]:
    import torch
    mod = importlib.import_module(module_name_for(mid))
    if hasattr(mod, "mamba_pure_default") and not mod.mamba_pure_default():
        raise RuntimeError("FYP_MAMBA_PURE must be 1 (the frozen models were trained on the pure PyTorch path).")
    blob = load_checkpoint(mdir / "checkpoint.pt")
    params = blob["params"]
    model = mod.build_model(mid, params).to(device)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    n = len(X)
    data = SimpleNamespace(structured=X, demographics=D, text=T, text_missing=TM,
                           targets={t: np.zeros(n, dtype=np.float32) for t in mod.PRIMARY_TASKS},
                           valid_arrays={t: np.ones(n, dtype=np.int8) for t in mod.PRIMARY_TASKS})
    rb = mdir / "runtime_batch_size.txt"
    bs = int(rb.read_text().strip()) if rb.exists() else int(params.get("batch_size", 512))
    loader = mod.make_loader(data, np.arange(n, dtype=np.int64), bs, False, device.type == "cuda")
    with torch.no_grad():
        pred, _ = mod.collect_predictions(model, loader, device, mod.amp_enabled(device))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {t: np.asarray(pred[t], dtype=np.float64).reshape(-1) for t in PRIMARY}


def score_tabular(mid: str, mdir: Path, Xtab: np.ndarray) -> dict[str, np.ndarray]:
    import joblib
    f1 = importlib.import_module(module_name_for(mid))
    out = {}
    for task in PRIMARY:
        est = joblib.load(mdir / f"{task}.joblib")
        out[task] = np.asarray(f1._predict(est, Xtab, task in BINARY), dtype=np.float64).reshape(-1)
        del est
    return out


def score_members(ids: list[str], dirs: dict[str, Path], X, D, T, TM, device) -> dict[str, dict[str, np.ndarray]]:
    out, Xtab = {}, None
    for mid in ids:
        t0 = time.time()
        if int(mid) <= 5:
            if Xtab is None:
                f1 = importlib.import_module(module_name_for(mid))
                Xtab = np.concatenate([f1._tabularize_structured(X), np.asarray(D, dtype=np.float32)], axis=1).astype(np.float32, copy=False)
            out[mid] = score_tabular(mid, dirs[mid], Xtab)
        else:
            out[mid] = score_neural(mid, dirs[mid], X, D, T, TM, device)
        for t in PRIMARY:
            if not np.isfinite(out[mid][t]).all():
                raise RuntimeError(f"Model {mid}: non-finite predictions for {t}")
        print(f"  scored {mid}  {dirs[mid].parent.name:34s} {time.time() - t0:6.1f}s", flush=True)
    return out


# ----------------------------------------------------------------------------
# Ensembles: same arithmetic as model_z_family.apply_recipe / z2_apply_from_components
# ----------------------------------------------------------------------------
def rank01(v: np.ndarray) -> np.ndarray:
    return pd.Series(np.asarray(v, dtype=float)).rank(method="average").to_numpy(float) / float(len(v))


def apply_recipe(recipe: dict, task: str, member: dict) -> tuple[np.ndarray, np.ndarray | None]:
    ids = list(recipe["members"])
    w = np.asarray([recipe["weights"][m] for m in ids], dtype=float)
    w = w / w.sum()
    raw = np.column_stack([member[m][task] for m in ids])
    if task in BINARY:
        # model_z_family.apply_recipe ranks only when blend == "rank_mean"; any other value
        # (including Z1's "weighted_rank_mean") is blended on the probability scale.
        pred = np.column_stack([rank01(raw[:, j]) for j in range(len(ids))]) @ w if recipe.get("blend", "rank_mean") == "rank_mean" else raw @ w
        return pred, np.clip(raw @ w, 0.0, 1.0)
    return raw @ w, None


def build_ensembles(recipes: dict, member: dict) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """Returns {model: {task: {"pred": ..., "prob": ...}}} for A3, A2, Z1, Z2_tuned (deterioration included)."""
    out = {p: {} for p in PARENTS}
    out["Z2_tuned"] = {}
    for task in PRIMARY:
        comp = {p: apply_recipe(recipes[p][task], task, member) for p in PARENTS}
        for p in PARENTS:
            out[p][task] = {"pred": comp[p][0], "prob": comp[p][1] if comp[p][1] is not None else comp[p][0]}
        w = np.asarray([recipes["Z2"][task]["weights"][p] for p in PARENTS], dtype=float)
        if task in BINARY:
            pred = np.column_stack([rank01(comp[p][0]) for p in PARENTS]) @ w
            prob = np.clip(np.column_stack([comp[p][1] for p in PARENTS]) @ w, 0.0, 1.0)
        else:
            pred = np.column_stack([comp[p][0] for p in PARENTS]) @ w
            prob = pred
        out["Z2_tuned"][task] = {"pred": pred, "prob": prob}
    for name in out:
        out[name][DET] = noisy_or({t: out[name][t]["prob"] for t in DET_PARTS})
    return out


def noisy_or(prob: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    p = 1.0 - np.prod(np.stack([1.0 - np.clip(prob[t], 0.0, 1.0) for t in DET_PARTS], axis=0), axis=0)
    return {"pred": p, "prob": p}


def member_as_model(m: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    out = {t: {"pred": m[t], "prob": m[t]} for t in PRIMARY}
    out[DET] = noisy_or({t: m[t] for t in DET_PARTS})
    return out


# ----------------------------------------------------------------------------
# Saved MIMIC-IV predictions
# ----------------------------------------------------------------------------
def read_saved_member(mdir: Path) -> pd.DataFrame:
    return pd.read_csv(mdir / "test_predictions.csv", usecols=["stay_id", "subject_id", "split", "task", "target", "validity", "prediction"])


def saved_member_arrays(df: pd.DataFrame, split: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    out, ids = {}, None
    for t in PRIMARY:
        z = df[(df["split"] == split) & (df["task"] == t)].sort_values("stay_id")
        if ids is None:
            ids = z["stay_id"].to_numpy()
        elif not np.array_equal(ids, z["stay_id"].to_numpy()):
            raise ValueError("stay_id alignment failure in saved member predictions")
        out[t] = z["prediction"].to_numpy(float)
    return ids, out


def load_cse(root: Path):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return importlib.import_module("complete_statistical_evaluation")


def validation_thresholds(root: Path, dirs: dict[str, Path], member_ids: list[str]) -> dict[tuple[str, str], float]:
    """Max-F1 thresholds on MIMIC-IV validation, exactly as complete_statistical_evaluation.py locks them."""
    cse = load_cse(root)
    thr: dict[tuple[str, str], float] = {}

    def lock(name: str, d: pd.DataFrame) -> None:
        for task in ALL_BINARY:
            sub = d[d["task"].eq(task)].copy()
            p = cse.probability_series(sub, task)
            yv = pd.to_numeric(sub["target"], errors="coerce").to_numpy(float)
            keep = sub["validity"].astype(bool).to_numpy() & np.isfinite(yv) & np.isfinite(p) & np.isin(yv, [0.0, 1.0])   # = cse.valid_binary
            thr[(name, task)] = cse.threshold_at_max_f1(yv[keep].astype(int), p[keep])

    for mid in member_ids:
        d = read_saved_member_full(dirs[mid])
        lock(mid, d[d["split"].eq("val")])
    z = root / "outputs" / "model_z_family"
    for name, stem in ENSEMBLE_FILES.items():
        d = pd.read_csv(z / f"{stem}_validation_predictions.csv")
        d = pd.concat([d, cse.derive_z2_deterioration_rows(d, "val")], ignore_index=True)
        lock(name, d)
    return thr


def read_saved_member_full(mdir: Path) -> pd.DataFrame:
    return pd.read_csv(mdir / "test_predictions.csv")


# ----------------------------------------------------------------------------
# Verification on MIMIC-IV
# ----------------------------------------------------------------------------
def verify_on_mimic4(root: Path, device, skip_members: bool) -> bool:
    recipes = load_recipes(root)
    dirs = discover_model_dirs(root)
    ids = members_needed(recipes)
    print(f"Members used by the locked recipes ({len(ids)}): {', '.join(ids)}")

    print("\n[a] Ensemble arithmetic: rebuild A2 / A3 / Z1 / Z2 from saved member test predictions")
    saved, stay = {}, None
    for mid in ids:
        s, arr = saved_member_arrays(read_saved_member(dirs[mid]), "test")
        if stay is None:
            stay = s
        elif not np.array_equal(stay, s):
            raise ValueError(f"Model {mid}: test stay_id order differs")
        saved[mid] = arr
    ens = build_ensembles(recipes, saved)
    ok_a = True
    z = root / "outputs" / "model_z_family"
    for name, stem in ENSEMBLE_FILES.items():
        ref = pd.read_csv(z / f"{stem}_test_predictions.csv")
        worst_pred, worst_prob = 0.0, 0.0
        for task in PRIMARY:
            r = ref[ref["task"].eq(task)].sort_values("stay_id")
            if not np.array_equal(r["stay_id"].to_numpy(), stay):
                raise ValueError(f"{name}/{task}: stay_id order differs from members")
            worst_pred = max(worst_pred, float(np.max(np.abs(r["prediction"].to_numpy(float) - ens[name][task]["pred"]))))
            if task in BINARY and "probability_mean" in r.columns:
                worst_prob = max(worst_prob, float(np.max(np.abs(r["probability_mean"].to_numpy(float) - ens[name][task]["prob"]))))
        good = worst_pred < 1e-6 and worst_prob < 1e-6
        ok_a &= good
        print(f"  {name:9s} max |prediction diff| {worst_pred:.3e}   max |probability_mean diff| {worst_prob:.3e}   {'ok' if good else 'MISMATCH'}")
    print("  PASS: ensemble arithmetic reproduces the saved MIMIC-IV files" if ok_a else "  FAIL: ensemble arithmetic differs from the saved MIMIC-IV files")
    if skip_members:
        return ok_a

    print("\n[b] Member scoring: re-score the MIMIC-IV test set from the checkpoints")
    v2 = root / "outputs" / "pipeline_a_v2"
    align = pd.read_csv(v2 / "a5_alignment_index_24h.csv", usecols=["stay_id", "split_24h"])
    test_idx = np.flatnonzero(align["split_24h"].astype(str).to_numpy() == "test")
    order = np.argsort(align["stay_id"].to_numpy()[test_idx], kind="mergesort")
    test_idx = test_idx[order]
    if not np.array_equal(align["stay_id"].to_numpy()[test_idx], stay):
        raise ValueError("MIMIC-IV test stay ids do not match the saved predictions")
    X = np.asarray(np.load(v2 / "structured_24h_processed.npy", mmap_mode="r")[test_idx], dtype=np.float32)
    D = np.asarray(np.load(v2 / "demographics_24h_processed.npy", mmap_mode="r")[test_idx], dtype=np.float32)
    T = TM = None
    if any(m in TEXT_MODELS for m in ids):
        f7 = importlib.import_module(module_name_for("32"))
        T_all, TM_all = f7.load_text_embeddings(align["stay_id"].to_numpy())
        T, TM = T_all[test_idx], TM_all[test_idx]
        del T_all, TM_all
    fresh = score_members(ids, dirs, X, D, T, TM, device)
    ok_b = True
    print(f"  {'model':6s} {'max |prob diff|':>16s} {'max LOS diff (h)':>17s} {'max LOS rel diff':>17s}")
    for mid in ids:
        dp = max(float(np.max(np.abs(fresh[mid][t] - saved[mid][t]))) for t in BINARY)
        dl = float(np.max(np.abs(fresh[mid][LOS] - saved[mid][LOS])))
        rl = float(np.max(np.abs(fresh[mid][LOS] - saved[mid][LOS]) / np.maximum(np.abs(saved[mid][LOS]), 1.0)))
        tol_p, tol_r = (1e-5, 1e-5) if int(mid) <= 5 else (5e-3, 2e-2)
        good = dp <= tol_p and rl <= tol_r
        ok_b &= good
        print(f"  {mid:6s} {dp:16.3e} {dl:17.3e} {rl:17.3e}   {'ok' if good else 'MISMATCH'}")
    print("  PASS: checkpoints reproduce the saved MIMIC-IV test predictions" if ok_b else "  FAIL: at least one member does not reproduce its saved predictions")
    if device.type != "cuda":
        print("  note: no GPU visible, so BF16 autocast was off; small differences for neural members are expected on CPU.")
    ok = ok_a and ok_b
    print("\nPASS: identical scoring path" if ok else "\nFAIL: stop and send this output")
    return ok


# ----------------------------------------------------------------------------
# Statistics (functions of complete_statistical_evaluation.py)
# ----------------------------------------------------------------------------
def _cse(root: str):
    if root not in sys.path:
        sys.path.insert(0, root)
    return importlib.import_module("complete_statistical_evaluation")


def binary_job(root: str, meta: dict, y, disc, prob, thr: float, seed: int, reps: int) -> dict:
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    cse = _cse(root)
    y = np.asarray(y).astype(int)
    row = {**meta, "n": int(len(y)), "events": int(y.sum()), "prevalence": float(y.mean()), "threshold": float(thr),
           "auroc": float(roc_auc_score(y, disc)), "auprc": float(average_precision_score(y, disc)),
           "brier": float(brier_score_loss(y, prob)), "ece": float(cse.ece_score(y, prob)),
           **cse.threshold_metrics(y, prob, thr), **cse.calibration_params(y, prob)}
    for metric, (lo, hi) in cse.bootstrap_core_binary_ci(y, disc, prob, reps, np.random.default_rng(seed)).items():
        row[f"{metric}_ci95_low"], row[f"{metric}_ci95_high"] = lo, hi
    for metric, (lo, hi) in cse.bootstrap_threshold_ci(y, prob, thr, reps, np.random.default_rng(seed + 2_000_000)).items():
        row[f"{metric}_ci95_low"], row[f"{metric}_ci95_high"] = lo, hi
    return row


def los_job(root: str, meta: dict, y, pred, seed: int, reps: int) -> dict:
    from sklearn.metrics import r2_score
    cse = _cse(root)
    e = pred - y
    row = {**meta, "n": int(len(y)), "mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e ** 2))),
           "median_ae": float(np.median(np.abs(e))), "p90_ae": float(np.quantile(np.abs(e), 0.9)), "mean_error": float(np.mean(e)),
           "r2": float(r2_score(y, pred)) if len(np.unique(y)) > 1 else float("nan")}
    for metric, (lo, hi) in cse.bootstrap_los_ci(y, pred, reps, np.random.default_rng(seed)).items():
        row[f"{metric}_ci95_low"], row[f"{metric}_ci95_high"] = lo, hi
    return row


def pair_job(root: str, meta: dict, y, disc_m, disc_r, prob_m, prob_r, t_m: float, t_r: float, seed: int, reps: int) -> list[dict]:
    cse = _cse(root)
    y = np.asarray(y).astype(int)
    rows = []
    pt, lo, hi, _ = cse.paired_bootstrap_delta(y, disc_m, disc_r, "auroc", None, None, reps, np.random.default_rng(seed + 901))
    rows.append({**meta, "test_type": "paired_DeLong_AUROC", "metric": "auroc", "n": int(len(y)), "delta_model_minus_reference": pt,
                 "ci95_low": lo, "ci95_high": hi, "p_value": cse.delong_pvalue(y, disc_m, disc_r)})
    cm, cr = (prob_m >= t_m).astype(int), (prob_r >= t_r).astype(int)
    rows.append({**meta, "test_type": "McNemar_exact", "metric": "classification_error", "n": int(len(y)),
                 "delta_model_minus_reference": float(np.mean(cm == y) - np.mean(cr == y)), "ci95_low": np.nan, "ci95_high": np.nan,
                 "p_value": cse.mcnemar_exact_pvalue(y, cm, cr)})
    for metric, pm, pr, tm, tr in [("auprc", disc_m, disc_r, None, None), ("brier", prob_m, prob_r, None, None),
                                   ("f1", prob_m, prob_r, t_m, t_r), ("sensitivity", prob_m, prob_r, t_m, t_r),
                                   ("specificity", prob_m, prob_r, t_m, t_r), ("precision", prob_m, prob_r, t_m, t_r),
                                   ("npv", prob_m, prob_r, t_m, t_r)]:
        pt, lo, hi, pb = cse.paired_bootstrap_delta(y, pm, pr, metric, tm, tr, reps, np.random.default_rng(seed + len(metric)))
        rows.append({**meta, "test_type": "paired_bootstrap", "metric": metric, "n": int(len(y)), "delta_model_minus_reference": pt,
                     "ci95_low": lo, "ci95_high": hi, "p_value": pb})
    return rows


def model_label(name: str, dirs: dict[str, Path]) -> tuple[str, str]:
    if name in dirs:
        return name, dirs[name].parent.name.split("_", 1)[1].replace("_", " ")
    return name, name


def run_statistics(root: Path, models: dict, labels: pd.DataFrame, cohorts: dict[str, np.ndarray], thr: dict, dirs: dict, a) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from joblib import Parallel, delayed
    cse = load_cse(root)
    seed0 = int(cse.SEED) + EXTERNAL_SEED_OFFSET
    y = {t: labels[t].to_numpy(float) for t in ALL_BINARY + (LOS,)}
    v = {t: labels[f"{t}_valid"].to_numpy(int) == 1 for t in ALL_BINARY + (LOS,)}
    bjobs, ljobs, pjobs = [], [], []
    for ci, (cname, cmask) in enumerate(cohorts.items()):
        names = list(models) if cname == "all" else [n for n in models if n in ENSEMBLE_FILES or n == REFERENCE]
        for name in names:
            mid, mname = model_label(name, dirs)
            sv = cse.model_seed_value(name)
            for ti, task in enumerate(ALL_BINARY):
                disc, prob = models[name][task]["pred"], models[name][task]["prob"]
                keep = cmask & v[task] & np.isin(y[task], [0.0, 1.0]) & np.isfinite(disc) & np.isfinite(prob)
                if keep.sum() == 0 or len(np.unique(y[task][keep])) < 2:
                    continue
                meta = {"dataset": "MIMIC-III CareVue", "cohort": cname, "model_id": mid, "model_name": mname, "task": task}
                bjobs.append(delayed(binary_job)(str(root), meta, y[task][keep], disc[keep], prob[keep], thr[(name, task)],
                                                 seed0 + ci * 50_000 + sv * 101 + ti * 1000, a.bootstrap_reps))
                if cname == "all" and name != REFERENCE and REFERENCE in models and (a.pairwise == "all" or name in ENSEMBLE_FILES):
                    rd, rp = models[REFERENCE][task]["pred"], models[REFERENCE][task]["prob"]
                    k2 = keep & np.isfinite(rd) & np.isfinite(rp)
                    pmeta = {"dataset": "MIMIC-III CareVue", "cohort": cname, "task": task, "model_id": mid, "reference_model_id": REFERENCE}
                    pjobs.append(delayed(pair_job)(str(root), pmeta, y[task][k2], disc[k2], rd[k2], prob[k2], rp[k2], thr[(name, task)],
                                                   thr[(REFERENCE, task)], seed0 + sv * 1000 + ti * 10_000, a.pairwise_bootstrap_reps))
            pred = models[name][LOS]["pred"]
            keep = cmask & v[LOS] & np.isfinite(y[LOS]) & np.isfinite(pred)
            if keep.sum():
                meta = {"dataset": "MIMIC-III CareVue", "cohort": cname, "model_id": mid, "model_name": mname, "task": LOS}
                ljobs.append(delayed(los_job)(str(root), meta, y[LOS][keep], pred[keep], seed0 + ci * 50_000 + sv * 73, a.bootstrap_reps))
    print(f"Statistics: {len(bjobs)} binary, {len(ljobs)} LOS, {len(pjobs)} paired comparisons on {a.jobs} workers", flush=True)
    par = Parallel(n_jobs=a.jobs, verbose=0)
    brows = par(bjobs)
    lrows = par(ljobs)
    prows = [r for rows in par(pjobs) for r in rows]
    pair = pd.DataFrame(prows)
    if len(pair):
        pair["q_value_bh_fdr"] = np.nan
        for _, idx in pair.groupby(["cohort", "task", "test_type"], dropna=False).groups.items():
            pair.loc[idx, "q_value_bh_fdr"] = cse.bh_fdr(pair.loc[idx, "p_value"]).to_numpy()
    return pd.DataFrame(brows), pd.DataFrame(lrows), pair


def curve_points(models: dict, labels: pd.DataFrame, out: Path) -> None:
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import precision_recall_curve, roc_curve
    roc, pr, cal = [], [], []
    for task in ALL_BINARY:
        yy = labels[task].to_numpy(float)
        keep = (labels[f"{task}_valid"].to_numpy(int) == 1) & np.isin(yy, [0.0, 1.0])
        yk = yy[keep].astype(int)
        if len(np.unique(yk)) < 2:
            continue
        d, p = models["Z2_tuned"][task]["pred"][keep], models["Z2_tuned"][task]["prob"][keep]
        base = {"dataset": "MIMIC-III CareVue", "model": "Z2_tuned", "task": task}
        f, t, th = roc_curve(yk, d)
        roc += [{**base, "fpr": x, "tpr": z, "threshold": q} for x, z, q in zip(f, t, th)]
        pc, rc, th = precision_recall_curve(yk, d)
        pr += [{**base, "precision": x, "recall": z, "threshold": th[i] if i < len(th) else np.nan} for i, (x, z) in enumerate(zip(pc, rc))]
        pt, pp = calibration_curve(yk, p, n_bins=10, strategy="uniform")
        cal += [{**base, "bin_index": i + 1, "mean_predicted_probability": pp[i], "observed_event_rate": pt[i]} for i in range(len(pt))]
    pd.DataFrame(roc).to_csv(out / "mimic3_z2_roc_curve_points.csv", index=False)
    pd.DataFrame(pr).to_csv(out / "mimic3_z2_pr_curve_points.csv", index=False)
    pd.DataFrame(cal).to_csv(out / "mimic3_z2_calibration_curve_points.csv", index=False)


def write_summary(out: Path, root: Path, binary: pd.DataFrame, los: pd.DataFrame, pair: pd.DataFrame, meta: dict) -> None:
    lines = ["MIMIC-III CareVue external validation: aggregate results (no patient rows)", "=" * 78, json.dumps(meta, indent=2), ""]
    ref = root / "outputs" / "complete_statistical_evaluation" / "complete_binary_statistics.csv"
    b = binary.copy()
    if ref.exists():
        m4 = pd.read_csv(ref, dtype={"model_id": str})
        m4 = m4[m4["split"].eq("test")][["model_id", "task", "auroc", "auprc"]].rename(columns={"auroc": "mimic4_test_auroc", "auprc": "mimic4_test_auprc"})
        m4["model_id"] = m4["model_id"].map(lambda s: s.zfill(2) if s.isdigit() else s)
        b = b.merge(m4, on=["model_id", "task"], how="left")
    cols = ["cohort", "model_id", "task", "n", "events", "prevalence", "auroc", "auroc_ci95_low", "auroc_ci95_high", "auprc", "auprc_ci95_low",
            "auprc_ci95_high", "brier", "ece", "calibration_intercept", "calibration_slope", "threshold", "sensitivity", "specificity", "precision",
            "npv", "f1"] + [c for c in ("mimic4_test_auroc", "mimic4_test_auprc") if c in b.columns]
    with pd.option_context("display.width", 400, "display.max_columns", 60, "display.max_rows", 500, "display.float_format", "{:.4f}".format):
        lines += ["BINARY TASKS", b[cols].sort_values(["cohort", "task", "model_id"]).to_string(index=False), ""]
        lines += ["LENGTH OF STAY (hours)", los.sort_values(["cohort", "model_id"]).to_string(index=False), ""]
        if len(pair):
            p = pair[pair["metric"].isin(["auroc", "auprc", "brier"])]
            lines += ["PAIRED COMPARISONS vs MODEL 32 (model minus reference)", p.sort_values(["task", "metric", "model_id"]).to_string(index=False), ""]
    (out / "mimic3_summary_for_paste.txt").write_text("\n".join(lines))
    z = b[(b["model_id"] == "Z2_tuned")]
    with pd.option_context("display.width", 250, "display.float_format", "{:.4f}".format):
        print("\nZ2_tuned on MIMIC-III CareVue")
        print(z[[c for c in ["cohort", "task", "n", "events", "auroc", "auroc_ci95_low", "auroc_ci95_high", "auprc", "brier", "ece",
                              "calibration_slope", "mimic4_test_auroc", "mimic4_test_auprc"] if c in z.columns]].to_string(index=False))
        print(los[los["model_id"] == "Z2_tuned"][["cohort", "n", "mae", "rmse", "median_ae", "mean_error", "r2"]].to_string(index=False))


def versions() -> dict:
    out = {"python": platform.python_version()}
    for m in ("numpy", "pandas", "scipy", "sklearn", "torch", "xgboost", "lightgbm", "joblib"):
        try:
            out[m] = importlib.import_module(m).__version__
        except Exception:
            out[m] = "not importable"
    return out


# ----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project-root", type=Path, default=Path.cwd())
    ap.add_argument("--m3-dir", type=Path, default=Path("outputs/external_mimic3_tempC"), help="MIMIC-III features (tensor, demographics, index, raw grid, text)")
    ap.add_argument("--out-dir", type=Path, default=None, help="default: outputs/external_mimic3_z2/<m3-dir name>__text_<mode>")
    ap.add_argument("--text", choices=["embeddings", "none"], default="embeddings", help="'none' feeds a zero vector with the missing-text flag set, as in the eICU run")
    ap.add_argument("--bootstrap-reps", type=int, default=1000)
    ap.add_argument("--pairwise-bootstrap-reps", type=int, default=1000)
    ap.add_argument("--pairwise", choices=["all", "ensembles"], default="all", help="which models are compared with Model 32")
    ap.add_argument("--jobs", type=int, default=min(16, os.cpu_count() or 1))
    ap.add_argument("--verify-on-mimic4", action="store_true")
    ap.add_argument("--skip-member-check", action="store_true", help="with --verify-on-mimic4: only check the ensemble arithmetic")
    ap.add_argument("--rescore", action="store_true", help="ignore cached member predictions in the output folder")
    a = ap.parse_args()

    root = a.project_root.resolve()
    os.chdir(root)
    sys.path.insert(0, str(root))
    os.environ["PYTHONPATH"] = str(root) + os.pathsep + os.environ.get("PYTHONPATH", "")
    os.environ.setdefault("FYP_MAMBA_PURE", "1")
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    if a.verify_on_mimic4:
        verify_on_mimic4(root, device, a.skip_member_check)
        return

    m3 = a.m3_dir if a.m3_dir.is_absolute() else root / a.m3_dir
    out = a.out_dir or (root / "outputs" / "external_mimic3_z2" / f"{m3.name}__text_{a.text}")
    (out / "members").mkdir(parents=True, exist_ok=True)

    recipes = load_recipes(root)
    dirs = discover_model_dirs(root)
    ids = members_needed(recipes)
    labels = pd.read_csv(m3 / "mimic3_index_24h.csv")
    n = len(labels)
    assert (labels["row_index"].to_numpy() == np.arange(n)).all(), "index is not in tensor row order"
    X = np.load(m3 / "mimic3_structured_24h_processed.npy", mmap_mode="r")
    D = np.load(m3 / "mimic3_demographics_24h_processed.npy", mmap_mode="r")
    assert X.shape == (n, 24, 110) and D.shape == (n, 4), (X.shape, D.shape)
    if a.text == "embeddings":
        tp, mp = m3 / "mimic3_text_embeddings.npy", m3 / "mimic3_text_missing.npy"
        if not tp.exists():
            raise FileNotFoundError(f"{tp} not found. Run build_mimic3_text_embeddings.py first, or use --text none.")
        T, TM = np.load(tp), np.load(mp).astype(np.float32)
    else:
        T, TM = np.zeros((n, 768), dtype=np.float32), np.ones(n, dtype=np.float32)
    assert T.shape == (n, 768) and TM.shape == (n,)
    text_share = float((TM == 0).mean())
    print(f"Cohort: {n:,} stays | members: {', '.join(ids)} | text: {a.text} ({text_share * 100:.1f}% of stays with text)")

    member = {}
    todo = []
    for mid in ids:
        f = out / "members" / f"{mid}.npz"
        if f.exists() and not a.rescore:
            z = np.load(f)
            member[mid] = {t: z[t].astype(np.float64) for t in PRIMARY}
        else:
            todo.append(mid)
    if todo:
        fresh = score_members(todo, dirs, X, D, T, TM, device)
        for mid in todo:
            np.savez_compressed(out / "members" / f"{mid}.npz", **fresh[mid])
            member[mid] = fresh[mid]

    models = build_ensembles(recipes, member)
    for mid in ids:
        models[mid] = member_as_model(member[mid])

    key = labels[["stay_id", "subject_id"]]
    frames = []
    for name in list(ENSEMBLE_FILES) + ids:
        for task in ALL_BINARY + (LOS,):
            frames.append(pd.DataFrame({"stay_id": key["stay_id"], "subject_id": key["subject_id"], "split": "external_mimic3_carevue", "model_id": name,
                                        "task": task, "target": labels[task].to_numpy(float), "validity": labels[f"{task}_valid"].to_numpy(int),
                                        "prediction": models[name][task]["pred"], "probability_mean": models[name][task]["prob"]}))
    pd.concat(frames, ignore_index=True).to_csv(out / "mimic3_predictions_long.csv", index=False)

    grid = np.load(m3 / "mimic3_hourly_grid_raw.npy", mmap_mode="r")
    has_vitals = np.isfinite(np.asarray(grid[:, :, 0])).any(axis=1)
    cohorts = {"all": np.ones(n, dtype=bool), "vitals_observed": has_vitals}
    thr = validation_thresholds(root, dirs, ids)
    binary, los, pair = run_statistics(root, models, labels, cohorts, thr, dirs, a)
    binary.to_csv(out / "mimic3_complete_binary_statistics.csv", index=False)
    los.to_csv(out / "mimic3_complete_los_statistics.csv", index=False)
    pair.to_csv(out / "mimic3_paired_comparisons_vs_model32.csv", index=False)
    pd.DataFrame([{"model_id": k[0], "task": k[1], "locked_threshold": val, "selection_rule": "MIMIC-IV validation max F1; smallest threshold on ties"}
                  for k, val in thr.items()]).to_csv(out / "mimic3_locked_validation_thresholds.csv", index=False)
    curve_points(models, labels, out)

    meta = {"dataset": "MIMIC-III v1.4, CareVue-only patients", "stays": int(n), "subjects": int(labels["subject_id"].nunique()),
            "stays_with_charted_vitals": int(has_vitals.sum()), "features_from": str(m3), "text_input": a.text, "stays_with_text_share": round(text_share, 4),
            "members_scored": ids, "recipes": RECIPE_FILES, "z1_blend_note": "Z1 recipe is applied on the probability scale, as model_z_family.apply_recipe does on MIMIC-IV",
            "thresholds": "locked on MIMIC-IV validation (max F1)", "bootstrap_reps": a.bootstrap_reps, "pairwise_bootstrap_reps": a.pairwise_bootstrap_reps,
            "bootstrap_unit": "stay (row), as in the MIMIC-IV evaluation", "external_refit": False, "external_tuning": False, "external_recalibration": False,
            "los_transform": "expm1(predicted_excess_log1p) + 24.0 hours", "device": str(device), "software": versions()}
    (out / "mimic3_external_validation_metadata.json").write_text(json.dumps(meta, indent=2))
    write_summary(out, root, binary, los, pair, meta)
    print(f"\nwrote {out}")
    print(f"paste back: {out / 'mimic3_summary_for_paste.txt'}")


if __name__ == "__main__":
    main()
