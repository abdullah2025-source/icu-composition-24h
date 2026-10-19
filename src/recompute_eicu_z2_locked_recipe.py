#!/usr/bin/env python3
"""
Recompute the eICU Z2_tuned results with the locked MIMIC-IV ensemble arithmetic.

Nothing is retrained and no model is re-scored: the script reads the member
predictions already saved in outputs/external_eicu_z2/members/*.npz.

Why
---
model_z_family.py (the code that produced the MIMIC-IV Z2 results) does this:
  * A3  : rank-mean of its members               (blend == "rank_mean")
  * A2  : mean of member probabilities           (blend == "raw_mean")
  * Z1  : weighted mean of member PROBABILITIES  (blend == "weighted_rank_mean",
          which apply_recipe does not treat as a rank blend)
  * Z2  : weighted mean of the RE-RANKED parent scores (rank / n of A3, A2, Z1)
The published eICU discrimination scores rank-blended Z1 and did not re-rank
the parents. This script applies the locked arithmetic instead.

Member files: 00 and 05 hold probabilities (predict_proba); the neural members
hold logits (direct forward output), so a sigmoid is applied to them, as in
eicu_z2_native_statistical_analysis.py.

What changes: the rank-space discrimination score (AUROC, AUPRC) of mortality,
ventilation and vasopressor. What must not change: probability_mean (Brier, ECE,
calibration, thresholded metrics), deterioration and LOS. The script checks
that against the published canonical file and prints the differences.

Statistics use the functions and seeds of complete_statistical_evaluation.py
(evaluate_eicu), so unchanged quantities reproduce the published values.
An extra hospital-cluster bootstrap is reported for AUROC and AUPRC.

    python recompute_eicu_z2_locked_recipe.py

Existing files are not overwritten. Only aggregate numbers are printed.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

TASKS = ("mortality_24h", "ventilation_onset_24h", "vasopressor_onset_24h")
DET = "deterioration_24h"
LOS = "los_hours_24h"
PARENTS = ("A3", "A2", "Z1")
TREE_MODELS = {"00", "04", "05"}                 # saved with predict_proba; all other members are logits
RECIPE_FILES = {"A2": "A2_recipes.json", "A3": "A3_recipes.json", "Z1": "Z1_tuned_recipes.json", "Z2": "Z2_tuned_recipes.json"}


def rank01(v: np.ndarray) -> np.ndarray:
    return pd.Series(np.asarray(v, dtype=float)).rank(method="average").to_numpy(float) / float(len(v))


def apply_recipe(recipe: dict, task: str, member) -> tuple[np.ndarray, np.ndarray]:
    """model_z_family.apply_recipe: rank only when blend == 'rank_mean'."""
    ids = list(recipe["members"])
    w = np.asarray([recipe["weights"][m] for m in ids], dtype=float)
    w = w / w.sum()
    raw = np.column_stack([member(m, task) for m in ids])
    if task == LOS:
        return raw @ w, raw @ w
    pred = np.column_stack([rank01(raw[:, j]) for j in range(len(ids))]) @ w if recipe.get("blend", "rank_mean") == "rank_mean" else raw @ w
    return pred, np.clip(raw @ w, 0.0, 1.0)


def cluster_boot(y: np.ndarray, s: np.ndarray, groups: np.ndarray, reps: int, seed: int) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score
    rng = np.random.default_rng(seed)
    order = np.argsort(groups, kind="mergesort")
    y, s, groups = y[order], s[order], groups[order]
    _, start, count = np.unique(groups, return_index=True, return_counts=True)
    auc, ap = [], []
    for _ in range(reps):
        pick = rng.integers(0, len(start), len(start))
        idx = np.concatenate([np.arange(start[k], start[k] + count[k]) for k in pick])
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        auc.append(roc_auc_score(yy, s[idx]))
        ap.append(average_precision_score(yy, s[idx]))
    q = lambda v: (float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975)))
    return {"auroc_hospital_ci95_low": q(auc)[0], "auroc_hospital_ci95_high": q(auc)[1],
            "auprc_hospital_ci95_low": q(ap)[0], "auprc_hospital_ci95_high": q(ap)[1], "hospital_bootstrap_reps": len(auc)}


def binary_row(root: str, task: str, y, disc, prob, thr: float, groups, reps: int, cluster_reps: int) -> dict:
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    if root not in sys.path:
        sys.path.insert(0, root)
    cse = importlib.import_module("complete_statistical_evaluation")
    ti = list(cse.BINARY_TASKS).index(task)
    row = {"dataset": "eICU-CRD", "model": "Z2_tuned", "task": task, "n": int(len(y)), "events": int(y.sum()), "prevalence": float(y.mean()),
           "auroc": float(roc_auc_score(y, disc)), "auprc": float(average_precision_score(y, disc)), "brier": float(brier_score_loss(y, prob)),
           "ece": float(cse.ece_score(y, prob)), "threshold_locked_from_mimic_validation": float(thr),
           "discrimination_source": "locked Z2 recipe (model_z_family arithmetic)", "probability_source": "Z2 probability_mean",
           **cse.threshold_metrics(y, prob, thr), **cse.calibration_params(y, prob)}
    for m, (lo, hi) in cse.bootstrap_core_binary_ci(y, disc, prob, reps, np.random.default_rng(cse.SEED + 900000 + ti)).items():
        row[f"{m}_ci95_low"], row[f"{m}_ci95_high"] = lo, hi
    for m, (lo, hi) in cse.bootstrap_threshold_ci(y, prob, thr, reps, np.random.default_rng(cse.SEED + 950000 + ti)).items():
        row[f"{m}_ci95_low"], row[f"{m}_ci95_high"] = lo, hi
    if cluster_reps > 0:
        row.update(cluster_boot(y, disc, groups, cluster_reps, 20260923 + ti))
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project-root", type=Path, default=Path.cwd())
    ap.add_argument("--bootstrap-reps", type=int, default=1000)
    ap.add_argument("--hospital-bootstrap-reps", type=int, default=1000, help="0 switches the hospital-cluster bootstrap off")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()

    root = a.project_root.resolve()
    os.chdir(root)
    sys.path.insert(0, str(root))
    os.environ["PYTHONPATH"] = str(root) + os.pathsep + os.environ.get("PYTHONPATH", "")
    cse = importlib.import_module("complete_statistical_evaluation")
    from joblib import Parallel, delayed

    zdir, inv, out = root / "outputs/model_z_family", root / "outputs/external_validation_inventory", root / "outputs/external_eicu_z2"
    stat = out / "statistical_analysis"
    recipes = {k: json.loads((zdir / f).read_text()) for k, f in RECIPE_FILES.items()}
    labels = pd.read_csv(inv / "eicu_labels_aligned_24h.csv")
    n = len(labels)

    cache: dict[str, dict] = {}

    def member(mid: str, task: str) -> np.ndarray:
        if mid not in cache:
            f = out / "members" / f"{mid}.npz"
            if not f.exists():
                raise FileNotFoundError(f"{f} is missing; the locked recipe for {task} needs member {mid}")
            cache[mid] = dict(np.load(f))
        z = cache[mid]
        if task == LOS:
            v = z[LOS].astype(float) if LOS in z else np.expm1(z["los_excess_log1p"].astype(float)) + 24.0
        else:
            v = z[task].astype(float)
            if mid not in TREE_MODELS:
                v = expit(v)
            if v.min() < 0 or v.max() > 1:
                raise ValueError(f"Member {mid}/{task} is not on the probability scale after conversion")
        if len(v) != n:
            raise ValueError(f"Member {mid}: {len(v)} rows, labels have {n}")
        return v

    print("Locked recipes")
    for t in TASKS + (LOS,):
        print(f"  {t:24s} Z2 weights {recipes['Z2'][t]['weights']} | blends: " + ", ".join(f"{p}={recipes[p][t].get('blend')}" for p in PARENTS))

    pred, prob = {}, {}
    for t in TASKS + (LOS,):
        comp = {p: apply_recipe(recipes[p][t], t, member) for p in PARENTS}
        w = np.asarray([recipes["Z2"][t]["weights"][p] for p in PARENTS], dtype=float)
        if t == LOS:
            pred[t] = np.column_stack([comp[p][0] for p in PARENTS]) @ w
        else:
            pred[t] = np.column_stack([rank01(comp[p][0]) for p in PARENTS]) @ w            # z2_apply_from_components
            prob[t] = np.clip(np.column_stack([comp[p][1] for p in PARENTS]) @ w, 0.0, 1.0)
    prob[DET] = 1.0 - np.prod(np.stack([1.0 - prob[t] for t in TASKS]), axis=0)
    pred[DET] = prob[DET]

    new = pd.DataFrame({"patientunitstayid": labels["patientunitstayid"]})
    for t in TASKS + (DET,):
        new[f"{t}_prediction"], new[f"{t}_probability_mean"] = pred[t], prob[t]
    new["los_prediction"] = pred[LOS]
    new.to_csv(out / "z2_tuned_eicu_predictions_locked_recipe.csv", index=False)

    # ---- what changed against the published canonical file
    old_path = out / "z2_tuned_eicu_predictions_canonical.csv"
    old = pd.read_csv(old_path) if old_path.exists() else None
    if old is not None:
        assert np.array_equal(old["patientunitstayid"].to_numpy(), new["patientunitstayid"].to_numpy()), "row order differs from the canonical file"
        print("\nCheck against the published canonical file (largest absolute difference)")
        for t in TASKS + (DET,):
            print(f"  {t:24s} probability_mean {np.max(np.abs(old[f'{t}_probability_mean'] - new[f'{t}_probability_mean'])):.3e}")
        print(f"  {'los_prediction':24s} hours            {np.max(np.abs(old['los_prediction'] - new['los_prediction'])):.3e}")

    # ---- thresholds locked on MIMIC-IV validation (as complete_statistical_evaluation does)
    zv = pd.read_csv(zdir / "Z2_tuned_validation_predictions.csv")
    zv = pd.concat([zv, cse.derive_z2_deterioration_rows(zv, "val")], ignore_index=True)
    thr = {}
    for t in TASKS + (DET,):
        sub = zv[zv["task"].eq(t)]
        yv, pv = pd.to_numeric(sub["target"], errors="coerce").to_numpy(float), pd.to_numeric(sub["probability_mean"], errors="coerce").to_numpy(float)
        keep = sub["validity"].astype(bool).to_numpy() & np.isfinite(yv) & np.isfinite(pv) & np.isin(yv, [0.0, 1.0])
        thr[t] = cse.threshold_at_max_f1(yv[keep].astype(int), pv[keep])

    hosp = pd.read_csv(inv / "eicu_cohort_skeleton.csv", usecols=["patientunitstayid", "hospitalid"])
    groups_all = labels[["patientunitstayid"]].merge(hosp, on="patientunitstayid", how="left", validate="one_to_one")["hospitalid"].to_numpy()
    if pd.isna(groups_all).any():
        raise ValueError("missing hospitalid")

    jobs, keeps = [], {}
    for t in TASKS + (DET,):
        y = labels[t].to_numpy(float)
        keep = np.isfinite(y) & np.isin(y, [0.0, 1.0]) & np.isfinite(pred[t]) & np.isfinite(prob[t])
        keeps[t] = keep
        jobs.append(delayed(binary_row)(str(root), t, y[keep].astype(int), pred[t][keep], prob[t][keep], thr[t], groups_all[keep],
                                        a.bootstrap_reps, a.hospital_bootstrap_reps))
    binary = pd.DataFrame(Parallel(n_jobs=a.jobs)(jobs))

    yl, pl = labels[LOS].to_numpy(float), pred[LOS]
    k = np.isfinite(yl) & np.isfinite(pl)
    from sklearn.metrics import r2_score
    e = pl[k] - yl[k]
    los = {"dataset": "eICU-CRD", "model": "Z2_tuned", "task": LOS, "n": int(k.sum()), "mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e ** 2))),
           "median_ae": float(np.median(np.abs(e))), "p90_ae": float(np.quantile(np.abs(e), 0.9)), "mean_error": float(np.mean(e)), "r2": float(r2_score(yl[k], pl[k]))}
    for m, (lo, hi) in cse.bootstrap_los_ci(yl[k], pl[k], a.bootstrap_reps, np.random.default_rng(cse.SEED + 970000)).items():
        los[f"{m}_ci95_low"], los[f"{m}_ci95_high"] = lo, hi

    stat.mkdir(parents=True, exist_ok=True)
    binary.to_csv(stat / "eicu_z2_locked_recipe_binary_statistics.csv", index=False)
    pd.DataFrame([los]).to_csv(stat / "eicu_z2_locked_recipe_los_statistics.csv", index=False)

    comp = binary[["task", "n", "events", "auroc", "auprc"]].rename(columns={"auroc": "auroc_locked", "auprc": "auprc_locked"})
    if old is not None:
        from sklearn.metrics import average_precision_score, roc_auc_score
        comp["auroc_published"] = [roc_auc_score(labels[t].to_numpy(float)[keeps[t]].astype(int), old[f"{t}_prediction"].to_numpy(float)[keeps[t]]) for t in comp["task"]]
        comp["auprc_published"] = [average_precision_score(labels[t].to_numpy(float)[keeps[t]].astype(int), old[f"{t}_prediction"].to_numpy(float)[keeps[t]]) for t in comp["task"]]
        comp["auroc_change"] = comp["auroc_locked"] - comp["auroc_published"]
        comp["auprc_change"] = comp["auprc_locked"] - comp["auprc_published"]
    comp.to_csv(stat / "eicu_z2_locked_recipe_vs_published.csv", index=False)

    with pd.option_context("display.width", 300, "display.max_columns", 60, "display.float_format", "{:.4f}".format):
        print("\nDISCRIMINATION: locked recipe against the published eICU numbers")
        print(comp.to_string(index=False))
        cols = ["task", "n", "events", "auroc", "auroc_ci95_low", "auroc_ci95_high", "auprc", "auprc_ci95_low", "auprc_ci95_high", "brier", "ece",
                "calibration_intercept", "calibration_slope", "threshold_locked_from_mimic_validation", "sensitivity", "specificity", "precision", "npv", "f1"]
        print("\nFULL STATISTICS (locked recipe, row bootstrap as in the master evaluation)")
        print(binary[cols].to_string(index=False))
        if a.hospital_bootstrap_reps > 0:
            print("\nHOSPITAL-CLUSTER 95% INTERVALS")
            print(binary[["task", "auroc", "auroc_hospital_ci95_low", "auroc_hospital_ci95_high", "auprc", "auprc_hospital_ci95_low", "auprc_hospital_ci95_high"]].to_string(index=False))
        print("\nLENGTH OF STAY (hours)")
        print(pd.DataFrame([los]).to_string(index=False))
    meta = {"dataset": "eICU-CRD", "model": "Z2_tuned", "retrained": False, "members_rescored": False,
            "ensemble_arithmetic": "model_z_family.apply_recipe + z2_apply_from_components (locked MIMIC-IV logic)",
            "z1_blend": "weighted mean on the probability scale", "z2_blend": "weighted mean of re-ranked parent scores",
            "neural_member_files": "logits, sigmoid applied", "tree_member_files": "probabilities",
            "thresholds": "locked on MIMIC-IV validation (max F1)", "bootstrap_reps": a.bootstrap_reps, "hospital_bootstrap_reps": a.hospital_bootstrap_reps,
            "readmission": "not evaluated on eICU (no valid label)"}
    (stat / "eicu_z2_locked_recipe_metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"\nwrote {out / 'z2_tuned_eicu_predictions_locked_recipe.csv'} and {stat}/eicu_z2_locked_recipe_*")


if __name__ == "__main__":
    main()
