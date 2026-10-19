#!/usr/bin/env python3
"""
Compact ROC and precision-recall curve points for Z2_tuned on the three datasets:
MIMIC-IV test, MIMIC-III CareVue (primary run) and eICU-CRD (locked recipe).

Nothing is re-scored. The script reads the saved Z2 predictions, computes the
exact ROC and PR curves, and keeps at most --points real curve points per curve
(the points nearest to an even grid on the x axis; no interpolation), so the
result is small enough for the master workbook.

    python build_z2_curve_package.py

Writes to outputs/curves_z2/ :
    z2_roc_pr_curve_points_compact.csv   dataset, task, curve, x, y
    z2_roc_pr_curve_summary.csv          n, events, AUROC, AUPRC per curve (for cross-checking)

The files contain curve coordinates only: no stay ids, no thresholds, no patient rows.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

PARTS = ("mortality_24h", "ventilation_onset_24h", "vasopressor_onset_24h")
TASKS = PARTS + ("readmission_30d_24h", "deterioration_24h")


def thin(x: np.ndarray, y: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Keep real curve points nearest to an even x grid, plus both ends; x must be sorted ascending."""
    if len(x) <= k:
        return x, y
    idx = np.unique(np.concatenate([[0, len(x) - 1], np.searchsorted(x, np.linspace(x[0], x[-1], k), side="left").clip(0, len(x) - 1)]))
    return x[idx], y[idx]


def curves(dataset: str, task: str, y: np.ndarray, score: np.ndarray, k: int) -> tuple[list[dict], dict]:
    y = y.astype(int)
    fpr, tpr, _ = roc_curve(y, score)
    prec, rec, _ = precision_recall_curve(y, score)
    order = np.argsort(rec, kind="mergesort")
    rec, prec = rec[order], prec[order]
    rows = []
    for name, (a, b) in (("roc", thin(fpr, tpr, k)), ("pr", thin(rec, prec, k))):
        rows += [{"dataset": dataset, "task": task, "curve": name, "x": float(u), "y": float(v)} for u, v in zip(a, b)]
    summary = {"dataset": dataset, "task": task, "n": int(len(y)), "events": int(y.sum()), "prevalence": float(y.mean()),
               "auroc": float(roc_auc_score(y, score)), "auprc": float(average_precision_score(y, score))}
    return rows, summary


def mimic4(root: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    d = pd.read_csv(root / "outputs/model_z_family/Z2_tuned_test_predictions.csv")
    out, parts = {}, {}
    for t in TASKS[:4]:
        z = d[d["task"].eq(t)].sort_values("stay_id")
        yv, valid = z["target"].to_numpy(float), z["validity"].astype(bool).to_numpy()
        keep = valid & np.isin(yv, [0.0, 1.0])
        out[t] = (yv[keep], z["prediction"].to_numpy(float)[keep])
        parts[t] = (z["stay_id"].to_numpy(), yv, valid, np.clip(z["probability_mean"].to_numpy(float), 0, 1))
    assert all(np.array_equal(parts[PARTS[0]][0], parts[t][0]) for t in PARTS)
    valid = np.all([parts[t][2] for t in PARTS], axis=0)
    y = 1.0 - np.prod([1.0 - parts[t][1] for t in PARTS], axis=0)
    p = 1.0 - np.prod([1.0 - parts[t][3] for t in PARTS], axis=0)
    keep = valid & np.isin(y, [0.0, 1.0])
    out["deterioration_24h"] = (y[keep], p[keep])
    return out


def mimic3(root: Path, run: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    f = root / "outputs/external_mimic3_z2" / run / "mimic3_predictions_long.csv"
    d = pd.read_csv(f, dtype={"model_id": str})
    d = d[d["model_id"].eq("Z2_tuned")]
    out = {}
    for t in TASKS:
        z = d[d["task"].eq(t)]
        yv = z["target"].to_numpy(float)
        keep = (z["validity"].to_numpy(int) == 1) & np.isin(yv, [0.0, 1.0])
        out[t] = (yv[keep], z["prediction"].to_numpy(float)[keep])
    return out


def eicu(root: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    pred = pd.read_csv(root / "outputs/external_eicu_z2/z2_tuned_eicu_predictions_locked_recipe.csv")
    lab = pd.read_csv(root / "outputs/external_validation_inventory/eicu_labels_aligned_24h.csv")
    assert np.array_equal(pred["patientunitstayid"].to_numpy(), lab["patientunitstayid"].to_numpy())
    out = {}
    for t in PARTS + ("deterioration_24h",):
        yv = lab[t].to_numpy(float)
        keep = np.isfinite(yv) & np.isin(yv, [0.0, 1.0])
        out[t] = (yv[keep], pred[f"{t}_prediction"].to_numpy(float)[keep])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project-root", type=Path, default=Path.cwd())
    ap.add_argument("--mimic3-run", default="external_mimic3_tempC__text_embeddings")
    ap.add_argument("--points", type=int, default=300)
    a = ap.parse_args()
    root = a.project_root.resolve()
    out = root / "outputs/curves_z2"
    out.mkdir(parents=True, exist_ok=True)

    rows, summ = [], []
    for name, data in (("MIMIC-IV test", mimic4(root)), ("MIMIC-III CareVue", mimic3(root, a.mimic3_run)), ("eICU-CRD", eicu(root))):
        for t, (y, s) in data.items():
            r, m = curves(name, t, y, s, a.points)
            rows += r
            summ.append(m)
    pd.DataFrame(rows).to_csv(out / "z2_roc_pr_curve_points_compact.csv", index=False, float_format="%.6f")
    s = pd.DataFrame(summ)
    s.to_csv(out / "z2_roc_pr_curve_summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print(s.to_string(index=False))
    print(f"\n{len(rows):,} curve points written to {out}")


if __name__ == "__main__":
    main()
