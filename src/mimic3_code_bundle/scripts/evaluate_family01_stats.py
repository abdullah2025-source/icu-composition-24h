#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
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
BASE = PROJECT_ROOT / "outputs" / "baseline_family_01_tabular"
OUT_DIR = BASE / "stats"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BINARY_TASKS = {
    "mortality_24h",
    "ventilation_onset_24h",
    "vasopressor_onset_24h",
    "readmission_30d_24h",
    "deterioration_24h",
}

def _safe_auroc(y, s):
    return float("nan") if len(np.unique(y)) < 2 else float(roc_auc_score(y, s))

def _safe_auprc(y, s):
    return float("nan") if len(np.unique(y)) < 2 else float(average_precision_score(y, s))

def _percentile_ci(vals, alpha=0.05):
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2)))

def bootstrap_classification(y_true, y_score, n_boot=1000, seed=20260923):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    mask = np.isfinite(y_true) & np.isfinite(y_score)
    y_true, y_score = y_true[mask], y_score[mask]
    n = len(y_true)
    if n < 10:
        return {k: float("nan") for k in [
            "auroc","auroc_ci_low","auroc_ci_high",
            "auprc","auprc_ci_low","auprc_ci_high",
            "brier","brier_ci_low","brier_ci_high","n","prevalence"
        ]}
    rng = np.random.default_rng(seed)
    auroc_p = _safe_auroc(y_true, y_score)
    auprc_p = _safe_auprc(y_true, y_score)
    brier_p = float(brier_score_loss(y_true, y_score))
    aurocs = np.empty(n_boot); auprcs = np.empty(n_boot); briers = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        aurocs[i] = _safe_auroc(y_true[idx], y_score[idx])
        auprcs[i] = _safe_auprc(y_true[idx], y_score[idx])
        briers[i] = brier_score_loss(y_true[idx], y_score[idx])
    alo, ahi = _percentile_ci(aurocs); plo, phi = _percentile_ci(auprcs); blo, bhi = _percentile_ci(briers)
    return {
        "auroc": auroc_p, "auroc_ci_low": alo, "auroc_ci_high": ahi,
        "auprc": auprc_p, "auprc_ci_low": plo, "auprc_ci_high": phi,
        "brier": brier_p, "brier_ci_low": blo, "brier_ci_high": bhi,
        "n": int(n), "prevalence": float(np.mean(y_true)),
    }

def bootstrap_regression(y_true, y_pred, n_boot=1000, seed=20260923):
    y_true = np.asarray(y_true).astype(float)
    y_pred = np.asarray(y_pred).astype(float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    n = len(y_true)
    if n < 10:
        return {k: float("nan") for k in [
            "mae","mae_ci_low","mae_ci_high",
            "rmse","rmse_ci_low","rmse_ci_high",
            "median_ae","median_ae_ci_low","median_ae_ci_high",
            "r2","r2_ci_low","r2_ci_high","n"
        ]}
    rng = np.random.default_rng(seed)
    mae_p = float(mean_absolute_error(y_true, y_pred))
    rmse_p = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    med_p = float(median_absolute_error(y_true, y_pred))
    r2_p = float(r2_score(y_true, y_pred)) if n > 1 else float("nan")
    maes = np.empty(n_boot); rmses = np.empty(n_boot); meds = np.empty(n_boot); r2s = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        maes[i] = mean_absolute_error(y_true[idx], y_pred[idx])
        rmses[i] = np.sqrt(mean_squared_error(y_true[idx], y_pred[idx]))
        meds[i] = median_absolute_error(y_true[idx], y_pred[idx])
        r2s[i] = r2_score(y_true[idx], y_pred[idx]) if len(idx) > 1 else float("nan")
    mlo, mhi = _percentile_ci(maes); rlo, rhi = _percentile_ci(rmses); dlo, dhi = _percentile_ci(meds); r2lo, r2hi = _percentile_ci(r2s)
    return {
        "mae": mae_p, "mae_ci_low": mlo, "mae_ci_high": mhi,
        "rmse": rmse_p, "rmse_ci_low": rlo, "rmse_ci_high": rhi,
        "median_ae": med_p, "median_ae_ci_low": dlo, "median_ae_ci_high": dhi,
        "r2": r2_p, "r2_ci_low": r2lo, "r2_ci_high": r2hi, "n": int(n),
    }

def expected_calibration_error(y_true, y_score, n_bins=10):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    mask = np.isfinite(y_true) & np.isfinite(y_score)
    y_true, y_score = y_true[mask], y_score[mask]
    if len(y_true) == 0:
        return float("nan")
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        in_bin = ((y_score >= lo) & (y_score < hi)) if i < n_bins - 1 else ((y_score >= lo) & (y_score <= hi))
        if not np.any(in_bin):
            continue
        ece += float(np.mean(in_bin)) * abs(float(np.mean(y_true[in_bin])) - float(np.mean(y_score[in_bin])))
    return float(ece)

def evaluate_one_mode(mode: str):
    mode_dir = BASE / mode
    if not mode_dir.exists():
        print(f"[SKIP] {mode_dir} does not exist")
        return []
    rows = []
    model_dirs = sorted([p for p in mode_dir.iterdir() if p.is_dir()])
    for model_dir in model_dirs:
        pred_path = (model_dir / "final_selected_configuration" / "test_predictions.csv") if mode == "tuned" else (model_dir / "test_predictions.csv")
        name = model_dir.name
        if not pred_path.exists():
            print(f"[SKIP] No predictions: {pred_path}")
            continue
        print(f"Evaluating {mode} / {name} ...")
        df = pd.read_csv(pred_path)
        if "split" in df.columns:
            df = df[df["split"] == "test"].copy()
        model_id = name.split("_")[0]
        for task, g in df.groupby("task"):
            if "validity" in g.columns:
                g = g[g["validity"] == 1].copy()
            y = g["target"].to_numpy()
            p = g["prediction"].to_numpy()
            row = {"mode": mode, "model_id": model_id, "model_name": name, "task": task}
            if task in BINARY_TASKS:
                row.update(bootstrap_classification(y, p, n_boot=1000))
                row["ece"] = expected_calibration_error(y, p)
            else:
                row.update(bootstrap_regression(y, p, n_boot=1000))
                row["ece"] = float("nan")
            rows.append(row)
    return rows

def main():
    all_rows = evaluate_one_mode("fixed") + evaluate_one_mode("tuned")
    if not all_rows:
        print("No results found.")
        return
    result = pd.DataFrame(all_rows)
    out_csv = OUT_DIR / "family01_all_metrics_with_stats.csv"
    result.to_csv(out_csv, index=False)
    print("\n" + "=" * 70)
    print(f"Saved → {out_csv}")
    print("=" * 70)
    print(result.head(15).to_string(index=False))

if __name__ == "__main__":
    main()
