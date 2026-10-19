from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
)

ROOT = Path("/home/abdullah2025/fyp_project")
INV = ROOT / "outputs/external_validation_inventory"
OUT = ROOT / "outputs/external_eicu_z2"
STAT = OUT / "statistical_analysis"
STAT.mkdir(parents=True, exist_ok=True)

SEED = 20260923
N_BOOT = 5000
rng = np.random.default_rng(SEED)

labels = pd.read_csv(INV / "eicu_labels_aligned_24h.csv")
cohort = pd.read_csv(
    INV / "eicu_cohort_skeleton.csv",
    usecols=["patientunitstayid", "hospitalid"],
)
pred = pd.read_csv(OUT / "z2_tuned_eicu_predictions.csv")

# Strict key-based alignment
df = (
    labels
    .merge(cohort, on="patientunitstayid", how="left", validate="one_to_one")
    .merge(pred, on="patientunitstayid", how="left", validate="one_to_one")
)

if len(df) != len(labels):
    raise RuntimeError("Prediction/label alignment changed row count.")

if df["hospitalid"].isna().any():
    raise RuntimeError("Missing hospitalid after alignment.")

# The binary Z2 outputs are ranking scores, not calibrated probabilities.
binary_score_cols = {
    "mortality_24h": "mortality_24h_score",
    "ventilation_onset_24h": "ventilation_onset_24h_score",
    "vasopressor_onset_24h": "vasopressor_onset_24h_score",
    "deterioration_24h": "deterioration_24h_score",
}

for c in binary_score_cols.values():
    if c not in df.columns:
        raise KeyError(f"Missing prediction column: {c}")

if "los_prediction" not in df.columns:
    raise KeyError("Missing prediction column: los_prediction")


def hospital_cluster_bootstrap(data, metric_fn):
    groups = {
        h: g.index.to_numpy()
        for h, g in data.groupby("hospitalid", sort=False)
    }
    hospitals = np.asarray(list(groups.keys()), dtype=object)

    boot = np.empty(N_BOOT, dtype=float)

    for b in range(N_BOOT):
        sampled = rng.choice(
            hospitals,
            size=len(hospitals),
            replace=True,
        )
        idx = np.concatenate([groups[h] for h in sampled])
        boot[b] = float(metric_fn(data.loc[idx]))

    return boot


def ci95(values):
    return (
        float(np.percentile(values, 2.5)),
        float(np.percentile(values, 97.5)),
    )


# ============================================================
# BINARY TASKS
# ============================================================
binary_rows = []
bootstrap_rows = []

for task, pred_col in binary_score_cols.items():
    work = df[
        ["patientunitstayid", "hospitalid", task, pred_col]
    ].copy()

    # eICU invalid binary labels are encoded as -1.
    work = work[
        np.isfinite(work[task])
        & (work[task] >= 0)
        & np.isfinite(work[pred_col])
    ].reset_index(drop=True)

    y = work[task].to_numpy(float)
    score = work[pred_col].to_numpy(float)

    if len(np.unique(y)) != 2:
        raise RuntimeError(f"{task}: bootstrap sample has invalid class structure.")

    auroc = roc_auc_score(y, score)
    auprc = average_precision_score(y, score)

    auroc_boot = hospital_cluster_bootstrap(
        work,
        lambda x: roc_auc_score(x[task], x[pred_col])
        if len(np.unique(x[task])) == 2
        else np.nan,
    )

    auprc_boot = hospital_cluster_bootstrap(
        work,
        lambda x: average_precision_score(x[task], x[pred_col])
        if len(np.unique(x[task])) == 2
        else np.nan,
    )

    auroc_boot = auroc_boot[np.isfinite(auroc_boot)]
    auprc_boot = auprc_boot[np.isfinite(auprc_boot)]

    auroc_ci = ci95(auroc_boot)
    auprc_ci = ci95(auprc_boot)

    binary_rows.append({
        "dataset": "eICU-CRD",
        "model": "Z2_tuned",
        "task": task,
        "valid_n": int(len(work)),
        "events": int(y.sum()),
        "event_rate": float(y.mean()),
        "auroc": float(auroc),
        "auroc_ci_low": auroc_ci[0],
        "auroc_ci_high": auroc_ci[1],
        "auprc": float(auprc),
        "auprc_ci_low": auprc_ci[0],
        "auprc_ci_high": auprc_ci[1],
        "bootstrap_replicates_requested": N_BOOT,
        "bootstrap_replicates_valid_auroc": int(len(auroc_boot)),
        "bootstrap_replicates_valid_auprc": int(len(auprc_boot)),
        "bootstrap_unit": "hospital",
        "prediction_scale": "ensemble_ranking_score",
    })

    for i, v in enumerate(auroc_boot):
        bootstrap_rows.append({
            "task": task,
            "metric": "AUROC",
            "replicate": i,
            "value": float(v),
        })

    for i, v in enumerate(auprc_boot):
        bootstrap_rows.append({
            "task": task,
            "metric": "AUPRC",
            "replicate": i,
            "value": float(v),
        })


# ============================================================
# LOS
# ============================================================
los = df[
    ["patientunitstayid", "hospitalid", "los_hours_24h", "los_prediction"]
].copy()

los = los[
    np.isfinite(los["los_hours_24h"])
    & np.isfinite(los["los_prediction"])
].reset_index(drop=True)

def mae_fn(x):
    return mean_absolute_error(x["los_hours_24h"], x["los_prediction"])

def rmse_fn(x):
    return float(
        np.sqrt(
            mean_squared_error(
                x["los_hours_24h"],
                x["los_prediction"],
            )
        )
    )

def bias_fn(x):
    return float(
        (
            x["los_prediction"].to_numpy(float)
            - x["los_hours_24h"].to_numpy(float)
        ).mean()
    )

mae_boot = hospital_cluster_bootstrap(los, mae_fn)
rmse_boot = hospital_cluster_bootstrap(los, rmse_fn)
bias_boot = hospital_cluster_bootstrap(los, bias_fn)

err = (
    los["los_prediction"].to_numpy(float)
    - los["los_hours_24h"].to_numpy(float)
)

mae = float(np.mean(np.abs(err)))
rmse = float(np.sqrt(np.mean(err ** 2)))
bias = float(err.mean())

mae_ci = ci95(mae_boot)
rmse_ci = ci95(rmse_boot)
bias_ci = ci95(bias_boot)

los_row = {
    "dataset": "eICU-CRD",
    "model": "Z2_tuned",
    "task": "los_hours_24h",
    "valid_n": int(len(los)),
    "mae": mae,
    "mae_ci_low": mae_ci[0],
    "mae_ci_high": mae_ci[1],
    "rmse": rmse,
    "rmse_ci_low": rmse_ci[0],
    "rmse_ci_high": rmse_ci[1],
    "mean_error": bias,
    "mean_error_ci_low": bias_ci[0],
    "mean_error_ci_high": bias_ci[1],
    "median_absolute_error": float(np.median(np.abs(err))),
    "p90_absolute_error": float(np.quantile(np.abs(err), 0.90)),
    "overprediction_rate": float((err > 0).mean()),
    "underprediction_rate": float((err < 0).mean()),
    "bootstrap_replicates_requested": N_BOOT,
    "bootstrap_unit": "hospital",
    "reporting_transform": "expm1(predicted_excess_log1p) + 24.0 hours",
}

for i, v in enumerate(mae_boot):
    bootstrap_rows.append({
        "task": "los_hours_24h",
        "metric": "MAE",
        "replicate": i,
        "value": float(v),
    })

for i, v in enumerate(rmse_boot):
    bootstrap_rows.append({
        "task": "los_hours_24h",
        "metric": "RMSE",
        "replicate": i,
        "value": float(v),
    })

for i, v in enumerate(bias_boot):
    bootstrap_rows.append({
        "task": "los_hours_24h",
        "metric": "MeanError",
        "replicate": i,
        "value": float(v),
    })


# ============================================================
# SAVE
# ============================================================
binary_out = pd.DataFrame(binary_rows)
los_out = pd.DataFrame([los_row])
boot_out = pd.DataFrame(bootstrap_rows)

binary_out.to_csv(
    STAT / "eicu_z2_discrimination_bootstrap_cis.csv",
    index=False,
)

los_out.to_csv(
    STAT / "eicu_z2_los_bootstrap_cis.csv",
    index=False,
)

boot_out.to_csv(
    STAT / "eicu_z2_bootstrap_distributions.csv",
    index=False,
)

metadata = {
    "dataset": "eICU-CRD",
    "model": "Z2_tuned",
    "bootstrap_replicates": N_BOOT,
    "bootstrap_unit": "hospital",
    "seed": SEED,
    "invalid_binary_label": -1,
    "los_transform": "expm1(predicted_excess_log1p) + 24.0",
    "external_refit": False,
    "external_tuning": False,
    "calibration_metrics_reported": False,
    "calibration_reason": (
        "Locked Z2 binary outputs are ranking scores rather than "
        "calibrated 0-1 probabilities. Brier score and calibration "
        "slope/intercept are therefore not computed from the raw score."
    ),
}

(STAT / "eicu_z2_inferential_analysis_metadata.json").write_text(
    json.dumps(metadata, indent=2)
)

print("=" * 90)
print("eICU Z2_tuned INFERENTIAL ANALYSIS")
print("=" * 90)

print("\nBINARY TASKS")
print(binary_out.to_string(index=False))

print("\nLOS")
print(los_out.to_string(index=False))

print("\nBootstrap distributions:", len(boot_out), "rows")
print("Output directory:", STAT)
