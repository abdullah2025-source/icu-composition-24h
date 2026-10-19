from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.linear_model import LogisticRegression

ROOT = Path("/home/abdullah2025/fyp_project")
EICU = ROOT / "outputs/external_validation_inventory"
Z2 = ROOT / "outputs/external_eicu_z2"
OUT = Z2 / "statistical_analysis"
OUT.mkdir(parents=True, exist_ok=True)

N_BOOT = 5000
SEED = 20260923

rng = np.random.default_rng(SEED)

labels = pd.read_csv(EICU / "eicu_labels_aligned_24h.csv")
cohort = pd.read_csv(
    EICU / "eicu_cohort_skeleton.csv",
    usecols=["patientunitstayid", "hospitalid"],
)

# ---------------------------------------------------------------------
# Load the already-computed locked Z2 predictions.
# ---------------------------------------------------------------------
# Replace these column names only if your saved final prediction file
# uses different names.
pred_file = Z2 / "z2_tuned_eicu_predictions.csv"

if pred_file.exists():
    pred = pd.read_csv(pred_file)
else:
    raise FileNotFoundError(
        f"Expected consolidated prediction file not found: {pred_file}\n"
        "Use the existing member predictions to create this file first."
    )

df = labels.merge(
    cohort,
    on="patientunitstayid",
    how="left",
    validate="one_to_one",
)

df = df.merge(
    pred,
    on="patientunitstayid",
    how="left",
    validate="one_to_one",
)

# ---------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------
def calibration_intercept(y, p):
    p = np.clip(np.asarray(p, float), 1e-7, 1 - 1e-7)
    y = np.asarray(y, float)

    x = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=2000,
    )
    lr.fit(x, y)

    return float(lr.intercept_[0])


def calibration_slope(y, p):
    p = np.clip(np.asarray(p, float), 1e-7, 1 - 1e-7)
    y = np.asarray(y, float)

    x = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=2000,
    )
    lr.fit(x, y)

    return float(lr.coef_[0, 0])


def bootstrap_ci(values, alpha=0.05):
    values = np.asarray(values, float)
    return (
        float(np.nanpercentile(values, 100 * alpha / 2)),
        float(np.nanpercentile(values, 100 * (1 - alpha / 2))),
    )


def hospital_cluster_bootstrap(
    d,
    metric_fn,
    n_boot=N_BOOT,
    seed=SEED,
):
    rng_local = np.random.default_rng(seed)

    hospitals = d["hospitalid"].dropna().unique()
    metric_values = []

    grouped = {
        h: g.index.to_numpy()
        for h, g in d.groupby("hospitalid", sort=False)
    }

    for _ in range(n_boot):
        sampled_hospitals = rng_local.choice(
            hospitals,
            size=len(hospitals),
            replace=True,
        )

        sampled_idx = np.concatenate(
            [grouped[h] for h in sampled_hospitals]
        )

        metric_values.append(
            metric_fn(d.loc[sampled_idx])
        )

    return np.asarray(metric_values, float)


# ---------------------------------------------------------------------
# Binary tasks
# ---------------------------------------------------------------------
binary_tasks = [
    "mortality_24h",
    "ventilation_onset_24h",
    "vasopressor_onset_24h",
    "deterioration_24h",
]

rows = []

for task in binary_tasks:
    pred_col = f"{task}_prediction"

    if pred_col not in df.columns:
        raise KeyError(f"Missing prediction column: {pred_col}")

    work = df[[task, pred_col, "hospitalid"]].copy()
    work = work[work[task] >= 0]
    work = work[np.isfinite(work[pred_col])]

    y = work[task].to_numpy(float)
    p = work[pred_col].to_numpy(float)

    def auroc_metric(x):
        return roc_auc_score(
            x[task],
            x[pred_col],
        )

    def auprc_metric(x):
        return average_precision_score(
            x[task],
            x[pred_col],
        )

    def brier_metric(x):
        return brier_score_loss(
            x[task],
            x[pred_col],
        )

    def cal_int_metric(x):
        return calibration_intercept(
            x[task],
            x[pred_col],
        )

    def cal_slope_metric(x):
        return calibration_slope(
            x[task],
            x[pred_col],
        )

    auroc_boot = hospital_cluster_bootstrap(work, auroc_metric)
    auprc_boot = hospital_cluster_bootstrap(work, auprc_metric)
    brier_boot = hospital_cluster_bootstrap(work, brier_metric)
    cint_boot = hospital_cluster_bootstrap(work, cal_int_metric)
    cslope_boot = hospital_cluster_bootstrap(work, cal_slope_metric)

    rows.append({
        "dataset": "eICU",
        "task": task,
        "n": len(work),
        "events": int(y.sum()),
        "event_rate": float(y.mean()),
        "auroc": float(roc_auc_score(y, p)),
        "auroc_ci_low": bootstrap_ci(auroc_boot)[0],
        "auroc_ci_high": bootstrap_ci(auroc_boot)[1],
        "auprc": float(average_precision_score(y, p)),
        "auprc_ci_low": bootstrap_ci(auprc_boot)[0],
        "auprc_ci_high": bootstrap_ci(auprc_boot)[1],
        "brier": float(brier_score_loss(y, p)),
        "brier_ci_low": bootstrap_ci(brier_boot)[0],
        "brier_ci_high": bootstrap_ci(brier_boot)[1],
        "calibration_intercept": calibration_intercept(y, p),
        "calibration_intercept_ci_low": bootstrap_ci(cint_boot)[0],
        "calibration_intercept_ci_high": bootstrap_ci(cint_boot)[1],
        "calibration_slope": calibration_slope(y, p),
        "calibration_slope_ci_low": bootstrap_ci(cslope_boot)[0],
        "calibration_slope_ci_high": bootstrap_ci(cslope_boot)[1],
    })

# ---------------------------------------------------------------------
# LOS
# ---------------------------------------------------------------------
los = df[["los_hours_24h", "los_prediction", "hospitalid"]].copy()
los = los[np.isfinite(los["los_hours_24h"])]
los = los[np.isfinite(los["los_prediction"])]

def los_mae(x):
    return mean_absolute_error(
        x["los_hours_24h"],
        x["los_prediction"],
    )

def los_rmse(x):
    return float(
        np.sqrt(
            mean_squared_error(
                x["los_hours_24h"],
                x["los_prediction"],
            )
        )
    )

def los_bias(x):
    return float(
        (
            x["los_prediction"].to_numpy(float)
            - x["los_hours_24h"].to_numpy(float)
        ).mean()
    )

mae_boot = hospital_cluster_bootstrap(los, los_mae)
rmse_boot = hospital_cluster_bootstrap(los, los_rmse)
bias_boot = hospital_cluster_bootstrap(los, los_bias)

err = (
    los["los_prediction"].to_numpy(float)
    - los["los_hours_24h"].to_numpy(float)
)

los_row = {
    "dataset": "eICU",
    "task": "los_hours_24h",
    "n": len(los),
    "mae": float(np.mean(np.abs(err))),
    "mae_ci_low": bootstrap_ci(mae_boot)[0],
    "mae_ci_high": bootstrap_ci(mae_boot)[1],
    "rmse": float(np.sqrt(np.mean(err ** 2))),
    "rmse_ci_low": bootstrap_ci(rmse_boot)[0],
    "rmse_ci_high": bootstrap_ci(rmse_boot)[1],
    "mean_error": float(err.mean()),
    "mean_error_ci_low": bootstrap_ci(bias_boot)[0],
    "mean_error_ci_high": bootstrap_ci(bias_boot)[1],
    "median_absolute_error": float(np.median(np.abs(err))),
    "p90_absolute_error": float(np.quantile(np.abs(err), 0.90)),
    "overprediction_rate": float((err > 0).mean()),
    "underprediction_rate": float((err < 0).mean()),
}

# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------
binary_out = pd.DataFrame(rows)
binary_out.to_csv(
    OUT / "eicu_z2_binary_statistical_analysis.csv",
    index=False,
)

pd.DataFrame([los_row]).to_csv(
    OUT / "eicu_z2_los_statistical_analysis.csv",
    index=False,
)

summary = {
    "model": "Z2_tuned",
    "dataset": "eICU-CRD",
    "bootstrap_replicates": N_BOOT,
    "bootstrap_unit": "hospital",
    "seed": SEED,
    "invalid_binary_label": -1,
    "los_transform": "expm1(predicted_excess_log1p) + 24.0",
    "no_external_refit": True,
    "no_external_tuning": True,
}

(OUT / "eicu_z2_statistical_analysis_metadata.json").write_text(
    json.dumps(summary, indent=2)
)

print("=" * 90)
print("eICU Z2_tuned STATISTICAL ANALYSIS COMPLETE")
print("=" * 90)
print(binary_out.to_string(index=False))
print()
print(pd.DataFrame([los_row]).to_string(index=False))
print()
print("Output directory:", OUT)
