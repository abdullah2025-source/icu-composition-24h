from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

ROOT = Path("/home/abdullah2025/fyp_project")
MODEL_ROOT = ROOT / "outputs/model_z_family"
OUT = ROOT / "outputs/external_eicu_z2"
STAT = OUT / "statistical_analysis"
INV = ROOT / "outputs/external_validation_inventory"

STAT.mkdir(parents=True, exist_ok=True)

SEED = 20260923
N_BOOT = 5000
rng = np.random.default_rng(SEED)

TASKS = [
    "mortality_24h",
    "ventilation_onset_24h",
    "vasopressor_onset_24h",
    "deterioration_24h",
]

EICU_SCORE_COLUMNS = {
    "mortality_24h": "mortality_24h_score",
    "ventilation_onset_24h": "ventilation_onset_24h_score",
    "vasopressor_onset_24h": "vasopressor_onset_24h_score",
    "deterioration_24h": "deterioration_24h_score",
}

# ------------------------------------------------------------
# MIMIC-IV validation predictions
# ------------------------------------------------------------
mimic_path = (
    MODEL_ROOT / "Z2_tuned_validation_predictions.csv"
)

mimic = pd.read_csv(mimic_path)

required_mimic = {
    "stay_id",
    "task",
    "target",
    "validity",
    "prediction",
    "probability_mean",
}

missing = required_mimic - set(mimic.columns)
if missing:
    raise RuntimeError(
        f"MIMIC validation file missing columns: {sorted(missing)}"
    )

print("MIMIC validation file:", mimic_path)
print("shape:", mimic.shape)

# ------------------------------------------------------------
# eICU
# ------------------------------------------------------------
eicu_pred = pd.read_csv(
    OUT / "z2_tuned_eicu_predictions.csv"
)

eicu_labels = pd.read_csv(
    INV / "eicu_labels_aligned_24h.csv"
)

hospital = pd.read_csv(
    INV / "eicu_cohort_skeleton.csv",
    usecols=["patientunitstayid", "hospitalid"],
)

eicu = (
    eicu_labels
    .merge(
        hospital,
        on="patientunitstayid",
        how="left",
        validate="one_to_one",
    )
    .merge(
        eicu_pred,
        on="patientunitstayid",
        how="left",
        validate="one_to_one",
    )
)

if eicu["hospitalid"].isna().any():
    raise RuntimeError("Missing eICU hospital IDs.")

# ------------------------------------------------------------
# Within-cohort percentile transformation
#
# This creates a dimensionless ranking coordinate so that
# MIMIC's validation ranking and eICU's external ranking do not
# depend on different cohort sizes.
# ------------------------------------------------------------
def percentile_score(values):
    values = np.asarray(values, dtype=float)
    ranks = pd.Series(values).rank(method="average").to_numpy(float)
    return (ranks - 0.5) / len(values)


# ------------------------------------------------------------
# Frozen calibration models
# ------------------------------------------------------------
calibration = {}
results = []

for task in TASKS:

    # ---------------------------
    # MIMIC validation subset
    # ---------------------------
    m = mimic[mimic["task"].eq(task)].copy()

    m = m[
        (m["validity"] == 1)
        & np.isfinite(m["target"])
        & np.isfinite(m["prediction"])
    ].copy()

    if len(m) == 0:
        raise RuntimeError(f"No valid MIMIC validation rows for {task}")

    y_m = m["target"].to_numpy(float)

    if not np.isin(np.unique(y_m), [0.0, 1.0]).all():
        raise RuntimeError(f"Unexpected MIMIC labels for {task}")

    # Raw Z2 ranking -> percentile
    pct_m = percentile_score(
        m["prediction"].to_numpy(float)
    )

    lp_m = logit(
        np.clip(pct_m, 1e-5, 1 - 1e-5)
    ).reshape(-1, 1)

    # Frozen Platt/logistic calibration
    model = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=5000,
    )
    model.fit(lp_m, y_m)

    intercept = float(model.intercept_[0])
    slope = float(model.coef_[0, 0])

    calibration[task] = {
        "intercept": intercept,
        "slope": slope,
        "mimic_valid_n": int(len(m)),
        "mimic_events": int(y_m.sum()),
    }

    # ---------------------------
    # eICU
    # ---------------------------
    score_col = EICU_SCORE_COLUMNS[task]

    work = eicu[
        [
            "patientunitstayid",
            "hospitalid",
            task,
            score_col,
        ]
    ].copy()

    work = work[
        (work[task] >= 0)
        & np.isfinite(work[task])
        & np.isfinite(work[score_col])
    ].reset_index(drop=True)

    y_e = work[task].to_numpy(float)
    score_e = work[score_col].to_numpy(float)

    pct_e = percentile_score(score_e)

    lp_e = logit(
        np.clip(pct_e, 1e-5, 1 - 1e-5)
    ).reshape(-1, 1)

    p_e = model.predict_proba(lp_e)[:, 1]

    # Sanity
    if not ((p_e >= 0).all() and (p_e <= 1).all()):
        raise RuntimeError(f"Invalid calibrated probability for {task}")

    # ---------------------------
    # External Brier score
    # ---------------------------
    brier = float(
        brier_score_loss(y_e, p_e)
    )

    # ---------------------------
    # External calibration intercept/slope
    # ---------------------------
    observed_cal = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=5000,
    )

    observed_cal.fit(
        logit(
            np.clip(p_e, 1e-7, 1 - 1e-7)
        ).reshape(-1, 1),
        y_e,
    )

    ext_intercept = float(observed_cal.intercept_[0])
    ext_slope = float(observed_cal.coef_[0, 0])

    # ---------------------------
    # Hospital-cluster bootstrap
    # ---------------------------
    groups = {
        h: g.index.to_numpy()
        for h, g in work.groupby("hospitalid", sort=False)
    }

    hospitals = np.asarray(
        list(groups.keys()),
        dtype=object,
    )

    boot_brier = []
    boot_intercept = []
    boot_slope = []

    for _ in range(N_BOOT):

        sampled = rng.choice(
            hospitals,
            size=len(hospitals),
            replace=True,
        )

        idx = np.concatenate(
            [groups[h] for h in sampled]
        )

        yb = y_e[idx]
        pb = p_e[idx]

        boot_brier.append(
            brier_score_loss(yb, pb)
        )

        # Require both classes for calibration fit.
        if len(np.unique(yb)) == 2:

            cal_b = LogisticRegression(
                penalty=None,
                solver="lbfgs",
                max_iter=5000,
            )

            cal_b.fit(
                logit(
                    np.clip(pb, 1e-7, 1 - 1e-7)
                ).reshape(-1, 1),
                yb,
            )

            boot_intercept.append(
                float(cal_b.intercept_[0])
            )

            boot_slope.append(
                float(cal_b.coef_[0, 0])
            )

    boot_brier = np.asarray(boot_brier)
    boot_intercept = np.asarray(boot_intercept)
    boot_slope = np.asarray(boot_slope)

    def ci(x):
        return (
            float(np.percentile(x, 2.5)),
            float(np.percentile(x, 97.5)),
        )

    brier_ci = ci(boot_brier)
    int_ci = ci(boot_intercept)
    slope_ci = ci(boot_slope)

    results.append({
        "dataset": "eICU-CRD",
        "model": "Z2_tuned",
        "task": task,
        "valid_n": int(len(work)),
        "events": int(y_e.sum()),
        "event_rate": float(y_e.mean()),

        "mimic_validation_n": int(len(m)),
        "mimic_validation_events": int(y_m.sum()),

        "frozen_calibration_intercept": intercept,
        "frozen_calibration_slope": slope,

        "eicu_brier_score": brier,
        "eicu_brier_ci_low": brier_ci[0],
        "eicu_brier_ci_high": brier_ci[1],

        "eicu_calibration_intercept": ext_intercept,
        "eicu_calibration_intercept_ci_low": int_ci[0],
        "eicu_calibration_intercept_ci_high": int_ci[1],

        "eicu_calibration_slope": ext_slope,
        "eicu_calibration_slope_ci_low": slope_ci[0],
        "eicu_calibration_slope_ci_high": slope_ci[1],

        "bootstrap_replicates": N_BOOT,
        "bootstrap_unit": "hospital",
        "external_refit": False,
        "calibration_training_data": "MIMIC-IV validation",
        "score_normalization": "within-cohort percentile rank",
        "calibration_method": "logistic/Platt mapping",
    })


result = pd.DataFrame(results)

result.to_csv(
    STAT / "eicu_z2_frozen_calibration.csv",
    index=False,
)

metadata = {
    "model": "Z2_tuned",
    "dataset": "eICU-CRD",
    "calibration_training_dataset": "MIMIC-IV validation",
    "external_refit": False,
    "external_tuning": False,
    "method": "logistic/Platt calibration",
    "raw_score_type": "ensemble ranking score",
    "score_normalization": "within-cohort percentile rank",
    "bootstrap_replicates": N_BOOT,
    "bootstrap_unit": "hospital",
    "seed": SEED,
    "metrics": [
        "Brier score",
        "calibration intercept",
        "calibration slope",
    ],
    "mimic_probability_mean_available": True,
    "mimic_probability_mean_used_for_calibration": False,
    "note": (
        "The MIMIC validation file contains probability_mean, but this "
        "analysis calibrates the locked ranking signal itself so that the "
        "external eICU score can be evaluated under the same frozen mapping."
    ),
}

(
    STAT / "eicu_z2_frozen_calibration_metadata.json"
).write_text(
    json.dumps(metadata, indent=2)
)

print("=" * 100)
print("eICU Z2_tuned FROZEN CALIBRATION COMPLETE")
print("=" * 100)
print(result.to_string(index=False))
print()
print("Output:")
print(STAT / "eicu_z2_frozen_calibration.csv")
