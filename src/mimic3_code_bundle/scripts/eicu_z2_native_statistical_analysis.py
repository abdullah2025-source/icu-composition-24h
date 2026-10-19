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
MODEL_ROOT = ROOT / "outputs/model_z_family"
MEMBERS = ROOT / "outputs/external_eicu_z2/members"
INV = ROOT / "outputs/external_validation_inventory"
OUT = ROOT / "outputs/external_eicu_z2"
STAT = OUT / "statistical_analysis"
STAT.mkdir(parents=True, exist_ok=True)

BOOTSTRAP_REPS = 1000
SEED = 20260923
rng = np.random.default_rng(SEED)

labels = pd.read_csv(INV / "eicu_labels_aligned_24h.csv")

recipes = {
    "A2": json.loads((MODEL_ROOT / "A2_recipes.json").read_text()),
    "A3": json.loads((MODEL_ROOT / "A3_recipes.json").read_text()),
    "Z1": json.loads((MODEL_ROOT / "Z1_tuned_recipes.json").read_text()),
    "Z2": json.loads((MODEL_ROOT / "Z2_tuned_recipes.json").read_text()),
}

from scipy.special import expit

TREE_MODELS = {"00", "05"}

def member_array(model_id, task):
    path = MEMBERS / f"{model_id}.npz"
    if not path.exists():
        raise FileNotFoundError(path)

    z = np.load(path)

    if task not in z:
        raise KeyError(
            f"{task} missing from {path}. "
            f"Available keys: {z.files}"
        )

    value = z[task].astype(float)

    # 00/05 were saved with predict_proba().
    # Neural members were saved from direct model.forward() output,
    # i.e. binary logits. The canonical family probability path
    # requires sigmoid(logit) before probability_mean blending.
    if task in {
        "mortality_24h",
        "ventilation_onset_24h",
        "vasopressor_onset_24h",
        "readmission_30d_24h",
    } and model_id not in TREE_MODELS:
        value = expit(value)

    return value


def member_los(model_id):
    path = MEMBERS / f"{model_id}.npz"
    z = np.load(path)

    if "los_hours_24h" in z:
        return z["los_hours_24h"].astype(float)

    return (
        np.expm1(z["los_excess_log1p"].astype(float))
        + 24.0
    )


def weighted_mean(parts):
    total = sum(w for _, w in parts)
    out = None

    for value, weight in parts:
        out = (
            weight * value
            if out is None
            else out + weight * value
        )

    return out / total


def rank_mean(parts):
    ranked = []

    for value, weight in parts:
        r = (
            pd.Series(value)
            .rank(method="average")
            .to_numpy(float)
            / len(value)
        )
        ranked.append((r, weight))

    return weighted_mean(ranked)


def build_parent(parent, task):
    spec = recipes[parent][task]

    members = [
        (mid, w)
        for mid, w in spec["weights"].items()
        if w > 0
    ]

    raw_parts = [
        (member_array(mid, task), w)
        for mid, w in members
    ]

    # Canonical family logic:
    # prediction is rank_mean for A3/Z1 and raw_mean for A2,
    # while probability_mean is always raw probability averaging
    # for binary tasks.
    if spec.get("blend", "rank_mean") == "rank_mean":
        pred = rank_mean(raw_parts)
    elif spec.get("blend") == "weighted_rank_mean":
        pred = rank_mean(raw_parts)
    else:
        pred = weighted_mean(raw_parts)

    prob = np.clip(
        weighted_mean(raw_parts),
        0.0,
        1.0,
    )

    return pred, prob


def build_z2_binary(task):
    # Build A3, A2, Z1 exactly from locked recipes.
    a3_pred, a3_prob = build_parent("A3", task)
    a2_pred, a2_prob = build_parent("A2", task)
    z1_pred, z1_prob = build_parent("Z1", task)

    w = recipes["Z2"][task]["weights"]

    # Canonical Z2 rank-space prediction.
    pred = (
        w["A3"] * a3_pred
        + w["A2"] * a2_pred
        + w["Z1"] * z1_pred
    )

    # Canonical Z2 probability_mean.
    prob = np.clip(
        w["A3"] * a3_prob
        + w["A2"] * a2_prob
        + w["Z1"] * z1_prob,
        0.0,
        1.0,
    )

    return pred, prob


# ------------------------------------------------------------------
# ECE: same 10-bin concept used by the family statistical layer.
# ------------------------------------------------------------------
def expected_calibration_error(y, p, bins=10):
    y = np.asarray(y, float)
    p = np.asarray(p, float)

    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0

    for i in range(bins):
        lo = edges[i]
        hi = edges[i + 1]

        if i == 0:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p > lo) & (p <= hi)

        if not np.any(mask):
            continue

        weight = mask.mean()
        mean_pred = p[mask].mean()
        mean_obs = y[mask].mean()

        ece += weight * abs(mean_pred - mean_obs)

    return float(ece)


# ------------------------------------------------------------------
# Bootstrap CI
# ------------------------------------------------------------------
def bootstrap_metric(y, p, metric_fn):
    n = len(y)
    values = np.empty(BOOTSTRAP_REPS, dtype=float)

    for b in range(BOOTSTRAP_REPS):
        idx = rng.integers(0, n, size=n)

        try:
            values[b] = metric_fn(
                y[idx],
                p[idx],
            )
        except Exception:
            values[b] = np.nan

    values = values[np.isfinite(values)]

    return (
        float(np.percentile(values, 2.5)),
        float(np.percentile(values, 97.5)),
        int(len(values)),
    )


def calibration_parameters(y, p):
    lp = np.log(
        np.clip(p, 1e-7, 1 - 1e-7)
        /
        np.clip(1 - p, 1e-7, 1 - 1e-7)
    ).reshape(-1, 1)

    model = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=5000,
    )

    model.fit(lp, y)

    return (
        float(model.intercept_[0]),
        float(model.coef_[0, 0]),
    )


# ------------------------------------------------------------------
# Build exact Z2 outputs
# ------------------------------------------------------------------
tasks = [
    "mortality_24h",
    "ventilation_onset_24h",
    "vasopressor_onset_24h",
]

predictions = {
    "patientunitstayid": labels["patientunitstayid"]
}

for task in tasks:
    pred, prob = build_z2_binary(task)

    predictions[f"{task}_prediction"] = pred
    predictions[f"{task}_probability_mean"] = prob

# Deterioration uses the canonical probability composition.
p_m = predictions["mortality_24h_probability_mean"]
p_v = predictions["ventilation_onset_24h_probability_mean"]
p_p = predictions["vasopressor_onset_24h_probability_mean"]

deterioration_prob = (
    1.0
    - (1.0 - p_m)
    * (1.0 - p_v)
    * (1.0 - p_p)
)

predictions["deterioration_24h_probability_mean"] = np.clip(
    deterioration_prob,
    0.0,
    1.0,
)

# Deterioration has no independent raw rank recipe in the locked Z2
# construction; the canonical statistical score is this probability.
predictions["deterioration_24h_prediction"] = (
    predictions["deterioration_24h_probability_mean"]
)

# LOS: exact locked reporting coordinate.
los_parts = []

for parent in ["A2", "A3", "Z1"]:
    spec = recipes[parent]["los_hours_24h"]

    parts = [
        (member_los(mid), w)
        for mid, w in spec["weights"].items()
        if w > 0
    ]

    parent_los = weighted_mean(parts)

    los_parts.append(
        (
            parent_los,
            recipes["Z2"]["los_hours_24h"]["weights"][parent],
        )
    )

predictions["los_prediction"] = weighted_mean(los_parts)

pred_df = pd.DataFrame(predictions)

# Integrity checks.
for c in pred_df.columns[1:]:
    if not np.isfinite(pred_df[c]).all():
        raise RuntimeError(f"Non-finite output: {c}")

for c in [
    "mortality_24h_probability_mean",
    "ventilation_onset_24h_probability_mean",
    "vasopressor_onset_24h_probability_mean",
    "deterioration_24h_probability_mean",
]:
    if not ((pred_df[c] >= 0).all() and (pred_df[c] <= 1).all()):
        raise RuntimeError(
            f"Invalid probability values in {c}: "
            f"{pred_df[c].min()} to {pred_df[c].max()}"
        )

pred_path = OUT / "z2_tuned_eicu_predictions_canonical.csv"
pred_df.to_csv(pred_path, index=False)


# ------------------------------------------------------------------
# Binary statistical analysis
# ------------------------------------------------------------------
binary_rows = []

for task in [
    "mortality_24h",
    "ventilation_onset_24h",
    "vasopressor_onset_24h",
    "deterioration_24h",
]:

    y = labels[task].to_numpy(float)

    score = pred_df[f"{task}_prediction"].to_numpy(float)
    prob = pred_df[f"{task}_probability_mean"].to_numpy(float)

    keep = (
        np.isfinite(y)
        & (y >= 0)
        & np.isfinite(score)
        & np.isfinite(prob)
    )

    y = y[keep]
    score = score[keep]
    prob = prob[keep]

    auroc = roc_auc_score(y, score)
    auprc = average_precision_score(y, score)
    brier = brier_score_loss(y, prob)
    ece = expected_calibration_error(y, prob, bins=10)

    auroc_ci = bootstrap_metric(
        y,
        score,
        roc_auc_score,
    )

    auprc_ci = bootstrap_metric(
        y,
        score,
        average_precision_score,
    )

    brier_ci = bootstrap_metric(
        y,
        prob,
        brier_score_loss,
    )

    ece_ci = bootstrap_metric(
        y,
        prob,
        lambda yy, pp: expected_calibration_error(
            yy, pp, bins=10
        ),
    )

    cal_int, cal_slope = calibration_parameters(y, prob)

    cal_int_boot = []
    cal_slope_boot = []

    for _ in range(BOOTSTRAP_REPS):
        idx = rng.integers(0, len(y), size=len(y))

        yy = y[idx]
        pp = prob[idx]

        if len(np.unique(yy)) < 2:
            continue

        try:
            ci, cs = calibration_parameters(yy, pp)
            cal_int_boot.append(ci)
            cal_slope_boot.append(cs)
        except Exception:
            pass

    cal_int_boot = np.asarray(cal_int_boot, float)
    cal_slope_boot = np.asarray(cal_slope_boot, float)

    binary_rows.append({
        "dataset": "eICU-CRD",
        "model": "Z2_tuned",
        "task": task,
        "valid_n": int(len(y)),
        "events": int(y.sum()),
        "prevalence": float(y.mean()),

        "auroc": float(auroc),
        "auroc_ci95_low": auroc_ci[0],
        "auroc_ci95_high": auroc_ci[1],

        "auprc": float(auprc),
        "auprc_ci95_low": auprc_ci[0],
        "auprc_ci95_high": auprc_ci[1],

        "brier": float(brier),
        "brier_ci95_low": brier_ci[0],
        "brier_ci95_high": brier_ci[1],

        "ece": float(ece),
        "ece_ci95_low": ece_ci[0],
        "ece_ci95_high": ece_ci[1],

        "calibration_intercept": float(cal_int),
        "calibration_intercept_ci95_low": float(
            np.percentile(cal_int_boot, 2.5)
        ),
        "calibration_intercept_ci95_high": float(
            np.percentile(cal_int_boot, 97.5)
        ),

        "calibration_slope": float(cal_slope),
        "calibration_slope_ci95_low": float(
            np.percentile(cal_slope_boot, 2.5)
        ),
        "calibration_slope_ci95_high": float(
            np.percentile(cal_slope_boot, 97.5)
        ),

        "bootstrap_reps": BOOTSTRAP_REPS,
        "probability_source": "native Z2 probability_mean",
    })


# ------------------------------------------------------------------
# LOS statistics
# ------------------------------------------------------------------
y_los = labels["los_hours_24h"].to_numpy(float)
p_los = pred_df["los_prediction"].to_numpy(float)

keep = np.isfinite(y_los) & np.isfinite(p_los)

y_los = y_los[keep]
p_los = p_los[keep]

err = p_los - y_los

def los_mae(y, p):
    return mean_absolute_error(y, p)

def los_rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))

def los_bias(y, p):
    return float(np.mean(p - y))

mae_ci = bootstrap_metric(y_los, p_los, los_mae)
rmse_ci = bootstrap_metric(y_los, p_los, los_rmse)
bias_ci = bootstrap_metric(y_los, p_los, los_bias)

los_row = {
    "dataset": "eICU-CRD",
    "model": "Z2_tuned",
    "task": "los_hours_24h",
    "valid_n": int(len(y_los)),

    "mae": float(np.mean(np.abs(err))),
    "mae_ci95_low": mae_ci[0],
    "mae_ci95_high": mae_ci[1],

    "rmse": float(np.sqrt(np.mean(err ** 2))),
    "rmse_ci95_low": rmse_ci[0],
    "rmse_ci95_high": rmse_ci[1],

    "mean_error": float(np.mean(err)),
    "mean_error_ci95_low": bias_ci[0],
    "mean_error_ci95_high": bias_ci[1],

    "median_absolute_error": float(
        np.median(np.abs(err))
    ),

    "p90_absolute_error": float(
        np.quantile(np.abs(err), 0.90)
    ),

    "overprediction_rate": float(
        np.mean(err > 0)
    ),

    "underprediction_rate": float(
        np.mean(err < 0)
    ),

    "bootstrap_reps": BOOTSTRAP_REPS,
    "reporting_transform":
        "expm1(predicted_excess_log1p) + 24.0 hours",
}


# ------------------------------------------------------------------
# Save final artifacts
# ------------------------------------------------------------------
binary_df = pd.DataFrame(binary_rows)

binary_df.to_csv(
    STAT / "eicu_z2_canonical_binary_statistics.csv",
    index=False,
)

pd.DataFrame([los_row]).to_csv(
    STAT / "eicu_z2_canonical_los_statistics.csv",
    index=False,
)

metadata = {
    "dataset": "eICU-CRD",
    "model": "Z2_tuned",
    "bootstrap_reps": BOOTSTRAP_REPS,
    "bootstrap_method": "row bootstrap percentile 95% CI",
    "seed": SEED,
    "invalid_binary_label": -1,
    "binary_prediction_source":
        "native model_z_family rank-space prediction",
    "probability_source":
        "native model_z_family probability_mean",
    "brier_source":
        "native Z2 probability_mean",
    "ece_bins": 10,
    "calibration_intercept_slope":
        "additional logistic calibration analysis on native probability_mean",
    "external_refit": False,
    "external_tuning": False,
    "los_transform":
        "expm1(predicted_excess_log1p) + 24.0 hours",
}

(STAT / "eicu_z2_canonical_statistical_metadata.json").write_text(
    json.dumps(metadata, indent=2)
)

print("=" * 100)
print("NATIVE Z2 eICU STATISTICAL ANALYSIS COMPLETE")
print("=" * 100)

print("\nBINARY:")
print(binary_df.to_string(index=False))

print("\nLOS:")
print(pd.DataFrame([los_row]).to_string(index=False))

print("\nCanonical predictions:")
print(pred_path)

print("\nStatistical output:")
print(STAT)
