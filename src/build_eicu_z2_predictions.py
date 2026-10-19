import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/abdullah2025/fyp_project")
MODEL_ROOT = ROOT / "outputs/model_z_family"
MEMBERS = ROOT / "outputs/external_eicu_z2/members"
INV = ROOT / "outputs/external_validation_inventory"
OUT = ROOT / "outputs/external_eicu_z2"

labels = pd.read_csv(INV / "eicu_labels_aligned_24h.csv")

recipes = {
    "A2": json.loads((MODEL_ROOT / "A2_recipes.json").read_text()),
    "A3": json.loads((MODEL_ROOT / "A3_recipes.json").read_text()),
    "Z1": json.loads((MODEL_ROOT / "Z1_tuned_recipes.json").read_text()),
    "Z2": json.loads((MODEL_ROOT / "Z2_tuned_recipes.json").read_text()),
}

def member_blob(model_id):
    path = MEMBERS / f"{model_id}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path)

def member_binary(model_id, task):
    return member_blob(model_id)[task].astype(float)

def member_los(model_id):
    z = member_blob(model_id)
    if "los_hours_24h" in z:
        return z["los_hours_24h"].astype(float)
    return np.expm1(z["los_excess_log1p"].astype(float)) + 24.0

def weighted_mean(parts):
    total = sum(w for _, w in parts)
    out = None
    for value, weight in parts:
        out = weight * value if out is None else out + weight * value
    return out / total

def rank_mean(parts):
    return weighted_mean([
        (pd.Series(value).rank(method="average").to_numpy(dtype=float), weight)
        for value, weight in parts
    ])

def parent_binary(parent, task, rank):
    weights = recipes[parent][task]["weights"]
    parts = [
        (member_binary(model_id, task), weight)
        for model_id, weight in weights.items()
        if weight > 0
    ]
    return rank_mean(parts) if rank else weighted_mean(parts)

def parent_los(parent):
    weights = recipes[parent]["los_hours_24h"]["weights"]
    return weighted_mean([
        (member_los(model_id), weight)
        for model_id, weight in weights.items()
        if weight > 0
    ])

rank_scores = {}
prob_scores = {}

for task in [
    "mortality_24h",
    "ventilation_onset_24h",
    "vasopressor_onset_24h",
]:
    z2_weights = recipes["Z2"][task]["weights"]

    rank_scores[task] = rank_mean([
        (parent_binary(parent, task, rank=True), weight)
        for parent, weight in z2_weights.items()
        if weight > 0
    ])

    prob_scores[task] = weighted_mean([
        (parent_binary(parent, task, rank=False), weight)
        for parent, weight in z2_weights.items()
        if weight > 0
    ])

los_weights = recipes["Z2"]["los_hours_24h"]["weights"]

z2_los = weighted_mean([
    (parent_los(parent), weight)
    for parent, weight in los_weights.items()
    if weight > 0
])

# Deterioration uses the probability-scale ensemble exactly as in
# the previously verified eICU reconstruction.
z2_deterioration_probability = (
    1.0
    - (1.0 - prob_scores["mortality_24h"])
    * (1.0 - prob_scores["ventilation_onset_24h"])
    * (1.0 - prob_scores["vasopressor_onset_24h"])
)

pred = pd.DataFrame({
    "patientunitstayid": labels["patientunitstayid"],

    # Discrimination score: rank scale, NOT probability.
    "mortality_24h_score": rank_scores["mortality_24h"],
    "ventilation_onset_24h_score": rank_scores["ventilation_onset_24h"],
    "vasopressor_onset_24h_score": rank_scores["vasopressor_onset_24h"],

    # Probability-scale ensemble components.
    "mortality_24h_component_score": prob_scores["mortality_24h"],
    "ventilation_onset_24h_component_score": prob_scores["ventilation_onset_24h"],
    "vasopressor_onset_24h_component_score": prob_scores["vasopressor_onset_24h"],

    "deterioration_24h_score": z2_deterioration_probability,

    # Correct locked LOS reporting scale.
    "los_prediction": z2_los,
})

# Basic integrity
if len(pred) != len(labels):
    raise RuntimeError("Prediction/label row count mismatch.")

if pred["patientunitstayid"].duplicated().any():
    raise RuntimeError("Duplicate patientunitstayid.")

for col in pred.columns[1:]:
    if not np.isfinite(pred[col].to_numpy(float)).all():
        raise RuntimeError(f"Non-finite values in {col}.")

dest = OUT / "z2_tuned_eicu_predictions.csv"
pred.to_csv(dest, index=False)

print("=" * 90)
print("CORRECTED LOCKED eICU Z2 PREDICTIONS")
print("=" * 90)
print("rows:", len(pred))
print("destination:", dest)

print("\nRANK-SCALE DISCRIMINATION SCORES:")
for c in [
    "mortality_24h_score",
    "ventilation_onset_24h_score",
    "vasopressor_onset_24h_score",
]:
    print(
        c,
        "min=", pred[c].min(),
        "max=", pred[c].max(),
        "mean=", pred[c].mean(),
    )

print("\nPROBABILITY-SCALE SCORES:")
for c in prob_cols:
    print(
        c,
        "min=", pred[c].min(),
        "max=", pred[c].max(),
        "mean=", pred[c].mean(),
    )

print("\nLOS:")
print(
    "min=", pred["los_prediction"].min(),
    "max=", pred["los_prediction"].max(),
    "mean=", pred["los_prediction"].mean(),
)

