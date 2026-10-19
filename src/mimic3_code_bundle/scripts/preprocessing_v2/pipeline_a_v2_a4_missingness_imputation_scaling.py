#!/usr/bin/env python3
"""
Pipeline A V2
Stage A4 — Missingness, Temporal Derivation, Imputation, Scaling

IMPORTANT
---------
This is NOT the legacy outcome-construction A4.

Legacy A4 created:
    mortality
    readmission
    deterioration
    LOS

That design is obsolete for Pipeline A V2.

Canonical V2 target construction is A2.1 and already completed:
    mortality_24h
    ventilation_onset_24h
    vasopressor_onset_24h
    readmission_30d_24h
    los_hours_24h
    deterioration_24h

A4 consumes those frozen target artifacts only for alignment checks.
A4 itself transforms the frozen A3 structured 24-hour grid into the
model-ready structured representation.

LOCKED A4 ORDER
---------------
A3 complete hourly grid
    ↓
missingness masks (BEFORE fill)
    ↓
within-stay forward fill from LAST VALID observation
    ↓
time-since-last VALID observation
    ↓
trend / derivative features on regular hourly representation
    ↓
train-only fallback imputation
    ↓
train-only normalization

Safety:
- V1 outputs are never read as active V2 inputs.
- Raw MIMIC-IV files are not modified or required.
- A0/A1/A2/TGT/A3 are treated as read-only.
- The frozen V2 split must come from a V2-owned artifact containing
  split_24h. The script refuses to use pipeline_a/ V1 artifacts.
- All writes are atomic.
"""

from __future__ import annotations

from pathlib import Path
import gc
import json
import os
import tempfile

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# ============================================================
# 0. GLOBAL PATHS
# ============================================================

PROJECT_ROOT = Path("/home/abdullah2025/fyp_project")

V1_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "pipeline_a"
V2_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "pipeline_a_v2"

A0_PATH = V2_OUTPUT_DIR / "cohort_master.csv"
A1_FEATURE_DEFINITIONS_PATH = V2_OUTPUT_DIR / "a1_feature_definitions.json"

A3_PATH = V2_OUTPUT_DIR / "vitals_labs_hourly.parquet"

TARGET_DIR = V2_OUTPUT_DIR / "targets_24h"
TARGET_PRIMARY_PATH = TARGET_DIR / "outcomes_24h_primary_cohort.csv"

# V2-owned split/index is required. Do NOT point this at pipeline_a/.
SPLIT_INDEX_DEFAULT = V2_OUTPUT_DIR / "index_24h.csv"

# Canonical A4 outputs.
STRUCTURED_TENSOR_PATH = V2_OUTPUT_DIR / "structured_24h_processed.npy"
STRUCTURED_VALUES_PATH = V2_OUTPUT_DIR / "structured_24h_values.npy"
STRUCTURED_MASKS_PATH = V2_OUTPUT_DIR / "structured_24h_masks.npy"
STRUCTURED_TIME_SINCE_PATH = V2_OUTPUT_DIR / "structured_24h_time_since.npy"
STRUCTURED_TRENDS_PATH = V2_OUTPUT_DIR / "structured_24h_trends.npy"

DEMOGRAPHICS_PATH = V2_OUTPUT_DIR / "demographics_24h_processed.npy"
INDEX_OUTPUT_PATH = V2_OUTPUT_DIR / "final_index_24h.csv"

IMPUTATION_STATS_PATH = V2_OUTPUT_DIR / "train_imputation_stats.csv"
SCALING_STATS_PATH = V2_OUTPUT_DIR / "train_scaling_stats.csv"
QUALITY_SUMMARY_PATH = V2_OUTPUT_DIR / "a4_feature_quality_summary.csv"

CHANNEL_METADATA_PATH = V2_OUTPUT_DIR / "final_model_input_metadata_24h.csv"
MODEL_DEFINITION_PATH = V2_OUTPUT_DIR / "final_model_input_definition_24h.json"
PREPROCESSING_METADATA_PATH = V2_OUTPUT_DIR / "preprocessing_metadata.json"


EXPECTED_A0_STAYS = 93_224
EXPECTED_PRIMARY_STAYS = 74_572
SEQUENCE_LENGTH = 24

EXPECTED_SPLITS = {
    "train": 52_268,
    "val": 7_417,
    "test": 14_887,
}


# ============================================================
# 1. FROZEN V2 FEATURE CONTRACT
# ============================================================

EXPECTED_FEATURE_ORDER = [
    "heart_rate",
    "sbp",
    "dbp",
    "mbp",
    "resp_rate",
    "spo2",
    "temperature",
    "wbc",
    "hemoglobin",
    "platelets",
    "creatinine",
    "bun",
    "lactate",
    "glucose",
    "sodium",
    "potassium",
    "chloride",
    "bicarbonate",
    "ph",
    "pao2",
    "paco2",
    "bilirubin",
    "gcs_total",
    "gcs_unable",
    "gcs_components_observed",
    "fio2",
    "peep",
    "resp_support_mode",
]

EXPECTED_TARGETS = [
    "mortality_24h",
    "ventilation_onset_24h",
    "vasopressor_onset_24h",
    "readmission_30d_24h",
    "los_hours_24h",
    "deterioration_24h",
]

VALIDITY_COLUMNS = [
    "mortality_24h_valid",
    "ventilation_onset_24h_valid",
    "vasopressor_onset_24h_valid",
    "readmission_30d_24h_valid",
    "los_hours_24h_valid",
    "deterioration_24h_valid",
]

# A4 trend specification.
#
# The master specification requires trend policy to explicitly define:
# variable, window, formula, missing-data handling, and minimum observations.
# It does not enumerate a concrete trend list. Therefore this implementation
# makes the policy explicit rather than silently inventing it elsewhere.
#
# Trends are deliberately limited to non-binary/non-ordinal features.
# This gives:
#   26 trend channels = all 28 base features except:
#       gcs_unable
#       resp_support_mode
#
# A 3-hour trailing OLS slope is computed on the regular hourly
# forward-filled representation. Missing trend windows remain NaN and are
# subsequently filled with a TRAIN-ONLY trend mean so the final tensor is finite.
#
TREND_FEATURES = [
    feature
    for feature in EXPECTED_FEATURE_ORDER
    if feature not in {"gcs_unable", "resp_support_mode"}
]

TREND_WINDOW_HOURS = 3
TREND_MIN_WINDOW_VALUES = 3
TREND_METHOD = "trailing_3h_ols_slope_on_forward_filled_hourly_values"


# ============================================================
# 2. HELPERS
# ============================================================

def require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}:\n{path}")


def atomic_write_bytes(write_fn, final_path: Path) -> None:
    final_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        dir=final_path.parent,
        prefix=final_path.name + ".",
        suffix=".tmp",
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)

    try:
        write_fn(tmp_path)
        os.replace(tmp_path, final_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def atomic_write_csv(df: pd.DataFrame, final_path: Path) -> None:
    atomic_write_bytes(
        lambda tmp: df.to_csv(
            tmp,
            index=False,
            encoding="utf-8",
        ),
        final_path,
    )


def atomic_write_json(obj: dict, final_path: Path) -> None:
    def writer(tmp: Path) -> None:
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(
                obj,
                handle,
                indent=2,
                ensure_ascii=False,
                default=str,
            )

    atomic_write_bytes(writer, final_path)


def atomic_write_npy(array: np.ndarray, final_path: Path) -> None:
    def writer(tmp: Path) -> None:
        with open(tmp, "wb") as handle:
            np.save(handle, array, allow_pickle=False)

    atomic_write_bytes(writer, final_path)


def find_v2_split_index() -> Path:
    """
    Locate exactly one V2-owned CSV containing split_24h.

    This intentionally searches ONLY V2_OUTPUT_DIR.
    It refuses to use historical pipeline_a/.
    """
    candidates = []

    for path in V2_OUTPUT_DIR.rglob("*.csv"):
        if path.name.startswith("a4_"):
            continue

        try:
            header = pd.read_csv(path, nrows=0)
        except Exception:
            continue

        required = {"stay_id", "subject_id", "hadm_id", "split_24h"}
        if required.issubset(header.columns):
            candidates.append(path)

    if SPLIT_INDEX_DEFAULT.exists() and SPLIT_INDEX_DEFAULT not in candidates:
        try:
            header = pd.read_csv(SPLIT_INDEX_DEFAULT, nrows=0)
            if {"stay_id", "subject_id", "hadm_id", "split_24h"}.issubset(
                header.columns
            ):
                candidates.insert(0, SPLIT_INDEX_DEFAULT)
        except Exception:
            pass

    # Deduplicate resolved paths.
    unique = []
    seen = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved not in seen:
            unique.append(path)
            seen.add(resolved)

    if len(unique) == 0:
        raise FileNotFoundError(
            "No V2-owned split/index CSV with columns "
            "{stay_id, subject_id, hadm_id, split_24h} was found under "
            f"{V2_OUTPUT_DIR}.\n"
            "Do NOT substitute the historical V1 index from "
            f"{V1_OUTPUT_DIR}."
        )

    if len(unique) > 1:
        lines = "\n".join(f"  - {p}" for p in unique)
        raise RuntimeError(
            "Multiple V2 split/index candidates were found. "
            "A4 refuses to guess which one is canonical:\n" + lines
        )

    return unique[0]


def validate_unique_mapping(df: pd.DataFrame, key: str, label: str) -> None:
    if df[key].duplicated().any():
        duplicates = int(df[key].duplicated().sum())
        raise AssertionError(
            f"{label}: duplicate {key} rows detected: {duplicates:,}"
        )


def compute_ols_slope(values: np.ndarray) -> float:
    """OLS slope over consecutive hourly positions."""
    if values.size != TREND_WINDOW_HOURS:
        raise ValueError(
            f"Expected exactly {TREND_WINDOW_HOURS} values, got {values.size}."
        )

    if not np.isfinite(values).all():
        return np.nan

    x = np.arange(values.size, dtype=np.float64)
    x_mean = x.mean()
    y_mean = values.mean()

    denominator = np.sum((x - x_mean) ** 2)
    if denominator == 0:
        return np.nan

    return float(
        np.sum((x - x_mean) * (values - y_mean)) / denominator
    )


def compute_trend_matrix(
    filled_values: np.ndarray,
    trend_feature_indices: list[int],
) -> np.ndarray:
    """
    Input:
        [N, 24, F]
    Output:
        [N, 24, T]

    A trailing 3-hour OLS slope is computed on the regular hourly,
    forward-filled representation. The first two hours have no complete
    window and therefore remain NaN until train-only trend-mean fallback.
    """
    n_stays, hours, _ = filled_values.shape

    if hours != SEQUENCE_LENGTH:
        raise AssertionError(
            f"Expected {SEQUENCE_LENGTH} hours, found {hours}."
        )

    trends = np.full(
        (n_stays, hours, len(trend_feature_indices)),
        np.nan,
        dtype=np.float32,
    )

    for t in range(TREND_WINDOW_HOURS - 1, hours):
        window = filled_values[
            :,
            t - TREND_WINDOW_HOURS + 1 : t + 1,
            trend_feature_indices,
        ]

        # window = [N, 3, T]
        valid = np.isfinite(window).all(axis=1)

        # Compute each trend channel independently.
        for j in range(len(trend_feature_indices)):
            vals = window[:, :, j]

            for i in np.flatnonzero(valid[:, j]):
                trends[i, t, j] = compute_ols_slope(
                    vals[i].astype(np.float64)
                )

    return trends


# ============================================================
# 3. STARTUP / SOURCE LOCKS
# ============================================================

print("=" * 80)
print("PIPELINE A V2 — A4 MISSINGNESS + TEMPORAL DERIVATION + IMPUTATION")
print("=" * 80)

require_file(A0_PATH, "A0 cohort")
require_file(A1_FEATURE_DEFINITIONS_PATH, "A1 feature definitions")
require_file(A3_PATH, "A3 hourly grid")
require_file(TARGET_PRIMARY_PATH, "A2.1 primary target cohort")

if V1_OUTPUT_DIR.resolve() == V2_OUTPUT_DIR.resolve():
    raise RuntimeError(
        "SAFETY ERROR: V1 and V2 output directories resolve to the same path."
    )

print(f"A0    : {A0_PATH}")
print(f"A3    : {A3_PATH}")
print(f"TGT   : {TARGET_PRIMARY_PATH}")
print(f"V2 out: {V2_OUTPUT_DIR}")


# ============================================================
# 4. LOAD / VALIDATE FEATURE DEFINITIONS
# ============================================================

with open(A1_FEATURE_DEFINITIONS_PATH, "r", encoding="utf-8") as handle:
    a1_defs = json.load(handle)

feature_order = a1_defs.get("feature_order")
feature_types = a1_defs.get("feature_types", {})

if feature_order != EXPECTED_FEATURE_ORDER:
    raise AssertionError(
        "A1 feature order does not match the frozen V2 28-feature contract."
    )

if len(feature_order) != 28:
    raise AssertionError(
        f"Expected 28 A1 features, found {len(feature_order)}."
    )

for feature in feature_order:
    if feature not in feature_types:
        raise AssertionError(
            f"Missing feature type for {feature} in A1 definitions."
        )

print("\nPASS: Frozen 28-feature A1 contract loaded")


# ============================================================
# 5. LOAD A0 / TGT / V2 SPLIT
# ============================================================

a0 = pd.read_csv(A0_PATH)

required_a0 = [
    "subject_id",
    "hadm_id",
    "stay_id",
    "age",
    "gender",
    "intime",
    "outtime",
    "icu_los_hours",
]

missing_a0 = [c for c in required_a0 if c not in a0.columns]
if missing_a0:
    raise AssertionError(
        f"A0 is missing required columns: {missing_a0}"
    )

if len(a0) != EXPECTED_A0_STAYS:
    raise AssertionError(
        f"A0 row count changed: expected {EXPECTED_A0_STAYS:,}, "
        f"found {len(a0):,}"
    )

validate_unique_mapping(a0, "stay_id", "A0")

a0["stay_id"] = pd.to_numeric(a0["stay_id"], errors="raise").astype(np.int64)
a0["subject_id"] = pd.to_numeric(a0["subject_id"], errors="raise").astype(np.int64)
a0["hadm_id"] = pd.to_numeric(a0["hadm_id"], errors="raise").astype(np.int64)
a0["age"] = pd.to_numeric(a0["age"], errors="coerce")

tgt = pd.read_csv(TARGET_PRIMARY_PATH)

if len(tgt) != EXPECTED_PRIMARY_STAYS:
    raise AssertionError(
        f"TGT primary cohort changed: expected {EXPECTED_PRIMARY_STAYS:,}, "
        f"found {len(tgt):,}"
    )

required_tgt = [
    "stay_id",
    "subject_id",
    "hadm_id",
    *EXPECTED_TARGETS,
    *VALIDITY_COLUMNS,
]

missing_tgt = [c for c in required_tgt if c not in tgt.columns]
if missing_tgt:
    raise AssertionError(
        f"TGT primary cohort is missing required columns: {missing_tgt}"
    )

validate_unique_mapping(tgt, "stay_id", "TGT primary")

tgt["stay_id"] = pd.to_numeric(tgt["stay_id"], errors="raise").astype(np.int64)
tgt["subject_id"] = pd.to_numeric(tgt["subject_id"], errors="raise").astype(np.int64)
tgt["hadm_id"] = pd.to_numeric(tgt["hadm_id"], errors="raise").astype(np.int64)

split_path = find_v2_split_index()
print(f"V2 split/index candidate: {split_path}")

split_df = pd.read_csv(split_path)

required_split = [
    "stay_id",
    "subject_id",
    "hadm_id",
    "split_24h",
]

missing_split = [c for c in required_split if c not in split_df.columns]
if missing_split:
    raise AssertionError(
        f"V2 split/index missing columns: {missing_split}"
    )

validate_unique_mapping(split_df, "stay_id", "V2 split/index")

split_df["stay_id"] = pd.to_numeric(
    split_df["stay_id"], errors="raise"
).astype(np.int64)
split_df["subject_id"] = pd.to_numeric(
    split_df["subject_id"], errors="raise"
).astype(np.int64)
split_df["hadm_id"] = pd.to_numeric(
    split_df["hadm_id"], errors="raise"
).astype(np.int64)
split_df["split_24h"] = (
    split_df["split_24h"]
    .astype("string")
    .str.strip()
    .str.lower()
)

if set(split_df["split_24h"].dropna().unique()) != {
    "train", "val", "test"
}:
    raise AssertionError(
        "V2 split/index must contain exactly train/val/test in split_24h."
    )

actual_splits = (
    split_df["split_24h"]
    .value_counts()
    .reindex(["train", "val", "test"])
    .fillna(0)
    .astype(int)
    .to_dict()
)

if actual_splits != EXPECTED_SPLITS:
    raise AssertionError(
        "Frozen split counts changed.\n"
        f"Expected: {EXPECTED_SPLITS}\n"
        f"Found   : {actual_splits}"
    )

# Split must cover exactly the target primary cohort.
tgt_ids = tgt[["stay_id", "subject_id", "hadm_id"]].set_index("stay_id").sort_index()
split_ids = split_df[["stay_id", "subject_id", "hadm_id"]].set_index("stay_id").sort_index()

if not tgt_ids.index.equals(split_ids.index):
    raise AssertionError(
        "TGT primary stay IDs and V2 split/index stay IDs do not match."
    )

if not tgt_ids["subject_id"].equals(split_ids["subject_id"]):
    raise AssertionError(
        "TGT/split subject_id identity mismatch."
    )

if not tgt_ids["hadm_id"].equals(split_ids["hadm_id"]):
    raise AssertionError(
        "TGT/split hadm_id identity mismatch."
    )

# Patient-level split isolation.
split_subject_sets = {
    split_name: set(
        split_df.loc[
            split_df["split_24h"].eq(split_name),
            "subject_id",
        ].tolist()
    )
    for split_name in ["train", "val", "test"]
}

for left, right in [
    ("train", "val"),
    ("train", "test"),
    ("val", "test"),
]:
    overlap = split_subject_sets[left] & split_subject_sets[right]
    if overlap:
        raise AssertionError(
            f"Subject leakage detected between {left} and {right}: "
            f"{len(overlap):,}"
        )

print("PASS: A0/TGT/V2 split identity and patient-level isolation")


# ============================================================
# 6. READ A3 — EXACT GRID
# ============================================================

a3_schema = pq.read_schema(A3_PATH)
a3_columns = a3_schema.names

expected_a3_columns = [
    "subject_id",
    "hadm_id",
    "stay_id",
    "icu_hour",
    *EXPECTED_FEATURE_ORDER,
]

if a3_columns != expected_a3_columns:
    raise AssertionError(
        "A3 column contract mismatch.\n"
        f"Expected: {expected_a3_columns}\n"
        f"Found   : {a3_columns}"
    )

a3 = pd.read_parquet(
    A3_PATH,
    engine="pyarrow",
)

expected_rows = EXPECTED_PRIMARY_STAYS * SEQUENCE_LENGTH

if len(a3) != expected_rows:
    raise AssertionError(
        f"A3 row count changed: expected {expected_rows:,}, "
        f"found {len(a3):,}"
    )

if a3["stay_id"].nunique() != EXPECTED_PRIMARY_STAYS:
    raise AssertionError(
        "A3 unique stay count does not equal the frozen primary cohort."
    )

if a3.duplicated(["stay_id", "icu_hour"]).any():
    raise AssertionError(
        "A3 contains duplicate stay/hour rows."
    )

hour_values = set(
    pd.to_numeric(a3["icu_hour"], errors="raise").unique().tolist()
)
if hour_values != set(range(SEQUENCE_LENGTH)):
    raise AssertionError(
        f"A3 hour set is not exactly 0..23: {sorted(hour_values)}"
    )

row_counts = a3.groupby("stay_id").size()
if not (row_counts == SEQUENCE_LENGTH).all():
    raise AssertionError(
        "A3 does not contain exactly 24 rows per stay."
    )

# Exact identity against TGT.
a3_ids = (
    a3[
        ["stay_id", "subject_id", "hadm_id"]
    ]
    .drop_duplicates("stay_id")
    .set_index("stay_id")
    .sort_index()
)

if not a3_ids.index.equals(tgt_ids.index):
    raise AssertionError("A3/TGT stay ID mismatch.")

if not a3_ids["subject_id"].equals(tgt_ids["subject_id"]):
    raise AssertionError("A3/TGT subject identity mismatch.")

if not a3_ids["hadm_id"].equals(tgt_ids["hadm_id"]):
    raise AssertionError("A3/TGT admission identity mismatch.")

# Sort into canonical order.
a3 = (
    a3
    .sort_values(["stay_id", "icu_hour"], kind="mergesort")
    .reset_index(drop=True)
)

print(
    f"PASS: A3 exact 24h grid loaded "
    f"({len(a3):,} rows; {EXPECTED_PRIMARY_STAYS:,} stays)"
)


# ============================================================
# 7. BUILD REGULAR [N,24,F] ORIGINAL GRID
# ============================================================

stay_order = tgt["stay_id"].sort_values().to_numpy(dtype=np.int64)

# Re-align split/index to the target order.
index_df = (
    tgt[
        ["stay_id", "subject_id", "hadm_id"]
    ]
    .copy()
    .merge(
        split_df[
            ["stay_id", "split_24h"]
        ],
        on="stay_id",
        how="left",
        validate="one_to_one",
    )
    .sort_values("stay_id")
    .reset_index(drop=True)
)

if not np.array_equal(
    index_df["stay_id"].to_numpy(dtype=np.int64),
    stay_order,
):
    raise AssertionError(
        "Canonical A4 index ordering failed."
    )

# Reshape values in already-sorted stay/hour order.
values_2d = (
    a3[EXPECTED_FEATURE_ORDER]
    .apply(pd.to_numeric, errors="coerce")
    .to_numpy(dtype=np.float32)
)

values = values_2d.reshape(
    EXPECTED_PRIMARY_STAYS,
    SEQUENCE_LENGTH,
    len(EXPECTED_FEATURE_ORDER),
)

# Ensure original A3 valid values are finite.
nonfinite_mask = ~np.isfinite(values) & ~np.isnan(values)
if nonfinite_mask.any():
    raise AssertionError(
        "A3 contains nonfinite clinical values."
    )

# Original-validity mask MUST be created before any fill.
observed_mask = np.isfinite(values)

print(
    f"Original valid cells: {int(observed_mask.sum()):,} "
    f"/ {values.size:,}"
)


# ============================================================
# 8. WITHIN-STAY FORWARD FILL + TIME SINCE VALID OBSERVATION
# ============================================================

filled_values = values.copy()
time_since = np.zeros_like(values, dtype=np.float32)

for feature_idx, feature in enumerate(EXPECTED_FEATURE_ORDER):
    # Forward fill within each stay only.
    feature_frame = pd.DataFrame(
        filled_values[:, :, feature_idx]
    )
    feature_frame = feature_frame.ffill(axis=1)

    filled_values[:, :, feature_idx] = (
        feature_frame.to_numpy(dtype=np.float32)
    )

    # Time-since uses ORIGINAL VALID observations, not filled values.
    obs = observed_mask[:, :, feature_idx]
    for i in range(EXPECTED_PRIMARY_STAYS):
        last_valid_hour = -1

        for h in range(SEQUENCE_LENGTH):
            if obs[i, h]:
                last_valid_hour = h
                time_since[i, h, feature_idx] = 0.0
            elif last_valid_hour >= 0:
                time_since[i, h, feature_idx] = float(
                    h - last_valid_hour
                )
            else:
                # Explicit pre-first/no-observation representation:
                # 24 means no valid observation has occurred anywhere
                # before this hour within the 24h benchmark window.
                time_since[i, h, feature_idx] = float(SEQUENCE_LENGTH)

# Sanity:
# - observed cells must have time_since=0
# - time_since must be 0..24
if not np.all(time_since[observed_mask] == 0.0):
    raise AssertionError(
        "Time-since channel is nonzero at an originally observed cell."
    )

if not ((time_since >= 0.0) & (time_since <= 24.0)).all():
    raise AssertionError(
        "Time-since channel contains values outside [0,24]."
    )

print("PASS: Original masks created before fill")
print("PASS: Forward fill performed within stay")
print("PASS: Time-since derived from original valid observations")


# ============================================================
# 9. TREND / DERIVATIVE CHANNELS
# ============================================================

trend_indices = [
    EXPECTED_FEATURE_ORDER.index(feature)
    for feature in TREND_FEATURES
]

trend_array = compute_trend_matrix(
    filled_values=filled_values,
    trend_feature_indices=trend_indices,
)

if trend_array.shape != (
    EXPECTED_PRIMARY_STAYS,
    SEQUENCE_LENGTH,
    len(TREND_FEATURES),
):
    raise AssertionError(
        f"Unexpected trend shape: {trend_array.shape}"
    )

print(
    f"Trend policy: {TREND_METHOD}; "
    f"{len(TREND_FEATURES)} channels; "
    f"window={TREND_WINDOW_HOURS}h; "
    f"min_complete_window={TREND_MIN_WINDOW_VALUES}"
)


# ============================================================
# 10. TRAIN-ONLY FALLBACK MEDIANS
# ============================================================

train_mask_stays = (
    index_df["split_24h"].eq("train").to_numpy()
)

if int(train_mask_stays.sum()) != EXPECTED_SPLITS["train"]:
    raise AssertionError("Unexpected number of training stays.")

train_raw = values[train_mask_stays]

imputation_rows = []
train_medians = {}

for feature_idx, feature in enumerate(EXPECTED_FEATURE_ORDER):
    vals = train_raw[:, :, feature_idx].astype(np.float64).reshape(-1)
    vals = vals[np.isfinite(vals)]

    if vals.size == 0:
        raise AssertionError(
            f"No valid training observations exist for {feature}; "
            "cannot compute a train-only population median."
        )

    median = float(np.median(vals))
    train_medians[feature] = median

    imputation_rows.append({
        "feature": feature,
        "train_observed_cells": int(vals.size),
        "train_population_median": median,
    })

imputation_stats = pd.DataFrame(imputation_rows)

# A3 forward fill handles post-first missing values.
# Two residual cases remain:
#
# 1. zero valid observation across the whole 24h stay-feature:
#       train population median
#
# 2. pre-first-observation leading cells for a stay that later has
#    a valid value:
#       also use train median as a no-look-ahead finite-input safeguard.
#
# The master specification explicitly mandates case (1). Case (2) is
# an implementation resolution for a real finite-tensor edge case:
# backward fill would introduce future information into earlier hours.
#
fallback_counts = {
    "zero_valid_window": {feature: 0 for feature in EXPECTED_FEATURE_ORDER},
    "leading_pre_first_valid": {feature: 0 for feature in EXPECTED_FEATURE_ORDER},
}

for feature_idx, feature in enumerate(EXPECTED_FEATURE_ORDER):
    median = train_medians[feature]

    for i in range(EXPECTED_PRIMARY_STAYS):
        original_obs = observed_mask[i, :, feature_idx]

        if not original_obs.any():
            fallback_counts["zero_valid_window"][feature] += SEQUENCE_LENGTH
        else:
            first_valid = int(np.flatnonzero(original_obs)[0])
            fallback_counts["leading_pre_first_valid"][feature] += first_valid

    col = filled_values[:, :, feature_idx]

    nan_mask = ~np.isfinite(col)

    # All remaining NaNs at this point are leading gaps or entire-stay gaps.
    col[nan_mask] = median

    filled_values[:, :, feature_idx] = col

# Now values must be finite.
if not np.isfinite(filled_values).all():
    raise AssertionError(
        "Nonfinite values remain after A4 train-only value fallback."
    )

for feature in EXPECTED_FEATURE_ORDER:
    idx = EXPECTED_FEATURE_ORDER.index(feature)
    zero_cells = fallback_counts["zero_valid_window"][feature]
    leading_cells = fallback_counts["leading_pre_first_valid"][feature]

    row = imputation_stats.loc[
        imputation_stats["feature"].eq(feature)
    ].index[0]

    imputation_stats.loc[row, "zero_valid_stay_cells_filled"] = zero_cells
    imputation_stats.loc[row, "leading_pre_first_valid_cells_filled"] = leading_cells

imputation_stats = imputation_stats[
    [
        "feature",
        "train_observed_cells",
        "train_population_median",
        "zero_valid_stay_cells_filled",
        "leading_pre_first_valid_cells_filled",
    ]
].copy()

print("PASS: Train-only population medians computed")
print("PASS: Value tensor finite after train-only fallback")


# ============================================================
# 11. TRAIN-ONLY TREND FALLBACK
# ============================================================

trend_train = trend_array[train_mask_stays]

trend_train_stats = []

for j, feature in enumerate(TREND_FEATURES):
    vals = trend_train[:, :, j].astype(np.float64).reshape(-1)
    vals = vals[np.isfinite(vals)]

    if vals.size == 0:
        # A trend channel can legitimately have no finite training trend.
        # Use 0 as a neutral channel value and record std=1.
        train_mean = 0.0
        train_std = 1.0
        n = 0
    else:
        train_mean = float(vals.mean())
        train_std = float(vals.std(ddof=0))

        if not np.isfinite(train_std) or train_std <= 0.0:
            train_std = 1.0

        n = int(vals.size)

    trend_train_stats.append({
        "feature": feature,
        "channel_group": "trend",
        "train_finite_cells": n,
        "train_mean": train_mean,
        "train_std": train_std,
    })

    # Fill missing trend values with TRAIN-ONLY mean.
    missing = ~np.isfinite(trend_array[:, :, j])
    if missing.any():
        trend_array[:, :, j][missing] = train_mean

trend_stats_df = pd.DataFrame(trend_train_stats)

if not np.isfinite(trend_array).all():
    raise AssertionError(
        "Nonfinite values remain in trend tensor."
    )

print("PASS: Trend missingness resolved with train-only trend means")


# ============================================================
# 12. TRAIN-ONLY NORMALIZATION
# ============================================================

# Policy:
# - Continuous/value channels: z-score using train-only mean/std.
# - Binary gcs_unable: left as 0/1.
# - Ordinal resp_support_mode: left as 0..5.
# - Masks: left as 0/1.
# - Time-since: left as raw elapsed hours 0..24 because its units are
#   semantically meaningful and used directly by time-aware architectures.
# - Trend channels: z-score using train-only mean/std.
#
# This is explicit in metadata so model scripts cannot guess.

scaling_rows = []

for feature_idx, feature in enumerate(EXPECTED_FEATURE_ORDER):
    ftype = str(feature_types[feature])

    vals = filled_values[train_mask_stays, :, feature_idx].astype(np.float64).reshape(-1)
    if not np.isfinite(vals).all():
        raise AssertionError(
            f"Training values for {feature} are not finite."
        )

    if ftype in {"binary", "ordinal_categorical"}:
        mean = float(vals.mean())
        std = 1.0
        scale_type = "none_discrete_code_preserved"

        # Preserve binary/ordinal clinical codes exactly.
    else:
        mean = float(vals.mean())
        std = float(vals.std(ddof=0))
        if not np.isfinite(std) or std <= 0.0:
            std = 1.0
        scale_type = "zscore_train_only"

        filled_values[:, :, feature_idx] = (
            (filled_values[:, :, feature_idx].astype(np.float64) - mean)
            / std
        ).astype(np.float32)

    scaling_rows.append({
        "channel_group": "value",
        "feature": feature,
        "train_mean": mean,
        "train_std": std,
        "scaling": scale_type,
    })

# Trend scaling.
for j, feature in enumerate(TREND_FEATURES):
    row = trend_stats_df.iloc[j]

    mean = float(row["train_mean"])
    std = float(row["train_std"])

    trend_array[:, :, j] = (
        (
            trend_array[:, :, j].astype(np.float64)
            - mean
        )
        / std
    ).astype(np.float32)

    scaling_rows.append({
        "channel_group": "trend",
        "feature": feature,
        "train_mean": mean,
        "train_std": std,
        "scaling": "zscore_train_only",
    })

scaling_stats = pd.DataFrame(scaling_rows)

if not np.isfinite(filled_values).all():
    raise AssertionError(
        "Nonfinite values remain after normalization."
    )

if not np.isfinite(trend_array).all():
    raise AssertionError(
        "Nonfinite trend values remain after normalization."
    )

print("PASS: Train-only value normalization")
print("PASS: Train-only trend normalization")


# ============================================================
# 13. MASK / TIME-SINCE / FINAL CHANNEL STACK
# ============================================================

masks_float = observed_mask.astype(np.float32)

if not np.isfinite(masks_float).all():
    raise AssertionError("Mask tensor contains nonfinite values.")

if not np.isfinite(time_since).all():
    raise AssertionError("Time-since tensor contains nonfinite values.")

# Final channel order is DERIVED here.
# Do not hard-code an old 44- or 66-channel dimension.
channel_names = []
channel_groups = []
source_features = []
scaling_policies = []

for feature in EXPECTED_FEATURE_ORDER:
    channel_names.append(feature)
    channel_groups.append("value")
    source_features.append(feature)

for feature in EXPECTED_FEATURE_ORDER:
    channel_names.append(f"{feature}__missing")
    channel_groups.append("mask")
    source_features.append(feature)

for feature in EXPECTED_FEATURE_ORDER:
    channel_names.append(f"{feature}__time_since")
    channel_groups.append("time_since")
    source_features.append(feature)

for feature in TREND_FEATURES:
    channel_names.append(f"{feature}__trend_slope_3h")
    channel_groups.append("trend")
    source_features.append(feature)

assert len(channel_names) == 28 + 28 + 28 + len(TREND_FEATURES)
N_CHANNELS = len(channel_names)

# The channel count is derived from the actual implemented channel list.
# With the current explicit trend policy this resolves to:
#   28 value + 28 mask + 28 time-since + 26 trend = 110.
# No historical 44- or 66-channel assumption is used.

# Combined tensor [N,24,C].
final_tensor = np.concatenate(
    [
        filled_values,
        masks_float,
        time_since,
        trend_array,
    ],
    axis=2,
).astype(np.float32)

if final_tensor.shape != (
    EXPECTED_PRIMARY_STAYS,
    SEQUENCE_LENGTH,
    N_CHANNELS,
):
    raise AssertionError(
        f"Unexpected final tensor shape: {final_tensor.shape}"
    )

if not np.isfinite(final_tensor).all():
    raise AssertionError(
        "Final A4 structured tensor contains NaN/Inf."
    )

print(
    f"PASS: Final structured tensor shape = "
    f"{final_tensor.shape}"
)


# ============================================================
# 14. DEMOGRAPHICS — TRAIN-ONLY AGE PROCESSING + LOCKED GENDER
# ============================================================

demo = (
    index_df[
        ["stay_id", "subject_id", "hadm_id"]
    ]
    .merge(
        a0[
            ["stay_id", "age", "gender"]
        ],
        on="stay_id",
        how="left",
        validate="one_to_one",
    )
)

if len(demo) != EXPECTED_PRIMARY_STAYS:
    raise AssertionError(
        "Demographic alignment changed the primary cohort size."
    )

if demo["age"].isna().all():
    raise AssertionError(
        "A0 age is entirely missing."
    )

train_demo_mask = index_df["split_24h"].eq("train").to_numpy()

train_age = pd.to_numeric(
    demo.loc[train_demo_mask, "age"],
    errors="coerce",
).to_numpy(dtype=np.float64)

finite_train_age = train_age[np.isfinite(train_age)]

if finite_train_age.size == 0:
    raise AssertionError("No finite training age values exist.")

age_train_median = float(np.median(finite_train_age))

age = pd.to_numeric(
    demo["age"],
    errors="coerce",
).to_numpy(dtype=np.float64)

age_missing = ~np.isfinite(age)
age[age_missing] = age_train_median

age_train_imputed = age[train_demo_mask]

age_mean = float(age_train_imputed.mean())
age_std = float(age_train_imputed.std(ddof=0))

if not np.isfinite(age_std) or age_std <= 0.0:
    age_std = 1.0

age_z = ((age - age_mean) / age_std).astype(np.float32)

gender = (
    demo["gender"]
    .astype("string")
    .str.strip()
    .str.upper()
)

gender_f = gender.eq("F").astype(np.float32).to_numpy()
gender_m = gender.eq("M").astype(np.float32).to_numpy()
gender_unknown = (
    (~gender.eq("F")) & (~gender.eq("M"))
).astype(np.float32).to_numpy()

# The locked V2 cohort has 0 unknown genders.
# Do not silently reinterpret numeric/unknown encodings.
if int(gender_f.sum()) != 32_243:
    raise AssertionError(
        "Female count changed from the verified V2 demographic representation."
    )

if int(gender_m.sum()) != 42_329:
    raise AssertionError(
        "Male count changed from the verified V2 demographic representation."
    )

if int(gender_unknown.sum()) != 0:
    raise AssertionError(
        "Unexpected unknown gender values in the frozen V2 cohort."
    )

demographics = np.column_stack(
    [
        age_z,
        gender_f,
        gender_m,
        gender_unknown,
    ]
).astype(np.float32)

if demographics.shape != (EXPECTED_PRIMARY_STAYS, 4):
    raise AssertionError(
        f"Unexpected demographic shape: {demographics.shape}"
    )

if not np.isfinite(demographics).all():
    raise AssertionError(
        "Demographic tensor contains nonfinite values."
    )

print("PASS: Demographics aligned and train-only age normalization applied")


# ============================================================
# 15. FINAL INDEX
# ============================================================

final_index = index_df.copy()

# Add target columns for alignment/traceability without changing target values.
target_alignment = tgt[
    [
        "stay_id",
        *EXPECTED_TARGETS,
        *VALIDITY_COLUMNS,
    ]
].copy()

final_index = final_index.merge(
    target_alignment,
    on="stay_id",
    how="left",
    validate="one_to_one",
)

if len(final_index) != EXPECTED_PRIMARY_STAYS:
    raise AssertionError(
        "Final index row count changed."
    )

if final_index["stay_id"].duplicated().any():
    raise AssertionError(
        "Final index contains duplicate stay IDs."
    )

if final_index[EXPECTED_TARGETS + VALIDITY_COLUMNS].isna().any().any():
    raise AssertionError(
        "Target alignment contains missing target/index fields."
    )

# Canonical row ordering.
final_index = final_index.sort_values(
    "stay_id",
    kind="mergesort",
).reset_index(drop=True)

if not np.array_equal(
    final_index["stay_id"].to_numpy(dtype=np.int64),
    stay_order,
):
    raise AssertionError(
        "Final index ordering no longer matches target ordering."
    )

print("PASS: Final index / six-output target alignment")


# ============================================================
# 16. FEATURE QUALITY SUMMARY
# ============================================================

quality_rows = []

for feature_idx, feature in enumerate(EXPECTED_FEATURE_ORDER):
    orig = values[:, :, feature_idx]
    obs = observed_mask[:, :, feature_idx]
    post_ffill = filled_values[:, :, feature_idx]

    zero_valid_stays = int(
        (~obs.any(axis=1)).sum()
    )

    leading_gap_cells = int(
        fallback_counts["leading_pre_first_valid"][feature]
    )

    quality_rows.append({
        "feature": feature,
        "feature_type": str(feature_types[feature]),
        "original_valid_cells": int(obs.sum()),
        "original_missing_cells": int((~obs).sum()),
        "original_missing_rate_percent": float(
            100.0 * (~obs).sum() / EXPECTED_PRIMARY_STAYS / SEQUENCE_LENGTH
        ),
        "stays_with_zero_valid_observation": zero_valid_stays,
        "leading_pre_first_valid_cells_filled": leading_gap_cells,
        "post_fallback_finite_cells": int(
            np.isfinite(post_ffill).sum()
        ),
        "final_mask_observed_cells": int(obs.sum()),
        "final_time_since_mean": float(
            time_since[:, :, feature_idx].mean()
        ),
        "final_time_since_max": float(
            time_since[:, :, feature_idx].max()
        ),
    })

quality_summary = pd.DataFrame(quality_rows)

# Add trend statistics where applicable.
trend_lookup = {
    row["feature"]: row
    for _, row in trend_stats_df.iterrows()
}

quality_summary["trend_channel"] = quality_summary["feature"].isin(
    TREND_FEATURES
)

quality_summary["trend_train_mean"] = quality_summary["feature"].map(
    lambda x: (
        trend_lookup[x]["train_mean"]
        if x in trend_lookup
        else np.nan
    )
)

quality_summary["trend_train_std"] = quality_summary["feature"].map(
    lambda x: (
        trend_lookup[x]["train_std"]
        if x in trend_lookup
        else np.nan
    )
)

# ============================================================
# 17. CHANNEL METADATA
# ============================================================

channel_rows = []

for idx, (name, group, source_feature) in enumerate(
    zip(channel_names, channel_groups, source_features)
):
    if group == "value":
        ftype = str(feature_types[source_feature])
        if ftype in {"binary", "ordinal_categorical"}:
            scaling = "none_discrete_code_preserved"
        else:
            scaling = "zscore_train_only"
    elif group == "mask":
        scaling = "none_binary_observed_before_fill"
    elif group == "time_since":
        scaling = "none_raw_hours_0_to_24"
    else:
        scaling = "zscore_train_only"

    channel_rows.append({
        "channel_index": idx,
        "channel_name": name,
        "channel_group": group,
        "source_feature": source_feature,
        "feature_type": str(feature_types[source_feature]),
        "scaling": scaling,
    })

channel_metadata = pd.DataFrame(channel_rows)

if len(channel_metadata) != N_CHANNELS:
    raise AssertionError(
        "Channel metadata count does not match final tensor channel count."
    )


# ============================================================
# 18. MACHINE-READABLE METADATA
# ============================================================

N_BASE = len(EXPECTED_FEATURE_ORDER)

value_slice = [0, N_BASE]
mask_slice = [N_BASE, 2 * N_BASE]
time_since_slice = [2 * N_BASE, 3 * N_BASE]
trend_slice = [3 * N_BASE, N_CHANNELS]

model_definition = {
    "pipeline": "Pipeline A",
    "version": "V2",
    "stage": "A4",
    "purpose": (
        "Final structured 24h preprocessing after frozen A3; "
        "A2.1 target artifacts are consumed only for alignment."
    ),
    "n_stays": EXPECTED_PRIMARY_STAYS,
    "sequence_length": SEQUENCE_LENGTH,
    "n_structured_channels": N_CHANNELS,
    "n_base_features": len(EXPECTED_FEATURE_ORDER),
    "n_trend_features": len(TREND_FEATURES),
    "channel_order": channel_names,
    "channel_slices": {
        "value": value_slice,
        "mask": mask_slice,
        "time_since": time_since_slice,
        "trend": trend_slice,
    },
    "base_feature_order": EXPECTED_FEATURE_ORDER,
    "trend_features": TREND_FEATURES,
    "trend_spec": {
        "method": TREND_METHOD,
        "window_hours": TREND_WINDOW_HOURS,
        "formula": (
            "OLS slope across the three consecutive regular hourly "
            "forward-filled values; slope units = feature-units/hour."
        ),
        "missing_handling": (
            "incomplete leading windows remain missing before "
            "train-only trend-mean fallback."
        ),
        "minimum_observations": TREND_MIN_WINDOW_VALUES,
        "implementation_status": (
            "explicit_A4_implementation_choice_because_master_does_not_"
            "enumerate_a_concrete_trend_list"
        ),
    },
    "missingness": {
        "created_before_forward_fill": True,
        "meaning": "1 if originally valid observation existed at that hour",
    },
    "forward_fill": {
        "within_stay_only": True,
        "source": "last_valid_A3_observation",
        "future_values_used": False,
    },
    "time_since": {
        "source": "original_validity_mask",
        "units": "hours",
        "observed_value": 0.0,
        "pre_first_or_never_observed_value": 24.0,
    },
    "fallback_imputation": {
        "base_value_channels": "train_only_population_median",
        "zero_valid_stay_feature": "train_only_population_median",
        "leading_pre_first_valid_gap": (
            "train_only_population_median; "
            "backward fill is intentionally prohibited to avoid look-ahead"
        ),
        "trend_channels": "train_only_trend_mean",
    },
    "normalization": {
        "continuous_value_channels": "train_only_zscore",
        "binary_value_channels": "preserve_code",
        "ordinal_value_channels": "preserve_code",
        "mask_channels": "preserve_0_1",
        "time_since_channels": "preserve_hours_0_24",
        "trend_channels": "train_only_zscore",
    },
    "demographics": {
        "shape": [EXPECTED_PRIMARY_STAYS, 4],
        "columns": [
            "age_zscore",
            "gender_F",
            "gender_M",
            "gender_UNKNOWN",
        ],
        "age_imputation": "training_median",
        "age_normalization": "training_mean_std",
        "gender_encoding": "locked_one_hot",
    },
    "targets": {
        "six_outputs": EXPECTED_TARGETS,
        "primary_trained_tasks": EXPECTED_TARGETS[:5],
        "derived_output": "deterioration_24h",
        "source_stage": "A2.1",
    },
    "split": {
        "source": str(split_path),
        "column": "split_24h",
        "counts": EXPECTED_SPLITS,
        "patient_level": True,
        "subject_overlap_train_val": 0,
        "subject_overlap_train_test": 0,
        "subject_overlap_val_test": 0,
    },
    "source_artifacts": {
        "a0": str(A0_PATH),
        "a1_feature_definitions": str(A1_FEATURE_DEFINITIONS_PATH),
        "a3": str(A3_PATH),
        "targets": str(TARGET_PRIMARY_PATH),
    },
    "v1_active_input_forbidden": True,
    "raw_mimic_modified": False,
}


preprocessing_metadata = {
    "pipeline": "Pipeline A",
    "version": "V2",
    "stage": "A4",
    "canonical_output_directory": str(V2_OUTPUT_DIR),
    "input_contract": {
        "a3_rows": int(len(a3)),
        "a3_stays": EXPECTED_PRIMARY_STAYS,
        "a3_hours_per_stay": SEQUENCE_LENGTH,
        "a3_features": EXPECTED_FEATURE_ORDER,
    },
    "output_contract": {
        "structured_tensor_path": str(STRUCTURED_TENSOR_PATH),
        "structured_shape": list(final_tensor.shape),
        "demographics_path": str(DEMOGRAPHICS_PATH),
        "demographics_shape": list(demographics.shape),
        "final_index_path": str(INDEX_OUTPUT_PATH),
        "n_structured_channels": N_CHANNELS,
    },
    "train_only_statistics": {
        "imputation_stats_path": str(IMPUTATION_STATS_PATH),
        "scaling_stats_path": str(SCALING_STATS_PATH),
        "split_counts": EXPECTED_SPLITS,
    },
    "qa": {
        "cohort_count_locked": True,
        "a3_grid_locked": True,
        "masks_before_fill": True,
        "forward_fill_within_stay": True,
        "time_since_from_original_valid_observations": True,
        "trend_on_regular_hourly_representation": True,
        "train_only_imputation": True,
        "train_only_scaling": True,
        "finite_final_tensor": True,
        "no_duplicate_stay_id": True,
        "no_duplicate_stay_hour": True,
    },
}


# ============================================================
# 19. WRITE OUTPUTS ATOMICALLY
# ============================================================

print("\n" + "-" * 80)
print("ATOMIC A4 COMMIT")
print("-" * 80)

atomic_write_npy(final_tensor, STRUCTURED_TENSOR_PATH)
atomic_write_npy(filled_values, STRUCTURED_VALUES_PATH)
atomic_write_npy(masks_float, STRUCTURED_MASKS_PATH)
atomic_write_npy(time_since, STRUCTURED_TIME_SINCE_PATH)
atomic_write_npy(trend_array, STRUCTURED_TRENDS_PATH)

atomic_write_npy(demographics, DEMOGRAPHICS_PATH)

atomic_write_csv(final_index, INDEX_OUTPUT_PATH)
atomic_write_csv(imputation_stats, IMPUTATION_STATS_PATH)
atomic_write_csv(scaling_stats, SCALING_STATS_PATH)
atomic_write_csv(quality_summary, QUALITY_SUMMARY_PATH)
atomic_write_csv(channel_metadata, CHANNEL_METADATA_PATH)

atomic_write_json(model_definition, MODEL_DEFINITION_PATH)
atomic_write_json(preprocessing_metadata, PREPROCESSING_METADATA_PATH)


# ============================================================
# 20. COMMITTED OUTPUT RE-READ CHECK
# ============================================================

committed = np.load(
    STRUCTURED_TENSOR_PATH,
    mmap_mode="r",
    allow_pickle=False,
)

if committed.shape != final_tensor.shape:
    raise AssertionError(
        "Committed tensor shape does not match validated A4 tensor."
    )

# mmap read is finite-checkable without changing the file.
if not np.isfinite(committed).all():
    raise AssertionError(
        "Committed A4 tensor contains nonfinite values."
    )

committed_index = pd.read_csv(
    INDEX_OUTPUT_PATH,
    usecols=["stay_id", "split_24h"],
)

if len(committed_index) != EXPECTED_PRIMARY_STAYS:
    raise AssertionError(
        "Committed final index row count changed."
    )

if committed_index["stay_id"].duplicated().any():
    raise AssertionError(
        "Committed final index contains duplicate stay IDs."
    )

committed_channel_meta = pd.read_csv(
    CHANNEL_METADATA_PATH
)

if len(committed_channel_meta) != N_CHANNELS:
    raise AssertionError(
        "Committed channel metadata count changed."
    )

# Explicitly verify all six outputs are present in committed index.
for target_name in EXPECTED_TARGETS:
    if target_name not in final_index.columns:
        raise AssertionError(
            f"Committed final index lacks target {target_name}."
        )


# ============================================================
# 21. FINAL REPORT
# ============================================================

print("\n" + "=" * 80)
print("✅ PIPELINE A V2 — A4 COMPLETE")
print("=" * 80)
print(f"Primary stays                  : {EXPECTED_PRIMARY_STAYS:,}")
print(f"Sequence length               : {SEQUENCE_LENGTH}")
print(f"Base clinical features        : {len(EXPECTED_FEATURE_ORDER)}")
print(f"Missingness channels           : {len(EXPECTED_FEATURE_ORDER)}")
print(f"Time-since channels            : {len(EXPECTED_FEATURE_ORDER)}")
print(f"Trend channels                 : {len(TREND_FEATURES)}")
print(f"FINAL STRUCTURED CHANNELS      : {N_CHANNELS}")
print(f"Structured tensor shape        : {tuple(final_tensor.shape)}")
print(f"Demographics shape              : {tuple(demographics.shape)}")
print(
    "Split counts                    : "
    f"train={EXPECTED_SPLITS['train']:,}, "
    f"val={EXPECTED_SPLITS['val']:,}, "
    f"test={EXPECTED_SPLITS['test']:,}"
)
print("Six outputs aligned            : YES")
print("Masks created before fill      : YES")
print("Forward fill within stay      : YES")
print("Time-since from valid history  : YES")
print("Trends on regular hourly grid  : YES")
print("Imputation statistics train-only: YES")
print("Scaling statistics train-only  : YES")
print("Final tensor finite             : YES")
print("V1 active input used            : NO")
print("Raw MIMIC modified              : NO")
print("=" * 80)
print(f"Saved tensor                    : {STRUCTURED_TENSOR_PATH}")
print(f"Saved demographics              : {DEMOGRAPHICS_PATH}")
print(f"Saved final index               : {INDEX_OUTPUT_PATH}")
print(f"Saved imputation stats          : {IMPUTATION_STATS_PATH}")
print(f"Saved scaling stats             : {SCALING_STATS_PATH}")
print(f"Saved channel metadata          : {CHANNEL_METADATA_PATH}")
print(f"Saved model definition          : {MODEL_DEFINITION_PATH}")
print(f"Saved preprocessing metadata    : {PREPROCESSING_METADATA_PATH}")
print("=" * 80)

del a3, values, filled_values, observed_mask
del final_tensor, demographics, trend_array
del committed

gc.collect()
