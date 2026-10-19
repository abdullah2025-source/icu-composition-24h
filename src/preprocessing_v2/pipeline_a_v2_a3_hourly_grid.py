#!/usr/bin/env python3

"""
Pipeline A V2
Stage A3 — Structured Hourly Aggregation + Complete 24h Grid

PURPOSE
-------
Build the frozen 24-hour structured representation used by Pipeline A V2.

A3 consumes ONLY the frozen A1 validated event stream and the frozen TGT
primary cohort. It performs:
    1. hourly aggregation
    2. complete 24-hour grid construction (hours 0..23)

A3 deliberately DOES NOT perform:
    - missingness-mask creation for the final model tensor
    - forward fill
    - time-since-last-observation
    - trend/derivative features
    - population-median imputation
    - scaling / normalization
    - text processing
    - target construction
    - model training

Those operations belong to A4 or later stages.

V2 SAFETY RULES
---------------
* Raw MIMIC-IV files are read only.
* A0/A1/A2/TGT artifacts are not modified.
* V1 outputs are never modified.
* The final A3 parquet is written atomically only after all hard checks pass.
* Feature definitions are loaded from the frozen A1 metadata rather than
  re-invented ad hoc in the sequence-construction code.

CANONICAL INPUTS
----------------
A0:
    /home/abdullah2025/fyp_project/outputs/pipeline_a_v2/cohort_master.csv

A1:
    /home/abdullah2025/fyp_project/outputs/pipeline_a_v2/a1_validated_events.parquet
    /home/abdullah2025/fyp_project/outputs/pipeline_a_v2/a1_feature_definitions.json

TGT:
    /home/abdullah2025/fyp_project/outputs/pipeline_a_v2/targets_24h/
        outcomes_24h_primary_cohort.csv

CANONICAL OUTPUTS
-----------------
    /home/abdullah2025/fyp_project/outputs/pipeline_a_v2/vitals_labs_hourly.parquet
    /home/abdullah2025/fyp_project/outputs/pipeline_a_v2/a3_hourly_summary.csv
    /home/abdullah2025/fyp_project/outputs/pipeline_a_v2/a3_metadata.json

The filename `vitals_labs_hourly.parquet` is retained for downstream
compatibility, but ONLY under the V2 output directory. It is not the V1 file.
"""

from pathlib import Path
import gc
import json
import math
import os
import warnings

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm


# ============================================================
# 0. GLOBAL PATHS + LOCKS
# ============================================================

PROJECT_ROOT = Path("/home/abdullah2025/fyp_project")
V1_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "pipeline_a"
V2_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "pipeline_a_v2"
TARGET_DIR = V2_OUTPUT_DIR / "targets_24h"

A0_COHORT_PATH = V2_OUTPUT_DIR / "cohort_master.csv"
A1_EVENT_PATH = V2_OUTPUT_DIR / "a1_validated_events.parquet"
A1_FEATURE_DEFINITION_PATH = (
    V2_OUTPUT_DIR / "a1_feature_definitions.json"
)
PRIMARY_TARGET_PATH = TARGET_DIR / "outcomes_24h_primary_cohort.csv"

OUT_PATH = V2_OUTPUT_DIR / "vitals_labs_hourly.parquet"
SUMMARY_PATH = V2_OUTPUT_DIR / "a3_hourly_summary.csv"
METADATA_PATH = V2_OUTPUT_DIR / "a3_metadata.json"

EXPECTED_A0_STAYS = 93_224
EXPECTED_PRIMARY_STAYS = 74_572
LANDMARK_HOURS = 24
GRID_HOURS = tuple(range(LANDMARK_HOURS))
EXPECTED_GRID_ROWS = EXPECTED_PRIMARY_STAYS * LANDMARK_HOURS

EVENT_BATCH_SIZE = 1_000_000

if V1_OUTPUT_DIR.resolve() == V2_OUTPUT_DIR.resolve():
    raise RuntimeError(
        "SAFETY ERROR: V1 and V2 output directories resolve to the same path."
    )

V2_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TARGET_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. HELPERS
# ============================================================

def require_file(path: Path, description: str):
    if not path.exists():
        raise FileNotFoundError(
            f"Required {description} not found:\n{path}"
        )


def atomic_dataframe_to_parquet(df: pd.DataFrame, final_path: Path):
    """Write parquet to a temporary sibling and atomically commit it."""
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = final_path.with_name(final_path.name + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    df.to_parquet(
        tmp_path,
        index=False,
        engine="pyarrow",
    )
    os.replace(tmp_path, final_path)


def atomic_dataframe_to_csv(df: pd.DataFrame, final_path: Path):
    """Write CSV to a temporary sibling and atomically commit it."""
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = final_path.with_name(final_path.name + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, final_path)


def atomic_json_write(obj, final_path: Path):
    """Write JSON to a temporary sibling and atomically commit it."""
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = final_path.with_name(final_path.name + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(
            obj,
            handle,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    os.replace(tmp_path, final_path)


def assert_same_values(left: pd.Series, right: pd.Series, label: str):
    """Strict equality including NA==NA for metadata audit fields."""
    same = left.eq(right) | (left.isna() & right.isna())
    if not bool(same.all()):
        mismatch_count = int((~same).sum())
        raise AssertionError(
            f"{label} mismatch detected in {mismatch_count:,} rows."
        )


def load_feature_definitions():
    with open(
        A1_FEATURE_DEFINITION_PATH,
        "r",
        encoding="utf-8",
    ) as handle:
        definitions = json.load(handle)

    feature_order = definitions.get("feature_order")
    feature_types = definitions.get("feature_types", {})
    declared_aggregation = definitions.get(
        "aggregation_rules_for_A2",
        {},
    )

    if not isinstance(feature_order, list):
        raise AssertionError(
            "A1 feature definition JSON does not contain a feature_order list."
        )

    if len(feature_order) != 28:
        raise AssertionError(
            f"Expected frozen V2 A1 feature count = 28; found {len(feature_order):,}."
        )

    if len(set(feature_order)) != len(feature_order):
        raise AssertionError("A1 feature_order contains duplicate feature names.")

    missing_types = [
        feature
        for feature in feature_order
        if feature not in feature_types
    ]
    if missing_types:
        raise AssertionError(
            "A1 feature definition is missing feature types for: "
            + ", ".join(missing_types)
        )

    # V2 stage-local aggregation policy.
    # The master spec explicitly locks median for continuous numeric variables
    # and maximum for ordinal respiratory support. For binary/count-valued
    # discrete features, max preserves a valid discrete code and avoids
    # fractional codes when multiple observations occur inside one hour.
    aggregation = {}
    for feature in feature_order:
        ftype = str(feature_types[feature])
        if ftype == "ordinal_categorical":
            aggregation[feature] = "max"
        elif ftype == "binary":
            aggregation[feature] = "max"
        elif ftype == "numeric_count_0_to_3":
            aggregation[feature] = "max"
        else:
            aggregation[feature] = "median"

    return (
        definitions,
        feature_order,
        feature_types,
        aggregation,
        declared_aggregation,
    )


# ============================================================
# 2. STARTUP
# ============================================================

print("=" * 80)
print("PIPELINE A V2 — A3 STRUCTURED HOURLY AGGREGATION + COMPLETE 24H GRID")
print("=" * 80)

for path, description in [
    (A0_COHORT_PATH, "A0 cohort"),
    (A1_EVENT_PATH, "A1 validated event stream"),
    (A1_FEATURE_DEFINITION_PATH, "A1 feature definition"),
    (PRIMARY_TARGET_PATH, "TGT primary target cohort"),
]:
    require_file(path, description)

(
    a1_definitions,
    feature_order,
    feature_types,
    aggregation_rules,
    declared_aggregation_rules,
) = load_feature_definitions()

print(f"A1 frozen feature count      : {len(feature_order)}")
print("Feature order:")
for idx, feature in enumerate(feature_order, start=1):
    print(f"  {idx:02d}. {feature}")

print("\nA3 aggregation rules:")
for feature in feature_order:
    print(
        f"  {feature:28s} "
        f"type={feature_types[feature]:24s} "
        f"rule={aggregation_rules[feature]}"
    )

# ============================================================
# 3. LOAD + VALIDATE FROZEN A0 / TGT PRIMARY COHORT
# ============================================================

print("\n" + "-" * 80)
print("SOURCE: FROZEN A0 + FROZEN TGT PRIMARY COHORT")
print("-" * 80)

cohort = pd.read_csv(A0_COHORT_PATH)
if len(cohort) != EXPECTED_A0_STAYS:
    raise AssertionError(
        f"Expected {EXPECTED_A0_STAYS:,} A0 rows; found {len(cohort):,}."
    )

required_a0 = [
    "subject_id",
    "hadm_id",
    "stay_id",
    "intime",
    "outtime",
    "icu_los_hours",
]
missing_a0 = [col for col in required_a0 if col not in cohort.columns]
if missing_a0:
    raise AssertionError(
        f"A0 cohort missing required columns: {missing_a0}"
    )

cohort["subject_id"] = pd.to_numeric(
    cohort["subject_id"], errors="raise"
).astype("int64")
cohort["hadm_id"] = pd.to_numeric(
    cohort["hadm_id"], errors="raise"
).astype("int64")
cohort["stay_id"] = pd.to_numeric(
    cohort["stay_id"], errors="raise"
).astype("int64")
cohort["intime"] = pd.to_datetime(
    cohort["intime"], errors="coerce"
)
cohort["outtime"] = pd.to_datetime(
    cohort["outtime"], errors="coerce"
)
cohort["icu_los_hours"] = pd.to_numeric(
    cohort["icu_los_hours"], errors="coerce"
)

if cohort["stay_id"].duplicated().any():
    raise AssertionError("Frozen A0 cohort contains duplicate stay_id.")
if cohort["intime"].isna().any() or cohort["outtime"].isna().any():
    raise AssertionError("Frozen A0 cohort contains missing ICU timestamps.")
if (cohort["outtime"] <= cohort["intime"]).any():
    raise AssertionError("Frozen A0 cohort contains outtime <= intime.")
if not np.isfinite(cohort["icu_los_hours"].to_numpy(dtype=np.float64)).all():
    raise AssertionError("Frozen A0 cohort contains non-finite ICU LOS.")
if (cohort["icu_los_hours"] < 6.0).any():
    raise AssertionError("Frozen A0 cohort contains ICU LOS < 6 hours.")

primary = pd.read_csv(
    PRIMARY_TARGET_PATH,
    usecols=["subject_id", "hadm_id", "stay_id"],
)

if len(primary) != EXPECTED_PRIMARY_STAYS:
    raise AssertionError(
        f"Expected {EXPECTED_PRIMARY_STAYS:,} final primary target rows; "
        f"found {len(primary):,}."
    )

for col in ["subject_id", "hadm_id", "stay_id"]:
    primary[col] = pd.to_numeric(
        primary[col], errors="raise"
    ).astype("int64")

if primary["stay_id"].duplicated().any():
    raise AssertionError(
        "TGT primary cohort contains duplicate stay_id."
    )

# Exact one-to-one identity audit against A0.
a0_key = cohort[
    ["stay_id", "subject_id", "hadm_id"]
].copy()

audit = primary.merge(
    a0_key,
    on="stay_id",
    how="left",
    validate="one_to_one",
    suffixes=("_target", "_a0"),
    indicator=True,
)

if (audit["_merge"] != "both").any():
    raise AssertionError(
        "Some TGT primary stays are not present in frozen A0 cohort."
    )

assert_same_values(
    audit["subject_id_target"],
    audit["subject_id_a0"],
    "subject_id TGT vs A0",
)
assert_same_values(
    audit["hadm_id_target"],
    audit["hadm_id_a0"],
    "hadm_id TGT vs A0",
)

del audit, a0_key, cohort

gc.collect()

primary_stays = primary["stay_id"].to_numpy(dtype=np.int64)
primary_stay_set = set(primary_stays.tolist())

if len(primary_stay_set) != EXPECTED_PRIMARY_STAYS:
    raise AssertionError("Primary stay_id set cardinality is incorrect.")

print(f"Frozen A0 stays             : {EXPECTED_A0_STAYS:,}")
print(f"Frozen TGT primary stays    : {EXPECTED_PRIMARY_STAYS:,}")
print("Exact stay/subject/hadm identity against A0: PASS")


# ============================================================
# 4. STREAM A1 VALIDATED EVENTS
# ============================================================

print("\n" + "-" * 80)
print("SOURCE: A1 VALIDATED EVENT STREAM")
print("-" * 80)

required_event_columns = {
    "subject_id",
    "hadm_id",
    "stay_id",
    "charttime",
    "event_time_hours",
    "icu_hour",
    "feature",
    "value",
    "value_text",
    "itemid",
    "source",
    "valueuom",
    "warning_flag",
}

parquet_file = pq.ParquetFile(A1_EVENT_PATH)
actual_columns = set(parquet_file.schema_arrow.names)
missing_event_columns = sorted(required_event_columns - actual_columns)
if missing_event_columns:
    raise AssertionError(
        "A1 event stream is missing required columns: "
        + ", ".join(missing_event_columns)
    )

read_columns = [
    "subject_id",
    "hadm_id",
    "stay_id",
    "charttime",
    "event_time_hours",
    "icu_hour",
    "feature",
    "value",
    "value_text",
    "itemid",
    "source",
    "valueuom",
    "warning_flag",
]

# Read only the columns needed by A3.  A1 is a single Parquet event stream,
# so use Arrow batch iteration to keep peak memory bounded while scanning it.
selected_parts = []
rows_scanned = 0
rows_primary = 0

batch_iter = parquet_file.iter_batches(
    batch_size=EVENT_BATCH_SIZE,
    columns=read_columns,
    use_threads=True,
)

for batch_index, batch in enumerate(
    tqdm(batch_iter, desc="Scanning A1 event batches"),
    start=1,
):
    df = batch.to_pandas()
    rows_scanned += len(df)

    if df.empty:
        del df
        continue

    df["stay_id"] = pd.to_numeric(
        df["stay_id"], errors="raise"
    ).astype("int64")

    df = df[
        df["stay_id"].isin(primary_stay_set)
    ].copy()

    if df.empty:
        del df
        continue

    rows_primary += len(df)

    # All A1 validated values entering A3 must be finite numeric values.
    df["value"] = pd.to_numeric(
        df["value"], errors="coerce"
    )
    if df["value"].isna().any():
        raise AssertionError(
            "A1 validated event stream contains null/non-numeric values "
            "after primary-cohort filtering."
        )

    if not np.isfinite(
        df["value"].to_numpy(dtype=np.float64)
    ).all():
        raise AssertionError(
            "A1 validated event stream contains non-finite values."
        )

    df["icu_hour"] = pd.to_numeric(
        df["icu_hour"], errors="coerce"
    )
    df["event_time_hours"] = pd.to_numeric(
        df["event_time_hours"], errors="coerce"
    )

    if df["icu_hour"].isna().any() or df["event_time_hours"].isna().any():
        raise AssertionError(
            "A1 validated event stream contains missing temporal fields."
        )

    df["icu_hour"] = df["icu_hour"].astype("int32")
    df["event_time_hours"] = df["event_time_hours"].astype("float64")

    # A3 is strictly first 24 ICU hours.
    if not (
        (df["event_time_hours"] >= 0.0)
        & (df["event_time_hours"] < 24.0)
    ).all():
        raise AssertionError(
            "A1 validated events for primary stays contain event times outside [0,24)."
        )

    expected_hour = np.floor(
        df["event_time_hours"].to_numpy(dtype=np.float64)
    ).astype(np.int32)

    if not np.array_equal(
        expected_hour,
        df["icu_hour"].to_numpy(dtype=np.int32),
    ):
        raise AssertionError(
            "A1 icu_hour does not equal floor(event_time_hours)."
        )

    if not (
        (df["icu_hour"] >= 0)
        & (df["icu_hour"] <= 23)
    ).all():
        raise AssertionError(
            "A1 validated events contain icu_hour outside 0..23."
        )

    # A1 validated primary stream should already have warning_flag=0.
    warning_values = pd.to_numeric(
        df["warning_flag"], errors="coerce"
    ).fillna(-1)
    if not warning_values.eq(0).all():
        raise AssertionError(
            "A1 primary validated event stream contains non-zero warning_flag values."
        )

    df["feature"] = df["feature"].astype(str)

    unknown_features = sorted(
        set(df["feature"]) - set(feature_order)
    )
    if unknown_features:
        raise AssertionError(
            "A1 primary event stream contains unknown features: "
            + ", ".join(unknown_features)
        )

    selected_parts.append(
        df[
            [
                "subject_id",
                "hadm_id",
                "stay_id",
                "icu_hour",
                "feature",
                "value",
            ]
        ].copy()
    )

    del df, batch

print(f"A1 rows scanned             : {rows_scanned:,}")
print(f"A1 rows for primary stays   : {rows_primary:,}")

if not selected_parts:
    raise RuntimeError(
        "No A1 validated events were found for the frozen primary cohort."
    )

events = pd.concat(
    selected_parts,
    ignore_index=True,
)

del selected_parts, parquet_file, batch_iter

gc.collect()

# Exact identity of primary rows in A1.
unknown_primary_stays = sorted(
    set(events["stay_id"].unique().tolist()) - primary_stay_set
)
if unknown_primary_stays:
    raise AssertionError(
        "Unexpected non-primary stay IDs remain in A3 event stream."
    )

print(f"Filtered validated events     : {len(events):,}")
print(f"Unique primary stays in events: {events['stay_id'].nunique():,}")


# ============================================================
# 5. HOURLY AGGREGATION
# ============================================================

print("\n" + "-" * 80)
print("HOURLY AGGREGATION")
print("-" * 80)

# Grouping keys include only the canonical temporal identity and feature.
# A1 has already removed invalid observations; consequently these are the
# values eligible to contribute to the hourly aggregate.

aggregated_parts = []

for feature in tqdm(
    feature_order,
    desc="Aggregating features",
):
    sub = events.loc[
        events["feature"].eq(feature),
        ["stay_id", "icu_hour", "value"],
    ].copy()

    if sub.empty:
        # Keep an empty representation for this feature; the complete grid
        # below will explicitly produce all-NaN cells for it.
        continue

    rule = aggregation_rules[feature]

    if rule == "median":
        agg = (
            sub.groupby(
                ["stay_id", "icu_hour"],
                as_index=False,
                observed=True,
            )["value"]
            .median()
            .rename(columns={"value": feature})
        )
    elif rule == "max":
        agg = (
            sub.groupby(
                ["stay_id", "icu_hour"],
                as_index=False,
                observed=True,
            )["value"]
            .max()
            .rename(columns={"value": feature})
        )
    else:
        raise AssertionError(
            f"Unsupported A3 aggregation rule '{rule}' for {feature}."
        )

    aggregated_parts.append(agg)
    del sub, agg

if aggregated_parts:
    hourly_observed = aggregated_parts[0]
    for part in aggregated_parts[1:]:
        hourly_observed = hourly_observed.merge(
            part,
            on=["stay_id", "icu_hour"],
            how="outer",
            validate="one_to_one",
        )
        del part
else:
    hourly_observed = pd.DataFrame(
        columns=["stay_id", "icu_hour"] + feature_order
    )

del aggregated_parts, events

gc.collect()

# Normalize identifier dtypes.
hourly_observed["stay_id"] = pd.to_numeric(
    hourly_observed["stay_id"], errors="raise"
).astype("int64")
hourly_observed["icu_hour"] = pd.to_numeric(
    hourly_observed["icu_hour"], errors="raise"
).astype("int32")

for feature in feature_order:
    if feature in hourly_observed.columns:
        hourly_observed[feature] = pd.to_numeric(
            hourly_observed[feature], errors="coerce"
        ).astype("float32")
    else:
        hourly_observed[feature] = np.nan

# Reorder columns exactly.
hourly_observed = hourly_observed[
    ["stay_id", "icu_hour"] + feature_order
].copy()

if hourly_observed.duplicated(
    subset=["stay_id", "icu_hour"]
).any():
    raise AssertionError(
        "Duplicate stay_id + icu_hour rows exist after hourly aggregation."
    )

# Discrete-code hard checks before the complete grid.
binary_features = [
    feature
    for feature in feature_order
    if feature_types[feature] == "binary"
]
count_features = [
    feature
    for feature in feature_order
    if feature_types[feature] == "numeric_count_0_to_3"
]
ordinal_features = [
    feature
    for feature in feature_order
    if feature_types[feature] == "ordinal_categorical"
]

for feature in binary_features:
    vals = hourly_observed[feature].dropna().to_numpy(dtype=np.float64)
    if vals.size and not np.isin(vals, [0.0, 1.0]).all():
        raise AssertionError(
            f"Binary feature {feature} contains values outside 0/1 after aggregation."
        )

for feature in count_features:
    vals = hourly_observed[feature].dropna().to_numpy(dtype=np.float64)
    if vals.size and not np.isin(vals, [0.0, 1.0, 2.0, 3.0]).all():
        raise AssertionError(
            f"Count feature {feature} contains values outside 0..3 after aggregation."
        )

for feature in ordinal_features:
    vals = hourly_observed[feature].dropna().to_numpy(dtype=np.float64)
    if vals.size and not np.isin(vals, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).all():
        raise AssertionError(
            f"Ordinal feature {feature} contains values outside 0..5 after aggregation."
        )

if "resp_support_mode" in hourly_observed.columns:
    resp_values = hourly_observed["resp_support_mode"].dropna().to_numpy(
        dtype=np.float64
    )
    if resp_values.size and not np.isin(
        resp_values,
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
    ).all():
        raise AssertionError(
            "resp_support_mode contains an impossible code after aggregation."
        )

print(
    f"Observed stay-hour cells      : {len(hourly_observed):,}"
)
print(
    f"Observed stays represented    : {hourly_observed['stay_id'].nunique():,}"
)


# ============================================================
# 6. COMPLETE 24-HOUR GRID
# ============================================================

print("\n" + "-" * 80)
print("COMPLETE 24-HOUR GRID: HOURS 0..23")
print("-" * 80)

# Build exactly 24 rows per frozen primary stay, regardless of how many
# observations exist. Missing cells remain NaN. No fill occurs here.
primary_frame = primary[["stay_id", "subject_id", "hadm_id"]].copy()

primary_frame = primary_frame.rename(
    columns={
        "stay_id": "stay_id",
    }
)

# Vectorized Cartesian product: primary stays × 24 hours.
grid = pd.DataFrame({
    "stay_id": np.repeat(
        primary_frame["stay_id"].to_numpy(dtype=np.int64),
        LANDMARK_HOURS,
    ),
    "icu_hour": np.tile(
        np.arange(LANDMARK_HOURS, dtype=np.int32),
        len(primary_frame),
    ),
})

# Attach subject/hospital admission identifiers exactly once per stay.
grid = grid.merge(
    primary_frame,
    on="stay_id",
    how="left",
    validate="many_to_one",
)

# Attach observed hourly aggregates. Any absent hour remains NaN.
grid = grid.merge(
    hourly_observed,
    on=["stay_id", "icu_hour"],
    how="left",
    validate="one_to_one",
)

del primary_frame, primary, hourly_observed

gc.collect()

# Canonical column order.
grid = grid[
    [
        "subject_id",
        "hadm_id",
        "stay_id",
        "icu_hour",
    ] + feature_order
].copy()

# Reorder primary identity first in a stable way after merge.
grid = grid.sort_values(
    ["stay_id", "icu_hour"],
    kind="mergesort",
).reset_index(drop=True)


# ============================================================
# 7. COMPLETE-GRID HARD CHECKS
# ============================================================

print("\n" + "-" * 80)
print("A3 HARD CHECKS")
print("-" * 80)

if len(grid) != EXPECTED_GRID_ROWS:
    raise AssertionError(
        f"Expected exactly {EXPECTED_GRID_ROWS:,} grid rows "
        f"({EXPECTED_PRIMARY_STAYS:,} stays × 24 hours); found {len(grid):,}."
    )

if grid["stay_id"].nunique() != EXPECTED_PRIMARY_STAYS:
    raise AssertionError(
        "A3 grid does not contain exactly the frozen 74,572 primary stays."
    )

if grid["stay_id"].isna().any():
    raise AssertionError("A3 grid contains missing stay_id.")
if grid["subject_id"].isna().any():
    raise AssertionError("A3 grid contains missing subject_id.")
if grid["hadm_id"].isna().any():
    raise AssertionError("A3 grid contains missing hadm_id.")
if grid["icu_hour"].isna().any():
    raise AssertionError("A3 grid contains missing icu_hour.")

if grid.duplicated(
    subset=["stay_id", "icu_hour"]
).any():
    raise AssertionError(
        "A3 grid contains duplicate stay_id + icu_hour rows."
    )

hour_values = grid["icu_hour"].to_numpy(dtype=np.int32)
if not np.isin(hour_values, np.arange(24, dtype=np.int32)).all():
    raise AssertionError("A3 grid contains an hour outside 0..23.")

hours_per_stay = grid.groupby("stay_id")["icu_hour"].agg(
    ["count", "min", "max"]
)

if not (hours_per_stay["count"] == 24).all():
    raise AssertionError("Not every primary stay has exactly 24 hourly rows.")
if not (hours_per_stay["min"] == 0).all():
    raise AssertionError("Some primary stays do not begin at hour 0.")
if not (hours_per_stay["max"] == 23).all():
    raise AssertionError("Some primary stays do not end at hour 23.")

del hours_per_stay

# Verify values are finite wherever observed; NaN is allowed and expected.
for feature in feature_order:
    values = grid[feature].to_numpy(dtype=np.float64)
    finite_or_nan = np.isfinite(values) | np.isnan(values)
    if not finite_or_nan.all():
        raise AssertionError(
            f"Feature {feature} contains invalid numeric states."
        )

# Re-check discrete feature domains on the completed grid (NaN allowed).
for feature in binary_features:
    vals = grid[feature].dropna().to_numpy(dtype=np.float64)
    if vals.size and not np.isin(vals, [0.0, 1.0]).all():
        raise AssertionError(
            f"Completed-grid binary feature {feature} contains non-binary values."
        )

for feature in count_features:
    vals = grid[feature].dropna().to_numpy(dtype=np.float64)
    if vals.size and not np.isin(vals, [0.0, 1.0, 2.0, 3.0]).all():
        raise AssertionError(
            f"Completed-grid count feature {feature} contains values outside 0..3."
        )

for feature in ordinal_features:
    vals = grid[feature].dropna().to_numpy(dtype=np.float64)
    if vals.size and not np.isin(vals, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).all():
        raise AssertionError(
            f"Completed-grid ordinal feature {feature} contains values outside 0..5."
        )

# Explicitly assert that A3 did not accidentally perform common A4 operations.
for feature in feature_order:
    # At this stage missing cells must still exist if the feature was not fully
    # observed across every stay/hour. A fully observed feature is acceptable,
    # so we only verify that no auxiliary mask/fill/time-since columns appeared.
    if f"{feature}_missing" in grid.columns:
        raise AssertionError(
            "A3 must not create final missingness-mask columns."
        )
    if f"{feature}_time_since" in grid.columns:
        raise AssertionError(
            "A3 must not create time-since columns."
        )

# Do not search for generic fragments such as "rate" across all columns:
# legitimate frozen A1 features include `heart_rate` and `resp_rate`.
# Instead, verify that A3 contains only the canonical identity columns and
# the 28 frozen A1 feature columns. Any later-stage auxiliary column (mask,
# time-since, trend, delta, slope, rate, z-score, normalized value, etc.)
# would therefore be rejected as an unexpected column without false
# positives on legitimate feature names.
allowed_columns = {
    "subject_id",
    "hadm_id",
    "stay_id",
    "icu_hour",
    *feature_order,
}

unexpected_columns = sorted(
    set(grid.columns) - allowed_columns
)

if unexpected_columns:
    raise AssertionError(
        "A3 output contains unexpected later-stage or auxiliary columns: "
        + ", ".join(unexpected_columns)
    )

print(f"Grid rows                     : {len(grid):,}")
print(f"Unique stays                  : {grid['stay_id'].nunique():,}")
print("24 rows/stay                  : PASS")
print("Hours 0..23                   : PASS")
print("Unique stay/hour              : PASS")
print("No A4 masks / fill / trends   : PASS")
print("Feature domain checks         : PASS")


# ============================================================
# 8. A3 SUMMARY / QA TABLE
# ============================================================

print("\n" + "-" * 80)
print("A3 FEATURE SUMMARY")
print("-" * 80)

summary_rows = []

total_grid_cells = EXPECTED_PRIMARY_STAYS * LANDMARK_HOURS

# Count event rows directly from the final aggregate source is no longer
# available, so use the number of nonmissing hourly cells as the canonical
# representation-level observation count, and retain event-row counts from
# a fresh small scan below only if needed. For A3 QA, cell counts are the
# relevant post-aggregation metric.
for feature in feature_order:
    nonmissing = int(grid[feature].notna().sum())
    missing = int(grid[feature].isna().sum())
    missing_rate = (
        100.0 * missing / total_grid_cells
        if total_grid_cells > 0
        else 0.0
    )

    summary_rows.append({
        "feature_index": int(feature_order.index(feature) + 1),
        "feature": feature,
        "feature_type": str(feature_types[feature]),
        "aggregation_rule": aggregation_rules[feature],
        "observed_hour_cells": nonmissing,
        "missing_hour_cells": missing,
        "missing_rate_percent": missing_rate,
    })

summary_df = pd.DataFrame(summary_rows)

if len(summary_df) != 28:
    raise AssertionError(
        "A3 summary does not contain exactly 28 features."
    )

if set(summary_df["feature"]) != set(feature_order):
    raise AssertionError(
        "A3 summary feature set does not match canonical A1 feature set."
    )

print(summary_df.to_string(index=False))


# ============================================================
# 9. METADATA
# ============================================================

metadata = {
    "pipeline": "Pipeline A",
    "version": "V2",
    "stage": "A3",
    "stage_name": "Structured hourly aggregation + complete 24h grid",
    "source_mimic_version": "3.1",
    "v1_output_directory_preserved": str(V1_OUTPUT_DIR),
    "v2_output_directory": str(V2_OUTPUT_DIR),
    "raw_mimic_read_only": True,
    "input_a1_event_stream": str(A1_EVENT_PATH),
    "input_a1_feature_definition": str(A1_FEATURE_DEFINITION_PATH),
    "input_primary_target_cohort": str(PRIMARY_TARGET_PATH),
    "a0_expected_stays": EXPECTED_A0_STAYS,
    "primary_expected_stays": EXPECTED_PRIMARY_STAYS,
    "landmark_hours": LANDMARK_HOURS,
    "hours": list(GRID_HOURS),
    "expected_grid_rows": EXPECTED_GRID_ROWS,
    "actual_grid_rows": int(len(grid)),
    "feature_count": len(feature_order),
    "feature_order": feature_order,
    "feature_types": {
        feature: str(feature_types[feature])
        for feature in feature_order
    },
    "aggregation_rules": aggregation_rules,
    "a1_declared_aggregation_rules": declared_aggregation_rules,
    "a1_declared_feature_count": a1_definitions.get("feature_count"),
    "a1_rows_scanned": int(rows_scanned),
    "a1_primary_event_rows_scanned": int(rows_primary),
    "a3_operations": [
        "hourly aggregation",
        "complete 24-hour grid",
    ],
    "a3_excluded_operations": [
        "final missingness-mask creation",
        "forward fill",
        "time-since-last-observation",
        "trend/derivative features",
        "population-median fallback",
        "normalization/scaling",
    ],
    "complete_grid_check_passed": True,
    "unique_stay_hour_check_passed": True,
    "feature_domain_checks_passed": True,
    "no_a4_operations_performed": True,
    "grid_rows_per_stay": 24,
    "output_path": str(OUT_PATH),
    "summary_path": str(SUMMARY_PATH),
    "metadata_path": str(METADATA_PATH),
}


# ============================================================
# 10. ATOMIC COMMIT
# ============================================================

print("\n" + "-" * 80)
print("ATOMIC OUTPUT COMMIT")
print("-" * 80)

atomic_dataframe_to_parquet(
    grid,
    OUT_PATH,
)

atomic_dataframe_to_csv(
    summary_df,
    SUMMARY_PATH,
)

atomic_json_write(
    metadata,
    METADATA_PATH,
)

# Verify committed parquet by re-reading only identity columns + one feature
# from the actual committed file. This catches accidental partial writes.
committed = pd.read_parquet(
    OUT_PATH,
    engine="pyarrow",
    columns=["subject_id", "hadm_id", "stay_id", "icu_hour", feature_order[0]],
)

if len(committed) != EXPECTED_GRID_ROWS:
    raise AssertionError(
        "Committed A3 parquet row count does not match the validated grid."
    )

if committed["stay_id"].nunique() != EXPECTED_PRIMARY_STAYS:
    raise AssertionError(
        "Committed A3 parquet unique stay count is incorrect."
    )

if committed.duplicated(
    subset=["stay_id", "icu_hour"]
).any():
    raise AssertionError(
        "Committed A3 parquet contains duplicate stay/hour rows."
    )

del committed, grid, summary_df

gc.collect()

print("\n" + "=" * 80)
print("✅ PIPELINE A V2 — A3 COMPLETE")
print("=" * 80)
print(f"Primary stays                 : {EXPECTED_PRIMARY_STAYS:,}")
print(f"Hours per stay                : {LANDMARK_HOURS}")
print(f"Grid rows                     : {EXPECTED_GRID_ROWS:,}")
print(f"Structured features           : {len(feature_order)}")
print(f"Saved hourly grid             : {OUT_PATH}")
print(f"Saved feature summary         : {SUMMARY_PATH}")
print(f"Saved metadata                : {METADATA_PATH}")
print("A3 performs aggregation + complete grid only.")
print("A4 must perform masks, forward-fill, time-since, trends, train-only")
print("fallback imputation, and train-only scaling.")
print("V1 outputs were not modified.")
print("Raw MIMIC-IV files were not modified.")
print("=" * 80)
