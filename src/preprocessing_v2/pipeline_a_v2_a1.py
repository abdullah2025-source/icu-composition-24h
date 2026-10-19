#!/usr/bin/env python3

"""
Pipeline A V2
Stage A1 — Raw Structured EHR Extraction + Quality Control

A1 PURPOSE
----------
A1 performs event-level extraction and quality control only.

A1 intentionally stops BEFORE:
    - hourly aggregation
    - complete hourly grid construction
    - missingness masks
    - forward fill
    - time-since-last-observation
    - trend/derivative features
    - train-only fallback imputation
    - train-only scaling

Those belong to A2 / A3 / A4.

V2 PRIMARY EVENT OUTPUT
-----------------------
    a1_validated_events.parquet

V2 QC / AUDIT OUTPUTS
---------------------
    feature_quality_summary.csv
    invalid_measurement_counts.csv
    a1_unit_audit.csv
    labevents_reference_audit.csv
    fio2_unit_audit.csv
    a1_feature_mapping_audit.csv
    a1_gcs_component_audit.csv
    a1_feature_definitions.json
    a1_metadata.json

IMPORTANT SAFETY RULES
----------------------
    * V1 outputs are never modified.
    * Raw MIMIC-IV files are read-only.
    * V2 writes only under pipeline_a_v2/.
    * Large outputs use temporary files + atomic replacement.
"""

from pathlib import Path
from collections import defaultdict
import json
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm


# ============================================================
# 0. GLOBAL PATHS
# ============================================================

PROJECT_ROOT = Path("/home/abdullah2025/fyp_project")

MIMIC_IV_DIR = (
    PROJECT_ROOT
    / "data"
    / "mimic_iv"
    / "mimic-iv-3.1"
)

HOSP_DIR = MIMIC_IV_DIR / "hosp"
ICU_DIR = MIMIC_IV_DIR / "icu"

V1_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "pipeline_a"
V2_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "pipeline_a_v2"

A0_COHORT_PATH = V2_OUTPUT_DIR / "cohort_master.csv"

A1_EVENT_OUTPUT = V2_OUTPUT_DIR / "a1_validated_events.parquet"
A1_FEATURE_QUALITY_OUTPUT = V2_OUTPUT_DIR / "feature_quality_summary.csv"
A1_INVALID_OUTPUT = V2_OUTPUT_DIR / "invalid_measurement_counts.csv"
A1_UNIT_AUDIT_OUTPUT = V2_OUTPUT_DIR / "a1_unit_audit.csv"
A1_LAB_REFERENCE_AUDIT_OUTPUT = (
    V2_OUTPUT_DIR / "labevents_reference_audit.csv"
)
A1_FIO2_AUDIT_OUTPUT = V2_OUTPUT_DIR / "fio2_unit_audit.csv"
A1_MAPPING_AUDIT_OUTPUT = (
    V2_OUTPUT_DIR / "a1_feature_mapping_audit.csv"
)
A1_GCS_AUDIT_OUTPUT = V2_OUTPUT_DIR / "a1_gcs_component_audit.csv"
A1_FEATURE_DEFINITION_OUTPUT = (
    V2_OUTPUT_DIR / "a1_feature_definitions.json"
)
A1_METADATA_OUTPUT = V2_OUTPUT_DIR / "a1_metadata.json"


if V1_OUTPUT_DIR.resolve() == V2_OUTPUT_DIR.resolve():
    raise RuntimeError(
        "SAFETY ERROR: V1 and V2 output directories are identical."
    )

V2_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. SOURCE FILES
# ============================================================

CHARTEVENTS_PATH = ICU_DIR / "chartevents.csv.gz"
LABEVENTS_PATH = HOSP_DIR / "labevents.csv.gz"
D_ITEMS_PATH = ICU_DIR / "d_items.csv.gz"
D_LABITEMS_PATH = HOSP_DIR / "d_labitems.csv.gz"

for path in [
    A0_COHORT_PATH,
    CHARTEVENTS_PATH,
    LABEVENTS_PATH,
    D_ITEMS_PATH,
    D_LABITEMS_PATH,
]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

print("=" * 80)
print("PIPELINE A V2 — A1 RAW EHR EXTRACTION + QC")
print("=" * 80)
print(f"V1 output directory : {V1_OUTPUT_DIR}")
print(f"V2 output directory : {V2_OUTPUT_DIR}")
print(f"Raw MIMIC-IV        : {MIMIC_IV_DIR}")


# ============================================================
# 2. FEATURE DEFINITIONS
# ============================================================

# Existing V1 chart features.
CHART_ITEMS = {
    "heart_rate": [220045],
    "sbp": [220179, 220050],
    "dbp": [220180, 220051],
    "mbp": [220052, 220181],
    "resp_rate": [220210, 224690],
    "spo2": [220277],
    "temperature": [223762, 226329],
}

# Existing V1 lab features.
LAB_ITEMS = {
    "wbc": [51300, 51301],
    "hemoglobin": [51222],
    "platelets": [51265],
    "creatinine": [50912],
    "bun": [51006],
    "lactate": [50813],
    "glucose": [50931, 52027],
    "sodium": [50824, 50983],
    "potassium": [50822, 50971],
    "chloride": [50806, 50902],
    "bicarbonate": [50882],
    "ph": [50820],
    "pao2": [50821],
    "paco2": [50818],
    "bilirubin": [50885],
}

# New V2 numeric respiratory features.
FIO2_ITEMS = [223835]
PEEP_ITEMS = [220339, 224700]

# New V2 GCS features.
GCS_ITEMS = {
    "gcs_verbal": [223900],
    "gcs_motor": [223901],
    "gcs_eyes": [220739],
}

# New V2 respiratory-support source items.
RESP_SUPPORT_ITEMS = {
    "o2_delivery_device": [226732],
    "ventilator_mode": [223849],
    "ventilator_mode_hamilton": [229314],
}


CHART_ITEM_TO_FEATURE = {
    itemid: feature
    for feature, itemids in CHART_ITEMS.items()
    for itemid in itemids
}

LAB_ITEM_TO_FEATURE = {
    itemid: feature
    for feature, itemids in LAB_ITEMS.items()
    for itemid in itemids
}

FIO2_ITEM_TO_FEATURE = {itemid: "fio2" for itemid in FIO2_ITEMS}
PEEP_ITEM_TO_FEATURE = {itemid: "peep" for itemid in PEEP_ITEMS}

CHART_NUMERIC_ITEM_TO_FEATURE = {
    **CHART_ITEM_TO_FEATURE,
    **FIO2_ITEM_TO_FEATURE,
    **PEEP_ITEM_TO_FEATURE,
}

GCS_ITEM_IDS = sorted({
    itemid
    for ids in GCS_ITEMS.values()
    for itemid in ids
})

RESP_SUPPORT_ITEM_IDS = sorted({
    itemid
    for ids in RESP_SUPPORT_ITEMS.values()
    for itemid in ids
})

CHART_NUMERIC_ITEM_IDS = sorted(CHART_NUMERIC_ITEM_TO_FEATURE)
LAB_ITEM_IDS = sorted(LAB_ITEM_TO_FEATURE)

ALL_CHART_ITEM_IDS = sorted(
    set(CHART_NUMERIC_ITEM_IDS)
    | set(GCS_ITEM_IDS)
    | set(RESP_SUPPORT_ITEM_IDS)
)


# ============================================================
# 3. V2 FEATURE ORDER
# ============================================================

BASE_FEATURE_ORDER = list(CHART_ITEMS.keys()) + list(LAB_ITEMS.keys())

FEATURE_ORDER = BASE_FEATURE_ORDER + [
    "gcs_total",
    "gcs_unable",
    "gcs_components_observed",
    "fio2",
    "peep",
    "resp_support_mode",
]

if len(BASE_FEATURE_ORDER) != 22:
    raise AssertionError(
        f"Expected 22 original clinical variables, found "
        f"{len(BASE_FEATURE_ORDER)}."
    )

if len(FEATURE_ORDER) != 28:
    raise AssertionError(
        f"Expected 28 total A1 features, found {len(FEATURE_ORDER)}."
    )


# ============================================================
# 4. FEATURE-SPECIFIC PLAUSIBILITY BOUNDS
# ============================================================

# These are deliberately generous outer bounds intended to reject
# obvious data-entry / archival artifacts, not to clip severe illness.
FEATURE_BOUNDS = {
    "heart_rate": (20.0, 300.0),
    "sbp": (20.0, 300.0),
    "dbp": (10.0, 200.0),
    "mbp": (20.0, 250.0),
    "resp_rate": (2.0, 70.0),
    "spo2": (50.0, 100.0),
    "temperature": (25.0, 45.0),
    "wbc": (0.1, 150.0),
    "hemoglobin": (2.0, 20.0),
    "platelets": (1.0, 2000.0),
    "creatinine": (0.1, 20.0),
    "bun": (1.0, 200.0),
    "lactate": (0.1, 30.0),
    "glucose": (10.0, 1000.0),
    "sodium": (100.0, 200.0),
    "potassium": (1.0, 10.0),
    "chloride": (60.0, 150.0),
    "bicarbonate": (2.0, 60.0),
    "ph": (6.5, 8.0),
    "pao2": (20.0, 600.0),
    "paco2": (10.0, 150.0),
    "bilirubin": (0.1, 50.0),
    # 20% is valid FiO2 in the MIMIC-style normalization rule below.
    "fio2": (20.0, 100.0),
    "peep": (0.0, 100.0),
}

MAX_RANGE_REJECTION_PERCENT = 30.0
MAX_PLAUSIBILITY_REJECTION_PERCENT = 30.0


# ============================================================
# 5. A2 AGGREGATION METADATA
# ============================================================

AGGREGATION_RULES = {
    feature: "median"
    for feature in FEATURE_ORDER
}

# Binary / ordinal features should not be median-aggregated.
AGGREGATION_RULES["gcs_unable"] = "max"
AGGREGATION_RULES["gcs_components_observed"] = "max"
AGGREGATION_RULES["resp_support_mode"] = "max"

FEATURE_TYPES = {
    feature: "numeric"
    for feature in FEATURE_ORDER
}
FEATURE_TYPES["gcs_unable"] = "binary"
FEATURE_TYPES["gcs_components_observed"] = "numeric_count_0_to_3"
FEATURE_TYPES["resp_support_mode"] = "ordinal_categorical"


# ============================================================
# 6. RESPIRATORY SUPPORT ORDINAL ENCODING
# ============================================================

# Ordered from no support to highest support.
RESP_SUPPORT_ENCODING = {
    "None": 0,
    "SupplementalOxygen": 1,
    "HFNC": 2,
    "NonInvasiveVent": 3,
    "InvasiveVent": 4,
    "Tracheostomy": 5,
}

RESP_SUPPORT_DECODING = {
    value: key
    for key, value in RESP_SUPPORT_ENCODING.items()
}

TRACHEOSTOMY_DEVICES = {
    "Tracheostomy tube",
    "Trach mask",
    "Trach mask ",
}

INVASIVE_DEVICES = {
    "Endotracheal tube",
}

NIV_DEVICES = {
    "Bipap mask",
    "Bipap mask ",
    "CPAP mask",
    "CPAP mask ",
}

HFNC_DEVICES = {
    "High flow nasal cannula",
}

SUPPLEMENTAL_OXYGEN_DEVICES = {
    "Non-rebreather",
    "Face tent",
    "Aerosol-cool",
    "Venti mask",
    "Venti mask ",
    "Medium conc mask",
    "Medium conc mask ",
    "Ultrasonic neb",
    "Vapomist",
    "Oxymizer",
    "High flow neb",
    "Nasal cannula",
}

NO_OXYGEN_DEVICES = {"None"}

INVASIVE_VENT_MODES = {
    "(S) CMV",
    "APRV",
    "APRV/Biphasic+ApnPress",
    "APRV/Biphasic+ApnVol",
    "APV (cmv)",
    "Ambient",
    "Apnea Ventilation",
    "CMV",
    "CMV/ASSIST",
    "CMV/ASSIST/AutoFlow",
    "CMV/AutoFlow",
    "CPAP/PPS",
    "CPAP/PSV",
    "CPAP/PSV+Apn TCPL",
    "CPAP/PSV+ApnPres",
    "CPAP/PSV+ApnVol",
    "MMV",
    "MMV/AutoFlow",
    "MMV/PSV",
    "MMV/PSV/AutoFlow",
    "P-CMV",
    "PCV+",
    "PCV+/PSV",
    "PCV+Assist",
    "PRES/AC",
    "PRVC/AC",
    "PRVC/SIMV",
    "PSV/SBT",
    "SIMV",
    "SIMV/AutoFlow",
    "SIMV/PRES",
    "SIMV/PSV",
    "SIMV/PSV/AutoFlow",
    "SIMV/VOL",
    "SYNCHRON MASTER",
    "SYNCHRON SLAVE",
    "VOL/AC",
}

HAMILTON_INVASIVE_MODES = {
    "APRV",
    "APV (cmv)",
    "Ambient",
    "(S) CMV",
    "P-CMV",
    "SIMV",
    "APV (simv)",
    "P-SIMV",
    "VS",
    "ASV",
}

HAMILTON_NIV_MODES = {
    "DuoPaP",
    "NIV",
    "NIV-ST",
}


# ============================================================
# 7. LOAD + VALIDATE A0 V2 COHORT
# ============================================================

print("\nLoading V2 A0 cohort...")

cohort = pd.read_csv(A0_COHORT_PATH)

required_cohort_columns = [
    "subject_id",
    "hadm_id",
    "stay_id",
    "intime",
    "outtime",
    "icu_los_hours",
]

missing_cohort_columns = [
    col for col in required_cohort_columns
    if col not in cohort.columns
]

if missing_cohort_columns:
    raise AssertionError(
        f"A0 cohort missing columns: {missing_cohort_columns}"
    )

cohort["intime"] = pd.to_datetime(
    cohort["intime"],
    errors="coerce",
)
cohort["outtime"] = pd.to_datetime(
    cohort["outtime"],
    errors="coerce",
)
cohort["icu_los_hours"] = pd.to_numeric(
    cohort["icu_los_hours"],
    errors="coerce",
)

if cohort["stay_id"].duplicated().any():
    raise AssertionError("A0 cohort contains duplicate stay_id.")

if cohort["intime"].isna().any():
    raise AssertionError("A0 cohort contains missing intime.")

if cohort["outtime"].isna().any():
    raise AssertionError("A0 cohort contains missing outtime.")

if (cohort["outtime"] <= cohort["intime"]).any():
    raise AssertionError("A0 cohort contains outtime <= intime.")

if (cohort["icu_los_hours"] < 6).any():
    raise AssertionError("A0 cohort contains ICU LOS < 6h.")

EXPECTED_A0_ROWS = 93224
if len(cohort) != EXPECTED_A0_ROWS:
    raise AssertionError(
        f"Unexpected A0 cohort size. Expected {EXPECTED_A0_ROWS:,}, "
        f"found {len(cohort):,}."
    )

print(f"A0 cohort rows : {len(cohort):,}")
print(f"Unique stays   : {cohort['stay_id'].nunique():,}")
print(f"Unique subjects: {cohort['subject_id'].nunique():,}")

# A1/A2/A3/A4 benchmark uses only the first 24 ICU hours.
stay_window = cohort[
    [
        "subject_id",
        "hadm_id",
        "stay_id",
        "intime",
        "outtime",
    ]
].copy()

stay_ids = set(cohort["stay_id"].astype("int64"))
hadm_ids = set(cohort["hadm_id"].astype("int64"))


# ============================================================
# 8. LOAD ITEM-DEFINITION TABLES + VALIDATE MAPPINGS
# ============================================================

print("\nLoading item-definition tables...")

d_items = pd.read_csv(
    D_ITEMS_PATH,
    compression="gzip",
    usecols=[
        "itemid",
        "label",
        "abbreviation",
        "linksto",
        "category",
        "unitname",
        "param_type",
        "lownormalvalue",
        "highnormalvalue",
    ],
    low_memory=False,
)

d_labitems = pd.read_csv(
    D_LABITEMS_PATH,
    compression="gzip",
    usecols=[
        "itemid",
        "label",
        "fluid",
        "category",
    ],
    low_memory=False,
)

chart_missing = sorted(
    set(ALL_CHART_ITEM_IDS)
    - set(d_items["itemid"].astype(int))
)
if chart_missing:
    raise AssertionError(
        f"Chart item IDs missing from d_items: {chart_missing}"
    )

chart_map = d_items[
    d_items["itemid"].isin(ALL_CHART_ITEM_IDS)
].copy()

if not (
    chart_map["linksto"]
    .fillna("")
    .str.lower()
    .eq("chartevents")
    .all()
):
    bad = chart_map[
        ~chart_map["linksto"]
        .fillna("")
        .str.lower()
        .eq("chartevents")
    ]
    raise AssertionError(
        "Some selected chart item IDs do not link to chartevents:\n"
        + bad.to_string(index=False)
    )

lab_missing = sorted(
    set(LAB_ITEM_IDS)
    - set(d_labitems["itemid"].astype(int))
)
if lab_missing:
    raise AssertionError(
        f"Lab item IDs missing from d_labitems: {lab_missing}"
    )

mapping_rows = []

for _, row in chart_map.iterrows():
    itemid = int(row["itemid"])

    if itemid in CHART_ITEM_TO_FEATURE:
        feature = CHART_ITEM_TO_FEATURE[itemid]
    elif itemid in FIO2_ITEM_TO_FEATURE:
        feature = "fio2"
    elif itemid in PEEP_ITEM_TO_FEATURE:
        feature = "peep"
    elif itemid in GCS_ITEM_IDS:
        feature = "gcs_component"
    elif itemid in RESP_SUPPORT_ITEM_IDS:
        feature = "resp_support_source"
    else:
        feature = "UNKNOWN"

    mapping_rows.append({
        "source": "chartevents",
        "itemid": itemid,
        "feature": feature,
        "label": row["label"],
        "abbreviation": row["abbreviation"],
        "linksto": row["linksto"],
        "category": row["category"],
        "unitname": row["unitname"],
        "param_type": row["param_type"],
        "lownormalvalue": row["lownormalvalue"],
        "highnormalvalue": row["highnormalvalue"],
    })

lab_map = d_labitems[
    d_labitems["itemid"].isin(LAB_ITEM_IDS)
].copy()

for _, row in lab_map.iterrows():
    itemid = int(row["itemid"])
    mapping_rows.append({
        "source": "labevents",
        "itemid": itemid,
        "feature": LAB_ITEM_TO_FEATURE[itemid],
        "label": row["label"],
        "abbreviation": None,
        "linksto": "labevents",
        "category": row["category"],
        "unitname": None,
        "param_type": "number",
        "lownormalvalue": None,
        "highnormalvalue": None,
    })

mapping_audit = pd.DataFrame(mapping_rows)


# ============================================================
# 9. UNIT + LAB-REFERENCE AUDIT STATE
# ============================================================

unit_values = defaultdict(set)
unit_counts = defaultdict(int)


def record_units(source, df):
    """Track distinct non-null valueuom representations by itemid."""
    if "valueuom" not in df.columns or df.empty:
        return

    for itemid, group in df.groupby("itemid", observed=True):
        itemid = int(itemid)
        values = (
            group["valueuom"]
            .dropna()
            .astype(str)
            .str.strip()
            .str.lower()
        )
        values = values[values != ""]

        for value in values.unique():
            unit_values[(source, itemid)].add(value)

        unit_counts[(source, itemid)] += len(values)


lab_reference_audit = {
    int(itemid): {
        "ref_lower_nonnull_count": 0,
        "ref_upper_nonnull_count": 0,
        "ref_lower_min": None,
        "ref_lower_max": None,
        "ref_upper_min": None,
        "ref_upper_max": None,
        "reference_range_pairs": set(),
        "flag_counts": defaultdict(int),
    }
    for itemid in LAB_ITEM_IDS
}


def record_lab_reference_audit(df):
    """
    Record labevents reference ranges and flags for supplementary QC.

    IMPORTANT: ref_range_lower/ref_range_upper/flag are NEVER used to
    exclude clinical observations from the primary event stream.
    """
    if df is None or df.empty:
        return

    for itemid, group in df.groupby("itemid", observed=True):
        itemid = int(itemid)
        state = lab_reference_audit[itemid]

        lower = pd.to_numeric(
            group["ref_range_lower"],
            errors="coerce",
        )
        upper = pd.to_numeric(
            group["ref_range_upper"],
            errors="coerce",
        )

        lower_finite = lower[np.isfinite(lower)]
        upper_finite = upper[np.isfinite(upper)]

        state["ref_lower_nonnull_count"] += int(lower_finite.size)
        state["ref_upper_nonnull_count"] += int(upper_finite.size)

        if lower_finite.size:
            lower_min = float(lower_finite.min())
            lower_max = float(lower_finite.max())
            state["ref_lower_min"] = (
                lower_min
                if state["ref_lower_min"] is None
                else min(state["ref_lower_min"], lower_min)
            )
            state["ref_lower_max"] = (
                lower_max
                if state["ref_lower_max"] is None
                else max(state["ref_lower_max"], lower_max)
            )

        if upper_finite.size:
            upper_min = float(upper_finite.min())
            upper_max = float(upper_finite.max())
            state["ref_upper_min"] = (
                upper_min
                if state["ref_upper_min"] is None
                else min(state["ref_upper_min"], upper_min)
            )
            state["ref_upper_max"] = (
                upper_max
                if state["ref_upper_max"] is None
                else max(state["ref_upper_max"], upper_max)
            )

        pair_mask = lower.notna() & upper.notna()
        if pair_mask.any():
            state["reference_range_pairs"].update(
                (
                    float(lower_value),
                    float(upper_value),
                )
                for lower_value, upper_value in zip(
                    lower.loc[pair_mask].tolist(),
                    upper.loc[pair_mask].tolist(),
                )
            )

        flags = (
            group["flag"]
            .dropna()
            .astype(str)
            .str.strip()
            .str.lower()
        )
        flags = flags[flags != ""]
        for flag in flags:
            state["flag_counts"][flag] += 1


# ============================================================
# 10. QUALITY COUNTERS
# ============================================================


def new_quality_record():
    return {
        "raw_count": 0,
        "valid_count": 0,
        "warning_count": 0,
        "parse_invalid_count": 0,
        "nonfinite_count": 0,
        "range_invalid_count": 0,
        "normalization_invalid_count": 0,
    }


quality = {
    feature: new_quality_record()
    for feature in (
        list(CHART_ITEMS.keys())
        + ["fio2", "peep"]
        + list(LAB_ITEMS.keys())
    )
}


# ============================================================
# 11. FiO2 REPRESENTATION AUDIT
# ============================================================

fio2_representation_counts = defaultdict(int)


def normalize_fio2(values):
    """
    Normalize FiO2 to percentage units.

    Supported MIMIC-style representations:
        0.20 <= x <= 1.00  -> x * 100
        1 < x < 20         -> invalid representation
        20 <= x <= 100     -> unchanged
        otherwise          -> invalid
    """
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    finite = np.isfinite(numeric)

    fraction = (
        finite
        & (numeric >= 0.20)
        & (numeric <= 1.00)
    )
    percentage = (
        finite
        & (numeric >= 20.0)
        & (numeric <= 100.0)
    )
    improper = (
        finite
        & (numeric > 1.0)
        & (numeric < 20.0)
    )
    outside = finite & ~(fraction | percentage | improper)

    fio2_representation_counts[
        "fraction_0.20_to_1.00"
    ] += int(fraction.sum())
    fio2_representation_counts[
        "percent_20_to_100"
    ] += int(percentage.sum())
    fio2_representation_counts[
        "improper_1_to_20"
    ] += int(improper.sum())
    fio2_representation_counts[
        "outside_supported_range"
    ] += int(outside.sum())

    out = numeric.copy()
    out.loc[fraction] = numeric.loc[fraction] * 100.0
    out.loc[improper | outside] = np.nan

    return out


# ============================================================
# 12. EVENT OUTPUT SCHEMA
# ============================================================

EVENT_SCHEMA = pa.schema([
    pa.field("subject_id", pa.int64()),
    pa.field("hadm_id", pa.int64()),
    pa.field("stay_id", pa.int64()),
    pa.field("charttime", pa.timestamp("ns")),
    pa.field("event_time_hours", pa.float32()),
    pa.field("icu_hour", pa.int32()),
    pa.field("feature", pa.string()),
    pa.field("value", pa.float32()),
    pa.field("value_text", pa.string()),
    pa.field("itemid", pa.int64()),
    pa.field("source", pa.string()),
    pa.field("valueuom", pa.string()),
    pa.field("warning_flag", pa.int8()),
])


def append_events(writer, df):
    """Append one validated event batch using a fixed Parquet schema."""
    if df is None or df.empty:
        return 0

    df = df.copy()

    df["subject_id"] = pd.to_numeric(
        df["subject_id"], errors="raise"
    ).astype("int64")
    df["hadm_id"] = pd.to_numeric(
        df["hadm_id"], errors="raise"
    ).astype("int64")
    df["stay_id"] = pd.to_numeric(
        df["stay_id"], errors="raise"
    ).astype("int64")
    df["charttime"] = pd.to_datetime(
        df["charttime"], errors="raise"
    )
    df["event_time_hours"] = pd.to_numeric(
        df["event_time_hours"], errors="raise"
    ).astype("float32")
    df["icu_hour"] = pd.to_numeric(
        df["icu_hour"], errors="raise"
    ).astype("int32")
    df["feature"] = df["feature"].astype(str)
    df["value"] = pd.to_numeric(
        df["value"], errors="raise"
    ).astype("float32")
    df["value_text"] = df["value_text"].where(
        df["value_text"].notna(),
        None,
    )
    df["itemid"] = pd.to_numeric(
        df["itemid"], errors="raise"
    ).astype("int64")
    df["source"] = df["source"].astype(str)
    df["valueuom"] = df["valueuom"].where(
        df["valueuom"].notna(),
        None,
    )
    df["warning_flag"] = pd.to_numeric(
        df["warning_flag"], errors="coerce"
    ).fillna(0).astype("int8")

    df = df[
        [
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
    ]

    if not np.isfinite(
        df["value"].to_numpy(dtype=np.float64)
    ).all():
        raise AssertionError(
            "Non-finite value entered validated event output."
        )

    if (
        df["feature"]
        .isin(FEATURE_ORDER)
        .eq(False)
        .any()
    ):
        bad_features = sorted(
            set(df.loc[
                ~df["feature"].isin(FEATURE_ORDER),
                "feature",
            ])
        )
        raise AssertionError(
            f"Unknown feature(s) entered validated event output: "
            f"{bad_features}"
        )

    table = pa.Table.from_pandas(
        df,
        schema=EVENT_SCHEMA,
        preserve_index=False,
    )
    writer.write_table(table)
    return len(df)


# ============================================================
# 13. TEMPORARY PARQUET OUTPUT
# ============================================================

TEMP_EVENT_OUTPUT = V2_OUTPUT_DIR / (
    "a1_validated_events.parquet.tmp"
)

if TEMP_EVENT_OUTPUT.exists():
    TEMP_EVENT_OUTPUT.unlink()

event_writer = pq.ParquetWriter(
    TEMP_EVENT_OUTPUT,
    EVENT_SCHEMA,
    compression="zstd",
)

total_written_events = 0
chart_raw_candidate_rows = 0
lab_raw_candidate_rows = 0

# Special chart features are retained only in memory because they are a
# much smaller subset than all of chartevents and need to be derived after
# numeric chart extraction has completed.
special_parts = []


# ============================================================
# 14. PROCESS CHARTEVENTS
# ============================================================

print("\n" + "-" * 80)
print("CHARTEVENTS — RAW EXTRACTION + QC")
print("-" * 80)

chart_usecols = [
    "stay_id",
    "charttime",
    "storetime",
    "itemid",
    "value",
    "valuenum",
    "valueuom",
    "warning",
]

chart_reader = pd.read_csv(
    CHARTEVENTS_PATH,
    compression="gzip",
    usecols=chart_usecols,
    chunksize=2_000_000,
    low_memory=False,
)

for chunk in tqdm(chart_reader, desc="Chartevents"):

    chunk = chunk[
        chunk["stay_id"].isin(stay_ids)
    ]
    if chunk.empty:
        continue

    chunk = chunk[
        chunk["itemid"].isin(ALL_CHART_ITEM_IDS)
    ]
    if chunk.empty:
        continue

    # Audit source units before any validity filtering.
    record_units("chartevents", chunk)

    chunk["charttime"] = pd.to_datetime(
        chunk["charttime"], errors="coerce"
    )
    chunk = chunk.dropna(subset=["charttime"])
    if chunk.empty:
        continue

    chunk = chunk.merge(
        stay_window,
        on="stay_id",
        how="inner",
        validate="many_to_one",
    )

    chunk = chunk[
        (chunk["charttime"] >= chunk["intime"])
        & (chunk["charttime"] < chunk["outtime"])
    ]
    if chunk.empty:
        continue

    delta_hours = (
        chunk["charttime"] - chunk["intime"]
    ).dt.total_seconds() / 3600.0

    chunk["event_time_hours"] = delta_hours.astype("float32")

    chunk = chunk[
        (chunk["event_time_hours"] >= 0.0)
        & (chunk["event_time_hours"] < 24.0)
    ].copy()
    if chunk.empty:
        continue

    chunk["icu_hour"] = np.floor(
        chunk["event_time_hours"]
    ).astype("int32")

    # --------------------------------------------------------
    # Numeric chart features: 22 original + FiO2 + PEEP
    # --------------------------------------------------------

    numeric_chunk = chunk[
        chunk["itemid"].isin(CHART_NUMERIC_ITEM_IDS)
    ].copy()

    if not numeric_chunk.empty:

        numeric_chunk["feature"] = (
            numeric_chunk["itemid"]
            .map(CHART_NUMERIC_ITEM_TO_FEATURE)
        )

        chart_raw_candidate_rows += len(numeric_chunk)
        validated_parts = []

        for feature, group in numeric_chunk.groupby(
            "feature",
            observed=True,
        ):
            feature = str(feature)
            group = group.copy()
            rec = quality[feature]

            rec["raw_count"] += len(group)

            warning_flag = (
                pd.to_numeric(
                    group["warning"],
                    errors="coerce",
                )
                .fillna(0)
                .astype(int)
                .gt(0)
            )
            rec["warning_count"] += int(
                warning_flag.sum()
            )

            raw_values = pd.to_numeric(
                group["valuenum"],
                errors="coerce",
            ).astype(float)

            parse_invalid = raw_values.isna()
            rec["parse_invalid_count"] += int(
                parse_invalid.sum()
            )

            finite = (
                raw_values.notna()
                & np.isfinite(raw_values)
            )
            nonfinite = (
                raw_values.notna()
                & ~np.isfinite(raw_values)
            )
            rec["nonfinite_count"] += int(
                nonfinite.sum()
            )

            if feature == "fio2":
                normalized_values = normalize_fio2(raw_values)
                normalization_invalid = (
                    raw_values.notna()
                    & np.isfinite(raw_values)
                    & normalized_values.isna()
                )
                rec["normalization_invalid_count"] += int(
                    normalization_invalid.sum()
                )
            else:
                normalized_values = raw_values.copy()

            lower, upper = FEATURE_BOUNDS[feature]

            in_range = (
                normalized_values.notna()
                & np.isfinite(normalized_values)
                & (normalized_values >= lower)
                & (normalized_values <= upper)
            )

            range_invalid = (
                normalized_values.notna()
                & ~in_range
            )
            rec["range_invalid_count"] += int(
                range_invalid.sum()
            )

            valid = (
                ~warning_flag
                & normalized_values.notna()
                & np.isfinite(normalized_values)
                & in_range
            )

            rec["valid_count"] += int(valid.sum())
            if not valid.any():
                continue

            validated_parts.append(pd.DataFrame({
                "subject_id": group.loc[
                    valid, "subject_id"
                ].astype("int64"),
                "hadm_id": group.loc[
                    valid, "hadm_id"
                ].astype("int64"),
                "stay_id": group.loc[
                    valid, "stay_id"
                ].astype("int64"),
                "charttime": group.loc[
                    valid, "charttime"
                ],
                "event_time_hours": group.loc[
                    valid, "event_time_hours"
                ].astype("float32"),
                "icu_hour": group.loc[
                    valid, "icu_hour"
                ].astype("int32"),
                "feature": feature,
                "value": normalized_values.loc[
                    valid
                ].astype("float32"),
                "value_text": None,
                "itemid": group.loc[
                    valid, "itemid"
                ].astype("int64"),
                "source": "chartevents",
                "valueuom": group.loc[
                    valid, "valueuom"
                ],
                "warning_flag": 0,
            }))

        if validated_parts:
            chart_validated = pd.concat(
                validated_parts,
                ignore_index=True,
            )
            total_written_events += append_events(
                event_writer,
                chart_validated,
            )

    # --------------------------------------------------------
    # Special chart features used to derive GCS/support mode.
    # --------------------------------------------------------

    special_chunk = chunk[
        chunk["itemid"].isin(
            GCS_ITEM_IDS + RESP_SUPPORT_ITEM_IDS
        )
    ].copy()

    if not special_chunk.empty:
        special_parts.append(
            special_chunk[
                [
                    "subject_id",
                    "hadm_id",
                    "stay_id",
                    "charttime",
                    "storetime",
                    "itemid",
                    "value",
                    "valuenum",
                    "valueuom",
                    "warning",
                    "event_time_hours",
                    "icu_hour",
                ]
            ]
        )

    del chunk
    del numeric_chunk
    del special_chunk


# ============================================================
# 15. PROCESS LABEVENTS
# ============================================================

print("\n" + "-" * 80)
print("LABEVENTS — RAW EXTRACTION + QC")
print("-" * 80)

lab_usecols = [
    "labevent_id",
    "subject_id",
    "hadm_id",
    "charttime",
    "itemid",
    "valuenum",
    "valueuom",
    "ref_range_lower",
    "ref_range_upper",
    "flag",
]

lab_reader = pd.read_csv(
    LABEVENTS_PATH,
    compression="gzip",
    usecols=lab_usecols,
    chunksize=2_000_000,
    low_memory=False,
)

for chunk in tqdm(lab_reader, desc="Labevents"):

    chunk = chunk[
        chunk["hadm_id"].isin(hadm_ids)
    ]
    if chunk.empty:
        continue

    chunk = chunk[
        chunk["itemid"].isin(LAB_ITEM_IDS)
    ]
    if chunk.empty:
        continue

    # Supplementary reference-range / flag audit only.
    # These fields do NOT participate in validity filtering.
    record_lab_reference_audit(chunk)

    # Audit units before validity filtering.
    record_units("labevents", chunk)

    chunk["charttime"] = pd.to_datetime(
        chunk["charttime"], errors="coerce"
    )
    chunk = chunk.dropna(subset=["charttime"])
    if chunk.empty:
        continue

    chunk = chunk.merge(
        stay_window,
        on=["subject_id", "hadm_id"],
        how="inner",
    )
    if chunk.empty:
        continue

    # A single hospital laboratory row must map to at most one ICU stay
    # inside the actual stay windows. This prevents duplicate lab events.
    chunk = chunk[
        (chunk["charttime"] >= chunk["intime"])
        & (chunk["charttime"] < chunk["outtime"])
    ].copy()
    if chunk.empty:
        continue

    duplicated_lab_ids = chunk.loc[
        chunk["labevent_id"].duplicated(keep=False),
        "labevent_id",
    ].unique()

    if len(duplicated_lab_ids) > 0:
        raise AssertionError(
            "A labevent_id matched multiple ICU stays inside the actual "
            "ICU windows. Refusing to duplicate laboratory observations. "
            f"Example IDs: {duplicated_lab_ids[:10].tolist()}"
        )

    delta_hours = (
        chunk["charttime"] - chunk["intime"]
    ).dt.total_seconds() / 3600.0

    chunk["event_time_hours"] = delta_hours.astype("float32")

    chunk = chunk[
        (chunk["event_time_hours"] >= 0.0)
        & (chunk["event_time_hours"] < 24.0)
    ].copy()
    if chunk.empty:
        continue

    chunk["icu_hour"] = np.floor(
        chunk["event_time_hours"]
    ).astype("int32")

    chunk["feature"] = chunk["itemid"].map(
        LAB_ITEM_TO_FEATURE
    )

    lab_raw_candidate_rows += len(chunk)
    validated_parts = []

    for feature, group in chunk.groupby(
        "feature",
        observed=True,
    ):
        feature = str(feature)
        group = group.copy()
        rec = quality[feature]

        rec["raw_count"] += len(group)

        # labevents has no MIMIC-IV warning boolean.
        warning_flag = pd.Series(
            False,
            index=group.index,
        )

        raw_values = pd.to_numeric(
            group["valuenum"],
            errors="coerce",
        ).astype(float)

        parse_invalid = raw_values.isna()
        rec["parse_invalid_count"] += int(
            parse_invalid.sum()
        )

        finite = (
            raw_values.notna()
            & np.isfinite(raw_values)
        )
        nonfinite = (
            raw_values.notna()
            & ~np.isfinite(raw_values)
        )
        rec["nonfinite_count"] += int(
            nonfinite.sum()
        )

        lower, upper = FEATURE_BOUNDS[feature]

        in_range = (
            raw_values.notna()
            & np.isfinite(raw_values)
            & (raw_values >= lower)
            & (raw_values <= upper)
        )

        range_invalid = (
            raw_values.notna()
            & ~in_range
        )
        rec["range_invalid_count"] += int(
            range_invalid.sum()
        )

        valid = (
            ~warning_flag
            & raw_values.notna()
            & np.isfinite(raw_values)
            & in_range
        )

        rec["valid_count"] += int(valid.sum())
        if not valid.any():
            continue

        validated_parts.append(pd.DataFrame({
            "subject_id": group.loc[
                valid, "subject_id"
            ].astype("int64"),
            "hadm_id": group.loc[
                valid, "hadm_id"
            ].astype("int64"),
            "stay_id": group.loc[
                valid, "stay_id"
            ].astype("int64"),
            "charttime": group.loc[
                valid, "charttime"
            ],
            "event_time_hours": group.loc[
                valid, "event_time_hours"
            ].astype("float32"),
            "icu_hour": group.loc[
                valid, "icu_hour"
            ].astype("int32"),
            "feature": feature,
            "value": raw_values.loc[
                valid
            ].astype("float32"),
            "value_text": None,
            "itemid": group.loc[
                valid, "itemid"
            ].astype("int64"),
            "source": "labevents",
            "valueuom": group.loc[
                valid, "valueuom"
            ],
            "warning_flag": 0,
        }))

    if validated_parts:
        lab_validated = pd.concat(
            validated_parts,
            ignore_index=True,
        )
        total_written_events += append_events(
            event_writer,
            lab_validated,
        )

    del chunk


# ============================================================
# 16. UNIT CONSISTENCY AUDIT + HARD GATE
# ============================================================

print("\n" + "-" * 80)
print("UNIT CONSISTENCY AUDIT")
print("-" * 80)

unit_audit_rows = []
unit_failures = []

for _, row in mapping_audit.iterrows():
    source = str(row["source"])
    itemid = int(row["itemid"])
    units_sorted = sorted(
        unit_values.get((source, itemid), set())
    )

    unit_audit_rows.append({
        "source": source,
        "itemid": itemid,
        "feature": str(row["feature"]),
        "observed_unit_count": len(units_sorted),
        "observed_units": ";".join(units_sorted),
        "d_items_unitname": (
            None
            if pd.isna(row["unitname"])
            else str(row["unitname"])
        ),
        "raw_nonnull_unit_count": unit_counts.get(
            (source, itemid),
            0,
        ),
    })

    if len(units_sorted) > 1:
        unit_failures.append({
            "source": source,
            "itemid": itemid,
            "feature": str(row["feature"]),
            "observed_units": units_sorted,
        })

unit_audit_df = pd.DataFrame(unit_audit_rows)

if unit_failures:
    event_writer.close()
    if TEMP_EVENT_OUTPUT.exists():
        TEMP_EVENT_OUTPUT.unlink()
    raise AssertionError(
        "UNIT DRIFT DETECTED. One or more item IDs have multiple "
        "distinct non-null valueuom representations. No A1 final "
        "output was committed.\n"
        + pd.DataFrame(unit_failures).to_string(index=False)
    )

print(
    "✅ No multi-unit itemid groups detected among observed "
    "non-null valueuom values."
)


# ============================================================
# 17. DERIVE GCS FEATURES
# ============================================================

print("\n" + "-" * 80)
print("GCS — DERIVED FEATURE CONSTRUCTION")
print("-" * 80)

gcs_warning_count = 0
gcs_raw_count_after_warning = 0

gcs_component_audit_rows = []
gcs_output = pd.DataFrame()

if special_parts:
    special_events = pd.concat(
        special_parts,
        ignore_index=True,
    )
else:
    special_events = pd.DataFrame(
        columns=[
            "subject_id",
            "hadm_id",
            "stay_id",
            "charttime",
            "storetime",
            "itemid",
            "value",
            "valuenum",
            "valueuom",
            "warning",
            "event_time_hours",
            "icu_hour",
        ]
    )

gcs_raw = special_events[
    special_events["itemid"].isin(GCS_ITEM_IDS)
].copy()

if not gcs_raw.empty:

    warning_mask = (
        pd.to_numeric(
            gcs_raw["warning"],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
        .gt(0)
    )
    gcs_warning_count = int(warning_mask.sum())
    gcs_raw = gcs_raw[~warning_mask].copy()
    gcs_raw_count_after_warning = len(gcs_raw)

    gcs_raw["charttime"] = pd.to_datetime(
        gcs_raw["charttime"], errors="coerce"
    )
    gcs_raw["storetime"] = pd.to_datetime(
        gcs_raw["storetime"], errors="coerce"
    )
    gcs_raw = gcs_raw.dropna(subset=["charttime"])

    # Keep the latest stored value for each stay/charttime/item.
    gcs_raw = gcs_raw.sort_values(
        [
            "stay_id",
            "charttime",
            "itemid",
            "storetime",
        ],
        na_position="first",
    )
    gcs_raw = gcs_raw.drop_duplicates(
        subset=[
            "stay_id",
            "charttime",
            "itemid",
        ],
        keep="last",
    )

    gcs_raw["component_value"] = pd.to_numeric(
        gcs_raw["valuenum"],
        errors="coerce",
    ).astype(float)

    # Official MIMIC-style special verbal convention.
    gcs_raw["verbal_ett_flag"] = 0
    verbal_mask = gcs_raw["itemid"].eq(223900)
    ett_mask = (
        verbal_mask
        & gcs_raw["value"].astype(str).str.strip().eq(
            "No Response-ETT"
        )
    )
    gcs_raw.loc[ett_mask, "component_value"] = 0.0
    gcs_raw.loc[ett_mask, "verbal_ett_flag"] = 1

    # Validate and then REMOVE invalid individual components.
    component_specs = [
        (223900, "gcs_verbal", 1.0, 5.0),
        (223901, "gcs_motor", 1.0, 6.0),
        (220739, "gcs_eyes", 1.0, 4.0),
    ]

    for itemid, component_name, lower, upper in component_specs:
        sub = gcs_raw[
            gcs_raw["itemid"].eq(itemid)
        ]

        ordinary_valid = (
            sub["component_value"].notna()
            & np.isfinite(sub["component_value"])
            & sub["component_value"].between(
                lower,
                upper,
            )
        )

        if itemid == 223900:
            valid_component = (
                ordinary_valid
                | (
                    sub["verbal_ett_flag"].eq(1)
                    & sub["component_value"].eq(0.0)
                )
            )
        else:
            valid_component = ordinary_valid

        gcs_component_audit_rows.append({
            "itemid": itemid,
            "component": component_name,
            "raw_rows_after_warning_filter": len(sub),
            "valid_component_rows": int(
                valid_component.sum()
            ),
            "invalid_component_rows": int(
                (~valid_component).sum()
            ),
            "no_response_ett_rows": (
                int(
                    sub["verbal_ett_flag"].eq(1).sum()
                )
                if itemid == 223900
                else 0
            ),
        })

    component_valid_mask = pd.Series(
        False,
        index=gcs_raw.index,
    )

    component_valid_mask |= (
        gcs_raw["itemid"].eq(223901)
        & gcs_raw["component_value"].between(1.0, 6.0)
    )
    component_valid_mask |= (
        gcs_raw["itemid"].eq(220739)
        & gcs_raw["component_value"].between(1.0, 4.0)
    )
    component_valid_mask |= (
        gcs_raw["itemid"].eq(223900)
        & (
            gcs_raw["component_value"].between(1.0, 5.0)
            | (
                gcs_raw["verbal_ett_flag"].eq(1)
                & gcs_raw["component_value"].eq(0.0)
            )
        )
    )
    component_valid_mask &= np.isfinite(
        gcs_raw["component_value"]
    )

    gcs_raw = gcs_raw[
        component_valid_mask
    ].copy()

    if not gcs_raw.empty:

        gcs_raw["component"] = gcs_raw[
            "itemid"
        ].map({
            223901: "motor",
            220739: "eyes",
            223900: "verbal",
        })

        # Every distinct GCS documentation time becomes an emitted GCS
        # event. Three target rows per timestamp let merge_asof carry each
        # component independently within the same 6-hour lookback.
        base = (
            gcs_raw[
                ["stay_id", "charttime"]
            ]
            .drop_duplicates()
            .sort_values(
                ["charttime", "stay_id"]
            )
        )

        component_names = pd.DataFrame({
            "component": [
                "motor",
                "eyes",
                "verbal",
            ],
            "_key": 1,
        })

        targets = (
            base.assign(_key=1)
            .merge(
                component_names,
                on="_key",
                how="inner",
                validate="many_to_many",
            )
            .drop(columns="_key")
            .sort_values(
                [
                    "charttime",
                    "stay_id",
                    "component",
                ]
            )
            .reset_index(drop=True)
        )

        component_long = gcs_raw[
            [
                "stay_id",
                "charttime",
                "component",
                "component_value",
                "verbal_ett_flag",
            ]
        ].copy()

        component_long["ett_flag"] = np.where(
            component_long["component"].eq("verbal"),
            component_long["verbal_ett_flag"].astype(float),
            np.nan,
        )

        component_long = component_long.sort_values(
            [
                "charttime",
                "stay_id",
                "component",
            ]
        ).reset_index(drop=True)

        # Vectorized replacement for the previous per-stay/per-component
        # merge_asof loops.
        merged_long = pd.merge_asof(
            targets,
            component_long[
                [
                    "stay_id",
                    "charttime",
                    "component",
                    "component_value",
                    "ett_flag",
                ]
            ],
            on="charttime",
            by=["stay_id", "component"],
            direction="backward",
            tolerance=pd.Timedelta(hours=6),
            allow_exact_matches=True,
        )

        gcs_wide = (
            merged_long
            .pivot_table(
                index=["stay_id", "charttime"],
                columns="component",
                values="component_value",
                aggfunc="last",
            )
            .reset_index()
        )

        gcs_ett = merged_long[
            merged_long["component"].eq("verbal")
        ][
            [
                "stay_id",
                "charttime",
                "ett_flag",
            ]
        ].copy()

        gcs_wide = gcs_wide.merge(
            gcs_ett,
            on=["stay_id", "charttime"],
            how="left",
            validate="one_to_one",
        )

        # This is the new transparency feature requested for V2.
        # It counts components actually available from current or <=6h
        # prior valid observations. It does NOT count defaulted normal
        # components as observed.
        gcs_wide["gcs_components_observed"] = (
            gcs_wide[
                [
                    "motor",
                    "eyes",
                    "verbal",
                ]
            ]
            .notna()
            .sum(axis=1)
            .astype("int8")
        )

        gcs_wide["gcs_unable"] = (
            gcs_wide["ett_flag"]
            .fillna(0)
            .astype("int8")
        )

        # At least one component must actually be available.
        gcs_wide = gcs_wide[
            gcs_wide["gcs_components_observed"] > 0
        ].copy()

        if not gcs_wide.empty:

            # Preserve the official-style MIMIC derived convention:
            # missing components may be defaulted to normal. The new
            # gcs_components_observed feature tells the model whether that
            # happened.
            normal_motor = gcs_wide["motor"].fillna(6.0)
            normal_verbal = gcs_wide["verbal"].fillna(5.0)
            normal_eyes = gcs_wide["eyes"].fillna(4.0)

            gcs_wide["gcs_total"] = np.where(
                gcs_wide["gcs_unable"].eq(1),
                15.0,
                normal_motor + normal_verbal + normal_eyes,
            ).astype("float32")

            # Hard semantic checks before committing derived events.
            if not np.isfinite(
                gcs_wide["gcs_total"].to_numpy(dtype=float)
            ).all():
                raise AssertionError(
                    "Non-finite gcs_total generated."
                )

            if (
                (gcs_wide["gcs_total"] < 3.0)
                | (gcs_wide["gcs_total"] > 15.0)
            ).any():
                raise AssertionError(
                    "gcs_total outside expected 3-15 range."
                )

            gcs_output = gcs_wide[
                [
                    "stay_id",
                    "charttime",
                    "gcs_total",
                    "gcs_unable",
                    "gcs_components_observed",
                ]
            ].copy()

if gcs_output.empty:
    print("No valid GCS-derived events found.")
else:

    gcs_output = gcs_output.merge(
        stay_window[
            [
                "subject_id",
                "hadm_id",
                "stay_id",
                "intime",
            ]
        ],
        on="stay_id",
        how="inner",
        validate="many_to_one",
    )

    gcs_output["event_time_hours"] = (
        (
            gcs_output["charttime"]
            - gcs_output["intime"]
        ).dt.total_seconds()
        / 3600.0
    ).astype("float32")

    gcs_output = gcs_output[
        (gcs_output["event_time_hours"] >= 0.0)
        & (gcs_output["event_time_hours"] < 24.0)
    ].copy()

    gcs_output["icu_hour"] = np.floor(
        gcs_output["event_time_hours"]
    ).astype("int32")

    for feature in [
        "gcs_total",
        "gcs_unable",
        "gcs_components_observed",
    ]:
        sub = gcs_output[
            [
                "subject_id",
                "hadm_id",
                "stay_id",
                "charttime",
                "event_time_hours",
                "icu_hour",
                feature,
            ]
        ].copy()

        sub["feature"] = feature
        sub["value"] = pd.to_numeric(
            sub[feature], errors="coerce"
        ).astype("float32")
        sub["value_text"] = None
        sub["itemid"] = -1
        sub["source"] = "chartevents_derived"
        sub["valueuom"] = None
        sub["warning_flag"] = 0

        sub = sub[
            [
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
        ]

        total_written_events += append_events(
            event_writer,
            sub,
        )


# ============================================================
# 18. DERIVE RESPIRATORY SUPPORT MODE
# ============================================================

print("\n" + "-" * 80)
print("RESPIRATORY SUPPORT — ORDINAL STATE")
print("-" * 80)

support_warning_count = 0
support_raw_count_after_warning = 0
support_output = pd.DataFrame()

support_raw = special_events[
    special_events["itemid"].isin(RESP_SUPPORT_ITEM_IDS)
].copy()

if not support_raw.empty:

    warning_mask = (
        pd.to_numeric(
            support_raw["warning"],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
        .gt(0)
    )
    support_warning_count = int(warning_mask.sum())
    support_raw = support_raw[~warning_mask].copy()
    support_raw_count_after_warning = len(support_raw)

    if not support_raw.empty:

        support_raw["support_value_norm"] = (
            support_raw["value"]
            .astype("string")
            .str.strip()
        )

        support_raw["support_state"] = pd.Series(
            pd.NA,
            index=support_raw.index,
            dtype="object",
        )

        # Device classification map; stripping whitespace makes the
        # classification robust to the observed trailing-space variants.
        device_map = {}
        for name, devices in [
            ("Tracheostomy", TRACHEOSTOMY_DEVICES),
            ("InvasiveVent", INVASIVE_DEVICES),
            ("NonInvasiveVent", NIV_DEVICES),
            ("HFNC", HFNC_DEVICES),
            ("SupplementalOxygen", SUPPLEMENTAL_OXYGEN_DEVICES),
            ("None", NO_OXYGEN_DEVICES),
        ]:
            for device in devices:
                device_map[device.strip()] = name

        device_mask = support_raw["itemid"].eq(226732)
        vent_mask = support_raw["itemid"].eq(223849)
        hamilton_mask = support_raw["itemid"].eq(229314)

        support_raw.loc[device_mask, "support_state"] = (
            support_raw.loc[
                device_mask,
                "support_value_norm",
            ].map(device_map)
        )

        standard_mode_map = {
            mode.strip(): "InvasiveVent"
            for mode in INVASIVE_VENT_MODES
        }
        hamilton_mode_map = {
            **{
                mode.strip(): "InvasiveVent"
                for mode in HAMILTON_INVASIVE_MODES
            },
            **{
                mode.strip(): "NonInvasiveVent"
                for mode in HAMILTON_NIV_MODES
            },
        }

        support_raw.loc[vent_mask, "support_state"] = (
            support_raw.loc[
                vent_mask,
                "support_value_norm",
            ].map(standard_mode_map)
        )
        support_raw.loc[hamilton_mask, "support_state"] = (
            support_raw.loc[
                hamilton_mask,
                "support_value_norm",
            ].map(hamilton_mode_map)
        )

        support_raw = support_raw[
            support_raw["support_state"].notna()
        ].copy()

        if not support_raw.empty:
            support_raw["support_code"] = (
                support_raw["support_state"]
                .map(RESP_SUPPORT_ENCODING)
                .astype("int8")
            )

            # Clinical hierarchy: trach > invasive > NIV > HFNC > O2 > none.
            support_output = (
                support_raw
                .groupby(
                    [
                        "subject_id",
                        "hadm_id",
                        "stay_id",
                        "charttime",
                    ],
                    as_index=False,
                    observed=True,
                )
                .agg(
                    support_code=(
                        "support_code",
                        "max",
                    )
                )
            )

            support_output["support_state"] = (
                support_output["support_code"]
                .map(RESP_SUPPORT_DECODING)
            )

if support_output.empty:
    print("No recognized respiratory-support states found.")
else:

    support_output = support_output.merge(
        stay_window[
            ["stay_id", "intime"]
        ],
        on="stay_id",
        how="inner",
        validate="many_to_one",
    )

    support_output["event_time_hours"] = (
        (
            support_output["charttime"]
            - support_output["intime"]
        ).dt.total_seconds()
        / 3600.0
    ).astype("float32")

    support_output = support_output[
        (support_output["event_time_hours"] >= 0.0)
        & (support_output["event_time_hours"] < 24.0)
    ].copy()

    support_output["icu_hour"] = np.floor(
        support_output["event_time_hours"]
    ).astype("int32")

    support_code_values = (
        support_output["support_code"]
        .to_numpy(dtype=np.int64)
    )
    if not ((support_code_values >= 0) & (support_code_values <= 5)).all():
        raise AssertionError(
            "resp_support_mode contains a value outside 0-5."
        )

    sub = pd.DataFrame({
        "subject_id": support_output["subject_id"],
        "hadm_id": support_output["hadm_id"],
        "stay_id": support_output["stay_id"],
        "charttime": support_output["charttime"],
        "event_time_hours": support_output["event_time_hours"],
        "icu_hour": support_output["icu_hour"],
        "feature": "resp_support_mode",
        "value": support_output["support_code"].astype("float32"),
        "value_text": support_output["support_state"],
        "itemid": -1,
        "source": "chartevents_derived",
        "valueuom": None,
        "warning_flag": 0,
    })

    total_written_events += append_events(
        event_writer,
        sub,
    )


# ============================================================
# 19. CLOSE TEMP EVENT WRITER
# ============================================================

event_writer.close()

if total_written_events <= 0:
    if TEMP_EVENT_OUTPUT.exists():
        TEMP_EVENT_OUTPUT.unlink()
    raise RuntimeError("A1 produced zero validated events.")


# ============================================================
# 20. QUALITY / RANGE / PLAUSIBILITY AUDIT
# ============================================================

print("\n" + "-" * 80)
print("TIER-1 RANGE / PLAUSIBILITY REJECTION AUDIT")
print("-" * 80)

quality_rows = []

for feature, rec in quality.items():

    raw_count = int(rec["raw_count"])
    valid_count = int(rec["valid_count"])
    invalid_count = raw_count - valid_count

    invalid_pct = (
        100.0 * invalid_count / raw_count
        if raw_count > 0
        else 0.0
    )
    range_invalid_pct = (
        100.0 * rec["range_invalid_count"] / raw_count
        if raw_count > 0
        else 0.0
    )
    normalization_invalid_pct = (
        100.0 * rec["normalization_invalid_count"] / raw_count
        if raw_count > 0
        else 0.0
    )
    plausibility_rejection_count = (
        int(rec["range_invalid_count"])
        + int(rec["normalization_invalid_count"])
    )
    plausibility_rejection_pct = (
        100.0 * plausibility_rejection_count / raw_count
        if raw_count > 0
        else 0.0
    )

    quality_rows.append({
        "feature": feature,
        "feature_type": FEATURE_TYPES[feature],
        "raw_count": raw_count,
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "invalid_percentage": invalid_pct,
        "warning_count": int(rec["warning_count"]),
        "parse_invalid_count": int(rec["parse_invalid_count"]),
        "nonfinite_count": int(rec["nonfinite_count"]),
        "range_invalid_count": int(rec["range_invalid_count"]),
        "range_invalid_percentage": range_invalid_pct,
        "normalization_invalid_count": int(
            rec["normalization_invalid_count"]
        ),
        "normalization_invalid_percentage": normalization_invalid_pct,
        "plausibility_rejection_count": plausibility_rejection_count,
        "plausibility_rejection_percentage": plausibility_rejection_pct,
    })

quality_df_numeric = pd.DataFrame(quality_rows)

range_gate_failures = quality_df_numeric[
    quality_df_numeric["range_invalid_percentage"]
    > MAX_RANGE_REJECTION_PERCENT
]

plausibility_gate_failures = quality_df_numeric[
    quality_df_numeric["plausibility_rejection_percentage"]
    > MAX_PLAUSIBILITY_REJECTION_PERCENT
]

if not range_gate_failures.empty:
    if TEMP_EVENT_OUTPUT.exists():
        TEMP_EVENT_OUTPUT.unlink()
    raise AssertionError(
        "TIER-1 RANGE REVIEW REQUIRED. At least one raw feature has "
        f">{MAX_RANGE_REJECTION_PERCENT:.0f}% rejected by its numeric "
        "range bound. Review the bound before continuing.\n\n"
        + range_gate_failures.to_string(index=False)
    )

if not plausibility_gate_failures.empty:
    if TEMP_EVENT_OUTPUT.exists():
        TEMP_EVENT_OUTPUT.unlink()
    raise AssertionError(
        "TIER-1 PLAUSIBILITY REVIEW REQUIRED. At least one raw feature "
        f"has >{MAX_PLAUSIBILITY_REJECTION_PERCENT:.0f}% of its raw finite "
        "measurements rejected by a plausibility-normalization rule.\n\n"
        + plausibility_gate_failures.to_string(index=False)
    )


# ============================================================
# 21. APPEND DERIVED FEATURE QUALITY
# ============================================================

derived_rows = []

gcs_count = 0 if gcs_output.empty else len(gcs_output)
support_count = 0 if support_output.empty else len(support_output)

for feature, count, feature_type in [
    ("gcs_total", gcs_count, "numeric"),
    ("gcs_unable", gcs_count, "binary"),
    (
        "gcs_components_observed",
        gcs_count,
        "numeric_count_0_to_3",
    ),
    (
        "resp_support_mode",
        support_count,
        "ordinal_categorical",
    ),
]:
    derived_rows.append({
        "feature": feature,
        "feature_type": feature_type,
        "raw_count": count,
        "valid_count": count,
        "invalid_count": 0,
        "invalid_percentage": 0.0,
        "warning_count": 0,
        "parse_invalid_count": 0,
        "nonfinite_count": 0,
        "range_invalid_count": 0,
        "range_invalid_percentage": 0.0,
        "normalization_invalid_count": 0,
        "normalization_invalid_percentage": 0.0,
        "plausibility_rejection_count": 0,
        "plausibility_rejection_percentage": 0.0,
    })

quality_df = pd.concat(
    [
        quality_df_numeric,
        pd.DataFrame(derived_rows),
    ],
    ignore_index=True,
)

quality_df = quality_df.drop_duplicates(
    subset=["feature"],
    keep="last",
)

quality_df["_feature_order"] = quality_df[
    "feature"
].map({feature: i for i, feature in enumerate(FEATURE_ORDER)})
quality_df = quality_df.sort_values(
    "_feature_order"
).drop(columns="_feature_order").reset_index(drop=True)

if set(quality_df["feature"]) != set(FEATURE_ORDER):
    raise AssertionError(
        "Feature QC summary does not contain exactly the canonical "
        "28 A1 features."
    )

invalid_df = quality_df.copy()
invalid_df["source"] = invalid_df["feature"].map(
    lambda feature: (
        "derived"
        if feature in {
            "gcs_total",
            "gcs_unable",
            "gcs_components_observed",
            "resp_support_mode",
        }
        else (
            "chartevents"
            if feature in CHART_NUMERIC_ITEM_TO_FEATURE.values()
            else "labevents"
        )
    )
)

invalid_df = invalid_df[
    [
        "feature",
        "source",
        "raw_count",
        "valid_count",
        "invalid_count",
        "invalid_percentage",
        "warning_count",
        "parse_invalid_count",
        "nonfinite_count",
        "range_invalid_count",
        "range_invalid_percentage",
        "normalization_invalid_count",
        "normalization_invalid_percentage",
        "plausibility_rejection_count",
        "plausibility_rejection_percentage",
    ]
].copy()


# ============================================================
# 22. LABEVENTS REFERENCE-RANGE / FLAG AUDIT TABLE
# ============================================================

lab_reference_audit_rows = []

for itemid in LAB_ITEM_IDS:
    itemid = int(itemid)
    state = lab_reference_audit[itemid]
    flag_values = sorted(state["flag_counts"].keys())

    lab_reference_audit_rows.append({
        "source": "labevents",
        "itemid": itemid,
        "feature": LAB_ITEM_TO_FEATURE[itemid],
        "ref_lower_nonnull_count": state[
            "ref_lower_nonnull_count"
        ],
        "ref_upper_nonnull_count": state[
            "ref_upper_nonnull_count"
        ],
        "ref_lower_min": state["ref_lower_min"],
        "ref_lower_max": state["ref_lower_max"],
        "ref_upper_min": state["ref_upper_min"],
        "ref_upper_max": state["ref_upper_max"],
        "distinct_reference_range_pairs": len(
            state["reference_range_pairs"]
        ),
        "reference_range_pairs": ";".join(
            f"{lower:g}|{upper:g}"
            for lower, upper in sorted(
                state["reference_range_pairs"]
            )
        ),
        "flag_nonnull_count": int(
            sum(state["flag_counts"].values())
        ),
        "flag_values": ";".join(flag_values),
        "reference_ranges_used_for_filtering": False,
        "flag_used_for_filtering": False,
    })

lab_reference_audit_df = pd.DataFrame(
    lab_reference_audit_rows
)


# ============================================================
# 23. FiO2 UNIT / REPRESENTATION AUDIT TABLE
# ============================================================

fio2_audit_rows = []

for representation, count in sorted(
    fio2_representation_counts.items()
):
    fio2_audit_rows.append({
        "feature": "fio2",
        "itemid": 223835,
        "representation": representation,
        "count": int(count),
        "normalization_rule": (
            "0.20-1.00 -> x100; 20-100 unchanged; "
            "1-20 invalid; outside supported range invalid"
        ),
        "lower_plausibility_bound_percent": 20.0,
        "upper_plausibility_bound_percent": 100.0,
    })

fio2_audit_df = pd.DataFrame(fio2_audit_rows)

if fio2_audit_df.empty:
    fio2_audit_df = pd.DataFrame(
        columns=[
            "feature",
            "itemid",
            "representation",
            "count",
            "normalization_rule",
            "lower_plausibility_bound_percent",
            "upper_plausibility_bound_percent",
        ]
    )


# ============================================================
# 24. GCS COMPONENT AUDIT TABLE
# ============================================================

if gcs_component_audit_rows:
    gcs_audit_df = pd.DataFrame(
        gcs_component_audit_rows
    )
else:
    gcs_audit_df = pd.DataFrame(
        columns=[
            "itemid",
            "component",
            "raw_rows_after_warning_filter",
            "valid_component_rows",
            "invalid_component_rows",
            "no_response_ett_rows",
        ]
    )


# ============================================================
# 25. FEATURE DEFINITIONS JSON
# ============================================================

feature_definitions = {
    "pipeline": "Pipeline A",
    "version": "V2",
    "stage": "A1",
    "feature_count": len(FEATURE_ORDER),
    "feature_order": FEATURE_ORDER,
    "features": {},
    "respiratory_support_encoding": RESP_SUPPORT_ENCODING,
    "aggregation_rules_for_A2": AGGREGATION_RULES,
    "feature_types": FEATURE_TYPES,
    "landmark_window_hours": 24,
    "trend_features": {
        "computed_in_A1": False,
        "computed_in_A2": False,
        "computed_in_A3": True,
        "specification_status": "deferred_to_A3",
    },
    "warning_policy": {
        "chartevents_warning_flag": (
            "exclude_from_primary_validated_stream"
        ),
        "labevents_warning_field_present": False,
        "warning_counts_preserved_in_QC": True,
    },
    "gcs_semantics": {
        "item_ids": {
            "verbal": 223900,
            "motor": 223901,
            "eyes": 220739,
        },
        "lookback_hours": 6,
        "no_response_ett_rule": (
            "No Response-ETT -> verbal component 0; overall GCS 15; "
            "gcs_unable=1."
        ),
        "missing_component_rule": (
            "Missing components may be defaulted to normal for gcs_total "
            "following the MIMIC-derived convention; the separate "
            "gcs_components_observed feature records how many components "
            "were actually available from current or <=6h prior valid "
            "observations."
        ),
        "gcs_components_observed_range": [0, 3],
        "vectorized_carry_forward": True,
    },
    "labevents_reference_audit": {
        "ref_range_lower_loaded": True,
        "ref_range_upper_loaded": True,
        "flag_loaded": True,
        "used_as_filter": False,
        "purpose": "supplementary unit/quality cross-check only",
    },
    "invalid_value_policy": {
        "invalid_values_written_to_primary_event_stream": False,
        "invalid_counts_preserved": True,
        "invalid_values_become_missing_at_A2": True,
    },
}

for feature in FEATURE_ORDER:
    feature_definitions["features"][feature] = {
        "feature_type": FEATURE_TYPES[feature],
        "aggregation": AGGREGATION_RULES[feature],
        "bounds": (
            list(FEATURE_BOUNDS[feature])
            if feature in FEATURE_BOUNDS
            else None
        ),
        "missingness_channel_planned": True,
        "time_since_channel_planned": True,
    }


# ============================================================
# 26. METADATA
# ============================================================

metadata = {
    "pipeline": "Pipeline A",
    "version": "V2",
    "stage": "A1",
    "source_mimic_version": "3.1",
    "v1_output_directory_preserved": str(V1_OUTPUT_DIR),
    "v2_output_directory": str(V2_OUTPUT_DIR),
    "raw_mimic_read_only": True,
    "a0_cohort_rows": int(len(cohort)),
    "a0_unique_stays": int(cohort["stay_id"].nunique()),
    "a0_unique_subjects": int(cohort["subject_id"].nunique()),
    "first_24_hours_only": True,
    "original_feature_count": 22,
    "total_a1_feature_count": 28,
    "total_validated_events_written": int(
        total_written_events
    ),
    "chart_numeric_raw_candidate_rows": int(
        chart_raw_candidate_rows
    ),
    "lab_numeric_raw_candidate_rows": int(
        lab_raw_candidate_rows
    ),
    "unit_audit_multi_unit_failures": len(
        unit_failures
    ),
    "labevents_reference_ranges_loaded": True,
    "labevents_reference_ranges_used_for_filtering": False,
    "labevents_flag_loaded": True,
    "labevents_flag_used_for_filtering": False,
    "gcs_warning_count": int(gcs_warning_count),
    "gcs_raw_count_after_warning_filter": int(
        gcs_raw_count_after_warning
    ),
    "gcs_derived_event_count": int(gcs_count),
    "resp_support_warning_count": int(
        support_warning_count
    ),
    "resp_support_raw_count_after_warning_filter": int(
        support_raw_count_after_warning
    ),
    "resp_support_derived_event_count": int(support_count),
    "tier1_max_range_rejection_percent": (
        MAX_RANGE_REJECTION_PERCENT
    ),
    "tier1_max_plausibility_rejection_percent": (
        MAX_PLAUSIBILITY_REJECTION_PERCENT
    ),
    "tier1_range_gate_passed": True,
    "tier1_plausibility_gate_passed": True,
    "notes": [
        "A1 performs raw extraction and feature QC only.",
        "A1 does not perform hourly median aggregation.",
        "A1 does not create the complete hourly grid.",
        "A1 does not forward-fill clinical model inputs.",
        "A1 does not create missingness masks for the final tensor.",
        "A1 does not compute time-since-last-observation.",
        "A1 does not compute trend features.",
        "A1 does not perform train-only imputation.",
        "A1 does not perform scaling.",
        "Invalid readings are excluded from the primary event stream and "
        "their counts are preserved in QC artifacts.",
        "FiO2 is normalized to percentage units before plausibility "
        "filtering; 20% is accepted and values >1 and <20 are treated "
        "as improper representations.",
        "GCS preserves the MIMIC-derived total-score convention while "
        "explicitly exposing gcs_components_observed.",
        "labevents ref_range_lower, ref_range_upper, and flag are audited "
        "but never used as exclusion criteria.",
        "GCS carry-forward uses grouped vectorized merge_asof rather than "
        "a Python loop over stays.",
        "Respiratory support classification is vectorized by item family "
        "and max-aggregated according to the clinical hierarchy.",
        "Existing V1 outputs are not modified.",
    ],
}


# ============================================================
# 27. ATOMIC OUTPUT HELPERS + COMMIT
# ============================================================


def atomic_csv_write(df, final_path):
    tmp_path = final_path.with_name(
        final_path.name + ".tmp"
    )
    if tmp_path.exists():
        tmp_path.unlink()
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, final_path)


def atomic_json_write(obj, final_path):
    tmp_path = final_path.with_name(
        final_path.name + ".tmp"
    )
    if tmp_path.exists():
        tmp_path.unlink()

    with open(
        tmp_path,
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            obj,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    os.replace(tmp_path, final_path)


# Keep the final event path untouched until the temporary file is complete.
os.replace(
    TEMP_EVENT_OUTPUT,
    A1_EVENT_OUTPUT,
)

atomic_csv_write(
    mapping_audit,
    A1_MAPPING_AUDIT_OUTPUT,
)
atomic_csv_write(
    unit_audit_df,
    A1_UNIT_AUDIT_OUTPUT,
)
atomic_csv_write(
    lab_reference_audit_df,
    A1_LAB_REFERENCE_AUDIT_OUTPUT,
)
atomic_csv_write(
    fio2_audit_df,
    A1_FIO2_AUDIT_OUTPUT,
)
atomic_csv_write(
    gcs_audit_df,
    A1_GCS_AUDIT_OUTPUT,
)
atomic_csv_write(
    quality_df,
    A1_FEATURE_QUALITY_OUTPUT,
)
atomic_csv_write(
    invalid_df,
    A1_INVALID_OUTPUT,
)
atomic_json_write(
    feature_definitions,
    A1_FEATURE_DEFINITION_OUTPUT,
)
atomic_json_write(
    metadata,
    A1_METADATA_OUTPUT,
)


# ============================================================
# 28. FINAL OUTPUT HARD CHECKS
# ============================================================

print("\n" + "-" * 80)
print("FINAL A1 HARD CHECKS")
print("-" * 80)

expected_outputs = [
    A1_EVENT_OUTPUT,
    A1_FEATURE_QUALITY_OUTPUT,
    A1_INVALID_OUTPUT,
    A1_UNIT_AUDIT_OUTPUT,
    A1_LAB_REFERENCE_AUDIT_OUTPUT,
    A1_FIO2_AUDIT_OUTPUT,
    A1_MAPPING_AUDIT_OUTPUT,
    A1_GCS_AUDIT_OUTPUT,
    A1_FEATURE_DEFINITION_OUTPUT,
    A1_METADATA_OUTPUT,
]

for path in expected_outputs:
    if not path.exists():
        raise AssertionError(
            f"Expected A1 output missing: {path}"
        )
    if path.stat().st_size <= 0:
        raise AssertionError(
            f"A1 output is empty: {path}"
        )

# Canonical Parquet schema.
event_schema_check = pq.read_schema(A1_EVENT_OUTPUT)
if event_schema_check != EVENT_SCHEMA:
    raise AssertionError(
        "Final event Parquet schema does not match the canonical A1 schema."
    )

# Reload the machine-readable QC table and verify all features.
saved_quality = pd.read_csv(
    A1_FEATURE_QUALITY_OUTPUT
)

if set(saved_quality["feature"]) != set(FEATURE_ORDER):
    raise AssertionError(
        "Saved feature_quality_summary.csv does not contain exactly "
        "the canonical 28 features."
    )

if (
    saved_quality["range_invalid_percentage"]
    > MAX_RANGE_REJECTION_PERCENT
).any():
    raise AssertionError(
        "Saved QC reports a Tier-1 range rejection rate >30%."
    )

if (
    saved_quality["plausibility_rejection_percentage"]
    > MAX_PLAUSIBILITY_REJECTION_PERCENT
).any():
    raise AssertionError(
        "Saved QC reports a Tier-1 plausibility rejection rate >30%."
    )

if not gcs_output.empty:
    observed = gcs_output[
        "gcs_components_observed"
    ].to_numpy(dtype=np.int64)
    if not np.isin(observed, [1, 2, 3]).all():
        raise AssertionError(
            "gcs_components_observed contains values outside 1-3 "
            "for emitted GCS events."
        )

    unable = gcs_output[
        "gcs_unable"
    ].to_numpy(dtype=np.int64)
    if not np.isin(unable, [0, 1]).all():
        raise AssertionError(
            "gcs_unable contains a value outside 0-1."
        )

    totals = gcs_output[
        "gcs_total"
    ].to_numpy(dtype=float)
    if not np.isfinite(totals).all():
        raise AssertionError(
            "Final gcs_total contains non-finite values."
        )
    if not ((totals >= 3.0) & (totals <= 15.0)).all():
        raise AssertionError(
            "Final gcs_total contains a value outside 3-15."
        )

if not support_output.empty:
    support_values = support_output[
        "support_code"
    ].to_numpy(dtype=np.int64)
    if not ((support_values >= 0) & (support_values <= 5)).all():
        raise AssertionError(
            "Final respiratory support codes are outside 0-5."
        )

# V2 event path must remain outside V1.
if A1_EVENT_OUTPUT.is_relative_to(V1_OUTPUT_DIR):
    raise AssertionError(
        "A1 output path incorrectly resolves inside V1."
    )

print("\n" + "=" * 80)
print("✅ PIPELINE A V2 — A1 COMPLETE")
print("=" * 80)
print(f"Original variables : 22")
print(f"V2 A1 features     : {len(FEATURE_ORDER)}")
print(f"Validated events   : {total_written_events:,}")
print(f"Chart raw candidates: {chart_raw_candidate_rows:,}")
print(f"Lab raw candidates  : {lab_raw_candidate_rows:,}")

print("\nFeature order:")
for i, feature in enumerate(FEATURE_ORDER, start=1):
    print(f"  {i:02d}. {feature}")

print("\nCreated:")
for path in expected_outputs:
    print(f"  {path}")

print("\nV1 output directory was NOT modified.")
print("Raw MIMIC-IV files were NOT modified.")

print("\nA1 deliberately stops before:")
print("  hourly aggregation")
print("  complete hourly grid")
print("  missingness masks")
print("  forward fill")
print("  time-since-last-observation")
print("  trend features")
print("  train-only imputation")
print("  train-only scaling")

print("=" * 80)
