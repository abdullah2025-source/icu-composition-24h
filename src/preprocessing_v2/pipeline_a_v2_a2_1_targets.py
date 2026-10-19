#!/usr/bin/env python3
"""
Pipeline A V2
Stage A2.1 — Independent 24-hour Target Construction

PURPOSE
-------
Construct the six Pipeline A clinical outputs from the frozen V2 A0 cohort
and raw outcome/treatment timing sources.

This stage is INDEPENDENT of A1 validated feature events and A2 text
embeddings.

CANONICAL V2 TARGET OUTPUTS
---------------------------
1. mortality_24h
2. ventilation_onset_24h
3. vasopressor_onset_24h
4. readmission_30d_24h
5. los_hours_24h
6. deterioration_24h (derived from the first three component outcomes)

IMPORTANT DESIGN DECISIONS
--------------------------
VENTILATION:
    - A1's support vocabulary is shared exactly here for consistency.
    - The target answers a different question:
          A1 = documented support state during the feature window.
          A2.1 = new invasive-ventilation episode after the 24h landmark.
    - Episode construction follows the MIT-LCP ventilation concept's
      interval/event logic, including the 14-hour gap rule.
    - A patient already on invasive ventilation at the 24h landmark is
      NOT assigned a negative onset label; ventilation_onset_24h_valid=0
      and the target value is -1.
    - Thus continuation and new onset are explicitly separated.

VASOPRESSOR:
    - Uses raw ICU inputevents because it contains administration
      starttime/endtime records.
    - Qualifying agents are the five vasopressor agents used for the
      Pipeline A target: norepinephrine, epinephrine, dopamine,
      phenylephrine, vasopressin.
    - Their item IDs are validated against d_items and MIT-LCP concepts.
    - For an onset target, dose unit conversion is NOT required to decide
      whether a qualifying administration interval exists. Units are still
      audited and retained in the audit output.
    - A patient already receiving a qualifying vasopressor at the 24h
      landmark has vasopressor_onset_24h_valid=0 and target value -1.

FULL-STAY TARGET WINDOW
----------------------
Treatment-onset detection uses the complete ICU stay:
    [intime, outtime)
It is NOT clipped to the first 24 hours.

24-HOUR PRIMARY COHORT
----------------------
The V2 A0 cohort remains the authoritative cohort.
For the 24h landmark benchmark, primary target rows are stays with
icu_los_hours >= 24.

The expected verified primary cohort size is 74,572.
Tier-3 features and target availability do not alter A0 membership.

READMISSION
-----------
readmission_30d_24h means a subsequent ICU stay beginning after the
index ICU outtime and within 30 days.

Follow-up is considered valid when:
    - a qualifying readmission is observed, OR
    - 30 days are observable, OR
    - the patient dies before 30 days, making subsequent readmission
      impossible.
Otherwise readmission_30d_24h_valid=0.

LOS QC
------
los_hours_24h is directly checked against outtime-intime.
No arbitrary upper LOS cutoff is imposed.

SOURCE PROVENANCE
-----------------
MIT-LCP MIMIC-IV concepts:
    ventilation.sql
    medication/vasoactive_agent.sql
    medication/norepinephrine.sql
    medication/epinephrine.sql
    medication/dopamine.sql
    medication/phenylephrine.sql
    medication/vasopressin.sql

The script uses raw local MIMIC-IV CSV files and ports only the required
timing/classification logic. It does not assume that local mimiciv_derived
tables are materialized.

RAW MIMIC FILES ARE READ ONLY.
"""

from pathlib import Path
from collections import defaultdict
import json
import os
import shutil
import tempfile
import warnings

import numpy as np
import pandas as pd


# ============================================================
# 0. GLOBAL PATHS / CONFIGURATION
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

V2_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "pipeline_a_v2"
)

TARGET_DIR = V2_OUTPUT_DIR / "targets_24h"

A0_COHORT_PATH = V2_OUTPUT_DIR / "cohort_master.csv"

ADMISSIONS_PATH = HOSP_DIR / "admissions.csv.gz"
ICUSTAYS_PATH = ICU_DIR / "icustays.csv.gz"
CHARTEVENTS_PATH = ICU_DIR / "chartevents.csv.gz"
INPUTEVENTS_PATH = ICU_DIR / "inputevents.csv.gz"
D_ITEMS_PATH = ICU_DIR / "d_items.csv.gz"

OUTCOMES_ALL_PATH = TARGET_DIR / "outcomes_24h.csv"
OUTCOMES_PRIMARY_PATH = TARGET_DIR / "outcomes_24h_primary_cohort.csv"
TARGET_NPZ_PATH = (
    TARGET_DIR
    / "five_task_component_targets_24h.npz"
)
SUMMARY_PATH = TARGET_DIR / "outcomes_24h_summary.csv"
DEFINITION_PATH = TARGET_DIR / "outcomes_24h_target_definition.json"
EVENT_AUDIT_PATH = TARGET_DIR / "target_event_audit_24h.csv"
SOURCE_AUDIT_PATH = TARGET_DIR / "target_source_audit_24h.csv"
VENT_AUDIT_PATH = TARGET_DIR / "ventilation_episode_audit_24h.csv"
VASO_AUDIT_PATH = TARGET_DIR / "vasopressor_interval_audit_24h.csv"
VASO_GAP_AUDIT_PATH = TARGET_DIR / "vasopressor_gap_bridge_audit_24h.csv"
UNIT_AUDIT_PATH = TARGET_DIR / "vasopressor_unit_audit_24h.csv"


CHUNK_SIZE = 2_000_000
EXPECTED_A0_ROWS = 93_224
# Small project-specific continuity tolerance for infusion-order renewal.
# Intervals separated by <1 hour are bridged to avoid mistaking
# documentation/order turnover for a clinical interruption.
VASO_MERGE_TOLERANCE_HOURS = 1.0
EXPECTED_PRIMARY_ROWS = 74_572
LANDMARK_HOURS = 24
READMISSION_DAYS = 30


# ============================================================
# 1. VENTILATION CLASSIFICATION VOCABULARY
#    Shared with the A1 V2 support feature.
# ============================================================

RESP_SUPPORT_ENCODING = {
    "None": 0,
    "SupplementalOxygen": 1,
    "HFNC": 2,
    "NonInvasiveVent": 3,
    "InvasiveVent": 4,
    "Tracheostomy": 5,
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

NO_OXYGEN_DEVICES = {
    "None",
}

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

VENT_SOURCE_ITEMS = {
    226732: "o2_delivery_device",
    223849: "ventilator_mode",
    229314: "ventilator_mode_hamilton",
}

VASOPRESSOR_ITEM_IDS = {
    221906: "norepinephrine",
    221289: "epinephrine",
    221662: "dopamine",
    221749: "phenylephrine",
    222315: "vasopressin",
}


def normalize_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def classify_support_value(itemid, value):
    """
    Shared A1 vocabulary. Returns the highest support category represented
    by the individual source value, or None when the value is not recognized.
    """
    value = normalize_text(value)

    if not value:
        return None

    if itemid == 226732:
        if value in TRACHEOSTOMY_DEVICES:
            return "Tracheostomy"
        if value in INVASIVE_DEVICES:
            return "InvasiveVent"
        if value in NIV_DEVICES:
            return "NonInvasiveVent"
        if value in HFNC_DEVICES:
            return "HFNC"
        if value in SUPPLEMENTAL_OXYGEN_DEVICES:
            return "SupplementalOxygen"
        if value in NO_OXYGEN_DEVICES:
            return "None"
        return None

    if itemid == 223849:
        if value in INVASIVE_VENT_MODES:
            return "InvasiveVent"
        return None

    if itemid == 229314:
        if value in HAMILTON_INVASIVE_MODES:
            return "InvasiveVent"
        if value in HAMILTON_NIV_MODES:
            return "NonInvasiveVent"
        return None

    return None


# ============================================================
# 2. COMMON HELPERS
# ============================================================

def assert_exists(paths):
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Required V2 target source files not found:\n"
            + "\n".join(missing)
        )


def atomic_replace_dataframe(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_name(path.name + ".tmp")

    if tmp.exists():
        tmp.unlink()

    df.to_csv(tmp, index=False)

    os.replace(tmp, path)


def atomic_replace_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_name(path.name + ".tmp")

    if tmp.exists():
        tmp.unlink()

    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(
            obj,
            f,
            indent=2,
            default=str,
        )

    os.replace(tmp, path)


def finite_binary_array(values, name):
    arr = np.asarray(values)

    if not np.isfinite(arr).all():
        raise AssertionError(
            f"{name} contains non-finite values."
        )

    unique = np.unique(arr)

    if not np.isin(unique, [-1, 0, 1]).all():
        raise AssertionError(
            f"{name} contains values outside -1/0/1: {unique}"
        )


# ============================================================
# 3. LOAD FROZEN V2 A0 COHORT
# ============================================================

print("=" * 80)
print("PIPELINE A V2 — A2.1 INDEPENDENT TARGET CONSTRUCTION")
print("=" * 80)

assert_exists([
    A0_COHORT_PATH,
    ADMISSIONS_PATH,
    ICUSTAYS_PATH,
    CHARTEVENTS_PATH,
    INPUTEVENTS_PATH,
    D_ITEMS_PATH,
])

TARGET_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

cohort = pd.read_csv(A0_COHORT_PATH)

if len(cohort) != EXPECTED_A0_ROWS:
    raise AssertionError(
        f"Expected {EXPECTED_A0_ROWS:,} A0 rows; "
        f"found {len(cohort):,}."
    )

required_cohort_cols = {
    "subject_id",
    "hadm_id",
    "stay_id",
    "intime",
    "outtime",
    "icu_los_hours",
    # A0 V2 carries the admission-level metadata needed by A2.1.
    "admittime",
    "dischtime",
    "deathtime",
    "hospital_expire_flag",
}

missing = required_cohort_cols - set(cohort.columns)

if missing:
    raise AssertionError(
        f"A0 cohort missing required columns: {sorted(missing)}"
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

for _col in ["admittime", "dischtime", "deathtime"]:
    cohort[_col] = pd.to_datetime(
        cohort[_col],
        errors="coerce",
    )

cohort["hospital_expire_flag"] = pd.to_numeric(
    cohort["hospital_expire_flag"],
    errors="coerce",
)

if cohort["stay_id"].duplicated().any():
    raise AssertionError(
        "A0 cohort contains duplicate stay_id."
    )

if cohort["intime"].isna().any() or cohort["outtime"].isna().any():
    raise AssertionError(
        "A0 cohort contains missing ICU timestamps."
    )

if (cohort["outtime"] <= cohort["intime"]).any():
    raise AssertionError(
        "A0 cohort contains outtime <= intime."
    )

# Preliminary 24-hour landmark-eligible cohort.
# The final model-target cohort cannot be fixed from ICU LOS alone because
# mortality timing must be checked first.
primary = cohort[
    cohort["icu_los_hours"] >= LANDMARK_HOURS
].copy()

EXPECTED_PRELIMINARY_24H_ROWS = 74_829

if len(primary) != EXPECTED_PRELIMINARY_24H_ROWS:
    raise AssertionError(
        f"Expected {EXPECTED_PRELIMINARY_24H_ROWS:,} preliminary 24h stays; "
        f"found {len(primary):,}."
    )

primary = primary.reset_index(drop=True)

primary["landmark_time"] = (
    primary["intime"]
    + pd.Timedelta(hours=LANDMARK_HOURS)
)

print(f"A0 cohort rows                 : {len(cohort):,}")
print(f"Preliminary >=24h rows         : {len(primary):,}")
print(f"Unique preliminary stays       : {primary['stay_id'].nunique():,}")


# ============================================================
# 4. LOAD ADMISSIONS / ICU STAYS
# ============================================================

print("\n" + "-" * 80)
print("SOURCE: ADMISSIONS + ICUSTAYS")
print("-" * 80)

admissions = pd.read_csv(
    ADMISSIONS_PATH,
    compression="gzip",
    usecols=[
        "subject_id",
        "hadm_id",
        "admittime",
        "dischtime",
        "deathtime",
        "hospital_expire_flag",
    ],
    low_memory=False,
)

admissions["admittime"] = pd.to_datetime(
    admissions["admittime"],
    errors="coerce",
)

admissions["dischtime"] = pd.to_datetime(
    admissions["dischtime"],
    errors="coerce",
)

admissions["deathtime"] = pd.to_datetime(
    admissions["deathtime"],
    errors="coerce",
)

if admissions["hadm_id"].duplicated().any():
    raise AssertionError(
        "Admissions contains duplicate hadm_id rows."
    )

# A0 already stores the admission-level metadata used by A2.1. Do not
# merge those columns back into primary: one admission can legitimately
# contain multiple ICU stays, and the repeated metadata would also create
# _x/_y column collisions. Instead, audit the frozen A0 values against the
# raw admissions table without modifying primary.
primary_admission_audit = primary[
    [
        "subject_id",
        "hadm_id",
        "admittime",
        "dischtime",
        "deathtime",
        "hospital_expire_flag",
    ]
].merge(
    admissions[
        [
            "subject_id",
            "hadm_id",
            "admittime",
            "dischtime",
            "deathtime",
            "hospital_expire_flag",
        ]
    ],
    on=["subject_id", "hadm_id"],
    how="left",
    validate="many_to_one",
    indicator=True,
    suffixes=("_a0", "_raw"),
)

if (primary_admission_audit["_merge"] != "both").any():
    missing_rows = int((primary_admission_audit["_merge"] != "both").sum())
    raise AssertionError(
        f"{missing_rows:,} preliminary 24h stays could not be linked to admissions."
    )

def _same_datetime(left, right):
    return left.eq(right) | (left.isna() & right.isna())

for _col in ["admittime", "dischtime", "deathtime"]:
    _left = primary_admission_audit[f"{_col}_a0"]
    _right = primary_admission_audit[f"{_col}_raw"]
    if not _same_datetime(_left, _right).all():
        raise AssertionError(
            f"A0/raw admissions mismatch detected in {_col}."
        )

_left_flag = pd.to_numeric(
    primary_admission_audit["hospital_expire_flag_a0"],
    errors="coerce",
)
_right_flag = pd.to_numeric(
    primary_admission_audit["hospital_expire_flag_raw"],
    errors="coerce",
)
if not (_left_flag.eq(_right_flag) | (_left_flag.isna() & _right_flag.isna())).all():
    raise AssertionError(
        "A0/raw admissions mismatch detected in hospital_expire_flag."
    )

print(
    "Admission metadata audit: "
    f"{len(primary_admission_audit):,} preliminary stays matched raw admissions; "
    "A0 metadata matches raw admissions."
)

# ------------------------------------------------------------
# Final 24h target-cohort reconciliation
# ------------------------------------------------------------
# The canonical V2 primary landmark cohort is defined from the preliminary
# >=24h cohort by removing two target-timing-invalid groups: (1) stays
# whose known hospital death occurred before the 24h landmark and (2)
# hospital-death records with missing deathtime, for which post-landmark
# mortality timing cannot be established. The counts are asserted against
# the already-verified event-time audit so an accidental cohort drift stops.
preliminary_primary = primary.copy()

mortality_known_time = preliminary_primary["deathtime"].notna()
death_before_landmark = (
    mortality_known_time
    & (preliminary_primary["deathtime"] < preliminary_primary["landmark_time"])
)
mortality_timing_unknown = (
    preliminary_primary["hospital_expire_flag"].astype(int).eq(1)
    & preliminary_primary["deathtime"].isna()
)

pre24_death_count = int(death_before_landmark.sum())
unknown_mortality_time_count = int(mortality_timing_unknown.sum())

print("\nTarget-eligibility audit before exclusions:")
print(f"  preliminary >=24h stays       : {len(preliminary_primary):,}")
print(f"  known death before 24h        : {pre24_death_count:,}")
print(f"  hospital death, no deathtime  : {unknown_mortality_time_count:,}")
print(
    f"  expected final primary       : "
    f"{len(preliminary_primary) - pre24_death_count - unknown_mortality_time_count:,}"
)

if pre24_death_count != 248:
    raise AssertionError(
        f"Expected 248 pre-24h deaths from the validated event-time audit; "
        f"found {pre24_death_count}."
    )

if unknown_mortality_time_count != 9:
    raise AssertionError(
        f"Expected 9 hospital-death records with missing deathtime; "
        f"found {unknown_mortality_time_count}."
    )

reconciled_primary_count = (
    len(preliminary_primary)
    - pre24_death_count
    - unknown_mortality_time_count
)

if reconciled_primary_count != EXPECTED_PRIMARY_ROWS:
    raise AssertionError(
        f"Primary cohort reconciliation failed: expected "
        f"{EXPECTED_PRIMARY_ROWS:,}, got {reconciled_primary_count:,}."
    )

primary = preliminary_primary.loc[
    ~death_before_landmark
    & ~mortality_timing_unknown
].copy().reset_index(drop=True)

if len(primary) != EXPECTED_PRIMARY_ROWS:
    raise AssertionError(
        f"Expected {EXPECTED_PRIMARY_ROWS:,} final primary 24h stays after "
        f"timing exclusions; found {len(primary):,}."
    )

print("\nFinal 24h target-cohort reconciliation:")
print(f"  preliminary >=24h stays       : {len(preliminary_primary):,}")
print(f"  pre-24h deaths excluded       : {pre24_death_count:,}")
print(f"  unknown death-time excluded   : {unknown_mortality_time_count:,}")
print(f"  final primary target cohort   : {len(primary):,}")

icustays = pd.read_csv(
    ICUSTAYS_PATH,
    compression="gzip",
    usecols=[
        "subject_id",
        "hadm_id",
        "stay_id",
        "intime",
        "outtime",
    ],
    low_memory=False,
)

icustays["intime"] = pd.to_datetime(
    icustays["intime"],
    errors="coerce",
)

icustays["outtime"] = pd.to_datetime(
    icustays["outtime"],
    errors="coerce",
)

if icustays["stay_id"].duplicated().any():
    raise AssertionError(
        "icustays contains duplicate stay_id rows."
    )

all_icu = icustays.sort_values(
    ["subject_id", "intime", "stay_id"]
).reset_index(drop=True)


# ============================================================
# 5. MORTALITY TARGET
# ============================================================

print("\n" + "-" * 80)
print("TARGET 1 — MORTALITY")
print("-" * 80)

death_time = primary["deathtime"]

hospital_expire = (
    pd.to_numeric(
        primary["hospital_expire_flag"],
        errors="coerce",
    )
    .fillna(0)
    .astype(int)
)

mortality = (
    death_time.notna()
    & (death_time >= primary["landmark_time"])
)

mortality_fallback = (
    hospital_expire.eq(1)
    & death_time.isna()
)

# Unknown death time has already been excluded from the final primary
# target cohort above. Keep the fallback count for QA, but never invent
# post-landmark timing from hospital_expire_flag alone.
if mortality_fallback.any():
    raise AssertionError(
        "Unexpected mortality fallback rows survived the final primary "
        "cohort timing exclusions."
    )

primary["mortality_24h"] = mortality.astype(
    np.float32
)

primary["mortality_24h_valid"] = True


# ============================================================
# 6. LOS TARGET + QC
# ============================================================

print("\n" + "-" * 80)
print("TARGET 5 — ICU LOS")
print("-" * 80)

# Compute LOS from the authoritative ICU timestamps in float64 first.
# Do the A0 consistency audit BEFORE any float32 storage conversion,
# because float32 cannot preserve sub-millisecond precision across the full
# observed LOS range and can create an artificial mismatch with A0.
los_from_timestamps = (
    (
        primary["outtime"]
        - primary["intime"]
    )
    .dt.total_seconds()
    / 3600.0
).astype(np.float64)

los_difference = (
    los_from_timestamps
    - primary["icu_los_hours"].astype(np.float64)
)

if not np.allclose(
    los_difference.to_numpy(),
    0.0,
    atol=1e-9,
    rtol=1e-12,
):
    max_abs_los_difference = float(
        np.nanmax(np.abs(los_difference.to_numpy()))
    )
    raise AssertionError(
        "LOS does not agree with outtime-intime before storage conversion; "
        f"max absolute difference={max_abs_los_difference:.12g} hours."
    )

# Store the target in float32 only after the precision-sensitive audit.
primary["los_hours_24h"] = los_from_timestamps.astype(np.float32)

if not np.isfinite(
    primary["los_hours_24h"].to_numpy()
).all():
    raise AssertionError(
        "LOS contains non-finite values."
    )

if (
    primary["los_hours_24h"] <= 0
).any():
    raise AssertionError(
        "LOS contains non-positive values."
    )

primary["los_hours_24h_valid"] = True


# ============================================================
# 7. READMISSION TARGET
# ============================================================

print("\n" + "-" * 80)
print("TARGET 4 — 30-DAY ICU READMISSION")
print("-" * 80)

# A subsequent ICU stay must begin after the index ICU outtime.
# We deliberately use the chronologically next ICU stay per subject rather
# than a merge_asof. This is deterministic and avoids any ambiguity caused
# by sorting requirements in grouped merge_asof operations.
future_icu = (
    all_icu[
        [
            "subject_id",
            "hadm_id",
            "stay_id",
            "intime",
            "outtime",
        ]
    ]
    .sort_values(
        ["subject_id", "intime", "stay_id"]
    )
    .copy()
)

future_icu["next_intime_for_subject"] = (
    future_icu
    .groupby("subject_id", observed=True)["intime"]
    .shift(-1)
)

future_icu["next_stay_id_for_subject"] = (
    future_icu
    .groupby("subject_id", observed=True)["stay_id"]
    .shift(-1)
)

next_icu = future_icu[
    [
        "stay_id",
        "next_intime_for_subject",
        "next_stay_id_for_subject",
    ]
].rename(
    columns={
        "next_intime_for_subject": "next_intime",
        "next_stay_id_for_subject": "candidate_stay_id",
    }
)

primary = primary.merge(
    next_icu,
    on="stay_id",
    how="left",
    validate="one_to_one",
)

primary["days_to_next_icu"] = (
    (
        primary["next_intime"]
        - primary["outtime"]
    ).dt.total_seconds()
    / 86400.0
)

readmission_observed = (
    primary["days_to_next_icu"].notna()
    & (primary["days_to_next_icu"] > 0)
    & (primary["days_to_next_icu"] <= READMISSION_DAYS)
)

# Dataset end is the latest observed hospital discharge time.
dataset_end = admissions["dischtime"].max()

if pd.isna(dataset_end):
    raise AssertionError(
        "Could not establish MIMIC dataset end from admissions.dischtime."
    )

thirty_day_end = (
    primary["outtime"]
    + pd.Timedelta(days=READMISSION_DAYS)
)

death_before_30d = (
    primary["deathtime"].notna()
    & (primary["deathtime"] > primary["outtime"])
    & (primary["deathtime"] < thirty_day_end)
)

followup_reaches_30d = (
    thirty_day_end <= dataset_end
)

primary["readmission_30d_24h_valid"] = (
    readmission_observed
    | death_before_30d
    | followup_reaches_30d
)

primary["readmission_30d_24h"] = np.where(
    primary["readmission_30d_24h_valid"],
    readmission_observed.astype(np.float32),
    -1.0,
).astype(np.float32)


# ============================================================
# 8. VENTILATION SOURCE AUDIT
# ============================================================

print("\n" + "-" * 80)
print("TARGET 2 — VENTILATION ONSET")
print("-" * 80)

d_items = pd.read_csv(
    D_ITEMS_PATH,
    compression="gzip",
    usecols=[
        "itemid",
        "label",
        "linksto",
        "category",
        "unitname",
        "param_type",
    ],
    low_memory=False,
)

vent_item_map = d_items[
    d_items["itemid"].isin(
        list(VENT_SOURCE_ITEMS.keys())
    )
].copy()

if set(VENT_SOURCE_ITEMS.keys()) - set(
    vent_item_map["itemid"].astype(int)
):
    missing = sorted(
        set(VENT_SOURCE_ITEMS.keys())
        - set(vent_item_map["itemid"].astype(int))
    )
    raise AssertionError(
        f"Ventilation source item IDs missing from d_items: {missing}"
    )

if not (
    vent_item_map["linksto"]
    .fillna("")
    .str.lower()
    .eq("chartevents")
    .all()
):
    raise AssertionError(
        "Ventilation source item(s) do not all link to chartevents."
    )

# Read all relevant ventilation documentation for primary stays.
primary_stay_ids = set(
    primary["stay_id"].astype("int64")
)

vent_parts = []

vent_reader = pd.read_csv(
    CHARTEVENTS_PATH,
    compression="gzip",
    usecols=[
        "stay_id",
        "charttime",
        "storetime",
        "itemid",
        "value",
        "warning",
    ],
    chunksize=CHUNK_SIZE,
    low_memory=False,
)

vent_raw_rows = 0
vent_warning_rows = 0

for chunk in vent_reader:
    chunk = chunk[
        chunk["stay_id"].isin(primary_stay_ids)
    ]

    if chunk.empty:
        continue

    chunk = chunk[
        chunk["itemid"].isin(
            list(VENT_SOURCE_ITEMS.keys())
        )
    ]

    if chunk.empty:
        continue

    vent_raw_rows += len(chunk)

    warning = (
        pd.to_numeric(
            chunk["warning"],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
        .gt(0)
    )

    vent_warning_rows += int(warning.sum())

    chunk = chunk.loc[
        ~warning
    ].copy()

    if chunk.empty:
        continue

    chunk["charttime"] = pd.to_datetime(
        chunk["charttime"],
        errors="coerce",
    )

    chunk["storetime"] = pd.to_datetime(
        chunk["storetime"],
        errors="coerce",
    )

    chunk = chunk.dropna(
        subset=["charttime"]
    ).copy()

    if chunk.empty:
        continue

    vent_parts.append(
        chunk[
            [
                "stay_id",
                "charttime",
                "storetime",
                "itemid",
                "value",
            ]
        ]
    )

if vent_parts:
    vent_raw = pd.concat(
        vent_parts,
        ignore_index=True,
    )
else:
    vent_raw = pd.DataFrame(
        columns=[
            "stay_id",
            "charttime",
            "storetime",
            "itemid",
            "value",
        ]
    )

del vent_parts


# ============================================================
# 9. DETERMINISTIC VENTILATION STATUS AT EACH DOCUMENTED TIME
# ============================================================

if not vent_raw.empty:

    # Latest stored value for each stay/charttime/item.
    vent_raw = vent_raw.sort_values(
        [
            "stay_id",
            "charttime",
            "itemid",
            "storetime",
        ],
        na_position="first",
    )

    vent_raw = vent_raw.drop_duplicates(
        subset=[
            "stay_id",
            "charttime",
            "itemid",
        ],
        keep="last",
    )

    vent_raw["support_state"] = [
        classify_support_value(
            int(itemid),
            value,
        )
        for itemid, value in zip(
            vent_raw["itemid"],
            vent_raw["value"],
        )
    ]

    vent_raw = vent_raw[
        vent_raw["support_state"].notna()
    ].copy()

else:

    vent_raw["support_state"] = pd.Series(
        dtype="object"
    )


# Multiple device/mode observations at one time collapse to the highest
# documented severity, matching the priority ordering used in the
# MIT-LCP ventilation concept.
if not vent_raw.empty:

    vent_priority = (
        vent_raw["support_state"]
        .map(RESP_SUPPORT_ENCODING)
    )

    vent_raw["support_code"] = (
        vent_priority
        .astype("int8")
    )

    vent_status = (
        vent_raw
        .sort_values(
            [
                "stay_id",
                "charttime",
                "support_code",
                "itemid",
            ]
        )
        .groupby(
            [
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

    vent_status["ventilation_status"] = (
        vent_status["support_code"]
        .map({
            code: state
            for state, code in RESP_SUPPORT_ENCODING.items()
        })
    )

else:

    vent_status = pd.DataFrame(
        columns=[
            "stay_id",
            "charttime",
            "support_code",
            "ventilation_status",
        ]
    )


# ============================================================
# 10. MIT-LCP-STYLE VENTILATION EPISODE CONSTRUCTION
# ============================================================

vent_episodes = []

if not vent_status.empty:

    vent_status = vent_status.sort_values(
        ["stay_id", "charttime"]
    ).reset_index(drop=True)

    vent_status["prev_charttime"] = (
        vent_status
        .groupby("stay_id")["charttime"]
        .shift(1)
    )

    vent_status["prev_status"] = (
        vent_status
        .groupby("stay_id")["ventilation_status"]
        .shift(1)
    )

    vent_status["next_charttime"] = (
        vent_status
        .groupby("stay_id")["charttime"]
        .shift(-1)
    )

    gap_from_prev = (
        (
            vent_status["charttime"]
            - vent_status["prev_charttime"]
        )
        .dt.total_seconds()
        / 3600.0
    )

    new_episode = (
        vent_status["prev_status"].isna()
        | gap_from_prev.ge(14.0)
        | (
            vent_status["prev_status"]
            != vent_status["ventilation_status"]
        )
    )

    vent_status["new_episode"] = (
        new_episode.astype(int)
    )

    vent_status["episode_seq"] = (
        vent_status
        .groupby("stay_id")["new_episode"]
        .cumsum()
    )

    gap_to_next = (
        (
            vent_status["next_charttime"]
            - vent_status["charttime"]
        )
        .dt.total_seconds()
        / 3600.0
    )

    vent_status["episode_end_candidate"] = np.where(
        (
            vent_status["next_charttime"].isna()
            | gap_to_next.ge(14.0)
        ),
        vent_status["charttime"],
        vent_status["next_charttime"],
    )

    vent_episodes = (
        vent_status
        .groupby(
            [
                "stay_id",
                "episode_seq",
            ],
            as_index=False,
            observed=True,
        )
        .agg(
            starttime=("charttime", "min"),
            endtime=(
                "episode_end_candidate",
                "max",
            ),
            ventilation_status=(
                "ventilation_status",
                "max",
            ),
        )
    )

    vent_episodes = pd.DataFrame(
        vent_episodes
    )

else:

    vent_episodes = pd.DataFrame(
        columns=[
            "stay_id",
            "episode_seq",
            "starttime",
            "endtime",
            "ventilation_status",
        ]
    )


# Restrict to the invasive ventilation category for target construction.
invasive_episodes = vent_episodes[
    vent_episodes["ventilation_status"]
    .eq("InvasiveVent")
].copy()

if not invasive_episodes.empty:
    invasive_episodes = invasive_episodes.merge(
        primary[
            [
                "stay_id",
                "intime",
                "outtime",
                "landmark_time",
            ]
        ],
        on="stay_id",
        how="left",
        validate="many_to_one",
    )

    # Clip only for auditing/landmark intersection; the original episode
    # start/end fields remain the MIT-LCP-style episode values.
    invasive_episodes["crosses_landmark"] = (
        (invasive_episodes["starttime"] < invasive_episodes["landmark_time"])
        & (invasive_episodes["endtime"] > invasive_episodes["landmark_time"])
    )

    invasive_episodes["post_landmark_onset"] = (
        (invasive_episodes["starttime"] > invasive_episodes["landmark_time"])
        & (invasive_episodes["starttime"] < invasive_episodes["outtime"])
    )


# ============================================================
# 11. VENTILATION ONSET LABEL + VALIDITY
# ============================================================

vent_stats = (
    primary[
        [
            "stay_id",
            "landmark_time",
            "outtime",
        ]
    ]
    .copy()
)

documented_vent_stays = set(
    invasive_episodes["stay_id"].astype("int64")
) if not invasive_episodes.empty else set()

# Any recognized ventilation state is evidence that the ventilation
# documentation channel was active during the stay.
documented_any_vent = set(
    vent_status["stay_id"].astype("int64")
) if not vent_status.empty else set()

if not invasive_episodes.empty:
    preexisting_vent = (
        invasive_episodes
        .groupby("stay_id")["crosses_landmark"]
        .any()
    )

    post_onset = (
        invasive_episodes
        .groupby("stay_id")["post_landmark_onset"]
        .any()
    )

else:
    preexisting_vent = pd.Series(
        dtype=bool
    )
    post_onset = pd.Series(
        dtype=bool
    )

vent_stats["preexisting_ventilation_at_24h"] = (
    vent_stats["stay_id"]
    .map(preexisting_vent)
    .fillna(False)
    .astype(bool)
)

vent_stats["vent_post24_onset_documented"] = (
    vent_stats["stay_id"]
    .map(post_onset)
    .fillna(False)
    .astype(bool)
)

vent_stats["vent_documentation_present"] = (
    vent_stats["stay_id"]
    .isin(documented_any_vent)
)

vent_stats["ventilation_onset_24h_valid"] = (
    vent_stats["vent_documentation_present"]
    & ~vent_stats["preexisting_ventilation_at_24h"]
)

vent_stats["ventilation_onset_24h"] = np.where(
    vent_stats["ventilation_onset_24h_valid"],
    vent_stats["vent_post24_onset_documented"].astype(np.float32),
    -1.0,
).astype(np.float32)

primary = primary.merge(
    vent_stats[
        [
            "stay_id",
            "preexisting_ventilation_at_24h",
            "vent_post24_onset_documented",
            "ventilation_onset_24h_valid",
            "ventilation_onset_24h",
        ]
    ],
    on="stay_id",
    how="left",
    validate="one_to_one",
)


# ============================================================
# 12. VASOPRESSOR SOURCE VALIDATION
# ============================================================

print("\n" + "-" * 80)
print("TARGET 3 — VASOPRESSOR ONSET")
print("-" * 80)

vaso_item_map = d_items[
    d_items["itemid"].isin(
        list(VASOPRESSOR_ITEM_IDS.keys())
    )
].copy()

missing_vaso = (
    set(VASOPRESSOR_ITEM_IDS.keys())
    - set(vaso_item_map["itemid"].astype(int))
)

if missing_vaso:
    raise AssertionError(
        f"Vasopressor item IDs missing from d_items: "
        f"{sorted(missing_vaso)}"
    )

if not (
    vaso_item_map["linksto"]
    .fillna("")
    .str.lower()
    .eq("inputevents")
    .all()
):
    raise AssertionError(
        "Selected vasopressor item IDs do not all link to inputevents."
    )

source_audit_rows = []

for _, row in vaso_item_map.sort_values("itemid").iterrows():
    itemid = int(row["itemid"])
    expected = VASOPRESSOR_ITEM_IDS[itemid]

    label = normalize_text(row["label"])

    if expected.lower() not in label.lower():
        raise AssertionError(
            f"d_items label mismatch for itemid={itemid}: "
            f"expected token={expected!r}, label={label!r}"
        )

    source_audit_rows.append({
        "source": "inputevents",
        "itemid": itemid,
        "expected_agent": expected,
        "label": label,
        "linksto": row["linksto"],
        "category": row["category"],
        "unitname": row["unitname"],
        "param_type": row["param_type"],
    })

source_audit_df = pd.DataFrame(
    source_audit_rows
)


# ============================================================
# 13. VASOPRESSOR UNIT / SOURCE AUDIT + INTERVAL EXTRACTION
# ============================================================

vaso_parts = []
vaso_unit_rows = []

vaso_reader = pd.read_csv(
    INPUTEVENTS_PATH,
    compression="gzip",
    usecols=[
        "stay_id",
        "itemid",
        "starttime",
        "endtime",
        "storetime",
        "amount",
        "amountuom",
        "rate",
        "rateuom",
        "patientweight",
        "statusdescription",
        "originalamount",
        "originalrate",
        "linkorderid",
    ],
    chunksize=CHUNK_SIZE,
    low_memory=False,
)

vaso_raw_rows = 0

for chunk in vaso_reader:

    chunk = chunk[
        chunk["stay_id"].isin(primary_stay_ids)
    ]

    if chunk.empty:
        continue

    chunk = chunk[
        chunk["itemid"].isin(
            list(VASOPRESSOR_ITEM_IDS.keys())
        )
    ]

    if chunk.empty:
        continue

    vaso_raw_rows += len(chunk)

    # Audit the raw unit representations. These are never used as
    # exclusion criteria for the onset flag.
    for itemid, grp in chunk.groupby(
        "itemid",
        observed=True,
    ):
        itemid = int(itemid)

        rate_units = (
            grp["rateuom"]
            .dropna()
            .astype(str)
            .str.strip()
        )

        amount_units = (
            grp["amountuom"]
            .dropna()
            .astype(str)
            .str.strip()
        )

        vaso_unit_rows.append({
            "itemid": itemid,
            "agent": VASOPRESSOR_ITEM_IDS[itemid],
            "distinct_rateuom": ";".join(
                sorted(
                    u for u in rate_units.unique()
                    if u
                )
            ),
            "distinct_amountuom": ";".join(
                sorted(
                    u for u in amount_units.unique()
                    if u
                )
            ),
            "rateuom_count": int(
                sum(bool(u) for u in rate_units.unique())
            ),
            "amountuom_count": int(
                sum(bool(u) for u in amount_units.unique())
            ),
            "rows_seen": int(len(grp)),
        })

    chunk["starttime"] = pd.to_datetime(
        chunk["starttime"],
        errors="coerce",
    )

    chunk["endtime"] = pd.to_datetime(
        chunk["endtime"],
        errors="coerce",
    )

    chunk["storetime"] = pd.to_datetime(
        chunk["storetime"],
        errors="coerce",
    )

    chunk["rate"] = pd.to_numeric(
        chunk["rate"],
        errors="coerce",
    )

    chunk["amount"] = pd.to_numeric(
        chunk["amount"],
        errors="coerce",
    )

    valid_times = (
        chunk["starttime"].notna()
        & chunk["endtime"].notna()
        & (chunk["endtime"] > chunk["starttime"])
    )

    # For binary onset, retain actual positive administration evidence.
    # We do NOT require one universal dose unit.
    positive_signal = (
        chunk["rate"].fillna(0).gt(0)
        | (
            chunk["rate"].isna()
            & chunk["amount"].fillna(0).gt(0)
        )
    )

    chunk = chunk[
        valid_times
        & positive_signal
    ].copy()

    if chunk.empty:
        continue

    # Clip intervals to their ICU stay before unioning them.
    chunk = chunk.merge(
        primary[
            [
                "stay_id",
                "intime",
                "outtime",
            ]
        ],
        on="stay_id",
        how="left",
        validate="many_to_one",
    )

    chunk["interval_start"] = chunk[
        ["starttime", "intime"]
    ].max(axis=1)

    chunk["interval_end"] = chunk[
        ["endtime", "outtime"]
    ].min(axis=1)

    chunk = chunk[
        chunk["interval_end"] > chunk["interval_start"]
    ].copy()

    if chunk.empty:
        continue

    vaso_parts.append(
        chunk[
            [
                "stay_id",
                "itemid",
                "starttime",
                "endtime",
                "interval_start",
                "interval_end",
                "rate",
                "amount",
                "rateuom",
                "amountuom",
                "linkorderid",
            ]
        ]
    )

if vaso_parts:
    vaso_raw = pd.concat(
        vaso_parts,
        ignore_index=True,
    )
else:
    vaso_raw = pd.DataFrame(
        columns=[
            "stay_id",
            "itemid",
            "starttime",
            "endtime",
            "interval_start",
            "interval_end",
            "rate",
            "amount",
            "rateuom",
            "amountuom",
            "linkorderid",
        ]
    )


if vaso_unit_rows:
    unit_audit_df = (
        pd.DataFrame(vaso_unit_rows)
        .groupby(
            [
                "itemid",
                "agent",
            ],
            as_index=False,
            observed=True,
        )
        .agg(
            distinct_rateuom=(
                "distinct_rateuom",
                lambda s: ";".join(
                    sorted(
                        set(
                            u
                            for x in s
                            for u in str(x).split(";")
                            if u
                        )
                    )
                ),
            ),
            distinct_amountuom=(
                "distinct_amountuom",
                lambda s: ";".join(
                    sorted(
                        set(
                            u
                            for x in s
                            for u in str(x).split(";")
                            if u
                        )
                    )
                ),
            ),
            rows_seen=("rows_seen", "sum"),
        )
    )
else:
    unit_audit_df = pd.DataFrame(
        columns=[
            "itemid",
            "agent",
            "distinct_rateuom",
            "distinct_amountuom",
            "rows_seen",
        ]
    )


# ============================================================
# 14. UNION VASOPRESSOR INTERVALS
# ============================================================

def union_intervals(group):
    """Union qualifying vasopressor intervals with a small continuity gap.

    A gap strictly below VASO_MERGE_TOLERANCE_HOURS is treated as order/
    documentation turnover rather than a clinical interruption. This is a
    project-specific QC rule, not an MIT-LCP published vasopressor gap rule.
    """
    group = group.sort_values(
        [
            "interval_start",
            "interval_end",
        ]
    )

    merged = []
    bridge_audits = []

    current_start = None
    current_end = None
    drugs = set()

    tolerance = pd.Timedelta(
        hours=VASO_MERGE_TOLERANCE_HOURS
    )

    for row in group.itertuples(index=False):
        start = row.interval_start
        end = row.interval_end

        if current_start is None:
            current_start = start
            current_end = end
            drugs = {
                VASOPRESSOR_ITEM_IDS[
                    int(row.itemid)
                ]
            }
            continue

        gap = start - current_end

        # Overlap/touching always merges. Positive gaps are bridged only when
        # they are strictly smaller than the project tolerance.
        if gap <= pd.Timedelta(0) or gap < tolerance:

            if gap > pd.Timedelta(0):
                bridge_audits.append({
                    "stay_id": int(row.stay_id),
                    "previous_interval_end": current_end,
                    "next_interval_start": start,
                    "gap_hours": gap.total_seconds() / 3600.0,
                    "bridged": True,
                })

            if end > current_end:
                current_end = end

            drugs.add(
                VASOPRESSOR_ITEM_IDS[
                    int(row.itemid)
                ]
            )

        else:
            merged.append({
                "stay_id": int(row.stay_id),
                "interval_start": current_start,
                "interval_end": current_end,
                "agents": ";".join(
                    sorted(drugs)
                ),
            })

            current_start = start
            current_end = end
            drugs = {
                VASOPRESSOR_ITEM_IDS[
                    int(row.itemid)
                ]
            }

    if current_start is not None:
        merged.append({
            "stay_id": int(
                group.iloc[0]["stay_id"]
            ),
            "interval_start": current_start,
            "interval_end": current_end,
            "agents": ";".join(
                sorted(drugs)
            ),
        })

    return merged, bridge_audits


vaso_bridge_count = 0
vaso_max_bridged_gap_hours = 0.0
vaso_gap_bridge_rows = []

if not vaso_raw.empty:

    interval_rows = []

    for _, grp in vaso_raw.groupby(
        "stay_id",
        observed=True,
    ):
        merged_rows, bridge_audits = union_intervals(grp)
        interval_rows.extend(merged_rows)
        vaso_gap_bridge_rows.extend(bridge_audits)

    vaso_bridge_count = len(vaso_gap_bridge_rows)

    if vaso_gap_bridge_rows:
        vaso_max_bridged_gap_hours = max(
            row["gap_hours"]
            for row in vaso_gap_bridge_rows
        )

    vaso_intervals = pd.DataFrame(interval_rows)

else:

    vaso_intervals = pd.DataFrame(
        columns=[
            "stay_id",
            "interval_start",
            "interval_end",
            "agents",
        ]
    )


# ============================================================
# 15. VASOPRESSOR ONSET LABEL + VALIDITY
# ============================================================

vaso_stats = primary[
    [
        "stay_id",
        "landmark_time",
        "outtime",
    ]
].copy()

if not vaso_intervals.empty:

    vaso_intervals = vaso_intervals.merge(
        primary[
            [
                "stay_id",
                "landmark_time",
                "outtime",
            ]
        ],
        on="stay_id",
        how="left",
        validate="many_to_one",
    )

    vaso_intervals["crosses_landmark"] = (
        (vaso_intervals["interval_start"] < vaso_intervals["landmark_time"])
        & (vaso_intervals["interval_end"] > vaso_intervals["landmark_time"])
    )

    vaso_intervals["post_landmark_onset"] = (
        (vaso_intervals["interval_start"] > vaso_intervals["landmark_time"])
        & (vaso_intervals["interval_start"] < vaso_intervals["outtime"])
    )

    vaso_preexisting = (
        vaso_intervals
        .groupby("stay_id")["crosses_landmark"]
        .any()
    )

    vaso_post_onset = (
        vaso_intervals
        .groupby("stay_id")["post_landmark_onset"]
        .any()
    )

else:

    vaso_preexisting = pd.Series(
        dtype=bool
    )

    vaso_post_onset = pd.Series(
        dtype=bool
    )

vaso_stats["preexisting_vasopressor_at_24h"] = (
    vaso_stats["stay_id"]
    .map(vaso_preexisting)
    .fillna(False)
    .astype(bool)
)

vaso_stats["vaso_post24_onset_documented"] = (
    vaso_stats["stay_id"]
    .map(vaso_post_onset)
    .fillna(False)
    .astype(bool)
)

# Once source QC has succeeded, a stay with no qualifying interval is a
# valid documented negative for this administration-based target.
vaso_stats["vasopressor_onset_24h_valid"] = (
    ~vaso_stats["preexisting_vasopressor_at_24h"]
)

vaso_stats["vasopressor_onset_24h"] = np.where(
    vaso_stats["vasopressor_onset_24h_valid"],
    vaso_stats["vaso_post24_onset_documented"].astype(np.float32),
    -1.0,
).astype(np.float32)

primary = primary.merge(
    vaso_stats[
        [
            "stay_id",
            "preexisting_vasopressor_at_24h",
            "vaso_post24_onset_documented",
            "vasopressor_onset_24h_valid",
            "vasopressor_onset_24h",
        ]
    ],
    on="stay_id",
    how="left",
    validate="one_to_one",
)


# ============================================================
# 16. DERIVED DETERIORATION OUTPUT
# ============================================================

print("\n" + "-" * 80)
print("OUTPUT 6 — DERIVED DETERIORATION")
print("-" * 80)

component_validity = (
    primary["mortality_24h_valid"].astype(bool)
    & primary["ventilation_onset_24h_valid"].astype(bool)
    & primary["vasopressor_onset_24h_valid"].astype(bool)
)

primary["deterioration_24h_valid"] = (
    component_validity
)

primary["deterioration_24h"] = np.where(
    component_validity,
    (
        primary["mortality_24h"].eq(1)
        | primary["ventilation_onset_24h"].eq(1)
        | primary["vasopressor_onset_24h"].eq(1)
    ).astype(np.float32),
    -1.0,
).astype(np.float32)

primary["death_after_24h"] = (
    primary["mortality_24h"].eq(1)
)


# ============================================================
# 17. TARGET HARD CHECKS
# ============================================================

print("\n" + "-" * 80)
print("FINAL A2.1 TARGET HARD CHECKS")
print("-" * 80)

target_columns = [
    "mortality_24h",
    "ventilation_onset_24h",
    "vasopressor_onset_24h",
    "readmission_30d_24h",
    "los_hours_24h",
    "deterioration_24h",
]

validity_columns = [
    "mortality_24h_valid",
    "ventilation_onset_24h_valid",
    "vasopressor_onset_24h_valid",
    "readmission_30d_24h_valid",
    "los_hours_24h_valid",
    "deterioration_24h_valid",
]

binary_target_columns = [
    "mortality_24h",
    "ventilation_onset_24h",
    "vasopressor_onset_24h",
    "readmission_30d_24h",
    "deterioration_24h",
]

for col in binary_target_columns:
    finite_binary_array(
        primary[col].to_numpy(),
        col,
    )

los_values = primary["los_hours_24h"].to_numpy(dtype=np.float64)
if not np.isfinite(los_values).all():
    raise AssertionError("los_hours_24h contains non-finite values.")
if not (los_values > 0).all():
    raise AssertionError("los_hours_24h contains non-positive values.")

for col in validity_columns:
    if not primary[col].astype(bool).isin([True, False]).all():
        raise AssertionError(
            f"{col} is not boolean-valued."
        )

if len(primary) != EXPECTED_PRIMARY_ROWS:
    raise AssertionError(
        "Primary cohort size changed during target construction."
    )

if primary["stay_id"].duplicated().any():
    raise AssertionError(
        "Duplicate stay_id after target construction."
    )

# Treatment onset cannot be positive when the target is invalid.
for value_col, valid_col in [
    (
        "ventilation_onset_24h",
        "ventilation_onset_24h_valid",
    ),
    (
        "vasopressor_onset_24h",
        "vasopressor_onset_24h_valid",
    ),
]:
    bad = (
        primary[value_col].eq(1)
        & ~primary[valid_col].astype(bool)
    )

    if bad.any():
        raise AssertionError(
            f"{value_col} contains positive labels with invalid mask."
        )

# Deterioration must agree with component outcomes when valid.
valid_det = primary["deterioration_24h_valid"].astype(bool)

expected_det = (
    primary.loc[valid_det, "mortality_24h"].eq(1)
    | primary.loc[valid_det, "ventilation_onset_24h"].eq(1)
    | primary.loc[valid_det, "vasopressor_onset_24h"].eq(1)
).astype(np.float32)

actual_det = primary.loc[
    valid_det,
    "deterioration_24h",
].astype(np.float32)

if not np.array_equal(
    expected_det.to_numpy(),
    actual_det.to_numpy(),
):
    raise AssertionError(
        "Deterioration output does not agree with component targets."
    )


# ============================================================
# 18. BUILD FULL A0 TARGET TABLE
# ============================================================

# Keep the entire frozen A0 cohort visible. Targets are populated only
# for the 24h primary cohort; non-primary rows have explicit eligibility=0.
all_outcomes = cohort[
    [
        "subject_id",
        "hadm_id",
        "stay_id",
        "intime",
        "outtime",
        "icu_los_hours",
    ]
].copy()

all_outcomes["primary_24h_eligible"] = (
    all_outcomes["icu_los_hours"] >= 24
)

target_cols_to_copy = [
    "stay_id",
    "landmark_time",
    "mortality_24h",
    "mortality_24h_valid",
    "ventilation_onset_24h",
    "ventilation_onset_24h_valid",
    "vasopressor_onset_24h",
    "vasopressor_onset_24h_valid",
    "readmission_30d_24h",
    "readmission_30d_24h_valid",
    "los_hours_24h",
    "los_hours_24h_valid",
    "deterioration_24h",
    "deterioration_24h_valid",
    "death_after_24h",
    "preexisting_ventilation_at_24h",
    "preexisting_vasopressor_at_24h",
]

target_transfer = primary[
    target_cols_to_copy
].copy()

all_outcomes = all_outcomes.merge(
    target_transfer,
    on="stay_id",
    how="left",
    validate="one_to_one",
)

# Non-primary stays explicitly carry NA targets rather than fabricated
# negatives. The primary model dataset below contains only valid 24h rows.
all_outcomes = all_outcomes.sort_values(
    "stay_id"
).reset_index(drop=True)

primary_output = primary.sort_values(
    "stay_id"
).reset_index(drop=True)


# ============================================================
# 19. TARGET SUMMARY / AUDITS
# ============================================================

summary_rows = []

for col in [
    "mortality_24h",
    "ventilation_onset_24h",
    "vasopressor_onset_24h",
    "readmission_30d_24h",
    "deterioration_24h",
]:
    s = primary_output[col]

    summary_rows.append({
        "target": col,
        "n_primary": len(primary_output),
        "n_valid": int(
            (
                primary_output[
                    {
                        "mortality_24h": "mortality_24h_valid",
                        "ventilation_onset_24h": "ventilation_onset_24h_valid",
                        "vasopressor_onset_24h": "vasopressor_onset_24h_valid",
                        "readmission_30d_24h": "readmission_30d_24h_valid",
                        "deterioration_24h": "deterioration_24h_valid",
                    }[col]
                ]
                .astype(bool)
            ).sum()
        ),
        "n_positive": int(
            s.eq(1).sum()
        ),
        "n_negative": int(
            s.eq(0).sum()
        ),
        "n_unknown": int(
            s.eq(-1).sum()
        ),
    })

summary_rows.append({
    "target": "los_hours_24h",
    "n_primary": len(primary_output),
    "n_valid": int(
        primary_output["los_hours_24h_valid"]
        .astype(bool)
        .sum()
    ),
    "n_positive": np.nan,
    "n_negative": np.nan,
    "n_unknown": int(
        primary_output["los_hours_24h"].isna().sum()
    ),
    "min": float(
        primary_output["los_hours_24h"].min()
    ),
    "median": float(
        primary_output["los_hours_24h"].median()
    ),
    "p99": float(
        primary_output["los_hours_24h"].quantile(0.99)
    ),
    "max": float(
        primary_output["los_hours_24h"].max()
    ),
})

summary_df = pd.DataFrame(
    summary_rows
)


event_audit = pd.DataFrame({
    "metric": [
        "A0_rows",
        "preliminary_24h_rows",
        "pre24_deaths_excluded_from_primary",
        "unknown_death_time_excluded_from_primary",
        "primary_24h_rows",
        "mortality_positive",
        "ventilation_valid",
        "ventilation_preexisting_invalid",
        "ventilation_positive",
        "vasopressor_valid",
        "vasopressor_preexisting_invalid",
        "vasopressor_positive",
        "readmission_valid",
        "readmission_positive",
        "deterioration_valid",
        "deterioration_positive",
        "chartevent_vent_raw_rows",
        "chartevent_vent_warning_rows_excluded",
        "inputevent_vasopressor_raw_rows",
        "vasopressor_intervals_bridged_by_tolerance",
        "vasopressor_max_bridged_gap_hours",
        "mortality_hospital_expire_flag_fallback",
    ],
    "value": [
        len(cohort),
        len(preliminary_primary),
        pre24_death_count,
        unknown_mortality_time_count,
        len(primary_output),
        int(
            primary_output["mortality_24h"].eq(1).sum()
        ),
        int(
            primary_output[
                "ventilation_onset_24h_valid"
            ].astype(bool).sum()
        ),
        int(
            primary_output[
                "preexisting_ventilation_at_24h"
            ].astype(bool).sum()
        ),
        int(
            primary_output["ventilation_onset_24h"].eq(1).sum()
        ),
        int(
            primary_output[
                "vasopressor_onset_24h_valid"
            ].astype(bool).sum()
        ),
        int(
            primary_output[
                "preexisting_vasopressor_at_24h"
            ].astype(bool).sum()
        ),
        int(
            primary_output[
                "vasopressor_onset_24h"
            ].eq(1).sum()
        ),
        int(
            primary_output[
                "readmission_30d_24h_valid"
            ].astype(bool).sum()
        ),
        int(
            primary_output[
                "readmission_30d_24h"
            ].eq(1).sum()
        ),
        int(
            primary_output[
                "deterioration_24h_valid"
            ].astype(bool).sum()
        ),
        int(
            primary_output[
                "deterioration_24h"
            ].eq(1).sum()
        ),
        vent_raw_rows,
        vent_warning_rows,
        vaso_raw_rows,
        vaso_bridge_count,
        vaso_max_bridged_gap_hours,
        int(
            (
                hospital_expire.eq(1)
                & death_time.isna()
            ).sum()
        ),
    ],
})


vent_audit = (
    invasive_episodes
    if not invasive_episodes.empty
    else pd.DataFrame(
        columns=[
            "stay_id",
            "episode_seq",
            "starttime",
            "endtime",
            "ventilation_status",
            "intime",
            "outtime",
            "landmark_time",
            "crosses_landmark",
            "post_landmark_onset",
        ]
    )
)

vaso_audit = vaso_intervals.copy()

vaso_gap_bridge_audit = pd.DataFrame(
    vaso_gap_bridge_rows,
    columns=[
        "stay_id",
        "previous_interval_end",
        "next_interval_start",
        "gap_hours",
        "bridged",
    ],
)


# ============================================================
# 20. BUILD MODEL TARGET NPZ
# ============================================================

npz_arrays = {
    "mortality_24h": primary_output[
        "mortality_24h"
    ].to_numpy(dtype=np.float32),

    "mortality_24h_valid": primary_output[
        "mortality_24h_valid"
    ].to_numpy(dtype=bool),

    "ventilation_onset_24h": primary_output[
        "ventilation_onset_24h"
    ].to_numpy(dtype=np.float32),

    "ventilation_onset_24h_valid": primary_output[
        "ventilation_onset_24h_valid"
    ].to_numpy(dtype=bool),

    "vasopressor_onset_24h": primary_output[
        "vasopressor_onset_24h"
    ].to_numpy(dtype=np.float32),

    "vasopressor_onset_24h_valid": primary_output[
        "vasopressor_onset_24h_valid"
    ].to_numpy(dtype=bool),

    "readmission_30d_24h": primary_output[
        "readmission_30d_24h"
    ].to_numpy(dtype=np.float32),

    "readmission_30d_24h_valid": primary_output[
        "readmission_30d_24h_valid"
    ].to_numpy(dtype=bool),

    "los_hours_24h": primary_output[
        "los_hours_24h"
    ].to_numpy(dtype=np.float32),

    "los_hours_24h_valid": primary_output[
        "los_hours_24h_valid"
    ].to_numpy(dtype=bool),

    "deterioration_24h": primary_output[
        "deterioration_24h"
    ].to_numpy(dtype=np.float32),

    "deterioration_24h_valid": primary_output[
        "deterioration_24h_valid"
    ].to_numpy(dtype=bool),

    "death_after_24h": primary_output[
        "death_after_24h"
    ].to_numpy(dtype=bool),

    "stay_id": primary_output[
        "stay_id"
    ].to_numpy(dtype=np.int64),

    "subject_id": primary_output[
        "subject_id"
    ].to_numpy(dtype=np.int64),

    "hadm_id": primary_output[
        "hadm_id"
    ].to_numpy(dtype=np.int64),
}


# ============================================================
# 21. TARGET DEFINITIONS / PROVENANCE
# ============================================================

definitions = {
    "stage": "A2.1",
    "purpose": "Independent Pipeline A V2 target construction",
    "a0_rows": int(len(cohort)),
    "preliminary_24h_rows": int(len(preliminary_primary)),
    "pre24_deaths_excluded_from_primary": pre24_death_count,
    "unknown_death_time_excluded_from_primary": unknown_mortality_time_count,
    "primary_24h_rows": int(len(primary_output)),
    "landmark_hours": LANDMARK_HOURS,
    "six_outputs": [
        "mortality_24h",
        "ventilation_onset_24h",
        "vasopressor_onset_24h",
        "readmission_30d_24h",
        "los_hours_24h",
        "deterioration_24h",
    ],
    "primary_trained_tasks": [
        "mortality_24h",
        "ventilation_onset_24h",
        "vasopressor_onset_24h",
        "readmission_30d_24h",
        "los_hours_24h",
    ],
    "derived_output": {
        "name": "deterioration_24h",
        "definition": (
            "mortality_24h OR ventilation_onset_24h OR "
            "vasopressor_onset_24h, when all three component "
            "validity masks are true"
        ),
    },
    "mortality_definition": (
        "Hospital mortality occurring at or after the 24-hour ICU landmark."
    ),
    "ventilation_definition": {
        "classification_decision": (
            "Use the same A1 V2 support vocabulary and source-item mappings "
            "for clinical consistency, while using the MIT-LCP ventilation "
            "episode/interval logic to determine whether a genuinely new "
            "invasive-ventilation episode began after the landmark."
        ),
        "question": (
            "Did a new invasive ventilation episode begin after the "
            "24-hour landmark?"
        ),
        "full_stay_window": "[intime, outtime)",
        "classification_vocabulary": (
            "Shared with the A1 V2 respiratory-support feature."
        ),
        "episode_logic": (
            "MIT-LCP ventilation-style event/interval logic with a "
            "14-hour documentation gap rule."
        ),
        "preexisting_rule": (
            "If an invasive ventilation episode crosses the 24-hour "
            "landmark, onset validity=0 and target=-1."
        ),
        "no_documentation_rule": (
            "If no recognized ventilation state is documented anywhere "
            "in the ICU stay, onset validity=0 and target=-1."
        ),
    },
    "vasopressor_definition": {
        "source_decision": (
            "Use raw ICU inputevents because onset requires administration "
            "start/end timing. Dose-unit fields are audited but binary onset "
            "does not require a universal dose conversion."
        ),
        "question": (
            "Did a new qualifying vasopressor administration interval "
            "begin after the 24-hour landmark?"
        ),
        "source": "icu.inputevents",
        "agents": {
            str(k): v
            for k, v in VASOPRESSOR_ITEM_IDS.items()
        },
        "preexisting_rule": (
            "If a qualifying vasopressor interval crosses the 24-hour "
            "landmark, onset validity=0 and target=-1."
        ),
        "interval_merge_tolerance_hours": VASO_MERGE_TOLERANCE_HOURS,
        "interval_merge_policy": (
            "Qualifying intervals separated by less than the 1-hour "
            "tolerance are bridged to avoid treating infusion-order or "
            "titration turnover as a clinical interruption."
        ),
        "unit_rule": (
            "rateuom and amountuom are audited but never used as a "
            "universal filtering criterion for binary onset."
        ),
    },
    "readmission_definition": (
        "A subsequent ICU stay beginning after the index ICU outtime "
        "and within 30 days, with explicit follow-up validity/censoring logic."
    ),
    "los_definition": (
        "ICU LOS in hours = outtime - intime."
    ),
    "derived_concept_policy": (
        "The script does not assume that local mimiciv_derived concept tables "
        "are materialized. It operates from the raw MIMIC-IV tables and "
        "documents the MIT-LCP concept references used for the required logic."
    ),
    "source_policy": {
        "ventilation": "raw chartevents + d_items",
        "vasopressor": "raw inputevents + d_items",
        "mortality": "raw admissions, with explicit post-landmark timing validity",
        "readmission": "raw icustays + admissions follow-up",
        "los": "A0 ICU timestamps, independently verified",
    },
    "mit_lcp_references": {
        "ventilation": (
            "https://github.com/MIT-LCP/mimic-code/blob/main/"
            "mimic-iv/concepts_postgres/treatment/ventilation.sql"
        ),
        "vasoactive_agent": (
            "https://github.com/MIT-LCP/mimic-code/blob/main/"
            "mimic-iv/concepts_postgres/medication/vasoactive_agent.sql"
        ),
        "norepinephrine": (
            "https://github.com/MIT-LCP/mimic-code/blob/main/"
            "mimic-iv/concepts_postgres/medication/norepinephrine.sql"
        ),
        "epinephrine": (
            "https://github.com/MIT-LCP/mimic-code/blob/main/"
            "mimic-iv/concepts_postgres/medication/epinephrine.sql"
        ),
        "dopamine": (
            "https://github.com/MIT-LCP/mimic-code/blob/main/"
            "mimic-iv/concepts_postgres/medication/dopamine.sql"
        ),
        "phenylephrine": (
            "https://github.com/MIT-LCP/mimic-code/blob/main/"
            "mimic-iv/concepts_postgres/medication/phenylephrine.sql"
        ),
        "vasopressin": (
            "https://github.com/MIT-LCP/mimic-code/blob/main/"
            "mimic-iv/concepts_postgres/medication/vasopressin.sql"
        ),
    },
    "cohort_policy": (
        "Target construction cannot change A0 membership. The 24h "
        "primary target table is a subset defined only by ICU LOS >=24h."
    ),
}


# ============================================================
# 22. ATOMIC COMMIT OF TARGET OUTPUTS
# ============================================================

# CSV outputs.
atomic_replace_dataframe(
    all_outcomes,
    OUTCOMES_ALL_PATH,
)

atomic_replace_dataframe(
    primary_output,
    OUTCOMES_PRIMARY_PATH,
)

atomic_replace_dataframe(
    summary_df,
    SUMMARY_PATH,
)

atomic_replace_dataframe(
    event_audit,
    EVENT_AUDIT_PATH,
)

atomic_replace_dataframe(
    source_audit_df,
    SOURCE_AUDIT_PATH,
)

atomic_replace_dataframe(
    vent_audit,
    VENT_AUDIT_PATH,
)

atomic_replace_dataframe(
    vaso_audit,
    VASO_AUDIT_PATH,
)

atomic_replace_dataframe(
    vaso_gap_bridge_audit,
    VASO_GAP_AUDIT_PATH,
)

atomic_replace_dataframe(
    unit_audit_df,
    UNIT_AUDIT_PATH,
)

atomic_replace_json(
    definitions,
    DEFINITION_PATH,
)

# NPZ uses a temporary file followed by atomic replace.
tmp_npz = TARGET_NPZ_PATH.with_name(
    TARGET_NPZ_PATH.name + ".tmp"
)

if tmp_npz.exists():
    tmp_npz.unlink()

np.savez_compressed(
    tmp_npz,
    **npz_arrays,
)

# np.savez_compressed appends .npz when the supplied filename lacks it.
generated_tmp_npz = Path(
    str(tmp_npz) + ".npz"
)

if not generated_tmp_npz.exists():
    raise AssertionError(
        "Temporary NPZ file was not created as expected."
    )

os.replace(
    generated_tmp_npz,
    TARGET_NPZ_PATH,
)


# ============================================================
# 23. FINAL OUTPUT REPORT
# ============================================================

print("\n" + "=" * 80)
print("✅ PIPELINE A V2 — A2.1 TARGET CONSTRUCTION COMPLETE")
print("=" * 80)

print(f"A0 cohort rows              : {len(cohort):,}")
print(f"Preliminary >=24h rows      : {len(preliminary_primary):,}")
print(f"Final primary target rows   : {len(primary_output):,}")
print(
    "Vasopressor merge tolerance: "
    f"{VASO_MERGE_TOLERANCE_HOURS:.1f}h"
)

print("\nSix Pipeline A outputs:")
for i, name in enumerate(
    [
        "mortality_24h",
        "ventilation_onset_24h",
        "vasopressor_onset_24h",
        "readmission_30d_24h",
        "los_hours_24h",
        "deterioration_24h",
    ],
    start=1,
):
    print(f"  {i}. {name}")

print("\nPrimary targets:")
for col in [
    "mortality_24h",
    "ventilation_onset_24h",
    "vasopressor_onset_24h",
    "readmission_30d_24h",
]:
    valid_col = (
        col.replace(
            "24h",
            "24h_valid"
        )
    )

    print(
        f"  {col}: "
        f"valid={int(primary_output[valid_col].astype(bool).sum()):,}, "
        f"positive={int(primary_output[col].eq(1).sum()):,}, "
        f"unknown={int(primary_output[col].eq(-1).sum()):,}"
    )

print(
    "\nLOS hours: "
    f"min={primary_output['los_hours_24h'].min():.6f}, "
    f"median={primary_output['los_hours_24h'].median():.6f}, "
    f"p99={primary_output['los_hours_24h'].quantile(0.99):.6f}, "
    f"max={primary_output['los_hours_24h'].max():.6f}"
)

print(
    "\nDeterioration: "
    f"valid={int(primary_output['deterioration_24h_valid'].astype(bool).sum()):,}, "
    f"positive={int(primary_output['deterioration_24h'].eq(1).sum()):,}, "
    f"unknown={int(primary_output['deterioration_24h'].eq(-1).sum()):,}"
)

print("\nCreated:")
for path in [
    OUTCOMES_ALL_PATH,
    OUTCOMES_PRIMARY_PATH,
    TARGET_NPZ_PATH,
    SUMMARY_PATH,
    DEFINITION_PATH,
    EVENT_AUDIT_PATH,
    SOURCE_AUDIT_PATH,
    VENT_AUDIT_PATH,
    VASO_AUDIT_PATH,
    VASO_GAP_AUDIT_PATH,
    UNIT_AUDIT_PATH,
]:
    print(f"  {path}")

print("\nRaw MIMIC-IV files were read only.")
print("A1/A2 feature artifacts are not modified.")
print("Target construction is independent of A1/A2 processed values.")
print("=" * 80)
