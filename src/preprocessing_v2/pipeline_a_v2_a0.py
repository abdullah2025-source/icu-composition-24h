#!/usr/bin/env python3

"""
Pipeline A V2
Stage A0 — Cohort Construction + Cohort QC

Purpose
-------
Create the V2 base adult ICU cohort from MIMIC-IV v3.1 while preserving
the existing scientific cohort definition from V1:

    - valid ICU intime/outtime
    - age >= 18
    - ICU LOS >= 6 hours

V2 changes at A0
----------------
1. Read only required columns.
2. Validate source-table uniqueness.
3. Use merge validation to prevent row multiplication.
4. Explicitly check merge row counts.
5. Validate timestamps.
6. Validate age / ICU LOS.
7. Do NOT introduce arbitrary maximum age or LOS limits.
8. Preserve raw gender in cohort metadata.
9. Save machine-readable QC metadata.
10. Write only to pipeline_a_v2.
11. Never modify pipeline_a or raw MIMIC-IV files.

Important
---------
This A0 cohort is the broad adult ICU >=6h cohort.
The later 24-hour benchmark landmark cohort and patient-level split
must be reproduced separately and verified against the established:

    74,572 total
    52,268 train
     7,417 validation
    14,887 test

Do not force those counts at A0.
"""

from pathlib import Path
import json
import os
import sys

import numpy as np
import pandas as pd


# ============================================================
# 0. PATHS
# ============================================================

PROJECT_ROOT = Path("/home/abdullah2025/fyp_project")

MIMIC_IV_DIR = PROJECT_ROOT / "data" / "mimic_iv" / "mimic-iv-3.1"
HOSP_DIR = MIMIC_IV_DIR / "hosp"
ICU_DIR = MIMIC_IV_DIR / "icu"

V1_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "pipeline_a"
V2_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "pipeline_a_v2"

COHORT_PATH = V2_OUTPUT_DIR / "cohort_master.csv"
QC_PATH = V2_OUTPUT_DIR / "a0_qc_summary.json"


# Safety guard: V2 must never point at V1.
if V2_OUTPUT_DIR.resolve() == V1_OUTPUT_DIR.resolve():
    raise RuntimeError(
        "SAFETY ERROR: V2 output directory resolves to V1 output directory."
    )


# ============================================================
# 1. SOURCE FILES
# ============================================================

PATIENTS_PATH = HOSP_DIR / "patients.csv.gz"
ADMISSIONS_PATH = HOSP_DIR / "admissions.csv.gz"
ICUSTAYS_PATH = ICU_DIR / "icustays.csv.gz"


required_sources = {
    "patients": PATIENTS_PATH,
    "admissions": ADMISSIONS_PATH,
    "icustays": ICUSTAYS_PATH,
}

print("=" * 72)
print("PIPELINE A V2 — A0 COHORT CONSTRUCTION")
print("=" * 72)

print("\nSource files:")
for name, path in required_sources.items():
    print(f"  {name:12s}: {path}")
    if not path.exists():
        raise FileNotFoundError(f"Missing required source file: {path}")


print("\nOutputs:")
print(f"  V1 preserved: {V1_OUTPUT_DIR}")
print(f"  V2 target   : {V2_OUTPUT_DIR}")


# ============================================================
# 2. CREATE ONLY V2 OUTPUT DIRECTORY
# ============================================================

V2_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 3. LOAD REQUIRED COLUMNS ONLY
# ============================================================

PATIENT_COLS = [
    "subject_id",
    "anchor_age",
    "gender",
    "anchor_year",
]

ADMISSION_COLS = [
    "subject_id",
    "hadm_id",
    "admittime",
    "dischtime",
    "deathtime",
    "hospital_expire_flag",
]

ICUSTAY_COLS = [
    "subject_id",
    "hadm_id",
    "stay_id",
    "intime",
    "outtime",
]


print("\nLoading MIMIC-IV tables...")

patients = pd.read_csv(
    PATIENTS_PATH,
    compression="gzip",
    usecols=PATIENT_COLS,
    low_memory=False,
)

admissions = pd.read_csv(
    ADMISSIONS_PATH,
    compression="gzip",
    usecols=ADMISSION_COLS,
    low_memory=False,
)

icustays = pd.read_csv(
    ICUSTAYS_PATH,
    compression="gzip",
    usecols=ICUSTAY_COLS,
    low_memory=False,
)

print(f"  patients   : {patients.shape}")
print(f"  admissions : {admissions.shape}")
print(f"  icustays   : {icustays.shape}")


# ============================================================
# 4. SOURCE TABLE UNIQUENESS CHECKS
# ============================================================

print("\nRunning source-table uniqueness checks...")

patients_duplicate_subjects = int(
    patients["subject_id"].duplicated().sum()
)

admissions_duplicate_hadm = int(
    admissions["hadm_id"].duplicated().sum()
)

admissions_duplicate_subject_hadm = int(
    admissions.duplicated(["subject_id", "hadm_id"]).sum()
)

icustays_duplicate_stay = int(
    icustays["stay_id"].duplicated().sum()
)

print(f"  duplicate patient subject_id       : {patients_duplicate_subjects}")
print(f"  duplicate admission hadm_id       : {admissions_duplicate_hadm}")
print(
    "  duplicate admission "
    f"(subject_id, hadm_id)              : {admissions_duplicate_subject_hadm}"
)
print(f"  duplicate ICU stay_id              : {icustays_duplicate_stay}")

if patients_duplicate_subjects != 0:
    raise AssertionError(
        "patients.csv.gz contains duplicate subject_id values."
    )

if admissions_duplicate_hadm != 0:
    raise AssertionError(
        "admissions.csv.gz contains duplicate hadm_id values."
    )

if admissions_duplicate_subject_hadm != 0:
    raise AssertionError(
        "admissions.csv.gz contains duplicate (subject_id, hadm_id)."
    )

if icustays_duplicate_stay != 0:
    raise AssertionError(
        "icustays.csv.gz contains duplicate stay_id values."
    )


# ============================================================
# 5. SOURCE ID / FOREIGN-KEY COVERAGE
# ============================================================

patient_subject_set = set(patients["subject_id"].dropna().astype(int))

admission_hadm_set = set(
    admissions["hadm_id"].dropna().astype(int)
)

admission_subject_hadm_set = set(
    zip(
        admissions["subject_id"].astype("Int64"),
        admissions["hadm_id"].astype("Int64"),
    )
)

missing_patient_subjects = int(
    (~icustays["subject_id"].isin(patient_subject_set)).sum()
)

icustay_pairs = list(
    zip(
        icustays["subject_id"].astype("Int64"),
        icustays["hadm_id"].astype("Int64"),
    )
)

missing_admission_pairs = int(
    sum(pair not in admission_subject_hadm_set for pair in icustay_pairs)
)

missing_admission_hadm = int(
    (~icustays["hadm_id"].isin(admission_hadm_set)).sum()
)

print("\nForeign-key coverage:")
print(f"  ICU stays with missing patient subject_id : {missing_patient_subjects}")
print(f"  ICU stays with missing admission pair    : {missing_admission_pairs}")
print(f"  ICU stays with missing admission hadm_id: {missing_admission_hadm}")

if missing_patient_subjects != 0:
    raise AssertionError(
        "Some ICU stays do not have a matching patient record."
    )

if missing_admission_pairs != 0:
    raise AssertionError(
        "Some ICU stays do not have a matching (subject_id, hadm_id) admission."
    )


# ============================================================
# 6. PRESERVE PRE-MERGE ROW COUNT
# ============================================================

n_icustays_before_merge = len(icustays)
n_unique_stays_before_merge = int(icustays["stay_id"].nunique())

if n_icustays_before_merge != n_unique_stays_before_merge:
    raise AssertionError(
        "icustays is not one-row-per-stay before merging."
    )


# ============================================================
# 7. MERGE — PATIENT DEMOGRAPHICS
# ============================================================

cohort = icustays.merge(
    patients,
    on="subject_id",
    how="left",
    validate="many_to_one",
    indicator="_patient_merge",
)

if len(cohort) != n_icustays_before_merge:
    raise AssertionError(
        "Patient merge changed ICU row count."
    )

patient_unmatched_after_merge = int(
    (cohort["_patient_merge"] != "both").sum()
)

if patient_unmatched_after_merge != 0:
    raise AssertionError(
        "Patient merge contains unmatched ICU stays."
    )

cohort = cohort.drop(columns="_patient_merge")


# ============================================================
# 8. MERGE — ADMISSIONS
# ============================================================

cohort = cohort.merge(
    admissions,
    on=["subject_id", "hadm_id"],
    how="left",
    validate="many_to_one",
    indicator="_admission_merge",
)

if len(cohort) != n_icustays_before_merge:
    raise AssertionError(
        "Admission merge changed ICU row count."
    )

admission_unmatched_after_merge = int(
    (cohort["_admission_merge"] != "both").sum()
)

if admission_unmatched_after_merge != 0:
    raise AssertionError(
        "Admission merge contains unmatched ICU stays."
    )

cohort = cohort.drop(columns="_admission_merge")


# ============================================================
# 9. POST-MERGE ID INTEGRITY
# ============================================================

if len(cohort) != n_icustays_before_merge:
    raise AssertionError(
        "Post-merge row count does not equal original icustays row count."
    )

if cohort["stay_id"].duplicated().any():
    raise AssertionError(
        "Duplicate stay_id values appeared after merging."
    )

if cohort[["subject_id", "hadm_id", "stay_id"]].isna().any().any():
    raise AssertionError(
        "Required ICU identifiers contain null values after merging."
    )


# ============================================================
# 10. TIMESTAMP PARSING
# ============================================================

timestamp_columns = [
    "intime",
    "outtime",
    "admittime",
    "dischtime",
    "deathtime",
]

for col in timestamp_columns:
    cohort[col] = pd.to_datetime(
        cohort[col],
        errors="coerce",
        utc=False,
    )


# ============================================================
# 10A. REQUIRED ICU TIMESTAMP EXCLUSION
# ============================================================
# Preserve the original V1 scientific rule:
# ICU stays with missing intime/outtime are excluded.
#
# IMPORTANT:
# Missing required timestamps are an exclusion condition,
# not a pipeline-failure condition.

missing_intime = int(cohort["intime"].isna().sum())
missing_outtime = int(cohort["outtime"].isna().sum())

print("\nICU timestamp checks BEFORE exclusion:")
print(f"  missing intime : {missing_intime}")
print(f"  missing outtime: {missing_outtime}")

rows_before_timestamp_exclusion = len(cohort)

cohort = cohort.dropna(
    subset=["intime", "outtime"]
).copy()

rows_removed_missing_timestamps = (
    rows_before_timestamp_exclusion - len(cohort)
)

print(f"  rows removed for missing ICU timestamps: "
      f"{rows_removed_missing_timestamps}")

# After applying the exclusion rule, the remaining cohort must
# have valid ICU timestamps.
remaining_missing_intime = int(cohort["intime"].isna().sum())
remaining_missing_outtime = int(cohort["outtime"].isna().sum())

if remaining_missing_intime != 0 or remaining_missing_outtime != 0:
    raise AssertionError(
        "Timestamp exclusion failed: missing ICU timestamps remain."
    )


# ============================================================
# 11. ICU TIME ORDER CHECK
# ============================================================

bad_time_order = int(
    (cohort["outtime"] <= cohort["intime"]).sum()
)

print(f"  outtime <= intime after exclusion: {bad_time_order}")

if bad_time_order != 0:
    raise AssertionError(
        "Found ICU stays where outtime <= intime."
    )


# ============================================================
# 12. AGE CALCULATION — PRESERVE V1 LOGIC
# ============================================================

required_age_columns = [
    "anchor_age",
    "anchor_year",
    "intime",
]

for col in required_age_columns:
    if cohort[col].isna().any():
        raise AssertionError(
            f"Required age field contains null values: {col}"
        )

cohort["age"] = (
    cohort["anchor_age"]
    + (
        cohort["intime"].dt.year
        - cohort["anchor_year"]
    )
)

age_nonfinite = int(
    (~np.isfinite(cohort["age"].astype(float))).sum()
)

print("\nAge:")
print(f"  non-finite: {age_nonfinite}")
print(f"  minimum   : {cohort['age'].min()}")
print(f"  maximum   : {cohort['age'].max()}")

if age_nonfinite != 0:
    raise AssertionError(
        "Non-finite calculated age values found."
    )


# ============================================================
# 13. ICU LOS CALCULATION
# ============================================================

cohort["icu_los_hours"] = (
    (cohort["outtime"] - cohort["intime"])
    .dt.total_seconds()
    / 3600.0
)

los_nonfinite = int(
    (~np.isfinite(cohort["icu_los_hours"].astype(float))).sum()
)

los_nonpositive = int(
    (cohort["icu_los_hours"] <= 0).sum()
)

print("\nICU LOS:")
print(f"  non-finite: {los_nonfinite}")
print(f"  <= 0      : {los_nonpositive}")
print(f"  minimum   : {cohort['icu_los_hours'].min()}")
print(f"  median    : {cohort['icu_los_hours'].median()}")
print(f"  p99       : {cohort['icu_los_hours'].quantile(0.99)}")
print(f"  maximum   : {cohort['icu_los_hours'].max()}")

if los_nonfinite != 0:
    raise AssertionError(
        "Non-finite ICU LOS values found."
    )

if los_nonpositive != 0:
    raise AssertionError(
        "Non-positive ICU LOS values found."
    )


# ============================================================
# 14. V1 SCIENTIFIC INCLUSION CRITERIA — PRESERVE
# ============================================================

n_before_filters = len(cohort)

missing_required_for_filter = int(
    cohort[["intime", "outtime", "age", "icu_los_hours"]]
    .isna()
    .any(axis=1)
    .sum()
)

# Valid ICU timestamps.
cohort = cohort.dropna(
    subset=["intime", "outtime"]
)

# Adult patients only.
cohort = cohort[
    cohort["age"] >= 18
]

# ICU stay at least 6 hours.
cohort = cohort[
    cohort["icu_los_hours"] >= 6
]

cohort = cohort.reset_index(drop=True)

n_after_filters = len(cohort)


# ============================================================
# 15. POST-FILTER HARD CHECKS
# ============================================================

if cohort.empty:
    raise AssertionError(
        "A0 produced an empty cohort."
    )

if int(cohort["stay_id"].duplicated().sum()) != 0:
    raise AssertionError(
        "Duplicate stay_id after A0 filtering."
    )

if int((cohort["age"] < 18).sum()) != 0:
    raise AssertionError(
        "A0 contains a patient younger than 18."
    )

if int((cohort["icu_los_hours"] < 6).sum()) != 0:
    raise AssertionError(
        "A0 contains an ICU stay shorter than 6 hours."
    )

if int((cohort["outtime"] <= cohort["intime"]).sum()) != 0:
    raise AssertionError(
        "A0 contains outtime <= intime."
    )


# ============================================================
# 16. GENDER — PRESERVE SOURCE VALUE
# ============================================================

gender_counts = (
    cohort["gender"]
    .astype("string")
    .fillna("MISSING")
    .value_counts(dropna=False)
    .to_dict()
)

print("\nGender values retained in A0:")
for key, value in sorted(gender_counts.items(), key=lambda x: str(x[0])):
    print(f"  {key}: {value}")


# ============================================================
# 17. RETAIN AUDITABLE COHORT METADATA
# ============================================================

COHORT_COLUMNS = [
    "subject_id",
    "hadm_id",
    "stay_id",
    "age",
    "gender",
    "intime",
    "outtime",
    "icu_los_hours",
    "admittime",
    "dischtime",
    "deathtime",
    "hospital_expire_flag",
]

missing_output_columns = [
    c for c in COHORT_COLUMNS
    if c not in cohort.columns
]

if missing_output_columns:
    raise AssertionError(
        f"Missing required output columns: {missing_output_columns}"
    )

cohort_master = cohort[COHORT_COLUMNS].copy()


# ============================================================
# 18. FINAL A0 SUMMARY
# ============================================================

subject_count = int(cohort_master["subject_id"].nunique())
hadm_count = int(cohort_master["hadm_id"].nunique())
stay_count = int(cohort_master["stay_id"].nunique())

summary = {
    "pipeline": "Pipeline A",
    "version": "V2",
    "stage": "A0",
    "description": "Adult ICU cohort construction with integrity/QC hardening",
    "mimic_version": "3.1",
    "source_root": str(MIMIC_IV_DIR),
    "source_files": {
        "patients": str(PATIENTS_PATH),
        "admissions": str(ADMISSIONS_PATH),
        "icustays": str(ICUSTAYS_PATH),
    },
    "v1_output_dir": str(V1_OUTPUT_DIR),
    "v2_output_dir": str(V2_OUTPUT_DIR),
    "inclusion_criteria": {
        "valid_icu_intime_outtime": True,
        "age_minimum_years": 18,
        "icu_los_minimum_hours": 6,
        "arbitrary_max_age": None,
        "arbitrary_max_icu_los": None,
    },
    "source_rows": {
        "patients": int(len(patients)),
        "admissions": int(len(admissions)),
        "icustays": int(len(icustays)),
    },
    "merge_qc": {
        "icu_rows_before_merge": int(n_icustays_before_merge),
        "icu_unique_stays_before_merge": int(n_unique_stays_before_merge),
        "patient_unmatched": int(patient_unmatched_after_merge),
        "admission_unmatched": int(admission_unmatched_after_merge),
        "missing_patient_subject_ids_in_icustays": int(missing_patient_subjects),
        "missing_admission_pairs_in_icustays": int(missing_admission_pairs),
        "missing_admission_hadm_ids_in_icustays": int(missing_admission_hadm),
        "duplicate_stays_after_merge": int(
            cohort["stay_id"].duplicated().sum()
        ),
    },
    "filter_qc": {
        "rows_before_filters": int(n_before_filters),
        "rows_with_missing_filter_fields_before_filter": int(
            missing_required_for_filter
        ),
        "rows_after_filters": int(n_after_filters),
    },
    "final_counts": {
        "rows": int(len(cohort_master)),
        "unique_subjects": subject_count,
        "unique_hadm_ids": hadm_count,
        "unique_stay_ids": stay_count,
    },
    "final_ranges": {
        "age_min": float(cohort_master["age"].min()),
        "age_median": float(cohort_master["age"].median()),
        "age_max": float(cohort_master["age"].max()),
        "icu_los_hours_min": float(cohort_master["icu_los_hours"].min()),
        "icu_los_hours_median": float(
            cohort_master["icu_los_hours"].median()
        ),
        "icu_los_hours_p99": float(
            cohort_master["icu_los_hours"].quantile(0.99)
        ),
        "icu_los_hours_max": float(
            cohort_master["icu_los_hours"].max()
        ),
    },
    "gender_counts": {
        str(k): int(v)
        for k, v in gender_counts.items()
    },
    "notes": [
        "Raw MIMIC-IV files are read-only.",
        "Existing pipeline_a V1 outputs are untouched.",
        "A0 does not create the final 24-hour benchmark cohort.",
        "The established 74,572 benchmark cohort and patient-level split "
        "must be verified at the later landmark/split stage.",
        "No arbitrary upper age or ICU LOS cutoff is applied.",
        "Raw gender is retained here; downstream demographic encoding remains separate.",
    ],
}


# ============================================================
# 19. ATOMIC WRITE — COHORT CSV
# ============================================================

tmp_cohort = COHORT_PATH.with_name(
    COHORT_PATH.name + ".tmp"
)

cohort_master.to_csv(
    tmp_cohort,
    index=False,
)

os.replace(
    tmp_cohort,
    COHORT_PATH,
)


# ============================================================
# 20. ATOMIC WRITE — QC JSON
# ============================================================

tmp_qc = QC_PATH.with_name(
    QC_PATH.name + ".tmp"
)

with open(tmp_qc, "w", encoding="utf-8") as f:
    json.dump(
        summary,
        f,
        indent=2,
        ensure_ascii=False,
    )

os.replace(
    tmp_qc,
    QC_PATH,
)


# ============================================================
# 21. FINAL OUTPUT VERIFICATION
# ============================================================

saved = pd.read_csv(COHORT_PATH)

if len(saved) != len(cohort_master):
    raise AssertionError(
        "Saved cohort row count does not match in-memory cohort."
    )

if int(saved["stay_id"].duplicated().sum()) != 0:
    raise AssertionError(
        "Saved cohort contains duplicate stay_id."
    )

if int((saved["age"] < 18).sum()) != 0:
    raise AssertionError(
        "Saved cohort contains age < 18."
    )

if int((saved["icu_los_hours"] < 6).sum()) != 0:
    raise AssertionError(
        "Saved cohort contains ICU LOS < 6h."
    )


print("\n" + "=" * 72)
print("✅ PIPELINE A V2 — A0 COMPLETE")
print("=" * 72)
print(f"Base cohort rows : {len(saved):,}")
print(f"Unique subjects  : {saved['subject_id'].nunique():,}")
print(f"Unique admissions: {saved['hadm_id'].nunique():,}")
print(f"Unique ICU stays : {saved['stay_id'].nunique():,}")
print(f"Saved cohort     : {COHORT_PATH}")
print(f"Saved QC         : {QC_PATH}")
print("\nV1 output directory was NOT modified.")
print("No raw MIMIC-IV files were modified.")
print("=" * 72)
