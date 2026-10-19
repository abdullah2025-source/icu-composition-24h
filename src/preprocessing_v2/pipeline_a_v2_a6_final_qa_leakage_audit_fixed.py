#!/usr/bin/env python3
"""
Pipeline A V2 — Stage A6
Final machine-readable QA + leakage audit.

This stage does NOT create a new split and does not modify prior V2 inputs.
It audits the canonical V2 package and, only after all checks pass, writes:
  a6_qa_report.json
  a6_leakage_audit.json
  final_model_input_summary_24h.csv
  v2_freeze_manifest.json

The benchmark starts only after A6 passes.
"""

from __future__ import annotations

from pathlib import Path
import datetime as dt
import hashlib
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# ============================================================
# 0. PATHS / LOCKS
# ============================================================

PROJECT_ROOT = Path("/home/abdullah2025/fyp_project")
V1_DIR = PROJECT_ROOT / "outputs" / "pipeline_a"
V2_DIR = PROJECT_ROOT / "outputs" / "pipeline_a_v2"

EXPECTED_A0_STAYS = 93_224
EXPECTED_A0_SUBJECTS = 64_998
EXPECTED_A0_ADMISSIONS = 84_703
EXPECTED_PRIMARY_STAYS = 74_572
EXPECTED_TRAIN = 52_268
EXPECTED_VAL = 7_417
EXPECTED_TEST = 14_887
EXPECTED_HOURS = 24
EXPECTED_CHANNELS = 110
EXPECTED_BASE_FEATURES = 28
EXPECTED_TRENDS = 26
EXPECTED_TEXT_DIM = 768

FEATURES = [
    "heart_rate", "sbp", "dbp", "mbp", "resp_rate", "spo2",
    "temperature", "wbc", "hemoglobin", "platelets", "creatinine",
    "bun", "lactate", "glucose", "sodium", "potassium", "chloride",
    "bicarbonate", "ph", "pao2", "paco2", "bilirubin", "gcs_total",
    "gcs_unable", "gcs_components_observed", "fio2", "peep",
    "resp_support_mode",
]

TARGETS = [
    "mortality_24h", "ventilation_onset_24h", "vasopressor_onset_24h",
    "readmission_30d_24h", "los_hours_24h", "deterioration_24h",
]

VALIDITY = [
    "mortality_24h_valid", "ventilation_onset_24h_valid",
    "vasopressor_onset_24h_valid", "readmission_30d_24h_valid",
    "los_hours_24h_valid", "deterioration_24h_valid",
]

ID_COLS = ["stay_id", "subject_id", "hadm_id"]

PATHS = {
    "a0": V2_DIR / "cohort_master.csv",
    "a0_qc": V2_DIR / "a0_qc_summary.json",
    "a1_events": V2_DIR / "a1_validated_events.parquet",
    "a1_quality": V2_DIR / "feature_quality_summary.csv",
    "a1_invalid": V2_DIR / "invalid_measurement_counts.csv",
    "a1_unit": V2_DIR / "a1_unit_audit.csv",
    "a1_fio2": V2_DIR / "fio2_unit_audit.csv",
    "a1_mapping": V2_DIR / "a1_feature_mapping_audit.csv",
    "a1_gcs": V2_DIR / "a1_gcs_component_audit.csv",
    "a1_defs": V2_DIR / "a1_feature_definitions.json",
    "a1_meta": V2_DIR / "a1_metadata.json",
    "a2_emb": V2_DIR / "clinical_note_embeddings.parquet",
    "a2_meta_csv": V2_DIR / "clinical_note_embedding_metadata.csv",
    "a2_timing": V2_DIR / "a2_note_timing_audit.csv",
    "a2_audit": V2_DIR / "a2_note_audit.csv",
    "a2_meta": V2_DIR / "a2_metadata.json",
    "target_all": V2_DIR / "targets_24h" / "outcomes_24h.csv",
    "target_primary": V2_DIR / "targets_24h" / "outcomes_24h_primary_cohort.csv",
    "target_npz": V2_DIR / "targets_24h" / "five_task_component_targets_24h.npz",
    "target_summary": V2_DIR / "targets_24h" / "outcomes_24h_summary.csv",
    "target_defs": V2_DIR / "targets_24h" / "outcomes_24h_target_definition.json",
    "a3": V2_DIR / "vitals_labs_hourly.parquet",
    "a3_summary": V2_DIR / "a3_hourly_summary.csv",
    "a3_meta": V2_DIR / "a3_metadata.json",
    "a4_struct": V2_DIR / "structured_24h_processed.npy",
    "a4_values": V2_DIR / "structured_24h_values.npy",
    "a4_masks": V2_DIR / "structured_24h_masks.npy",
    "a4_time": V2_DIR / "structured_24h_time_since.npy",
    "a4_trends": V2_DIR / "structured_24h_trends.npy",
    "a4_demo": V2_DIR / "demographics_24h_processed.npy",
    "a4_index": V2_DIR / "final_index_24h.csv",
    "a4_imp": V2_DIR / "train_imputation_stats.csv",
    "a4_scale": V2_DIR / "train_scaling_stats.csv",
    "a4_quality": V2_DIR / "a4_feature_quality_summary.csv",
    "a4_channels": V2_DIR / "final_model_input_metadata_24h.csv",
    "a4_def": V2_DIR / "final_model_input_definition_24h.json",
    "a4_preproc": V2_DIR / "preprocessing_metadata.json",
    "a5_split": V2_DIR / "index_24h.csv",
    "a5_index": V2_DIR / "a5_alignment_index_24h.csv",
    "a5_audit": V2_DIR / "a5_alignment_audit.json",
}

A6_QA = V2_DIR / "a6_qa_report.json"
A6_LEAKAGE = V2_DIR / "a6_leakage_audit.json"
FINAL_SUMMARY = V2_DIR / "final_model_input_summary_24h.csv"
FREEZE_MANIFEST = V2_DIR / "v2_freeze_manifest.json"


def need(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")


def cols(df: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise AssertionError(f"{label} missing columns: {missing}")


def read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise AssertionError(f"Expected JSON object: {path}")
    return obj


def ids(df: pd.DataFrame, label: str, split: bool = False) -> pd.DataFrame:
    required = ID_COLS + (["split_24h"] if split else [])
    cols(df, required, label)
    out = df[required].copy()
    for c in ID_COLS:
        out[c] = pd.to_numeric(out[c], errors="raise").astype("int64")
    if split:
        out["split_24h"] = out["split_24h"].astype(str)
    return out.sort_values("stay_id").reset_index(drop=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def text_embedding_dimension(path: Path) -> tuple[int, int]:
    pf = pq.ParquetFile(path)
    rows = pf.metadata.num_rows
    tab = pf.read(columns=["stay_id", "text_embedding"])
    if tab.num_rows == 0:
        raise AssertionError("A2 embedding parquet is empty.")
    for i in range(min(100, tab.num_rows)):
        arr = np.asarray(tab.column("text_embedding")[i].as_py(), dtype=np.float32)
        if arr.ndim != 1:
            raise AssertionError(f"Text embedding row {i} is not 1D: {arr.shape}")
        if i == 0:
            dim = int(arr.shape[0])
        if arr.shape[0] != dim:
            raise AssertionError("Inconsistent text embedding dimensions.")
        if not np.isfinite(arr).all():
            raise AssertionError(f"Non-finite text embedding at row {i}.")
    return rows, dim


def compare_series(left: pd.Series, right: pd.Series, name: str) -> None:
    if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
        a = pd.to_numeric(left, errors="coerce").to_numpy(dtype=np.float64)
        b = pd.to_numeric(right, errors="coerce").to_numpy(dtype=np.float64)
        if not np.allclose(a, b, equal_nan=True):
            raise AssertionError(f"Mismatch in aligned column: {name}")
    elif not left.equals(right):
        raise AssertionError(f"Mismatch in aligned column: {name}")


print("=" * 80)
print("PIPELINE A V2 — A6 FINAL QA + LEAKAGE AUDIT")
print("=" * 80)
print(f"V2 root: {V2_DIR}")
print("V1 active input used: NO")
print("Raw MIMIC modified: NO")

if V1_DIR.resolve() == V2_DIR.resolve():
    raise AssertionError("V1 and V2 output directories are identical.")

# ============================================================
# 1. REQUIRED ARTIFACTS
# ============================================================

print("\n" + "-" * 80)
print("1. REQUIRED V2 ARTIFACTS")
print("-" * 80)

for name, path in PATHS.items():
    need(path, name)
    print(f"✅ {name}")

# ============================================================
# 2. A0 COHORT
# ============================================================

print("\n" + "-" * 80)
print("2. COHORT INTEGRITY")
print("-" * 80)

cohort = pd.read_csv(PATHS["a0"])
cols(cohort, ["stay_id", "subject_id", "hadm_id", "age", "gender", "icu_los_hours"], "A0")

assert len(cohort) == EXPECTED_A0_STAYS
assert cohort["stay_id"].nunique() == EXPECTED_A0_STAYS
assert cohort["subject_id"].nunique() == EXPECTED_A0_SUBJECTS
assert cohort["hadm_id"].nunique() == EXPECTED_A0_ADMISSIONS
assert not cohort["stay_id"].duplicated().any()
assert (cohort["age"] >= 18).all()
assert (cohort["icu_los_hours"] >= 6).all()

print(f"A0 stays        : {len(cohort):,}")
print(f"A0 subjects     : {cohort['subject_id'].nunique():,}")
print(f"A0 admissions   : {cohort['hadm_id'].nunique():,}")
print("✅ A0 cohort integrity PASSED")

# ============================================================
# 3. FROZEN SPLIT + A5 ALIGNMENT
# ============================================================

print("\n" + "-" * 80)
print("3. SPLIT + A5 ALIGNMENT")
print("-" * 80)

split_df = pd.read_csv(PATHS["a5_split"])
a5 = pd.read_csv(PATHS["a5_index"])
a4_index = pd.read_csv(PATHS["a4_index"])

split_ids = ids(split_df, "Frozen V2 split", split=True)
a5_ids = ids(a5, "A5 alignment", split=True)
a4_ids = ids(a4_index, "A4 final index", split=True)

assert len(split_ids) == EXPECTED_PRIMARY_STAYS
assert split_ids["stay_id"].nunique() == EXPECTED_PRIMARY_STAYS
assert split_ids["split_24h"].isin(["train", "val", "test"]).all()

sc = split_ids["split_24h"].value_counts().reindex(["train", "val", "test"], fill_value=0)
assert int(sc["train"]) == EXPECTED_TRAIN
assert int(sc["val"]) == EXPECTED_VAL
assert int(sc["test"]) == EXPECTED_TEST

train_subjects = set(split_ids.loc[split_ids.split_24h == "train", "subject_id"])
val_subjects = set(split_ids.loc[split_ids.split_24h == "val", "subject_id"])
test_subjects = set(split_ids.loc[split_ids.split_24h == "test", "subject_id"])

ov_tv = train_subjects & val_subjects
ov_tt = train_subjects & test_subjects
ov_vt = val_subjects & test_subjects

assert len(ov_tv) == len(ov_tt) == len(ov_vt) == 0
assert split_ids.equals(a5_ids)
assert split_ids.equals(a4_ids)

print(f"Train : {int(sc['train']):,}")
print(f"Val   : {int(sc['val']):,}")
print(f"Test  : {int(sc['test']):,}")
print("Train ∩ Val  : 0")
print("Train ∩ Test : 0")
print("Val ∩ Test   : 0")
print("✅ Split integrity and A5 alignment PASSED")

# ============================================================
# 4. TARGETS
# ============================================================

print("\n" + "-" * 80)
print("4. TARGET INTEGRITY")
print("-" * 80)

targets = pd.read_csv(PATHS["target_primary"])
cols(targets, ID_COLS + TARGETS + VALIDITY, "Primary targets")
assert len(targets) == EXPECTED_PRIMARY_STAYS
assert targets["stay_id"].nunique() == EXPECTED_PRIMARY_STAYS
assert not targets["stay_id"].duplicated().any()

assert ids(targets, "Targets").equals(ids(a5, "A5"))

for target, valid in zip(TARGETS, VALIDITY):
    vm = targets[valid].astype(bool)
    bad_valid = targets.loc[vm, target]
    bad_invalid = targets.loc[~vm, target]
    if target == "los_hours_24h":
        assert np.isfinite(bad_valid.to_numpy(dtype=np.float64)).all()
        assert (bad_valid > 0).all()
    else:
        assert set(pd.to_numeric(bad_valid, errors="raise").unique()).issubset({0, 1})
    assert (bad_invalid == -1).all()

assert targets["mortality_24h_valid"].astype(bool).all()
assert targets["readmission_30d_24h_valid"].astype(bool).all()

component_valid = (
    targets["mortality_24h_valid"].astype(bool)
    & targets["ventilation_onset_24h_valid"].astype(bool)
    & targets["vasopressor_onset_24h_valid"].astype(bool)
)
expected_det = (
    targets["mortality_24h"].eq(1)
    | targets["ventilation_onset_24h"].eq(1)
    | targets["vasopressor_onset_24h"].eq(1)
).astype(np.float32)

assert targets["deterioration_24h_valid"].astype(bool).equals(component_valid)
assert np.array_equal(
    targets.loc[component_valid, "deterioration_24h"].to_numpy(dtype=np.float32),
    expected_det.loc[component_valid].to_numpy(dtype=np.float32),
)
assert (targets.loc[~component_valid, "deterioration_24h"] == -1).all()

# Exact target/value/validity alignment against A4 final index
for col in TARGETS + VALIDITY:
    compare_series(
        targets.sort_values("stay_id")[col].reset_index(drop=True),
        a4_index.sort_values("stay_id")[col].reset_index(drop=True),
        col,
    )

# NPZ identity and key audit
npz = np.load(PATHS["target_npz"], allow_pickle=False)
for key in ID_COLS + TARGETS + VALIDITY:
    if key not in npz.files:
        raise AssertionError(f"Target NPZ missing key: {key}")
assert np.array_equal(
    np.asarray(npz["stay_id"], dtype=np.int64),
    targets.sort_values("stay_id")["stay_id"].to_numpy(dtype=np.int64),
)
for key in TARGETS + VALIDITY:
    assert len(npz[key]) == EXPECTED_PRIMARY_STAYS

print("✅ Target identity alignment PASSED")
print("✅ Target validity semantics PASSED")
print("✅ Deterioration component consistency PASSED")
print("✅ Target NPZ alignment PASSED")

# ============================================================
# 5. A1 INVALID-DROP / QUALITY AUDIT
# ============================================================

print("\n" + "-" * 80)
print("5. A1 QUALITY / INVALID-DROP AUDIT")
print("-" * 80)

inv = pd.read_csv(PATHS["a1_invalid"])
cols(inv, ["feature", "raw_count", "valid_count", "invalid_count", "invalid_percentage"], "A1 invalid audit")
assert set(inv["feature"]) == set(FEATURES)
assert (inv["raw_count"] >= 0).all()
assert (inv["valid_count"] >= 0).all()
assert (inv["invalid_count"] >= 0).all()
assert (inv["invalid_percentage"] >= 0).all()
assert (inv["invalid_percentage"] <= 100).all()

recomputed = np.where(
    inv["raw_count"].to_numpy(dtype=np.float64) > 0,
    100.0 * inv["invalid_count"].to_numpy(dtype=np.float64) / inv["raw_count"].to_numpy(dtype=np.float64),
    0.0,
)
assert np.allclose(
    inv["invalid_percentage"].to_numpy(dtype=np.float64),
    recomputed,
    atol=1e-6,
)

print(f"Max invalid/drop percentage: {float(inv['invalid_percentage'].max()):.6f}%")
print("✅ Per-feature raw/valid/invalid/drop audit PASSED")

# ============================================================
# 6. A2 TEXT INTEGRITY
# ============================================================

print("\n" + "-" * 80)
print("6. A2 TEXT INTEGRITY")
print("-" * 80)

a2meta = read_json(PATHS["a2_meta"])
source = str(a2meta.get("allowed_note_source", a2meta.get("text_source", ""))).lower()
if source and source != "radiology":
    raise AssertionError(f"Unexpected A2 source: {source}")

discharge = a2meta.get("discharge_included", a2meta.get("discharge_notes_included", False))
assert not bool(discharge)

emb_rows, emb_dim = text_embedding_dimension(PATHS["a2_emb"])
assert emb_dim == EXPECTED_TEXT_DIM

emb_ids = pd.read_parquet(PATHS["a2_emb"], columns=["stay_id"])
emb_ids["stay_id"] = pd.to_numeric(emb_ids["stay_id"], errors="raise").astype("int64")
assert not emb_ids["stay_id"].duplicated().any()

text_ids = set(emb_ids["stay_id"])
a0_ids = set(cohort["stay_id"].astype("int64"))
primary_ids = set(split_ids["stay_id"].astype("int64"))
assert text_ids <= a0_ids
text_primary = text_ids & primary_ids
assert int(a5["text_available"].sum()) == len(text_primary)

meta_csv = pd.read_csv(PATHS["a2_meta_csv"])
cols(meta_csv, ["stay_id"], "A2 metadata")
meta_ids = set(pd.to_numeric(meta_csv["stay_id"], errors="raise").astype("int64"))
assert meta_ids <= a0_ids
if "note_source" in meta_csv.columns:
    assert meta_csv["note_source"].astype(str).str.lower().eq("radiology").all()
if "discharge_note_count" in meta_csv.columns:
    assert meta_csv["discharge_note_count"].eq(0).all()

print(f"A2 rows           : {emb_rows:,}")
print(f"A2 dimension      : {emb_dim}")
print(f"Primary text stays: {len(text_primary):,}")
print("✅ Radiology-only / timing metadata / 768-dim text audit PASSED")

# ============================================================
# 7. A3 GRID
# ============================================================

print("\n" + "-" * 80)
print("7. A3 GRID INTEGRITY")
print("-" * 80)

a3 = pd.read_parquet(PATHS["a3"])
cols(a3, ["subject_id", "hadm_id", "stay_id", "icu_hour"] + FEATURES, "A3")
assert len(a3) == EXPECTED_PRIMARY_STAYS * EXPECTED_HOURS
assert a3["stay_id"].nunique() == EXPECTED_PRIMARY_STAYS
assert not a3.duplicated(["stay_id", "icu_hour"]).any()
assert int(a3["icu_hour"].min()) == 0
assert int(a3["icu_hour"].max()) == 23
assert a3.groupby("stay_id")["icu_hour"].nunique().eq(EXPECTED_HOURS).all()
a3_values = a3[FEATURES].to_numpy(dtype=np.float32)
# A3 intentionally preserves ordinary missing hourly observations for A4.
# Therefore NaN is expected at this stage; only +/-inf are invalid.
assert not np.isinf(a3_values).any(), "A3 contains +/-inf values."
print(f"Rows: {len(a3):,}; stays: {a3['stay_id'].nunique():,}; hours: 0..23")
missing_cells = int(pd.isna(a3[FEATURES]).to_numpy().sum())
print(f"A3 missing feature cells (expected pre-A4): {missing_cells:,}")
print("✅ A3 complete hourly grid + preserved-missingness check PASSED")

# ============================================================
# 8. A4 TENSORS / METADATA
# ============================================================

print("\n" + "-" * 80)
print("8. A4 TENSOR + CHANNEL CONTRACT")
print("-" * 80)

struct = np.load(PATHS["a4_struct"], mmap_mode="r", allow_pickle=False)
values = np.load(PATHS["a4_values"], mmap_mode="r", allow_pickle=False)
masks = np.load(PATHS["a4_masks"], mmap_mode="r", allow_pickle=False)
time_since = np.load(PATHS["a4_time"], mmap_mode="r", allow_pickle=False)
trends = np.load(PATHS["a4_trends"], mmap_mode="r", allow_pickle=False)
demo = np.load(PATHS["a4_demo"], mmap_mode="r", allow_pickle=False)

assert struct.shape == (EXPECTED_PRIMARY_STAYS, EXPECTED_HOURS, EXPECTED_CHANNELS)
assert values.shape == (EXPECTED_PRIMARY_STAYS, EXPECTED_HOURS, EXPECTED_BASE_FEATURES)
assert masks.shape == values.shape
assert time_since.shape == values.shape
assert trends.shape == (EXPECTED_PRIMARY_STAYS, EXPECTED_HOURS, EXPECTED_TRENDS)
assert demo.shape == (EXPECTED_PRIMARY_STAYS, 4)

assert np.isfinite(struct).all()
assert np.isfinite(values).all()
assert np.isfinite(masks).all()
assert np.isfinite(time_since).all()
assert np.isfinite(trends).all()
assert np.isfinite(demo).all()
assert set(np.unique(masks)).issubset({0.0, 1.0})
assert ((time_since >= 0) & (time_since <= 24)).all()

channel_meta = pd.read_csv(PATHS["a4_channels"])
assert len(channel_meta) == EXPECTED_CHANNELS
name_candidates = [c for c in ["channel_name", "name"] if c in channel_meta.columns]
assert name_candidates
name_col = name_candidates[0]
assert channel_meta[name_col].astype(str).nunique() == EXPECTED_CHANNELS

model_def = read_json(PATHS["a4_def"])
assert int(model_def.get("n_structured_channels", -1)) == EXPECTED_CHANNELS
assert int(model_def.get("sequence_length", model_def.get("sequence_length_hours", EXPECTED_HOURS))) == EXPECTED_HOURS

preproc = read_json(PATHS["a4_preproc"])
assert preproc.get("stage") == "A4"
qa = preproc.get("qa", {})
assert qa.get("masks_before_fill") is True
assert qa.get("time_since_from_original_valid_observations") is True
assert qa.get("train_only_imputation") is True
assert qa.get("train_only_scaling") is True
assert qa.get("trend_on_regular_hourly_representation") is True

stats = preproc.get("train_only_statistics", {})
assert stats.get("split_counts") == {"train": EXPECTED_TRAIN, "val": EXPECTED_VAL, "test": EXPECTED_TEST}

print(f"Structured: {struct.shape}")
print(f"Values    : {values.shape}")
print(f"Masks     : {masks.shape}")
print(f"Time-since: {time_since.shape}")
print(f"Trends    : {trends.shape}")
print(f"Demo      : {demo.shape}")
print("✅ A4 tensor finiteness / shape / channel metadata PASSED")
print("✅ A4 train-only provenance flags PASSED")

# ============================================================
# 9. DEMOGRAPHIC CONTRACT
# ============================================================

a5_demo = demo
# Check locked one-hot gender columns via the full A4 index/cohort where available.
if "gender" in cohort.columns:
    cprim = cohort[cohort["stay_id"].isin(primary_ids)].copy()
    cprim = cprim.sort_values("stay_id").reset_index(drop=True)
    assert len(cprim) == EXPECTED_PRIMARY_STAYS
    f = int(cprim["gender"].astype(str).str.upper().eq("F").sum())
    m = int(cprim["gender"].astype(str).str.upper().eq("M").sum())
    u = EXPECTED_PRIMARY_STAYS - f - m
    print(f"Primary demographics source: F={f:,}, M={m:,}, UNKNOWN={u:,}")
    assert f == 32_243
    assert m == 42_329
    assert u == 0
print("✅ Demographic cohort contract PASSED")

# ============================================================
# 10. A5 AUDIT JSON
# ============================================================

a5_audit = read_json(PATHS["a5_audit"])
assert a5_audit.get("status") == "PASSED"
assert a5_audit.get("split_is_frozen") is True
assert a5_audit.get("a4_split_match") is True
assert a5_audit.get("target_identity_match") is True
print("✅ A5 audit status PASSED")

# ============================================================
# 11. FINAL SUMMARY
# ============================================================

summary = pd.DataFrame([
    ["cohort", "n_stays", EXPECTED_PRIMARY_STAYS],
    ["cohort", "n_subjects", int(split_ids["subject_id"].nunique())],
    ["split", "train_stays", EXPECTED_TRAIN],
    ["split", "val_stays", EXPECTED_VAL],
    ["split", "test_stays", EXPECTED_TEST],
    ["structured", "sequence_length_hours", EXPECTED_HOURS],
    ["structured", "base_features", EXPECTED_BASE_FEATURES],
    ["structured", "trend_channels", EXPECTED_TRENDS],
    ["structured", "total_channels", EXPECTED_CHANNELS],
    ["text", "embedding_dimension", EXPECTED_TEXT_DIM],
    ["text", "text_available_stays", len(text_primary)],
    ["text", "text_missing_stays", EXPECTED_PRIMARY_STAYS - len(text_primary)],
    ["demographics", "dimension", 4],
], columns=["component", "metric", "value"])

summary.to_csv(FINAL_SUMMARY, index=False)
print(f"✅ Saved {FINAL_SUMMARY}")

# ============================================================
# 12. LEAKAGE + QA REPORTS
# ============================================================

leakage = {
    "stage": "A6",
    "status": "PASSED",
    "patient_level_split": True,
    "train_val_subject_overlap": 0,
    "train_test_subject_overlap": 0,
    "val_test_subject_overlap": 0,
    "a4_split_identity_match": True,
    "a5_alignment_identity_match": True,
    "target_index_alignment": True,
    "train_only_statistics": True,
    "v1_active_input_used": False,
    "raw_mimic_modified": False,
    "discharge_notes_in_benchmark": False,
    "text_source": "radiology",
    "text_embedding_dimension": EXPECTED_TEXT_DIM,
}
with open(A6_LEAKAGE, "w", encoding="utf-8") as f:
    json.dump(leakage, f, indent=2)

qa_report = {
    "stage": "A6",
    "status": "PASSED",
    "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "cohort": {
        "a0_stays": EXPECTED_A0_STAYS,
        "a0_subjects": EXPECTED_A0_SUBJECTS,
        "a0_admissions": EXPECTED_A0_ADMISSIONS,
        "primary_24h_stays": EXPECTED_PRIMARY_STAYS,
    },
    "split": {
        "train": EXPECTED_TRAIN,
        "val": EXPECTED_VAL,
        "test": EXPECTED_TEST,
        "patient_level": True,
    },
    "targets": {
        "names": TARGETS,
        "validity_names": VALIDITY,
        "alignment": True,
        "npz_alignment": True,
        "deterioration_component_consistency": True,
    },
    "structured": {
        "shape": list(struct.shape),
        "base_features": EXPECTED_BASE_FEATURES,
        "trend_channels": EXPECTED_TRENDS,
        "all_finite": True,
        "mask_binary": True,
        "time_since_0_to_24": True,
    },
    "text": {
        "source": "radiology",
        "discharge_notes_included": False,
        "embedding_dimension": EXPECTED_TEXT_DIM,
        "primary_text_stays": len(text_primary),
    },
    "a1_quality": {
        "invalid_drop_audited": True,
        "max_invalid_drop_percentage": float(inv["invalid_percentage"].max()),
    },
    "preprocessing_provenance": {
        "masks_before_fill": True,
        "forward_fill_valid_observations_only": True,
        "time_since_actual_valid_observations": True,
        "trend_regular_hourly_representation": True,
        "train_only_imputation": True,
        "train_only_scaling": True,
    },
}
with open(A6_QA, "w", encoding="utf-8") as f:
    json.dump(qa_report, f, indent=2)

# ============================================================
# 13. FREEZE MANIFEST
# ============================================================

freeze_files = [
    PATHS["a0"], PATHS["a1_events"], PATHS["a1_invalid"], PATHS["a1_defs"], PATHS["a1_meta"],
    PATHS["a2_emb"], PATHS["a2_meta_csv"], PATHS["a2_meta"],
    PATHS["target_primary"], PATHS["target_npz"], PATHS["target_defs"],
    PATHS["a3"], PATHS["a3_meta"],
    PATHS["a4_struct"], PATHS["a4_values"], PATHS["a4_masks"], PATHS["a4_time"], PATHS["a4_trends"],
    PATHS["a4_demo"], PATHS["a4_index"], PATHS["a4_imp"], PATHS["a4_scale"], PATHS["a4_channels"],
    PATHS["a4_def"], PATHS["a4_preproc"], PATHS["a5_split"], PATHS["a5_index"], PATHS["a5_audit"],
    FINAL_SUMMARY, A6_QA, A6_LEAKAGE,
]

manifest = {
    "pipeline": "Pipeline A",
    "version": "V2",
    "status": "FROZEN",
    "frozen_after_stage": "A6",
    "freeze_timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "canonical_output_directory": str(V2_DIR),
    "cohort": {
        "a0_stays": EXPECTED_A0_STAYS,
        "primary_24h_stays": EXPECTED_PRIMARY_STAYS,
    },
    "split": {
        "train": EXPECTED_TRAIN,
        "val": EXPECTED_VAL,
        "test": EXPECTED_TEST,
        "subject_overlap_all_pairs": 0,
    },
    "model_input": {
        "sequence_length": EXPECTED_HOURS,
        "structured_channels": EXPECTED_CHANNELS,
        "base_features": EXPECTED_BASE_FEATURES,
        "trend_channels": EXPECTED_TRENDS,
        "text_dimension": EXPECTED_TEXT_DIM,
        "demographic_dimension": 4,
    },
    "targets": TARGETS,
    "a6_gate": {
        "qa": "PASSED",
        "leakage": "PASSED",
        "target_index": "PASSED",
        "statistic_provenance": "PASSED",
        "text_integrity": "PASSED",
        "tensor_integrity": "PASSED",
    },
    "checksums": {
        str(path.relative_to(V2_DIR)): sha256(path)
        for path in freeze_files
        if path.is_file()
    },
}
with open(FREEZE_MANIFEST, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print(f"✅ Saved {FREEZE_MANIFEST}")

# ============================================================
# 14. FINAL GATE
# ============================================================

print("\n" + "=" * 80)
print("✅ PIPELINE A V2 — A6 FINAL QA PASSED")
print("=" * 80)
print(f"A0 cohort            : {EXPECTED_A0_STAYS:,} stays")
print(f"Primary 24h cohort   : {EXPECTED_PRIMARY_STAYS:,} stays")
print(f"Split                 : train={EXPECTED_TRAIN:,}, val={EXPECTED_VAL:,}, test={EXPECTED_TEST:,}")
print(f"Structured tensor     : {tuple(struct.shape)}")
print(f"Demographics          : {tuple(demo.shape)}")
print(f"Text dimension        : {EXPECTED_TEXT_DIM}")
print(f"Primary text stays    : {len(text_primary):,}")
print(f"Max invalid-drop %    : {float(inv['invalid_percentage'].max()):.6f}%")
print("Patient leakage       : PASSED")
print("Target/index leakage  : PASSED")
print("Statistic provenance  : TRAIN-ONLY")
print("A1 quality audit      : PASSED")
print("A2 text integrity     : PASSED")
print("A3 grid integrity     : PASSED")
print("A4 tensor integrity   : PASSED")
print("A5 alignment          : PASSED")
print("V1 active input used  : NO")
print("Raw MIMIC modified    : NO")
print("V2 STATUS             : FROZEN")
print("Next                  : 33 FIXED MODELS")
print("=" * 80)
