# ============================================================
# PIPELINE A V2 — A5
# MULTIMODAL ALIGNMENT + FROZEN SPLIT VERIFICATION
# ============================================================

import json
import os

import numpy as np
import pandas as pd


print("=" * 80)
print("PIPELINE A V2 — A5 MULTIMODAL ALIGNMENT + FROZEN SPLIT AUDIT")
print("=" * 80)


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = "/home/abdullah2025/fyp_project"
V2_DIR = os.path.join(PROJECT_ROOT, "outputs", "pipeline_a_v2")
TARGET_DIR = os.path.join(V2_DIR, "targets_24h")

V2_SPLIT = os.path.join(V2_DIR, "index_24h.csv")
A4_FINAL_INDEX = os.path.join(V2_DIR, "final_index_24h.csv")
A4_STRUCTURED = os.path.join(V2_DIR, "structured_24h_processed.npy")
A4_DEMOGRAPHICS = os.path.join(V2_DIR, "demographics_24h_processed.npy")
A2_EMBEDDINGS = os.path.join(V2_DIR, "clinical_note_embeddings.parquet")
A2_METADATA = os.path.join(V2_DIR, "clinical_note_embedding_metadata.csv")
TARGETS_PATH = os.path.join(TARGET_DIR, "outcomes_24h_primary_cohort.csv")
TARGET_DEFINITION = os.path.join(TARGET_DIR, "outcomes_24h_target_definition.json")
A0_COHORT = os.path.join(V2_DIR, "cohort_master.csv")

A5_ALIGNMENT_INDEX = os.path.join(V2_DIR, "a5_alignment_index_24h.csv")
A5_AUDIT_JSON = os.path.join(V2_DIR, "a5_alignment_audit.json")


# ============================================================
# 2. FROZEN EXPECTATIONS
# ============================================================

EXPECTED_STAYS = 74572
EXPECTED_TRAIN = 52268
EXPECTED_VAL = 7417
EXPECTED_TEST = 14887

EXPECTED_SPLITS = {
    "train": EXPECTED_TRAIN,
    "val": EXPECTED_VAL,
    "test": EXPECTED_TEST,
}

TARGET_COLUMNS = [
    "stay_id",
    "subject_id",
    "hadm_id",
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
]

IDENTITY_COLUMNS = ["stay_id", "subject_id", "hadm_id"]
SPLIT_COLUMNS = ["stay_id", "subject_id", "hadm_id", "split_24h"]


# ============================================================
# 3. HELPERS
# ============================================================

def require_file(path: str, label: str) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label} not found:\n{path}")
    print(f"✅ {label}")


def normalize_ids(df: pd.DataFrame, columns) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        out[col] = pd.to_numeric(out[col], errors="raise").astype("int64")
    return out


def sorted_reset(df: pd.DataFrame, columns) -> pd.DataFrame:
    return df[columns].sort_values("stay_id").reset_index(drop=True)


# ============================================================
# 4. CHECK INPUTS
# ============================================================

print("\n" + "-" * 80)
print("1. CHECKING V2 INPUT ARTIFACTS")
print("-" * 80)

required_files = {
    "V2 frozen split": V2_SPLIT,
    "A4 final index": A4_FINAL_INDEX,
    "A4 structured tensor": A4_STRUCTURED,
    "A4 demographics": A4_DEMOGRAPHICS,
    "A2 embeddings": A2_EMBEDDINGS,
    "A2 metadata": A2_METADATA,
    "A2.1 primary targets": TARGETS_PATH,
    "A2.1 target definition": TARGET_DEFINITION,
}

for label, path in required_files.items():
    require_file(path, label)


# ============================================================
# 5. LOAD + VERIFY FROZEN V2 SPLIT
# ============================================================

print("\n" + "-" * 80)
print("2. VERIFYING FROZEN V2 SPLIT")
print("-" * 80)

split_df = pd.read_csv(V2_SPLIT)
missing = [c for c in SPLIT_COLUMNS if c not in split_df.columns]
if missing:
    raise AssertionError(f"V2 split missing columns: {missing}")

split_df = normalize_ids(split_df[SPLIT_COLUMNS], IDENTITY_COLUMNS)

if len(split_df) != EXPECTED_STAYS:
    raise AssertionError(
        f"Expected {EXPECTED_STAYS:,} split rows, got {len(split_df):,}"
    )

if split_df[IDENTITY_COLUMNS].isna().any().any():
    raise AssertionError("Frozen V2 split contains missing identity values.")

if split_df["split_24h"].isna().any():
    raise AssertionError("Frozen V2 split contains missing split labels.")

if split_df["stay_id"].duplicated().any():
    raise AssertionError("Duplicate stay_id values in frozen V2 split.")

if set(split_df["split_24h"].unique()) != {"train", "val", "test"}:
    raise AssertionError("Unexpected split labels in frozen V2 split.")

split_counts = (
    split_df["split_24h"]
    .value_counts()
    .reindex(["train", "val", "test"], fill_value=0)
)

for name, expected in EXPECTED_SPLITS.items():
    actual = int(split_counts[name])
    print(f"{name.capitalize():5s}: {actual:,} stays (expected {expected:,})")
    if actual != expected:
        raise AssertionError(f"Frozen {name} count mismatch.")


# ============================================================
# 6. PATIENT-LEVEL LEAKAGE CHECK
# ============================================================

print("\n" + "-" * 80)
print("3. VERIFYING PATIENT-LEVEL ISOLATION")
print("-" * 80)

train_subjects = set(split_df.loc[split_df.split_24h == "train", "subject_id"])
val_subjects = set(split_df.loc[split_df.split_24h == "val", "subject_id"])
test_subjects = set(split_df.loc[split_df.split_24h == "test", "subject_id"])

train_val_overlap = train_subjects & val_subjects
train_test_overlap = train_subjects & test_subjects
val_test_overlap = val_subjects & test_subjects

print("Train ∩ Val  :", len(train_val_overlap))
print("Train ∩ Test :", len(train_test_overlap))
print("Val ∩ Test   :", len(val_test_overlap))

if train_val_overlap or train_test_overlap or val_test_overlap:
    raise AssertionError("Patient leakage detected across V2 splits.")

if (
    len(train_subjects) + len(val_subjects) + len(test_subjects)
    != split_df["subject_id"].nunique()
):
    raise AssertionError("Patient split sets do not partition all subjects.")

print("✅ Patient-level leakage check PASSED")


# ============================================================
# 7. LOAD + VERIFY A4 FINAL INDEX
# ============================================================

print("\n" + "-" * 80)
print("4. VERIFYING A4 FINAL INDEX")
print("-" * 80)

a4_index = pd.read_csv(A4_FINAL_INDEX)

missing = [c for c in TARGET_COLUMNS if c not in a4_index.columns]
if missing:
    raise AssertionError(f"A4 final index missing columns: {missing}")

if len(a4_index) != EXPECTED_STAYS:
    raise AssertionError("A4 final index row count mismatch.")

a4_index = normalize_ids(a4_index, IDENTITY_COLUMNS)

if a4_index["stay_id"].duplicated().any():
    raise AssertionError("Duplicate stay_id values in A4 final index.")

frozen_vs_a4 = sorted_reset(split_df, SPLIT_COLUMNS).equals(
    sorted_reset(a4_index, SPLIT_COLUMNS)
)

if not frozen_vs_a4:
    raise AssertionError(
        "A4 final index does not exactly match the frozen V2 split."
    )

print("✅ A4 split assignments exactly match frozen V2 split")


# ============================================================
# 8. LOAD + VERIFY TARGET PACKAGE
# ============================================================

print("\n" + "-" * 80)
print("5. VERIFYING A2.1 TARGET PACKAGE")
print("-" * 80)

targets = pd.read_csv(TARGETS_PATH)
missing = [c for c in TARGET_COLUMNS if c not in targets.columns]
if missing:
    raise AssertionError(f"Target package missing columns: {missing}")

if len(targets) != EXPECTED_STAYS:
    raise AssertionError("Target package row count mismatch.")

targets = normalize_ids(targets[TARGET_COLUMNS], IDENTITY_COLUMNS)

if targets["stay_id"].duplicated().any():
    raise AssertionError("Duplicate stay_id values in target package.")

if not sorted_reset(targets, IDENTITY_COLUMNS).equals(
    sorted_reset(a4_index, IDENTITY_COLUMNS)
):
    raise AssertionError("Target identities do not exactly match A4 identities.")

print("✅ Target identities exactly aligned with A4")


target_split_join = targets[IDENTITY_COLUMNS].merge(
    split_df,
    on=IDENTITY_COLUMNS,
    how="left",
    validate="one_to_one",
)

if target_split_join["split_24h"].isna().any():
    raise AssertionError("Some target rows failed to map to frozen split.")

print("✅ Every target row maps to the frozen split")


# ============================================================
# 9. STRUCTURED TENSOR CHECK
# ============================================================

print("\n" + "-" * 80)
print("6. VERIFYING A4 STRUCTURED TENSOR")
print("-" * 80)

structured = np.load(A4_STRUCTURED, mmap_mode="r")
print("Structured tensor shape:", structured.shape)

if structured.ndim != 3:
    raise AssertionError(f"Expected 3D structured tensor, got {structured.ndim}D.")

if structured.shape[0] != EXPECTED_STAYS:
    raise AssertionError("Structured tensor stay dimension mismatch.")

if structured.shape[1] != 24:
    raise AssertionError("Structured tensor must contain 24 hourly positions.")

if not np.isfinite(structured).all():
    raise AssertionError("Structured tensor contains non-finite values.")

print("✅ Structured tensor aligned and finite")


# ============================================================
# 10. DEMOGRAPHICS CHECK
# ============================================================

print("\n" + "-" * 80)
print("7. VERIFYING A4 DEMOGRAPHICS")
print("-" * 80)

demographics = np.load(A4_DEMOGRAPHICS, mmap_mode="r")
print("Demographics shape:", demographics.shape)

if demographics.shape != (EXPECTED_STAYS, 4):
    raise AssertionError(
        f"Unexpected demographics shape: {demographics.shape}"
    )

if not np.isfinite(demographics).all():
    raise AssertionError("Demographics contain non-finite values.")

print("✅ Demographics aligned and finite")


# ============================================================
# 11. TEXT EMBEDDINGS CHECK
# ============================================================

print("\n" + "-" * 80)
print("8. VERIFYING A2 TEXT EMBEDDINGS")
print("-" * 80)

emb = pd.read_parquet(A2_EMBEDDINGS)

if "stay_id" not in emb.columns:
    raise AssertionError("A2 embeddings missing stay_id.")

emb["stay_id"] = pd.to_numeric(
    emb["stay_id"], errors="raise"
).astype("int64")

if emb["stay_id"].duplicated().any():
    raise AssertionError(
        "Duplicate stay_id values in A2 embeddings."
    )

# A2 is intentionally built from the full frozen A0 cohort.
# Therefore it can legitimately contain text-bearing stays that do not
# belong to the final 24-hour primary cohort (e.g. stays excluded by the
# A2.1 target-eligibility reconciliation). Those rows must not enlarge A5.
a0 = pd.read_csv(
    A0_COHORT,
    usecols=["stay_id"],
)

a0["stay_id"] = pd.to_numeric(
    a0["stay_id"], errors="raise"
).astype("int64")

if a0["stay_id"].duplicated().any():
    raise AssertionError(
        "Duplicate stay_id values in A0 cohort."
    )

a0_stays = set(a0["stay_id"])
primary_stays = set(split_df["stay_id"])
text_stays = set(emb["stay_id"])

# First prove A2 did not contain IDs outside the frozen A0 cohort.
unknown_text_stays_vs_a0 = text_stays - a0_stays

if unknown_text_stays_vs_a0:
    sample = sorted(unknown_text_stays_vs_a0)[:20]
    raise AssertionError(
        "Text embeddings contain stay_ids outside frozen A0 cohort: "
        f"{sample}"
    )

# A2 may contain valid A0 stays excluded from the 24-hour primary cohort.
text_stays_outside_primary = text_stays - primary_stays
text_stays_primary = text_stays & primary_stays

print(
    f"A2 text stays (full A0 scope) : {len(text_stays):,}"
)
print(
    f"A2 text stays in primary 24h  : {len(text_stays_primary):,}"
)
print(
    f"A2 text stays outside primary : {len(text_stays_outside_primary):,}"
)

if text_stays_outside_primary:
    print(
        "ℹ️ Extra A2 text rows are retained in A2 but ignored by A5 "
        "because A5 is restricted to the frozen 24h primary cohort."
    )

print("✅ A2 text IDs are valid within frozen A0 cohort")


# ============================================================
# 12. TEXT METADATA CHECK
# ============================================================

print("\n" + "-" * 80)
print("9. VERIFYING A2 TEXT METADATA")
print("-" * 80)

a2_meta = pd.read_csv(A2_METADATA)

if "stay_id" in a2_meta.columns:
    a2_meta["stay_id"] = pd.to_numeric(
        a2_meta["stay_id"], errors="raise"
    ).astype("int64")

    # Metadata may be note-level and therefore legitimately contain
    # repeated stay_id values. Do NOT require stay_id uniqueness here.
    unknown_meta_stays = set(a2_meta["stay_id"]) - a0_stays

    if unknown_meta_stays:
        sample = sorted(unknown_meta_stays)[:20]
        raise AssertionError(
            "A2 metadata contains stay_ids outside frozen A0 cohort: "
            f"{sample}"
        )

    metadata_primary = set(a2_meta["stay_id"]) & primary_stays

    print(
        f"A2 metadata rows             : {len(a2_meta):,}"
    )
    print(
        f"Metadata stays in primary    : {len(metadata_primary):,}"
    )

print("✅ A2 metadata IDs are valid within frozen A0 cohort")


# ============================================================
# 13. BUILD A5 ALIGNMENT INDEX
# ============================================================

print("\n" + "-" * 80)
print("10. BUILDING A5 ALIGNMENT INDEX")
print("-" * 80)

alignment = a4_index.copy()
alignment["text_available"] = (
    alignment["stay_id"].isin(text_stays_primary).astype("int8")
)
alignment = alignment.sort_values("stay_id").reset_index(drop=True)

if len(alignment) != EXPECTED_STAYS:
    raise AssertionError("A5 alignment row count mismatch.")

if alignment["stay_id"].duplicated().any():
    raise AssertionError("Duplicate stay_id values in A5 alignment.")

if not alignment["split_24h"].isin(["train", "val", "test"]).all():
    raise AssertionError("Invalid split labels in A5 alignment.")

alignment.to_csv(A5_ALIGNMENT_INDEX, index=False)
print(f"✅ Saved: {A5_ALIGNMENT_INDEX}")


# ============================================================
# 14. SAVE AUDIT JSON
# ============================================================

text_available = int(alignment["text_available"].sum())
text_missing = int((alignment["text_available"] == 0).sum())

# Re-check final alignment before committing the audit record.
assert sorted_reset(alignment, SPLIT_COLUMNS).equals(
    sorted_reset(split_df, SPLIT_COLUMNS)
)
assert sorted_reset(alignment, IDENTITY_COLUMNS).equals(
    sorted_reset(targets, IDENTITY_COLUMNS)
)

# The target definition file is required as an input/provenance artifact.
with open(TARGET_DEFINITION, "r", encoding="utf-8") as f:
    target_definition = json.load(f)

if not isinstance(target_definition, dict):
    raise AssertionError("Target definition JSON is not a JSON object.")

audit = {
    "stage": "A5",
    "status": "PASSED",
    "split_source": V2_SPLIT,
    "split_is_frozen": True,
    "n_stays": EXPECTED_STAYS,
    "train_stays": EXPECTED_TRAIN,
    "val_stays": EXPECTED_VAL,
    "test_stays": EXPECTED_TEST,
    "structured_shape": list(structured.shape),
    "demographics_shape": list(demographics.shape),
    "text_stays_available_full_a0": len(text_stays),
    "text_stays_outside_primary": len(text_stays_outside_primary),
    "text_stays_available_primary": len(text_stays_primary),
    "text_available_stays": text_available,
    "text_missing_stays": text_missing,
    "patient_overlap_train_val": len(train_val_overlap),
    "patient_overlap_train_test": len(train_test_overlap),
    "patient_overlap_val_test": len(val_test_overlap),
    "a4_split_match": True,
    "target_identity_match": True,
    "target_definition_loaded": True,
    "structured_finite": True,
    "demographics_finite": True,
    "duplicate_stay_ids": 0,
    "legacy_train_test_split_used": False,
}

with open(A5_AUDIT_JSON, "w", encoding="utf-8") as f:
    json.dump(audit, f, indent=2)


# ============================================================
# 15. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("✅ PIPELINE A V2 — A5 ALIGNMENT AUDIT PASSED")
print("=" * 80)
print(f"Total stays       : {EXPECTED_STAYS:,}")
print(f"Train stays       : {EXPECTED_TRAIN:,}")
print(f"Validation stays  : {EXPECTED_VAL:,}")
print(f"Test stays        : {EXPECTED_TEST:,}")
print(f"Structured tensor : {structured.shape}")
print(f"Demographics      : {demographics.shape}")
print(f"Text available    : {text_available:,} primary-cohort stays")
print("Patient leakage   : PASSED")
print("Split consistency : PASSED")
print("Target alignment  : PASSED")
print("Identity checks   : PASSED")
print(f"Alignment saved   : {A5_ALIGNMENT_INDEX}")
print(f"Audit saved       : {A5_AUDIT_JSON}")
print("=" * 80)
