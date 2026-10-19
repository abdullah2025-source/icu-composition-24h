#!/usr/bin/env python3

"""
Pipeline A V2
Stage A2 — Leakage-Safe Clinical Radiology Text Embeddings

Purpose
-------
Build leakage-safe per-ICU-stay clinical text embeddings for the fixed
24-hour benchmark.

LOCKED TEXT POLICY
------------------
Primary benchmark text source:
    radiology.csv.gz ONLY

Discharge summaries are deliberately excluded from the benchmark text pool.
They must NOT be loaded, filtered by time, encoded, or pooled here.

Information-availability rule:
    storetime when present, otherwise charttime

Benchmark window:
    [intime, intime + 24h)

Multi-ICU-stay note assignment:
    A radiology note may match more than one ICU stay in the same admission.
    After the leakage-safe time test, assign such a note deterministically to
    the matching stay with the latest ICU intime. Ties are resolved by the
    larger stay_id.

Output
------
clinical_note_embeddings.parquet
    stay_id + 768-dimensional pooled embedding

Additional QA
--------------
clinical_note_embedding_metadata.csv
a2_note_timing_audit.csv
a2_note_audit.csv
a2_metadata.json

Pooling
-------
The supplied pooling method is preserved as deterministic norm-softmax
pooling across note embeddings within a stay. It is NOT learned attention.

Safety
------
- V1 output is never modified.
- Raw MIMIC-IV-Note files are read-only.
- A2 writes to the canonical V2 location only.
- Output files are written atomically after hard checks.
"""

from pathlib import Path
import gc
import json
import os
import re
import tempfile

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel


# ============================================================
# 0. PATHS
# ============================================================

PROJECT_ROOT = Path("/home/abdullah2025/fyp_project")

MIMIC_NOTE_DIR = (
    PROJECT_ROOT
    / "data"
    / "mimic_iv"
    / "mimic-iv-note-deidentified-free-text-clinical-notes-2.2"
    / "note"
)

V1_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "pipeline_a"
V2_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "pipeline_a_v2"

A0_COHORT_PATH = V2_OUTPUT_DIR / "cohort_master.csv"

# IMPORTANT: radiology ONLY for the fixed benchmark.
RADIOLOGY_PATH = MIMIC_NOTE_DIR / "radiology.csv.gz"

OUT_PATH = V2_OUTPUT_DIR / "clinical_note_embeddings.parquet"
METADATA_CSV_PATH = V2_OUTPUT_DIR / "clinical_note_embedding_metadata.csv"
AUDIT_PATH = V2_OUTPUT_DIR / "a2_note_audit.csv"
TIMING_AUDIT_PATH = V2_OUTPUT_DIR / "a2_note_timing_audit.csv"
METADATA_PATH = V2_OUTPUT_DIR / "a2_metadata.json"

V2_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if V1_OUTPUT_DIR.resolve() == V2_OUTPUT_DIR.resolve():
    raise RuntimeError(
        "SAFETY ERROR: V1 and V2 output directories are identical."
    )


# ============================================================
# 1. CONFIGURATION
# ============================================================

MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
EXPECTED_HIDDEN_SIZE = 768
MAX_LENGTH = 512
INITIAL_BATCH_SIZE = 64
LANDMARK_HOURS = 24
REQUIRE_GPU = True
EXPECTED_A0_ROWS = 93224

# Explicit benchmark note policy.
ALLOWED_NOTE_SOURCE = "radiology"
DISCHARGE_INCLUDED = False


# ============================================================
# 2. ATOMIC OUTPUT HELPERS
# ============================================================

def atomic_write_parquet(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        suffix=".parquet",
        prefix=output_path.stem + ".",
        dir=output_path.parent,
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)

    try:
        df.to_parquet(tmp_path, index=False, engine="pyarrow")
        os.replace(tmp_path, output_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def atomic_write_csv(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        suffix=".csv",
        prefix=output_path.stem + ".",
        dir=output_path.parent,
        delete=False,
        mode="w",
        encoding="utf-8",
        newline="",
    ) as tmp:
        tmp_path = Path(tmp.name)

    try:
        df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, output_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def atomic_write_json(obj: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        suffix=".json",
        prefix=output_path.stem + ".",
        dir=output_path.parent,
        delete=False,
        mode="w",
        encoding="utf-8",
    ) as tmp:
        tmp_path = Path(tmp.name)
        json.dump(obj, tmp, indent=2, default=str)

    try:
        os.replace(tmp_path, output_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


# ============================================================
# 3. TEXT / MODEL HELPERS
# ============================================================

def clean_note(text) -> str:
    if not isinstance(text, str):
        return ""

    text = text.lower()

    # Remove MIMIC de-identification markers.
    text = re.sub(r"\[\*\*.*?\*\*\]", "", text)

    # Normalize whitespace while preserving clinical content.
    text = re.sub(r"\s+", " ", text).strip()

    return text


def load_radiology_table(path: Path) -> pd.DataFrame:
    """Load radiology.csv.gz only, with explicit schema validation."""
    if not path.exists():
        raise FileNotFoundError(
            f"Required radiology table not found:\n{path}"
        )

    header = pd.read_csv(
        path,
        compression="gzip",
        nrows=0,
    )

    required = {
        "note_id",
        "subject_id",
        "hadm_id",
        "note_type",
        "charttime",
        "storetime",
        "text",
    }

    missing = sorted(required - set(header.columns))
    if missing:
        raise AssertionError(
            f"radiology.csv.gz missing required columns: {missing}"
        )

    df = pd.read_csv(
        path,
        compression="gzip",
        usecols=[
            "note_id",
            "subject_id",
            "hadm_id",
            "note_type",
            "charttime",
            "storetime",
            "text",
        ],
        low_memory=False,
    )

    df["note_source"] = ALLOWED_NOTE_SOURCE
    return df


def encode_texts(
    texts,
    tokenizer,
    model,
    device,
    initial_batch_size: int,
) -> np.ndarray:
    """Encode texts with automatic batch-size reduction on GPU OOM."""
    if not texts:
        raise RuntimeError("No text rows were supplied for encoding.")

    batch_size = int(initial_batch_size)

    while batch_size >= 1:
        embeddings = []
        failed = False

        try:
            total_batches = (
                len(texts) + batch_size - 1
            ) // batch_size

            with torch.inference_mode():
                for start in tqdm(
                    range(0, len(texts), batch_size),
                    total=total_batches,
                    desc=f"Encoding notes (batch={batch_size})",
                ):
                    batch = texts[start : start + batch_size]

                    encoded = tokenizer(
                        batch,
                        padding=True,
                        truncation=True,
                        max_length=MAX_LENGTH,
                        return_tensors="pt",
                    )

                    encoded = {
                        key: value.to(device)
                        for key, value in encoded.items()
                    }

                    outputs = model(**encoded)

                    cls_embeddings = (
                        outputs.last_hidden_state[:, 0, :]
                        .detach()
                        .cpu()
                        .float()
                        .numpy()
                    )

                    embeddings.append(cls_embeddings)

                    del encoded
                    del outputs
                    del cls_embeddings

            result = np.vstack(embeddings).astype(
                np.float32,
                copy=False,
            )

            return result

        except RuntimeError as exc:
            is_oom = (
                "out of memory" in str(exc).lower()
                and torch.cuda.is_available()
            )

            if not is_oom:
                raise

            failed = True

            del embeddings
            gc.collect()
            torch.cuda.empty_cache()

            if batch_size == 1:
                raise RuntimeError(
                    "GPU out-of-memory even at batch_size=1."
                ) from exc

            batch_size = max(1, batch_size // 2)

            print(
                "\n⚠️ CUDA/ROCm OOM; reducing batch size to "
                f"{batch_size} and retrying from the beginning."
            )

        finally:
            if failed:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    raise RuntimeError("Encoding failed for an unknown reason.")


def norm_softmax_pooling(embeddings: np.ndarray) -> np.ndarray:
    """Deterministic norm-softmax pooling; not learned attention."""
    if embeddings.ndim != 2:
        raise AssertionError(
            f"Expected [n_notes, hidden_size], got {embeddings.shape}"
        )

    if embeddings.shape[1] != EXPECTED_HIDDEN_SIZE:
        raise AssertionError(
            "Unexpected embedding width before pooling: "
            f"{embeddings.shape[1]}"
        )

    if not np.isfinite(embeddings).all():
        raise AssertionError(
            "Non-finite note embedding detected before pooling."
        )

    scores = np.linalg.norm(
        embeddings,
        axis=1,
    ).astype(np.float64)

    scores -= np.max(scores)

    weights = np.exp(scores)
    weight_sum = float(weights.sum())

    if not np.isfinite(weight_sum) or weight_sum <= 0:
        raise AssertionError("Invalid pooling weights.")

    weights /= weight_sum

    pooled = (
        weights[:, None]
        * embeddings
    ).sum(axis=0)

    if not np.isfinite(pooled).all():
        raise AssertionError(
            "Non-finite pooled embedding produced."
        )

    return pooled.astype(np.float32)


# ============================================================
# 4. STARTUP / GPU CHECK
# ============================================================

print("=" * 80)
print("PIPELINE A V2 — A2 CLINICAL RADIOLOGY TEXT EMBEDDINGS")
print("=" * 80)
print(f"V1 output directory : {V1_OUTPUT_DIR}")
print(f"V2 output directory : {V2_OUTPUT_DIR}")
print(f"MIMIC-IV-Note dir   : {MIMIC_NOTE_DIR}")
print(f"Text source         : {ALLOWED_NOTE_SOURCE} ONLY")
print(f"Discharge included  : {DISCHARGE_INCLUDED}")
print(f"Model               : {MODEL_NAME}")
print(f"Landmark            : first {LANDMARK_HOURS} ICU hours")
print("Availability rule   : storetime, charttime fallback")
print(f"Max token length    : {MAX_LENGTH}")
print(
    "Multi-stay rule     : latest qualifying ICU intime, then max stay_id"
)

if REQUIRE_GPU and not torch.cuda.is_available():
    raise RuntimeError(
        "A2 requires a GPU allocation. "
        "torch.cuda.is_available() is False. "
        "Do NOT run A2 in a CPU-only Slurm allocation."
    )

device = torch.device("cuda")

print(f"Device              : {device}")
print(f"PyTorch              : {torch.__version__}")
print(f"HIP version          : {getattr(torch.version, 'hip', None)}")

print(
    f"GPU                  : {torch.cuda.get_device_name(0)}"
)
print(
    f"GPU count            : {torch.cuda.device_count()}"
)


# ============================================================
# 5. LOAD V2 A0 COHORT
# ============================================================

print("\n" + "-" * 80)
print("1. LOAD V2 A0 COHORT")
print("-" * 80)

if not A0_COHORT_PATH.exists():
    raise FileNotFoundError(
        f"A0 cohort not found:\n{A0_COHORT_PATH}"
    )

cohort = pd.read_csv(A0_COHORT_PATH)

required_cohort_cols = [
    "subject_id",
    "hadm_id",
    "stay_id",
    "intime",
    "outtime",
]

missing = [
    c for c in required_cohort_cols
    if c not in cohort.columns
]

if missing:
    raise AssertionError(
        f"A0 cohort missing columns: {missing}"
    )

cohort["subject_id"] = pd.to_numeric(
    cohort["subject_id"],
    errors="raise",
).astype("int64")

cohort["hadm_id"] = pd.to_numeric(
    cohort["hadm_id"],
    errors="raise",
).astype("int64")

cohort["stay_id"] = pd.to_numeric(
    cohort["stay_id"],
    errors="raise",
).astype("int64")

cohort["intime"] = pd.to_datetime(
    cohort["intime"],
    errors="coerce",
)

cohort["outtime"] = pd.to_datetime(
    cohort["outtime"],
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

if len(cohort) != EXPECTED_A0_ROWS:
    raise AssertionError(
        f"Unexpected A0 cohort size: expected {EXPECTED_A0_ROWS:,}, "
        f"found {len(cohort):,}."
    )

cohort["landmark_end"] = (
    cohort["intime"]
    + pd.Timedelta(hours=LANDMARK_HOURS)
)

stay_window = cohort[
    [
        "subject_id",
        "hadm_id",
        "stay_id",
        "intime",
        "outtime",
        "landmark_end",
    ]
].copy()

print(
    f"A0 cohort rows      : {len(cohort):,}"
)
print(
    f"Unique ICU stays    : {cohort['stay_id'].nunique():,}"
)
print(
    f"Unique subjects     : {cohort['subject_id'].nunique():,}"
)


# ============================================================
# 6. LOAD RADIOLOGY ONLY
# ============================================================

print("\n" + "-" * 80)
print("2. LOAD MIMIC-IV-NOTE TABLE")
print("-" * 80)

radiology = load_radiology_table(RADIOLOGY_PATH)

raw_radiology_count = len(radiology)

print(
    f"radiology notes    : {raw_radiology_count:,}"
)
print(
    "discharge notes    : 0 (deliberately excluded from benchmark)"
)

# Hard policy guard: this code path contains radiology only.
if set(radiology["note_source"].dropna().unique()) != {"radiology"}:
    raise AssertionError(
        "Text-source policy violation: non-radiology source present."
    )


# ============================================================
# 7. NOTE ID / TIMESTAMP QC
# ============================================================

print("\n" + "-" * 80)
print("3. NOTE TIMESTAMP / ID QC")
print("-" * 80)

radiology["subject_id"] = pd.to_numeric(
    radiology["subject_id"],
    errors="coerce",
)

radiology["hadm_id"] = pd.to_numeric(
    radiology["hadm_id"],
    errors="coerce",
)

radiology["charttime"] = pd.to_datetime(
    radiology["charttime"],
    errors="coerce",
)

radiology["storetime"] = pd.to_datetime(
    radiology["storetime"],
    errors="coerce",
)

usable_id_rows = radiology[
    [
        "subject_id",
        "hadm_id",
        "note_id",
        "charttime",
    ]
].notna().all(axis=1)

before_usable_filter = len(radiology)
rows_removed_unusable_ids = int((~usable_id_rows).sum())

radiology = radiology[usable_id_rows].copy()

radiology["subject_id"] = radiology["subject_id"].astype("int64")
radiology["hadm_id"] = radiology["hadm_id"].astype("int64")
radiology["note_id"] = radiology["note_id"].astype(str)

if radiology["note_id"].duplicated().any():
    dup_count = int(radiology["note_id"].duplicated().sum())
    raise AssertionError(
        "radiology.csv.gz contains duplicate note_id values. "
        f"Duplicate rows detected: {dup_count:,}."
    )

note_id_duplicates_removed = 0

print(
    f"Rows loaded                  : {before_usable_filter:,}"
)
print(
    f"Rows with usable IDs/time    : {len(radiology):,}"
)
print(
    f"Rows removed for unusable ID/time: {rows_removed_unusable_ids:,}"
)
print(
    f"Duplicate note_id rows removed : {note_id_duplicates_removed:,}"
)

if radiology.empty:
    raise RuntimeError(
        "No radiology notes remain after ID/timestamp QC."
    )


# ============================================================
# 8. LINK TO ICU STAYS + LEAKAGE-SAFE 24H WINDOW
# ============================================================

print("\n" + "-" * 80)
print("4. LINK RADIOLOGY NOTES TO ICU STAYS")
print("-" * 80)

notes = radiology.merge(
    stay_window,
    on=["subject_id", "hadm_id"],
    how="inner",
    validate="many_to_many",
)

del radiology

gc.collect()

matched_note_rows_before_assignment = len(notes)
matched_stays_before_assignment = notes["stay_id"].nunique()

print(
    f"Matched note rows before time filter : "
    f"{matched_note_rows_before_assignment:,}"
)
print(
    f"Matched stays before time filter     : "
    f"{matched_stays_before_assignment:,}"
)

# Information availability: storetime preferred; charttime fallback.
notes["availability_time"] = notes["storetime"].fillna(
    notes["charttime"]
)

notes["availability_source"] = np.where(
    notes["storetime"].notna(),
    "storetime",
    "charttime",
)

notes["hours_from_intime"] = (
    notes["availability_time"]
    - notes["intime"]
).dt.total_seconds() / 3600.0

notes_before_intime = int(
    (
        notes["availability_time"]
        < notes["intime"]
    ).sum()
)

notes_after_or_at_landmark = int(
    (
        notes["availability_time"]
        >= notes["landmark_end"]
    ).sum()
)

notes_inside_candidate_window = notes[
    (
        notes["availability_time"]
        >= notes["intime"]
    )
    & (
        notes["availability_time"]
        < notes["landmark_end"]
    )
].copy()

candidate_rows_before_multistay_resolution = len(
    notes_inside_candidate_window
)

print(
    f"Notes before ICU intake        : {notes_before_intime:,}"
)
print(
    f"Notes at/after 24h landmark    : {notes_after_or_at_landmark:,}"
)
print(
    "Notes in leakage-safe [intime, intime+24h) window: "
    f"{candidate_rows_before_multistay_resolution:,}"
)

if notes_inside_candidate_window.empty:
    raise RuntimeError(
        "No radiology notes remain inside the leakage-safe 24-hour window."
    )


# ------------------------------------------------------------
# Deterministic multi-stay assignment
# ------------------------------------------------------------
# A note associated with the same hospital admission may match more than one
# ICU stay. We deliberately do not let merge row ordering decide its owner.
# For every qualifying note, assign it to the matching stay with the latest
# ICU intime. If intimes tie, use the larger stay_id.
notes_inside_candidate_window = (
    notes_inside_candidate_window
    .sort_values(
        [
            "note_id",
            "intime",
            "stay_id",
        ],
        ascending=[True, True, True],
        kind="mergesort",
    )
)

multistay_candidate_notes = int(
    notes_inside_candidate_window
    .groupby("note_id", sort=False)["stay_id"]
    .nunique()
    .gt(1)
    .sum()
)

notes_inside_candidate_window = (
    notes_inside_candidate_window
    .drop_duplicates(
        subset=["note_id"],
        keep="last",
    )
    .reset_index(drop=True)
)

assigned_unique_notes = len(notes_inside_candidate_window)
assigned_unique_stays = (
    notes_inside_candidate_window["stay_id"].nunique()
)

print(
    f"Notes matching >1 ICU stay      : {multistay_candidate_notes:,}"
)
print(
    f"Unique notes after stay assignment: {assigned_unique_notes:,}"
)
print(
    f"Unique stays with eligible notes : {assigned_unique_stays:,}"
)

# Every final note must satisfy its selected stay's leakage-safe window.
if not (
    notes_inside_candidate_window["availability_time"]
    >= notes_inside_candidate_window["intime"]
).all():
    raise AssertionError(
        "Final note assignment contains availability_time before intime."
    )

if not (
    notes_inside_candidate_window["availability_time"]
    < notes_inside_candidate_window["landmark_end"]
).all():
    raise AssertionError(
        "Final note assignment contains availability_time at/after landmark."
    )

if notes_inside_candidate_window["note_id"].duplicated().any():
    raise AssertionError(
        "Final note set still contains duplicate note_id values."
    )


# ============================================================
# 9. TEXT CLEANING / DEDUPLICATION
# ============================================================

print("\n" + "-" * 80)
print("5. CLEAN NOTE TEXT")
print("-" * 80)

notes = notes_inside_candidate_window

del notes_inside_candidate_window

gc.collect()

notes["text_clean"] = notes["text"].map(clean_note)

empty_text_rows = int(
    (notes["text_clean"].str.len() == 0).sum()
)
trivial_text_rows = int(
    (notes["text_clean"].str.len() <= 20).sum()
)

notes = notes[
    notes["text_clean"].str.len() > 20
].copy()

print(
    f"Empty cleaned notes removed   : {empty_text_rows:,}"
)
print(
    f"<=20-character notes removed  : {trivial_text_rows:,}"
)
print(
    f"Usable radiology note rows    : {len(notes):,}"
)

if notes.empty:
    raise RuntimeError(
        "No usable radiology note text remains after cleaning."
    )

# No second note_id deduplication should normally be necessary. A hard check
# keeps provenance deterministic rather than silently dropping rows.
if notes["note_id"].duplicated().any():
    raise AssertionError(
        "Duplicate note_id appeared after text cleaning."
    )


# Identical text from different radiology note IDs/timestamps is retained.
# Timing and note provenance remain meaningful.


# ============================================================
# 10. NOTE TIMING / TYPE AUDITS
# ============================================================

print("\n" + "-" * 80)
print("6. NOTE TIMING AUDIT")
print("-" * 80)

# Track whether storetime was used on the final benchmark note rows.
storetime_count = int(
    (notes["availability_source"] == "storetime").sum()
)
charttime_fallback_count = int(
    (notes["availability_source"] == "charttime").sum()
)

store_before_chart_count = int(
    (
        notes["storetime"].notna()
        & notes["charttime"].notna()
        & (notes["storetime"] < notes["charttime"])
    ).sum()
)

note_type_audit = (
    notes.groupby(
        ["note_source", "note_type"],
        as_index=False,
        dropna=False,
    )
    .agg(
        notes=("note_id", "nunique"),
        stays=("stay_id", "nunique"),
        median_hours_from_intime=("hours_from_intime", "median"),
        min_hours_from_intime=("hours_from_intime", "min"),
        max_hours_from_intime=("hours_from_intime", "max"),
    )
)

# Explicit policy proof: there must be no discharge source at this stage.
if not note_type_audit.empty:
    if not (
        note_type_audit["note_source"].eq("radiology")
    ).all():
        raise AssertionError(
            "A2 note audit contains a non-radiology source."
        )

atomic_write_csv(note_type_audit, AUDIT_PATH)

# Correct timing counts: these are measured before text-cleaning removal.
timing_rows = [
    {
        "metric": "raw_radiology_notes_loaded",
        "value": int(raw_radiology_count),
    },
    {
        "metric": "radiology_rows_with_usable_ids_and_charttime",
        "value": int(before_usable_filter - rows_removed_unusable_ids),
    },
    {
        "metric": "radiology_rows_removed_for_unusable_id_or_charttime",
        "value": int(rows_removed_unusable_ids),
    },
    {
        "metric": "duplicate_note_id_rows_removed",
        "value": 0,
    },
    {
        "metric": "matched_note_rows_before_time_filter",
        "value": int(matched_note_rows_before_assignment),
    },
    {
        "metric": "matched_stays_before_time_filter",
        "value": int(matched_stays_before_assignment),
    },
    {
        "metric": "notes_available_before_selected_icu_intime",
        "value": int(notes_before_intime),
    },
    {
        "metric": "notes_at_or_after_24h_landmark",
        "value": int(notes_after_or_at_landmark),
    },
    {
        "metric": "candidate_notes_in_24h_window_before_multistay_resolution",
        "value": int(candidate_rows_before_multistay_resolution),
    },
    {
        "metric": "notes_matching_multiple_icu_stays",
        "value": int(multistay_candidate_notes),
    },
    {
        "metric": "unique_notes_after_deterministic_stay_assignment",
        "value": int(assigned_unique_notes),
    },
    {
        "metric": "stays_with_eligible_radiology_notes",
        "value": int(assigned_unique_stays),
    },
    {
        "metric": "empty_cleaned_notes_removed",
        "value": int(empty_text_rows),
    },
    {
        "metric": "short_notes_20_or_fewer_characters_removed",
        "value": int(trivial_text_rows),
    },
    {
        "metric": "final_usable_radiology_note_rows_before_encoding",
        "value": int(len(notes)),
    },
    {
        "metric": "notes_using_storetime_as_availability",
        "value": int(storetime_count),
    },
    {
        "metric": "notes_using_charttime_fallback_as_availability",
        "value": int(charttime_fallback_count),
    },
    {
        "metric": "notes_with_storetime_earlier_than_charttime",
        "value": int(store_before_chart_count),
    },
    {
        "metric": "discharge_notes_in_benchmark_text_pool",
        "value": 0,
    },
]

timing_audit = pd.DataFrame(timing_rows)
atomic_write_csv(timing_audit, TIMING_AUDIT_PATH)

print(
    f"Saved timing audit : {TIMING_AUDIT_PATH}"
)
print(
    f"Saved note audit   : {AUDIT_PATH}"
)


# ============================================================
# 11. LOAD BIOCLINICALBERT
# ============================================================

print("\n" + "-" * 80)
print("7. LOAD BIOCLINICALBERT")
print("-" * 80)

print(f"Loading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print(f"Loading model    : {MODEL_NAME}")
bert_model = AutoModel.from_pretrained(MODEL_NAME)
bert_model = bert_model.to(device)
bert_model.eval()

hidden_size = int(
    bert_model.config.hidden_size
)

if hidden_size != EXPECTED_HIDDEN_SIZE:
    raise AssertionError(
        "Unexpected BioClinicalBERT hidden size: "
        f"expected {EXPECTED_HIDDEN_SIZE}, got {hidden_size}."
    )

max_position_embeddings = int(
    bert_model.config.max_position_embeddings
)

if MAX_LENGTH > max_position_embeddings:
    raise AssertionError(
        f"MAX_LENGTH={MAX_LENGTH} exceeds model max_position_embeddings="
        f"{max_position_embeddings}."
    )

print(
    f"Hidden size        : {hidden_size}"
)
print(
    f"Model max positions: {max_position_embeddings}"
)
print("✅ BioClinicalBERT loaded")


# ============================================================
# 12. ENCODE NOTE TEXT
# ============================================================

print("\n" + "-" * 80)
print("8. ENCODE RADIOLOGY NOTES")
print("-" * 80)

texts = notes["text_clean"].tolist()

note_embeddings = encode_texts(
    texts=texts,
    tokenizer=tokenizer,
    model=bert_model,
    device=device,
    initial_batch_size=INITIAL_BATCH_SIZE,
)

expected_embedding_shape = (
    len(notes),
    EXPECTED_HIDDEN_SIZE,
)

if note_embeddings.shape != expected_embedding_shape:
    raise AssertionError(
        f"Unexpected note embedding shape: {note_embeddings.shape}; "
        f"expected {expected_embedding_shape}."
    )

if not np.isfinite(note_embeddings).all():
    raise AssertionError(
        "Non-finite note embeddings detected after encoding."
    )

print(
    f"Note embeddings shape : {note_embeddings.shape}"
)
print("✅ Individual radiology note embeddings complete")


# ============================================================
# 13. PER-STAY DETERMINISTIC NORM-SOFTMAX POOLING
# ============================================================

print("\n" + "-" * 80)
print("9. PER-STAY DETERMINISTIC NORM-SOFTMAX POOLING")
print("-" * 80)

order = np.argsort(
    notes["stay_id"].to_numpy(dtype=np.int64),
    kind="mergesort",
)

notes_sorted = (
    notes.iloc[order]
    .reset_index(drop=True)
)

embeddings_sorted = note_embeddings[order]

stay_embeddings = []
stay_rows = []

for stay_id, grp in tqdm(
    notes_sorted.groupby("stay_id", sort=False),
    desc="Pooling stays",
):
    idx = grp.index.to_numpy(dtype=np.int64)

    pooled = norm_softmax_pooling(
        embeddings_sorted[idx]
    )

    if pooled.shape != (EXPECTED_HIDDEN_SIZE,):
        raise AssertionError(
            f"Invalid pooled embedding shape for stay {stay_id}: "
            f"{pooled.shape}"
        )

    stay_embeddings.append(
        pooled.tolist()
    )

    stay_rows.append(
        {
            "stay_id": int(stay_id),
            "note_count": int(len(idx)),
            "first_note_availability": grp["availability_time"].min(),
            "last_note_availability": grp["availability_time"].max(),
            "radiology_note_count": int(len(idx)),
            "discharge_note_count": 0,
            "note_source": "radiology",
        }
    )

print(
    f"Stays with text    : {len(stay_rows):,}"
)

if not stay_rows:
    raise RuntimeError(
        "No stays received pooled radiology text embeddings."
    )


# ============================================================
# 14. BUILD OUTPUTS
# ============================================================

print("\n" + "-" * 80)
print("10. BUILD AND SAVE A2 OUTPUTS")
print("-" * 80)

main_output = pd.DataFrame(
    {
        "stay_id": [
            row["stay_id"]
            for row in stay_rows
        ],
        "text_embedding": stay_embeddings,
    }
)

main_output["stay_id"] = main_output["stay_id"].astype("int64")

if main_output["stay_id"].duplicated().any():
    raise AssertionError(
        "Duplicate stay_id in final A2 output."
    )

vector_lengths = np.array(
    [
        len(v)
        for v in main_output["text_embedding"]
    ],
    dtype=np.int32,
)

if not np.all(
    vector_lengths == EXPECTED_HIDDEN_SIZE
):
    raise AssertionError(
        "One or more final embeddings are not 768-dimensional."
    )

for vec in main_output["text_embedding"]:
    arr = np.asarray(
        vec,
        dtype=np.float32,
    )

    if not np.isfinite(arr).all():
        raise AssertionError(
            "Non-finite final text embedding detected."
        )

metadata_df = pd.DataFrame(stay_rows)

metadata_df["stay_id"] = metadata_df[
    "stay_id"
].astype("int64")

for col in [
    "note_count",
    "radiology_note_count",
    "discharge_note_count",
]:
    metadata_df[col] = metadata_df[col].astype("int32")

# Strong safety check: every metadata row must be radiology-only.
if not metadata_df["note_source"].eq("radiology").all():
    raise AssertionError(
        "Final metadata contains a non-radiology note source."
    )

if not metadata_df["discharge_note_count"].eq(0).all():
    raise AssertionError(
        "Final metadata contains a nonzero discharge_note_count."
    )

# Main output is intentionally only stay_id + embedding, preserving the
# downstream schema used by the supplied pipeline.
atomic_write_parquet(
    main_output,
    OUT_PATH,
)

atomic_write_csv(
    metadata_df,
    METADATA_CSV_PATH,
)

meta = {
    "stage": "A2",
    "pipeline": "Pipeline A V2",
    "benchmark_text_policy": "radiology_only",
    "note_sources": ["radiology"],
    "discharge_notes_in_benchmark": False,
    "detail_tables_embedded": False,
    "mimic_note_radiology_table": str(RADIOLOGY_PATH),
    "model_name": MODEL_NAME,
    "device": str(device),
    "torch_version": torch.__version__,
    "hip_version": getattr(torch.version, "hip", None),
    "hidden_size": EXPECTED_HIDDEN_SIZE,
    "max_length": MAX_LENGTH,
    "landmark_hours": LANDMARK_HOURS,
    "benchmark_interval": "[intime, intime + 24h)",
    "availability_time_rule": "storetime when present, otherwise charttime",
    "multistay_note_assignment_rule": (
        "Among stays whose 24h window contains the note availability time, "
        "assign to latest ICU intime; ties resolved by largest stay_id."
    ),
    "pooling_method": "deterministic_norm_softmax_pooling",
    "pooling_is_learned_attention": False,
    "raw_radiology_notes_loaded": int(raw_radiology_count),
    "matched_note_rows_before_time_filter": int(
        matched_note_rows_before_assignment
    ),
    "candidate_notes_before_multistay_resolution": int(
        candidate_rows_before_multistay_resolution
    ),
    "notes_matching_multiple_icu_stays": int(
        multistay_candidate_notes
    ),
    "benchmark_notes_before_text_cleaning": int(
        assigned_unique_notes
    ),
    "benchmark_notes_after_text_cleaning": int(
        len(notes)
    ),
    "stays_with_text_embeddings": int(
        len(main_output)
    ),
    "a0_stays": int(len(cohort)),
    "output_path": str(OUT_PATH),
    "metadata_path": str(METADATA_CSV_PATH),
    "timing_audit_path": str(TIMING_AUDIT_PATH),
    "note_audit_path": str(AUDIT_PATH),
    "v1_output_modified": False,
    "raw_mimic_modified": False,
}

atomic_write_json(
    meta,
    METADATA_PATH,
)


# ============================================================
# 15. FINAL A2 HARD CHECKS
# ============================================================

print("\n" + "-" * 80)
print("FINAL A2 HARD CHECKS")
print("-" * 80)

if len(main_output) != main_output["stay_id"].nunique():
    raise AssertionError(
        "Final A2 stay_id uniqueness check failed."
    )

if len(main_output) != len(metadata_df):
    raise AssertionError(
        "Embedding output / metadata row mismatch."
    )

# Main output must be a subset of A0 stays.
a0_stay_ids = set(
    cohort["stay_id"].astype("int64")
)

out_stay_ids = set(
    main_output["stay_id"].astype("int64")
)

if not out_stay_ids.issubset(a0_stay_ids):
    raise AssertionError(
        "A2 output contains stay IDs outside the A0 cohort."
    )

# Final output rows are radiology-only by construction.
if not metadata_df["note_source"].eq("radiology").all():
    raise AssertionError(
        "Final A2 metadata contains a non-radiology source."
    )

if int(metadata_df["discharge_note_count"].sum()) != 0:
    raise AssertionError(
        "Discharge notes are present in the final A2 metadata."
    )

if not OUT_PATH.exists():
    raise AssertionError(
        "Main A2 parquet output was not created."
    )

if not METADATA_CSV_PATH.exists():
    raise AssertionError(
        "A2 stay metadata was not created."
    )

if not TIMING_AUDIT_PATH.exists():
    raise AssertionError(
        "A2 timing audit was not created."
    )

if not AUDIT_PATH.exists():
    raise AssertionError(
        "A2 note audit was not created."
    )

if not METADATA_PATH.exists():
    raise AssertionError(
        "A2 metadata JSON was not created."
    )

print(
    "Output rows          :",
    f"{len(main_output):,}",
)
print(
    "Unique stays         :",
    f"{main_output['stay_id'].nunique():,}",
)
print(
    "Embedding dimension  :",
    EXPECTED_HIDDEN_SIZE,
)
print(
    "Text source          :",
    "radiology ONLY",
)
print(
    "Discharge notes      :",
    0,
)
print(
    "Main output          :",
    OUT_PATH,
)
print(
    "Stay metadata        :",
    METADATA_CSV_PATH,
)
print(
    "Timing audit         :",
    TIMING_AUDIT_PATH,
)
print(
    "Note audit           :",
    AUDIT_PATH,
)
print(
    "Metadata JSON        :",
    METADATA_PATH,
)
print(
    "V1 output modified   : False"
)
print(
    "Raw MIMIC modified   : False"
)

print("\n" + "=" * 80)
print("✅ PIPELINE A V2 — A2 COMPLETE (RADIOLOGY-ONLY)")
print("=" * 80)


# ============================================================
# 16. CLEANUP
# ============================================================

del texts
del note_embeddings
del notes_sorted
del embeddings_sorted
del notes
del metadata_df
del main_output
del stay_rows

gc.collect()

torch.cuda.empty_cache()
