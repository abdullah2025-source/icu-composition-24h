
# ==================================================================================================
# MODEL 26 DEFINITIVE COMPUTE PROVENANCE AUDIT
# ==================================================================================================
#
# Purpose:
#   Resolve whether the tuned Mamba-2 benchmark artifact was:
#     - tuned/selected/trained on CUDA,
#     - tuned/selected/trained on CPU,
#     - or has mixed/incomplete evidence.
#
# No retraining. No benchmark metric recalculation.
# ==================================================================================================

from pathlib import Path
import json
import re
import sqlite3
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path("outputs")
SEL = ROOT / "benchmark_selection"
SEL.mkdir(parents=True, exist_ok=True)
MODEL_DIR = ROOT / "baseline_family_06_state_space_24_27"

FIXED_DIR = MODEL_DIR / "fixed" / "26_Mamba-2"
TUNED_DIR = MODEL_DIR / "tuned" / "26_Mamba-2"
FINAL_DIR = TUNED_DIR / "final_selected_configuration"


def read_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def print_json_section(title, obj):
    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)
    print(json.dumps(obj, indent=2, default=str))


def safe_read_csv(path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"CSV READ ERROR: {path}: {e}")
        return None


print("\n" + "=" * 120)
print("MODEL 26 DEFINITIVE COMPUTE PROVENANCE AUDIT")
print("=" * 120)


# ==================================================================================================
# 1. REQUIRED PATH CHECK
# ==================================================================================================

paths = {
    "fixed_metadata": FIXED_DIR / "metadata.json",
    "tuned_metadata": FINAL_DIR / "metadata.json",
    "best_params": TUNED_DIR / "best_params.json",
    "study_db": TUNED_DIR / "study.db",
    "study_storage": TUNED_DIR / "study_storage.txt",
    "optuna_trials": TUNED_DIR / "optuna_trials.csv",
    "fixed_history": FIXED_DIR / "training_history.csv",
    "tuned_history": FINAL_DIR / "training_history.csv",
    "fixed_metrics": FIXED_DIR / "metrics.csv",
    "tuned_metrics": FINAL_DIR / "metrics.csv",
}

print("\nFILE INVENTORY:")
for name, path in paths.items():
    print(
        f"{name:18s} | "
        f"{'FOUND' if path.exists() else 'MISSING'} | "
        f"{path}"
    )


# ==================================================================================================
# 2. FIXED METADATA
# ==================================================================================================

fixed_meta = (
    read_json(paths["fixed_metadata"])
    if paths["fixed_metadata"].exists()
    else {}
)

tuned_meta = (
    read_json(paths["tuned_metadata"])
    if paths["tuned_metadata"].exists()
    else {}
)

print_json_section(
    "FIXED MODEL 26 METADATA",
    {
        "device": fixed_meta.get("device"),
        "gpu_name": fixed_meta.get("gpu_name"),
        "amp_mode": fixed_meta.get("amp_mode"),
        "validation_selection_score": fixed_meta.get("validation_selection_score"),
        "best_epoch": fixed_meta.get("best_epoch"),
        "test_used_for_selection": fixed_meta.get("test_used_for_selection"),
        "run_params": fixed_meta.get("run_params"),
    },
)


# ==================================================================================================
# 3. TUNED FINAL METADATA
# ==================================================================================================

print_json_section(
    "TUNED FINAL MODEL 26 METADATA",
    {
        "device": tuned_meta.get("device"),
        "gpu_name": tuned_meta.get("gpu_name"),
        "amp_mode": tuned_meta.get("amp_mode"),
        "validation_selection_score": tuned_meta.get("validation_selection_score"),
        "best_epoch": tuned_meta.get("best_epoch"),
        "n_optuna_trials_target": tuned_meta.get("n_optuna_trials_target"),
        "n_optuna_trials_completed": tuned_meta.get("n_optuna_trials_completed"),
        "n_optuna_trials": tuned_meta.get("n_optuna_trials"),
        "test_used_for_selection": tuned_meta.get("test_used_for_selection"),
        "run_params": tuned_meta.get("run_params"),
    },
)


# ==================================================================================================
# 4. OPTUNA TRIAL DATABASE INSPECTION
# ==================================================================================================

print("\n" + "=" * 120)
print("OPTUNA STUDY DATABASE INSPECTION")
print("=" * 120)

db = paths["study_db"]

if not db.exists():
    print("study.db: MISSING")
else:
    try:
        conn = sqlite3.connect(str(db))

        tables = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name",
            conn,
        )

        print("\nTABLES:")
        print(tables.to_string(index=False))

        # Study-level information.
        if "studies" in tables["name"].tolist():
            studies = pd.read_sql_query(
                "SELECT * FROM studies",
                conn,
            )
            print("\nSTUDIES:")
            print(studies.to_string(index=False))

        # User attributes sometimes contain provenance.
        if "study_user_attributes" in tables["name"].tolist():
            attrs = pd.read_sql_query(
                "SELECT * FROM study_user_attributes",
                conn,
            )
            print("\nSTUDY USER ATTRIBUTES:")
            print(attrs.to_string(index=False))

        if "trial_user_attributes" in tables["name"].tolist():
            trial_attrs = pd.read_sql_query(
                "SELECT * FROM trial_user_attributes",
                conn,
            )

            print("\nTRIAL USER ATTRIBUTES:")
            if trial_attrs.empty:
                print("NONE")
            else:
                print(trial_attrs.to_string(index=False))

        if "trials" in tables["name"].tolist():
            trials = pd.read_sql_query(
                "SELECT * FROM trials ORDER BY trial_id",
                conn,
            )

            print("\nTRIAL COUNT:", len(trials))

            if "state" in trials.columns:
                print("\nTRIAL STATES:")
                print(trials["state"].value_counts(dropna=False).to_string())

        conn.close()

    except Exception as e:
        print("SQLite inspection ERROR:", e)


# ==================================================================================================
# 5. OPTUNA CSV INSPECTION
# ==================================================================================================

print("\n" + "=" * 120)
print("OPTUNA TRIAL CSV INSPECTION")
print("=" * 120)

trial_csv = paths["optuna_trials"]

if trial_csv.exists():
    df = safe_read_csv(trial_csv)

    if df is not None:
        print("\nCOLUMNS:")
        print(list(df.columns))

        print("\nROWS:", len(df))

        # Print any columns that may encode device/provenance.
        provenance_cols = [
            c for c in df.columns
            if any(
                token in str(c).lower()
                for token in [
                    "device",
                    "gpu",
                    "cuda",
                    "roc",
                    "amp",
                    "worker",
                    "host",
                    "source",
                ]
            )
        ]

        print("\nPOTENTIAL PROVENANCE COLUMNS:")
        print(provenance_cols if provenance_cols else "NONE")

        if provenance_cols:
            print("\nPROVENANCE VALUES:")
            for c in provenance_cols:
                print(f"\n{c}:")
                print(df[c].value_counts(dropna=False).to_string())

        # Show trial-number / objective information if available.
        display_cols = [
            c for c in [
                "number",
                "trial_id",
                "value",
                "state",
                "datetime_start",
                "datetime_complete",
            ]
            if c in df.columns
        ]

        if display_cols:
            print("\nTRIAL SUMMARY:")
            print(df[display_cols].tail(10).to_string(index=False))

        complete = df.copy()
        if "state" in complete.columns:
            complete = complete[complete["state"].astype(str).str.contains("COMPLETE", na=False)]
        if not complete.empty and "value" in complete.columns:
            complete["value"] = pd.to_numeric(complete["value"], errors="coerce")
            complete = complete.dropna(subset=["value"])
            if not complete.empty:
                best_row = complete.loc[complete["value"].idxmax()]
                print("\nOPTUNA BEST FROM CSV:")
                print({
                    "trial_number": best_row.get("number"),
                    "objective_value": best_row.get("value"),
                    "state": best_row.get("state"),
                })
else:
    print("optuna_trials.csv: MISSING")


# ==================================================================================================
# 6. STUDY STORAGE TEXT
# ==================================================================================================

print("\n" + "=" * 120)
print("STUDY STORAGE TEXT")
print("=" * 120)

storage = paths["study_storage"]

if storage.exists():
    try:
        print(storage.read_text(encoding="utf-8"))
    except Exception as e:
        print("READ ERROR:", e)
else:
    print("study_storage.txt: MISSING")


# ==================================================================================================
# 7. TRAINING HISTORY / METRICS INSPECTION
# ==================================================================================================

for label, path in [
    ("FIXED TRAINING HISTORY", paths["fixed_history"]),
    ("TUNED TRAINING HISTORY", paths["tuned_history"]),
    ("FIXED METRICS", paths["fixed_metrics"]),
    ("TUNED METRICS", paths["tuned_metrics"]),
]:
    print("\n" + "=" * 120)
    print(label)
    print("=" * 120)

    if not path.exists():
        print("MISSING:", path)
        continue

    df = safe_read_csv(path)
    if df is None:
        continue

    print("ROWS:", len(df))
    print("COLUMNS:", list(df.columns))

    provenance_cols = [
        c for c in df.columns
        if any(
            token in str(c).lower()
            for token in [
                "device",
                "gpu",
                "cuda",
                "roc",
                "amp",
                "worker",
                "host",
            ]
        )
    ]

    if provenance_cols:
        print("PROVENANCE COLUMNS:", provenance_cols)
        for c in provenance_cols:
            print(f"\n{c}:")
            print(df[c].value_counts(dropna=False).to_string())
    else:
        print("PROVENANCE COLUMNS: NONE")

    print("\nTAIL:")
    print(df.tail(10).to_string(index=False))


# ==================================================================================================
# 8. RECURSIVE TEXT PROVENANCE SCAN
# ==================================================================================================

print("\n" + "=" * 120)
print("RECURSIVE MODEL 26 TEXT PROVENANCE SCAN")
print("=" * 120)

patterns = [
    re.compile(r"\bcuda\b", re.IGNORECASE),
    re.compile(r"\bcpu\b", re.IGNORECASE),
    re.compile(r"MI210", re.IGNORECASE),
    re.compile(r"rocm", re.IGNORECASE),
    re.compile(r"BF16", re.IGNORECASE),
]

text_extensions = {
    ".txt",
    ".log",
    ".json",
    ".csv",
    ".yaml",
    ".yml",
}

matches = []

for p in TUNED_DIR.rglob("*"):
    if not p.is_file():
        continue
    if p.suffix.lower() not in text_extensions:
        continue

    try:
        content = p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        continue

    for lineno, line in enumerate(content.splitlines(), start=1):
        hits = [
            pattern.pattern
            for pattern in patterns
            if pattern.search(line)
        ]

        if hits:
            matches.append({
                "file": str(p),
                "line": lineno,
                "hits": ",".join(hits),
                "text": line[:500],
            })

print("Text provenance matches:", len(matches))

if matches:
    scan_df = pd.DataFrame(matches)
    print(scan_df.to_string(index=False))
else:
    print("NONE")


# ==================================================================================================
# 9. DEFINITIVE INTERPRETATION
# ==================================================================================================

print("\n" + "=" * 120)
print("DEFINITIVE MODEL 26 COMPUTE-PROVENANCE INTERPRETATION")
print("=" * 120)

fixed_device = str(fixed_meta.get("device", "")).lower()
tuned_device = str(tuned_meta.get("device", "")).lower()

fixed_cuda = fixed_device == "cuda"
fixed_cpu = "cpu" in fixed_device

tuned_cuda = tuned_device == "cuda"
tuned_cpu = "cpu" in tuned_device

test_flags = [
    fixed_meta.get("test_used_for_selection"),
    tuned_meta.get("test_used_for_selection"),
]

print("Fixed artifact device:", fixed_meta.get("device"))
print("Fixed artifact GPU:", fixed_meta.get("gpu_name"))
print("Tuned final artifact device:", tuned_meta.get("device"))
print("Tuned final artifact GPU:", tuned_meta.get("gpu_name"))
print("Tuned trials completed:", tuned_meta.get("n_optuna_trials_completed"))
print("Tuned test_used_for_selection:", tuned_meta.get("test_used_for_selection"))

if tuned_cpu and not tuned_cuda:
    status = "TUNED_FINAL_CPU_CONFIRMED"
    reason = (
        "The selected tuned final Model 26 artifact explicitly records device=cpu "
        "and gpu_name=null. The fixed Model 26 artifact separately records CUDA/MI210. "
        "Therefore the completed benchmark contains mixed compute provenance: fixed=CUDA, "
        "tuned-final=CPU."
    )
elif tuned_cuda and not tuned_cpu:
    status = "TUNED_FINAL_CUDA_CONFIRMED"
    reason = (
        "The selected tuned final Model 26 artifact explicitly records CUDA. "
        "The fixed artifact is also inspected separately."
    )
else:
    status = "TUNED_FINAL_PROVENANCE_INCONCLUSIVE"
    reason = (
        "The tuned final metadata does not provide a single unambiguous CPU/CUDA state."
    )

if True in test_flags:
    reason += " WARNING: a test_used_for_selection=True flag exists."
elif test_flags and all(v is False for v in test_flags):
    reason += " Both fixed and tuned metadata report test_used_for_selection=False."
else:
    reason += " Some test-selection provenance is missing."

print("\nSTATUS:", status)
print("REASON:", reason)

print(
    "NOTE: The selected final Model 26 artifact is explicitly CUDA-confirmed when "
    "the status above is TUNED_FINAL_CUDA_CONFIRMED. The Optuna database/CSV inspected "
    "here do not contain per-trial device fields, so the compute device used by individual "
    "tuning trials is not independently established."
)


# ==================================================================================================
# 10. SAVE SUMMARY
# ==================================================================================================

summary = {
    "status": status,
    "reason": reason,
    "fixed": {
        "device": fixed_meta.get("device"),
        "gpu_name": fixed_meta.get("gpu_name"),
        "amp_mode": fixed_meta.get("amp_mode"),
        "validation_selection_score": fixed_meta.get("validation_selection_score"),
        "test_used_for_selection": fixed_meta.get("test_used_for_selection"),
    },
    "tuned_final": {
        "device": tuned_meta.get("device"),
        "gpu_name": tuned_meta.get("gpu_name"),
        "amp_mode": tuned_meta.get("amp_mode"),
        "validation_selection_score": tuned_meta.get("validation_selection_score"),
        "n_optuna_trials_completed": tuned_meta.get("n_optuna_trials_completed"),
        "n_optuna_trials_target": tuned_meta.get("n_optuna_trials_target"),
        "test_used_for_selection": tuned_meta.get("test_used_for_selection"),
    },
}

summary_path = SEL / "FINAL_model26_definitive_provenance_summary.json"

with summary_path.open("w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, default=str)

print("\nSaved:")
print(summary_path)

print("\n" + "=" * 120)
print("END MODEL 26 DEFINITIVE PROVENANCE AUDIT")
print("=" * 120)
