# ==================================================================================================
# FINAL CANDIDATE / THESIS-DECISION AUDIT — SCRIPT 2
# ==================================================================================================
#
# Purpose
# -------
# Deep-dive audit after Script 1 has produced the canonical 33-model benchmark.
#
# This version keeps the original audit structure and adds:
#   1. Strict prediction/target alias separation.
#   2. Robust validity parsing (avoids "false"/"0" -> True).
#   3. Prediction-file test-split verification.
#   4. Explicit PASS / FAIL / INCOMPLETE leakage status.
#   5. LOS-unit verification before LOS paired bootstrap.
#   6. Explicit patient-level vs stay-level bootstrap provenance.
#
# Run corrected Script 1 FIRST.
#
# ==================================================================================================

from pathlib import Path
import json
import re
import warnings

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
)

warnings.filterwarnings("ignore")


# ==================================================================================================
# 0. PROJECT PATHS
# ==================================================================================================

ROOT = Path("outputs")
SEL = ROOT / "benchmark_selection"
SEL.mkdir(parents=True, exist_ok=True)


# ==================================================================================================
# 1. MODEL REGISTRY
# ==================================================================================================

MODEL_NAMES = {
    "00": "XGBoost",
    "01": "Logistic Regression",
    "02": "Regularized Logistic Regression",
    "03": "Random Forest",
    "04": "Extra Trees",
    "05": "LightGBM",
    "06": "GRU",
    "07": "LSTM",
    "08": "BiGRU",
    "09": "GRU-D",
    "10": "TCN",
    "11": "ModernTCN",
    "12": "CNN+GRU",
    "13": "RETAIN",
    "14": "AdaCare",
    "15": "StageNet",
    "16": "ConCare",
    "17": "Transformer",
    "18": "Transformer+Conv",
    "19": "Transformer+LSTM",
    "20": "TFT",
    "21": "PatchTST",
    "22": "TimesNet",
    "23": "iTransformer",
    "24": "S4D",
    "25": "Mamba-1",
    "26": "Mamba-2",
    "27": "Mamba-3",
    "28": "Hybrid Mamba-Transformer",
    "29": "Multimodal Progression",
    "30": "Cross-Attention Multimodal",
    "31": "Gated Fusion/Cross-Attn",
    "32": "MoE Modality Fusion",
}

EXPECTED_MODEL_IDS = {f"{i:02d}" for i in range(33)}

BINARY_TASKS = [
    "mortality_24h",
    "ventilation_onset_24h",
    "vasopressor_onset_24h",
    "readmission_30d_24h",
]

PRIMARY_TASKS = BINARY_TASKS + ["los_hours_24h"]

candidate_ids = ["32", "31", "00", "05", "13", "24", "29"]

BOOTSTRAP_REPS = 5000
BOOTSTRAP_SEED = 20260923


# ==================================================================================================
# 2. HELPERS
# ==================================================================================================

def normalize_model_id(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(2)
    )


def first_existing(df, candidates, required=True):
    for col in candidates:
        if col in df.columns:
            return col
    if required:
        raise KeyError(
            f"Required column not found. Tried={candidates}. "
            f"Available={list(df.columns)}"
        )
    return None


def canonical_task_name(value: str) -> str:
    s = str(value).strip()
    aliases = {
        "mortality": "mortality_24h",
        "mortality_24h": "mortality_24h",
        "ventilation": "ventilation_onset_24h",
        "ventilation_onset": "ventilation_onset_24h",
        "ventilation_onset_24h": "ventilation_onset_24h",
        "vasopressor": "vasopressor_onset_24h",
        "vasopressor_onset": "vasopressor_onset_24h",
        "vasopressor_onset_24h": "vasopressor_onset_24h",
        "readmission": "readmission_30d_24h",
        "readmission_30d": "readmission_30d_24h",
        "readmission_30d_24h": "readmission_30d_24h",
        "los": "los_hours_24h",
        "icu_los": "los_hours_24h",
        "los_hours": "los_hours_24h",
        "los_hours_24h": "los_hours_24h",
    }
    return aliases.get(s, s)


def infer_stay_id_column(df):
    for col in ["stay_id", "icustay_id", "stayid"]:
        if col in df.columns:
            return col
    return None


def infer_subject_id_column(df):
    for col in ["subject_id", "subjectid", "patient_id", "patientid"]:
        if col in df.columns:
            return col
    return None


def file_model_match(path: Path, model_id: str) -> bool:
    normalized = str(path).replace("\\", "/")
    pattern = rf"(?:^|/){re.escape(model_id)}_[^/]+(?:/|$)"
    return re.search(pattern, normalized) is not None


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return np.nan


def normalize_validity(series: pd.Series) -> pd.Series:
    """
    Explicitly converts boolean/numeric/text validity values.
    Avoids pandas astype(bool), where non-empty strings such as "false"
    would incorrectly become True.
    """
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    numeric = pd.to_numeric(series, errors="coerce")
    out = numeric.eq(1)
    unresolved = numeric.isna()

    if unresolved.any():
        text = series.astype(str).str.strip().str.lower()
        out.loc[unresolved] = text.loc[unresolved].isin(
            ["true", "t", "yes", "y", "valid", "1"]
        )

    return out.fillna(False).astype(bool)


def infer_prediction_split_status(df: pd.DataFrame):
    """
    Returns:
      status in {"TEST_ONLY", "NON_TEST_PRESENT", "UNKNOWN"}
      column used, unique values
    """
    split_col = first_existing(
        df,
        [
            "split",
            "dataset_split",
            "partition",
            "subset",
            "set",
            "data_split",
        ],
        required=False,
    )
    if split_col is None:
        return "UNKNOWN", None, []

    vals = (
        df[split_col]
        .dropna()
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"test_assumed": "test"})
        .unique()
        .tolist()
    )

    non_test = [v for v in vals if v not in {"test"}]
    if non_test:
        return "NON_TEST_PRESENT", split_col, sorted(vals)

    return "TEST_ONLY", split_col, sorted(vals)


def los_units_status(df: pd.DataFrame):
    """
    Prediction files are acceptable for LOS paired MAE only when there is
    evidence that LOS prediction/target values are raw hours.

    Explicit unit columns containing 'hour'/'hr' are accepted.
    Otherwise status is UNKNOWN rather than silently assuming units.
    """
    unit_col = first_existing(
        df,
        [
            "los_units",
            "los_unit",
            "los_hours_unit",
            "prediction_units",
            "target_units",
            "unit",
        ],
        required=False,
    )

    if unit_col is None:
        return "UNKNOWN", None, []

    vals = (
        df[unit_col]
        .dropna()
        .astype(str)
        .str.strip()
        .str.lower()
        .unique()
        .tolist()
    )

    ok_tokens = {"hour", "hours", "hr", "hrs", "hour(s)", "hours_icu"}
    accepted = all(
        any(tok == x or tok in x for tok in ok_tokens)
        for x in vals
    ) if vals else False

    return (
        "RAW_HOURS" if accepted else "NON_HOURS_OR_UNKNOWN",
        unit_col,
        sorted(vals),
    )


# ==================================================================================================
# 3. LOAD SCRIPT 1 OUTPUTS
# ==================================================================================================

print("\n" + "=" * 120)
print("LOADING SCRIPT 1 FINAL BENCHMARK OUTPUTS")
print("=" * 120)

rank_path = SEL / "FINAL_33MODEL_tuned_test_ranking.csv"
details_path = SEL / "FINAL_33MODEL_all_modes_test_ranking.csv"

if not rank_path.exists():
    raise FileNotFoundError(f"Missing: {rank_path}. Run corrected Script 1 first.")

if not details_path.exists():
    raise FileNotFoundError(f"Missing: {details_path}. Run corrected Script 1 first.")

rank = pd.read_csv(rank_path)
details = pd.read_csv(details_path)

print(
    "\nNOTE: FINAL_33MODEL_all_modes_test_ranking.csv is used only as an "
    "aggregate model/mode audit table. Task-level statistics are loaded "
    "from the canonical Family 1-7 statistical-evaluation files below."
)

rank["model_id"] = normalize_model_id(
    rank[first_existing(rank, ["model_id", "_model_id"])]
)
details["model_id"] = normalize_model_id(
    details[first_existing(details, ["model_id", "_model_id"])]
)

rank_name_col = first_existing(
    rank, ["model_name", "_model_name_canonical"], required=False
)
rank["model_name"] = (
    rank[rank_name_col] if rank_name_col is not None
    else rank["model_id"].map(MODEL_NAMES)
)

detail_name_col = first_existing(
    details, ["model_name", "_model_name_canonical"], required=False
)
details["model_name"] = (
    details[detail_name_col] if detail_name_col is not None
    else details["model_id"].map(MODEL_NAMES)
)

detail_mode_col = first_existing(details, ["mode", "_mode_norm"], required=False)
if detail_mode_col is not None:
    details["mode_norm"] = (
        details[detail_mode_col].astype(str).str.strip().str.lower()
    )
else:
    details["mode_norm"] = np.nan

# IMPORTANT:
# FINAL_33MODEL_all_modes_test_ranking.csv is a model/mode summary table.
# It does NOT contain one row per task. Task-level results are therefore
# loaded below from the canonical family statistical-evaluation files.


# ==================================================================================================
# 4. HARD 33-MODEL COVERAGE GUARD
# ==================================================================================================

print("\n" + "=" * 120)
print("33-MODEL COVERAGE GUARD")
print("=" * 120)

found_rank_ids = set(rank["model_id"].astype(str).str.zfill(2))
missing_rank_ids = sorted(EXPECTED_MODEL_IDS - found_rank_ids)
unexpected_rank_ids = sorted(found_rank_ids - EXPECTED_MODEL_IDS)

print("Expected model count :", len(EXPECTED_MODEL_IDS))
print("Found model count    :", len(found_rank_ids))
print("Missing models       :", ", ".join(missing_rank_ids) if missing_rank_ids else "NONE")
print("Unexpected model IDs :", ", ".join(unexpected_rank_ids) if unexpected_rank_ids else "NONE")

if missing_rank_ids or unexpected_rank_ids:
    raise RuntimeError("33-model coverage is incomplete/invalid. Correct Script 1 first.")

print("\n33-MODEL COVERAGE CHECK: PASS ✅")


# ==================================================================================================
# 5. RANKING / CANDIDATE REPORTS
# ==================================================================================================

ranking_required = [
    "model_id", "model_name", "family",
    "mean_binary_auprc", "mean_binary_auroc",
    "mean_brier", "mean_ece", "los_mae",
]
missing = [c for c in ranking_required if c not in rank.columns]
if missing:
    raise RuntimeError(f"Final ranking missing required columns: {missing}")

rank_display = (
    rank[ranking_required]
    .copy()
    .sort_values("mean_binary_auprc", ascending=False)
)

print("\n" + "=" * 120)
print("1. FINAL TUNED 33-MODEL BENCHMARK")
print("=" * 120)
print(rank_display.to_string(index=False))

candidate_rank = (
    rank[rank["model_id"].isin(candidate_ids)][ranking_required]
    .copy()
    .sort_values("mean_binary_auprc", ascending=False)
)

print("\n" + "=" * 120)
print("2. PRIMARY CANDIDATE COMPARISON")
print("=" * 120)
print(candidate_rank.to_string(index=False))


# ==================================================================================================
# 6. TASK-LEVEL METRICS + CIs
# ==================================================================================================
#
# IMPORTANT:
# Script 1's FINAL_33MODEL_all_modes_test_ranking.csv is an aggregate
# model/mode table and has no task column. Task-level metrics must therefore
# come directly from the canonical family statistical-evaluation files.
#
# Family 1 and Family 2 canonical files do not contain an explicit split
# column; their rows are test_assumed. This matches the corrected Script 1
# interpretation and does NOT recalculate any statistics.
# ==================================================================================================

print("\n" + "=" * 120)
print("3. TASK-LEVEL METRICS + 95% CIs")
print("=" * 120)


CANONICAL_TASK_SOURCES = {
    "01": ROOT / "baseline_family_01_tabular" / "stats" / "family01_all_metrics_with_stats.csv",
    "02": ROOT / "baseline_family_02_rnn" / "stats" / "family02_all_metrics_with_stats.csv",
    "03": ROOT / "baseline_family_03_tcn_10_12" / "family03_statistical_evaluation.csv",
    "04": ROOT / "baseline_family_04_clinical_sequence_13_16" / "family04_statistical_evaluation.csv",
    "05": ROOT / "baseline_family_05_transformer_modern_17_23" / "family05_statistical_evaluation.csv",
    "06": ROOT / "baseline_family_06_state_space_24_27" / "family06_statistical_evaluation.csv",
}


def discover_family07_task_source(root: Path) -> Path:
    """
    Family 7 has the same canonical task-statistics role as Families 3-6,
    but its directory/file naming has varied across the completed runs.

    Discovery is restricted to outputs/ and requires a CSV whose filename
    contains family07 plus statistical/evaluation terminology. No metrics
    are recalculated.
    """
    exact = sorted(root.rglob("family07_statistical_evaluation.csv"))
    if exact:
        if len(exact) > 1:
            print(
                "\nWARNING: Multiple exact Family 7 task-stat files found; "
                f"using shortest canonical-looking path: {exact[0]}"
            )
        return exact[0]

    candidates = []
    for p in root.rglob("*.csv"):
        name = p.name.lower()
        path_text = str(p).replace("\\", "/").lower()

        family7_match = (
            "family07" in name
            or "family_07" in path_text
            or "baseline_family_07" in path_text
        )
        evaluation_match = (
            "statistical" in name
            or "evaluation" in name
            or "metrics_with_stats" in name
        )

        if family7_match and evaluation_match:
            candidates.append(p)

    if not candidates:
        raise FileNotFoundError(
            "Could not discover the canonical Family 7 task-statistics CSV "
            "under outputs/. Search was restricted to filenames/paths "
            "containing Family 7 plus statistical/evaluation terminology."
        )

    # Prefer exact family07 naming, then statistical-evaluation naming,
    # then the shortest path.
    candidates.sort(
        key=lambda p: (
            p.name.lower() != "family07_statistical_evaluation.csv",
            "statistical_evaluation" not in p.name.lower(),
            len(str(p)),
            str(p),
        )
    )

    if len(candidates) > 1:
        print(
            "\nFamily 7 task-statistics discovery candidates:"
        )
        for c in candidates:
            print(f"  - {c}")
        print(f"Using: {candidates[0]}")

    return candidates[0]


CANONICAL_TASK_SOURCES["07"] = discover_family07_task_source(ROOT)


def _standardize_task_stats_columns(df: pd.DataFrame, family_num: str) -> pd.DataFrame:
    """
    Normalize canonical family statistical-evaluation files without
    recalculating any metric/statistic.
    """
    out = df.copy()

    # Case-insensitive canonical names.
    rename = {}
    lookup = {str(c).strip().lower(): c for c in out.columns}

    aliases = {
        "task": ["task"],
        "split": ["split", "dataset_split", "partition"],
        "mode": ["mode", "run_mode"],
        "model_id": ["model_id", "_model_id"],
        "model_name": ["model_name", "_model_name_canonical"],
        "auprc": ["auprc", "AUPRC"],
        "auroc": ["auroc", "AUROC"],
        "brier": ["brier", "Brier"],
        "ece": ["ece", "ECE"],
        "mae": ["mae", "MAE"],
    }

    # CI aliases are handled separately below.
    for canonical, candidates in aliases.items():
        for candidate in candidates:
            key = str(candidate).lower()
            if key in lookup:
                rename[lookup[key]] = canonical
                break

    # Common CI spelling variants.
    ci_aliases = {
        "auprc_ci95_low": [
            "auprc_ci95_low", "auprc_ci_low", "AUPRC_CI95_low", "AUPRC_CI_low"
        ],
        "auprc_ci95_high": [
            "auprc_ci95_high", "auprc_ci_high", "AUPRC_CI95_high", "AUPRC_CI_high"
        ],
        "auroc_ci95_low": [
            "auroc_ci95_low", "auroc_ci_low", "AUROC_CI95_low", "AUROC_CI_low"
        ],
        "auroc_ci95_high": [
            "auroc_ci95_high", "auroc_ci_high", "AUROC_CI95_high", "AUROC_CI_high"
        ],
        "brier_ci95_low": [
            "brier_ci95_low", "brier_ci_low", "Brier_CI95_low", "Brier_CI_low"
        ],
        "brier_ci95_high": [
            "brier_ci95_high", "brier_ci_high", "Brier_CI95_high", "Brier_CI_high"
        ],
        "mae_ci95_low": [
            "mae_ci95_low", "mae_ci_low", "MAE_CI95_low", "MAE_CI_low"
        ],
        "mae_ci95_high": [
            "mae_ci95_high", "mae_ci_high", "MAE_CI95_high", "MAE_CI_high"
        ],
        "rmse_ci95_low": [
            "rmse_ci95_low", "rmse_ci_low", "RMSE_CI95_low", "RMSE_CI_low"
        ],
        "rmse_ci95_high": [
            "rmse_ci95_high", "rmse_ci_high", "RMSE_CI95_high", "RMSE_CI_high"
        ],
        "median_ae_ci95_low": [
            "median_ae_ci95_low", "median_ae_ci_low"
        ],
        "median_ae_ci95_high": [
            "median_ae_ci95_high", "median_ae_ci_high"
        ],
        "r2_ci95_low": [
            "r2_ci95_low", "r2_ci_low"
        ],
        "r2_ci95_high": [
            "r2_ci95_high", "r2_ci_high"
        ],
    }

    for canonical, candidates in ci_aliases.items():
        for candidate in candidates:
            key = str(candidate).lower()
            if key in lookup:
                rename[lookup[key]] = canonical
                break

    out = out.rename(columns=rename)

    required = ["task", "mode", "model_id", "auprc", "auroc", "brier", "ece"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise RuntimeError(
            f"Canonical task-stat file for Family {family_num} is missing "
            f"required columns {missing}. Available={list(df.columns)}"
        )

    out["model_id"] = normalize_model_id(out["model_id"])
    out["task"] = out["task"].astype(str).str.strip().map(canonical_task_name)
    out["mode"] = out["mode"].astype(str).str.strip().str.lower()

    if "split" not in out.columns:
        out["split"] = "test_assumed"
    else:
        out["split"] = (
            out["split"].astype(str).str.strip().str.lower()
            .replace({"test_assumed": "test_assumed"})
        )

    if "model_name" not in out.columns:
        out["model_name"] = out["model_id"].map(MODEL_NAMES)

    out["_source_family"] = family_num
    out["_source_path"] = str(CANONICAL_TASK_SOURCES[family_num])

    return out


canonical_task_frames = []

for family_num, source_path in CANONICAL_TASK_SOURCES.items():
    print(f"\nFamily {family_num}: {source_path}")

    if not source_path.exists():
        raise FileNotFoundError(
            f"Canonical task-statistics source missing for Family {family_num}:\n{source_path}"
        )

    raw = pd.read_csv(source_path)
    norm = _standardize_task_stats_columns(raw, family_num)

    print(
        f"  rows={len(norm)} | "
        f"models={norm['model_id'].nunique()} | "
        f"tasks={norm['task'].nunique()} | "
        f"modes={sorted(norm['mode'].dropna().unique().tolist())} | "
        f"splits={sorted(norm['split'].dropna().unique().tolist())}"
    )

    canonical_task_frames.append(norm)


task_stats = pd.concat(
    canonical_task_frames,
    ignore_index=True,
)


# Keep only the one canonical tuned/test-assumed record per model/task.
task_stats = task_stats[
    task_stats["mode"].eq("tuned")
    & task_stats["split"].isin(["test", "test_assumed"])
    & task_stats["task"].isin(set(PRIMARY_TASKS))
].copy()


# Sanity check: one row per model/task after canonical filtering.
duplicate_task_rows = (
    task_stats.groupby(["model_id", "task"]).size()
    .reset_index(name="n")
)
duplicate_task_rows = duplicate_task_rows[
    duplicate_task_rows["n"] > 1
]

if not duplicate_task_rows.empty:
    print("\nWARNING: duplicate model/task rows detected across canonical sources:")
    print(duplicate_task_rows.to_string(index=False))
else:
    print("\nCanonical task-statistics uniqueness check: PASS ✅")


expected_task_pairs = len(EXPECTED_MODEL_IDS) * len(PRIMARY_TASKS)
found_task_pairs = (
    task_stats[["model_id", "task"]]
    .drop_duplicates()
    .shape[0]
)

print(
    f"Canonical tuned held-out task rows: {len(task_stats)} "
    f"(expected model-task pairs = {expected_task_pairs}, "
    f"found unique pairs = {found_task_pairs})"
)


task_rows = task_stats[
    task_stats["model_id"].isin(candidate_ids)
].copy()


metric_candidates = [
    "model_id", "model_name", "task",
    "auroc", "auroc_ci95_low", "auroc_ci95_high",
    "auroc_ci_low", "auroc_ci_high",
    "auprc", "auprc_ci95_low", "auprc_ci95_high",
    "auprc_ci_low", "auprc_ci_high",
    "brier", "brier_ci95_low", "brier_ci95_high",
    "brier_ci_low", "brier_ci_high",
    "ece",
    "mae", "mae_ci95_low", "mae_ci95_high",
    "mae_ci_low", "mae_ci_high",
    "rmse", "rmse_ci95_low", "rmse_ci95_high",
    "rmse_ci_low", "rmse_ci_high",
    "median_ae", "median_ae_ci95_low", "median_ae_ci95_high",
    "median_ae_ci_low", "median_ae_ci_high",
    "r2", "r2_ci95_low", "r2_ci95_high",
    "r2_ci_low", "r2_ci_high",
    "n", "prevalence",
    "_source_family", "_source_path",
]

metric_cols = [c for c in metric_candidates if c in task_rows.columns]

if task_rows.empty:
    print("WARNING: No candidate task rows were found in canonical family statistics.")
else:
    print(
        task_rows[metric_cols]
        .sort_values(["model_id", "task"])
        .to_string(index=False)
    )


candidate_task_path = SEL / "FINAL_candidate_deep_dive_task_metrics.csv"
task_rows.to_csv(candidate_task_path, index=False)
print(f"\nSaved: {candidate_task_path}")


# ==================================================================================================
# 7. TASK WINNERS ACROSS ALL 33
# ==================================================================================================

# 7. TASK WINNERS ACROSS ALL 33
# ==================================================================================================

print("\n" + "=" * 120)
print("4. TASK-LEVEL TOP RESULTS ACROSS ALL 33 MODELS")
print("=" * 120)

all_test = task_stats[
    task_stats["task"].isin(set(BINARY_TASKS))
].copy()

all_test["auprc"] = pd.to_numeric(
    all_test["auprc"],
    errors="coerce",
)

for task in BINARY_TASKS:
    rows = (
        all_test[all_test["task"].eq(task)]
        .dropna(subset=["auprc"])
        .sort_values("auprc", ascending=False)
    )

    print(f"\n{task}")
    if rows.empty:
        print("NO VALID RESULT")
    else:
        print(
            rows[["model_id", "model_name", "auprc"]]
            .head(5)
            .to_string(index=False)
        )


# ==================================================================================================
# ==================================================================================================
# 8. EXACT TUNED PARAMETERS / PROVENANCE
# ==================================================================================================

print("\n" + "=" * 120)
print("5. EXACT TUNED PARAMETERS / FINAL PROVENANCE")
print("=" * 120)


def find_best_params(model_id):
    candidates = []
    for p in ROOT.rglob("best_params.json"):
        normalized = str(p).replace("\\", "/")
        if not file_model_match(p, model_id):
            continue
        if "tuned" not in normalized.lower():
            continue
        if "cpu_audit" in normalized.lower():
            continue
        candidates.append(p)

    candidates.sort(
        key=lambda p: (
            "final_selected_configuration" not in str(p),
            len(str(p)),
            str(p),
        )
    )
    return candidates


def find_final_metadata(model_id):
    candidates = []
    for p in ROOT.rglob("metadata.json"):
        normalized = str(p).replace("\\", "/")
        if not file_model_match(p, model_id):
            continue
        if "tuned" not in normalized.lower():
            continue
        if "final_selected_configuration" not in normalized:
            continue
        if "cpu_audit" in normalized.lower():
            continue
        candidates.append(p)

    candidates.sort(key=lambda p: (len(str(p)), str(p)))
    return candidates


for mid in candidate_ids:
    print("\n" + "=" * 35)
    print(f"{mid} — {MODEL_NAMES[mid]}")
    print("=" * 35)

    best_files = find_best_params(mid)
    if best_files:
        p = best_files[0]
        print("\nBEST_PARAMS:")
        print(p)
        try:
            print(json.dumps(read_json(p), indent=2, default=str))
        except Exception as e:
            print("READ ERROR:", e)
    else:
        print("\nbest_params.json: NOT FOUND")

    meta_files = find_final_metadata(mid)
    if meta_files:
        p = meta_files[0]
        print("\nMETADATA:")
        print(p)
        try:
            obj = read_json(p)
            important = {
                "device": obj.get("device"),
                "gpu_name": obj.get("gpu_name"),
                "amp_mode": obj.get("amp_mode"),
                "validation_selection_score": obj.get("validation_selection_score"),
                "best_epoch": obj.get("best_epoch"),
                "n_optuna_trials_target": obj.get("n_optuna_trials_target"),
                "n_optuna_trials_completed": obj.get("n_optuna_trials_completed"),
                "n_optuna_trials": obj.get("n_optuna_trials"),
                "test_used_for_selection": obj.get("test_used_for_selection"),
                "run_params": obj.get("run_params"),
            }
            print(json.dumps(important, indent=2, default=str))
        except Exception as e:
            print("READ ERROR:", e)
    else:
        print("\nmetadata.json: NOT FOUND")


# ==================================================================================================
# 9. MODEL 29 OLD CPU VS CURRENT GPU/BF16
# ==================================================================================================

print("\n" + "=" * 120)
print("6. MODEL 29 — OLD CPU AUDIT VS CURRENT GPU/BF16")
print("=" * 120)

old_cpu_meta = list(
    ROOT.rglob(
        "29_Multimodal_Progression/final_selected_configuration_cpu_audit/metadata.json"
    )
)
new_gpu_meta = list(
    ROOT.rglob(
        "29_Multimodal_Progression/final_selected_configuration/metadata.json"
    )
)

print("\nOLD CPU AUDIT:")
if old_cpu_meta:
    for p in old_cpu_meta:
        try:
            obj = read_json(p)
            print(
                "PATH:", p,
                "\ndevice=", obj.get("device"),
                "| gpu=", obj.get("gpu_name"),
                "| amp=", obj.get("amp_mode"),
                "| val_sel=", obj.get("validation_selection_score"),
                "| best_epoch=", obj.get("best_epoch"),
            )
        except Exception as e:
            print("READ ERROR:", e)
else:
    print("OLD CPU AUDIT METADATA NOT FOUND")

print("\nCURRENT GPU/BF16:")
if new_gpu_meta:
    for p in new_gpu_meta:
        try:
            obj = read_json(p)
            print(
                "PATH:", p,
                "\ndevice=", obj.get("device"),
                "| gpu=", obj.get("gpu_name"),
                "| amp=", obj.get("amp_mode"),
                "| val_sel=", obj.get("validation_selection_score"),
                "| best_epoch=", obj.get("best_epoch"),
                "| trials=",
                obj.get("n_optuna_trials_completed"),
                "/",
                obj.get("n_optuna_trials_target"),
            )
        except Exception as e:
            print("READ ERROR:", e)
else:
    print("CURRENT GPU/BF16 METADATA NOT FOUND")


# ==================================================================================================
# 10. MODEL 29 PROVENANCE HARD CHECK
# ==================================================================================================

print("\n" + "=" * 120)
print("MODEL 29 GPU/BF16 PROVENANCE CHECK")
print("=" * 120)

model29_gpu_pass = False

for p in new_gpu_meta:
    try:
        obj = read_json(p)

        device_ok = str(obj.get("device", "")).lower() == "cuda"
        gpu_ok = "mi210" in str(obj.get("gpu_name", "")).lower()
        amp_ok = "bf16" in str(obj.get("amp_mode", "")).lower()
        trials_ok = (
            obj.get("n_optuna_trials_target") == 30
            and obj.get("n_optuna_trials_completed") == 30
        )
        not_test_selected = obj.get("test_used_for_selection") is False

        model29_gpu_pass = (
            device_ok and gpu_ok and amp_ok and trials_ok and not_test_selected
        )

        for key in [
            "device",
            "gpu_name",
            "amp_mode",
            "validation_selection_score",
            "best_epoch",
            "n_optuna_trials_completed",
            "n_optuna_trials_target",
            "test_used_for_selection",
        ]:
            print(f"{key}: {obj.get(key)}")
        break
    except Exception as e:
        print("READ ERROR:", e)

print(
    "\nMODEL 29 GPU/BF16 PROVENANCE:",
    "PASS ✅" if model29_gpu_pass else "FAIL / INCOMPLETE ❌"
)


# ==================================================================================================
# 11. PREDICTION FILE DISCOVERY + TEST-SPLIT AUDIT
# ==================================================================================================

print("\n" + "=" * 120)
print("7. PREDICTION FILE / TEST-SPLIT / PAIRING AUDIT")
print("=" * 120)


def find_prediction_file(model_id):
    candidates = []

    for wanted_name in ["test_predictions.csv", "predictions.csv"]:
        for p in ROOT.rglob(wanted_name):
            normalized = str(p).replace("\\", "/")
            if not file_model_match(p, model_id):
                continue
            if "tuned" not in normalized.lower():
                continue
            if "cpu_audit" in normalized.lower():
                continue
            candidates.append(p)

        if candidates:
            break

    candidates.sort(
        key=lambda p: (
            "final_selected_configuration" not in str(p),
            p.name != "test_predictions.csv",
            len(str(p)),
            str(p),
        )
    )

    return candidates[0] if candidates else None


prediction_paths = {}
prediction_audit_rows = []

for mid in candidate_ids:
    p = find_prediction_file(mid)

    print(f"\n{mid} | {MODEL_NAMES[mid]}")

    if p is None:
        print("PREDICTION FILE: NOT FOUND")
        prediction_audit_rows.append({
            "model_id": mid,
            "model_name": MODEL_NAMES[mid],
            "path": None,
            "compatible_candidate_prediction": False,
            "reason": "Prediction file not found.",
        })
        continue

    try:
        dfp = pd.read_csv(p)
        split_status, split_col, split_values = infer_prediction_split_status(dfp)
        los_status, los_unit_col, los_unit_values = los_units_status(dfp)

        print("PATH:", p)
        print("ROWS:", len(dfp))
        print("COLUMNS:", list(dfp.columns))
        print("TEST-SPLIT STATUS:", split_status)
        print("SPLIT COLUMN:", split_col)
        print("SPLIT VALUES:", split_values)

        # Explicit split column with test+val is acceptable because the
        # standardizer filters the artifact to split == test before pairing.
        test_verified = split_status in {"TEST_ONLY", "NON_TEST_PRESENT"} and split_col is not None

        if split_status == "NON_TEST_PRESENT":
            print("BOOTSTRAP ELIGIBILITY: YES — explicit test rows will be filtered")
        elif split_status == "UNKNOWN":
            print("BOOTSTRAP ELIGIBILITY: INCOMPLETE — test split not verifiable")
        else:
            print("BOOTSTRAP ELIGIBILITY: YES — test-only artifact")

        id_candidates = [
            c for c in [
                "subject_id", "stay_id", "hadm_id", "row_id", "index"
            ]
            if c in dfp.columns
        ]
        print("ID CANDIDATES:", id_candidates)
        print("LOS UNITS STATUS:", los_status)
        print("LOS UNIT COLUMN:", los_unit_col)
        print("LOS UNIT VALUES:", los_unit_values)

        prediction_paths[mid] = p

        prediction_audit_rows.append({
            "model_id": mid,
            "model_name": MODEL_NAMES[mid],
            "path": str(p),
            "rows": len(dfp),
            "test_split_status": split_status,
            "split_column": split_col,
            "split_values": json.dumps(split_values),
            "test_only_verified": test_verified,
            "los_units_status": los_status,
            "los_unit_column": los_unit_col,
            "los_unit_values": json.dumps(los_unit_values),
            "compatible_candidate_prediction": test_verified,
            "reason": (
                "Explicit split column available; test rows will be filtered before pairing."
                if test_verified and split_status == "NON_TEST_PRESENT"
                else (
                    "Test-only prediction artifact."
                    if test_verified
                    else
                    "Prediction artifact exists but held-out test split is not safely verified."
                )
            ),
        })

    except Exception as e:
        print("READ ERROR:", e)
        prediction_audit_rows.append({
            "model_id": mid,
            "model_name": MODEL_NAMES[mid],
            "path": str(p),
            "compatible_candidate_prediction": False,
            "reason": f"Prediction file read error: {e}",
        })


# ==================================================================================================
# 12. PREDICTION SCHEMA NORMALIZATION
# ==================================================================================================
#
# Standard output:
#   stay_id
#   subject_id
#   task
#   target
#   prediction
#   validity
#
# Strictly avoids using the same ambiguous bare task column as both target
# and prediction.
# ==================================================================================================

def find_column_by_aliases(df, aliases):
    lookup = {str(c).lower(): c for c in df.columns}
    for alias in aliases:
        key = str(alias).lower()
        if key in lookup:
            return lookup[key]
    return None


def task_column_aliases(task, kind):
    mapping = {
        "mortality_24h": {
            "prediction": [
                "pred_mortality_24h",
                "prob_mortality_24h",
                "mortality_24h_pred",
                "mortality_24h_prob",
                "mortality_pred",
                "mortality_prob",
                "p_mortality",
            ],
            "target": [
                "target_mortality_24h",
                "mortality_24h_target",
                "mortality_24h_true",
                "y_mortality_24h",
                "y_mortality",
            ],
            "validity": [
                "mortality_24h_valid",
                "valid_mortality_24h",
                "mortality_valid",
                "validity_mortality_24h",
            ],
        },
        "ventilation_onset_24h": {
            "prediction": [
                "pred_ventilation_onset_24h",
                "prob_ventilation_onset_24h",
                "ventilation_onset_24h_pred",
                "ventilation_onset_24h_prob",
                "ventilation_pred",
                "ventilation_prob",
                "p_ventilation",
            ],
            "target": [
                "target_ventilation_onset_24h",
                "ventilation_onset_24h_target",
                "ventilation_onset_24h_true",
                "y_ventilation_onset_24h",
                "y_ventilation",
            ],
            "validity": [
                "ventilation_onset_24h_valid",
                "valid_ventilation_onset_24h",
                "ventilation_valid",
                "validity_ventilation_onset_24h",
            ],
        },
        "vasopressor_onset_24h": {
            "prediction": [
                "pred_vasopressor_onset_24h",
                "prob_vasopressor_onset_24h",
                "vasopressor_onset_24h_pred",
                "vasopressor_onset_24h_prob",
                "vasopressor_pred",
                "vasopressor_prob",
                "p_vasopressor",
            ],
            "target": [
                "target_vasopressor_onset_24h",
                "vasopressor_onset_24h_target",
                "vasopressor_onset_24h_true",
                "y_vasopressor_onset_24h",
                "y_vasopressor",
            ],
            "validity": [
                "vasopressor_onset_24h_valid",
                "valid_vasopressor_onset_24h",
                "vasopressor_valid",
                "validity_vasopressor_onset_24h",
            ],
        },
        "readmission_30d_24h": {
            "prediction": [
                "pred_readmission_30d_24h",
                "prob_readmission_30d_24h",
                "readmission_30d_24h_pred",
                "readmission_30d_24h_prob",
                "readmission_pred",
                "readmission_prob",
                "p_readmission",
            ],
            "target": [
                "target_readmission_30d_24h",
                "readmission_30d_24h_target",
                "readmission_30d_24h_true",
                "y_readmission_30d_24h",
                "y_readmission",
            ],
            "validity": [
                "readmission_30d_24h_valid",
                "valid_readmission_30d_24h",
                "readmission_valid",
                "validity_readmission_30d_24h",
            ],
        },
        "los_hours_24h": {
            "prediction": [
                "pred_los_hours_24h",
                "los_hours_24h_pred",
                "los_pred",
                "los_prediction",
                "prediction_los_hours_24h",
            ],
            "target": [
                "target_los_hours_24h",
                "los_hours_24h_target",
                "los_hours_24h_true",
                "y_los_hours_24h",
            ],
            "validity": [
                "los_hours_24h_valid",
                "valid_los_hours_24h",
                "los_valid",
                "validity_los_hours_24h",
            ],
        },
    }
    return mapping.get(task, {}).get(kind, [])


def standardize_prediction_file(path):
    df = pd.read_csv(path)

    split_status, split_col, split_values = infer_prediction_split_status(df)

    # Prediction artifacts in the completed runs contain both validation and
    # test rows, with an explicit split column. This is safe for pairing:
    # retain only rows explicitly marked as test.
    if split_status == "NON_TEST_PRESENT" and split_col is not None:
        split_norm = (
            df[split_col]
            .astype(str)
            .str.strip()
            .str.lower()
        )
        test_df = df[split_norm.eq("test")].copy()

        if test_df.empty:
            return None, {
                "path": str(path),
                "rows": len(df),
                "format": None,
                "compatible": False,
                "reason": (
                    "Split column exists but contains no rows explicitly marked 'test'. "
                    f"values={split_values}"
                ),
            }

        df = test_df
        split_filter_applied = True
    elif split_status == "TEST_ONLY":
        split_filter_applied = False
    else:
        return None, {
            "path": str(path),
            "rows": len(df),
            "format": None,
            "compatible": False,
            "reason": (
                "Test split could not be verified because no explicit usable "
                f"split column was found. status={split_status}, values={split_values}"
            ),
        }

    stay_col = infer_stay_id_column(df)
    subject_col = infer_subject_id_column(df)

    audit = {
        "path": str(path),
        "rows": int(len(df)),
        "format": None,
        "compatible": False,
        "reason": None,
        "stay_id_column": stay_col,
        "subject_id_column": subject_col,
    }

    if stay_col is None:
        audit["reason"] = "No stay_id-like column."
        return None, audit

    # ------------------------------------------------------------------
    # A. Long format
    # ------------------------------------------------------------------
    required_long = {"task", "target", "prediction"}

    if required_long.issubset(set(df.columns)):
        audit["format"] = "long"

        out = pd.DataFrame()
        out["stay_id"] = df[stay_col]
        out["subject_id"] = (
            df[subject_col] if subject_col is not None else df[stay_col]
        )
        out["task"] = df["task"].astype(str).map(canonical_task_name)
        out["target"] = pd.to_numeric(df["target"], errors="coerce")
        out["prediction"] = pd.to_numeric(df["prediction"], errors="coerce")

        if "validity" in df.columns:
            out["validity"] = normalize_validity(df["validity"])
        else:
            out["validity"] = True

        out = out[out["task"].isin(PRIMARY_TASKS)].copy()

        if out.empty:
            audit["reason"] = "Long-form file has no recognized primary tasks."
            return None, audit

        audit["compatible"] = True
        audit["test_rows_only"] = True
        audit["split_filter_applied"] = split_filter_applied
        audit["reason"] = (
            "Supported long-form prediction schema after explicit test-row filtering."
            if split_filter_applied
            else "Supported long-form test prediction schema."
        )
        return out[
            ["stay_id", "subject_id", "task", "target", "prediction", "validity"]
        ], audit

    # ------------------------------------------------------------------
    # B. Wide format
    # ------------------------------------------------------------------
    audit["format"] = "wide"

    rows = []

    for task in PRIMARY_TASKS:
        pred_col = find_column_by_aliases(
            df, task_column_aliases(task, "prediction")
        )
        target_col = find_column_by_aliases(
            df, task_column_aliases(task, "target")
        )
        validity_col = find_column_by_aliases(
            df, task_column_aliases(task, "validity")
        )

        if pred_col is None:
            continue

        tmp = pd.DataFrame()
        tmp["stay_id"] = df[stay_col]
        tmp["subject_id"] = (
            df[subject_col] if subject_col is not None else df[stay_col]
        )
        tmp["task"] = task
        tmp["prediction"] = pd.to_numeric(df[pred_col], errors="coerce")

        if target_col is not None:
            tmp["target"] = pd.to_numeric(df[target_col], errors="coerce")
        else:
            tmp["target"] = np.nan

        if validity_col is not None:
            tmp["validity"] = normalize_validity(df[validity_col])
        else:
            tmp["validity"] = tmp["target"].notna()

        rows.append(tmp)

    if not rows:
        audit["reason"] = "No recognized task-specific prediction columns."
        return None, audit

    out = pd.concat(rows, ignore_index=True)

    if out["target"].notna().sum() == 0:
        audit["reason"] = (
            "Prediction columns found but no truth/target columns were found."
        )
        return None, audit

    audit["compatible"] = True
    audit["test_rows_only"] = True
    audit["split_filter_applied"] = split_filter_applied
    audit["reason"] = (
        "Supported wide-form prediction schema after explicit test-row filtering."
        if split_filter_applied
        else "Supported wide-form test prediction schema."
    )
    return out[
        ["stay_id", "subject_id", "task", "target", "prediction", "validity"]
    ], audit


# ==================================================================================================
# 13. STANDARDIZE CANDIDATE PREDICTIONS
# ==================================================================================================

print("\n" + "=" * 120)
print("STANDARDIZED PREDICTION SCHEMA AUDIT")
print("=" * 120)

standardized_predictions = {}
final_prediction_audits = []

for mid in candidate_ids:
    base = next(
        (x for x in prediction_audit_rows if x["model_id"] == mid),
        None,
    )

    print(f"\n{mid} | {MODEL_NAMES[mid]}")

    if mid not in prediction_paths:
        print("COMPATIBLE: NO")
        print("REASON: Prediction file not found.")
        final_prediction_audits.append(
            {
                "model_id": mid,
                "model_name": MODEL_NAMES[mid],
                "path": None,
                "compatible": False,
                "reason": "Prediction file not found.",
            }
        )
        continue

    std_df, audit = standardize_prediction_file(prediction_paths[mid])
    audit["model_id"] = mid
    audit["model_name"] = MODEL_NAMES[mid]

    if base:
        audit["test_split_status"] = base.get("test_split_status")
        audit["test_only_verified"] = base.get("test_only_verified")
        audit["los_units_status"] = base.get("los_units_status")
        audit["original_rows"] = base.get("rows")

    final_prediction_audits.append(audit)

    print("PATH:", prediction_paths[mid])
    print("FORMAT:", audit.get("format"))
    print("COMPATIBLE:", audit.get("compatible"))
    print("REASON:", audit.get("reason"))

    if std_df is not None:
        standardized_predictions[mid] = std_df
        print("STANDARDIZED ROWS:", len(std_df))
        print("TASKS:", sorted(std_df["task"].dropna().unique().tolist()))

prediction_audit_df = pd.DataFrame(final_prediction_audits)
prediction_audit_path = SEL / "FINAL_candidate_prediction_schema_audit.csv"
prediction_audit_df.to_csv(prediction_audit_path, index=False)
print(f"\nSaved: {prediction_audit_path}")


# ==================================================================================================
# 14. EXACT STAY-LEVEL PAIRING
# ==================================================================================================

def prepare_pair(df_a, df_b, task):
    a = df_a[df_a["task"].eq(task)].copy()
    b = df_b[df_b["task"].eq(task)].copy()

    if a.empty or b.empty:
        return None

    def collapse_exact_duplicates(df):
        dup_mask = df["stay_id"].duplicated(keep=False)
        if not dup_mask.any():
            return df

        for sid in df.loc[dup_mask, "stay_id"].astype(str).unique():
            subset = df[df["stay_id"].astype(str).eq(sid)]
            if len(
                subset[["target", "prediction", "validity"]].drop_duplicates()
            ) > 1:
                return None

        return df.drop_duplicates(
            subset=["stay_id", "task"], keep="first"
        )

    a = collapse_exact_duplicates(a)
    b = collapse_exact_duplicates(b)

    if a is None or b is None:
        return None

    merged = a.merge(
        b,
        on="stay_id",
        how="inner",
        suffixes=("_a", "_b"),
    )

    if merged.empty:
        return None

    truth_a = merged["target_a"]
    truth_b = merged["target_b"]

    both_finite = np.isfinite(truth_a) & np.isfinite(truth_b)
    if both_finite.any():
        if not np.array_equal(
            truth_a[both_finite].to_numpy(),
            truth_b[both_finite].to_numpy(),
        ):
            return None

    merged["target"] = merged["target_a"]
    merged["subject_id"] = merged["subject_id_a"]

    valid = (
        merged["validity_a"].astype(bool)
        & merged["validity_b"].astype(bool)
        & np.isfinite(merged["target"].to_numpy())
        & np.isfinite(merged["prediction_a"].to_numpy())
        & np.isfinite(merged["prediction_b"].to_numpy())
    )

    merged = merged[valid].copy()

    if merged.empty:
        return None

    merged = merged.drop_duplicates("stay_id", keep="first")

    return merged[
        [
            "stay_id",
            "subject_id",
            "target",
            "prediction_a",
            "prediction_b",
        ]
    ].reset_index(drop=True)


# ==================================================================================================
# 15. PATIENT-LEVEL BOOTSTRAP
# ==================================================================================================

def bootstrap_cluster_weights(subject_values, rng):
    subject_values = np.asarray(subject_values)
    unique_subjects, inverse = np.unique(
        subject_values, return_inverse=True
    )
    n_subjects = len(unique_subjects)

    sampled = rng.integers(0, n_subjects, size=n_subjects)
    counts = np.bincount(sampled, minlength=n_subjects)
    weights = counts[inverse]

    return weights


def weighted_auprc(y, pred, weights):
    try:
        valid = weights > 0
        yy = np.asarray(y)[valid]
        pp = np.asarray(pred)[valid]
        ww = np.asarray(weights)[valid]

        if len(np.unique(yy)) < 2:
            return np.nan

        return float(
            average_precision_score(
                yy, pp, sample_weight=ww
            )
        )
    except Exception:
        return np.nan


def weighted_mae(y, pred, weights):
    try:
        valid = weights > 0
        yy = np.asarray(y)[valid]
        pp = np.asarray(pred)[valid]
        ww = np.asarray(weights)[valid]

        if len(yy) == 0:
            return np.nan

        return float(
            np.average(np.abs(yy - pp), weights=ww)
        )
    except Exception:
        return np.nan


def bootstrap_pair_metrics(pair_df, reps=5000, seed=20260923):
    rng = np.random.default_rng(seed)

    y = pair_df["target"].to_numpy(dtype=float)
    pred_a = pair_df["prediction_a"].to_numpy(dtype=float)
    pred_b = pair_df["prediction_b"].to_numpy(dtype=float)
    subjects = pair_df["subject_id"].to_numpy()

    unique_subjects, inverse = np.unique(
        subjects, return_inverse=True
    )
    n_subjects = len(unique_subjects)

    if len(np.unique(y)) >= 2:
        point_a = float(average_precision_score(y, pred_a))
        point_b = float(average_precision_score(y, pred_b))
    else:
        point_a = np.nan
        point_b = np.nan

    deltas = []

    for _ in range(reps):
        sampled = rng.integers(
            0, n_subjects, size=n_subjects
        )
        counts = np.bincount(
            sampled, minlength=n_subjects
        )
        weights = counts[inverse]

        a = weighted_auprc(y, pred_a, weights)
        b = weighted_auprc(y, pred_b, weights)

        if np.isfinite(a) and np.isfinite(b):
            deltas.append(a - b)

    deltas = np.asarray(deltas, dtype=float)

    return {
        "n_rows": len(pair_df),
        "n_subjects": int(pair_df["subject_id"].nunique()),
        "bootstrap_unit": "subject_id",
        "point_a": point_a,
        "point_b": point_b,
        "point_delta_a_minus_b": (
            point_a - point_b
            if np.isfinite(point_a) and np.isfinite(point_b)
            else np.nan
        ),
        "delta_ci95_low": (
            float(np.quantile(deltas, 0.025))
            if len(deltas) else np.nan
        ),
        "delta_ci95_high": (
            float(np.quantile(deltas, 0.975))
            if len(deltas) else np.nan
        ),
        "prob_delta_gt_0": (
            float(np.mean(deltas > 0))
            if len(deltas) else np.nan
        ),
        "bootstrap_successful_reps": int(len(deltas)),
    }


# ==================================================================================================
# 16. PAIRED BOOTSTRAP
# ==================================================================================================

print("\n" + "=" * 120)
print("8. 5,000-RESAMPLE PATIENT-LEVEL PAIRED BOOTSTRAP")
print("=" * 120)

print("Unit = subject_id.")
print("Pairing = exact stay_id intersection.")
print("Classification metric = AUPRC.")
print(f"Resamples = {BOOTSTRAP_REPS}.")
print(f"Seed = {BOOTSTRAP_SEED}.")

bootstrap_pairs = [
    ("32", "00", "32 vs 00"),
    ("32", "05", "32 vs 05"),
    ("32", "31", "32 vs 31"),
    ("32", "13", "32 vs 13"),
    ("32", "24", "32 vs 24"),
    ("29", "32", "29 vs 32"),
]

bootstrap_results = []

for model_a, model_b, label in bootstrap_pairs:
    print("\n" + "-" * 100)
    print(label)
    print("-" * 100)

    if model_a not in standardized_predictions:
        bootstrap_results.append({
            "comparison": label,
            "model_a": model_a,
            "model_b": model_b,
            "task": None,
            "status": "SKIPPED",
            "reason": "Model A prediction artifact incompatible/unavailable.",
        })
        print("SKIP:", model_a)
        continue

    if model_b not in standardized_predictions:
        bootstrap_results.append({
            "comparison": label,
            "model_a": model_a,
            "model_b": model_b,
            "task": None,
            "status": "SKIPPED",
            "reason": "Model B prediction artifact incompatible/unavailable.",
        })
        print("SKIP:", model_b)
        continue

    df_a = standardized_predictions[model_a]
    df_b = standardized_predictions[model_b]

    for task_idx, task in enumerate(BINARY_TASKS):
        pair_df = prepare_pair(df_a, df_b, task)

        if pair_df is None:
            bootstrap_results.append({
                "comparison": label,
                "model_a": model_a,
                "model_b": model_b,
                "task": task,
                "status": "SKIPPED",
                "reason": "Exact paired stay-level truth/prediction table unavailable.",
            })
            print(f"{task:28s} -> NOT COMPATIBLE / NOT PAIRED")
            continue

        result = bootstrap_pair_metrics(
            pair_df,
            reps=BOOTSTRAP_REPS,
            seed=BOOTSTRAP_SEED + int(model_a) * 1000
            + int(model_b) * 10 + task_idx,
        )

        result.update({
            "comparison": label,
            "model_a": model_a,
            "model_a_name": MODEL_NAMES[model_a],
            "model_b": model_b,
            "model_b_name": MODEL_NAMES[model_b],
            "task": task,
            "status": "COMPLETED",
        })
        bootstrap_results.append(result)

        print(
            f"{task:28s} -> "
            f"n_rows={result['n_rows']} "
            f"n_subjects={result['n_subjects']} "
            f"A={result['point_a']:.6f} "
            f"B={result['point_b']:.6f} "
            f"delta={result['point_delta_a_minus_b']:.6f} "
            f"CI=[{result['delta_ci95_low']:.6f}, "
            f"{result['delta_ci95_high']:.6f}] "
            f"P(delta>0)={result['prob_delta_gt_0']:.4f}"
        )

bootstrap_df = pd.DataFrame(bootstrap_results)
bootstrap_path = SEL / "FINAL_candidate_paired_bootstrap_5000.csv"
bootstrap_df.to_csv(bootstrap_path, index=False)
print(f"\nSaved: {bootstrap_path}")


# ==================================================================================================
# 17. OPTIONAL LOS PAIRED AUDIT
# ==================================================================================================

print("\n" + "=" * 120)
print("9. OPTIONAL LOS PAIRED-METRIC AUDIT")
print("=" * 120)

los_results = []

for model_a, model_b, label in bootstrap_pairs:
    if model_a not in standardized_predictions or model_b not in standardized_predictions:
        continue

    # Require explicit raw-hours evidence for both artifacts.
    statuses = {}
    for mid in [model_a, model_b]:
        p = prediction_paths.get(mid)
        if p is None:
            statuses[mid] = "MISSING"
            continue
        try:
            df_tmp = pd.read_csv(p)
            statuses[mid] = los_units_status(df_tmp)[0]
        except Exception:
            statuses[mid] = "READ_ERROR"

    if statuses.get(model_a) != "RAW_HOURS" or statuses.get(model_b) != "RAW_HOURS":
        print(
            f"{label:12s} -> LOS pairing SKIPPED: "
            f"raw-hour units not explicitly verified; statuses={statuses}"
        )
        continue

    pair_df = prepare_pair(
        standardized_predictions[model_a],
        standardized_predictions[model_b],
        "los_hours_24h",
    )

    if pair_df is None:
        print(f"{label:12s} -> LOS pairing unavailable")
        continue

    y = pair_df["target"].to_numpy(dtype=float)
    pa = pair_df["prediction_a"].to_numpy(dtype=float)
    pb = pair_df["prediction_b"].to_numpy(dtype=float)
    subjects = pair_df["subject_id"].to_numpy()

    rng = np.random.default_rng(
        BOOTSTRAP_SEED + int(model_a) * 100 + int(model_b)
    )

    point_a = float(mean_absolute_error(y, pa))
    point_b = float(mean_absolute_error(y, pb))

    unique_subjects, inverse = np.unique(
        subjects, return_inverse=True
    )
    n_subjects = len(unique_subjects)

    deltas = []

    for _ in range(BOOTSTRAP_REPS):
        sampled = rng.integers(
            0, n_subjects, size=n_subjects
        )
        counts = np.bincount(
            sampled, minlength=n_subjects
        )
        weights = counts[inverse]

        mae_a = weighted_mae(y, pa, weights)
        mae_b = weighted_mae(y, pb, weights)

        if np.isfinite(mae_a) and np.isfinite(mae_b):
            deltas.append(mae_a - mae_b)

    deltas = np.asarray(deltas, dtype=float)

    if len(deltas):
        row = {
            "comparison": label,
            "model_a": model_a,
            "model_a_name": MODEL_NAMES[model_a],
            "model_b": model_b,
            "model_b_name": MODEL_NAMES[model_b],
            "task": "los_hours_24h",
            "n_rows": len(pair_df),
            "n_subjects": int(pair_df["subject_id"].nunique()),
            "bootstrap_unit": "subject_id",
            "los_units_verified": "raw_hours",
            "point_mae_a": point_a,
            "point_mae_b": point_b,
            "point_delta_a_minus_b": point_a - point_b,
            "delta_ci95_low": float(np.quantile(deltas, 0.025)),
            "delta_ci95_high": float(np.quantile(deltas, 0.975)),
            "prob_delta_lt_0": float(np.mean(deltas < 0)),
            "bootstrap_successful_reps": len(deltas),
            "status": "COMPLETED",
        }
        los_results.append(row)

        print(
            f"{label:12s} -> "
            f"MAE_A={point_a:.4f} "
            f"MAE_B={point_b:.4f} "
            f"delta(A-B)={point_a-point_b:.4f} "
            f"CI=[{row['delta_ci95_low']:.4f}, "
            f"{row['delta_ci95_high']:.4f}]"
        )
    else:
        print(f"{label:12s} -> LOS bootstrap produced no valid replicates")

los_df = pd.DataFrame(los_results)
los_path = SEL / "FINAL_candidate_los_paired_bootstrap_5000.csv"
los_df.to_csv(los_path, index=False)
print(f"\nSaved: {los_path}")


# ==================================================================================================
# 18. FINAL CONSISTENCY CHECKS
# ==================================================================================================

print("\n" + "=" * 120)
print("10. FINAL AUDIT CONSISTENCY CHECKS")
print("=" * 120)

candidate_missing = sorted(
    set(candidate_ids) - set(rank["model_id"])
)

print(
    "Candidate models present:",
    "PASS ✅" if not candidate_missing
    else f"FAIL ❌ missing={candidate_missing}",
)

print(
    "Model 29 GPU/BF16 provenance:",
    "PASS ✅" if model29_gpu_pass
    else "FAIL / INCOMPLETE ❌",
)


# ------------------------------------------------------------------
# Strict leakage status:
#   TRUE  -> FAIL
#   FALSE -> PASS
#   missing/unknown -> INCOMPLETE
# ------------------------------------------------------------------

leakage_fail = []
leakage_incomplete = []
leakage_values = {}

for mid in candidate_ids:
    meta_files = find_final_metadata(mid)

    if not meta_files:
        leakage_incomplete.append(mid)
        leakage_values[mid] = None
        continue

    values = []
    for p in meta_files:
        try:
            values.append(read_json(p).get("test_used_for_selection"))
        except Exception:
            pass

    values = list(dict.fromkeys(values))

    if True in values:
        leakage_fail.append(mid)
        leakage_values[mid] = True
    elif values and all(v is False for v in values):
        leakage_values[mid] = False
    else:
        leakage_incomplete.append(mid)
        leakage_values[mid] = values


if leakage_fail:
    leakage_status = "FAIL"
elif leakage_incomplete:
    leakage_status = "INCOMPLETE"
else:
    leakage_status = "PASS"

print("Candidate test-used-for-selection check:", leakage_status)
print("  FAIL models:", leakage_fail if leakage_fail else "NONE")
print("  INCOMPLETE models:", leakage_incomplete if leakage_incomplete else "NONE")


compatible_prediction_models = sorted(standardized_predictions.keys())

print(
    "Compatible candidate prediction files:",
    ", ".join(compatible_prediction_models)
    if compatible_prediction_models else "NONE",
)

completed_bootstrap = int(
    (
        bootstrap_df["status"].astype(str).eq("COMPLETED").sum()
        if not bootstrap_df.empty else 0
    )
)

print(
    "Completed task-level paired bootstrap comparisons:",
    completed_bootstrap,
)

print(
    "LOS paired comparisons completed:",
    int(len(los_df)),
)


# ==================================================================================================
# 19. SAVE FINAL SUMMARY
# ==================================================================================================

candidate_summary_path = SEL / "FINAL_candidate_summary_audit.csv"

candidate_summary = (
    rank[rank["model_id"].isin(candidate_ids)]
    .copy()
    .sort_values("mean_binary_auprc", ascending=False)
)

candidate_summary.to_csv(candidate_summary_path, index=False)


audit_summary = {
    "benchmark_coverage": {
        "expected_model_count": len(EXPECTED_MODEL_IDS),
        "found_model_count": len(found_rank_ids),
        "missing_models": missing_rank_ids,
        "unexpected_models": unexpected_rank_ids,
        "pass": not missing_rank_ids and not unexpected_rank_ids,
    },
    "candidate_ids": candidate_ids,
    "model_29_gpu_bf16_provenance_pass": model29_gpu_pass,
    "candidate_test_selection_leakage": {
        "status": leakage_status,
        "fail_models": leakage_fail,
        "incomplete_models": leakage_incomplete,
        "values": leakage_values,
    },
    "compatible_prediction_models": compatible_prediction_models,
    "bootstrap_definition": {
        "reps": BOOTSTRAP_REPS,
        "seed": BOOTSTRAP_SEED,
        "pairing": "exact stay_id",
        "bootstrap_unit": "subject_id",
    },
    "completed_bootstrap_task_results": completed_bootstrap,
    "completed_los_bootstrap_comparisons": int(len(los_df)),
    "los_policy": (
        "Only run paired LOS bootstrap when prediction/target units are "
        "explicitly verified as raw hours."
    ),
    "files": {
        "candidate_summary": str(candidate_summary_path),
        "candidate_task_metrics": str(candidate_task_path),
        "prediction_schema_audit": str(prediction_audit_path),
        "paired_bootstrap": str(bootstrap_path),
        "los_bootstrap": str(los_path),
    },
}

audit_json_path = SEL / "FINAL_candidate_decision_audit.json"

with audit_json_path.open("w", encoding="utf-8") as f:
    json.dump(audit_summary, f, indent=2, default=str)


# ==================================================================================================
# 20. FINAL STATUS
# ==================================================================================================

print("\n" + "=" * 120)
print("AUDIT 2 COMPLETE")
print("=" * 120)

print("\nSaved files:")
print(f"  {candidate_summary_path}")
print(f"  {candidate_task_path}")
print(f"  {prediction_audit_path}")
print(f"  {bootstrap_path}")
print(f"  {los_path}")
print(f"  {audit_json_path}")

print("\nIMPORTANT THESIS DECISION STATUS:")
print("  1. The 33-model benchmark must be complete before the proposed model is locked.")
print("  2. Model 29's old CPU result must not be used for benchmark ranking.")
print("  3. The corrected GPU/BF16 Model 29 result is the authoritative Model 29 result.")
print("  4. Deterioration remains a derived evaluation output, not a trained primary head.")
print("  5. LOS MAE remains monitored/report-only and is not the model-selection criterion.")
print("  6. Paired bootstrap results are reported only when exact stay pairing and compatible")
print("     truth/prediction artifacts are available.")
print("  7. LOS paired bootstrap is reported only when raw-hour units are explicitly verified.")
print("  8. Missing test-selection metadata is INCOMPLETE, not PASS.")

print(
    "\nDo not lock the proposed thesis architecture solely from this script if "
    "prediction-schema, test-split, leakage, or paired-bootstrap audits are incomplete."
)

print("\n" + "=" * 120)
print("END OF SCRIPT 2")
print("=" * 120)
