
# ==================================================================================================
# LOS UNIT + MODEL 26 COMPUTE PROVENANCE AUDIT
# ==================================================================================================
#
# Purpose
# -------
# Focused follow-up audit after Script 2:
#
# A) LOS paired audit
#    - Inspect saved prediction artifacts for candidate models and the refreshed Model 26.
#    - Filter split == test and task == los_hours_24h.
#    - Recompute raw MAE directly from saved target/prediction columns.
#    - Compare against canonical LOS MAE from Script 1 (including refreshed Model 26).
#    - Test whether predictions/targets appear to be:
#         1) raw hours
#         2) log1p(hours)
#         3) frozen V2 shifted transform: log1p(max(los_hours - 24, 0))
#            with inverse expm1(x) + 24
#    - Never silently relabel units; report EVIDENCE / INCONCLUSIVE.
#
# B) Model 26 provenance
#    - Find all Model 26 tuned/final metadata/config files.
#    - Report device/GPU/AMP/trials/selection flags.
#    - Search Model 26 directories for additional provenance/log artifacts.
#    - Identify whether available evidence supports CPU, CUDA, or INCONCLUSIVE.
#    - Check whether the selected Model 26 result used test data for selection.
#
# This script does NOT retrain anything and does NOT alter benchmark metrics.
#
# ==================================================================================================

from pathlib import Path
import json
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")

ROOT = Path("outputs")
SEL = ROOT / "benchmark_selection"
SEL.mkdir(parents=True, exist_ok=True)

MODEL_IDS = ["32", "31", "00", "05", "13", "24", "26", "29"]
MODEL_26_ID = "26"

MODEL_NAMES = {
    "00": "XGBoost",
    "05": "LightGBM",
    "13": "RETAIN",
    "24": "S4D",
    "26": "Mamba-2",
    "29": "Multimodal Progression",
    "31": "Gated Fusion/Cross-Attn",
    "32": "MoE Modality Fusion",
}

CANONICAL_LOS = {
    "00": 63.381006,
    "05": 63.065625,
    "13": 56.917704,
    "24": 56.652037,
    "26": 57.05433760165303,
    "29": 59.610515,
    "31": 56.467099,
    "32": 56.678587,
}

EXPECTED_LOS_TEST_ROWS = 14887
TOLERANCE_HOURS = 0.01

# Canonical LOS MAE values above are copied from the completed benchmark
# audit output. This script only compares saved test predictions against
# those frozen values; it does not recompute or replace benchmark statistics.


# ==================================================================================================
# HELPERS
# ==================================================================================================

def read_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def file_model_match(path: Path, model_id: str) -> bool:
    normalized = str(path).replace("\\", "/")
    pattern = rf"(?:^|/){re.escape(model_id)}_[^/]+(?:/|$)"
    return re.search(pattern, normalized) is not None


def find_prediction_file(model_id):
    candidates = []

    for p in ROOT.rglob("test_predictions.csv"):
        normalized = str(p).replace("\\", "/").lower()

        if not file_model_match(p, model_id):
            continue

        if "tuned" not in normalized:
            continue

        if "cpu_audit" in normalized:
            continue

        candidates.append(p)

    candidates.sort(
        key=lambda p: (
            "final_selected_configuration" not in str(p),
            len(str(p)),
            str(p),
        )
    )

    return candidates[0] if candidates else None


def normalize_split(df):
    if "split" not in df.columns:
        return None
    return (
        df["split"]
        .astype(str)
        .str.strip()
        .str.lower()
    )


def summarize_numeric(s):
    """
    Accept pandas Series, NumPy arrays, lists, or other array-like inputs.
    """
    x = pd.to_numeric(pd.Series(np.asarray(s).reshape(-1)), errors="coerce")
    x = x.dropna().to_numpy(dtype=float)

    if len(x) == 0:
        return {
            "n": 0,
            "min": np.nan,
            "p01": np.nan,
            "p50": np.nan,
            "p99": np.nan,
            "max": np.nan,
            "mean": np.nan,
        }

    return {
        "n": len(x),
        "min": float(np.min(x)),
        "p01": float(np.quantile(x, 0.01)),
        "p50": float(np.quantile(x, 0.50)),
        "p99": float(np.quantile(x, 0.99)),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
    }


def mae_if_finite(y, p):
    mask = np.isfinite(y) & np.isfinite(p)
    if mask.sum() == 0:
        return np.nan, 0
    return float(mean_absolute_error(y[mask], p[mask])), int(mask.sum())


# ==================================================================================================
# 1. LOS PREDICTION UNIT AUDIT
# ==================================================================================================

print("\n" + "=" * 120)
print("A. LOS UNIT / PAIRED-MAE AUDIT")
print("=" * 120)

los_rows = []

for mid in MODEL_IDS:
    p = find_prediction_file(mid)

    print("\n" + "-" * 100)
    print(f"{mid} — {MODEL_NAMES[mid]}")
    print("-" * 100)

    if p is None:
        print("Prediction file: NOT FOUND")
        los_rows.append({
            "model_id": mid,
            "model_name": MODEL_NAMES[mid],
            "status": "INCOMPLETE",
            "reason": "Prediction file not found.",
        })
        continue

    print("Prediction file:", p)

    try:
        df = pd.read_csv(p)

        if "split" not in df.columns:
            print("STATUS: INCOMPLETE — no split column")
            los_rows.append({
                "model_id": mid,
                "model_name": MODEL_NAMES[mid],
                "path": str(p),
                "status": "INCOMPLETE",
                "reason": "No explicit split column.",
            })
            continue

        split = normalize_split(df)
        test = df.loc[split.eq("test")].copy()

        print("Original rows:", len(df))
        print("Test rows:", len(test))

        if "task" not in test.columns:
            print("STATUS: INCOMPLETE — no task column")
            los_rows.append({
                "model_id": mid,
                "model_name": MODEL_NAMES[mid],
                "path": str(p),
                "status": "INCOMPLETE",
                "reason": "No task column.",
            })
            continue

        los = test[
            test["task"]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("los_hours_24h")
        ].copy()

        print("LOS test rows:", len(los))

        if "target" not in los.columns or "prediction" not in los.columns:
            print("STATUS: INCOMPLETE — target/prediction missing")
            los_rows.append({
                "model_id": mid,
                "model_name": MODEL_NAMES[mid],
                "path": str(p),
                "status": "INCOMPLETE",
                "reason": "LOS target/prediction columns missing.",
            })
            continue

        y = pd.to_numeric(los["target"], errors="coerce").to_numpy(dtype=float)
        pred = pd.to_numeric(los["prediction"], errors="coerce").to_numpy(dtype=float)

        finite = np.isfinite(y) & np.isfinite(pred)
        y = y[finite]
        pred = pred[finite]

        print("\nTARGET SUMMARY:")
        print(json.dumps(summarize_numeric(y), indent=2))

        print("\nPREDICTION SUMMARY:")
        print(json.dumps(summarize_numeric(pred), indent=2))

        raw_mae, n_raw = mae_if_finite(y, pred)

        # Candidate transform interpretations.
        pred_expm1 = np.expm1(np.clip(pred, -50, 50))
        pred_shifted = np.expm1(np.clip(pred, -50, 50)) + 24.0

        target_expm1 = np.expm1(np.clip(y, -50, 50))
        target_shifted = np.expm1(np.clip(y, -50, 50)) + 24.0

        mae_exp_pred = np.nan
        mae_shifted_pred = np.nan
        mae_exp_target = np.nan
        mae_shifted_target = np.nan

        # If target is raw hours, compare inverse transforms of prediction.
        if np.nanmedian(y) > 10:
            mae_exp_pred, _ = mae_if_finite(y, pred_expm1)
            mae_shifted_pred, _ = mae_if_finite(y, pred_shifted)

        # If target itself is transformed, compare transformed target to raw or inverse pred.
        mae_target_log_to_pred, _ = mae_if_finite(target_expm1, pred)
        mae_target_shifted_to_pred, _ = mae_if_finite(target_shifted, pred)

        canonical = CANONICAL_LOS.get(mid, np.nan)

        comparisons = {
            "raw_pred_vs_raw_target": raw_mae,
            "expm1_pred_vs_raw_target": mae_exp_pred,
            "expm1_plus_24_pred_vs_raw_target": mae_shifted_pred,
            "expm1_target_vs_prediction": mae_target_log_to_pred,
            "expm1_target_plus_24_vs_prediction": mae_target_shifted_to_pred,
        }

        print("\nMAE INTERPRETATION CHECKS:")
        for k, v in comparisons.items():
            print(f"  {k}: {v}")

        print("\nCANONICAL LOS MAE:", canonical)

        # Primary evidence:
        # direct raw prediction MAE should match the canonical test LOS MAE
        # if the saved prediction column is already in raw hours.
        direct_diff = (
            abs(raw_mae - canonical)
            if np.isfinite(raw_mae) and np.isfinite(canonical)
            else np.nan
        )

        if np.isfinite(direct_diff) and direct_diff <= TOLERANCE_HOURS:
            unit_status = "SUPPORTED_RAW_HOURS"
            reason = (
                "Direct test-set MAE from saved prediction column matches "
                "the canonical LOS MAE within tolerance; saved prediction "
                "is therefore evidenced to be in raw hours."
            )
        else:
            # Check whether an inverse transform reproduces the canonical MAE.
            transform_candidates = {
                "expm1_pred_vs_raw_target": mae_exp_pred,
                "expm1_plus_24_pred_vs_raw_target": mae_shifted_pred,
                "raw_pred_vs_raw_target": raw_mae,
            }

            matching = [
                name for name, value in transform_candidates.items()
                if np.isfinite(value)
                and np.isfinite(canonical)
                and abs(value - canonical) <= TOLERANCE_HOURS
            ]

            if matching:
                unit_status = "TRANSFORM_EVIDENCE"
                reason = (
                    "A transform of the saved prediction column reproduces "
                    "the canonical LOS MAE; inspect the matching transform "
                    "before reporting units."
                )
            else:
                unit_status = "INCONCLUSIVE"
                reason = (
                    "No tested interpretation reproduces canonical LOS MAE "
                    "within tolerance."
                )

        print("\nUNIT STATUS:", unit_status)
        print("REASON:", reason)

        los_rows.append({
            "model_id": mid,
            "model_name": MODEL_NAMES[mid],
            "path": str(p),
            "original_rows": len(df),
            "test_rows": len(test),
            "los_test_rows": len(los),
            "finite_pairs": len(y),
            "raw_mae": raw_mae,
            "canonical_los_mae": canonical,
            "direct_mae_abs_diff": direct_diff,
            "mae_expm1_prediction": mae_exp_pred,
            "mae_expm1_plus24_prediction": mae_shifted_pred,
            "mae_expm1_target_vs_prediction": mae_target_log_to_pred,
            "mae_expm1_plus24_target_vs_prediction": mae_target_shifted_to_pred,
            "target_p50": summarize_numeric(y)["p50"],
            "target_p99": summarize_numeric(y)["p99"],
            "prediction_p50": summarize_numeric(pred)["p50"],
            "prediction_p99": summarize_numeric(pred)["p99"],
            "unit_status": unit_status,
            "reason": reason,
        })

    except Exception as e:
        print("READ/ANALYSIS ERROR:", e)
        los_rows.append({
            "model_id": mid,
            "model_name": MODEL_NAMES[mid],
            "path": str(p),
            "status": "ERROR",
            "reason": str(e),
        })

los_audit_df = pd.DataFrame(los_rows)
los_audit_path = SEL / "FINAL_los_unit_audit.csv"
los_audit_df.to_csv(los_audit_path, index=False)

print("\nSaved LOS audit:")
print(los_audit_path)


# ==================================================================================================
# 2. LOS CROSS-MODEL CONSISTENCY
# ==================================================================================================

print("\n" + "=" * 120)
print("LOS CROSS-MODEL CONSISTENCY")
print("=" * 120)

if not los_audit_df.empty:
    display_cols = [
        c for c in [
            "model_id",
            "model_name",
            "los_test_rows",
            "raw_mae",
            "canonical_los_mae",
            "direct_mae_abs_diff",
            "unit_status",
        ]
        if c in los_audit_df.columns
    ]

    print(
        los_audit_df[display_cols].to_string(index=False)
    )

    if "unit_status" in los_audit_df.columns:
        raw_supported = los_audit_df[
            los_audit_df["unit_status"].eq("SUPPORTED_RAW_HOURS")
        ]

        print(
            "\nModels directly supported as raw-hour predictions:",
            ", ".join(raw_supported["model_id"].astype(str).tolist())
            if not raw_supported.empty else "NONE"
        )
    else:
        print(
            "\nNo completed LOS unit analyses were available; "
            "continue to Model 26 provenance."
        )


# ==================================================================================================
# 3. MODEL 26 PROVENANCE
# ==================================================================================================

print("\n" + "=" * 120)
print("B. MODEL 26 — MAMBA-2 COMPUTE PROVENANCE AUDIT")
print("=" * 120)

model26_root_candidates = []

for p in ROOT.rglob("*"):
    s = str(p).replace("\\", "/")

    if file_model_match(p, MODEL_26_ID):
        model26_root_candidates.append(p)

# Deduplicate by resolved string without assuming files exist after symlinks.
unique_paths = {}
for p in model26_root_candidates:
    unique_paths[str(p)] = p

model26_root_candidates = sorted(
    unique_paths.values(),
    key=lambda p: (len(str(p)), str(p))
)

print("Model 26-related paths discovered:", len(model26_root_candidates))

metadata_paths = []
params_paths = []
log_paths = []

for p in model26_root_candidates:
    if p.is_file():
        name = p.name.lower()
        if name == "metadata.json":
            metadata_paths.append(p)
        elif name == "best_params.json":
            params_paths.append(p)
        elif any(
            token in name
            for token in [
                ".log",
                "log.txt",
                "stdout",
                "stderr",
                "console",
                "train",
                "run",
            ]
        ):
            log_paths.append(p)

print("\nMetadata files:")
for p in metadata_paths:
    print(" ", p)

print("\nBest-params files:")
for p in params_paths:
    print(" ", p)

print("\nPotential log/provenance files:")
for p in log_paths[:100]:
    print(" ", p)

model26_metadata_rows = []

for p in metadata_paths:
    try:
        obj = read_json(p)

        row = {
            "path": str(p),
            "device": obj.get("device"),
            "gpu_name": obj.get("gpu_name"),
            "amp_mode": obj.get("amp_mode"),
            "validation_selection_score": obj.get("validation_selection_score"),
            "best_epoch": obj.get("best_epoch"),
            "n_optuna_trials_target": obj.get("n_optuna_trials_target"),
            "n_optuna_trials_completed": obj.get("n_optuna_trials_completed"),
            "n_optuna_trials": obj.get("n_optuna_trials"),
            "test_used_for_selection": obj.get("test_used_for_selection"),
            "run_params": json.dumps(obj.get("run_params"), default=str),
        }

        model26_metadata_rows.append(row)

        print("\nMETADATA:", p)
        print(json.dumps(
            {
                k: v for k, v in row.items()
                if k not in {"path", "run_params"}
            },
            indent=2,
            default=str,
        ))

        if obj.get("run_params") is not None:
            print("run_params:")
            print(json.dumps(obj.get("run_params"), indent=2, default=str))

    except Exception as e:
        print("\nMetadata read error:", p, e)

model26_meta_df = pd.DataFrame(model26_metadata_rows)
model26_meta_path = SEL / "FINAL_model26_metadata_audit.csv"
model26_meta_df.to_csv(model26_meta_path, index=False)

print("\nSaved Model 26 metadata audit:")
print(model26_meta_path)


# ==================================================================================================
# 4. MODEL 26 CONFIG + ARTIFACT INVENTORY
# ==================================================================================================

print("\n" + "=" * 120)
print("MODEL 26 ARTIFACT INVENTORY")
print("=" * 120)

inventory_rows = []

for p in model26_root_candidates:
    if p.is_file():
        try:
            stat = p.stat()
            inventory_rows.append({
                "path": str(p),
                "name": p.name,
                "suffix": p.suffix,
                "size_bytes": stat.st_size,
                "mtime_epoch": stat.st_mtime,
            })
        except Exception:
            pass

inventory_df = pd.DataFrame(inventory_rows)

if not inventory_df.empty:
    inventory_df = inventory_df.sort_values(
        ["name", "path"]
    )

print(
    inventory_df.to_string(index=False)
    if not inventory_df.empty
    else "No Model 26 files discovered."
)

inventory_path = SEL / "FINAL_model26_artifact_inventory.csv"
inventory_df.to_csv(inventory_path, index=False)

print("\nSaved inventory:")
print(inventory_path)


# ==================================================================================================
# 5. MODEL 26 PROVENANCE CONCLUSION
# ==================================================================================================

print("\n" + "=" * 120)
print("MODEL 26 PROVENANCE CONCLUSION")
print("=" * 120)

provenance_status = "INCONCLUSIVE"
provenance_reason = "No sufficient Model 26 provenance evidence found."

if not model26_meta_df.empty:
    devices = (
        model26_meta_df["device"]
        .dropna()
        .astype(str)
        .str.lower()
        .unique()
        .tolist()
    )

    gpus = (
        model26_meta_df["gpu_name"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    test_flags = model26_meta_df["test_used_for_selection"].dropna().tolist()

    cuda_evidence = any(d == "cuda" for d in devices)
    cpu_evidence = any(
        ("cpu" in d) or (d == "cpu") for d in devices
    )

    if cuda_evidence and not cpu_evidence:
        provenance_status = "CUDA_SUPPORTED"
        provenance_reason = (
            "Final/selected metadata contains CUDA device evidence and no "
            "CPU device evidence."
        )
    elif cpu_evidence and not cuda_evidence:
        provenance_status = "CPU_SUPPORTED"
        provenance_reason = (
            "Final/selected metadata contains CPU device evidence and no "
            "CUDA device evidence."
        )
    elif cuda_evidence and cpu_evidence:
        provenance_status = "MIXED_ARTIFACTS"
        provenance_reason = (
            "Both CPU and CUDA metadata exist. The selected final artifact "
            "must be identified explicitly before claiming compute provenance."
        )

    if True in test_flags:
        provenance_reason += " WARNING: test_used_for_selection=True found."
    elif test_flags and all(v is False for v in test_flags):
        provenance_reason += " All recorded test_used_for_selection flags are False."

print("STATUS:", provenance_status)
print("REASON:", provenance_reason)

# Final human-readable summary JSON.
summary = {
    "los_audit": {
        "path": str(los_audit_path),
        "models_audited": MODEL_IDS,
        "raw_hour_supported_models": (
            los_audit_df.loc[
                los_audit_df["unit_status"].eq("SUPPORTED_RAW_HOURS"),
                "model_id",
            ].astype(str).tolist()
            if "unit_status" in los_audit_df.columns
            else []
        ),
        "completed_rows": int(len(los_audit_df)),
    },
    "model26": {
        "provenance_status": provenance_status,
        "provenance_reason": provenance_reason,
        "metadata_audit": str(model26_meta_path),
        "artifact_inventory": str(inventory_path),
    },
}

summary_path = SEL / "FINAL_los_and_model26_provenance_summary.json"

with summary_path.open("w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, default=str)

print("\nSaved summary:")
print(summary_path)

print("\n" + "=" * 120)
print("END OF LOS + MODEL 26 PROVENANCE AUDIT")
print("=" * 120)
