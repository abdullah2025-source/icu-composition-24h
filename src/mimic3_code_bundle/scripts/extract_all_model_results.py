#!/usr/bin/env python3
"""
One-pass documentation extractor for every completed CHI result.

Uses the real output tree. Does not retrain or retune.
Primary rows are tuned and fixed baselines, proposed models, Family Y,
Family Z, Z4, and Z2 Ultimate. Seed caches and Z3 trial folders are skipped.

Binary metrics: AUPRC, AUROC, Brier, ECE, prevalence, precision/recall/F1/
accuracy and confusion counts at 0.5.
LOS metrics: MAE, RMSE, median AE, R2.

Brier, ECE, and the 0.5 threshold are only meaningful when prediction is a
probability. Rank-blend rows (A3, Z1, Z2, Y0) are included for documentation,
but those calibration numbers must not be treated as calibrated probabilities.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


PROJECT_ROOT = Path("/home/abdullah2025/fyp_project")
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
DOC_ROOT = OUTPUT_ROOT / "documentation"

BINARY_TASKS = (
    "mortality_24h",
    "ventilation_onset_24h",
    "vasopressor_onset_24h",
    "readmission_30d_24h",
)
LOS_TASK = "los_hours_24h"
TASKS = BINARY_TASKS + (LOS_TASK,)
METRIC_BINARY = (
    "auprc", "auroc", "brier", "ece", "prevalence",
    "precision", "recall", "f1", "accuracy",
    "tn", "fp", "fn", "tp", "n",
)
METRIC_LOS = ("mae", "rmse", "median_ae", "r2", "n")


def _ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    if len(y) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    n = len(y)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p <= hi if hi == 1.0 else p < hi)
        if not np.any(m):
            continue
        total += (m.sum() / n) * abs(float(p[m].mean()) - float(y[m].mean()))
    return float(total)


def _threshold_metrics(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict:
    empty = {k: float("nan") for k in ("precision", "recall", "f1", "accuracy", "tn", "fp", "fn", "tp")}
    if len(y) == 0:
        return empty
    pred = (np.clip(p, 0.0, 1.0) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y.astype(int), pred, labels=[0, 1]).ravel()
    return {
        "precision": float(precision_score(y.astype(int), pred, zero_division=0)),
        "recall": float(recall_score(y.astype(int), pred, zero_division=0)),
        "f1": float(f1_score(y.astype(int), pred, zero_division=0)),
        "accuracy": float(accuracy_score(y.astype(int), pred)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def score_frame(df: pd.DataFrame, split: str) -> dict:
    out = {f"{split}_n_rows": 0}
    if df is None or df.empty or "target" not in df.columns or "prediction" not in df.columns:
        return out
    z = df
    if "split" in z.columns:
        z = z[z["split"].astype(str) == split]
    if "task" not in z.columns:
        return out
    out[f"{split}_n_rows"] = int(len(z))

    aps, aucs, briers, eces = [], [], [], []
    for task in TASKS:
        zt = z[z["task"].astype(str) == task]
        if "validity" in zt.columns:
            zt = zt[zt["validity"].astype(bool)]
        y = zt["target"].to_numpy(float) if len(zt) else np.array([])
        p = zt["prediction"].to_numpy(float) if len(zt) else np.array([])
        keep = np.isfinite(y) & np.isfinite(p)
        y, p = y[keep], p[keep]
        n = int(len(y))
        out[f"{split}_{task}_n"] = n

        if task in BINARY_TASKS:
            if n == 0 or len(np.unique(y)) < 2:
                for metric in METRIC_BINARY:
                    out[f"{split}_{task}_{metric}"] = float("nan")
                aps.append(float("nan"))
                aucs.append(float("nan"))
                briers.append(float("nan"))
                eces.append(float("nan"))
                continue
            y_int = y.astype(int)
            p_clip = np.clip(p, 0.0, 1.0)
            ap = float(average_precision_score(y_int, p))
            auc = float(roc_auc_score(y_int, p))
            brier = float(brier_score_loss(y_int, p_clip))
            ece = _ece(y_int, p_clip)
            thr = _threshold_metrics(y, p, 0.5)
            out[f"{split}_{task}_auprc"] = ap
            out[f"{split}_{task}_auroc"] = auc
            out[f"{split}_{task}_brier"] = brier
            out[f"{split}_{task}_ece"] = ece
            out[f"{split}_{task}_prevalence"] = float(y.mean())
            for key, value in thr.items():
                out[f"{split}_{task}_{key}"] = value
            aps.append(ap)
            aucs.append(auc)
            briers.append(brier)
            eces.append(ece)
        else:
            if n == 0:
                for metric in METRIC_LOS:
                    out[f"{split}_{task}_{metric}"] = float("nan")
                continue
            mae = float(mean_absolute_error(y, p))
            out[f"{split}_{task}_mae"] = mae
            out[f"{split}_{task}_rmse"] = float(np.sqrt(mean_squared_error(y, p)))
            out[f"{split}_{task}_median_ae"] = float(median_absolute_error(y, p))
            out[f"{split}_{task}_r2"] = float(r2_score(y, p)) if n > 1 else float("nan")

    out[f"{split}_mean_binary_auprc"] = float(np.nanmean(aps)) if aps else float("nan")
    out[f"{split}_mean_binary_auroc"] = float(np.nanmean(aucs)) if aucs else float("nan")
    out[f"{split}_mean_binary_brier"] = float(np.nanmean(briers)) if briers else float("nan")
    out[f"{split}_mean_binary_ece"] = float(np.nanmean(eces)) if eces else float("nan")
    return out


def read_pred(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "task" not in df.columns or "prediction" not in df.columns:
        raise ValueError(f"not a task prediction file: {path}")
    return df


def add_row(rows, *, name, family, stage, test_path=None, val_path=None):
    rec = {
        "name": name,
        "family": family,
        "stage": stage,
        "test_file": str(test_path.relative_to(PROJECT_ROOT)) if test_path else "",
        "val_file": str(val_path.relative_to(PROJECT_ROOT)) if val_path else "",
    }
    try:
        if test_path and test_path.exists():
            test_df = read_pred(test_path)
            rec.update(score_frame(test_df, "test"))
            if not val_path and "split" in test_df.columns and (test_df["split"].astype(str) == "val").any():
                rec.update(score_frame(test_df, "val"))
        if val_path and val_path.exists():
            rec.update(score_frame(read_pred(val_path), "val"))
    except Exception as exc:
        rec["error"] = str(exc)
        print(f"[SKIP] {name}: {exc}")
        return
    rows.append(rec)
    print(
        f"[OK] {family:22} {stage:16} {name:42} "
        f"val={rec.get('val_mean_binary_auprc', float('nan'))} "
        f"test={rec.get('test_mean_binary_auprc', float('nan'))}"
    )


def baseline_rows(rows):
    for root in sorted(OUTPUT_ROOT.glob("baseline_family_*")):
        for stage in ("tuned", "fixed"):
            stage_root = root / stage
            if not stage_root.exists():
                continue
            for model_dir in sorted(p for p in stage_root.iterdir() if p.is_dir()):
                if stage == "tuned":
                    final = model_dir / "final_selected_configuration"
                    if not final.exists():
                        continue
                    test_path = final / "test_predictions.csv"
                else:
                    test_path = model_dir / "test_predictions.csv"
                    if not test_path.exists():
                        test_path = model_dir / "predictions.csv"
                if test_path.exists():
                    add_row(rows, name=model_dir.name, family=root.name, stage=stage, test_path=test_path)


def proposed_rows(rows):
    root = OUTPUT_ROOT / "proposed_model_family"
    if root.exists():
        for model_dir in sorted((root / "fixed").glob("*")):
            test_path = model_dir / "test_predictions.csv"
            if test_path.exists():
                add_row(rows, name=f"proposed_{model_dir.name}", family="proposed_model_family", stage="fixed", test_path=test_path)
        tuned = root / "tuned" / "E_Full-E" / "final_selected_configuration" / "test_predictions.csv"
        if tuned.exists():
            add_row(rows, name="E_Full-E", family="proposed_model_family", stage="tuned", test_path=tuned)
    single = OUTPUT_ROOT / "proposed_model" / "fixed" / "final_selected_configuration" / "test_predictions.csv"
    if single.exists():
        add_row(rows, name="proposed_model_fixed", family="proposed_model", stage="fixed", test_path=single)


def pair_family(rows, folder: str, family: str, names: list[str], stage: str):
    root = OUTPUT_ROOT / folder
    for name in names:
        test_path = root / f"{name}_test_predictions.csv"
        val_path = root / f"{name}_validation_predictions.csv"
        if test_path.exists() or val_path.exists():
            add_row(
                rows,
                name=name,
                family=family,
                stage=stage,
                test_path=test_path if test_path.exists() else None,
                val_path=val_path if val_path.exists() else None,
            )


def y_rows(rows):
    pair_family(rows, "model_y_family", "family_y", [
        "A1_best_single", "A2_probability_mean_max3", "A3_rank_max5",
        "A4_gain_only_max3", "A5_error_correlation_max3",
        "A6_weighted_rank_same_members", "A7_los_hybrid_validation_gated",
        "Y0_rank_max3", "U1_multiseed_pool", "U2_focal_pool",
        "U3_clinical_tree_pool", "U1_U2_U3_combined_pool",
        "C0_Y0", "C1_Y0_U1", "C2_Y0_U1_U2", "C3_Y0_U1_U2_U3",
        "C4_Y0_U1_U2_U3_U4", "C5_Y0_U1_U2_U3_U4_U5",
    ], "selected")
    pair_family(rows, "model_y_v2_stacked_ensemble", "family_y_v2", ["ModelY0", "ModelY2"], "selected")


def z_rows(rows):
    pair_family(rows, "model_z_family", "family_z", [
        "A2_probability_mean_max3", "A3_rank_max5",
        "Z1_fixed", "Z1_tuned", "Z2_fixed", "Z2_tuned",
        "Z3_fixed", "Z3_tuned", "Z3_tuned_raw_multimodal",
    ], "selected")
    pair_family(rows, "model_z4_final", "family_z4", [
        "A2_probability_mean_max3", "A3_rank_max5",
        "Z1_tuned", "Z2_tuned", "Z4_fixed", "Z4_tuned",
    ], "selected")
    z3_test = OUTPUT_ROOT / "model_z4_final" / "Z3_seed_summary_test.csv"
    z3_val = OUTPUT_ROOT / "model_z4_final" / "Z3_seed_summary_validation.csv"
    if z3_test.exists() or z3_val.exists():
        add_row(
            rows,
            name="Z3_seed8_summary",
            family="family_z4",
            stage="seed_summary",
            test_path=z3_test if z3_test.exists() else None,
            val_path=z3_val if z3_val.exists() else None,
        )
    pair_family(
        rows,
        "model_z2_ultimate",
        "z2_ultimate",
        ["Z2_ultimate", "Z2_ultimate_fixed_equal10"],
        "validation_only",
    )


def copy_stats():
    frames = []
    for path in sorted(OUTPUT_ROOT.glob("**/stats/*stats*.csv")):
        df = pd.read_csv(path)
        df.insert(0, "stats_file", str(path.relative_to(PROJECT_ROOT)))
        frames.append(df)
        print(f"[STATS] {path}")
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(DOC_ROOT / "all_family_stats_copies.csv", index=False)


def write_long_metrics(metrics: pd.DataFrame, out_path: Path) -> None:
    rows = []
    for _, record in metrics.iterrows():
        for split in ("val", "test"):
            for task in TASKS:
                n = record.get(f"{split}_{task}_n")
                if pd.isna(n):
                    continue
                row = {
                    "name": record["name"],
                    "family": record["family"],
                    "stage": record["stage"],
                    "split": split,
                    "task": task,
                    "n": int(n),
                }
                if task in BINARY_TASKS:
                    for metric in ("auprc", "auroc", "brier", "ece", "prevalence", "precision", "recall", "f1", "accuracy", "tn", "fp", "fn", "tp"):
                        row[metric] = record.get(f"{split}_{task}_{metric}")
                else:
                    for metric in ("mae", "rmse", "median_ae", "r2"):
                        row[metric] = record.get(f"{split}_{task}_{metric}")
                rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False)


def markdown_table(df: pd.DataFrame) -> str:
    cols = [
        "rank_test_auprc", "family", "stage", "name",
        "val_mean_binary_auprc", "test_mean_binary_auprc",
        "test_mean_binary_auroc", "test_mean_binary_brier", "test_mean_binary_ece",
        "test_los_hours_24h_mae", "test_los_hours_24h_rmse", "test_los_hours_24h_r2",
        "test_mortality_24h_auprc", "test_mortality_24h_auroc",
        "test_ventilation_onset_24h_auprc", "test_ventilation_onset_24h_auroc",
        "test_vasopressor_onset_24h_auprc", "test_vasopressor_onset_24h_auroc",
        "test_readmission_30d_24h_auprc", "test_readmission_30d_24h_auroc",
    ]
    use = [c for c in cols if c in df.columns]
    show = df[use].copy()
    for col in use:
        if col not in ("rank_test_auprc", "family", "stage", "name"):
            show[col] = pd.to_numeric(show[col], errors="coerce").map(
                lambda x: "" if pd.isna(x) else f"{x:.6f}"
            )
    lines = ["| " + " | ".join(use) + " |", "| " + " | ".join("---" for _ in use) + " |"]
    for _, row in show.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in use) + " |")
    return "\n".join(lines)


def main():
    DOC_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    baseline_rows(rows)
    proposed_rows(rows)
    y_rows(rows)
    z_rows(rows)

    metrics = pd.DataFrame(rows)
    if metrics.empty:
        raise SystemExit("No prediction files scored")
    metrics = metrics.sort_values(
        ["test_mean_binary_auprc", "val_mean_binary_auprc"],
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)
    metrics.insert(0, "rank_test_auprc", np.arange(1, len(metrics) + 1))

    wide_path = DOC_ROOT / "all_models_metrics.csv"
    long_path = DOC_ROOT / "all_models_metrics_long.csv"
    metrics.to_csv(wide_path, index=False)
    write_long_metrics(metrics, long_path)
    copy_stats()

    z2 = metrics[metrics["name"].astype(str).eq("Z2_tuned") & metrics["family"].eq("family_z")]
    note = {
        "locked_validation_selection": "family_z / Z2_tuned",
        "selection_metric": "validation mean binary AUPRC",
        "z2_ultimate_test_file_present": bool(
            (OUTPUT_ROOT / "model_z2_ultimate" / "Z2_ultimate_test_predictions.csv").exists()
        ),
        "z2_tuned_rows": z2[
            ["name", "family", "val_mean_binary_auprc", "test_mean_binary_auprc", "test_file", "val_file"]
        ].to_dict("records") if len(z2) else [],
        "calibration_warning": (
            "Brier, ECE, and threshold-0.5 metrics assume prediction is a probability. "
            "Rank-blend outputs are not probabilities."
        ),
        "warning": (
            "Test ranking is documentation only. External validation uses frozen "
            "family_z Z2_tuned, selected on validation, not the highest test row."
        ),
    }
    (DOC_ROOT / "documentation_notes.json").write_text(json.dumps(note, indent=2, default=str))
    (DOC_ROOT / "all_models_comparison.md").write_text(
        "\n".join([
            "# All completed model results",
            "",
            "Wide table is ranked by test mean binary AUPRC for documentation.",
            "Locked model remains family_z Z2_tuned because it won on validation.",
            "Brier, ECE, and 0.5-threshold metrics are diagnostic. Rank blends are not probabilities.",
            "",
            markdown_table(metrics),
            "",
        ])
    )

    print("\n===== TOP 12 BY TEST MEAN BINARY AUPRC =====")
    cols = [
        "rank_test_auprc", "family", "stage", "name",
        "val_mean_binary_auprc", "test_mean_binary_auprc",
        "test_mean_binary_brier", "test_mean_binary_ece",
        "test_los_hours_24h_mae", "test_los_hours_24h_rmse",
    ]
    print(metrics[cols].head(12).to_string(index=False))
    print(f"\nWrote {wide_path}")
    print(f"Wrote {long_path}")


if __name__ == "__main__":
    main()
