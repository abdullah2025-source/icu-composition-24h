#!/usr/bin/env python3
"""
Cluster-bootstrap confidence intervals for every completed model.

Does not retrain. Families 1 and 2 already have stats files; this writes a
new comparable table for every tuned baseline, fixed baseline, proposed model,
Family Y, Family Z, Z4, and Z2 Ultimate.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    roc_auc_score,
)


PROJECT_ROOT = Path("/home/abdullah2025/fyp_project")
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
DOC_ROOT = OUTPUT_ROOT / "documentation" / "stats"
N_BOOT = 1000
SEED = 20261001

BINARY_TASKS = (
    "mortality_24h",
    "ventilation_onset_24h",
    "vasopressor_onset_24h",
    "readmission_30d_24h",
)
LOS_TASK = "los_hours_24h"
TASKS = BINARY_TASKS + (LOS_TASK,)


def discover() -> list[tuple[str, str, str, Path, Path | None]]:
    rows = []
    for root in sorted(OUTPUT_ROOT.glob("baseline_family_*")):
        for stage in ("tuned", "fixed"):
            stage_root = root / stage
            if not stage_root.exists():
                continue
            for model_dir in sorted(p for p in stage_root.iterdir() if p.is_dir()):
                if stage == "tuned":
                    final = model_dir / "final_selected_configuration"
                    test_path = final / "test_predictions.csv"
                    val_path = None
                else:
                    test_path = model_dir / "test_predictions.csv"
                    if not test_path.exists():
                        test_path = model_dir / "predictions.csv"
                    val_path = None
                if test_path.exists():
                    rows.append((model_dir.name, root.name, stage, test_path, val_path))

    proposed = OUTPUT_ROOT / "proposed_model_family"
    if proposed.exists():
        for model_dir in sorted((proposed / "fixed").glob("*")):
            path = model_dir / "test_predictions.csv"
            if path.exists():
                rows.append((f"proposed_{model_dir.name}", "proposed_model_family", "fixed", path, None))
        tuned = proposed / "tuned" / "E_Full-E" / "final_selected_configuration" / "test_predictions.csv"
        if tuned.exists():
            rows.append(("E_Full-E", "proposed_model_family", "tuned", tuned, None))

    def pair(folder, family, names, stage):
        root = OUTPUT_ROOT / folder
        for name in names:
            test_path = root / f"{name}_test_predictions.csv"
            val_path = root / f"{name}_validation_predictions.csv"
            if test_path.exists() or val_path.exists():
                rows.append((
                    name,
                    family,
                    stage,
                    test_path if test_path.exists() else None,
                    val_path if val_path.exists() else None,
                ))

    pair("model_y_family", "family_y", [
        "A1_best_single", "A2_probability_mean_max3", "A3_rank_max5",
        "A4_gain_only_max3", "A5_error_correlation_max3",
        "A6_weighted_rank_same_members", "A7_los_hybrid_validation_gated",
        "Y0_rank_max3", "U1_multiseed_pool", "U2_focal_pool",
        "U3_clinical_tree_pool", "U1_U2_U3_combined_pool",
        "C0_Y0", "C1_Y0_U1", "C2_Y0_U1_U2", "C3_Y0_U1_U2_U3",
        "C4_Y0_U1_U2_U3_U4", "C5_Y0_U1_U2_U3_U4_U5",
    ], "selected")
    pair("model_y_v2_stacked_ensemble", "family_y_v2", ["ModelY0", "ModelY2"], "selected")
    pair("model_z_family", "family_z", [
        "A2_probability_mean_max3", "A3_rank_max5",
        "Z1_fixed", "Z1_tuned", "Z2_fixed", "Z2_tuned",
        "Z3_fixed", "Z3_tuned", "Z3_tuned_raw_multimodal",
    ], "selected")
    pair("model_z4_final", "family_z4", [
        "Z4_fixed", "Z4_tuned",
    ], "selected")
    pair("model_z2_ultimate", "z2_ultimate", [
        "Z2_ultimate", "Z2_ultimate_fixed_equal10",
    ], "validation_only")
    return rows


def prepare(df: pd.DataFrame, split: str) -> pd.DataFrame:
    z = df.copy()
    if "split" in z.columns:
        z = z[z["split"].astype(str) == split]
    if "validity" in z.columns:
        z = z[z["validity"].astype(bool)]
    z = z[z["task"].isin(TASKS)].copy()
    id_col = "subject_id" if "subject_id" in z.columns else "stay_id"
    z = z[np.isfinite(z["target"]) & np.isfinite(z["prediction"])].copy()
    z["cluster_id"] = z[id_col].astype(str)
    if "probability_mean" in z.columns:
        z["calibration"] = z["probability_mean"]
        z.loc[~np.isfinite(z["calibration"]), "calibration"] = z.loc[~np.isfinite(z["calibration"]), "prediction"]
    else:
        z["calibration"] = z["prediction"]
    z["calibration"] = np.clip(z["calibration"].to_numpy(float), 0.0, 1.0)
    return z[["cluster_id", "task", "target", "prediction", "calibration"]]


def point_metrics(z: pd.DataFrame) -> dict:
    out = {}
    aps, aucs = [], []
    for task in BINARY_TASKS:
        zt = z[z["task"] == task]
        y = zt["target"].to_numpy(float)
        p = zt["prediction"].to_numpy(float)
        c = zt["calibration"].to_numpy(float)
        if len(y) == 0 or len(np.unique(y)) < 2:
            out[f"{task}_auprc"] = np.nan
            out[f"{task}_auroc"] = np.nan
            out[f"{task}_brier"] = np.nan
            continue
        out[f"{task}_auprc"] = float(average_precision_score(y.astype(int), p))
        out[f"{task}_auroc"] = float(roc_auc_score(y.astype(int), p))
        out[f"{task}_brier"] = float(brier_score_loss(y.astype(int), c))
        aps.append(out[f"{task}_auprc"])
        aucs.append(out[f"{task}_auroc"])
    los = z[z["task"] == LOS_TASK]
    out["los_hours_24h_mae"] = (
        float(mean_absolute_error(los["target"], los["prediction"])) if len(los) else np.nan
    )
    out["mean_binary_auprc"] = float(np.nanmean(aps)) if aps else np.nan
    out["mean_binary_auroc"] = float(np.nanmean(aucs)) if aucs else np.nan
    return out


def bootstrap(z: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    clusters = z["cluster_id"].drop_duplicates().to_numpy()
    grouped = {cid: part.index.to_numpy() for cid, part in z.groupby("cluster_id", sort=False)}
    rng = np.random.default_rng(seed)
    point = point_metrics(z)
    keys = list(point)
    draws = {key: np.empty(n_boot) for key in keys}
    for i in range(n_boot):
        chosen = rng.choice(clusters, size=len(clusters), replace=True)
        idx = np.concatenate([grouped[cid] for cid in chosen])
        sample = point_metrics(z.loc[idx])
        for key in keys:
            draws[key][i] = sample[key]
        if (i + 1) % 100 == 0:
            print(f"    bootstrap {i + 1}/{n_boot}", flush=True)
    rows = []
    for key in keys:
        values = draws[key][np.isfinite(draws[key])]
        rows.append({
            "metric": key,
            "point": point[key],
            "ci_low": float(np.quantile(values, 0.025)) if len(values) else np.nan,
            "ci_high": float(np.quantile(values, 0.975)) if len(values) else np.nan,
            "n_boot_finite": int(len(values)),
        })
    return pd.DataFrame(rows)


def load_split(test_path: Path | None, val_path: Path | None, split: str) -> pd.DataFrame:
    if split == "test" and test_path is not None:
        df = pd.read_csv(test_path)
        prepared = prepare(df, "test")
        if len(prepared):
            return prepared
        if "split" not in df.columns:
            return prepare(df.assign(split="test"), "test")
    if split == "val":
        if val_path is not None:
            return prepare(pd.read_csv(val_path), "val")
        if test_path is not None:
            df = pd.read_csv(test_path)
            if "split" in df.columns:
                return prepare(df, "val")
    return pd.DataFrame()


def main():
    DOC_ROOT.mkdir(parents=True, exist_ok=True)
    frames = []
    jobs = discover()
    print(f"[JOBS] {len(jobs)} model files", flush=True)
    for i, (name, family, stage, test_path, val_path) in enumerate(jobs, 1):
        for split in ("val", "test"):
            z = load_split(test_path, val_path, split)
            if z.empty:
                print(f"[MISS] {i}/{len(jobs)} {family} {name} {split}", flush=True)
                continue
            print(f"[RUN] {i}/{len(jobs)} {family} {stage} {name} {split} rows={len(z)}", flush=True)
            stats = bootstrap(z, N_BOOT, SEED)
            stats.insert(0, "split", split)
            stats.insert(0, "stage", stage)
            stats.insert(0, "family", family)
            stats.insert(0, "name", name)
            stats["n_boot"] = N_BOOT
            stats["cluster"] = "subject_id"
            stats["test_file"] = "" if test_path is None else str(test_path.relative_to(PROJECT_ROOT))
            stats["val_file"] = "" if val_path is None else str(val_path.relative_to(PROJECT_ROOT))
            frames.append(stats)
            partial = pd.concat(frames, ignore_index=True)
            partial.to_csv(DOC_ROOT / "all_models_bootstrap_stats.csv", index=False)
    if not frames:
        raise SystemExit("No prediction files scored")
    out = pd.concat(frames, ignore_index=True)
    out_path = DOC_ROOT / "all_models_bootstrap_stats.csv"
    out.to_csv(out_path, index=False)
    print(f"[DONE] {out_path} rows={len(out)}", flush=True)


if __name__ == "__main__":
    main()
