#!/usr/bin/env python3
"""
Full server audit — READ-ONLY.

Discovers every trained-model folder under outputs/ and records which
artifacts are present. Writes only to outputs/documentation/.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path("/home/abdullah2025/fyp_project")
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
DOC_ROOT = OUTPUT_ROOT / "documentation"

MODEL_MARKER_FILES = [
    "checkpoint.pt",
    "metadata.json",
    "best_params.json",
    "metrics.csv",
    "training_history.csv",
    "test_predictions.csv",
    "predictions.csv",
    "validation_predictions.csv",
    "architecture_manifest.json",
    "study.db",
    "optuna_trials.csv",
    "final_result.json",
    "family_summary.json",
]
SKIP_DIRS = {
    "documentation",
    "pipeline_a_v2",
    "pipeline_a",
    "benchmark_selection",
    "targets_24h",
    "split_24h",
}


def file_info(path: Path) -> dict:
    if not path.exists():
        return {"exists": False}
    st = path.stat()
    return {
        "exists": True,
        "size_bytes": st.st_size,
        "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime)),
    }


def fingerprint_csv(path: Path, n: int = 200) -> str:
    try:
        head = path.read_text(errors="ignore").splitlines()[:n]
        return hashlib.md5("\n".join(head).encode("utf-8")).hexdigest()[:12]
    except Exception:
        return ""


def is_model_dir(path: Path) -> bool:
    if path.name in SKIP_DIRS:
        return False
    return any((path / marker).exists() for marker in MODEL_MARKER_FILES)


def infer_family(path: Path) -> str:
    for part in path.parts:
        low = part.lower()
        if low.startswith(("baseline_family_", "proposed_model", "model_y", "model_z")):
            return part
    return "other"


def infer_stage(path: Path) -> str:
    text = str(path).lower()
    if "/tuned/" in text and "final_selected_configuration" in text:
        return "tuned_final"
    if "/fixed/" in text:
        return "fixed"
    if "/tuned/" in text:
        return "tuned_other"
    if "seed" in text:
        return "seed"
    if "trial" in text:
        return "trial"
    return "other"


def scan_models() -> pd.DataFrame:
    rows = []
    for path in sorted(p for p in OUTPUT_ROOT.rglob("*") if p.is_dir()):
        if any(skip in path.parts for skip in SKIP_DIRS) or not is_model_dir(path):
            continue
        rec = {
            "model_dir": str(path.relative_to(PROJECT_ROOT)),
            "family": infer_family(path),
            "stage": infer_stage(path),
        }
        for marker in MODEL_MARKER_FILES:
            info = file_info(path / marker)
            rec[f"has_{marker}"] = info["exists"]
            if info["exists"]:
                rec[f"size_{marker}"] = info["size_bytes"]
                rec[f"mtime_{marker}"] = info["mtime"]
        for pred_name in ("test_predictions.csv", "validation_predictions.csv", "predictions.csv"):
            pred = path / pred_name
            if not pred.exists():
                continue
            try:
                head = pd.read_csv(pred, nrows=5)
                cols = set(head.columns)
                rec[f"{pred_name}_cols"] = "|".join(sorted(cols))
                for col in ("split", "task", "prediction", "probability_mean", "stay_id", "target", "validity"):
                    rec[f"{pred_name}_has_{col}"] = col in cols
                full = pd.read_csv(pred)
                rec[f"{pred_name}_n_rows"] = len(full)
                if "split" in full.columns:
                    for split in ("train", "val", "test"):
                        rec[f"{pred_name}_rows_{split}"] = int((full["split"].astype(str) == split).sum())
                rec[f"{pred_name}_fingerprint"] = fingerprint_csv(pred)
            except Exception as exc:
                rec[f"{pred_name}_read_error"] = str(exc)
        rows.append(rec)
        print(f"[MODEL] {rec['model_dir']}", flush=True)
    return pd.DataFrame(rows)


def scan_files() -> pd.DataFrame:
    rows = []
    for path in sorted(p for p in OUTPUT_ROOT.rglob("*") if p.is_file()):
        if "documentation" in path.parts:
            continue
        rec = {
            "path": str(path.relative_to(PROJECT_ROOT)),
            "suffix": path.suffix,
            "size_bytes": path.stat().st_size,
            "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(path.stat().st_mtime)),
        }
        if path.suffix == ".csv":
            try:
                head = pd.read_csv(path, nrows=5)
                rec["columns_head"] = "|".join(head.columns.tolist())
                rec["n_rows_sampled"] = len(head)
            except Exception as exc:
                rec["read_error"] = str(exc)
        rows.append(rec)
    return pd.DataFrame(rows)


def build_duplicates(files_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rel in files_df.loc[files_df["suffix"].eq(".csv"), "path"]:
        fp = fingerprint_csv(PROJECT_ROOT / rel)
        if fp:
            rows.append({"path": rel, "fingerprint": fp})
    if not rows:
        return pd.DataFrame(columns=["fingerprint", "paths", "count"])
    groups = []
    for fp, grp in pd.DataFrame(rows).groupby("fingerprint"):
        if len(grp) > 1:
            groups.append({
                "fingerprint": fp,
                "count": len(grp),
                "paths": " ;; ".join(grp["path"].tolist()),
            })
    return pd.DataFrame(groups) if groups else pd.DataFrame(columns=["fingerprint", "paths", "count"])


def make_summary(models_df: pd.DataFrame, files_df: pd.DataFrame, dups_df: pd.DataFrame) -> str:
    lines = [
        "=" * 100,
        "SERVER AUDIT SUMMARY",
        "=" * 100,
        f"Total model directories discovered : {len(models_df)}",
        f"Total files under outputs/ excluding documentation : {len(files_df)}",
        f"Duplicate-content CSV groups : {len(dups_df)}",
        "",
        "=== MODELS BY FAMILY ===",
    ]
    for family, grp in models_df.groupby("family"):
        lines.append(f"  {family:45} {len(grp):4d} directories")
    lines += ["", "=== MODELS BY STAGE ==="]
    for stage, grp in models_df.groupby("stage"):
        lines.append(f"  {stage:15} {len(grp):4d}")
    has_test = models_df.get("has_test_predictions.csv", pd.Series(False, index=models_df.index)).fillna(False)
    has_val = models_df.get("has_validation_predictions.csv", pd.Series(False, index=models_df.index)).fillna(False)
    has_ckpt = models_df.get("has_checkpoint.pt", pd.Series(False, index=models_df.index)).fillna(False)
    has_meta = models_df.get("has_metadata.json", pd.Series(False, index=models_df.index)).fillna(False)
    lines += [
        "",
        "=== ARTIFACT COMPLETENESS ===",
        f"  With checkpoint.pt           : {int(has_ckpt.sum())}",
        f"  With metadata.json           : {int(has_meta.sum())}",
        f"  With test_predictions.csv    : {int(has_test.sum())}",
        f"  With validation_predictions  : {int(has_val.sum())}",
        f"  With both val and test       : {int((has_test & has_val).sum())}",
        f"  Test only                    : {int((has_test & ~has_val).sum())}",
        f"  Val only                     : {int((~has_test & has_val).sum())}",
        "",
        f"=== checkpoint.pt WITHOUT test_predictions.csv ({int((has_ckpt & ~has_test).sum())}) ===",
    ]
    for _, row in models_df[has_ckpt & ~has_test].iterrows():
        lines.append(f"  {row['model_dir']}")
    lines += ["", "=== PREDICTION FILES MISSING probability_mean ==="]
    for _, row in models_df.iterrows():
        for pred_name in ("test_predictions.csv", "validation_predictions.csv", "predictions.csv"):
            flag = row.get(f"{pred_name}_has_probability_mean")
            if flag is False:
                lines.append(f"  {row['model_dir']}/{pred_name}")
    lines += ["", f"=== DUPLICATE-CONTENT CSV GROUPS: {len(dups_df)} ==="]
    for _, row in dups_df.head(30).iterrows():
        lines.append(f"  [{row['fingerprint']}] {row['count']} copies")
        for path in str(row["paths"]).split(" ;; "):
            lines.append(f"    {path}")
    if len(dups_df) > 30:
        lines.append(f"  ... +{len(dups_df) - 30} more in audit_duplicates.csv")
    lines += ["", "=== LARGEST 20 CSV FILES ==="]
    big = files_df[files_df["suffix"].eq(".csv")].nlargest(20, "size_bytes")
    for _, row in big.iterrows():
        lines.append(f"  {row['size_bytes'] / 1024 / 1024:7.2f} MB  {row['path']}")
    return "\n".join(lines) + "\n"


def main():
    DOC_ROOT.mkdir(parents=True, exist_ok=True)
    print("Scanning model directories...", flush=True)
    models_df = scan_models()
    print(f" found {len(models_df)} model directories", flush=True)
    print("Scanning files...", flush=True)
    files_df = scan_files()
    print(f" found {len(files_df)} files", flush=True)
    print("Detecting duplicates...", flush=True)
    dups_df = build_duplicates(files_df)
    print(f" found {len(dups_df)} duplicate groups", flush=True)
    models_df.to_csv(DOC_ROOT / "audit_models.csv", index=False)
    files_df.to_csv(DOC_ROOT / "audit_files.csv", index=False)
    dups_df.to_csv(DOC_ROOT / "audit_duplicates.csv", index=False)
    summary = make_summary(models_df, files_df, dups_df)
    (DOC_ROOT / "audit_summary.txt").write_text(summary)
    print(summary)
    print(f"Artifacts written to: {DOC_ROOT}", flush=True)


if __name__ == "__main__":
    main()
