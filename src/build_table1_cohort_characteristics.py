#!/usr/bin/env python3
"""
Table 1 (thesis Section 3.7): cohort characteristics of the MIMIC-IV primary
24-hour cohort, overall and by split. Aggregates only: no identifiers, no
timestamps and no rows are written, so the output can leave the server.

    python build_table1_cohort_characteristics.py

Reads (read-only):
    outputs/pipeline_a_v2/cohort_master.csv
    outputs/pipeline_a_v2/a5_alignment_index_24h.csv
    outputs/pipeline_a_v2/clinical_note_embedding_metadata.csv
    outputs/pipeline_a_v2/vitals_labs_hourly.parquet          (first-24 h severity)
    data/mimic_iv/mimic-iv-3.1/hosp/admissions.csv.gz         (admission type, race, insurance)
    data/mimic_iv/mimic-iv-3.1/icu/icustays.csv.gz            (first care unit)

Writes:
    outputs/table1/table1_mimic4_primary.csv   characteristic, level, overall, train, val, test
    outputs/table1/table1_mimic4_primary.txt   the same table, printed

Cells with fewer than 10 stays are shown as "<10" (small-cell rule).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SPLITS = ("train", "val", "test")
SMALL = 10
FIRST24 = ["heart_rate", "mbp", "resp_rate", "spo2", "temperature", "lactate", "creatinine", "gcs_total"]


def fmt_n_pct(k: int, n: int) -> str:
    if k < SMALL:
        return f"<{SMALL}"
    return f"{k:,} ({100 * k / n:.1f}%)"


def fmt_med_iqr(x: pd.Series, digits: int = 1) -> str:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) < SMALL:
        return f"n<{SMALL}"
    q1, q2, q3 = np.percentile(x, [25, 50, 75])
    return f"{q2:.{digits}f} [{q1:.{digits}f}–{q3:.{digits}f}]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, default=Path.cwd())
    a = ap.parse_args()
    root = a.project_root.resolve()
    v2 = root / "outputs" / "pipeline_a_v2"
    raw = root / "data" / "mimic_iv" / "mimic-iv-3.1"
    out_dir = root / "outputs" / "table1"
    out_dir.mkdir(parents=True, exist_ok=True)

    idx = pd.read_csv(v2 / "a5_alignment_index_24h.csv")
    a0 = pd.read_csv(v2 / "cohort_master.csv", usecols=["stay_id", "age", "gender"])
    df = idx.merge(a0, on="stay_id", how="left", validate="one_to_one")

    adm = pd.read_csv(raw / "hosp" / "admissions.csv.gz",
                      usecols=["hadm_id", "admission_type", "race", "insurance"])
    df = df.merge(adm, on="hadm_id", how="left", validate="many_to_one")
    icu = pd.read_csv(raw / "icu" / "icustays.csv.gz", usecols=["stay_id", "first_careunit"])
    df = df.merge(icu, on="stay_id", how="left", validate="one_to_one")

    meta = pd.read_csv(v2 / "clinical_note_embedding_metadata.csv", usecols=["stay_id", "note_count"])
    df = df.merge(meta, on="stay_id", how="left", validate="one_to_one")
    df["note_count"] = df["note_count"].fillna(0)

    grid = pd.read_parquet(v2 / "vitals_labs_hourly.parquet", columns=["stay_id", "icu_hour"] + FIRST24)
    sev = grid.groupby("stay_id").agg(
        hr_max=("heart_rate", "max"), mbp_min=("mbp", "min"), rr_max=("resp_rate", "max"),
        spo2_min=("spo2", "min"), temp_max=("temperature", "max"), lactate_max=("lactate", "max"),
        creat_max=("creatinine", "max"), gcs_min=("gcs_total", "min"),
    ).reset_index()
    df = df.merge(sev, on="stay_id", how="left", validate="one_to_one")

    assert len(df) == 74_572 and df["stay_id"].is_unique

    rows = []

    def add(char: str, level: str, fn):
        r = {"characteristic": char, "level": level, "overall": fn(df)}
        for s in SPLITS:
            r[s] = fn(df[df["split_24h"].eq(s)])
        rows.append(r)

    add("ICU stays", "n", lambda d: f"{len(d):,}")
    add("Patients", "n", lambda d: f"{d['subject_id'].nunique():,}")
    add("Age, years", "median [IQR]", lambda d: fmt_med_iqr(d["age"]))
    add("Sex", "female", lambda d: fmt_n_pct(int(d["gender"].eq("F").sum()), len(d)))
    for lvl in df["admission_type"].value_counts().index:
        add("Admission type", str(lvl), lambda d, lvl=lvl: fmt_n_pct(int(d["admission_type"].eq(lvl).sum()), len(d)))
    for lvl in df["first_careunit"].value_counts().index:
        add("First ICU unit", str(lvl), lambda d, lvl=lvl: fmt_n_pct(int(d["first_careunit"].eq(lvl).sum()), len(d)))
    race = df["race"].fillna("UNKNOWN").str.split(" - ").str[0].str.split("/").str[0]
    df["race_group"] = race
    for lvl in df["race_group"].value_counts().index[:6]:
        add("Race (recorded)", str(lvl), lambda d, lvl=lvl: fmt_n_pct(int(d["race_group"].eq(lvl).sum()), len(d)))
    for lvl in df["insurance"].fillna("UNKNOWN").value_counts().index:
        add("Insurance", str(lvl), lambda d, lvl=lvl: fmt_n_pct(int(d["insurance"].fillna("UNKNOWN").eq(lvl).sum()), len(d)))
    add("Radiology text in first 24 h", "stays with ≥1 report", lambda d: fmt_n_pct(int(d["text_available"].eq(1).sum()), len(d)))
    add("Radiology reports per stay with text", "median [IQR]", lambda d: fmt_med_iqr(d.loc[d["note_count"] > 0, "note_count"], 0))
    for col, lab in [("hr_max", "Heart rate max, bpm"), ("mbp_min", "Mean BP min, mmHg"), ("rr_max", "Resp. rate max, /min"),
                     ("spo2_min", "SpO2 min, %"), ("temp_max", "Temperature max, °C"), ("lactate_max", "Lactate max, mmol/L"),
                     ("creat_max", "Creatinine max, mg/dL"), ("gcs_min", "GCS min")]:
        add("First 24 h", lab + ", median [IQR] (among measured)", lambda d, col=col: fmt_med_iqr(d[col]))
    add("ICU length of stay, h", "median [IQR]", lambda d: fmt_med_iqr(d["los_hours_24h"]))
    for t, lab in [("mortality_24h", "Hospital mortality"), ("ventilation_onset_24h", "Ventilation onset"),
                   ("vasopressor_onset_24h", "Vasopressor onset"), ("readmission_30d_24h", "30-day ICU readmission"),
                   ("deterioration_24h", "Deterioration")]:
        add(lab, "valid stays", lambda d, t=t: fmt_n_pct(int(d[f"{t}_valid"].astype(bool).sum()), len(d)))
        add(lab, "events / valid", lambda d, t=t: fmt_n_pct(int(d.loc[d[f"{t}_valid"].astype(bool), t].eq(1).sum()),
                                                           int(d[f"{t}_valid"].astype(bool).sum())))

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "table1_mimic4_primary.csv", index=False)
    text = table.to_string(index=False)
    (out_dir / "table1_mimic4_primary.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"\nwritten: {out_dir}")


if __name__ == "__main__":
    main()
