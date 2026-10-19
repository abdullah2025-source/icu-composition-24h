#!/usr/bin/env python3
"""Aggregate-only audit of the eICU-CRD cohort build (thesis Section 7.3).
Prints shapes, column names and counts. No identifiers or rows are printed."""
from pathlib import Path
import pandas as pd

D = Path("outputs/external_validation_inventory")
Z = Path("outputs/external_eicu_z2")

def show(name, df):
    print(f"\n=== {name}: {df.shape[0]:,} rows x {df.shape[1]} cols")
    print("columns:", list(df.columns))
    for c in df.columns:
        lc = c.lower()
        if any(k in lc for k in ("hospital", "patientunitstayid", "uniquepid", "patienthealthsystemstayid", "stay_id", "subject")):
            print(f"  unique {c}: {df[c].nunique():,}")
        if any(k in lc for k in ("valid", "split", "unittype", "unitdischargestatus", "hospitaldischargestatus", "excluded", "reason", "flag", "include")):
            vc = df[c].value_counts(dropna=False)
            if len(vc) <= 30:
                print(f"  value counts {c}:\n" + vc.to_string())
        if any(k in lc for k in ("age", "los", "hours", "offset")) and pd.api.types.is_numeric_dtype(df[c]):
            q = df[c].describe(percentiles=[.25, .5, .75])
            print(f"  {c}: n={int(q['count']):,} min={q['min']:.1f} p25={q['25%']:.1f} median={q['50%']:.1f} p75={q['75%']:.1f} max={q['max']:.1f}")

for f in ["eicu_cohort_skeleton.csv", "eicu_structured_index.csv", "eicu_labels_aligned_24h.csv",
          "eicu_labels_six_tasks.csv", "eicu_labels_five_plus_derived.csv"]:
    p = D / f
    if p.exists():
        show(f, pd.read_csv(p, low_memory=False))

for f in ["eicu_readmission_limitation.txt", "eicu_columns.txt"]:
    p = D / f
    if p.exists():
        print(f"\n=== {f} (first 60 lines)")
        print("\n".join(p.read_text(errors="replace").splitlines()[:60]))

for f in ["statistical_analysis/eicu_z2_canonical_statistical_metadata.json",
          "statistical_analysis/eicu_z2_inferential_analysis_metadata.json"]:
    p = Z / f
    if p.exists():
        print(f"\n=== {f}")
        print(p.read_text()[:3000])
