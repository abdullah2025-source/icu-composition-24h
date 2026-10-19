#!/usr/bin/env python3
"""
MIMIC-III CareVue external cohort: model inputs in the frozen Pipeline A V2 format.

For every stay in the label file this builds
    structured tensor  [N, 24, 110]  = 28 values + 28 masks + 28 time-since + 26 trends
    demographics       [N, 4]        = age_zscore, gender_F, gender_M, gender_UNKNOWN
exactly as MIMIC-IV stages A1 -> A3 -> A4 do, with two differences that cannot be avoided:
    * CareVue item IDs replace the MetaVision item IDs (mapping below);
    * NOTHING is fitted here. Medians, means and standard deviations are read from the
      MIMIC-IV training statistics (train_imputation_stats.csv, train_scaling_stats.csv)
      and the age statistics are recomputed from the MIMIC-IV training stays.

Steps (same order as V2)
    A1  events in [intime, intime + 24 h): unit handling, error-flag and plausibility filter
    A3  hourly median (max for code features), complete 24-hour grid
    A4  mask -> forward fill -> time since -> 3 h trend -> train-median fallback -> z-score

Run
    python build_mimic3_carevue_features_24h.py --verify-on-mimic4      # optional, recommended once
    python build_mimic3_carevue_features_24h.py --data-dir external_validation_data/mimic-iii-clinical-database-1.4

--verify-on-mimic4 re-applies the A4 code in this file to the MIMIC-IV hourly grid and
compares the result with the frozen MIMIC-IV tensor, so you can prove the transform is identical.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HOURS = 24
FEATURES = ["heart_rate", "sbp", "dbp", "mbp", "resp_rate", "spo2", "temperature", "wbc", "hemoglobin", "platelets", "creatinine", "bun",
            "lactate", "glucose", "sodium", "potassium", "chloride", "bicarbonate", "ph", "pao2", "paco2", "bilirubin", "gcs_total",
            "gcs_unable", "gcs_components_observed", "fio2", "peep", "resp_support_mode"]
TREND_FEATURES = [f for f in FEATURES if f not in {"gcs_unable", "resp_support_mode"}]
MAX_AGG = {"gcs_unable", "gcs_components_observed", "resp_support_mode"}

# ---- CareVue item mapping (MIMIC-IV MetaVision item in the comment) -------------------------
CHART_NUMERIC = {
    "heart_rate": [211],                      # 220045
    "sbp": [51, 442, 455, 6701],              # 220050 arterial, 220179 non-invasive
    "dbp": [8368, 8440, 8441, 8555],          # 220051, 220180
    "mbp": [52, 443, 456, 6702],              # 220052, 220181
    "resp_rate": [618, 615],                  # 220210, 224690
    "spo2": [646],                            # 220277
    "temperature": [676, 678],                # 223762 (C); 678 is Fahrenheit and is converted
    "fio2": [190, 3420, 3422],                # 223835; 190 is a fraction, handled by the V2 FiO2 rule
    "peep": [505, 506],                       # 220339, 224700
}
TEMP_F_ITEMS = {678}
LAB_NUMERIC = {  # LABEVENTS item IDs are the same in MIMIC-III and MIMIC-IV (52027 does not exist in MIMIC-III)
    "wbc": [51300, 51301], "hemoglobin": [51222], "platelets": [51265], "creatinine": [50912], "bun": [51006], "lactate": [50813],
    "glucose": [50931], "sodium": [50824, 50983], "potassium": [50822, 50971], "chloride": [50806, 50902], "bicarbonate": [50882],
    "ph": [50820], "pao2": [50821], "paco2": [50818], "bilirubin": [50885],
}
GCS_VERBAL, GCS_MOTOR, GCS_EYES = 723, 454, 184     # 223900, 223901, 220739
O2_DEVICE, VENT_MODE = 467, 720                     # 226732, 223849
# Same six-level code as V2: None 0, SupplementalOxygen 1, HFNC 2, NonInvasiveVent 3, InvasiveVent 4, Tracheostomy 5.
O2_DEVICE_CODE = {
    "trach mask": 5,
    "bipap mask": 3, "cpap mask": 3,
    "non-rebreather": 1, "face tent": 1, "aerosol-cool": 1, "venti mask": 1, "medium conc mask": 1, "ultrasonic neb": 1,
    "hi flow neb": 1, "nasal cannula": 1,
}   # 'T-Piece', 'TranstrachealCat', 'Heated Neb', 'Other/Remarks' have no V2 category and are ignored.

BOUNDS = {"heart_rate": (20, 300), "sbp": (20, 300), "dbp": (10, 200), "mbp": (20, 250), "resp_rate": (2, 70), "spo2": (50, 100),
          "temperature": (25, 45), "wbc": (0.1, 150), "hemoglobin": (2, 20), "platelets": (1, 2000), "creatinine": (0.1, 20),
          "bun": (1, 200), "lactate": (0.1, 30), "glucose": (10, 1000), "sodium": (100, 200), "potassium": (1, 10),
          "chloride": (60, 150), "bicarbonate": (2, 60), "ph": (6.5, 8.0), "pao2": (20, 600), "paco2": (10, 150),
          "bilirubin": (0.1, 50), "fio2": (20, 100), "peep": (0, 100)}

CHART_ITEM_TO_FEATURE = {i: f for f, ids in CHART_NUMERIC.items() for i in ids}
LAB_ITEM_TO_FEATURE = {i: f for f, ids in LAB_NUMERIC.items() for i in ids}
CHART_ITEMS = set(CHART_ITEM_TO_FEATURE) | {GCS_VERBAL, GCS_MOTOR, GCS_EYES, O2_DEVICE, VENT_MODE}


# =========================================================================== A4 (vectorised, no fitting)
def load_v2_stats(v2: Path) -> dict:
    imp = pd.read_csv(v2 / "train_imputation_stats.csv").set_index("feature")["train_population_median"]
    sc = pd.read_csv(v2 / "train_scaling_stats.csv")
    val = sc[sc.channel_group == "value"].set_index("feature")
    trd = sc[sc.channel_group == "trend"].set_index("feature")
    assert list(val.index) == FEATURES, "value scaling rows do not match the frozen 28-feature order"
    assert list(trd.index) == TREND_FEATURES, "trend scaling rows do not match the frozen 26 trend features"
    return {"median": imp.reindex(FEATURES).to_numpy(np.float64),
            "val_mean": val["train_mean"].to_numpy(np.float64), "val_std": val["train_std"].to_numpy(np.float64),
            "val_zscore": val["scaling"].eq("zscore_train_only").to_numpy(),
            "trd_mean": trd["train_mean"].to_numpy(np.float64), "trd_std": trd["train_std"].to_numpy(np.float64)}


def a4_transform(values: np.ndarray, st: dict) -> np.ndarray:
    """values: [N, 24, 28] float32 with NaN for unobserved cells. Returns the [N, 24, 110] tensor."""
    n = values.shape[0]
    assert values.shape[1:] == (HOURS, len(FEATURES))
    mask = np.isfinite(values)                                                     # 1. masks before any fill
    hour = np.arange(HOURS, dtype=np.int32)[None, :, None]
    last = np.maximum.accumulate(np.where(mask, hour, -1), axis=1)                 # hour of last valid observation
    filled = np.take_along_axis(values, np.clip(last, 0, None), axis=1)            # 2. forward fill within stay
    filled = np.where(last >= 0, filled, np.nan).astype(np.float32)
    time_since = np.where(last >= 0, hour - last, HOURS).astype(np.float32)        # 3. time since last valid value
    tidx = [FEATURES.index(f) for f in TREND_FEATURES]                             # 4. trailing 3 h OLS slope
    f64 = filled[:, :, tidx].astype(np.float64)
    trend = np.full((n, HOURS, len(tidx)), np.nan, dtype=np.float64)
    trend[:, 2:, :] = (f64[:, 2:, :] - f64[:, :-2, :]) / 2.0                        # slope over x = 0,1,2
    trend[:, 2:, :][~np.isfinite(f64[:, :-2, :])] = np.nan                         # window must be complete
    vals = filled.astype(np.float64)                                               # 5. train-median fallback
    vals = np.where(np.isfinite(vals), vals, st["median"][None, None, :])
    trend = np.where(np.isfinite(trend), trend, st["trd_mean"][None, None, :])     # 6. trend fallback = train mean
    z = (vals - st["val_mean"][None, None, :]) / st["val_std"][None, None, :]      # 7. train-only z-score
    vals = np.where(st["val_zscore"][None, None, :], z, vals).astype(np.float32)   #    code features unchanged
    trend = ((trend - st["trd_mean"][None, None, :]) / st["trd_std"][None, None, :]).astype(np.float32)
    out = np.concatenate([vals, mask.astype(np.float32), time_since, trend], axis=2).astype(np.float32)
    assert out.shape == (n, HOURS, 110) and np.isfinite(out).all()
    return out


def mimic4_age_stats(v2: Path) -> tuple[float, float, float]:
    a0 = pd.read_csv(v2 / "cohort_master.csv", usecols=["stay_id", "age"])
    idx = pd.read_csv(v2 / "index_24h.csv", usecols=["stay_id", "split_24h"])
    age = idx[idx.split_24h.astype(str).str.lower().eq("train")].merge(a0, on="stay_id", how="left")["age"].astype(float).to_numpy()
    med = float(np.nanmedian(age))
    age = np.where(np.isfinite(age), age, med)
    return med, float(age.mean()), float(age.std(ddof=0))


def verify_on_mimic4(v2: Path, n_check: int) -> None:
    st = load_v2_stats(v2)
    a3 = pd.read_parquet(v2 / "vitals_labs_hourly.parquet").sort_values(["stay_id", "icu_hour"], kind="mergesort")
    n = a3["stay_id"].nunique()
    grid = a3[FEATURES].to_numpy(np.float32).reshape(n, HOURS, len(FEATURES))
    ref = np.load(v2 / "structured_24h_processed.npy", mmap_mode="r")
    k = min(n_check, n)
    mine = a4_transform(grid[:k], st)
    diff = np.abs(mine - np.asarray(ref[:k]))
    names = ["value", "mask", "time_since", "trend"]
    sl = [slice(0, 28), slice(28, 56), slice(56, 84), slice(84, 110)]
    print(f"A4 check on {k:,} MIMIC-IV stays: max abs difference per block")
    for nm, s in zip(names, sl):
        print(f"  {nm:10s} {diff[:, :, s].max():.3e}")
    print("PASS: identical transform" if diff.max() < 1e-3 else "FAIL: transform differs from the frozen A4 — do not use the output")


# =========================================================================== A1 (CareVue events)
def find_table(root: Path, stem: str) -> Path:
    names = {stem.lower() + ext for ext in ("", ".csv", ".csv.gz")}
    hits = [p for p in root.rglob("*") if p.is_file() and p.name.lower() in names]
    if not hits:
        raise FileNotFoundError(f"Could not find {stem} under {root}")
    return sorted(hits, key=lambda p: len(p.parts))[0]


def read_filtered(path: Path, cols: list[str], itemids: set[int], key: str, keys: set[int], chunksize: int = 2_000_000) -> pd.DataFrame:
    wanted, parts, n_read = set(cols), [], 0
    for chunk in pd.read_csv(path, usecols=lambda c: c.lower() in wanted, chunksize=chunksize, low_memory=False):
        chunk.columns = [c.lower() for c in chunk.columns]
        n_read += len(chunk)
        chunk = chunk[chunk["itemid"].isin(itemids)]
        chunk = chunk[pd.to_numeric(chunk[key], errors="coerce").isin(keys)]
        if not chunk.empty:
            parts.append(chunk)
        print(f"  {path.name}: {n_read:,} rows read, {sum(len(p) for p in parts):,} kept", end="\r", flush=True)
    print()
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)
    out["charttime"] = pd.to_datetime(out["charttime"], errors="coerce")
    return out.dropna(subset=["charttime"])


def normalize_fio2(x: pd.Series) -> pd.Series:
    """V2 rule: 0.20-1.00 is a fraction (x100); 20-100 is percent; everything else invalid."""
    x = pd.to_numeric(x, errors="coerce").astype(float)
    out = x.copy()
    frac = (x >= 0.20) & (x <= 1.00)
    out[frac] = x[frac] * 100.0
    out[~(frac | ((x >= 20.0) & (x <= 100.0)))] = np.nan
    return out


def window(ev: pd.DataFrame, stays: pd.DataFrame, on: list[str]) -> pd.DataFrame:
    """Keep events inside the stay and inside the first 24 ICU hours; add icu_hour."""
    ev = ev.merge(stays[list(dict.fromkeys(on + ["icustay_id", "intime", "outtime"]))], on=on, how="inner")
    ev = ev[(ev["charttime"] >= ev["intime"]) & (ev["charttime"] < ev["outtime"])]
    h = (ev["charttime"] - ev["intime"]).dt.total_seconds() / 3600.0
    ev = ev[(h >= 0) & (h < HOURS)].copy()
    ev["icu_hour"] = np.floor(h[ev.index]).astype("int32")
    return ev


def numeric_events(ev: pd.DataFrame, item_map: dict, audit: dict, source: str, celsius_only: bool) -> pd.DataFrame:
    ev = ev[ev["itemid"].isin(item_map)].copy()
    ev["feature"] = ev["itemid"].map(item_map)
    ev["v"] = pd.to_numeric(ev["valuenum"], errors="coerce").astype(float)
    flagged = pd.to_numeric(ev["error"], errors="coerce").fillna(0).astype(int).gt(0) if "error" in ev.columns else pd.Series(False, index=ev.index)
    is_f = ev["itemid"].isin(TEMP_F_ITEMS)
    if celsius_only:
        ev = ev[~is_f]; flagged = flagged[ev.index]; is_f = is_f[ev.index]
    ev.loc[is_f, "v"] = (ev.loc[is_f, "v"] - 32.0) * 5.0 / 9.0
    fio = ev["feature"].eq("fio2")
    ev.loc[fio, "v"] = normalize_fio2(ev.loc[fio, "v"])
    lo = ev["feature"].map(lambda f: BOUNDS[f][0]).astype(float)
    hi = ev["feature"].map(lambda f: BOUNDS[f][1]).astype(float)
    ok = ~flagged & np.isfinite(ev["v"]) & (ev["v"] >= lo) & (ev["v"] <= hi)
    for f, g in ev.groupby("feature"):
        audit[f] = {"source": source, "items": sorted(int(i) for i in g["itemid"].unique()), "raw_rows": int(len(g)),
                    "error_flagged": int(flagged[g.index].sum()), "valid_rows": int(ok[g.index].sum()),
                    "rejected_percent": round(100.0 * (1 - ok[g.index].mean()), 3)}
    return ev.loc[ok, ["icustay_id", "icu_hour", "feature", "v"]].rename(columns={"v": "value"})


def gcs_events(ch: pd.DataFrame, audit: dict) -> pd.DataFrame:
    g = ch[ch["itemid"].isin([GCS_VERBAL, GCS_MOTOR, GCS_EYES])].copy()
    if "error" in g.columns:
        g = g[pd.to_numeric(g["error"], errors="coerce").fillna(0).astype(int).eq(0)]
    g = g.drop_duplicates(subset=["icustay_id", "charttime", "itemid"], keep="last")
    g["c"] = pd.to_numeric(g["valuenum"], errors="coerce").astype(float)
    ett = g["itemid"].eq(GCS_VERBAL) & g["value"].astype(str).str.contains("ET/Trach", case=False, regex=False)
    g.loc[ett, "c"] = 0.0                                   # V2: intubated verbal = 0, total forced to 15, gcs_unable = 1
    g["ett"] = ett.astype(int)
    rng = {GCS_VERBAL: (0.0, 5.0), GCS_MOTOR: (1.0, 6.0), GCS_EYES: (1.0, 4.0)}
    lo, hi = g["itemid"].map(lambda i: rng[i][0]), g["itemid"].map(lambda i: rng[i][1])
    valid = np.isfinite(g["c"]) & (g["c"] >= lo) & (g["c"] <= hi) & (~g["itemid"].eq(GCS_VERBAL) | g["c"].ge(1.0) | ett)
    audit["gcs_components"] = {"raw_rows": int(len(g)), "valid_rows": int(valid.sum()), "ett_verbal_rows": int(ett.sum())}
    g = g[valid]
    if g.empty:
        return pd.DataFrame(columns=["icustay_id", "icu_hour", "feature", "value"])
    base = g[["icustay_id", "charttime", "icu_hour"]].drop_duplicates(["icustay_id", "charttime"]).sort_values("charttime")
    tol = pd.Timedelta(hours=6)
    for item, name in [(GCS_MOTOR, "motor"), (GCS_EYES, "eyes"), (GCS_VERBAL, "verbal")]:
        comp = g.loc[g["itemid"].eq(item), ["icustay_id", "charttime", "c"] + (["ett"] if item == GCS_VERBAL else [])]
        comp = comp.rename(columns={"c": name}).sort_values("charttime")
        base = pd.merge_asof(base, comp, on="charttime", by="icustay_id", direction="backward", tolerance=tol)
    base["ett"] = base["ett"].fillna(0).astype(int)
    n_obs = base[["motor", "eyes", "verbal"]].notna().sum(axis=1)
    base = base[n_obs > 0].copy()
    total = base["motor"].fillna(6.0) + base["verbal"].fillna(5.0) + base["eyes"].fillna(4.0)
    base["gcs_total"] = np.where(base["ett"].eq(1), 15.0, total)
    base["gcs_unable"] = base["ett"].astype(float)
    base["gcs_components_observed"] = n_obs[base.index].astype(float)
    audit["gcs_derived_observations"] = int(len(base))
    long = base.melt(id_vars=["icustay_id", "icu_hour"], value_vars=["gcs_total", "gcs_unable", "gcs_components_observed"],
                     var_name="feature", value_name="value")
    return long


def support_events(ch: pd.DataFrame, audit: dict) -> pd.DataFrame:
    s = ch[ch["itemid"].isin([O2_DEVICE, VENT_MODE])].copy()
    if "error" in s.columns:
        s = s[pd.to_numeric(s["error"], errors="coerce").fillna(0).astype(int).eq(0)]
    val = s["value"].astype(str).str.strip().str.lower()
    code = pd.Series(np.nan, index=s.index)
    dev = s["itemid"].eq(O2_DEVICE)
    code[dev] = val[dev].map(O2_DEVICE_CODE)
    mode = s["itemid"].eq(VENT_MODE) & s["value"].notna() & ~val.eq("other/remarks")
    code[mode] = 4.0
    s["code"] = code
    audit["resp_support"] = {"source_rows": int(len(s)), "recognised_rows": int(code.notna().sum()),
                             "code_counts": {str(int(k)): int(v) for k, v in code.dropna().value_counts().sort_index().items()},
                             "note": "CareVue has no 'None' device value, so code 0 never occurs"}
    s = s.dropna(subset=["code"])
    s = s.groupby(["icustay_id", "charttime", "icu_hour"], as_index=False)["code"].max()
    return s.assign(feature="resp_support_mode").rename(columns={"code": "value"})[["icustay_id", "icu_hour", "feature", "value"]]


# =========================================================================== A3
def hourly_grid(events: pd.DataFrame, stay_ids: np.ndarray) -> np.ndarray:
    pos = pd.Series(np.arange(len(stay_ids)), index=stay_ids)
    grid = np.full((len(stay_ids), HOURS, len(FEATURES)), np.nan, dtype=np.float32)
    for j, f in enumerate(FEATURES):
        sub = events[events["feature"].eq(f)]
        if sub.empty:
            continue
        agg = sub.groupby(["icustay_id", "icu_hour"])["value"].agg("max" if f in MAX_AGG else "median").reset_index()
        grid[pos.loc[agg["icustay_id"]].to_numpy(), agg["icu_hour"].to_numpy(), j] = agg["value"].to_numpy(np.float32)
    return grid


# =========================================================================== main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, help="MIMIC-III folder")
    ap.add_argument("--labels", type=Path, default=Path("outputs/external_mimic3/mimic3_carevue_labels_aligned_24h.csv"))
    ap.add_argument("--v2-dir", type=Path, default=Path("outputs/pipeline_a_v2"), help="frozen MIMIC-IV V2 artifacts")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/external_mimic3"))
    ap.add_argument("--temperature-celsius-only", action="store_true", help="Use only the Celsius item (as MIMIC-IV does); default also converts Fahrenheit")
    ap.add_argument("--verify-on-mimic4", action="store_true", help="Only check that the A4 code here reproduces the frozen MIMIC-IV tensor")
    ap.add_argument("--verify-stays", type=int, default=20000)
    a = ap.parse_args()

    if a.verify_on_mimic4:
        verify_on_mimic4(a.v2_dir, a.verify_stays)
        return
    if a.data_dir is None:
        ap.error("--data-dir is required")
    a.out_dir.mkdir(parents=True, exist_ok=True)
    st = load_v2_stats(a.v2_dir)
    age_med, age_mean, age_std = mimic4_age_stats(a.v2_dir)

    lab = pd.read_csv(a.labels, parse_dates=["intime", "outtime"]).sort_values("icustay_id").reset_index(drop=True)
    assert lab["icustay_id"].is_unique
    stays = lab[["subject_id", "hadm_id", "icustay_id", "intime", "outtime"]].copy()
    stay_ids = lab["icustay_id"].to_numpy(np.int64)
    print(f"Cohort: {len(lab):,} stays")

    audit: dict = {}
    print("Reading CHARTEVENTS (slow step)")
    ch = read_filtered(find_table(a.data_dir, "CHARTEVENTS"), ["icustay_id", "itemid", "charttime", "value", "valuenum", "error"],
                       CHART_ITEMS, "icustay_id", set(stay_ids.tolist()))
    ch["icustay_id"] = ch["icustay_id"].astype("int64")
    ch = window(ch, stays, ["icustay_id"])
    print("Reading LABEVENTS")
    lb = read_filtered(find_table(a.data_dir, "LABEVENTS"), ["subject_id", "hadm_id", "itemid", "charttime", "valuenum"],
                       set(LAB_ITEM_TO_FEATURE), "hadm_id", set(lab["hadm_id"].astype(int)))
    lb["hadm_id"] = lb["hadm_id"].astype("int64")
    lb = window(lb, stays, ["subject_id", "hadm_id"])

    events = pd.concat([numeric_events(ch, CHART_ITEM_TO_FEATURE, audit, "chartevents", a.temperature_celsius_only),
                        numeric_events(lb, LAB_ITEM_TO_FEATURE, audit, "labevents", False),
                        gcs_events(ch, audit), support_events(ch, audit)], ignore_index=True)
    del ch, lb
    grid = hourly_grid(events, stay_ids)
    tensor = a4_transform(grid, st)

    age = pd.to_numeric(lab["age"], errors="coerce").to_numpy(np.float64)
    age = np.where(np.isfinite(age), age, age_med)
    g = lab["gender"].astype(str).str.strip().str.upper()
    demo = np.column_stack([(age - age_mean) / age_std, g.eq("F"), g.eq("M"), ~(g.eq("F") | g.eq("M"))]).astype(np.float32)
    assert demo.shape == (len(lab), 4) and np.isfinite(demo).all()

    np.save(a.out_dir / "mimic3_structured_24h_processed.npy", tensor, allow_pickle=False)
    np.save(a.out_dir / "mimic3_demographics_24h_processed.npy", demo, allow_pickle=False)
    np.save(a.out_dir / "mimic3_hourly_grid_raw.npy", grid, allow_pickle=False)
    lab.assign(row_index=np.arange(len(lab))).to_csv(a.out_dir / "mimic3_index_24h.csv", index=False)

    obs = np.isfinite(grid)
    ref_missing = {}
    p = a.v2_dir / "a3_hourly_summary.csv"
    if p.exists():
        ref_missing = pd.read_csv(p).set_index("feature")["missing_rate_percent"].to_dict()
    table = []
    for j, f in enumerate(FEATURES):
        table.append({"feature": f, "mimic3_missing_hour_cells_percent": round(100.0 * (1 - obs[:, :, j].mean()), 3),
                      "mimic4_missing_hour_cells_percent": round(float(ref_missing[f]), 3) if f in ref_missing else None,
                      "mimic3_stays_never_observed_percent": round(100.0 * (~obs[:, :, j].any(axis=1)).mean(), 3),
                      "mimic3_raw_median": round(float(np.nanmedian(grid[:, :, j])), 4) if obs[:, :, j].any() else None,
                      "mimic4_train_median": round(float(st["median"][j]), 4)})
    pd.DataFrame(table).to_csv(a.out_dir / "mimic3_feature_coverage_vs_mimic4.csv", index=False)
    report = {"n_stays": int(len(lab)), "tensor_shape": list(tensor.shape), "demographics_shape": list(demo.shape),
              "statistics_source": "MIMIC-IV V2 training split only (train_imputation_stats.csv, train_scaling_stats.csv, cohort_master.csv + index_24h.csv)",
              "mimic4_train_age": {"median": age_med, "mean": age_mean, "std": age_std},
              "mimic3_age": {"mean": float(age.mean()), "female_share": float(g.eq("F").mean())},
              "temperature_items": "Celsius only" if a.temperature_celsius_only else "Celsius + Fahrenheit converted",
              "event_audit": audit, "features_with_no_data": [f for j, f in enumerate(FEATURES) if not obs[:, :, j].any()],
              "text": "not built here; score with the missing-text setting or build radiology embeddings in a separate step"}
    (a.out_dir / "mimic3_feature_build_report.json").write_text(json.dumps(report, indent=2))
    print(pd.DataFrame(table).to_string(index=False))
    print(f"\nwrote {a.out_dir / 'mimic3_structured_24h_processed.npy'}  shape {tensor.shape}")
    if report["features_with_no_data"]:
        print("WARNING: no data at all for:", report["features_with_no_data"])


if __name__ == "__main__":
    main()
