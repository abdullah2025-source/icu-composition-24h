#!/usr/bin/env python3
"""
MIMIC-III CareVue external cohort + six 24-hour task labels, aligned to the
Pipeline A V2 (MIMIC-IV) target definitions in pipeline_a_v2_a2_1_targets.py.

This script builds labels only. It does not extract features or score a model.

Why CareVue only: MetaVision patients (2008-2012) can also be in MIMIC-IV, so
only dbsource == 'carevue' stays (2001-2008) are an independent cohort.

Target definitions (same as MIMIC-IV V2; -1 = invalid, never a negative):

  mortality_24h          hospital death at or after the 24 h landmark (any time
                         later in the admission, not "within the next 24 h")
  ventilation_onset_24h  a new invasive-ventilation episode starts after the
                         landmark, anywhere in the rest of the ICU stay.
                         Invalid if an episode crosses the landmark, or if the
                         stay has no respiratory documentation at all.
  vasopressor_onset_24h  a new vasopressor interval starts after the landmark.
                         Invalid if an interval crosses the landmark.
  readmission_30d_24h    next ICU stay of the same patient starts within 30 days
                         after the index ICU outtime (any ICU stay, any dbsource)
  los_hours_24h          total ICU length of stay in hours (outtime - intime),
                         NOT the time remaining after the landmark
  deterioration_24h      OR of mortality / ventilation / vasopressor when all
                         three are valid, else -1

Cohort (same rules as MIMIC-IV V2): every adult ICU stay (not only the first
per admission) with ICU LOS >= 24 h; stays with death before the landmark and
hospital deaths with missing death time are excluded.

Differences from MIMIC-IV that cannot be removed (report them in the thesis):
  * CareVue charts infusions by charttime only (no start/end time), so
    vasopressor intervals are rebuilt from consecutive charted rates.
  * CareVue has no O2-delivery "Endotracheal tube" item; invasive ventilation
    is taken from ventilator mode (720) and device = 'Ventilator' (467).
  * MIMIC-III dates are shifted per patient, so there is no dataset end date;
    30-day follow-up is assumed complete for every stay.

Usage:
    python build_mimic3_carevue_labels_24h.py --data-dir /path/to/mimic-iii-1.4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

LANDMARK_HOURS = 24
READMISSION_DAYS = 30

# Five qualifying agents, same as MIMIC-IV V2. Inotropes (dobutamine 30042/30306,
# milrinone 30125) are excluded. CareVue itemids from MIT-LCP vasopressor concepts.
VASOPRESSOR_CV = {
    "norepinephrine": {30047, 30120},
    "epinephrine": {30044, 30119, 30309},
    "phenylephrine": {30127, 30128},
    "vasopressin": {30051, 42273, 42802},
    "dopamine": {30043, 30307},
}
VASO_ITEM_TO_AGENT = {i: a for a, ids in VASOPRESSOR_CV.items() for i in ids}

# Invasive ventilation, CareVue.
VENT_MODE_ITEM = 720            # Ventilator Mode
O2_DEVICE_ITEM = 467            # O2 Delivery Device ('Ventilator' = invasive)
EXTUBATION_ITEM = 640           # 'Extubated', 'Self Extubation'
# MIT-LCP ventilation-classification settings items (CareVue). Used only with
# --vent-definition mitlcp, because some are also charted during NIV.
VENT_SETTINGS_ITEMS = {
    445, 448, 449, 450, 1340, 1486, 1600,          # minute volume
    639, 654, 681, 682, 683, 684,                  # tidal volume
    218, 436, 535, 444, 459,                       # airway pressures
    221, 1, 1211, 1655, 2000,                      # inspiratory pressure
    543, 5865, 5866,                               # plateau / APRV
    60, 437, 505, 506, 686,                        # PEEP
    3459, 501, 502, 503, 223, 667, 668, 669, 670, 671, 672,
}
VENT_ALL_ITEMS = {VENT_MODE_ITEM, O2_DEVICE_ITEM, EXTUBATION_ITEM} | VENT_SETTINGS_ITEMS

BINARY_TASKS = ["mortality_24h", "ventilation_onset_24h", "vasopressor_onset_24h", "readmission_30d_24h", "deterioration_24h"]
ALL_TASKS = BINARY_TASKS + ["los_hours_24h"]


# --------------------------------------------------------------------------- IO
def find_table(root: Path, stem: str) -> Path:
    names = {stem.lower() + ext for ext in ("", ".csv", ".csv.gz")}
    hits = [p for p in root.rglob("*") if p.is_file() and p.name.lower() in names]
    if not hits:
        raise FileNotFoundError(f"Could not find {stem} under {root}")
    return sorted(hits, key=lambda p: len(p.parts))[0]


def read_small(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.lower() for c in df.columns]
    return df


def read_events(path: Path, cols: list[str], itemids: set[int], stay_ids: set[int], chunksize: int = 2_000_000) -> pd.DataFrame:
    """Stream a large event table, keeping only wanted columns, items and stays."""
    wanted = set(cols)
    parts, n_read = [], 0
    reader = pd.read_csv(path, usecols=lambda c: c.lower() in wanted, chunksize=chunksize, low_memory=False)
    for chunk in reader:
        chunk.columns = [c.lower() for c in chunk.columns]
        n_read += len(chunk)
        chunk = chunk[chunk["itemid"].isin(itemids)]
        chunk = chunk[pd.to_numeric(chunk["icustay_id"], errors="coerce").isin(stay_ids)]
        if not chunk.empty:
            parts.append(chunk)
        print(f"  {path.name}: {n_read:,} rows read, {sum(len(p) for p in parts):,} kept", end="\r", flush=True)
    print()
    if not parts:
        return pd.DataFrame(columns=cols)
    out = pd.concat(parts, ignore_index=True)
    out["icustay_id"] = out["icustay_id"].astype("int64")
    out["charttime"] = pd.to_datetime(out["charttime"], errors="coerce")
    return out.dropna(subset=["charttime"])


# --------------------------------------------------------------------------- cohort
def age_years(intime: pd.Series, dob: pd.Series) -> pd.Series:
    """Age without Timedelta overflow (MIMIC-III shifts DOB ~300 years for age > 89)."""
    age = intime.dt.year - dob.dt.year
    before_birthday = (intime.dt.month < dob.dt.month) | ((intime.dt.month == dob.dt.month) & (intime.dt.day < dob.dt.day))
    return (age - before_birthday.astype(int)).astype(float)


def build_cohort(patients: pd.DataFrame, admissions: pd.DataFrame, icustays: pd.DataFrame, drop_mixed_era: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    flow: dict = {}
    icu = icustays.copy()
    for c in ("intime", "outtime"):
        icu[c] = pd.to_datetime(icu[c], errors="coerce")
    icu = icu.dropna(subset=["subject_id", "hadm_id", "icustay_id", "intime"])
    all_icu = icu[["subject_id", "hadm_id", "icustay_id", "intime", "outtime", "dbsource"]].copy()   # for readmission
    flow["icu_stays_all"] = int(len(icu))

    adm = admissions.copy()
    for c in ("admittime", "dischtime", "deathtime"):
        adm[c] = pd.to_datetime(adm[c], errors="coerce")
    pat = patients.copy()
    pat["dob"] = pd.to_datetime(pat["dob"], errors="coerce")

    df = icu[icu["dbsource"].astype(str).str.lower().eq("carevue")].copy()
    flow["carevue_stays"] = int(len(df))
    # A patient who also has a MetaVision-era ICU stay (2008+) can be in MIMIC-IV under another
    # subject_id. The PhysioNet CareVue subset removes such patients; do the same on full MIMIC-III.
    mixed = set(icu.loc[~icu["dbsource"].astype(str).str.lower().eq("carevue"), "subject_id"])
    flow["carevue_stays_of_patients_with_metavision_era_stay"] = int(df["subject_id"].isin(mixed).sum())
    if drop_mixed_era:
        df = df[~df["subject_id"].isin(mixed)]
    flow["carevue_only_patients_stays"] = int(len(df))
    df = df.dropna(subset=["outtime"])
    df = df[df["outtime"] > df["intime"]]
    flow["with_valid_outtime"] = int(len(df))

    df = df.merge(adm[["subject_id", "hadm_id", "admittime", "dischtime", "deathtime", "hospital_expire_flag"]],
                  on=["subject_id", "hadm_id"], how="inner", validate="many_to_one")
    df = df.merge(pat[["subject_id", "gender", "dob"]], on="subject_id", how="left", validate="many_to_one")
    df["age"] = age_years(df["intime"], df["dob"])
    df = df[df["age"] >= 18]
    df["age"] = df["age"].clip(upper=91)          # MIMIC-IV caps anchor_age at 91
    flow["adult"] = int(len(df))

    df["icu_los_hours"] = (df["outtime"] - df["intime"]).dt.total_seconds() / 3600.0
    df = df[df["icu_los_hours"] >= LANDMARK_HOURS]
    flow["icu_los_ge_24h"] = int(len(df))

    df["landmark_time"] = df["intime"] + pd.Timedelta(hours=LANDMARK_HOURS)
    died_before = df["deathtime"].notna() & (df["deathtime"] < df["landmark_time"])
    flow["excluded_death_before_24h"] = int(died_before.sum())
    df = df[~died_before]
    missing_dt = df["hospital_expire_flag"].fillna(0).astype(int).eq(1) & df["deathtime"].isna()
    flow["excluded_hospital_death_missing_deathtime"] = int(missing_dt.sum())
    df = df[~missing_dt]
    flow["primary_cohort"] = int(len(df))
    flow["primary_subjects"] = int(df["subject_id"].nunique())
    if not df["icustay_id"].is_unique:
        raise AssertionError("icustay_id is not unique in the cohort")
    return df.reset_index(drop=True), all_icu, flow


# --------------------------------------------------------------------------- episodes
def build_episodes(ev: pd.DataFrame, keys: list[str], gap_hours: float) -> pd.DataFrame:
    """ev has columns keys + ['time', 'on'] (on = True: treatment documented; False: stop marker).

    A new episode starts at an 'on' row when it is the first row of the key, when
    the previous row is a stop marker, or when the gap to the previous row is
    >= gap_hours. An episode ends at the stop marker that directly follows it
    within the gap, otherwise at its last 'on' row.
    """
    cols = keys + ["start", "end"]
    if ev.empty:
        return pd.DataFrame(columns=cols)
    gap = pd.Timedelta(hours=gap_hours)
    ev = ev.assign(_off=(~ev["on"]).astype(int)).sort_values(keys + ["time", "_off"]).reset_index(drop=True)
    g = ev.groupby(keys, sort=False)
    prev_time, prev_on = g["time"].shift(), g["on"].shift()
    delta = ev["time"] - prev_time
    new = ev["on"] & (prev_on.isna() | prev_on.eq(False) | (delta >= gap))
    ev["ep"] = np.where(ev["on"], new.cumsum(), np.nan)
    ev["ep_ff"] = ev.groupby(keys, sort=False)["ep"].ffill()
    closing = (~ev["on"]) & prev_on.eq(True) & (delta < gap)
    stop_time = ev[closing].groupby("ep_ff")["time"].min()
    on_rows = ev[ev["on"]]
    eps = on_rows.groupby("ep").agg(**{k: (k, "first") for k in keys}, start=("time", "min"), last=("time", "max"))
    eps["end"] = stop_time.reindex(eps.index).fillna(eps["last"])
    return eps[cols].reset_index(drop=True)


def union_intervals(iv: pd.DataFrame, key: str, bridge_hours: float) -> pd.DataFrame:
    """Merge overlapping intervals of one stay; intervals closer than bridge_hours are joined."""
    if iv.empty:
        return iv
    iv = iv.sort_values([key, "start", "end"]).reset_index(drop=True)
    run_end = iv.groupby(key)["end"].cummax()
    prev_end = run_end.groupby(iv[key]).shift()
    new = prev_end.isna() | ((iv["start"] - prev_end) >= pd.Timedelta(hours=bridge_hours))
    grp = new.cumsum()
    return iv.groupby(grp).agg(**{key: (key, "first")}, start=("start", "min"), end=("end", "max")).reset_index(drop=True)


def onset_flags(intervals: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    """Per stay: does an interval cross the landmark, and does one start after it (same strict rules as V2)."""
    out = cohort[["icustay_id"]].copy()
    if intervals.empty:
        return out.assign(crosses=False, post_onset=False, n_intervals=0)
    iv = intervals.merge(cohort[["icustay_id", "landmark_time"]], on="icustay_id", how="inner")
    iv["crosses"] = (iv["start"] < iv["landmark_time"]) & (iv["end"] > iv["landmark_time"])
    iv["post"] = iv["start"] > iv["landmark_time"]
    agg = iv.groupby("icustay_id").agg(crosses=("crosses", "any"), post_onset=("post", "any"), n_intervals=("start", "size")).reset_index()
    out = out.merge(agg, on="icustay_id", how="left")
    out["crosses"] = out["crosses"].fillna(False).astype(bool)
    out["post_onset"] = out["post_onset"].fillna(False).astype(bool)
    out["n_intervals"] = out["n_intervals"].fillna(0).astype(int)
    return out


def clip_to_stay(ev: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    """Keep events inside [intime, outtime), the V2 full-stay target window."""
    ev = ev.merge(cohort[["icustay_id", "intime", "outtime"]], on="icustay_id", how="inner")
    return ev[(ev["charttime"] >= ev["intime"]) & (ev["charttime"] < ev["outtime"])].drop(columns=["intime", "outtime"])


# --------------------------------------------------------------------------- ventilation
def ventilation_intervals(chart: pd.DataFrame, definition: str, gap_hours: float) -> tuple[pd.DataFrame, set[int], dict]:
    ch = chart.copy()
    val = ch["value"].astype(str).str.strip().str.lower()
    is_mode = ch["itemid"].eq(VENT_MODE_ITEM) & ch["value"].notna() & ~val.eq("other/remarks")
    is_dev_vent = ch["itemid"].eq(O2_DEVICE_ITEM) & val.eq("ventilator")
    is_intub = ch["itemid"].eq(EXTUBATION_ITEM) & val.eq("intubated")          # item 640 is a general events item
    is_extub = ch["itemid"].eq(EXTUBATION_ITEM) & val.isin(["extubated", "self extubation"])
    is_setting = ch["itemid"].isin(VENT_SETTINGS_ITEMS) & ch["value"].notna()
    on = is_mode | is_dev_vent | is_intub | (is_setting if definition == "mitlcp" else False)
    # Stop markers: extubation, or an O2 device that means the patient is off the ventilator.
    # 'Other/Remarks' is uninformative and 'T-Piece' is a breathing trial on a still-intubated
    # patient, so neither ends an episode.
    dev_named = ch["itemid"].eq(O2_DEVICE_ITEM) & ch["value"].notna() & ~val.isin(["ventilator", "other/remarks", "t-piece", "nan", ""])
    off = is_extub | dev_named
    documented = set(ch.loc[is_mode | is_dev_vent | is_intub | is_extub | dev_named | (ch["itemid"].eq(VENT_MODE_ITEM) & ch["value"].notna()),
                            "icustay_id"].astype(int))
    ev = pd.concat([ch.loc[on, ["icustay_id", "charttime"]].assign(on=True),
                    ch.loc[off & ~on, ["icustay_id", "charttime"]].assign(on=False)], ignore_index=True)
    ev = ev.rename(columns={"charttime": "time"}).drop_duplicates()
    eps = build_episodes(ev, ["icustay_id"], gap_hours)
    audit = {"definition": definition, "gap_hours": gap_hours, "rows_mode_720": int(is_mode.sum()), "rows_device_ventilator_467": int(is_dev_vent.sum()),
             "rows_intubated_640": int(is_intub.sum()), "rows_extubated_640": int(is_extub.sum()), "rows_o2_device_off_markers_467": int(dev_named.sum()),
             "rows_settings_items": int(is_setting.sum()), "rows_stop_markers": int((off & ~on).sum()),
             "episodes": int(len(eps)), "stays_with_any_respiratory_documentation": len(documented)}
    return eps, documented, audit


# --------------------------------------------------------------------------- vasopressors
def vasopressor_intervals(inp: pd.DataFrame, gap_hours: float, bridge_hours: float) -> tuple[pd.DataFrame, dict]:
    iv = inp.copy()
    iv["agent"] = iv["itemid"].map(VASO_ITEM_TO_AGENT)
    rate = pd.to_numeric(iv.get("rate"), errors="coerce")
    amount = pd.to_numeric(iv.get("amount"), errors="coerce")
    stopped = iv.get("stopped", pd.Series("", index=iv.index)).astype(str).str.strip().str.lower()
    is_stop = stopped.isin(["stopped", "d/c'd"]) | rate.eq(0)
    is_on = ~is_stop & ((rate > 0) | (rate.isna() & (amount > 0)))
    ev = pd.concat([iv.loc[is_on, ["icustay_id", "agent", "charttime"]].assign(on=True),
                    iv.loc[is_stop, ["icustay_id", "agent", "charttime"]].assign(on=False)], ignore_index=True)
    ev = ev.rename(columns={"charttime": "time"}).drop_duplicates()
    per_agent = build_episodes(ev, ["icustay_id", "agent"], gap_hours)
    merged = union_intervals(per_agent[["icustay_id", "start", "end"]], "icustay_id", bridge_hours)
    audit = {"gap_hours": gap_hours, "bridge_hours": bridge_hours, "rows_active": int(is_on.sum()), "rows_stop": int(is_stop.sum()),
             "rows_ignored_no_rate_or_amount": int((~is_on & ~is_stop).sum()),
             "rows_by_itemid": {str(k): int(v) for k, v in iv["itemid"].value_counts().sort_index().items()},
             "intervals_per_agent": {str(k): int(v) for k, v in per_agent["agent"].value_counts().items()} if len(per_agent) else {},
             "intervals_after_union": int(len(merged))}
    return merged, audit


# --------------------------------------------------------------------------- labels
def add_labels(df: pd.DataFrame, all_icu: pd.DataFrame, vent: pd.DataFrame, vent_documented: set[int], vaso: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # 1. mortality: hospital death at or after the landmark (cohort already excludes earlier deaths).
    out["mortality_24h"] = (out["deathtime"].notna() & (out["deathtime"] >= out["landmark_time"])).astype(int)
    out["mortality_24h_valid"] = 1

    # 5. LOS: total ICU hours, as in V2 (the models report expm1(pred) + 24).
    out["los_hours_24h"] = out["icu_los_hours"].astype(float)
    out["los_hours_24h_valid"] = 1

    # 4. readmission: chronologically next ICU stay of the subject, any dbsource / any length.
    fut = all_icu.sort_values(["subject_id", "intime", "icustay_id"]).copy()
    fut["next_intime"] = fut.groupby("subject_id")["intime"].shift(-1)
    out = out.merge(fut[["icustay_id", "next_intime"]], on="icustay_id", how="left", validate="one_to_one")
    days = (out["next_intime"] - out["outtime"]).dt.total_seconds() / 86400.0
    out["days_to_next_icu"] = days
    out["readmission_30d_24h"] = (days.notna() & (days > 0) & (days <= READMISSION_DAYS)).astype(int)
    # No dataset end date exists in MIMIC-III (per-patient date shift): follow-up is treated as complete.
    out["readmission_30d_24h_valid"] = 1

    # 2. ventilation onset.
    v = vent.rename(columns={"crosses": "vent_crosses_landmark", "post_onset": "vent_post_onset", "n_intervals": "vent_episodes"})
    out = out.merge(v, on="icustay_id", how="left", validate="one_to_one")
    out["vent_documentation_present"] = out["icustay_id"].isin(vent_documented)
    v_valid = out["vent_documentation_present"] & ~out["vent_crosses_landmark"]
    out["ventilation_onset_24h_valid"] = v_valid.astype(int)
    out["ventilation_onset_24h"] = np.where(v_valid, out["vent_post_onset"].astype(int), -1)

    # 3. vasopressor onset: no interval at all is a valid negative.
    p = vaso.rename(columns={"crosses": "vaso_crosses_landmark", "post_onset": "vaso_post_onset", "n_intervals": "vaso_intervals"})
    out = out.merge(p, on="icustay_id", how="left", validate="one_to_one")
    p_valid = ~out["vaso_crosses_landmark"]
    out["vasopressor_onset_24h_valid"] = p_valid.astype(int)
    out["vasopressor_onset_24h"] = np.where(p_valid, out["vaso_post_onset"].astype(int), -1)

    # 6. deterioration: OR of the three components, valid only when all three are valid.
    comp = ["mortality_24h", "ventilation_onset_24h", "vasopressor_onset_24h"]
    d_valid = np.logical_and.reduce([out[c + "_valid"].eq(1).to_numpy() for c in comp])
    d_event = np.logical_or.reduce([out[c].eq(1).to_numpy() for c in comp])
    out["deterioration_24h_valid"] = d_valid.astype(int)
    out["deterioration_24h"] = np.where(d_valid, d_event.astype(int), -1)
    return out


def summarize(df: pd.DataFrame) -> dict:
    out = {}
    for t in BINARY_TASKS:
        valid = df[t + "_valid"].eq(1)
        n, e = int(valid.sum()), int((valid & df[t].eq(1)).sum())
        out[t] = {"valid": n, "invalid": int((~valid).sum()), "events": e, "prevalence": round(e / n, 6) if n else None}
    los = df["los_hours_24h"]
    out["los_hours_24h"] = {"valid": int(los.notna().sum()), "min": float(los.min()), "median": float(los.median()), "mean": float(los.mean()),
                            "p99": float(los.quantile(0.99)), "max": float(los.max())}
    return out


def qc(df: pd.DataFrame) -> None:
    for t in BINARY_TASKS:
        v = df[t + "_valid"].eq(1)
        assert df.loc[v, t].isin([0, 1]).all(), f"{t}: valid rows must be 0/1"
        assert df.loc[~v, t].eq(-1).all(), f"{t}: invalid rows must be -1"
    assert (df["los_hours_24h"] >= LANDMARK_HOURS).all(), "LOS below 24 h in cohort"
    assert (df["mortality_24h"].eq(1) <= df["hospital_expire_flag"].fillna(0).eq(1) | df["deathtime"].notna()).all()
    d = df["deterioration_24h_valid"].eq(1)
    assert (df.loc[d, "deterioration_24h"] >= df.loc[d, "mortality_24h"]).all(), "deterioration must include mortality"


# MIMIC-IV V2 reference prevalences among valid rows (handoff specification), for a quick sanity comparison.
MIMIC4_REFERENCE = {"mortality_24h": 8673 / 74572, "ventilation_onset_24h": 3978 / 47300, "vasopressor_onset_24h": 4882 / 64498,
                    "readmission_30d_24h": 10801 / 74572, "deterioration_24h": 6836 / 43619}
MIMIC4_VALID_SHARE = {"ventilation_onset_24h": 47300 / 74572, "vasopressor_onset_24h": 64498 / 74572, "deterioration_24h": 43619 / 74572}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/external_mimic3"))
    ap.add_argument("--vent-definition", choices=["strict", "mitlcp"], default="strict",
                    help="strict: ventilator mode (720) or device = Ventilator (467). mitlcp: also ventilator-settings items.")
    ap.add_argument("--vent-gap-hours", type=float, default=14.0, help="Documentation gap that starts a new ventilation episode (V2 uses 14).")
    ap.add_argument("--vaso-gap-hours", type=float, default=8.0, help="Charting gap that starts a new CareVue infusion interval (MIT-LCP uses 8).")
    ap.add_argument("--vaso-bridge-hours", type=float, default=1.0, help="Intervals closer than this are joined (V2 uses 1).")
    ap.add_argument("--keep-mixed-era-patients", action="store_true",
                    help="Keep CareVue stays of patients who also have a MetaVision-era stay (default: drop them, they can be in MIMIC-IV).")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    print("Reading PATIENTS / ADMISSIONS / ICUSTAYS")
    cohort, all_icu, flow = build_cohort(read_small(find_table(a.data_dir, "PATIENTS")), read_small(find_table(a.data_dir, "ADMISSIONS")),
                                         read_small(find_table(a.data_dir, "ICUSTAYS")), drop_mixed_era=not a.keep_mixed_era_patients)
    print(json.dumps(flow, indent=2))
    stay_ids = set(cohort["icustay_id"].astype(int))

    print("Reading CHARTEVENTS (ventilation items; this is the slow step)")
    chart = read_events(find_table(a.data_dir, "CHARTEVENTS"), ["icustay_id", "itemid", "charttime", "value"], VENT_ALL_ITEMS, stay_ids)
    chart = clip_to_stay(chart, cohort)
    print("Reading INPUTEVENTS_CV (vasopressors)")
    inp = read_events(find_table(a.data_dir, "INPUTEVENTS_CV"), ["icustay_id", "itemid", "charttime", "rate", "amount", "stopped"],
                      set(VASO_ITEM_TO_AGENT), stay_ids)
    inp = clip_to_stay(inp, cohort)

    vent_iv, vent_doc, vent_audit = ventilation_intervals(chart, a.vent_definition, a.vent_gap_hours)
    vaso_iv, vaso_audit = vasopressor_intervals(inp, a.vaso_gap_hours, a.vaso_bridge_hours)
    labeled = add_labels(cohort, all_icu, onset_flags(vent_iv, cohort), vent_doc, onset_flags(vaso_iv, cohort))
    qc(labeled)

    labeled["stay_id"] = labeled["icustay_id"]        # MIMIC-IV style key for downstream scripts
    labeled["split"] = "external_mimic3_carevue"
    keep = ["subject_id", "hadm_id", "icustay_id", "stay_id", "split", "dbsource", "gender", "age", "intime", "outtime", "landmark_time",
            "admittime", "dischtime", "deathtime", "hospital_expire_flag", "icu_los_hours"]
    for t in ALL_TASKS:
        keep += [t, t + "_valid"]
    keep += ["days_to_next_icu", "vent_documentation_present", "vent_crosses_landmark", "vent_post_onset", "vent_episodes",
             "vaso_crosses_landmark", "vaso_post_onset", "vaso_intervals"]
    labeled = labeled.sort_values("icustay_id")
    dest = a.out_dir / "mimic3_carevue_labels_aligned_24h.csv"
    labeled[keep].to_csv(dest, index=False)
    vent_iv.to_csv(a.out_dir / "mimic3_carevue_ventilation_episodes.csv", index=False)
    vaso_iv.to_csv(a.out_dir / "mimic3_carevue_vasopressor_intervals.csv", index=False)

    stats = summarize(labeled)
    compare = {t: {"mimic3_prevalence": stats[t]["prevalence"], "mimic4_prevalence": round(MIMIC4_REFERENCE[t], 6),
                   "mimic3_valid_share": round(stats[t]["valid"] / len(labeled), 4), "mimic4_valid_share": round(MIMIC4_VALID_SHARE.get(t, 1.0), 4)}
               for t in BINARY_TASKS}
    report = {"cohort_flow": flow, "labels": stats, "comparison_with_mimic4_v2": compare, "ventilation_audit": vent_audit, "vasopressor_audit": vaso_audit,
              "settings": {"landmark_hours": LANDMARK_HOURS, "readmission_days": READMISSION_DAYS, "vent_definition": a.vent_definition,
                           "vent_gap_hours": a.vent_gap_hours, "vaso_gap_hours": a.vaso_gap_hours, "vaso_bridge_hours": a.vaso_bridge_hours},
              "notes": ["Targets follow pipeline_a_v2_a2_1_targets.py; -1 = invalid.",
                        "Readmission follow-up assumed complete (MIMIC-III has no dataset end date).",
                        "Vasopressor intervals are rebuilt from charted rates (CareVue has no start/end times)."]}
    (a.out_dir / "mimic3_carevue_label_summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"labels": stats, "comparison_with_mimic4_v2": compare}, indent=2))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
