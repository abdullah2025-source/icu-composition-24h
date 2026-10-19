#!/usr/bin/env python3
"""One-file Model Y family: locked Y0, ablations, statistics, and extensions.

Y0 is the primary result. Ablations change one choice. U3 trains two tree
models on temporal aggregates from the frozen arrays. U1 and U2 are included
and run only when their prediction files already exist. Test scores do not
replace Y0.

    python -u model_y_family.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, mean_absolute_error, roc_auc_score

PROJECT_ROOT = Path("/home/abdullah2025/fyp_project")
SEARCH_ROOT = PROJECT_ROOT / "outputs"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "model_y_family"
STACK_ROOT = PROJECT_ROOT / "outputs" / "model_y_v2_stacked_ensemble"
V2_DIR = PROJECT_ROOT / "outputs" / "pipeline_a_v2"
SEED = 20260923
EXPECTED_VAL = 7_417
EXPECTED_TEST = 14_887
BINARY = ("mortality_24h", "ventilation_onset_24h", "vasopressor_onset_24h", "readmission_30d_24h")
DETERIORATION = BINARY[:3]
LOS = "los_hours_24h"
TASKS = BINARY + (LOS,)
IDS = [f"{i:02d}" for i in range(33)]


def discover(ids: Sequence[str]) -> Dict[str, dict]:
    found = {mid: [] for mid in ids}
    for path in SEARCH_ROOT.glob("**/final_selected_configuration/test_predictions.csv"):
        mid = next((m.group(1) for part in reversed(path.parts) if (m := re.match(r"^(\d{2})_", part))), None)
        if mid in found:
            found[mid].append(path)
    out = {}
    for mid in ids:
        paths = sorted(found[mid])
        family7 = [p for p in paths if "baseline_family_07_advanced_multimodal_28_32" in str(p)]
        if len(family7) == 1:
            paths = family7
        if len(paths) != 1:
            raise FileNotFoundError(f"Model {mid} needs one tuned prediction file, found {paths}")
        frame = pd.read_csv(paths[0])
        required = {"stay_id", "subject_id", "split", "task", "target", "validity", "prediction"}
        if required - set(frame.columns) or frame.duplicated(["stay_id", "split", "task"]).any():
            raise ValueError(f"Bad prediction table: {paths[0]}")
        out[mid] = {"id": mid, "path": paths[0], "df": frame}
    return out


def task_frame(candidate: dict, split: str, task: str) -> pd.DataFrame:
    frame = candidate["df"]
    out = frame[(frame["split"] == split) & (frame["task"] == task)].sort_values("stay_id").reset_index(drop=True)
    if out.empty or out["stay_id"].duplicated().any():
        raise ValueError(f"Bad rows for {candidate['id']} {split} {task}")
    return out


def align(candidates: Dict[str, dict], ids: Sequence[str], split: str, task: str) -> pd.DataFrame:
    first = task_frame(candidates[ids[0]], split, task)
    out = first[["stay_id", "subject_id", "target", "validity"]].copy()
    out[f"pred_{ids[0]}"] = first["prediction"].to_numpy(float)
    for mid in ids[1:]:
        frame = task_frame(candidates[mid], split, task)
        if not np.array_equal(out["stay_id"], frame["stay_id"]):
            raise ValueError(f"Alignment failed for {mid} {split} {task}")
        out[f"pred_{mid}"] = frame["prediction"].to_numpy(float)
    return out


def scores(candidates: Dict[str, dict], ids: Sequence[str]) -> pd.DataFrame:
    rows = []
    for mid in ids:
        for task in TASKS:
            frame = task_frame(candidates[mid], "val", task)
            keep = frame["validity"].astype(bool).to_numpy() & np.isfinite(frame["target"]) & np.isfinite(frame["prediction"])
            y = frame.loc[keep, "target"].to_numpy(float)
            p = frame.loc[keep, "prediction"].to_numpy(float)
            if task in BINARY:
                rows.append({"model_id": mid, "task": task, "val_auprc": float(average_precision_score(y.astype(int), p)), "val_mae": np.nan})
            else:
                rows.append({"model_id": mid, "task": task, "val_auprc": np.nan, "val_mae": float(mean_absolute_error(y, p))})
    return pd.DataFrame(rows)


def rank_blend(columns: Sequence[np.ndarray], weights: Sequence[float] | None = None) -> np.ndarray:
    ranks = [pd.Series(col).rank(method="average").to_numpy() / float(len(col)) for col in columns]
    stacked = np.stack(ranks, axis=0)
    if weights is None:
        return stacked.mean(axis=0)
    w = np.asarray(weights, dtype=float)
    return np.average(stacked, axis=0, weights=w / w.sum())


def corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) == 0 or np.std(b) == 0:
        return 1.0
    return abs(float(np.corrcoef(a, b)[0, 1]))


def choose(candidates, table, ids, task: str, max_members: int, mode: str, blend: str) -> dict:
    aligned = align(candidates, ids, "val", task)
    y = aligned["target"].to_numpy(float)
    valid = aligned["validity"].astype(bool).to_numpy()
    if task in BINARY:
        ranking = table[table.task == task].sort_values("val_auprc", ascending=False)

        def score(members: Sequence[str]) -> float:
            cols = [aligned[f"pred_{mid}"].to_numpy(float) for mid in members]
            pred = rank_blend(cols) if blend == "rank_mean" else np.mean(np.stack(cols), axis=0)
            return float(average_precision_score(y[valid].astype(int), pred[valid]))

        selected = [str(ranking.iloc[0].model_id)]
        current = score(selected)
        for mid in ranking.model_id.astype(str).tolist()[1:]:
            if len(selected) >= max_members:
                break
            gain = score(selected + [mid]) - current
            pred_corr = [corr(aligned.loc[valid, f"pred_{old}"].to_numpy(float), aligned.loc[valid, f"pred_{mid}"].to_numpy(float)) for old in selected]
            err_corr = [corr(aligned.loc[valid, f"pred_{old}"].to_numpy(float) - y[valid], aligned.loc[valid, f"pred_{mid}"].to_numpy(float) - y[valid]) for old in selected]
            take = (mode == "y0" and (gain > 0.001 or (gain > 0 and min(pred_corr) < 0.90))) or (mode == "gain_only" and gain > 0.001) or (mode == "error_corr" and gain > 0.002 and min(err_corr) < 0.70)
            if take:
                selected.append(mid)
                current = score(selected)
        return {"task": task, "members": selected, "validation_score": current, "blend": blend}
    ranking = table[table.task == LOS].sort_values("val_mae")
    selected = [str(ranking.iloc[0].model_id)]
    current = float(mean_absolute_error(y[valid], aligned.loc[valid, f"pred_{selected[0]}"].to_numpy(float)))
    for mid in ranking.model_id.astype(str).tolist()[1:]:
        if len(selected) >= max_members:
            break
        cols = [aligned.loc[valid, f"pred_{member}"].to_numpy(float) for member in selected + [mid]]
        candidate = float(mean_absolute_error(y[valid], np.mean(np.stack(cols), axis=0)))
        gain = current - candidate
        pred_corr = [corr(aligned.loc[valid, f"pred_{old}"].to_numpy(float), aligned.loc[valid, f"pred_{mid}"].to_numpy(float)) for old in selected]
        err_corr = [corr(aligned.loc[valid, f"pred_{old}"].to_numpy(float) - y[valid], aligned.loc[valid, f"pred_{mid}"].to_numpy(float) - y[valid]) for old in selected]
        take = (mode == "y0" and (gain > 0.05 or (gain > 0 and min(pred_corr) < 0.90))) or (mode == "gain_only" and gain > 0.05) or (mode == "error_corr" and gain > 0.10 and min(err_corr) < 0.70)
        if take:
            selected.append(mid)
            current = candidate
    return {"task": LOS, "members": selected, "validation_score": current, "blend": "probability_mean"}


def apply_recipe(candidates, recipe: dict, split: str, weights: Sequence[float] | None = None) -> pd.DataFrame:
    aligned = align(candidates, recipe["members"], split, recipe["task"])
    cols = [aligned[f"pred_{mid}"].to_numpy(float) for mid in recipe["members"]]
    prediction = rank_blend(cols, weights) if recipe["blend"] == "rank_mean" else np.mean(np.stack(cols), axis=0)
    out = aligned[["stay_id", "subject_id", "target", "validity"]].copy()
    out["task"] = recipe["task"]
    out["prediction"] = prediction
    if recipe["task"] in BINARY:
        out["probability_mean"] = np.mean(np.stack(cols), axis=0)
    return out


def assemble(candidates, table, ids, max_members: int, mode: str, blend: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    recipes = {task: choose(candidates, table, ids, task, max_members, mode, blend) for task in BINARY}
    recipes[LOS] = choose(candidates, table, ids, LOS, max_members, mode, "probability_mean")
    val = pd.concat([apply_recipe(candidates, recipes[task], "val") for task in TASKS], ignore_index=True)
    test = pd.concat([apply_recipe(candidates, recipes[task], "test") for task in TASKS], ignore_index=True)
    return val, test, recipes


def metrics(frame: pd.DataFrame) -> dict:
    out = {}
    for task in TASKS:
        rows = frame[(frame.task == task) & frame.validity.astype(bool)]
        y = rows.target.to_numpy(float)
        p = rows.prediction.to_numpy(float)
        keep = np.isfinite(y) & np.isfinite(p)
        y, p = y[keep], p[keep]
        if task in BINARY:
            out[f"{task}_auprc"] = float(average_precision_score(y.astype(int), p))
            out[f"{task}_auroc"] = float(roc_auc_score(y.astype(int), p))
            out[f"{task}_brier"] = float(brier_score_loss(y.astype(int), np.clip(p, 0, 1)))
        else:
            out[f"{task}_mae"] = float(mean_absolute_error(y, p))
    out["mean_binary_auprc"] = float(np.mean([out[f"{task}_auprc"] for task in BINARY]))
    out["mean_binary_auroc"] = float(np.mean([out[f"{task}_auroc"] for task in BINARY]))
    return out


def bootstrap(left: pd.DataFrame, right: pd.DataFrame, task: str, reps: int, seed: int) -> dict:
    a = left[left.task == task].sort_values("stay_id")
    b = right[right.task == task].sort_values("stay_id")
    if not np.array_equal(a["stay_id"].to_numpy(), b["stay_id"].to_numpy()):
        raise ValueError(f"Bootstrap alignment failure for {task}")
    keep = a.validity.astype(bool).to_numpy() & b.validity.astype(bool).to_numpy()
    y = a.loc[keep, "target"].to_numpy(float)
    py = a.loc[keep, "prediction"].to_numpy(float)
    pr = b.loc[keep, "prediction"].to_numpy(float)
    subjects = a.loc[keep, "subject_id"].to_numpy()
    groups = {s: np.flatnonzero(subjects == s) for s in np.unique(subjects)}
    keys = np.array(list(groups))
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(reps):
        idx = np.concatenate([groups[s] for s in rng.choice(keys, size=len(keys), replace=True)])
        if task in BINARY:
            yy = y[idx].astype(int)
            if len(np.unique(yy)) < 2:
                continue
            diffs.append(average_precision_score(yy, py[idx]) - average_precision_score(yy, pr[idx]))
        else:
            diffs.append(mean_absolute_error(y[idx], py[idx]) - mean_absolute_error(y[idx], pr[idx]))
    arr = np.asarray(diffs)
    point = (average_precision_score(y.astype(int), py) - average_precision_score(y.astype(int), pr)) if task in BINARY else (mean_absolute_error(y, py) - mean_absolute_error(y, pr))
    return {"task": task, "delta": float(point), "ci95_low": float(np.quantile(arr, 0.025)), "ci95_high": float(np.quantile(arr, 0.975)), "prob_better": float(np.mean(arr > 0) if task in BINARY else np.mean(arr < 0))}


def reference(candidates, split: str) -> pd.DataFrame:
    parts = []
    for task in TASKS:
        frame = task_frame(candidates["32"], split, task)[["stay_id", "subject_id", "target", "validity", "prediction"]].copy()
        frame["task"] = task
        parts.append(frame)
    return pd.concat(parts, ignore_index=True)


def los_hybrid(y0_val, y0_test) -> tuple[pd.DataFrame, pd.DataFrame, dict] | None:
    val_path = STACK_ROOT / "ModelY2_validation_predictions.csv"
    test_path = STACK_ROOT / "ModelY2_test_predictions.csv"
    if not val_path.exists() or not test_path.exists():
        return None
    y2_val, y2_test = pd.read_csv(val_path), pd.read_csv(test_path)

    def mae(frame):
        rows = frame[(frame.task == LOS) & frame.validity.astype(bool)]
        return float(mean_absolute_error(rows.target, rows.prediction))

    use_y2 = mae(y2_val) < mae(y0_val)
    source_val, source_test = (y2_val, y2_test) if use_y2 else (y0_val, y0_test)
    return (
        pd.concat([y0_val[y0_val.task != LOS], source_val[source_val.task == LOS]], ignore_index=True),
        pd.concat([y0_test[y0_test.task != LOS], source_test[source_test.task == LOS]], ignore_index=True),
        {"selected_on_validation": "Y2" if use_y2 else "Y0", "validation_mae_y0": mae(y0_val), "validation_mae_y2": mae(y2_val)},
    )


def resolve_column(frame: pd.DataFrame, name: str) -> str:
    for candidate in (name, f"{name}_y", f"{name}_outcome", f"{name}_x"):
        if candidate in frame.columns:
            return candidate
    matches = [col for col in frame.columns if str(col).lower().replace(" ", "_") == name]
    if len(matches) == 1:
        return matches[0]
    raise KeyError(f"Missing {name}. Available columns: {list(frame.columns)}")


def load_cohort():
    structured = np.load(V2_DIR / "structured_24h_processed.npy", mmap_mode="r")
    demographics = np.load(V2_DIR / "demographics_24h_processed.npy", mmap_mode="r")
    alignment = pd.read_csv(V2_DIR / "a5_alignment_index_24h.csv")
    outcome_paths = [V2_DIR / "targets_24h" / "outcomes_24h_primary_cohort.csv", *sorted((V2_DIR / "targets_24h").glob("*.csv"))]
    outcomes = None
    for outcome_path in outcome_paths:
        if not outcome_path.exists():
            continue
        frame = pd.read_csv(outcome_path)
        if {"stay_id", "subject_id"}.issubset(frame.columns) and any("mortality" in str(col).lower() for col in frame.columns):
            outcomes = frame
            print(f"outcome table: {outcome_path}", flush=True)
            break
    if outcomes is None:
        raise FileNotFoundError("No outcome table with stay_id and a mortality column was found")
    cohort = alignment.merge(outcomes, on=["stay_id", "subject_id"], how="left", validate="one_to_one", suffixes=("", "_outcome"))
    for task in TASKS:
        cohort[task] = cohort[resolve_column(cohort, task)]
        cohort[f"{task}_valid"] = cohort[resolve_column(cohort, f"{task}_valid")]
    if "split_24h" not in cohort.columns:
        cohort["split_24h"] = cohort[resolve_column(cohort, "split")]
    return structured, demographics, cohort


def channel_map() -> dict:
    definition = json.loads((V2_DIR / "final_model_input_definition_24h.json").read_text())
    found = {}

    def walk(obj):
        if isinstance(obj, dict):
            name = obj.get("name") or obj.get("channel") or obj.get("variable") or obj.get("label")
            index = obj.get("index", obj.get("channel_index", obj.get("position")))
            if name is not None and index is not None:
                found[str(name).lower()] = int(index)
            for value in obj.values():
                walk(value)
        elif isinstance(obj, list):
            for i, value in enumerate(obj):
                if isinstance(value, str):
                    found.setdefault(value.lower(), i)
                else:
                    walk(value)

    walk(definition)
    aliases = {
        "hr": ("heart_rate", "heartrate", "hr"),
        "sbp": ("sbp", "systolic_bp", "systolic", "nibp_systolic"),
        "map": ("map", "mean_arterial_pressure", "mbp"),
        "rr": ("resp_rate", "respiratory_rate", "rr"),
        "spo2": ("spo2", "o2sat", "oxygen_saturation"),
        "fio2": ("fio2", "fio2_set"),
        "temp": ("temperature", "temp_c", "temp"),
        "wbc": ("wbc", "white_blood_cell", "leukocytes"),
        "platelets": ("platelet", "platelets"),
        "bilirubin": ("bilirubin", "total_bilirubin"),
        "creatinine": ("creatinine",),
        "bun": ("bun", "urea_nitrogen", "blood_urea_nitrogen"),
        "neutrophils": ("neutrophil", "neutrophils"),
        "lymphocytes": ("lymphocyte", "lymphocytes"),
        "gcs": ("gcs", "glasgow"),
        "ph": ("ph", "arterial_ph"),
        "sodium": ("sodium", "na"),
        "potassium": ("potassium",),
        "hematocrit": ("hematocrit", "hct"),
        "pao2": ("pao2", "po2"),
    }
    selected = {}
    for key, names in aliases.items():
        for name, index in found.items():
            if any(alias in name for alias in names) and 0 <= index < 28:
                selected[key] = index
                break
    return selected


def _series(values: np.ndarray, index: int | None) -> np.ndarray | None:
    if index is None:
        return None
    return values[:, :, index]


def _points(x: np.ndarray | None, cuts: list, higher_worse: bool) -> np.ndarray | None:
    if x is None:
        return None
    stat = np.nanmax(x, axis=1) if higher_worse else np.nanmin(x, axis=1)
    score = np.zeros(len(stat), dtype=np.float32)
    for point, threshold in cuts:
        score = np.where(stat > threshold if higher_worse else stat < threshold, point, score)
    return score


def clinical_matrix(structured, demographics) -> tuple[np.ndarray, list]:
    values = np.asarray(structured[:, :, :28], dtype=np.float32)
    masks = np.asarray(structured[:, :, 28:56], dtype=np.float32)
    mapping = channel_map()
    base = [
        np.nanmean(values[:, :6], axis=1), np.nanmean(values[:, -6:], axis=1),
        np.nanmean(values[:, -6:], axis=1) - np.nanmean(values[:, :6], axis=1),
        np.nanmean(values, axis=1), np.nanstd(values, axis=1), np.nanmean(masks, axis=1),
        np.asarray(demographics, dtype=np.float32),
    ]
    hr, sbp, mean_ap, rr = (_series(values, mapping.get(k)) for k in ("hr", "sbp", "map", "rr"))
    spo2, fio2, temp, wbc = (_series(values, mapping.get(k)) for k in ("spo2", "fio2", "temp", "wbc"))
    platelets, bilirubin, creatinine, bun = (_series(values, mapping.get(k)) for k in ("platelets", "bilirubin", "creatinine", "bun"))
    neutrophils, lymphocytes, gcs = (_series(values, mapping.get(k)) for k in ("neutrophils", "lymphocytes", "gcs"))
    ph, sodium, potassium, hematocrit, pao2 = (_series(values, mapping.get(k)) for k in ("ph", "sodium", "potassium", "hematocrit", "pao2"))
    clinical = []
    names = ["temporal_aggregates", "demographics"]
    if hr is not None and sbp is not None:
        clinical.append((np.nanmean(hr, axis=1) / np.clip(np.nanmean(sbp, axis=1), 1, None))[:, None]); names.append("shock_index")
    if spo2 is not None and fio2 is not None and rr is not None:
        clinical.append((np.nanmean(spo2, axis=1) / np.clip(np.nanmean(fio2, axis=1) * np.nanmean(rr, axis=1), 1e-3, None))[:, None]); names.append("rox")
    if neutrophils is not None and lymphocytes is not None:
        clinical.append((np.nanmean(neutrophils, axis=1) / np.clip(np.nanmean(lymphocytes, axis=1), 1e-3, None))[:, None]); names.append("nlr")
    if bun is not None and creatinine is not None:
        clinical.append((np.nanmean(bun, axis=1) / np.clip(np.nanmean(creatinine, axis=1), 1e-3, None))[:, None]); names.append("bun_creatinine")
    qsofa = []
    if rr is not None: qsofa.append((np.nanmax(rr, axis=1) >= 22).astype(np.float32))
    if sbp is not None: qsofa.append((np.nanmin(sbp, axis=1) <= 100).astype(np.float32))
    if gcs is not None: qsofa.append((np.nanmin(gcs, axis=1) < 15).astype(np.float32))
    if qsofa:
        clinical.append(np.sum(np.stack(qsofa), axis=0)[:, None]); names.append("qsofa")
    sirs = []
    if temp is not None: sirs.append(((np.nanmax(temp, axis=1) > 38) | (np.nanmin(temp, axis=1) < 36)).astype(np.float32))
    if hr is not None: sirs.append((np.nanmax(hr, axis=1) > 90).astype(np.float32))
    if rr is not None: sirs.append((np.nanmax(rr, axis=1) > 20).astype(np.float32))
    if wbc is not None: sirs.append(((np.nanmax(wbc, axis=1) > 12) | (np.nanmin(wbc, axis=1) < 4)).astype(np.float32))
    if sirs:
        clinical.append(np.sum(np.stack(sirs), axis=0)[:, None]); names.append("sirs")
    sofa_parts = [
        _points(pao2 / np.clip(fio2, 0.21, None) if pao2 is not None and fio2 is not None else None, [(1, 400), (2, 300), (3, 200), (4, 100)], False),
        _points(platelets, [(1, 150), (2, 100), (3, 50), (4, 20)], False),
        _points(bilirubin, [(1, 1.2), (2, 2), (3, 6), (4, 12)], True),
        _points(mean_ap, [(1, 70)], False),
        _points(gcs, [(1, 15), (2, 13), (3, 10), (4, 6)], False),
        _points(creatinine, [(1, 1.2), (2, 2), (3, 3.5), (4, 5)], True),
    ]
    sofa = [part for part in sofa_parts if part is not None]
    if sofa:
        clinical.append(np.sum(np.stack(sofa), axis=0)[:, None]); names.append("sofa_available_components")
    apache = [part for part in (
        _points(temp, [(1, 38.4), (2, 38.9), (3, 40.9)], True), _points(mean_ap, [(2, 159), (3, 129)], False),
        _points(hr, [(2, 109), (3, 139), (4, 179)], True), _points(rr, [(1, 24), (3, 34), (4, 49)], True),
        _points(ph, [(2, 7.49), (3, 7.59), (4, 7.69)], True), _points(sodium, [(1, 154), (2, 159), (3, 179)], True),
        _points(potassium, [(1, 5.4), (3, 6.9)], True), _points(hematocrit, [(2, 49.9), (4, 59.9)], True), _points(wbc, [(1, 19.9), (2, 39.9)], True),
    ) if part is not None]
    if apache:
        clinical.append(np.stack(apache, axis=1)); names.append("apache_ii_available_components")
    matrix = np.nan_to_num(np.concatenate(base + clinical, axis=1), nan=0.0, posinf=0.0, neginf=0.0)
    (OUTPUT_ROOT / "clinical_feature_coverage.json").write_text(json.dumps({"mapped_channels": mapping, "feature_groups": names}, indent=2))
    return matrix, names


def fit_tree(kind: str, task: str, x: np.ndarray, y: np.ndarray, valid: np.ndarray, seed: int, sample_weight: np.ndarray | None = None):
    import lightgbm as lgb
    import xgboost as xgb
    weight = None if sample_weight is None else sample_weight[valid]
    if kind == "xgb" and task in BINARY:
        model = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, tree_method="hist", random_state=seed, n_jobs=8)
        model.fit(x[valid], y[valid].astype(int), sample_weight=weight)
    elif kind == "xgb":
        model = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, tree_method="hist", random_state=seed, n_jobs=8)
        model.fit(x[valid], y[valid], sample_weight=weight)
    elif task in BINARY:
        model = lgb.LGBMClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, random_state=seed, n_jobs=8, verbose=-1)
        model.fit(x[valid], y[valid].astype(int), sample_weight=weight)
    else:
        model = lgb.LGBMRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, random_state=seed, n_jobs=8, verbose=-1)
        model.fit(x[valid], y[valid], sample_weight=weight)
    return model


def prediction_table(candidates, cohort, features, model, task: str) -> pd.DataFrame:
    rows = []
    order = pd.Series(np.arange(len(cohort)), index=cohort["stay_id"])
    pred = model.predict_proba(features)[:, 1] if task in BINARY else model.predict(features)
    for split_name in ("val", "test"):
        ref = task_frame(candidates["32"], split_name, task)
        piece = ref[["stay_id", "subject_id", "target", "validity"]].copy()
        piece["split"] = split_name
        piece["task"] = task
        piece["prediction"] = pred[order.loc[piece["stay_id"]].to_numpy()]
        rows.append(piece)
    return pd.concat(rows, ignore_index=True)


def train_tree_family(candidates, features, cohort, names: list, seeds: list, focal: bool) -> Dict[str, dict]:
    train = cohort["split_24h"].astype(str).to_numpy() == "train"
    added = {}
    for seed in seeds:
        for kind in ("xgb", "lgb"):
            rows = []
            for task in TASKS:
                y = cohort[task].to_numpy(float)
                valid = train & (cohort[f"{task}_valid"].to_numpy(int) == 1) & np.isfinite(y)
                weight = None
                if focal and task in ("vasopressor_onset_24h", "readmission_30d_24h"):
                    positive = y == 1
                    weight = np.ones(len(y), dtype=np.float32)
                    weight[positive] = 4.0
                model = fit_tree(kind, task, features, y, valid, seed, weight)
                rows.append(prediction_table(candidates, cohort, features, model, task))
            label = f"{'U2' if focal else 'U1'}{kind[0].upper()}_seed{seed}"
            added[label] = {"id": label, "path": "trained_in_model_y_family", "df": pd.concat(rows, ignore_index=True)}
            print(f"trained {label}", flush=True)
    return added


def train_neural_members(member_ids: list, seeds: list, focal: bool) -> Dict[str, dict]:
    import importlib
    import sys
    from contextlib import contextmanager
    sys.path.insert(0, str(PROJECT_ROOT))
    modules = []
    for path in sorted(PROJECT_ROOT.glob("family_*.py")):
        source = path.read_text(errors="ignore")
        if "def train_one" in source and "def fixed_params" in source:
            modules.append(importlib.import_module(path.stem))
    added = {}
    wanted = [mid for mid in member_ids if mid in {f"{i:02d}" for i in range(33)}]
    print(f"{'U2' if focal else 'U1'} neural members requested: {wanted}", flush=True)

    @contextmanager
    def focal_patch(module):
        if not focal or not hasattr(module, "F"):
            yield
            return
        original = module.F.binary_cross_entropy_with_logits
        def focal_loss(logits, targets, *args, **kwargs):
            bce = original(logits, targets, reduction="none")
            prob = module.torch.sigmoid(logits)
            pt = prob * targets + (1 - prob) * (1 - targets)
            return (0.25 * (1 - pt).pow(2) * bce).mean()
        module.F.binary_cross_entropy_with_logits = focal_loss
        try:
            yield
        finally:
            module.F.binary_cross_entropy_with_logits = original

    for module in modules:
        for mid in wanted:
            if mid not in getattr(module, "MODEL_CONFIGS", {}):
                continue
            for seed in seeds:
                label = f"{'U2N' if focal else 'U1N'}_{mid}_seed{seed}"
                run_dir = OUTPUT_ROOT / ("focal" if focal else "multiseed") / f"{mid}_seed{seed}"
                saved = run_dir / "test_predictions.csv"
                if saved.exists():
                    frame = pd.read_csv(saved)
                    if {"val", "test"}.issubset(set(frame["split"].astype(str))):
                        added[label] = {"id": label, "path": run_dir, "df": frame}
                        print(f"reused {label}", flush=True)
                        continue
                module.SEED = seed
                module.set_seed(seed)
                with focal_patch(module):
                    data = module.load_frozen_v2()
                    device = module.get_device()
                    params = module.fixed_params(mid)
                    run_dir.mkdir(parents=True, exist_ok=True)
                    model, score, epoch, _ = module.train_one(mid, params, data, device, run_dir=run_dir)
                    rows = []
                    for split_name, index in (("val", data.val_idx), ("test", data.test_idx)):
                        metrics_df, pred_df = module.predict_and_write(model, data, device, index, split_name, run_dir, params["batch_size"])
                        rows.append(pred_df)
                    added[label] = {"id": label, "path": run_dir, "df": pd.concat(rows, ignore_index=True)}
                    print(f"trained {label} validation={score:.5f} epoch={epoch}", flush=True)
    return added



def write_results(frames, recipes, candidates) -> None:
    ref = reference(candidates, "test")
    ref_metrics = metrics(ref)
    rows, stats = [], []
    for i, (name, (val, test)) in enumerate(frames.items()):
        val.to_csv(OUTPUT_ROOT / f"{name}_validation_predictions.csv", index=False)
        test.to_csv(OUTPUT_ROOT / f"{name}_test_predictions.csv", index=False)
        test_metrics = metrics(test)
        rows.append({"model": name, "validation_mean_auprc": metrics(val)["mean_binary_auprc"], "test_mean_auprc": test_metrics["mean_binary_auprc"], "test_mean_auroc": test_metrics["mean_binary_auroc"], "test_los_mae": test_metrics[f"{LOS}_mae"], "delta_vs_model32": test_metrics["mean_binary_auprc"] - ref_metrics["mean_binary_auprc"]})
        for j, task in enumerate(TASKS):
            item = bootstrap(test, ref, task, 1000, SEED + 100 * i + j)
            item["model"] = name
            stats.append(item)
    comparison = pd.DataFrame(rows).sort_values("validation_mean_auprc", ascending=False)
    statistics = pd.DataFrame(stats)
    comparison.to_csv(OUTPUT_ROOT / "ablation_comparison.csv", index=False)
    statistics.to_csv(OUTPUT_ROOT / "paired_bootstrap_vs_model32.csv", index=False)
    (OUTPUT_ROOT / "recipes.json").write_text(json.dumps(recipes, indent=2, default=str))
    print(comparison.to_string(index=False), flush=True)
    print(statistics.to_string(index=False), flush=True)


def add_pool(frames, recipes, candidates, added, label):
    if not added:
        print(f"{label} produced no models", flush=True)
        return
    pool = dict(candidates)
    pool.update(added)
    pool_scores = scores(pool, list(pool))
    val, test, recipe = assemble(pool, pool_scores, list(pool), 3, "y0", "rank_mean")
    frames[label] = (val, test)
    recipes[label] = recipe



def build_cumulative_chain(candidates, added_u1, added_u2, added_u3, y0_val, y0_test):
    frames, recipes = {}, {}
    frames["C0_Y0"] = (y0_val, y0_test)
    recipes["C0_Y0"] = {"note": "locked Y0 recipe"}
    pool = dict(candidates)
    stages = (("C1_Y0_U1", added_u1), ("C2_Y0_U1_U2", added_u2), ("C3_Y0_U1_U2_U3", added_u3))
    scores_c3 = None
    pool_c3 = pool
    for label, added in stages:
        if added:
            pool.update(added)
        pool_scores = scores(pool, list(pool))
        val, test, recipe = assemble(pool, pool_scores, list(pool), 3, "y0", "rank_mean")
        frames[label] = (val, test)
        recipes[label] = recipe
        scores_c3 = pool_scores
        pool_c3 = dict(pool)
    val, test, recipe = assemble(pool_c3, scores_c3, list(pool_c3), 3, "error_corr", "rank_mean")
    frames["C4_Y0_U1_U2_U3_U4"] = (val, test)
    recipes["C4_Y0_U1_U2_U3_U4"] = recipe
    hybrid = los_hybrid(val, test)
    if hybrid:
        frames["C5_Y0_U1_U2_U3_U4_U5"] = hybrid[:2]
        recipes["C5_Y0_U1_U2_U3_U4_U5"] = hybrid[2]
    else:
        print("C5 skipped: ModelY2 prediction files were not found", flush=True)
    return frames, recipes


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    candidates = discover(IDS)
    table = scores(candidates, IDS)
    specs = {
        "Y0_rank_max3": (3, "y0", "rank_mean"),
        "A1_best_single": (1, "y0", "rank_mean"),
        "A2_probability_mean_max3": (3, "y0", "probability_mean"),
        "A3_rank_max5": (5, "y0", "rank_mean"),
        "A4_gain_only_max3": (3, "gain_only", "rank_mean"),
        "A5_error_correlation_max3": (3, "error_corr", "rank_mean"),
    }
    frames, recipes = {}, {}
    for name, (max_members, mode, blend) in specs.items():
        val, test, recipe = assemble(candidates, table, IDS, max_members, mode, blend)
        frames[name] = (val, test)
        recipes[name] = recipe
    y0_val, y0_test = frames["Y0_rank_max3"]
    y0_recipes = recipes["Y0_rank_max3"]
    weights = {task: [float(table[(table.model_id == mid) & (table.task == task)].val_auprc.iloc[0]) for mid in y0_recipes[task]["members"]] for task in BINARY}
    parts = {split: [] for split in ("val", "test")}
    for task in BINARY:
        for split in parts:
            parts[split].append(apply_recipe(candidates, y0_recipes[task], split, weights[task]))
    for split in parts:
        parts[split].append(apply_recipe(candidates, y0_recipes[LOS], split))
    frames["A6_weighted_rank_same_members"] = (pd.concat(parts["val"], ignore_index=True), pd.concat(parts["test"], ignore_index=True))
    recipes["A6_weighted_rank_same_members"] = {"weights": weights}
    hybrid = los_hybrid(y0_val, y0_test)
    if hybrid:
        frames["A7_los_hybrid_validation_gated"] = hybrid[:2]
        recipes["A7_los_hybrid_validation_gated"] = hybrid[2]
    structured, demographics, cohort = load_cohort()
    features, feature_names = clinical_matrix(structured, demographics)
    print(f"clinical feature groups: {feature_names}", flush=True)
    write_results(frames, recipes, candidates)
    u3 = train_tree_family(candidates, features, cohort, feature_names, [SEED], False)
    add_pool(frames, recipes, candidates, u3, "U3_clinical_tree_pool")
    write_results(frames, recipes, candidates)
    members = sorted({mid for recipe in y0_recipes.values() for mid in recipe["members"]})
    seeds = [SEED + i for i in range(5)]
    u1 = {**train_tree_family(candidates, features, cohort, feature_names, seeds, False), **train_neural_members(members, seeds, False)}
    add_pool(frames, recipes, candidates, u1, "U1_multiseed_pool")
    write_results(frames, recipes, candidates)
    u2 = {**train_tree_family(candidates, features, cohort, feature_names, seeds, True), **train_neural_members(members, seeds[:1], True)}
    add_pool(frames, recipes, candidates, u2, "U2_focal_pool")
    write_results(frames, recipes, candidates)
    add_pool(frames, recipes, candidates, {**u1, **u2, **u3}, "U1_U2_U3_combined_pool")
    chain_frames, chain_recipes = build_cumulative_chain(candidates, u1, u2, u3, y0_val, y0_test)
    frames.update(chain_frames)
    recipes.update(chain_recipes)
    write_results(frames, recipes, candidates)
    print("Reported model is the highest validation mean AUPRC row, including C3 and C5.")
    print(f"output: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
