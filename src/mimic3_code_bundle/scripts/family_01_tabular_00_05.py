
#!/usr/bin/env python3
"""
Pipeline A V2 — Family 1: Tabular / Tree / Linear Baselines (Models 00–05)

Models
------
00 XGBoost
01 Logistic Regression
02 Regularized Logistic Regression
03 Random Forest
04 Extra Trees
05 LightGBM

Design
------
- Consumes ONLY the frozen V2 model package.
- No preprocessing/statistics are recomputed from validation/test.
- Uses the frozen split_24h from a5_alignment_index_24h.csv.
- Uses the canonical 24 x 110 structured tensor.
- Uses a documented, deterministic tabular summary:
      [last_hour, mean_24h, std_24h, min_24h, max_24h] per channel
  => 550 structured features
  + 4 frozen demographic features
  => 554 tabular features.
- Text is not added to this tree/linear family because the V2 handoff
  explicitly separates tree/linear tabular handling from the multimodal
  architecture family. The exact tabular summary is recorded in metadata.
- Five primary tasks are fitted independently:
      mortality, ventilation onset, vasopressor onset,
      30-day readmission, LOS.
- Models 01–02 use LogisticRegression for binary tasks and a matching
  linear-regression counterpart for LOS (Ridge / ElasticNet). The frozen
  C parameter is mapped to alpha=1/C for the LOS counterpart.
- Ventilation and vasopressor rows with validity=0 are excluded from
  fitting/evaluation.
- Deterioration is NEVER trained. It is derived at evaluation time from
  mortality/ventilation/vasopressor probabilities.
- Fixed mode uses the preregistered settings.
- Tuned mode uses Optuna on TRAIN + validation only; TEST is untouched.
- The tuned objective is the mean validation AUPRC across the four
  binary primary tasks, matching the frozen benchmark selection rule.
- Test evaluation is performed only after the configuration is frozen.

CHI / AMD
---------
Expected project environment:
    /home/abdullah2025/fyp_project/mamba_env

The script does not require NVIDIA CUDA. These classical models are CPU
or CPU-parallel by default. Deep model families will use the CHI AMD GPU
environment separately.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import optuna
except ImportError as exc:
    optuna = None

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None


# ---------------------------------------------------------------------
# Fixed V2 paths from the canonical CHI specification
# ---------------------------------------------------------------------

PROJECT_ROOT = Path("/home/abdullah2025/fyp_project")
V2_DIR = PROJECT_ROOT / "outputs" / "pipeline_a_v2"

ALIGNMENT_PATH = V2_DIR / "a5_alignment_index_24h.csv"
STRUCTURED_PATH = V2_DIR / "structured_24h_processed.npy"
DEMOGRAPHICS_PATH = V2_DIR / "demographics_24h_processed.npy"
INPUT_METADATA_PATH = V2_DIR / "final_model_input_metadata_24h.csv"
INPUT_DEFINITION_PATH = V2_DIR / "final_model_input_definition_24h.json"
OUTCOMES_PATH = V2_DIR / "targets_24h" / "outcomes_24h_primary_cohort.csv"

RESULTS_ROOT = PROJECT_ROOT / "outputs" / "baseline_family_01_tabular"
SEED = 20260923

EXPECTED_N = 74_572
EXPECTED_TRAIN = 52_268
EXPECTED_VAL = 7_417
EXPECTED_TEST = 14_887
EXPECTED_SHAPE = (EXPECTED_N, 24, 110)

BINARY_TASKS = (
    "mortality_24h",
    "ventilation_onset_24h",
    "vasopressor_onset_24h",
    "readmission_30d_24h",
)
ALL_PRIMARY_TASKS = BINARY_TASKS + ("los_hours_24h",)

VALID_COL = {
    "mortality_24h": "mortality_24h_valid",
    "ventilation_onset_24h": "ventilation_onset_24h_valid",
    "vasopressor_onset_24h": "vasopressor_onset_24h_valid",
    "readmission_30d_24h": "readmission_30d_24h_valid",
    "los_hours_24h": "los_hours_24h_valid",
}


@dataclass(frozen=True)
class FamilyConfig:
    model_id: str
    name: str
    fixed_params: Dict[str, Any]


# Registered Optuna search spaces from the V2 canonical specification.
# "depth 8–None" for RF/Extra Trees is not a continuous numeric range, so
# the implementation exposes a documented categorical set ending in None.
REGISTERED_TUNING_SPACES: Dict[str, Dict[str, Any]] = {
    "00": {
        "max_depth": "int[4,10]",
        "learning_rate": "float[0.01,0.15]",
        "n_estimators": "int[400,2000]",
        "subsample": "float[0.6,1.0]",
        "colsample_bytree": "fixed=0.8",
    },
    "01": {
        "C": "logfloat[1e-3,20]",
        "penalty": "fixed=l2",
    },
    "02": {
        "C": "logfloat[1e-3,10]",
        "l1_ratio": "float[0.1,0.9]",
        "penalty": "fixed=elasticnet",
    },
    "03": {
        "n_estimators": "int[300,1200]",
        "max_depth": "categorical=[8,10,12,14,16,20,24,32,None]",
        "min_samples_leaf": "int[2,12]",
        "max_features": "fixed=sqrt",
    },
    "04": {
        "n_estimators": "int[300,1200]",
        "max_depth": "categorical=[8,10,12,14,16,20,24,32,None]",
        "min_samples_leaf": "int[2,12]",
        "max_features": "fixed=sqrt",
    },
    "05": {
        "num_leaves": "int[31,127]",
        "learning_rate": "float[0.01,0.12]",
        "n_estimators": "fixed=1200",
        "min_data_in_leaf": "int[20,100]",
    },
}

FAMILY_CONFIGS: Dict[str, FamilyConfig] = {
    "00": FamilyConfig(
        "00",
        "XGBoost",
        {
            "max_depth": 6,
            "learning_rate": 0.05,
            "n_estimators": 1000,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
    ),
    "01": FamilyConfig(
        "01",
        "Logistic Regression",
        {
            "C": 1.0,
            "penalty": "l2",
            "max_iter": 2000,
        },
    ),
    "02": FamilyConfig(
        "02",
        "Regularized Logistic Regression",
        {
            "C": 0.3,
            "penalty": "elasticnet",
            "l1_ratio": 0.5,
            "max_iter": 2000,
        },
    ),
    "03": FamilyConfig(
        "03",
        "Random Forest",
        {
            "n_estimators": 600,
            "max_depth": 14,
            "min_samples_leaf": 4,
            "max_features": "sqrt",
        },
    ),
    "04": FamilyConfig(
        "04",
        "Extra Trees",
        {
            "n_estimators": 600,
            "max_depth": 14,
            "min_samples_leaf": 4,
            "max_features": "sqrt",
        },
    ),
    "05": FamilyConfig(
        "05",
        "LightGBM",
        {
            "num_leaves": 63,
            "learning_rate": 0.04,
            "n_estimators": 1200,
            "min_data_in_leaf": 50,
        },
    ),
}


@dataclass
class PreparedData:
    X: np.ndarray
    demographics: np.ndarray
    stay_id: np.ndarray
    subject_id: np.ndarray
    split: np.ndarray
    outcomes: pd.DataFrame
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required frozen V2 artifact not found: {path}")


def _set_seeds(seed: int = SEED) -> None:
    np.random.seed(seed)


def _validity_mask(df: pd.DataFrame, task: str) -> np.ndarray:
    """Use explicit validity if present; otherwise only allow documented
    always-valid tasks to default to all-valid."""
    col = VALID_COL[task]
    if col in df.columns:
        return df[col].to_numpy(dtype=np.int8) == 1

    if task in {"mortality_24h", "readmission_30d_24h", "los_hours_24h"}:
        return np.ones(len(df), dtype=bool)

    raise KeyError(
        f"Missing required validity column '{col}' for task '{task}'. "
        "V2 requires explicit validity for onset tasks."
    )


def _check_alignment_columns(df: pd.DataFrame) -> None:
    required = {"stay_id", "subject_id", "split_24h"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Missing required V2 alignment columns: {missing}")


def _check_outcome_columns(df: pd.DataFrame) -> None:
    required = {"stay_id", "subject_id", *ALL_PRIMARY_TASKS}
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Missing required V2 outcome columns: {missing}")


def _tabularize_structured(
    X_seq: np.ndarray,
    batch_size: int = 4096,
) -> np.ndarray:
    """
    Frozen deterministic tabular summary for models 00–05.

    For each of the 110 channels:
        1) last_hour
        2) mean over 24 hours
        3) std over 24 hours
        4) min over 24 hours
        5) max over 24 hours

    No statistic is fitted from train/validation/test. This is a pure
    architecture-specific transformation of the frozen V2 tensor.
    """
    n = X_seq.shape[0]
    out = np.empty((n, 110 * 5), dtype=np.float32)

    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        chunk = np.asarray(X_seq[start:stop], dtype=np.float32)

        last_hour = chunk[:, -1, :]
        mean = np.mean(chunk, axis=1, dtype=np.float32)
        std = np.std(chunk, axis=1, dtype=np.float32)
        min_ = np.min(chunk, axis=1)
        max_ = np.max(chunk, axis=1)

        out[start:stop] = np.concatenate(
            [last_hour, mean, std, min_, max_],
            axis=1,
        ).astype(np.float32, copy=False)

    return out


def load_frozen_v2() -> PreparedData:
    for path in (
        ALIGNMENT_PATH,
        STRUCTURED_PATH,
        DEMOGRAPHICS_PATH,
        INPUT_METADATA_PATH,
        INPUT_DEFINITION_PATH,
        OUTCOMES_PATH,
    ):
        _require_file(path)

    alignment = pd.read_csv(ALIGNMENT_PATH)
    outcomes = pd.read_csv(OUTCOMES_PATH)

    # The canonical contract uses the alignment file for identity/split and
    # outcome values from the frozen targets package. Join by stay_id, never
    # blindly by row position.
    _check_alignment_columns(alignment)

    _check_outcome_columns(outcomes)

    outcomes_required = [
        "stay_id",
        "subject_id",
        *ALL_PRIMARY_TASKS,
    ]
    outcome_small = outcomes[
        [c for c in outcomes_required if c in outcomes.columns]
        + [c for c in outcomes.columns if c.endswith("_valid")]
    ].copy()

    if outcome_small["stay_id"].duplicated().any():
        raise ValueError("V2 outcome file has duplicate stay_id rows.")

    merged = alignment.merge(
        outcome_small,
        on=["stay_id", "subject_id"],
        how="left",
        suffixes=("", "_outcome"),
        validate="one_to_one",
    )

    if merged[list(ALL_PRIMARY_TASKS)].isna().any().any():
        missing_rows = merged[list(ALL_PRIMARY_TASKS)].isna().any(axis=1).sum()
        raise ValueError(
            f"V2 alignment/outcome join left {missing_rows} rows with missing targets."
        )

    X_seq = np.load(STRUCTURED_PATH, mmap_mode="r")
    D = np.load(DEMOGRAPHICS_PATH, mmap_mode="r")

    if tuple(X_seq.shape) != EXPECTED_SHAPE:
        raise ValueError(f"Unexpected structured shape: {X_seq.shape}; expected {EXPECTED_SHAPE}")

    if tuple(D.shape) != (EXPECTED_N, 4):
        raise ValueError(
            f"Unexpected demographics shape: {D.shape}; expected {(EXPECTED_N, 4)}"
        )

    if len(merged) != EXPECTED_N:
        raise ValueError(f"Unexpected V2 row count: {len(merged)} != {EXPECTED_N}")

    split_values = merged["split_24h"].astype(str).to_numpy()
    split_counts = pd.Series(split_values).value_counts().to_dict()

    expected_counts = {
        "train": EXPECTED_TRAIN,
        "val": EXPECTED_VAL,
        "test": EXPECTED_TEST,
    }
    if split_counts != expected_counts:
        raise ValueError(
            f"Frozen split counts do not match V2 contract: "
            f"{split_counts} != {expected_counts}"
        )

    # The canonical V2 package is stay-ordered. Do not reorder the frozen
    # tensors independently from alignment rows.
    if merged["stay_id"].duplicated().any():
        raise ValueError("Duplicate stay_id in alignment.")

    if merged["subject_id"].duplicated().sum() == 0:
        # This is not an error; subject_id can occur across multiple stays.
        pass

    X_tab = _tabularize_structured(X_seq)

    # Demographics are already train-only standardized in frozen V2.
    X_tab = np.concatenate(
        [X_tab, np.asarray(D, dtype=np.float32)],
        axis=1,
    ).astype(np.float32, copy=False)

    train_idx = np.flatnonzero(split_values == "train")
    val_idx = np.flatnonzero(split_values == "val")
    test_idx = np.flatnonzero(split_values == "test")

    return PreparedData(
        X=X_tab,
        demographics=np.asarray(D, dtype=np.float32),
        stay_id=merged["stay_id"].to_numpy(),
        subject_id=merged["subject_id"].to_numpy(),
        split=split_values,
        outcomes=merged,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )


def _make_model(
    model_id: str,
    task: str,
    params: Dict[str, Any],
    seed: int = SEED,
) -> Any:
    classification = task in BINARY_TASKS

    if model_id == "00":
        if xgb is None:
            raise ImportError("xgboost is required for model 00.")
        common = dict(
            random_state=seed,
            n_jobs=-1,
            tree_method="hist",
            verbosity=0,
        )
        if classification:
            return xgb.XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                **common,
                **params,
            )
        return xgb.XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            **common,
            **params,
        )

    if model_id in {"01", "02"}:
        if classification:
            if model_id == "01":
                estimator = LogisticRegression(
                    solver="lbfgs",
                    random_state=seed,
                    **params,
                )
            else:
                estimator = LogisticRegression(
                    solver="saga",
                    random_state=seed,
                    **params,
                )
        else:
            # LOS is continuous. V2 requires LOS regression metrics for
            # every baseline, so models 01–02 use the matching linear
            # regression counterpart for LOS. C is mapped to alpha=1/C.
            C = float(params["C"])
            alpha = 1.0 / C
            if model_id == "01":
                estimator = Ridge(alpha=alpha)
            else:
                estimator = ElasticNet(
                    alpha=alpha,
                    l1_ratio=float(params["l1_ratio"]),
                    max_iter=max(2000, int(params.get("max_iter", 2000))),
                    random_state=seed,
                )

        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", estimator),
            ]
        )

    if model_id == "03":
        if classification:
            return RandomForestClassifier(
                random_state=seed,
                n_jobs=-1,
                **params,
            )
        return RandomForestRegressor(
            random_state=seed,
            n_jobs=-1,
            **params,
        )

    if model_id == "04":
        if classification:
            return ExtraTreesClassifier(
                random_state=seed,
                n_jobs=-1,
                **params,
            )
        return ExtraTreesRegressor(
            random_state=seed,
            n_jobs=-1,
            **params,
        )

    if model_id == "05":
        if lgb is None:
            raise ImportError("lightgbm is required for model 05.")
        common = dict(
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
        )
        if classification:
            return lgb.LGBMClassifier(
                objective="binary",
                **common,
                **params,
            )
        return lgb.LGBMRegressor(
            objective="regression",
            **common,
            **params,
        )

    raise ValueError(f"Unsupported model_id={model_id}")


def _fit_one(
    model_id: str,
    task: str,
    params: Dict[str, Any],
    data: PreparedData,
) -> Tuple[Any, Dict[str, Any]]:
    model = _make_model(model_id, task, params)

    valid = _validity_mask(data.outcomes, task)

    train_keep = valid[data.train_idx]
    train_idx = data.train_idx[train_keep]

    if task in BINARY_TASKS:
        y = data.outcomes[task].to_numpy(dtype=np.float32)
        y_train = y[train_idx].astype(int)

        if np.unique(y_train).size < 2:
            raise ValueError(f"Task {task} has fewer than 2 classes in training.")

        model.fit(data.X[train_idx], y_train)
    else:
        y = data.outcomes[task].to_numpy(dtype=np.float32)
        y_train = y[train_idx]

        finite = np.isfinite(y_train)
        train_idx = train_idx[finite]
        y_train = y_train[finite]
        model.fit(data.X[train_idx], y_train)

    return model, {
        "task": task,
        "train_valid_n": int(len(train_idx)),
    }


def _predict(model: Any, X: np.ndarray, classification: bool) -> np.ndarray:
    if classification:
        return model.predict_proba(X)[:, 1].astype(np.float64)
    return model.predict(X).astype(np.float64)


def _safe_auc(y_true: np.ndarray, pred: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, pred))


def _safe_pr(y_true: np.ndarray, pred: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, pred))


def evaluate_task(
    task: str,
    y: np.ndarray,
    valid: np.ndarray,
    pred: np.ndarray,
    split_name: str,
) -> Dict[str, Any]:
    keep = valid & np.isfinite(y) & np.isfinite(pred)
    yt = y[keep]
    pp = pred[keep]

    result: Dict[str, Any] = {
        "task": task,
        "split": split_name,
        "valid_n": int(len(yt)),
    }

    if task in BINARY_TASKS:
        result.update(
            {
                "auroc": _safe_auc(yt.astype(int), pp),
                "auprc": _safe_pr(yt.astype(int), pp),
                "brier": float(brier_score_loss(yt.astype(int), pp)),
                "prevalence": float(np.mean(yt)),
            }
        )
    else:
        result.update(
            {
                "mae": float(mean_absolute_error(yt, pp)),
                "rmse": float(np.sqrt(mean_squared_error(yt, pp))),
                "median_ae": float(median_absolute_error(yt, pp)),
                "r2": float(r2_score(yt, pp)) if len(yt) > 1 else float("nan"),
                "prevalence": float("nan"),
            }
        )

    return result


def evaluate_all(
    models: Dict[str, Any],
    data: PreparedData,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    pred_rows = []

    y_arrays = {
        task: data.outcomes[task].to_numpy(dtype=np.float64)
        for task in ALL_PRIMARY_TASKS
    }

    splits = {
        "val": data.val_idx,
        "test": data.test_idx,
    }

    for split_name, idx in splits.items():
        for task, model in models.items():
            classification = task in BINARY_TASKS
            valid = _validity_mask(data.outcomes, task)

            pred = _predict(model, data.X[idx], classification)
            y = y_arrays[task][idx]
            valid_split = valid[idx]

            m = evaluate_task(task, y, valid_split, pred, split_name)
            metric_rows.append(m)

            for row_i, original_idx in enumerate(idx):
                pred_rows.append(
                    {
                        "stay_id": data.stay_id[original_idx],
                        "subject_id": data.subject_id[original_idx],
                        "split": split_name,
                        "task": task,
                        "target": float(y[row_i]),
                        "validity": int(valid_split[row_i]),
                        "prediction": float(pred[row_i]),
                    }
                )

    metrics = pd.DataFrame(metric_rows)

    # Deterioration is a derived output only.
    if {"mortality_24h", "ventilation_onset_24h", "vasopressor_onset_24h"}.issubset(models):
        for split_name, idx in splits.items():
            component_preds = {}
            component_valid = {}
            for task in BINARY_TASKS[:3]:
                component_preds[task] = _predict(
                    models[task],
                    data.X[idx],
                    classification=True,
                )
                component_valid[task] = _validity_mask(data.outcomes, task)[idx]

            common_valid = (
                component_valid["mortality_24h"]
                & component_valid["ventilation_onset_24h"]
                & component_valid["vasopressor_onset_24h"]
            )

            y_components = [
                data.outcomes[t].to_numpy(dtype=int)[idx]
                for t in BINARY_TASKS[:3]
            ]
            y_det = (
                1
                - (1 - y_components[0])
                * (1 - y_components[1])
                * (1 - y_components[2])
            ).astype(int)

            p_det = (
                1
                - (1 - component_preds["mortality_24h"])
                * (1 - component_preds["ventilation_onset_24h"])
                * (1 - component_preds["vasopressor_onset_24h"])
            )

            det_m = evaluate_task(
                "deterioration_24h",
                y_det.astype(float),
                common_valid,
                p_det,
                split_name,
            )
            metrics = pd.concat(
                [metrics, pd.DataFrame([det_m])],
                ignore_index=True,
            )

            for row_i, original_idx in enumerate(idx):
                pred_rows.append(
                    {
                        "stay_id": data.stay_id[original_idx],
                        "subject_id": data.subject_id[original_idx],
                        "split": split_name,
                        "task": "deterioration_24h",
                        "target": float(y_det[row_i]),
                        "validity": int(common_valid[row_i]),
                        "prediction": float(p_det[row_i]),
                    }
                )

    # Build this AFTER the derived deterioration rows have been appended.
    predictions = pd.DataFrame(pred_rows)
    return metrics, predictions


def _validation_selection_score(metrics: pd.DataFrame) -> float:
    rows = metrics[
        (metrics["split"] == "val")
        & metrics["task"].isin(BINARY_TASKS)
    ].copy()

    if len(rows) != 4:
        return float("-inf")

    vals = rows["auprc"].to_numpy(dtype=float)
    if not np.all(np.isfinite(vals)):
        return float("-inf")

    return float(np.mean(vals))


def _tunable_params(model_id: str, trial: "optuna.Trial") -> Dict[str, Any]:
    if model_id == "00":
        return {
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15),
            "n_estimators": trial.suggest_int("n_estimators", 400, 2000),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            # Not in the registered search space; remain at the fixed value.
            "colsample_bytree": 0.8,
        }

    if model_id == "01":
        return {
            "C": trial.suggest_float("C", 1e-3, 20.0, log=True),
            "penalty": "l2",
            "max_iter": 2000,
        }

    if model_id == "02":
        return {
            "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
            "penalty": "elasticnet",
            "l1_ratio": trial.suggest_float("l1_ratio", 0.1, 0.9),
            "max_iter": 2000,
        }

    if model_id in {"03", "04"}:
        # The specification gives "depth 8–None"; this is not a continuous
        # numeric interval. We expose bounded depths plus None explicitly.
        depth_choices = [8, 10, 12, 14, 16, 20, 24, 32, None]
        return {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1200),
            "max_depth": trial.suggest_categorical(
                "max_depth", depth_choices
            ),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 12),
            "max_features": "sqrt",
        }

    if model_id == "05":
        return {
            "num_leaves": trial.suggest_int("num_leaves", 31, 127),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.12),
            "n_estimators": 1200,
            "min_data_in_leaf": trial.suggest_int(
                "min_data_in_leaf", 20, 100
            ),
        }

    raise ValueError(model_id)


def _run_once(
    model_id: str,
    params: Dict[str, Any],
    data: PreparedData,
    out_dir: Path,
    mode: str,
) -> Dict[str, Any]:
    model_cfg = FAMILY_CONFIGS[model_id]

    out_dir.mkdir(parents=True, exist_ok=True)

    _set_seeds(SEED)
    models: Dict[str, Any] = {}

    fit_rows = []
    start = time.time()

    for task in ALL_PRIMARY_TASKS:
        task_start = time.time()
        print(
            f"[{model_id} {model_cfg.name}] {task} — fitting...",
            flush=True,
        )
        model, fit_meta = _fit_one(model_id, task, params, data)
        models[task] = model
        fit_rows.append(
            {
                **fit_meta,
                "model_id": model_id,
                "model_name": model_cfg.name,
                "mode": mode,
                "fit_seconds": time.time() - task_start,
            }
        )
        joblib.dump(
            model,
            out_dir / f"{task}.joblib",
            compress=3,
        )
        print(
            f"[{model_id} {model_cfg.name}] {task} — complete "
            f"({time.time() - task_start:.1f}s)",
            flush=True,
        )

    metrics, predictions = evaluate_all(models, data)
    selection_score = _validation_selection_score(metrics)

    metrics.to_csv(out_dir / "metrics.csv", index=False)
    predictions.to_csv(out_dir / "test_predictions.csv", index=False)
    pd.DataFrame(fit_rows).to_csv(
        out_dir / "training_history.csv",
        index=False,
    )

    tabular_spec = {
        "source_tensor": str(STRUCTURED_PATH),
        "source_shape": list(EXPECTED_SHAPE),
        "summary_method": [
            "last_hour",
            "mean_24h",
            "std_24h",
            "min_24h",
            "max_24h",
        ],
        "structured_summary_width": 550,
        "demographics_included": True,
        "demographics_width": 4,
        "final_tabular_width": 554,
        "text_included": False,
        "note": (
            "The V2 handoff requires a documented fixed tabular summary "
            "for tree/linear models but does not prescribe one exact formula. "
            "This formula is therefore a model-side fixed definition and is "
            "recorded for reproducibility."
        ),
    }
    (out_dir / "tabularization_spec.json").write_text(
        json.dumps(tabular_spec, indent=2),
        encoding="utf-8",
    )

    metadata = {
        "model_id": model_id,
        "model_name": model_cfg.name,
        "mode": mode,
        "seed": SEED,
        "device": "cpu_classical_family",
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "sklearn": sklearn_version,
            "xgboost": getattr(xgb, "__version__", None),
            "lightgbm": getattr(lgb, "__version__", None),
            "optuna": getattr(optuna, "__version__", None),
        },
        "v2_paths": {
            "alignment": str(ALIGNMENT_PATH),
            "structured": str(STRUCTURED_PATH),
            "demographics": str(DEMOGRAPHICS_PATH),
            "input_metadata": str(INPUT_METADATA_PATH),
            "input_definition": str(INPUT_DEFINITION_PATH),
            "outcomes": str(OUTCOMES_PATH),
        },
        "split_counts": {
            "train": int(len(data.train_idx)),
            "val": int(len(data.val_idx)),
            "test": int(len(data.test_idx)),
        },
        "fixed_registered_params": model_cfg.fixed_params,
        "registered_optuna_search_space": REGISTERED_TUNING_SPACES[model_id],
        "used_params": params,
        "primary_tasks": list(ALL_PRIMARY_TASKS),
        "linear_los_implementation": {
            "model_01": "Ridge with alpha=1/C",
            "model_02": "ElasticNet with alpha=1/C and registered l1_ratio",
            "reason": (
                "LogisticRegression only accepts discrete labels; V2 requires "
                "LOS regression metrics for every baseline."
            ),
        },
        "checkpoint_selection_rule": (
            "Mean validation AUPRC across mortality, ventilation onset, "
            "vasopressor onset and readmission"
        ),
        "validation_selection_score": selection_score,
        "tabularization": tabular_spec,
        "total_runtime_seconds": time.time() - start,
        "fixed_tuned_phase_separation": True,
        "test_used_for_selection": False,
    }

    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str),
        encoding="utf-8",
    )

    return {
        "model_id": model_id,
        "model_name": model_cfg.name,
        "mode": mode,
        "validation_selection_score": selection_score,
        "output_dir": str(out_dir),
    }


def run_fixed(
    model_id: str,
    data: PreparedData,
) -> Dict[str, Any]:
    cfg = FAMILY_CONFIGS[model_id]
    out_dir = RESULTS_ROOT / "fixed" / f"{model_id}_{cfg.name.replace(' ', '_')}"
    return _run_once(
        model_id=model_id,
        params=dict(cfg.fixed_params),
        data=data,
        out_dir=out_dir,
        mode="fixed",
    )


def run_tuned(
    model_id: str,
    data: PreparedData,
    n_trials: int,
) -> Dict[str, Any]:
    if optuna is None:
        raise ImportError(
            "Optuna is required for tuned mode. Install it in the CHI project environment."
        )

    cfg = FAMILY_CONFIGS[model_id]
    study_dir = RESULTS_ROOT / "tuned" / f"{model_id}_{cfg.name.replace(' ', '_')}"
    study_dir.mkdir(parents=True, exist_ok=True)

    trial_log = []

    def objective(trial: "optuna.Trial") -> float:
        params = _tunable_params(model_id, trial)

        trial_models: Dict[str, Any] = {}
        trial_start = time.time()

        for task in ALL_PRIMARY_TASKS:
            model, _ = _fit_one(model_id, task, params, data)
            trial_models[task] = model

        val_metrics, _ = evaluate_all_on_validation_only(
            trial_models,
            data,
        )

        score = _validation_selection_score(val_metrics)

        trial.set_user_attr(
            "validation_mean_binary_auprc",
            score,
        )
        trial.set_user_attr(
            "elapsed_seconds",
            time.time() - trial_start,
        )

        trial_log.append(
            {
                "trial_number": trial.number,
                "value": score,
                "params": json.dumps(params, sort_keys=True),
            }
        )
        return score

    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=f"pipeline_a_v2_{model_id}_tuned",
    )
    study.optimize(objective, n_trials=n_trials)

    pd.DataFrame(trial_log).to_csv(
        study_dir / "optuna_trials.csv",
        index=False,
    )
    (study_dir / "best_params.json").write_text(
        json.dumps(study.best_trial.params, indent=2),
        encoding="utf-8",
    )

    # Convert Optuna params into the exact estimator parameter dictionary.
    best_params = _tunable_params_from_best_trial(model_id, study.best_trial)

    result = _run_once(
        model_id=model_id,
        params=best_params,
        data=data,
        out_dir=study_dir / "final_selected_configuration",
        mode="tuned",
    )

    result["best_trial_number"] = study.best_trial.number
    result["best_trial_value"] = float(study.best_value)
    return result


def _tunable_params_from_best_trial(
    model_id: str,
    trial: "optuna.trial.FrozenTrial",
) -> Dict[str, Any]:
    p = trial.params

    if model_id == "00":
        return {
            "max_depth": int(p["max_depth"]),
            "learning_rate": float(p["learning_rate"]),
            "n_estimators": int(p["n_estimators"]),
            "subsample": float(p["subsample"]),
            "colsample_bytree": 0.8,
        }

    if model_id == "01":
        return {
            "C": float(p["C"]),
            "penalty": "l2",
            "max_iter": 2000,
        }

    if model_id == "02":
        return {
            "C": float(p["C"]),
            "penalty": "elasticnet",
            "l1_ratio": float(p["l1_ratio"]),
            "max_iter": 2000,
        }

    if model_id in {"03", "04"}:
        return {
            "n_estimators": int(p["n_estimators"]),
            "max_depth": p["max_depth"],
            "min_samples_leaf": int(p["min_samples_leaf"]),
            "max_features": "sqrt",
        }

    if model_id == "05":
        return {
            "num_leaves": int(p["num_leaves"]),
            "learning_rate": float(p["learning_rate"]),
            "n_estimators": 1200,
            "min_data_in_leaf": int(p["min_data_in_leaf"]),
        }

    raise ValueError(model_id)


def evaluate_all_on_validation_only(
    models: Dict[str, Any],
    data: PreparedData,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    idx = data.val_idx
    metric_rows = []
    pred_rows = []

    for task, model in models.items():
        classification = task in BINARY_TASKS
        valid = _validity_mask(data.outcomes, task)[idx]
        y = data.outcomes[task].to_numpy(dtype=np.float64)[idx]
        pred = _predict(model, data.X[idx], classification)

        metric_rows.append(
            evaluate_task(task, y, valid, pred, "val")
        )

    required_components = {
        "mortality_24h",
        "ventilation_onset_24h",
        "vasopressor_onset_24h",
    }
    if required_components.issubset(models):
        component_preds = {
            task: _predict(models[task], data.X[idx], classification=True)
            for task in BINARY_TASKS[:3]
        }
        component_valid = {
            task: _validity_mask(data.outcomes, task)[idx]
            for task in BINARY_TASKS[:3]
        }
        common_valid = (
            component_valid["mortality_24h"]
            & component_valid["ventilation_onset_24h"]
            & component_valid["vasopressor_onset_24h"]
        )
        y_components = [
            data.outcomes[t].to_numpy(dtype=int)[idx]
            for t in BINARY_TASKS[:3]
        ]
        y_det = (
            1
            - (1 - y_components[0])
            * (1 - y_components[1])
            * (1 - y_components[2])
        ).astype(int)
        p_det = (
            1
            - (1 - component_preds["mortality_24h"])
            * (1 - component_preds["ventilation_onset_24h"])
            * (1 - component_preds["vasopressor_onset_24h"])
        )

        metric_rows.append(
            evaluate_task(
                "deterioration_24h",
                y_det.astype(float),
                common_valid,
                p_det,
                "val",
            )
        )

        for row_i, original_idx in enumerate(idx):
            pred_rows.append(
                {
                    "stay_id": data.stay_id[original_idx],
                    "subject_id": data.subject_id[original_idx],
                    "split": "val",
                    "task": "deterioration_24h",
                    "target": float(y_det[row_i]),
                    "validity": int(common_valid[row_i]),
                    "prediction": float(p_det[row_i]),
                }
            )

    metrics = pd.DataFrame(metric_rows)
    return metrics, pd.DataFrame(pred_rows)


def run_many(
    model_ids: Iterable[str],
    mode: str,
    data: PreparedData,
    n_trials: int,
) -> None:
    summaries = []

    for model_id in model_ids:
        if mode == "fixed":
            result = run_fixed(model_id, data)
        else:
            result = run_tuned(model_id, data, n_trials=n_trials)

        summaries.append(result)

        pd.DataFrame(summaries).to_csv(
            RESULTS_ROOT / f"family_01_{mode}_summary.csv",
            index=False,
        )

        print(json.dumps(result, indent=2, default=str))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline A V2 Family 1: models 00–05"
    )
    parser.add_argument(
        "--model",
        default="all",
        choices=["all", *FAMILY_CONFIGS.keys()],
        help="Model ID or all.",
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["fixed", "tuned"],
        help="Fixed preregistered benchmark or Optuna tuned phase.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=30,
        help="Optuna trials per model in tuned mode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.n_trials < 1:
        raise ValueError("--n-trials must be >= 1")

    _set_seeds(SEED)

    print("Loading frozen Pipeline A V2 artifacts...")
    data = load_frozen_v2()
    print(
        f"Loaded X_tab={data.X.shape}; "
        f"train={len(data.train_idx)}, "
        f"val={len(data.val_idx)}, "
        f"test={len(data.test_idx)}"
    )

    if args.model == "all":
        model_ids = list(FAMILY_CONFIGS.keys())
    else:
        model_ids = [args.model]

    run_many(
        model_ids=model_ids,
        mode=args.mode,
        data=data,
        n_trials=args.n_trials,
    )


if __name__ == "__main__":
    main()
