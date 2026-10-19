from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

ROOT = Path("/home/abdullah2025/fyp_project")
OUT = ROOT / "outputs" / "xai_z2_complete"
PLOTS = OUT / "plots"
OUT.mkdir(parents=True, exist_ok=True)
PLOTS.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))

import family_01_tabular_00_05 as f01
import family_06_state_space_24_27 as f06
import family_07_advanced_multimodal_28_32 as f07

warnings.filterwarnings("ignore")

SEED = 20260923
rng = np.random.default_rng(SEED)

PRIMARY_TASKS = [
    "mortality_24h",
    "ventilation_onset_24h",
    "vasopressor_onset_24h",
    "readmission_30d_24h",
    "los_hours_24h",
]

BINARY_TASKS = [
    "mortality_24h",
    "ventilation_onset_24h",
    "vasopressor_onset_24h",
    "readmission_30d_24h",
]

ALL_TASKS = PRIMARY_TASKS + ["deterioration_24h"]

ALIGNMENT_PATH = ROOT / "outputs/pipeline_a_v2/a5_alignment_index_24h.csv"
INPUT_DEF_PATH = ROOT / "outputs/pipeline_a_v2/final_model_input_definition_24h.json"
A1_MAP_PATH = ROOT / "outputs/pipeline_a_v2/a1_feature_mapping_audit.csv"

PARITY_SUMMARY_PATH = OUT / "20_deep_xai_prediction_parity_summary.csv"

TREE_DIRS = {
    "00": ROOT / "outputs/baseline_family_01_tabular/tuned/00_XGBoost/final_selected_configuration",
    "05": ROOT / "outputs/baseline_family_01_tabular/tuned/05_LightGBM/final_selected_configuration",
}

S4D_DIR = ROOT / "outputs/baseline_family_06_state_space_24_27/tuned/24_S4D/final_selected_configuration"
MOE_DIR = ROOT / "outputs/baseline_family_07_advanced_multimodal_28_32/tuned/32_MoE_Modality_Fusion/final_selected_configuration"


# ---------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------
def json_dump(obj: Any, path: Path) -> None:
    path.write_text(
        json.dumps(obj, indent=2, default=str),
        encoding="utf-8",
    )


def require_parity() -> None:
    if not PARITY_SUMMARY_PATH.exists():
        raise RuntimeError(
            f"Missing parity summary: {PARITY_SUMMARY_PATH}"
        )

    df = pd.read_csv(PARITY_SUMMARY_PATH)

    required = {"model_id", "overall_parity"}
    if not required.issubset(df.columns):
        raise RuntimeError("Parity summary schema is incomplete.")

    expected = {"00", "05", "24", "32"}

    # CSV parsing may convert identifiers such as 00 and 05
    # into integers 0 and 5. Normalize them back to the
    # canonical two-digit model IDs used throughout the project.
    df["_model_id_norm"] = (
        df["model_id"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(2)
    )

    got = set(df["_model_id_norm"])

    if not expected.issubset(got):
        raise RuntimeError(
            f"Parity summary does not contain all required models: {sorted(got)}"
        )

    failed = df.loc[
        ~df["_model_id_norm"].isin(expected)
        | ~df["overall_parity"].astype(bool)
    ]

    if not failed.empty:
        raise RuntimeError(
            "Deep-XAI parity gate failed. "
            "Do not generate final explanations before parity is fixed."
        )


def load_alignment() -> pd.DataFrame:
    df = pd.read_csv(ALIGNMENT_PATH)
    df["stay_id"] = df["stay_id"].astype(np.int64)
    df["subject_id"] = df["subject_id"].astype(np.int64)
    df["split_24h"] = df["split_24h"].astype(str)
    return df


ALIGN = load_alignment()


def valid_indices(
    split: str,
    task: str,
) -> np.ndarray:
    mask = (
        (ALIGN["split_24h"].to_numpy() == split)
        & ALIGN[f"{task}_valid"].astype(bool).to_numpy()
    )
    return np.flatnonzero(mask)


def sample_indices(
    split: str,
    task: str,
    n: int = 512,
) -> np.ndarray:
    idx = valid_indices(split, task)

    if len(idx) <= n:
        return idx.copy()

    local_rng = np.random.default_rng(
        SEED
        + abs(hash((split, task))) % 100000
    )

    selected = local_rng.choice(
        idx,
        size=n,
        replace=False,
    )

    return np.sort(selected)


def identity_frame(
    indices: np.ndarray,
    split: str,
) -> pd.DataFrame:
    x = ALIGN.iloc[np.asarray(indices)].copy()

    if not np.all(x["split_24h"].to_numpy() == split):
        raise RuntimeError("Split identity mismatch.")

    return x[
        ["stay_id", "subject_id"]
    ].reset_index(drop=True)


def task_scalar(
    out: Dict[str, torch.Tensor],
    task: str,
) -> torch.Tensor:
    if task in BINARY_TASKS:
        return out[task]

    return out["los_excess_log1p"]


def reported_prediction(
    out: Dict[str, torch.Tensor],
    task: str,
) -> torch.Tensor:
    if task in BINARY_TASKS:
        return torch.sigmoid(out[task])

    return torch.expm1(out["los_excess_log1p"]) + 24.0


def find_string_lists(
    obj: Any,
    target_len: int,
    path: str = "",
) -> List[Tuple[str, List[str]]]:
    found = []

    if isinstance(obj, list):
        if (
            len(obj) == target_len
            and all(isinstance(v, str) for v in obj)
        ):
            found.append((path, obj))

        for i, v in enumerate(obj):
            found.extend(
                find_string_lists(
                    v,
                    target_len,
                    f"{path}[{i}]",
                )
            )

    elif isinstance(obj, dict):
        for k, v in obj.items():
            found.extend(
                find_string_lists(
                    v,
                    target_len,
                    f"{path}.{k}" if path else k,
                )
            )

    return found


def feature_names_110() -> Tuple[List[str], str]:
    if INPUT_DEF_PATH.exists():
        try:
            obj = json.loads(
                INPUT_DEF_PATH.read_text()
            )

            candidates = find_string_lists(
                obj,
                110,
            )

            if candidates:
                candidates.sort(
                    key=lambda x: (
                        "feature" not in x[0].lower(),
                        "channel" not in x[0].lower(),
                        len(x[0]),
                    )
                )

                return candidates[0][1], candidates[0][0]

        except Exception:
            pass

    return (
        [f"structured_channel_{i:03d}" for i in range(110)],
        "fallback",
    )


def demographic_names() -> List[str]:
    if INPUT_DEF_PATH.exists():
        try:
            obj = json.loads(
                INPUT_DEF_PATH.read_text()
            )

            candidates = find_string_lists(
                obj,
                4,
            )

            if candidates:
                candidates.sort(
                    key=lambda x: (
                        "demo" not in x[0].lower(),
                        len(x[0]),
                    )
                )

                return candidates[0][1]
        except Exception:
            pass

    return [
        "demographic_00",
        "demographic_01",
        "demographic_02",
        "demographic_03",
    ]


STRUCTURED_NAMES, STRUCTURED_NAME_SOURCE = feature_names_110()
DEMO_NAMES = demographic_names()


def feature_root_name(name: str) -> str:
    s = str(name).lower()

    for prefix in (
        "last_hour::",
        "mean_24h::",
        "std_24h::",
        "min_24h::",
        "max_24h::",
        "value::",
        "values::",
        "time_since::",
        "trend::",
        "mask::",
        "value_",
        "time_",
        "trend_",
        "mask_",
    ):
        if s.startswith(prefix):
            s = s[len(prefix):]

    return s.strip()


def clinical_category_map() -> Dict[str, str]:
    if not A1_MAP_PATH.exists():
        return {}

    df = pd.read_csv(A1_MAP_PATH)

    if "feature" not in df.columns:
        return {}

    result = {}

    for _, row in df.iterrows():
        feat = str(row.get("feature", "")).strip().lower()
        cat = str(row.get("category", "")).strip()

        if feat and cat:
            result[feat] = cat

    return result


CATEGORY_MAP = clinical_category_map()


def category_for_feature(name: str) -> str:
    root = feature_root_name(name)

    if root in CATEGORY_MAP:
        return CATEGORY_MAP[root]

    for key, category in CATEGORY_MAP.items():
        if key and key in root:
            return category

    return "Unmapped"


# ---------------------------------------------------------------------
# Validation-based methodological lock
# ---------------------------------------------------------------------
def ig_once(
    model,
    task: str,
    x: torch.Tensor,
    d: torch.Tensor,
    base_x: torch.Tensor,
    base_d: torch.Tensor,
    steps: int,
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    delta_x = x - base_x
    delta_d = d - base_d

    grad_x = torch.zeros_like(x)
    grad_d = torch.zeros_like(d)

    for alpha in torch.linspace(
        1.0 / steps,
        1.0,
        steps,
        device=x.device,
    ):
        xi = (
            base_x
            + alpha * delta_x
        ).detach().requires_grad_(True)

        di = (
            base_d
            + alpha * delta_d
        ).detach().requires_grad_(True)

        out = model(
            xi,
            di,
        )

        scalar = task_scalar(
            out,
            task,
        )

        gx, gd = torch.autograd.grad(
            scalar.sum(),
            [xi, di],
            retain_graph=False,
            create_graph=False,
        )

        grad_x += gx.detach()
        grad_d += gd.detach()

    attr_x = delta_x * grad_x / float(steps)
    attr_d = delta_d * grad_d / float(steps)

    with torch.no_grad():
        input_out = (
            task_scalar(
                model(x, d),
                task,
            )
            .detach()
            .cpu()
            .numpy()
        )

        base_out = (
            task_scalar(
                model(base_x, base_d),
                task,
            )
            .detach()
            .cpu()
            .numpy()
        )

    completeness = (
        attr_x.detach()
        .sum(dim=(1, 2))
        .cpu()
        .numpy()
        + attr_d.detach()
        .sum(dim=1)
        .cpu()
        .numpy()
        + base_out
        - input_out
    )

    return (
        attr_x.detach(),
        attr_d.detach(),
        completeness,
    )


def load_s4d_for_xai():
    metadata = json.loads(
        (S4D_DIR / "metadata.json").read_text()
    )

    params = dict(
        metadata["run_params"]
    )

    model = f06.build_model(
        "24",
        params,
    )

    checkpoint = torch.load(
        S4D_DIR / "checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["state_dict"],
        strict=True,
    )

    device = f06.get_device()

    model = model.to(device)
    model.eval()

    return (
        model,
        device,
        params,
    )


def validation_xai_lock() -> Dict[str, Any]:
    model, device, params = load_s4d_for_xai()

    val_idx = np.asarray(
        ALIGN.index[
            ALIGN["split_24h"] == "val"
        ]
    )

    x_val = np.asarray(
        np.load(
            ROOT / "outputs/pipeline_a_v2/structured_24h_processed.npy",
            mmap_mode="r",
        ),
        dtype=np.float32,
    )

    d_val = np.asarray(
        np.load(
            ROOT / "outputs/pipeline_a_v2/demographics_24h_processed.npy",
            mmap_mode="r",
        ),
        dtype=np.float32,
    )

    # Validation-only diagnostic subset.
    chosen = []
    for task in PRIMARY_TASKS:
        idx = sample_indices(
            "val",
            task,
            n=32,
        )
        chosen.extend(idx.tolist())

    chosen = np.unique(chosen)

    xv = torch.tensor(
        x_val[chosen],
        dtype=torch.float32,
        device=device,
    )

    dv = torch.tensor(
        d_val[chosen],
        dtype=torch.float32,
        device=device,
    )

    mean_x = torch.tensor(
        np.mean(
            x_val[val_idx],
            axis=0,
        ),
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    mean_d = torch.tensor(
        np.mean(
            d_val[val_idx],
            axis=0,
        ),
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    zero_x = torch.zeros_like(
        mean_x
    )

    zero_d = torch.zeros_like(
        mean_d
    )

    candidate_baselines = {
        "validation_mean": (
            mean_x,
            mean_d,
        ),
        "zero": (
            zero_x,
            zero_d,
        ),
    }

    baseline_scores = {}

    for name, (bx0, bd0) in candidate_baselines.items():
        bx = bx0.expand_as(xv)
        bd = bd0.expand_as(dv)

        task_errors = []

        with torch.enable_grad():
            for task in PRIMARY_TASKS:
                try:
                    _, _, residual = ig_once(
                        model,
                        task,
                        xv,
                        dv,
                        bx,
                        bd,
                        steps=16,
                    )

                    # Scale-independent completeness error.
                    with torch.no_grad():
                        in_out = (
                            task_scalar(
                                model(xv, dv),
                                task,
                            )
                            .detach()
                            .cpu()
                            .numpy()
                        )

                        base_out = (
                            task_scalar(
                                model(bx, bd),
                                task,
                            )
                            .detach()
                            .cpu()
                            .numpy()
                        )

                    denom = (
                        np.abs(
                            in_out - base_out
                        )
                        + 1e-6
                    )

                    rel = np.abs(
                        residual
                    ) / denom

                    task_errors.extend(
                        rel[np.isfinite(rel)].tolist()
                    )

                except Exception:
                    continue

        baseline_scores[name] = {
            "mean_relative_completeness_error": float(
                np.mean(task_errors)
            ) if task_errors else float("inf"),
            "p90_relative_completeness_error": float(
                np.quantile(task_errors, 0.90)
            ) if task_errors else float("inf"),
            "n_measurements": len(task_errors),
        }

    selected_baseline = min(
        baseline_scores,
        key=lambda x: (
            baseline_scores[x]["mean_relative_completeness_error"],
            baseline_scores[x]["p90_relative_completeness_error"],
        ),
    )

    bx0, bd0 = candidate_baselines[
        selected_baseline
    ]

    # Select the smallest step count satisfying
    # the fixed validation convergence rule.
    step_candidates = [64, 128, 256, 512]
    step_scores = {}

    for steps in step_candidates:
        errors = []

        with torch.enable_grad():
            for task in PRIMARY_TASKS:
                try:
                    bx = bx0.expand_as(xv)
                    bd = bd0.expand_as(dv)

                    _, _, residual = ig_once(
                        model,
                        task,
                        xv,
                        dv,
                        bx,
                        bd,
                        steps=steps,
                    )

                    with torch.no_grad():
                        in_out = (
                            task_scalar(
                                model(xv, dv),
                                task,
                            )
                            .detach()
                            .cpu()
                            .numpy()
                        )

                        base_out = (
                            task_scalar(
                                model(bx, bd),
                                task,
                            )
                            .detach()
                            .cpu()
                            .numpy()
                        )

                    denom = (
                        np.abs(
                            in_out - base_out
                        )
                        + 1e-6
                    )

                    rel = (
                        np.abs(residual)
                        / denom
                    )

                    errors.extend(
                        rel[
                            np.isfinite(rel)
                        ].tolist()
                    )

                except Exception:
                    continue

        step_scores[str(steps)] = {
            "mean_relative_completeness_error": float(
                np.mean(errors)
            ) if errors else float("inf"),
            "p90_relative_completeness_error": float(
                np.quantile(errors, 0.90)
            ) if errors else float("inf"),
            "n_measurements": len(errors),
        }

    # Validation-only convergence lock.
    # We prefer the smallest step count reaching 1% p90
    # relative completeness error. If the criterion is not
    # reached, choose the empirically best tested setting and
    # record that the convergence target was not achieved.
    convergence_target = 1e-2

    selected_steps = min(
        step_candidates,
        key=lambda k: (
            step_scores[str(k)]["p90_relative_completeness_error"],
            k,
        ),
    )

    convergence_target_reached = False

    for steps in step_candidates:
        score = step_scores[str(steps)]
        if (
            score["p90_relative_completeness_error"]
            <= convergence_target
        ):
            selected_steps = steps
            convergence_target_reached = True
            break

    result = {
        "seed": SEED,
        "sample_per_task": 512,
        "tree_shap_feature_perturbation": "tree_path_dependent",
        "tree_shap_output_scale": "raw",
        "ig_baseline_candidates": list(
            candidate_baselines.keys()
        ),
        "ig_baseline_selection_rule": (
            "Minimum validation mean relative completeness "
            "error, tie-broken by p90 error."
        ),
        "selected_ig_baseline": selected_baseline,
        "baseline_validation_scores": baseline_scores,
        "ig_step_candidates": step_candidates,
        "ig_step_selection_rule": (
            "Smallest tested step count whose validation p90 "
            "relative completeness error <= 1e-2; otherwise "
            "the empirically best tested step count."
        ),
        "ig_convergence_target_p90_relative_error": 1e-2,
        "ig_convergence_target_reached": convergence_target_reached,
        "selected_ig_steps": selected_steps,
        "ig_step_validation_scores": step_scores,
        "temporal_occlusion_window_hours": 1,
        "temporal_occlusion_reference": (
            f"selected IG baseline: {selected_baseline}"
        ),
        "moe_modality_reference": (
            "validation mean for structured and demographics; "
            "zero text embedding + text_missing=1 for text ablation."
        ),
        "moe_gradient_method": "input_x_gradient",
        "router_method": (
            "exact locked router logits/probabilities from Model 32 "
            "evaluation mode; top-k taken from the locked model."
        ),
        "test_selection_locked_after_validation": True,
        "input_feature_name_source": STRUCTURED_NAME_SOURCE,
    }

    json_dump(
        result,
        OUT / "21_xai_method_lock.json",
    )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# ---------------------------------------------------------------------
# Tree SHAP
# ---------------------------------------------------------------------
def run_tree_shap() -> None:
    """
    Native Tree-SHAP using the exact locked estimators.

    Model 00:
        XGBoost Booster.predict(pred_contribs=True)

    Model 05:
        LightGBM LGBMModel.predict(pred_contrib=True)

    The final contribution column is the expected-value/bias term.
    Feature contributions are stored separately.
    """

    method_lock = json.loads(
        (OUT / "21_xai_method_lock.json").read_text()
    )

    sample_n = int(
        method_lock["sample_per_task"]
    )

    structured = np.load(
        ROOT / "outputs/pipeline_a_v2/structured_24h_processed.npy",
        mmap_mode="r",
    )

    demo = np.load(
        ROOT / "outputs/pipeline_a_v2/demographics_24h_processed.npy",
        mmap_mode="r",
    )

    x_summary = f01._tabularize_structured(
        structured,
        batch_size=4096,
    )

    x_tab = np.concatenate(
        [
            x_summary,
            np.asarray(demo, dtype=np.float32),
        ],
        axis=1,
    ).astype(np.float32, copy=False)

    if x_tab.shape != (74572, 554):
        raise RuntimeError(
            f"Unexpected tree input shape: {x_tab.shape}"
        )

    feature_names = []

    for stat in [
        "last_hour",
        "mean_24h",
        "std_24h",
        "min_24h",
        "max_24h",
    ]:
        feature_names.extend(
            [
                f"{stat}::{name}"
                for name in STRUCTURED_NAMES
            ]
        )

    feature_names.extend(DEMO_NAMES)

    if len(feature_names) != 554:
        raise RuntimeError(
            f"Expected 554 feature names, got {len(feature_names)}"
        )

    global_rows = []
    local_rows = []
    expected_rows = []

    for model_id, folder in TREE_DIRS.items():

        model_name = (
            "XGBoost"
            if model_id == "00"
            else "LightGBM"
        )

        for task in PRIMARY_TASKS:

            print(
                f"[Tree XAI] model={model_id} task={task}",
                flush=True,
            )

            idx = sample_indices(
                "test",
                task,
                sample_n,
            )

            X = x_tab[idx]

            model = joblib_load(
                folder / f"{task}.joblib"
            )

            if model_id == "00":
                import xgboost as xgb

                booster = model.get_booster()

                dm = xgb.DMatrix(
                    X,
                    feature_names=feature_names,
                )

                contribution = booster.predict(
                    dm,
                    pred_contribs=True,
                )

            elif model_id == "05":
                contribution = model.predict(
                    X,
                    pred_contrib=True,
                )

            else:
                raise RuntimeError(
                    f"Unsupported Tree XAI model: {model_id}"
                )

            contribution = np.asarray(
                contribution,
                dtype=np.float64,
            )

            expected_shape = (
                len(X),
                555,
            )

            if contribution.shape != expected_shape:
                raise RuntimeError(
                    f"{model_id}/{task}: contribution shape "
                    f"{contribution.shape}, expected {expected_shape}"
                )

            if not np.isfinite(
                contribution
            ).all():
                raise RuntimeError(
                    f"{model_id}/{task}: non-finite Tree-SHAP contributions"
                )

            shap_values = contribution[:, :-1]
            expected_value = contribution[:, -1]

            mean_abs = np.mean(
                np.abs(shap_values),
                axis=0,
            )

            mean_signed = np.mean(
                shap_values,
                axis=0,
            )

            for j, name in enumerate(
                feature_names
            ):
                global_rows.append(
                    {
                        "model_id": model_id,
                        "model_name": model_name,
                        "split": "test",
                        "task": task,
                        "feature_index": j,
                        "feature_name": name,
                        "clinical_category": category_for_feature(
                            name
                        ),
                        "mean_abs_shap": float(
                            mean_abs[j]
                        ),
                        "mean_signed_shap": float(
                            mean_signed[j]
                        ),
                    }
                )

            for row_i, global_i in enumerate(
                idx
            ):
                vals = shap_values[row_i]

                top = np.argsort(
                    np.abs(vals)
                )[::-1][:15]

                for rank, j in enumerate(
                    top,
                    1,
                ):
                    local_rows.append(
                        {
                            "model_id": model_id,
                            "model_name": model_name,
                            "split": "test",
                            "task": task,
                            "stay_id": int(
                                ALIGN.iloc[global_i]["stay_id"]
                            ),
                            "subject_id": int(
                                ALIGN.iloc[global_i]["subject_id"]
                            ),
                            "rank": rank,
                            "feature_index": int(j),
                            "feature_name": feature_names[j],
                            "clinical_category": (
                                category_for_feature(
                                    feature_names[j]
                                )
                            ),
                            "shap_value": float(
                                vals[j]
                            ),
                            "abs_shap_value": float(
                                abs(vals[j])
                            ),
                        }
                    )

                expected_rows.append(
                    {
                        "model_id": model_id,
                        "model_name": model_name,
                        "split": "test",
                        "task": task,
                        "stay_id": int(
                            ALIGN.iloc[global_i]["stay_id"]
                        ),
                        "subject_id": int(
                            ALIGN.iloc[global_i]["subject_id"]
                        ),
                        "expected_value": float(
                            expected_value[row_i]
                        ),
                    }
                )

    pd.DataFrame(
        global_rows
    ).to_csv(
        OUT / "22_tree_shap_global.csv",
        index=False,
    )

    pd.DataFrame(
        local_rows
    ).to_csv(
        OUT / "23_tree_shap_local.csv",
        index=False,
    )

    pd.DataFrame(
        expected_rows
    ).drop_duplicates(
        subset=[
            "model_id",
            "task",
            "stay_id",
            "subject_id",
        ]
    ).to_csv(
        OUT / "23b_tree_shap_expected_values.csv",
        index=False,
    )

    print(
        "[Tree XAI] Native Tree-SHAP completed.",
        flush=True,
    )


def joblib_load(path: Path):
    import joblib
    return joblib.load(path)


def normalize_shap_values(
    values,
    expected_shape: Tuple[int, int],
) -> np.ndarray:
    if isinstance(values, list):
        if len(values) == 2:
            values = values[-1]
        else:
            values = values[0]

    values = np.asarray(values)

    if values.shape == expected_shape:
        return values

    if values.ndim == 3:
        # Common binary SHAP representation:
        # [n, features, classes]
        if values.shape[0] == expected_shape[0]:
            if values.shape[1] == expected_shape[1]:
                return values[:, :, -1]

    raise RuntimeError(
        f"Unexpected SHAP shape {values.shape}; "
        f"expected {expected_shape}."
    )


# ---------------------------------------------------------------------
# S4D Integrated Gradients + temporal occlusion
# ---------------------------------------------------------------------
def run_s4d_deep_xai(
    method_lock: Dict[str, Any],
) -> None:
    model, device, params = load_s4d_for_xai()

    structured = np.load(
        ROOT / "outputs/pipeline_a_v2/structured_24h_processed.npy",
        mmap_mode="r",
    )

    demographics = np.load(
        ROOT / "outputs/pipeline_a_v2/demographics_24h_processed.npy",
        mmap_mode="r",
    )

    x_base = np.asarray(
        np.mean(
            structured[
                ALIGN["split_24h"].to_numpy() == "val"
            ],
            axis=0,
        ),
        dtype=np.float32,
    )

    d_base = np.asarray(
        np.mean(
            demographics[
                ALIGN["split_24h"].to_numpy() == "val"
            ],
            axis=0,
        ),
        dtype=np.float32,
    )

    if method_lock["selected_ig_baseline"] == "zero":
        x_ref = np.zeros_like(x_base)
        d_ref = np.zeros_like(d_base)
    else:
        x_ref = x_base
        d_ref = d_base

    steps = int(
        method_lock["selected_ig_steps"]
    )

    global_rows = []
    temporal_rows = []
    occlusion_rows = []
    local_rows = []

    for task in PRIMARY_TASKS:
        idx = sample_indices(
            "test",
            task,
            int(method_lock["sample_per_task"]),
        )

        for start in range(
            0,
            len(idx),
            16,
        ):
            sub = idx[
                start : start + 16
            ]

            xb = torch.tensor(
                np.asarray(
                    structured[sub],
                    dtype=np.float32,
                ),
                device=device,
            )

            db = torch.tensor(
                np.asarray(
                    demographics[sub],
                    dtype=np.float32,
                ),
                device=device,
            )

            bx = torch.tensor(
                x_ref,
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0).expand_as(xb)

            bd = torch.tensor(
                d_ref,
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0).expand_as(db)

            with torch.enable_grad():
                attr_x, attr_d, residual = ig_once(
                    model,
                    task,
                    xb,
                    db,
                    bx,
                    bd,
                    steps=steps,
                )

            ax = (
                attr_x
                .detach()
                .cpu()
                .numpy()
            )

            ad = (
                attr_d
                .detach()
                .cpu()
                .numpy()
            )

            # Feature-level global attribution.
            feature_score = np.mean(
                np.abs(ax),
                axis=(0, 1),
            )

            feature_signed = np.mean(
                ax,
                axis=(0, 1),
            )

            for j in range(110):
                global_rows.append(
                    {
                        "model_id": "24",
                        "model_name": "S4D",
                        "split": "test",
                        "task": task,
                        "feature_index": j,
                        "feature_name": STRUCTURED_NAMES[j],
                        "clinical_category": category_for_feature(
                            STRUCTURED_NAMES[j]
                        ),
                        "mean_abs_ig": float(
                            feature_score[j]
                        ),
                        "mean_signed_ig": float(
                            feature_signed[j]
                        ),
                    }
                )

            time_score = np.mean(
                np.abs(ax),
                axis=(0, 2),
            )

            for h in range(24):
                temporal_rows.append(
                    {
                        "model_id": "24",
                        "model_name": "S4D",
                        "split": "test",
                        "task": task,
                        "hour_index": h,
                        "hour_number": h + 1,
                        "mean_abs_ig": float(
                            time_score[h]
                        ),
                    }
                )

            # Local feature ranking.
            local_feature_score = np.sum(
                np.abs(ax),
                axis=1,
            )

            for i, global_i in enumerate(sub):
                top = np.argsort(
                    local_feature_score[i]
                )[::-1][:15]

                for rank, j in enumerate(top, 1):
                    local_rows.append(
                        {
                            "model_id": "24",
                            "model_name": "S4D",
                            "split": "test",
                            "task": task,
                            "stay_id": int(
                                ALIGN.iloc[global_i]["stay_id"]
                            ),
                            "subject_id": int(
                                ALIGN.iloc[global_i]["subject_id"]
                            ),
                            "rank": rank,
                            "feature_index": int(j),
                            "feature_name": STRUCTURED_NAMES[j],
                            "clinical_category": category_for_feature(
                                STRUCTURED_NAMES[j]
                            ),
                            "signed_attribution": float(
                                np.sum(
                                    ax[i, :, j]
                                )
                            ),
                            "abs_attribution": float(
                                np.sum(
                                    np.abs(
                                        ax[i, :, j]
                                    )
                                )
                            ),
                        }
                    )

                local_rows.append(
                    {
                        "model_id": "24",
                        "model_name": "S4D",
                        "split": "test",
                        "task": task,
                        "stay_id": int(
                            ALIGN.iloc[global_i]["stay_id"]
                        ),
                        "subject_id": int(
                            ALIGN.iloc[global_i]["subject_id"]
                        ),
                        "rank": 0,
                        "feature_index": -1,
                        "feature_name": "__completeness_residual__",
                        "clinical_category": "XAI diagnostic",
                        "signed_attribution": float(
                            residual[i]
                        ),
                        "abs_attribution": float(
                            abs(residual[i])
                        ),
                    }
                )

    # One-hour temporal occlusion.
    for task in PRIMARY_TASKS:
        idx = sample_indices(
            "test",
            task,
            int(method_lock["sample_per_task"]),
        )

        for start in range(
            0,
            len(idx),
            16,
        ):
            sub = idx[
                start : start + 16
            ]

            xb = torch.tensor(
                np.asarray(
                    structured[sub],
                    dtype=np.float32,
                ),
                device=device,
            )

            db = torch.tensor(
                np.asarray(
                    demographics[sub],
                    dtype=np.float32,
                ),
                device=device,
            )

            base_x_batch = torch.tensor(
                x_ref,
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0).expand_as(xb)

            with torch.no_grad():
                original_out = reported_prediction(
                    model(xb, db),
                    task,
                ).detach().cpu().numpy()

            for hour in range(24):
                xo = xb.clone()
                xo[:, hour, :] = (
                    base_x_batch[
                        :, hour, :
                    ]
                )

                with torch.no_grad():
                    occ_out = reported_prediction(
                        model(xo, db),
                        task,
                    ).detach().cpu().numpy()

                delta = (
                    original_out
                    - occ_out
                )

                occlusion_rows.append(
                    {
                        "model_id": "24",
                        "model_name": "S4D",
                        "split": "test",
                        "task": task,
                        "hour_index": hour,
                        "hour_number": hour + 1,
                        "mean_signed_change": float(
                            np.mean(delta)
                        ),
                        "mean_abs_change": float(
                            np.mean(np.abs(delta))
                        ),
                    }
                )

    pd.DataFrame(global_rows).to_csv(
        OUT / "24_s4d_ig_global.csv",
        index=False,
    )

    pd.DataFrame(local_rows).to_csv(
        OUT / "25_s4d_ig_local.csv",
        index=False,
    )

    pd.DataFrame(temporal_rows).to_csv(
        OUT / "26_s4d_ig_temporal.csv",
        index=False,
    )

    pd.DataFrame(occlusion_rows).groupby(
        [
            "model_id",
            "model_name",
            "split",
            "task",
            "hour_index",
            "hour_number",
        ],
        as_index=False,
    ).agg(
        mean_signed_change=("mean_signed_change", "mean"),
        mean_abs_change=("mean_abs_change", "mean"),
    ).to_csv(
        OUT / "27_s4d_temporal_occlusion.csv",
        index=False,
    )

    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------
# Model 32 — modality ablation, router and gradient attribution
# ---------------------------------------------------------------------
def load_moe_for_xai():
    metadata = json.loads(
        (MOE_DIR / "metadata.json").read_text()
    )

    params = dict(
        metadata["run_params"]
    )

    model = f07.build_model(
        "32",
        params,
    )

    checkpoint = torch.load(
        MOE_DIR / "checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["state_dict"],
        strict=True,
    )

    device = f07.get_device()

    model = model.to(device)
    model.eval()

    return (
        model,
        device,
        params,
    )


def run_moe_deep_xai(
    method_lock: Dict[str, Any],
) -> None:
    model, device, params = load_moe_for_xai()
    data = f07.load_frozen_v2()

    structured = np.asarray(
        data.structured,
        dtype=np.float32,
    )

    demographics = np.asarray(
        data.demographics,
        dtype=np.float32,
    )

    text = np.asarray(
        data.text,
        dtype=np.float32,
    )

    text_missing = np.asarray(
        data.text_missing,
        dtype=np.float32,
    )

    val_mask = (
        ALIGN["split_24h"].to_numpy()
        == "val"
    )

    structured_ref = np.mean(
        structured[val_mask],
        axis=0,
    ).astype(np.float32)

    demo_ref = np.mean(
        demographics[val_mask],
        axis=0,
    ).astype(np.float32)

    text_zero = np.zeros(
        text.shape[1],
        dtype=np.float32,
    )

    modality_rows = []
    gradient_rows = []
    local_rows = []
    router_rows = []

    # -------------------------------------------------------------
    # Modality ablation
    # -------------------------------------------------------------
    for task in PRIMARY_TASKS:
        idx = sample_indices(
            "test",
            task,
            int(method_lock["sample_per_task"]),
        )

        for start in range(
            0,
            len(idx),
            16,
        ):
            sub = idx[
                start : start + 16
            ]

            xb = torch.tensor(
                structured[sub],
                dtype=torch.float32,
                device=device,
            )

            db = torch.tensor(
                demographics[sub],
                dtype=torch.float32,
                device=device,
            )

            tb = torch.tensor(
                text[sub],
                dtype=torch.float32,
                device=device,
            )

            mb = torch.tensor(
                text_missing[sub],
                dtype=torch.float32,
                device=device,
            )

            with torch.no_grad():
                base_out = reported_prediction(
                    model(
                        xb,
                        db,
                        tb,
                        mb,
                    ),
                    task,
                )

                sx = torch.tensor(
                    structured_ref,
                    dtype=torch.float32,
                    device=device,
                ).unsqueeze(0).expand_as(xb)

                sd = db

                no_struct_out = reported_prediction(
                    model(
                        sx,
                        sd,
                        tb,
                        mb,
                    ),
                    task,
                )

                dx = xb

                dd = torch.tensor(
                    demo_ref,
                    dtype=torch.float32,
                    device=device,
                ).unsqueeze(0).expand_as(db)

                no_demo_out = reported_prediction(
                    model(
                        dx,
                        dd,
                        tb,
                        mb,
                    ),
                    task,
                )

                tt = torch.tensor(
                    text_zero,
                    dtype=torch.float32,
                    device=device,
                ).unsqueeze(0).expand_as(tb)

                mm = torch.ones_like(
                    mb
                )

                no_text_out = reported_prediction(
                    model(
                        xb,
                        db,
                        tt,
                        mm,
                    ),
                    task,
                )

            variants = {
                "structured_removed": no_struct_out,
                "demographics_removed": no_demo_out,
                "text_removed": no_text_out,
            }

            base_np = (
                base_out.detach()
                .cpu()
                .numpy()
            )

            for modality, out in variants.items():
                delta = (
                    base_np
                    - out.detach()
                    .cpu()
                    .numpy()
                )

                for i, global_i in enumerate(sub):
                    modality_rows.append(
                        {
                            "model_id": "32",
                            "model_name": "MoE Modality Fusion",
                            "split": "test",
                            "task": task,
                            "stay_id": int(
                                ALIGN.iloc[global_i]["stay_id"]
                            ),
                            "subject_id": int(
                                ALIGN.iloc[global_i]["subject_id"]
                            ),
                            "ablation": modality,
                            "baseline_prediction": float(
                                base_np[i]
                            ),
                            "ablated_prediction": float(
                                out[i].detach().cpu().item()
                            ),
                            "signed_change": float(
                                delta[i]
                            ),
                            "abs_change": float(
                                abs(delta[i])
                            ),
                        }
                    )

    # -------------------------------------------------------------
    # Router analysis on a deterministic test cohort
    # -------------------------------------------------------------
    router_candidate = np.flatnonzero(
        ALIGN["split_24h"].to_numpy() == "test"
    )

    if len(router_candidate) > 1024:
        router_candidate = np.sort(
            np.random.default_rng(
                SEED + 777
            ).choice(
                router_candidate,
                size=1024,
                replace=False,
            )
        )

    hook_values = []

    def router_hook(module, inputs, output):
        hook_values.append(
            output.detach()
            .cpu()
            .numpy()
        )

    handle = model.router.register_forward_hook(
        router_hook
    )

    for start in range(
        0,
        len(router_candidate),
        32,
    ):
        sub = router_candidate[
            start : start + 32
        ]

        xb = torch.tensor(
            structured[sub],
            dtype=torch.float32,
            device=device,
        )

        db = torch.tensor(
            demographics[sub],
            dtype=torch.float32,
            device=device,
        )

        tb = torch.tensor(
            text[sub],
            dtype=torch.float32,
            device=device,
        )

        mb = torch.tensor(
            text_missing[sub],
            dtype=torch.float32,
            device=device,
        )

        hook_values.clear()

        with torch.no_grad():
            _ = model(
                xb,
                db,
                tb,
                mb,
            )

        logits = hook_values[-1]

        probs = torch.softmax(
            torch.tensor(logits),
            dim=-1,
        ).numpy()

        top1 = np.argmax(
            probs,
            axis=1,
        )

        for i, global_i in enumerate(sub):
            ent = -np.sum(
                probs[i]
                * np.log(
                    np.clip(
                        probs[i],
                        1e-12,
                        1.0,
                    )
                )
            )

            router_rows.append(
                {
                    "model_id": "32",
                    "model_name": "MoE Modality Fusion",
                    "split": "test",
                    "stay_id": int(
                        ALIGN.iloc[global_i]["stay_id"]
                    ),
                    "subject_id": int(
                        ALIGN.iloc[global_i]["subject_id"]
                    ),
                    "text_missing": int(
                        text_missing[global_i]
                    ),
                    "expert_0_weight": float(
                        probs[i, 0]
                    ),
                    "expert_1_weight": float(
                        probs[i, 1]
                    ),
                    "top1_expert": int(
                        top1[i]
                    ),
                    "router_entropy": float(
                        ent
                    ),
                }
            )

    handle.remove()

    # -------------------------------------------------------------
    # Input x gradient attribution
    # -------------------------------------------------------------
    for task in PRIMARY_TASKS:
        idx = sample_indices(
            "test",
            task,
            int(method_lock["sample_per_task"]),
        )

        for start in range(
            0,
            len(idx),
            16,
        ):
            sub = idx[
                start : start + 16
            ]

            xb = torch.tensor(
                structured[sub],
                dtype=torch.float32,
                device=device,
            ).requires_grad_(True)

            db = torch.tensor(
                demographics[sub],
                dtype=torch.float32,
                device=device,
            ).requires_grad_(True)

            tb = torch.tensor(
                text[sub],
                dtype=torch.float32,
                device=device,
            ).requires_grad_(True)

            mb = torch.tensor(
                text_missing[sub],
                dtype=torch.float32,
                device=device,
            )

            model.zero_grad(
                set_to_none=True
            )

            out = model(
                xb,
                db,
                tb,
                mb,
            )

            scalar = task_scalar(
                out,
                task,
            )

            scalar.sum().backward()

            ax = (
                xb.detach()
                * xb.grad.detach()
            ).cpu().numpy()

            ad = (
                db.detach()
                * db.grad.detach()
            ).cpu().numpy()

            at = (
                tb.detach()
                * tb.grad.detach()
            ).cpu().numpy()

            feature_score = np.mean(
                np.abs(ax),
                axis=(0, 1),
            )

            feature_signed = np.mean(
                ax,
                axis=(0, 1),
            )

            for j in range(110):
                gradient_rows.append(
                    {
                        "model_id": "32",
                        "model_name": "MoE Modality Fusion",
                        "split": "test",
                        "task": task,
                        "modality": "structured",
                        "feature_index": j,
                        "feature_name": STRUCTURED_NAMES[j],
                        "clinical_category": category_for_feature(
                            STRUCTURED_NAMES[j]
                        ),
                        "mean_abs_input_gradient": float(
                            feature_score[j]
                        ),
                        "mean_signed_input_gradient": float(
                            feature_signed[j]
                        ),
                    }
                )

            demo_score = np.mean(
                np.abs(ad),
                axis=0,
            )

            demo_signed = np.mean(
                ad,
                axis=0,
            )

            for j in range(4):
                gradient_rows.append(
                    {
                        "model_id": "32",
                        "model_name": "MoE Modality Fusion",
                        "split": "test",
                        "task": task,
                        "modality": "demographics",
                        "feature_index": j,
                        "feature_name": DEMO_NAMES[j],
                        "clinical_category": "Demographics",
                        "mean_abs_input_gradient": float(
                            demo_score[j]
                        ),
                        "mean_signed_input_gradient": float(
                            demo_signed[j]
                        ),
                    }
                )

            text_score = float(
                np.mean(
                    np.abs(at)
                )
            )

            text_signed = float(
                np.mean(at)
            )

            gradient_rows.append(
                {
                    "model_id": "32",
                    "model_name": "MoE Modality Fusion",
                    "split": "test",
                    "task": task,
                    "modality": "clinical_text",
                    "feature_index": -1,
                    "feature_name": "clinical_note_embedding",
                    "clinical_category": "Clinical text",
                    "mean_abs_input_gradient": text_score,
                    "mean_signed_input_gradient": text_signed,
                }
            )

            local_score = np.sum(
                np.abs(ax),
                axis=1,
            )

            for i, global_i in enumerate(sub):
                top = np.argsort(
                    local_score[i]
                )[::-1][:15]

                for rank, j in enumerate(top, 1):
                    local_rows.append(
                        {
                            "model_id": "32",
                            "model_name": "MoE Modality Fusion",
                            "split": "test",
                            "task": task,
                            "stay_id": int(
                                ALIGN.iloc[global_i]["stay_id"]
                            ),
                            "subject_id": int(
                                ALIGN.iloc[global_i]["subject_id"]
                            ),
                            "rank": rank,
                            "feature_index": int(j),
                            "feature_name": STRUCTURED_NAMES[j],
                            "clinical_category": category_for_feature(
                                STRUCTURED_NAMES[j]
                            ),
                            "attribution": float(
                                np.sum(
                                    ax[i, :, j]
                                )
                            ),
                            "abs_attribution": float(
                                np.sum(
                                    np.abs(
                                        ax[i, :, j]
                                    )
                                )
                            ),
                        }
                    )

    pd.DataFrame(modality_rows).to_csv(
        OUT / "28_moe_modality_ablation.csv",
        index=False,
    )

    pd.DataFrame(router_rows).to_csv(
        OUT / "29_moe_router_summary.csv",
        index=False,
    )

    pd.DataFrame(gradient_rows).to_csv(
        OUT / "30_moe_gradient_global.csv",
        index=False,
    )

    pd.DataFrame(local_rows).to_csv(
        OUT / "31_moe_gradient_local.csv",
        index=False,
    )

    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------
# Plots + report
# ---------------------------------------------------------------------
def make_plots() -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(
            "Matplotlib is required only for the report/plot stage. "
            "Install it with: python -m pip install matplotlib"
        ) from exc

    # Tree SHAP summary
    tree = pd.read_csv(
        OUT / "22_tree_shap_global.csv"
    )

    for model_id in sorted(
        tree["model_id"].unique()
    ):
        sdf = tree[
            tree["model_id"] == model_id
        ].copy()

        agg = (
            sdf.groupby(
                [
                    "feature_name",
                    "clinical_category",
                ],
                as_index=False,
            )["mean_abs_shap"]
            .mean()
            .sort_values(
                "mean_abs_shap",
                ascending=False,
            )
            .head(15)
        )

        plt.figure(
            figsize=(9, 6)
        )

        plt.barh(
            agg["feature_name"][::-1],
            agg["mean_abs_shap"][::-1],
        )

        plt.xlabel(
            "Mean absolute SHAP value"
        )

        plt.title(
            f"Model {model_id} — global Tree SHAP"
        )

        plt.tight_layout()

        plt.savefig(
            PLOTS
            / f"tree_shap_global_{model_id}.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

    # S4D temporal attribution
    ig = pd.read_csv(
        OUT / "26_s4d_ig_temporal.csv"
    )
    occ = pd.read_csv(
        OUT / "27_s4d_temporal_occlusion.csv"
    )

    for task in PRIMARY_TASKS:
        a = ig[
            ig["task"] == task
        ]

        b = occ[
            occ["task"] == task
        ]

        plt.figure(
            figsize=(10, 5)
        )

        plt.plot(
            a["hour_number"],
            a["mean_abs_ig"],
            marker="o",
            label="Integrated Gradients",
        )

        plt.plot(
            b["hour_number"],
            b["mean_abs_change"],
            marker="s",
            label="1-hour occlusion",
        )

        plt.xlabel(
            "Hour in 24-hour input window"
        )

        plt.ylabel(
            "Attribution / prediction-change magnitude"
        )

        plt.title(
            f"Model 24 S4D — temporal XAI: {task}"
        )

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            PLOTS
            / f"s4d_temporal_{task}.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

    # MoE modality ablation
    moe = pd.read_csv(
        OUT / "28_moe_modality_ablation.csv"
    )

    agg = (
        moe.groupby(
            ["task", "ablation"],
            as_index=False,
        )["abs_change"]
        .mean()
    )

    tasks = list(
        agg["task"].unique()
    )

    plt.figure(
        figsize=(12, 6)
    )

    x = np.arange(
        len(tasks)
    )

    modalities = sorted(
        agg["ablation"].unique()
    )

    width = 0.25

    for k, modality in enumerate(
        modalities
    ):
        vals = []

        for task in tasks:
            row = agg[
                (agg["task"] == task)
                & (
                    agg["ablation"]
                    == modality
                )
            ]

            vals.append(
                float(
                    row["abs_change"].iloc[0]
                )
            )

        plt.bar(
            x + (k - 1) * width,
            vals,
            width=width,
            label=modality,
        )

    plt.xticks(
        x,
        tasks,
        rotation=25,
        ha="right",
    )

    plt.ylabel(
        "Mean absolute prediction change"
    )

    plt.title(
        "Model 32 — modality ablation"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        PLOTS
        / "moe_modality_ablation.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # Router weights
    router = pd.read_csv(
        OUT / "29_moe_router_summary.csv"
    )

    present = router[
        router["text_missing"] == 0
    ]

    missing = router[
        router["text_missing"] == 1
    ]

    means = pd.DataFrame(
        {
            "group": [
                "Text present",
                "Text missing",
            ],
            "Expert 0": [
                present["expert_0_weight"].mean()
                if not present.empty
                else np.nan,
                missing["expert_0_weight"].mean()
                if not missing.empty
                else np.nan,
            ],
            "Expert 1": [
                present["expert_1_weight"].mean()
                if not present.empty
                else np.nan,
                missing["expert_1_weight"].mean()
                if not missing.empty
                else np.nan,
            ],
        }
    )

    plt.figure(
        figsize=(7, 5)
    )

    x = np.arange(
        len(means)
    )

    width = 0.35

    plt.bar(
        x - width / 2,
        means["Expert 0"],
        width,
        label="Expert 0",
    )

    plt.bar(
        x + width / 2,
        means["Expert 1"],
        width,
        label="Expert 1",
    )

    plt.xticks(
        x,
        means["group"],
    )

    plt.ylabel(
        "Mean router weight"
    )

    plt.title(
        "Model 32 — router weights by text availability"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        PLOTS
        / "moe_router_text_availability.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def write_final_reports(
    method_lock: Dict[str, Any]
) -> None:
    tree = pd.read_csv(
        OUT / "22_tree_shap_global.csv"
    )

    s4d = pd.read_csv(
        OUT / "24_s4d_ig_global.csv"
    )

    moe_ab = pd.read_csv(
        OUT / "28_moe_modality_ablation.csv"
    )

    router = pd.read_csv(
        OUT / "29_moe_router_summary.csv"
    )

    parity = pd.read_csv(
        PARITY_SUMMARY_PATH
    )

    status_rows = [
        {
            "model_id": "00",
            "model_name": "XGBoost",
            "deep_xai_method": "Tree SHAP",
            "status": "completed",
            "parity_verified": True,
        },
        {
            "model_id": "05",
            "model_name": "LightGBM",
            "deep_xai_method": "Tree SHAP",
            "status": "completed",
            "parity_verified": True,
        },
        {
            "model_id": "24",
            "model_name": "S4D",
            "deep_xai_method": (
                "Integrated Gradients + "
                "1-hour temporal occlusion"
            ),
            "status": "completed",
            "parity_verified": True,
        },
        {
            "model_id": "32",
            "model_name": "MoE Modality Fusion",
            "deep_xai_method": (
                "Modality ablation + router/gating "
                "+ input×gradient"
            ),
            "status": "completed",
            "parity_verified": True,
        },
    ]

    pd.DataFrame(status_rows).to_csv(
        OUT / "33_deep_xai_status.csv",
        index=False,
    )

    gaps = [
        {
            "gap_id": "GAP_CONCEPT_MAP",
            "status": "documented",
            "description": (
                "A1 feature mapping is available for clinical "
                "feature/domain annotation, but it is not a formal "
                "concept-level representation."
            ),
            "impact": (
                "Concept-level counterfactual / ConceptCF-style "
                "analysis is not claimed."
            ),
        },
    ]

    pd.DataFrame(gaps).to_csv(
        OUT / "34_deep_xai_evidence_gaps.csv",
        index=False,
    )

    top_tree = (
        tree.groupby(
            [
                "model_id",
                "feature_name",
                "clinical_category",
            ],
            as_index=False,
        )["mean_abs_shap"]
        .mean()
        .sort_values(
            [
                "model_id",
                "mean_abs_shap",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .groupby(
            "model_id",
            as_index=False,
        )
        .head(10)
    )

    top_s4d = (
        s4d.groupby(
            [
                "task",
                "feature_name",
                "clinical_category",
            ],
            as_index=False,
        )["mean_abs_ig"]
        .mean()
        .sort_values(
            [
                "task",
                "mean_abs_ig",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .groupby(
            "task",
            as_index=False,
        )
        .head(10)
    )

    modality_summary = (
        moe_ab.groupby(
            [
                "task",
                "ablation",
            ],
            as_index=False,
        )["abs_change"]
        .mean()
        .rename(
            columns={
                "abs_change":
                    "mean_absolute_prediction_change"
            }
        )
    )

    router_summary = {
        "n_cases": int(len(router)),
        "mean_expert_0_weight": float(
            router["expert_0_weight"].mean()
        ),
        "mean_expert_1_weight": float(
            router["expert_1_weight"].mean()
        ),
        "mean_router_entropy": float(
            router["router_entropy"].mean()
        ),
        "text_present_n": int(
            (router["text_missing"] == 0).sum()
        ),
        "text_missing_n": int(
            (router["text_missing"] == 1).sum()
        ),
    }

    report = []
    report.append(
        "# Z2_tuned Complete XAI Final Report\n"
    )

    report.append(
        "## 1. Scope and locked predictor\n\n"
        "The primary predictive model is the locked `Z2_tuned` "
        "ensemble. Deep XAI is performed on four provenance-selected "
        "base models: 00 XGBoost, 05 LightGBM, 24 S4D, and 32 MoE "
        "Modality Fusion. These explanations are provenance/component "
        "explanations supporting interpretation of Z2_tuned; they are "
        "not claimed to be direct gradients or SHAP values of the "
        "Z2_tuned ensemble itself.\n"
    )

    report.append(
        "## 2. Prediction-parity gate\n\n"
        "The exact locked checkpoints/models were reconstructed using "
        "the original project source implementations and compared "
        "against the saved prediction artifacts using "
        "`stay_id + subject_id + split + task`. All four models passed "
        "identity, validity, and numerical prediction parity.\n\n"
        + parity.to_markdown(index=False)
    )

    report.append(
        "\n## 3. Validation-frozen XAI settings\n\n"
        "The XAI configuration below was selected/frozen before the "
        "test explanations were generated.\n\n"
        + json.dumps(
            method_lock,
            indent=2,
        )
    )

    report.append(
        "\n## 4. Model 00 and Model 05 — Tree SHAP\n\n"
        "Tree SHAP was applied on the locked 554-dimensional tabular "
        "contract using the exact serialized estimators. SHAP values "
        "are interpreted on the model raw-output scale: classification "
        "models on their tree output scale and LOS on the regression "
        "output scale. Global importance is the mean absolute SHAP "
        "value across the deterministic test explanation sample.\n\n"
        "### Top features\n\n"
        + top_tree.to_markdown(index=False)
    )

    report.append(
        "\n## 5. Model 24 — S4D\n\n"
        "Integrated Gradients were applied to the exact 24×110 structured "
        "sequence plus four demographic inputs. Temporal attribution was "
        "summarized by hour and independently checked with one-hour "
        "temporal occlusion. Feature-level attribution is aggregated over "
        "the 24-hour sequence.\n\n"
        "### Top structured features\n\n"
        + top_s4d.to_markdown(index=False)
    )

    report.append(
        "\n## 6. Model 32 — MoE Modality Fusion\n\n"
        "The analysis uses the exact clinical-note parquet source, the "
        "locked checkpoint, and the explicit missing-text flag. "
        "Single-modality ablations quantify sensitivity to structured "
        "data, demographics, and clinical text. Router analysis reports "
        "the exact shared expert probabilities in evaluation mode. "
        "Input×gradient provides additional structured and multimodal "
        "attribution evidence.\n\n"
        "### Mean modality-ablation effect\n\n"
        + modality_summary.to_markdown(index=False)
        + "\n\n### Router summary\n\n"
        + json.dumps(
            router_summary,
            indent=2,
        )
    )

    report.append(
        "\n## 7. Clinical feature mapping limitation\n\n"
        "The available `a1_feature_mapping_audit.csv` is used only "
        "for clinical domain/category annotation. It is not treated "
        "as a formal concept map. Therefore the final XAI package "
        "does not claim a ConceptCF-style concept-level counterfactual "
        "representation where none was actually implemented.\n"
    )

    report.append(
        "\n## 8. Final XAI status\n\n"
        "Tier 1 ensemble-level XAI is complete. Tier 2 provenance tracing "
        "is complete. Tier 3 deep model-specific XAI is complete for "
        "Models 00, 05, 24, and 32. The only explicitly documented "
        "limitation is the absence of a formal concept-level representation "
        "for concept counterfactual explanations.\n"
    )

    (OUT / "35_DEEP_XAI_FINAL_REPORT.md").write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    old_report = OUT / "16_XAI_REPORT.md"
    combined_report = OUT / "16_XAI_REPORT_FINAL.md"

    if old_report.exists():
        previous = old_report.read_text(
            encoding="utf-8"
        )

        combined_report.write_text(
            previous
            + "\n\n---\n\n"
            + "\n".join(report),
            encoding="utf-8",
        )

    manifest = {
        "project": "Pipeline A V2 / Z2_tuned",
        "xai_status": "complete_with_documented_concept_mapping_limitation",
        "parity_gate": "passed_for_00_05_24_32",
        "deep_xai_models": [
            "00",
            "05",
            "24",
            "32",
        ],
        "artifacts": sorted(
            str(p.relative_to(OUT))
            for p in OUT.rglob("*")
            if p.is_file()
        ),
        "method_lock": method_lock,
        "feature_name_source": STRUCTURED_NAME_SOURCE,
    }

    json_dump(
        manifest,
        OUT / "36_deep_xai_package_manifest.json",
    )


def package_final() -> None:
    import zipfile

    zip_path = (
        ROOT
        / "outputs"
        / "FYP_Z2_COMPLETE_XAI_FINAL.zip"
    )

    with zipfile.ZipFile(
        zip_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as z:
        for p in OUT.rglob("*"):
            if p.is_file():
                z.write(
                    p,
                    p.relative_to(ROOT),
                )

    print("\nFINAL PACKAGE")
    print(zip_path)
    print("size_mb:", round(
        zip_path.stat().st_size / 1024 / 1024,
        2,
    ))


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--stage",
        required=True,
        choices=[
            "lock",
            "test",
            "report",
            "all",
        ],
    )

    args = parser.parse_args()

    if args.stage in (
        "lock",
        "test",
        "report",
        "all",
    ):
        require_parity()

    if args.stage in (
        "lock",
        "all",
    ):
        print("=" * 100)
        print("VALIDATION-BASED XAI METHOD LOCK")
        print("=" * 100)

        lock = validation_xai_lock()

        print(
            json.dumps(
                lock,
                indent=2,
            )
        )

    if args.stage in (
        "test",
        "all",
    ):
        lock_path = OUT / "21_xai_method_lock.json"

        if not lock_path.exists():
            raise RuntimeError(
                "Run --stage lock first."
            )

        lock = json.loads(
            lock_path.read_text()
        )

        print("=" * 100)
        print("MODEL 00 / 05 — TREE SHAP")
        print("=" * 100)

        run_tree_shap()

        print("=" * 100)
        print("MODEL 24 — S4D IG + TEMPORAL OCCLUSION")
        print("=" * 100)

        run_s4d_deep_xai(
            lock
        )

        print("=" * 100)
        print("MODEL 32 — MOE MODALITY / ROUTER / GRADIENT XAI")
        print("=" * 100)

        run_moe_deep_xai(
            lock
        )

    if args.stage in (
        "report",
        "all",
    ):
        lock_path = OUT / "21_xai_method_lock.json"

        if not lock_path.exists():
            raise RuntimeError(
                "Run --stage lock first."
            )

        lock = json.loads(
            lock_path.read_text()
        )

        required = [
            "22_tree_shap_global.csv",
            "23_tree_shap_local.csv",
            "24_s4d_ig_global.csv",
            "25_s4d_ig_local.csv",
            "26_s4d_ig_temporal.csv",
            "27_s4d_temporal_occlusion.csv",
            "28_moe_modality_ablation.csv",
            "29_moe_router_summary.csv",
            "30_moe_gradient_global.csv",
            "31_moe_gradient_local.csv",
        ]

        missing = [
            name
            for name in required
            if not (OUT / name).exists()
        ]

        if missing:
            raise RuntimeError(
                "Deep-XAI outputs missing:\n"
                + "\n".join(missing)
            )

        print("=" * 100)
        print("FINAL XAI REPORT / PLOTS / PACKAGE")
        print("=" * 100)

        make_plots()
        write_final_reports(
            lock
        )
        package_final()

        print("\nXAI FINALIZATION COMPLETE")


if __name__ == "__main__":
    main()
