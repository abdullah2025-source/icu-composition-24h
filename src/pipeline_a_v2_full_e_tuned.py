#!/usr/bin/env python3
"""
Pipeline A V2 — Full-E Tuned Model

This is the tuning-stage driver for the EXACT Full-E architecture contained in:
    pipeline_a_v2_proposed_model_family.py

It does NOT redesign Full-E. It imports the existing family implementation and
tunes only training/model hyperparameters around that fixed architecture.

Primary selection metric:
    mean validation AUPRC across:
        mortality_24h
        ventilation_onset_24h
        vasopressor_onset_24h
        readmission_30d_24h

Test data are never used for Optuna selection.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib.util
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


PROJECT_ROOT = Path("/home/abdullah2025/fyp_project")
FAMILY_SCRIPT = PROJECT_ROOT / "pipeline_a_v2_proposed_model_family.py"
TUNED_ROOT = PROJECT_ROOT / "outputs" / "proposed_model_family" / "tuned" / "E_Full-E"
TRIALS_ROOT = TUNED_ROOT / "trials"
FINAL_ROOT = TUNED_ROOT / "final_selected_configuration"
STUDY_DB = TUNED_ROOT / "study.db"
STUDY_NAME_DEFAULT = "pipeline_a_v2_full_e"

SEED = 20260923
FIXED_REFERENCE_DEFAULT = 0.346098


def load_family_module():
    """Load the exact family script as a Python module without changing its architecture."""
    if not FAMILY_SCRIPT.exists():
        raise FileNotFoundError(
            f"Required family source not found: {FAMILY_SCRIPT}\n"
            "Place pipeline_a_v2_proposed_model_family.py in /home/abdullah2025/fyp_project."
        )

    spec = importlib.util.spec_from_file_location("pipeline_a_v2_proposed_model_family", FAMILY_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load family script: {FAMILY_SCRIPT}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


FAMILY = load_family_module()

# The uploaded family-screening file references stable_seed() in evaluation but
# does not define it. Adding it to the imported module preserves all existing
# architecture/evaluation code while fixing that helper at runtime.
if not hasattr(FAMILY, "stable_seed"):
    def stable_seed(label: str) -> int:
        digest = hashlib.sha256(f"{SEED}|{label}".encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "little") % (2**32 - 1)
    FAMILY.stable_seed = stable_seed


def parse_args():
    p = argparse.ArgumentParser(
        description="Pipeline A V2 — exact Full-E architecture with Optuna tuning"
    )
    p.add_argument("--mode", required=True, choices=["audit", "tuned"])
    p.add_argument("--n-trials", type=int, default=30)
    p.add_argument("--bootstrap-reps", type=int, default=2000)
    p.add_argument("--study-name", default=STUDY_NAME_DEFAULT)
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def require_optuna():
    if FAMILY.optuna is None:
        raise RuntimeError("Optuna is required. Install it in mamba_env.")


def set_all_seeds(seed: int) -> None:
    FAMILY.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def fixed_reference_score() -> float:
    summary_path = (
        PROJECT_ROOT
        / "outputs"
        / "proposed_model_family"
        / "family_fixed_summary.csv"
    )
    if summary_path.exists():
        try:
            df = pd.read_csv(summary_path)
            row = df[df["version"].astype(str).str.upper().eq("E")]
            if not row.empty:
                value = float(row["validation_mean_auprc"].iloc[0])
                if np.isfinite(value):
                    return value
        except Exception:
            pass
    return FIXED_REFERENCE_DEFAULT


def assert_full_e_architecture() -> int:
    """
    Guard against accidental architecture drift.

    Full-E fixed screening had 2,858,060 trainable parameters. The model
    parameter count is independent of dropout, so this is a strong invariant.
    """
    FAMILY.require_torch()
    model = FAMILY.FamilyModel("E", dropout=float(FAMILY.FIRST_RUN["dropout"]))
    count = FAMILY.model_parameter_count(model)
    del model

    expected = 2_858_060
    if count != expected:
        raise RuntimeError(
            f"Full-E architecture drift detected: parameter_count={count}, expected={expected}."
        )
    return count


def trial_params(trial) -> Dict[str, Any]:
    """
    Registered tuning space.

    Architecture is unchanged. These are optimization/training hyperparameters
    consistent with the project-wide tuning protocol.
    """
    return {
        "lr": trial.suggest_float("lr", 1e-4, 1e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [256, 512, 1024]),
        "dropout": trial.suggest_float("dropout", 0.05, 0.30),
        "scheduler": trial.suggest_categorical("scheduler", ["cosine", "onecycle"]),
        "grad_clip": trial.suggest_float("grad_clip", 0.5, 2.0),
        "patience": trial.suggest_int("patience", 5, 10),
        "max_epochs": 50,
        "seed": SEED,
        "amp": "BF16",
    }


def train_trial(
    data,
    device,
    params: Dict[str, Any],
    trial,
) -> float:
    """
    Train one Full-E trial and return the best validation mean AUPRC.

    No test rows are touched here.
    No checkpoint is persisted for ordinary trials.
    """
    set_all_seeds(SEED)

    model = None
    try:
        model = FAMILY.FamilyModel("E", dropout=float(params["dropout"])).to(device)

        optimizer = FAMILY.build_optimizer(
            model,
            float(params["lr"]),
            float(params["weight_decay"]),
        )

        steps_per_epoch = math.ceil(
            len(data.train_idx) / int(params["batch_size"])
        )

        scheduler = FAMILY.build_scheduler(
            optimizer,
            str(params["scheduler"]),
            int(params["max_epochs"]),
            steps_per_epoch,
            float(params["lr"]),
        )

        train_params = dict(params)
        train_params["optimizer"] = optimizer
        train_params["scheduler_obj"] = scheduler

        best_score = float("-inf")
        no_improve = 0

        for epoch in range(1, int(params["max_epochs"]) + 1):
            train_loss = FAMILY.run_epoch_train(
                model,
                data,
                data.train_idx,
                device,
                train_params,
            )

            val_score, val_los_mae, _ = FAMILY.run_validation(
                model,
                data,
                data.val_idx,
                device,
                params,
                bootstrap_reps=0,
            )

            trial.report(float(val_score), step=epoch)

            print(
                f"[Full-E][trial={trial.number:02d}] "
                f"epoch={epoch:02d} "
                f"train_loss={train_loss:.6f} "
                f"val_mean_AUPRC={val_score:.6f} "
                f"val_LOS_MAE={val_los_mae:.4f}",
                flush=True,
            )

            if trial.should_prune():
                raise FAMILY.optuna.TrialPruned(
                    f"Median-pruned at epoch {epoch}; val_mean_AUPRC={val_score:.6f}"
                )

            if val_score > best_score:
                best_score = float(val_score)
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= int(params["patience"]):
                break

        return float(best_score)

    finally:
        if model is not None:
            del model
        gc.collect()
        if FAMILY.torch.cuda.is_available():
            try:
                FAMILY.torch.cuda.synchronize()
                FAMILY.torch.cuda.empty_cache()
            except Exception:
                pass


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def final_retrain(
    data,
    device,
    params: Dict[str, Any],
):
    """
    Retrain the exact Full-E architecture using the best Optuna parameters.
    The checkpoint is selected on validation mean AUPRC.
    Test remains untouched until after checkpoint selection.
    """
    FINAL_ROOT.mkdir(parents=True, exist_ok=True)

    # Reuse the family's canonical trainer so the final training behavior,
    # early stopping, checkpointing, and scheduler semantics remain identical.
    model, best_score, best_epoch, history, training_seconds = FAMILY.train_family_model(
        data,
        params,
        device,
        "E",
        FINAL_ROOT,
    )

    return model, float(best_score), int(best_epoch), history, float(training_seconds)


def save_tuned_final_outputs(
    data,
    model,
    device,
    params: Dict[str, Any],
    best_score: float,
    best_epoch: int,
    training_seconds: float,
    bootstrap_reps: int,
    study,
    architecture_param_count: int,
):
    """
    Save the standard family outputs plus tuning provenance.
    """
    # Reuse the canonical output writer to preserve exactly the same metrics,
    # predictions, calibration, text-stratified outputs, and LOS convention.
    row = FAMILY.save_family_outputs(
        data,
        model,
        "E",
        data.val_idx,
        data.test_idx,
        device,
        params,
        FINAL_ROOT,
        best_score,
        best_epoch,
        training_seconds,
        bootstrap_reps,
    )

    meta_path = FINAL_ROOT / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    meta.update(
        {
            "mode": "tuned",
            "tuning": {
                "study_name": study.study_name,
                "study_db": str(STUDY_DB),
                "n_trials_target": int(len(study.trials)),
                "n_trials_completed": int(
                    sum(t.state == FAMILY.optuna.trial.TrialState.COMPLETE for t in study.trials)
                ),
                "n_trials_pruned": int(
                    sum(t.state == FAMILY.optuna.trial.TrialState.PRUNED for t in study.trials)
                ),
                "best_trial_number": int(study.best_trial.number),
                "best_validation_mean_auprc": float(study.best_value),
                "best_params": dict(study.best_trial.params),
                "selection_metric": (
                    "Mean validation AUPRC across mortality, ventilation, "
                    "vasopressor, readmission"
                ),
                "test_used_for_selection": False,
                "architecture_locked": True,
            },
            "fixed_reference": {
                "validation_mean_auprc": float(fixed_reference_score()),
                "tuned_beats_fixed_on_validation": bool(
                    best_score > fixed_reference_score()
                ),
                "decision_rule": (
                    "Tuned Full-E is retained only if its validation mean AUPRC "
                    "strictly exceeds the fixed Full-E validation mean AUPRC."
                ),
            },
            "architecture": {
                **meta.get("architecture", {}),
                "architecture_status": "same_as_fixed_full_E",
                "parameter_count_invariant": int(architecture_param_count),
            },
            "test_used_for_selection": False,
            "project_selection_metric": (
                "Mean validation AUPRC across mortality, ventilation, "
                "vasopressor, readmission"
            ),
        }
    )

    meta_path.write_text(
        json.dumps(meta, indent=2, default=str),
        encoding="utf-8",
    )

    # Preserve a copy of the study's trial table.
    trials_df = study.trials_dataframe(attrs=("number", "value", "state"))
    trials_df.to_csv(TUNED_ROOT / "optuna_trials.csv", index=False)

    save_json(
        TUNED_ROOT / "best_params.json",
        {
            "model": "Full-E",
            "architecture_locked": True,
            "parameter_count": architecture_param_count,
            "best_trial": int(study.best_trial.number),
            "best_validation_mean_auprc": float(study.best_value),
            "params": dict(study.best_trial.params),
            "fixed_reference_validation_mean_auprc": float(fixed_reference_score()),
            "tuned_beats_fixed_on_validation": bool(
                study.best_value > fixed_reference_score()
            ),
        },
    )

    save_json(
        TUNED_ROOT / "tuning_summary.json",
        {
            "study_name": study.study_name,
            "study_db": str(STUDY_DB),
            "n_trials_target": len(study.trials),
            "n_trials_completed": int(
                sum(t.state == FAMILY.optuna.trial.TrialState.COMPLETE for t in study.trials)
            ),
            "n_trials_pruned": int(
                sum(t.state == FAMILY.optuna.trial.TrialState.PRUNED for t in study.trials)
            ),
            "best_trial": int(study.best_trial.number),
            "best_validation_mean_auprc": float(study.best_value),
            "fixed_validation_mean_auprc": float(fixed_reference_score()),
            "tuned_beats_fixed": bool(study.best_value > fixed_reference_score()),
            "test_used_for_selection": False,
            "architecture_parameter_count": architecture_param_count,
        },
    )

    return row


def run_audit(data):
    audit = FAMILY.contract_audit(data)
    print(json.dumps(audit, indent=2))
    if not audit["all_pass"]:
        raise RuntimeError("Frozen V2 contract audit failed.")


def run_tuned(args):
    require_optuna()
    FAMILY.require_torch()

    if args.n_trials <= 0:
        raise ValueError("--n-trials must be > 0")
    if args.bootstrap_reps < 0:
        raise ValueError("--bootstrap-reps must be >= 0")

    TUNED_ROOT.mkdir(parents=True, exist_ok=True)
    TRIALS_ROOT.mkdir(parents=True, exist_ok=True)

    set_all_seeds(args.seed)

    data = FAMILY.load_frozen_v2()
    audit = FAMILY.save_contract_audit(data)
    print("=" * 100)
    print("PIPELINE A V2 — FULL-E TUNED MODEL")
    print("=" * 100)
    print(json.dumps(audit, indent=2))

    if not audit["all_pass"]:
        raise RuntimeError("Frozen V2 contract audit failed.")

    device = FAMILY.get_device()
    print(f"device={device}", flush=True)
    if device.type == "cuda":
        print(f"GPU={FAMILY.torch.cuda.get_device_name(0)}", flush=True)
        print(
            f"BF16 supported={FAMILY.torch.cuda.is_bf16_supported()}",
            flush=True,
        )
    FAMILY.validate_gpu_bf16(device, "Full-E tuning")

    architecture_param_count = assert_full_e_architecture()
    print(
        f"Full-E architecture parameter count verified: {architecture_param_count}",
        flush=True,
    )

    storage = f"sqlite:///{STUDY_DB}"

    sampler = FAMILY.optuna.samplers.TPESampler(
        seed=args.seed,
        multivariate=True,
        group=True,
    )
    pruner = FAMILY.optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=5,
        interval_steps=1,
    )

    study = FAMILY.optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
    )

    complete = sum(
        t.state == FAMILY.optuna.trial.TrialState.COMPLETE
        for t in study.trials
    )
    remaining = max(0, args.n_trials - len(study.trials))

    print(
        json.dumps(
            {
                "study_name": study.study_name,
                "storage": str(STUDY_DB),
                "existing_trials": len(study.trials),
                "existing_complete_trials": int(complete),
                "target_total_trials": int(args.n_trials),
                "remaining_trials_to_target": int(remaining),
                "fixed_reference_validation_mean_auprc": fixed_reference_score(),
                "architecture_parameter_count": architecture_param_count,
            },
            indent=2,
        ),
        flush=True,
    )

    if remaining > 0:
        tuning_start = time.time()

        def objective(trial):
            params = trial_params(trial)
            trial.set_user_attr("model", "Full-E")
            trial.set_user_attr("architecture_locked", True)
            trial.set_user_attr("parameter_count", architecture_param_count)
            trial.set_user_attr("params", json.dumps(params, sort_keys=True))
            return train_trial(data, device, params, trial)

        study.optimize(
            objective,
            n_trials=remaining,
            gc_after_trial=True,
            show_progress_bar=False,
        )

        tuning_seconds = time.time() - tuning_start
        save_json(
            TUNED_ROOT / "optuna_runtime.json",
            {
                "tuning_seconds": tuning_seconds,
                "target_total_trials": args.n_trials,
                "total_trials_in_study": len(study.trials),
            },
        )

    if not study.trials:
        raise RuntimeError("Optuna study contains no trials.")

    best = study.best_trial
    best_params = dict(best.params)
    best_params.update(
        {
            "max_epochs": 50,
            "seed": SEED,
            "amp": "BF16",
        }
    )

    print("=" * 100)
    print("FULL-E OPTUNA BEST TRIAL")
    print("=" * 100)
    print(
        json.dumps(
            {
                "best_trial": int(best.number),
                "best_validation_mean_auprc": float(best.value),
                "fixed_validation_mean_auprc": float(fixed_reference_score()),
                "tuned_beats_fixed_on_validation": bool(
                    best.value > fixed_reference_score()
                ),
                "params": best_params,
            },
            indent=2,
        ),
        flush=True,
    )

    # Final retrain with best hyperparameters.
    model, best_score, best_epoch, history, train_seconds = final_retrain(
        data,
        device,
        best_params,
    )

    # Definitive metrics + predictions only after validation checkpoint selection.
    row = save_tuned_final_outputs(
        data,
        model,
        device,
        best_params,
        best_score,
        best_epoch,
        train_seconds,
        args.bootstrap_reps,
        study,
        architecture_param_count,
    )

    print("=" * 100)
    print("FULL-E TUNED FINAL RESULT")
    print("=" * 100)
    print(json.dumps(row, indent=2))

    comparison = pd.DataFrame(
        [
            {
                "model": "Full-E fixed",
                "validation_mean_auprc": fixed_reference_score(),
                "status": "reference",
            },
            {
                "model": "Full-E tuned",
                "validation_mean_auprc": float(best_score),
                "status": (
                    "beats_fixed"
                    if best_score > fixed_reference_score()
                    else "does_not_beat_fixed"
                ),
            },
        ]
    )
    comparison.to_csv(TUNED_ROOT / "tuned_vs_fixed_validation_comparison.csv", index=False)

    del model
    gc.collect()
    if FAMILY.torch.cuda.is_available():
        try:
            FAMILY.torch.cuda.synchronize()
            FAMILY.torch.cuda.empty_cache()
        except Exception:
            pass


def main():
    args = parse_args()

    if args.mode == "audit":
        FAMILY.require_torch()
        data = FAMILY.load_frozen_v2()
        run_audit(data)
        return

    run_tuned(args)


if __name__ == "__main__":
    main()
