# ==================================================================================================
# FINAL BENCHMARK AUDIT — CANONICAL FAMILY STATISTICS ONLY
# ==================================================================================================

from pathlib import Path
import pandas as pd
import numpy as np
import re
import json


ROOT = Path("outputs")
REPORT_DIR = ROOT / "benchmark_selection"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


MODEL_NAMES = {
    "00": "XGBoost",
    "01": "Logistic Regression",
    "02": "Regularized Logistic Regression",
    "03": "Random Forest",
    "04": "Extra Trees",
    "05": "LightGBM",
    "06": "GRU",
    "07": "LSTM",
    "08": "BiGRU",
    "09": "GRU-D",
    "10": "TCN",
    "11": "ModernTCN",
    "12": "CNN+GRU",
    "13": "RETAIN",
    "14": "AdaCare",
    "15": "StageNet",
    "16": "ConCare",
    "17": "Transformer",
    "18": "Transformer+Conv",
    "19": "Transformer+LSTM",
    "20": "TFT",
    "21": "PatchTST",
    "22": "TimesNet",
    "23": "iTransformer",
    "24": "S4D",
    "25": "Mamba-1",
    "26": "Mamba-2",
    "27": "Mamba-3",
    "28": "Hybrid Mamba-Transformer",
    "29": "Multimodal Progression",
    "30": "Cross-Attention Multimodal",
    "31": "Gated Fusion/Cross-Attn",
    "32": "MoE Modality Fusion",
}

FAMILY_NAMES = {
    1: "Tabular / Tree / Linear",
    2: "RNN",
    3: "TCN / CNN",
    4: "Clinical Sequence",
    5: "Transformer / Modern Time-Series",
    6: "State-Space / Selective SSM",
    7: "Advanced Multimodal",
}

# IMPORTANT:
# These are already-generated canonical statistical artifacts.
# Family 1 statistics are NOT calculated here.
# Family 1 CIs come from evaluate_family01_stats.py.
FAMILY_FILES = {
    1: ROOT / "baseline_family_01_tabular" / "stats" / "family01_all_metrics_with_stats.csv",
    2: ROOT / "baseline_family_02_rnn" / "stats" / "family02_all_metrics_with_stats.csv",
    3: ROOT / "baseline_family_03_tcn_10_12" / "family03_statistical_evaluation.csv",
    4: ROOT / "baseline_family_04_clinical_sequence_13_16" / "family04_statistical_evaluation.csv",
    5: ROOT / "baseline_family_05_transformer_modern_17_23" / "family05_statistical_evaluation.csv",
    6: ROOT / "baseline_family_06_state_space_24_27" / "family06_statistical_evaluation.csv",
    7: ROOT / "baseline_family_07_advanced_multimodal_28_32" / "family07_statistical_evaluation.csv",
}

EXPECTED_FAMILY_IDS = {
    1: ["00", "01", "02", "03", "04", "05"],
    2: ["06", "07", "08", "09"],
    3: ["10", "11", "12"],
    4: ["13", "14", "15", "16"],
    5: ["17", "18", "19", "20", "21", "22", "23"],
    6: ["24", "25", "26", "27"],
    7: ["28", "29", "30", "31", "32"],
}

BINARY_TASKS = {
    "mortality_24h",
    "ventilation_onset_24h",
    "vasopressor_onset_24h",
    "readmission_30d_24h",
}

EXPECTED_CANONICAL_ROWS = 672
EXPECTED_FINAL_TEST_ROWS = 396
EXPECTED_RANKING_ROWS = 66


def family_num(mid):
    n = int(mid)
    if 0 <= n <= 5:
        return 1
    if 6 <= n <= 9:
        return 2
    if 10 <= n <= 12:
        return 3
    if 13 <= n <= 16:
        return 4
    if 17 <= n <= 23:
        return 5
    if 24 <= n <= 27:
        return 6
    if 28 <= n <= 32:
        return 7
    return None


def norm_text(x):
    if x is None or pd.isna(x):
        return ""
    return re.sub(r"[^a-z0-9]+", "", str(x).lower())


NAME_TO_ID = {
    norm_text(v): k
    for k, v in MODEL_NAMES.items()
}


ALIASES = {
    "06": ["gru"],
    "07": ["lstm"],
    "08": ["bigru", "bi_gru"],
    "09": ["gru-d", "grud", "gru_d"],
    "10": ["tcn"],
    "11": ["moderntcn", "modern_tcn"],
    "12": ["cnngru", "cnn_gru", "cnn+gru"],
    "13": ["retain"],
    "14": ["adacare", "ada_care"],
    "15": ["stagenet", "stage_net"],
    "16": ["concare", "con_care"],
    "17": ["transformer"],
    "18": ["transformerconv", "transformer+conv"],
    "19": ["transformerlstm", "transformer+lstm"],
    "20": ["tft"],
    "21": ["patchtst"],
    "22": ["timesnet"],
    "23": ["itransformer"],
    "24": ["s4d"],
    "25": ["mamba1", "mamba-1"],
    "26": ["mamba2", "mamba-2"],
    "27": ["mamba3", "mamba-3"],
    "28": ["hybridmambatransformer"],
    "29": ["multimodalprogression"],
    "30": ["crossattentionmultimodal"],
    "31": ["gatedfusioncrossattn", "gatedfusioncrossattention"],
    "32": ["moemodalityfusion"],
}


def infer_id(row, expected_ids):
    for key in ["model_id", "_model_id"]:
        if key not in row.index:
            continue
        value = row[key]
        if pd.isna(value):
            continue

        s = str(value).strip()
        match = re.fullmatch(r"\d+(\.0)?", s)
        if match:
            mid = s.split(".")[0].zfill(2)
            if mid in expected_ids:
                return mid

        s_norm = norm_text(s)
        if s_norm in NAME_TO_ID and NAME_TO_ID[s_norm] in expected_ids:
            return NAME_TO_ID[s_norm]

    for key in ["model_name", "_model_name"]:
        if key not in row.index:
            continue
        value = row[key]
        if pd.isna(value):
            continue

        s = str(value)
        sn = norm_text(s)

        for mid in expected_ids:
            canonical = norm_text(MODEL_NAMES[mid])
            if canonical and (
                canonical == sn
                or canonical in sn
                or sn in canonical
            ):
                return mid

        for mid in expected_ids:
            for alias in ALIASES.get(mid, []):
                if norm_text(alias) in sn:
                    return mid

        match = re.search(r"(?<!\d)(\d{1,2})(?:_|$)", s)
        if match:
            mid = match.group(1).zfill(2)
            if mid in expected_ids:
                return mid

    return None


def find_col(df, exact=(), contains=()):
    low = {str(c).lower(): c for c in df.columns}

    for x in exact:
        if x.lower() in low:
            return low[x.lower()]

    for c in df.columns:
        lc = str(c).lower()
        if any(token.lower() in lc for token in contains):
            return c

    return None


def ci_col(df, metric, side):
    candidates = [
        f"{metric}_ci95_{side}",
        f"{metric}_ci_{side}",
        f"{metric}_ci{side}",
        f"{metric}_{side}",
    ]

    low = {str(c).lower(): c for c in df.columns}

    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]

    for c in df.columns:
        lc = str(c).lower()
        if metric.lower() in lc and "ci" in lc and side.lower() in lc:
            if side == "low" and "high" not in lc:
                return c
            if side == "high" and "low" not in lc:
                return c

    return None


def main():
    print("\n" + "=" * 110)
    print("FINAL BENCHMARK AUDIT — CANONICAL FAMILY STATISTICS ONLY")
    print("=" * 110)

    frames = []

    for fam, path in FAMILY_FILES.items():
        print(f"\nFAMILY {fam}: {path}")

        if not path.exists():
            raise SystemExit(
                f"ERROR: REQUIRED CANONICAL FILE MISSING:\n{path}"
            )

        df = pd.read_csv(path).copy()
        df["_family_source"] = fam
        df["_source_path"] = str(path)

        print(f"  rows={len(df)}")
        print(f"  columns={list(df.columns)}")

        frames.append(df)

    all_df = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    task_col = find_col(
        all_df,
        exact=("task", "target", "outcome"),
        contains=("task", "target", "outcome"),
    )

    split_col = find_col(
        all_df,
        exact=("split",),
        contains=("split",),
    )

    mode_col = find_col(
        all_df,
        exact=("mode",),
        contains=("mode",),
    )

    model_id_col = find_col(
        all_df,
        exact=("model_id",),
        contains=("model_id",),
    )

    model_name_col = find_col(
        all_df,
        exact=("model_name",),
        contains=("model_name",),
    )

    auprc_col = find_col(
        all_df,
        exact=("auprc",),
        contains=("auprc",),
    )

    auroc_col = find_col(
        all_df,
        exact=("auroc",),
        contains=("auroc",),
    )

    brier_col = find_col(
        all_df,
        exact=("brier",),
        contains=("brier",),
    )

    ece_col = find_col(
        all_df,
        exact=("ece",),
        contains=("ece",),
    )

    mae_col = find_col(
        all_df,
        exact=("mae",),
        contains=("mae",),
    )

    if task_col is None or mode_col is None or auprc_col is None:
        raise SystemExit(
            "ERROR: required columns missing. "
            f"task={task_col}, mode={mode_col}, auprc={auprc_col}"
        )

    print("\nDetected columns:")
    print(" task   =", task_col)
    print(" split  =", split_col)
    print(" mode   =", mode_col)
    print(" id     =", model_id_col)
    print(" name   =", model_name_col)
    print(" auprc  =", auprc_col)
    print(" auroc  =", auroc_col)
    print(" brier  =", brier_col)
    print(" ece    =", ece_col)
    print(" mae    =", mae_col)

    # Copy/normalize existing CI columns only.
    # No statistics are calculated here.
    for metric in [
        "auprc",
        "auroc",
        "brier",
        "mae",
        "rmse",
        "median_ae",
        "r2",
    ]:
        for side in ["low", "high"]:
            src = ci_col(all_df, metric, side)
            dst = f"{metric}_ci95_{side}"
            if src is not None and dst not in all_df.columns:
                all_df[dst] = all_df[src]

    all_df["_model_id"] = None

    for fam in sorted(FAMILY_FILES):
        mask = all_df["_family_source"].eq(fam)
        expected = EXPECTED_FAMILY_IDS[fam]
        subset = all_df.loc[mask].copy()

        subset["_tmp_model_id"] = subset.apply(
            lambda r: infer_id(r, expected),
            axis=1,
        )

        all_df.loc[mask, "_model_id"] = subset["_tmp_model_id"].values

    all_df["_model_id"] = all_df["_model_id"].astype("string")
    all_df["_model_name_canonical"] = all_df["_model_id"].map(MODEL_NAMES)

    all_df["_family_num"] = all_df["_model_id"].apply(
        lambda x: family_num(str(x)) if pd.notna(x) else np.nan
    )

    all_df["_family_name"] = all_df["_family_num"].map(FAMILY_NAMES)

    all_df["_task_norm"] = (
        all_df[task_col]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    all_df["_mode_norm"] = (
        all_df[mode_col]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # ------------------------------------------------------------------
    # CRITICAL SPLIT FIX
    #
    # Family 1 and Family 2 canonical stats files have no split column.
    # They are already held-out evaluation/statistics artifacts.
    # Therefore their split is represented as test_assumed.
    #
    # Families 3-7 have explicit split columns and preserve them.
    # ------------------------------------------------------------------

    all_df["_split_norm"] = "test_assumed"

    if split_col is None:
        raise SystemExit(
            "ERROR: no split column was detected anywhere, but Families 3-7 "
            "require explicit validation/test split information."
        )

    for fam in [3, 4, 5, 6, 7]:
        fam_mask = all_df["_family_source"].eq(fam)

        all_df.loc[fam_mask, "_split_norm"] = (
            all_df.loc[fam_mask, split_col]
            .astype(str)
            .str.strip()
            .str.lower()
        )

    # Explicit documentation for Family 1 and Family 2.
    all_df.loc[
        all_df["_family_source"].eq(1),
        "_split_norm",
    ] = "test_assumed"

    all_df.loc[
        all_df["_family_source"].eq(2),
        "_split_norm",
    ] = "test_assumed"

    print("\n" + "=" * 110)
    print("CANONICAL SOURCE SANITY CHECK")
    print("=" * 110)

    for fam in range(1, 8):
        fam_df = all_df[all_df["_family_source"].eq(fam)]

        print(
            f"Family {fam}: "
            f"rows={len(fam_df)} "
            f"test_assumed={int((fam_df['_split_norm'] == 'test_assumed').sum())} "
            f"test={int((fam_df['_split_norm'] == 'test').sum())} "
            f"val={int((fam_df['_split_norm'] == 'val').sum())}"
        )

    print("\n" + "=" * 110)
    print("MODEL COVERAGE BEFORE DEDUPLICATION")
    print("=" * 110)

    for fam in range(1, 8):
        expected = set(EXPECTED_FAMILY_IDS[fam])

        found = set(
            all_df.loc[
                all_df["_family_source"].eq(fam),
                "_model_id",
            ]
            .dropna()
            .astype(str)
        )

        missing = sorted(expected - found)

        print(
            f"Family {fam}: "
            f"expected={len(expected)} "
            f"found={len(found)} "
            f"missing={missing if missing else 'NONE'}"
        )

        if missing:
            raise SystemExit(
                f"ERROR: Family {fam} is missing expected model IDs: {missing}"
            )

    dedupe_keys = [
        "_model_id",
        "_mode_norm",
        "_split_norm",
        "_task_norm",
    ]

    before = len(all_df)

    all_df = all_df.drop_duplicates(
        subset=dedupe_keys,
        keep="first",
    )

    after = len(all_df)

    print("\nDedupe:")
    print(" before =", before)
    print(" after  =", after)
    print(" removed =", before - after)

    if after != EXPECTED_CANONICAL_ROWS:
        raise SystemExit(
            "ERROR: canonical row count changed unexpectedly. "
            f"Expected {EXPECTED_CANONICAL_ROWS}, got {after}."
        )

    test_df = all_df[
        all_df["_split_norm"].isin(["test", "test_assumed"])
    ].copy()

    print("\nFinal held-out/test-assumed rows:", len(test_df))

    if len(test_df) != EXPECTED_FINAL_TEST_ROWS:
        raise SystemExit(
            "ERROR: final test row count is wrong. "
            f"Expected {EXPECTED_FINAL_TEST_ROWS}, got {len(test_df)}."
        )

    binary_df = test_df[
        test_df["_task_norm"].isin(BINARY_TASKS)
    ].copy()

    for col in [
        auprc_col,
        auroc_col,
        brier_col,
        ece_col,
        mae_col,
    ]:
        if col is not None:
            binary_df[col] = pd.to_numeric(
                binary_df[col],
                errors="coerce",
            )

    group = [
        "_model_id",
        "_model_name_canonical",
        "_family_num",
        "_family_name",
        "_mode_norm",
    ]

    agg_spec = {
        "mean_binary_auprc": (auprc_col, "mean"),
        "n_binary_tasks": ("_task_norm", "nunique"),
    }

    if auroc_col:
        agg_spec["mean_binary_auroc"] = (auroc_col, "mean")

    if brier_col:
        agg_spec["mean_brier"] = (brier_col, "mean")

    if ece_col:
        agg_spec["mean_ece"] = (ece_col, "mean")

    rank = (
        binary_df
        .groupby(group, dropna=False)
        .agg(**agg_spec)
        .reset_index()
    )

    if mae_col is not None:
        los = test_df[
            test_df["_task_norm"].eq("los_hours_24h")
        ].copy()

        los[mae_col] = pd.to_numeric(
            los[mae_col],
            errors="coerce",
        )

        los_summary = (
            los.groupby(group, dropna=False)[mae_col]
            .mean()
            .reset_index()
            .rename(columns={mae_col: "los_mae"})
        )

        rank = rank.merge(
            los_summary,
            on=group,
            how="left",
        )
    else:
        rank["los_mae"] = np.nan

    rank = rank[
        rank["n_binary_tasks"] == 4
    ].copy()

    rank["model_id"] = (
        rank["_model_id"]
        .astype(str)
        .str.zfill(2)
    )

    rank["model_name"] = rank["_model_name_canonical"]
    rank["family"] = rank["_family_name"]
    rank["mode"] = rank["_mode_norm"]

    rank = rank.sort_values(
        ["mean_binary_auprc", "mean_binary_auroc"],
        ascending=[False, False],
    )

    present_ids = set(
        rank["model_id"]
        .dropna()
        .astype(str)
    )

    missing_ids = [
        mid
        for mid in MODEL_NAMES
        if mid not in present_ids
    ]

    print("\n" + "=" * 110)
    print("FINAL 33-MODEL COVERAGE")
    print("=" * 110)

    print("Expected:", len(MODEL_NAMES))
    print("Found:   ", len(present_ids))

    if missing_ids:
        print("MISSING:", ", ".join(missing_ids))
        raise SystemExit(
            "STOP: final 33-model coverage is incomplete."
        )

    print("ALL 33 MODELS FOUND ✅")

    if len(rank) != EXPECTED_RANKING_ROWS:
        raise SystemExit(
            "ERROR: expected 66 model/mode ranking rows "
            f"(33 models × 2 modes), got {len(rank)}."
        )

    for mode_name in ["fixed", "tuned"]:
        sub = rank[
            rank["mode"].eq(mode_name)
        ].copy()

        if len(sub) != 33:
            raise SystemExit(
                f"ERROR: {mode_name} ranking contains "
                f"{len(sub)} models instead of 33."
            )

        print("\n" + "=" * 110)
        print(f"{mode_name.upper()} TEST RANKING")
        print("=" * 110)

        cols = [
            "model_id",
            "model_name",
            "family",
            "mean_binary_auprc",
            "mean_binary_auroc",
            "mean_brier",
            "mean_ece",
            "los_mae",
        ]

        print(
            sub[cols]
            .head(33)
            .to_string(index=False)
        )

    tuned = rank[
        rank["mode"].eq("tuned")
    ].copy()

    print("\n" + "=" * 110)
    print("FINAL TUNED TEST RANKING — 33 MODELS")
    print("=" * 110)

    print(
        tuned[
            [
                "model_id",
                "model_name",
                "family",
                "mean_binary_auprc",
                "mean_binary_auroc",
                "mean_brier",
                "mean_ece",
                "los_mae",
            ]
        ].to_string(index=False)
    )

    pivot = rank.pivot_table(
        index=[
            "model_id",
            "model_name",
            "family",
        ],
        columns="mode",
        values=[
            "mean_binary_auprc",
            "mean_binary_auroc",
            "mean_brier",
            "mean_ece",
            "los_mae",
        ],
        aggfunc="first",
    )

    pivot.columns = [
        "_".join(
            str(x)
            for x in col
            if str(x) not in ["", "nan"]
        )
        for col in pivot.columns
    ]

    pivot = pivot.reset_index()

    if (
        "mean_binary_auprc_fixed" in pivot.columns
        and
        "mean_binary_auprc_tuned" in pivot.columns
    ):
        pivot["auprc_delta_tuned_minus_fixed"] = (
            pivot["mean_binary_auprc_tuned"]
            - pivot["mean_binary_auprc_fixed"]
        )

    print("\n" + "=" * 110)
    print("TUNED MINUS FIXED")
    print("=" * 110)

    show = [
        "model_id",
        "model_name",
        "family",
        "mean_binary_auprc_fixed",
        "mean_binary_auprc_tuned",
        "auprc_delta_tuned_minus_fixed",
    ]

    show = [
        c for c in show
        if c in pivot.columns
    ]

    print(
        pivot
        .sort_values(
            "auprc_delta_tuned_minus_fixed",
            ascending=False,
        )[show]
        .to_string(index=False)
    )

    structured = tuned[
        tuned["family"].isin(
            [FAMILY_NAMES[i] for i in range(2, 7)]
        )
    ].copy()

    print("\n" + "=" * 110)
    print("TOP STRUCTURED-ONLY — FAMILIES 2–6")
    print("=" * 110)

    print(
        structured[
            [
                "model_id",
                "model_name",
                "family",
                "mean_binary_auprc",
                "mean_binary_auroc",
                "mean_brier",
                "mean_ece",
                "los_mae",
            ]
        ]
        .head(15)
        .to_string(index=False)
    )

    fam7 = tuned[
        tuned["family"].eq(FAMILY_NAMES[7])
    ].copy()

    print("\n" + "=" * 110)
    print("FAMILY 7 — MULTIMODAL")
    print("=" * 110)

    print(
        fam7[
            [
                "model_id",
                "model_name",
                "mean_binary_auprc",
                "mean_binary_auroc",
                "mean_brier",
                "mean_ece",
                "los_mae",
            ]
        ].to_string(index=False)
    )

    fam1 = tuned[
        tuned["family"].eq(FAMILY_NAMES[1])
    ].copy()

    print("\n" + "=" * 110)
    print("FAMILY 1 — TABULAR")
    print("=" * 110)

    print(
        fam1[
            [
                "model_id",
                "model_name",
                "mean_binary_auprc",
                "mean_binary_auroc",
                "mean_brier",
                "mean_ece",
                "los_mae",
            ]
        ].to_string(index=False)
    )

    family_summary = (
        tuned
        .groupby("family", dropna=False)
        .agg(
            family_mean_auprc=("mean_binary_auprc", "mean"),
            family_median_auprc=("mean_binary_auprc", "median"),
            best_model_auprc=("mean_binary_auprc", "max"),
            family_mean_auroc=("mean_binary_auroc", "mean"),
            family_mean_brier=("mean_brier", "mean"),
            family_mean_ece=("mean_ece", "mean"),
            family_mean_los_mae=("los_mae", "mean"),
        )
        .reset_index()
        .sort_values(
            "best_model_auprc",
            ascending=False,
        )
    )

    print("\n" + "=" * 110)
    print("FAMILY-LEVEL TUNED SUMMARY")
    print("=" * 110)

    print(
        family_summary.to_string(index=False)
    )

    print("\n" + "=" * 110)
    print("TASK-LEVEL WINNERS — TUNED TEST AUPRC")
    print("=" * 110)

    task_tuned = test_df[
        test_df["_mode_norm"].eq("tuned")
        &
        test_df["_task_norm"].isin(BINARY_TASKS)
    ].copy()

    task_tuned[auprc_col] = pd.to_numeric(
        task_tuned[auprc_col],
        errors="coerce",
    )

    for task in sorted(BINARY_TASKS):
        rows = (
            task_tuned[
                task_tuned["_task_norm"].eq(task)
            ]
            .dropna(subset=[auprc_col])
            .sort_values(
                auprc_col,
                ascending=False,
            )
        )

        if rows.empty:
            print(task, "NO DATA")
            continue

        r = rows.iloc[0]

        print(
            f"{task:28s} -> "
            f"{str(r['_model_id']).zfill(2)} "
            f"{r['_model_name_canonical']} "
            f"AUPRC={r[auprc_col]:.6f}"
        )

    print("\n" + "=" * 110)
    print("MODEL 29 GPU/BF16 PROVENANCE CHECK")
    print("=" * 110)

    meta_files = [
        p
        for p in ROOT.rglob("metadata.json")
        if (
            "29_Multimodal_Progression" in str(p)
            and "tuned" in str(p)
            and "final_selected_configuration" in str(p)
            and "cpu_audit" not in str(p)
        )
    ]

    if not meta_files:
        print("ERROR: Model 29 final tuned metadata not found.")
    else:
        p = sorted(meta_files)[0]
        obj = json.loads(
            p.read_text(
                encoding="utf-8"
            )
        )

        print("Path:", p)
        print("device:", obj.get("device"))
        print("gpu_name:", obj.get("gpu_name"))
        print("amp_mode:", obj.get("amp_mode"))
        print(
            "validation_selection_score:",
            obj.get("validation_selection_score"),
        )
        print("best_epoch:", obj.get("best_epoch"))
        print(
            "n_trials_completed:",
            obj.get("n_optuna_trials_completed"),
        )
        print(
            "n_trials_target:",
            obj.get("n_optuna_trials_target"),
        )

        model29_pass = (
            obj.get("device") == "cuda"
            and obj.get("gpu_name") == "AMD Instinct MI210"
            and obj.get("amp_mode") == "BF16 autocast"
            and obj.get("n_optuna_trials_completed") == 30
            and obj.get("n_optuna_trials_target") == 30
            and obj.get("test_used_for_selection") is False
        )

        print(
            "MODEL 29 GPU/BF16 PROVENANCE:",
            "PASS ✅" if model29_pass else "FAIL ❌",
        )

    print("\n" + "=" * 110)
    print("TUNED CONFIGURATION PROVENANCE")
    print("=" * 110)

    for mid in MODEL_NAMES:
        if mid in {"00", "01", "02", "03", "04", "05"}:
            continue

        matches = [
            p
            for p in ROOT.rglob("metadata.json")
            if (
                re.search(
                    rf"(?:^|/){mid}_[^/]+/",
                    str(p).replace("\\", "/"),
                )
                and "tuned" in str(p)
                and "final_selected_configuration" in str(p)
                and "cpu_audit" not in str(p)
            )
        ]

        if not matches:
            print(
                mid,
                MODEL_NAMES[mid],
                "METADATA NOT FOUND",
            )
            continue

        p = sorted(matches)[0]

        try:
            obj = json.loads(
                p.read_text(
                    encoding="utf-8"
                )
            )

            print(
                f"{mid} | "
                f"{MODEL_NAMES[mid]:30s} | "
                f"device={obj.get('device')} | "
                f"gpu={obj.get('gpu_name')} | "
                f"amp={obj.get('amp_mode')} | "
                f"trials={obj.get('n_optuna_trials_completed')}/"
                f"{obj.get('n_optuna_trials_target')}"
            )

        except Exception as e:
            print(mid, "ERROR:", e)

    # Save final artifacts.
    rank.to_csv(
        REPORT_DIR / "FINAL_33MODEL_all_modes_test_ranking.csv",
        index=False,
    )

    tuned.to_csv(
        REPORT_DIR / "FINAL_33MODEL_tuned_test_ranking.csv",
        index=False,
    )

    pivot.to_csv(
        REPORT_DIR / "FINAL_33MODEL_fixed_vs_tuned.csv",
        index=False,
    )

    structured.to_csv(
        REPORT_DIR / "FINAL_structured_only_F2_F6.csv",
        index=False,
    )

    fam7.to_csv(
        REPORT_DIR / "FINAL_family7_multimodal.csv",
        index=False,
    )

    fam1.to_csv(
        REPORT_DIR / "FINAL_family1_tabular.csv",
        index=False,
    )

    family_summary.to_csv(
        REPORT_DIR / "FINAL_family_summary.csv",
        index=False,
    )

    print("\n" + "=" * 110)
    print("AUDIT 1 COMPLETE")
    print("=" * 110)
    print("Canonical statistics source files were used directly.")
    print("Family 1 statistics were NOT recalculated by this script.")
    print(
        "Family 1 stats source:",
        FAMILY_FILES[1],
    )
    print(
        "Family 1 statistical calculation remains external:",
        "evaluate_family01_stats.py",
    )
    print("Family 2 no-split records were treated as test_assumed.")
    print("Final model count:", len(present_ids))
    print("Final test rows:", len(test_df))
    print("Saved to:", REPORT_DIR)
    print("=" * 110)


if __name__ == "__main__":
    main()
