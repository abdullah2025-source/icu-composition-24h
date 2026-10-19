#!/usr/bin/env python3
"""
FYP server provenance extractor, v2 (targeted).

Answers the open items of the thesis workbook (Training_Protocol, yellow cells):

  A. Search ranges of every family (incl. 01 and 03), from the FINAL scripts
  B. How Models 00-05 turn the [N,24,110] tensor into tabular features + their LOS objective
  C. Selected hyperparameters, number of Optuna trials, best trial number per model
  D. Identical tuned training settings across models (TFT / Model 32, StageNet / PatchTST)
  E. Per-task choices of A2 / A3 / Z1 / Z2 / Z3 / Z4
  F. Bootstrap resampling unit of the statistical evaluation
  G. A1 event counts (validated vs chart / lab candidates)
  H. Software versions

Rules it follows:
  * Reads code, JSON configs and small aggregate tables only.
  * NEVER opens anything under the raw data folder or the Python environment.
  * NEVER copies rows from a CSV that has stay_id / subject_id / hadm_id columns
    (patient-level MIMIC data must not leave the server).
  * Writes evidence with source path + line number. It does not guess values.

Run on the server with the project Python:

    cd /home/abdullah2025/fyp_project
    ./mamba_env/bin/python provenance_extract_v2.py

Outputs go to <root>/provenance_audit_v2/ and a zip next to it.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import platform
import re
import sys
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

DEFAULT_ROOT = "/home/abdullah2025/fyp_project"
OUT_NAME = "provenance_audit_v2"

# Never descend into these (environment, raw data, caches, own output).
EXCLUDE_DIRS = {
    ".git", ".cache", "__pycache__", ".ipynb_checkpoints", "node_modules", "site-packages",
    "mamba_env", "env", "venv", ".venv", "conda", "miniconda3", "data", "mimic_iv", "eicu",
    "checkpoints", "wandb", "provenance_audit", OUT_NAME,
}
ID_COLUMNS = {"stay_id", "subject_id", "hadm_id", "patientunitstayid", "patienthealthsystemstayid", "uniquepid"}
MAX_PY_BYTES = 3 * 1024 * 1024
MAX_JSON_BYTES = 20 * 1024 * 1024
MAX_SMALL_CSV_BYTES = 300 * 1024
MAX_SMALL_CSV_ROWS = 2000
CELL_LIMIT = 6000

SKIPPED: list[dict[str, Any]] = []


# ----------------------------------------------------------------------------- helpers
def rel(p: Path, root: Path) -> str:
    try:
        return str(p.relative_to(root))
    except ValueError:
        return str(p)


def walk(root: Path) -> Iterable[Path]:
    for base, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith(OUT_NAME))
        for fn in sorted(files):
            yield Path(base) / fn


def read_text(p: Path, limit: int, root: Path, why: str) -> str | None:
    try:
        size = p.stat().st_size
        if size > limit:
            SKIPPED.append({"relative_path": rel(p, root), "size_bytes": size, "reason": f"too large for {why}"})
            return None
        return p.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        SKIPPED.append({"relative_path": rel(p, root), "size_bytes": "", "reason": f"read error: {e}"})
        return None


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = []
        for r in rows:
            for k in r:
                if k not in fields:
                    fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields or ["note"], extrasaction="ignore")
        w.writeheader()
        if not rows:
            w.writerow({(fields or ["note"])[0]: "NO EVIDENCE FOUND"})
        for r in rows:
            w.writerow({k: clip(r.get(k, "")) for k in fields})


def clip(v: Any) -> Any:
    s = v if isinstance(v, str) else None
    if s is not None and len(s) > CELL_LIMIT:
        return s[:CELL_LIMIT] + f" ...<truncated {len(s) - CELL_LIMIT} chars>"
    return v


def flatten(obj: Any, prefix: str = "", max_list: int = 200) -> Iterable[tuple[str, Any]]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from flatten(v, f"{prefix}.{k}" if prefix else str(k), max_list)
    elif isinstance(obj, list):
        if len(obj) <= max_list:
            for i, v in enumerate(obj):
                yield from flatten(v, f"{prefix}[{i}]", max_list)
        else:
            yield prefix, f"<list of {len(obj)} items, not expanded>"
    else:
        yield prefix, obj


def load_json(p: Path, root: Path) -> Any | None:
    t = read_text(p, MAX_JSON_BYTES, root, "JSON")
    if t is None:
        return None
    try:
        return json.loads(t)
    except Exception as e:  # noqa: BLE001
        SKIPPED.append({"relative_path": rel(p, root), "size_bytes": len(t), "reason": f"JSON parse error: {e}"})
        return None


def model_dir_of(p: Path) -> str:
    """First path part that looks like '06_GRU'; else the parent folder name."""
    for part in p.parts:
        if re.match(r"^\d{2}_[A-Za-z]", part):
            return part
    return p.parent.name


def stage_of(p: Path) -> str:
    parts = {x.lower() for x in p.parts}
    for s in ("tuned", "fixed"):
        if s in parts:
            return s
    return ""


def small_safe_csv(p: Path, root: Path) -> tuple[list[str], list[dict[str, str]]] | None:
    """Rows of a small aggregate CSV. Refuses files with patient identifiers."""
    try:
        size = p.stat().st_size
        with p.open("r", newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            header = [h or "" for h in (reader.fieldnames or [])]
            if {h.strip().lower() for h in header} & ID_COLUMNS:
                SKIPPED.append({"relative_path": rel(p, root), "size_bytes": size,
                                "reason": "has patient identifier columns: listed only, rows not copied"})
                return None
            if size > MAX_SMALL_CSV_BYTES:
                SKIPPED.append({"relative_path": rel(p, root), "size_bytes": size, "reason": "CSV too large: listed only"})
                return None
            rows = []
            for i, r in enumerate(reader):
                if i >= MAX_SMALL_CSV_ROWS:
                    break
                rows.append(r)
            return header, rows
    except OSError:
        return None


# ----------------------------------------------------------------------------- A. search spaces (AST)
class CodeVisitor(ast.NodeVisitor):
    """Collects suggest_* calls with their full arguments and the if-branch they sit in."""

    def __init__(self, text: str, path: str):
        self.text, self.path = text, path
        self.conds: list[str] = []
        self.funcs: list[str] = []
        self.assigns: list[dict[str, str]] = [{}]
        self.suggest: list[dict[str, Any]] = []
        self.argparse: list[dict[str, Any]] = []

    def _func(self, node):
        self.funcs.append(node.name)
        local: dict[str, str] = {}
        for n in ast.walk(node):
            if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
                local.setdefault(n.targets[0].id, ast.unparse(n.value))
        self.assigns.append(local)
        self.generic_visit(node)
        self.assigns.pop()
        self.funcs.pop()

    visit_FunctionDef = _func
    visit_AsyncFunctionDef = _func

    def visit_If(self, node: ast.If):
        test = ast.unparse(node.test)
        self.visit(node.test)
        self.conds.append(test)
        for n in node.body:
            self.visit(n)
        self.conds.pop()
        self.conds.append(f"NOT({test})")
        for n in node.orelse:
            self.visit(n)
        self.conds.pop()

    def visit_Call(self, node: ast.Call):
        f = node.func
        if isinstance(f, ast.Attribute) and f.attr.startswith("suggest_"):
            args = [ast.unparse(a) for a in node.args]
            kwargs = {k.arg or "**": ast.unparse(k.value) for k in node.keywords}
            name = args[0].strip("'\"") if args else kwargs.get("name", "").strip("'\"")
            resolved = []
            for a in node.args[1:]:
                if isinstance(a, ast.Name) and a.id in self.assigns[-1]:
                    resolved.append(f"{a.id} = {self.assigns[-1][a.id]}")
            positive = [c for c in self.conds if not c.startswith("NOT(")]
            self.suggest.append({
                "relative_path": self.path, "line_number": node.lineno,
                "function": ".".join(self.funcs),
                "branch": positive[-1] if positive else "(all models in this function)",
                "full_condition_chain": " AND ".join(self.conds),
                "parameter": name, "method": f.attr,
                "range_or_choices": ", ".join(args[1:]),
                "keywords": json.dumps(kwargs) if kwargs else "",
                "variables_resolved": " | ".join(resolved),
                "full_call": ast.unparse(node),
            })
        if isinstance(f, ast.Attribute) and f.attr == "add_argument":
            self.argparse.append({"relative_path": self.path, "line_number": node.lineno, "full_call": ast.unparse(node)})
        self.generic_visit(node)


CONST_NAME = re.compile(r"^[A-Z][A-Z0-9_]{2,}$")
TAB_FUNC = re.compile(r"(tabular|summar|flatten|feature|aggregate|pool|build_x|make_x|to_2d|design_matrix)", re.I)
TAB_FILE = re.compile(r"(family[_-]?0?1\b|family[_-]?0?1[_-]|tabular|xgboost|lightgbm|random_forest|logistic)", re.I)
TAB_CONTENT = re.compile(r"(XGB|xgboost|lightgbm|LGBM|RandomForest|ExtraTrees|LogisticRegression)")
LOS_LINE = re.compile(r"(objective\s*=|['\"]objective['\"]|reg:|Regressor\(|log1p|expm1|squared_?error|absolute_?error|huber|eval_metric|"
                      r"\bRidge\b|\bElasticNet\b|SGDRegressor|los_hours|criterion\s*=)", re.I)
BOOT_FILE = re.compile(r"(statistic|bootstrap|evaluat|paired|delong|external|eicu|model_z|model_y)", re.I)
BOOT_FUNC = re.compile(r"(boot|resampl|paired|percentile|_ci\b|ci_|confidence|delong|mcnemar)", re.I)
BOOT_LINE = re.compile(r"(rng\.(integers|choice|permutation)|np\.random\.|default_rng|RandomState|resample\(|\.unique\(|"
                       r"subject_id|stay_id|groupby\(|cluster|n_boot|bootstrap_reps|replace\s*=\s*True)", re.I)


def scan_python(root: Path):
    suggest, consts, argp, tab_funcs, tab_los, boot_funcs, boot_lines, scripts = [], [], [], [], [], [], [], []
    for p in walk(root):
        if p.suffix.lower() != ".py":
            continue
        text = read_text(p, MAX_PY_BYTES, root, "Python source")
        if text is None:
            continue
        rp = rel(p, root)
        st = p.stat()
        scripts.append({"relative_path": rp, "size_bytes": st.st_size,
                        "modified_time": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
                        "sha256": hashlib.sha256(text.encode("utf-8", "replace")).hexdigest(),
                        "has_optuna_suggest": "yes" if ".suggest_" in text else "no"})
        try:
            tree = ast.parse(text)
        except SyntaxError as e:
            SKIPPED.append({"relative_path": rp, "size_bytes": st.st_size, "reason": f"Python syntax error: {e}"})
            continue
        v = CodeVisitor(text, rp)
        v.visit(tree)
        suggest += v.suggest
        argp += v.argparse
        # module-level UPPER_CASE constants (SEED, N_TRIALS, MODEL_CONFIGS, BOOTSTRAP_REPS ...)
        for node in tree.body:
            tgt = None
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                tgt, val = node.targets[0].id, node.value
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
                tgt, val = node.target.id, node.value
            if tgt and CONST_NAME.match(tgt):
                consts.append({"relative_path": rp, "line_number": node.lineno, "name": tgt, "value": ast.unparse(val)})
        lines = text.splitlines()
        is_tab = bool(TAB_FILE.search(rp)) or bool(TAB_CONTENT.search(text))
        is_boot = bool(BOOT_FILE.search(rp))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                src = ast.get_source_segment(text, node) or ""
                if is_tab and TAB_FUNC.search(node.name):
                    tab_funcs.append({"relative_path": rp, "line_number": node.lineno, "function": node.name, "source": src})
                if is_boot and BOOT_FUNC.search(node.name):
                    boot_funcs.append({"relative_path": rp, "line_number": node.lineno, "function": node.name, "source": src})
        for i, line in enumerate(lines, 1):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if is_tab and LOS_LINE.search(line):
                tab_los.append({"relative_path": rp, "line_number": i, "line": s})
            if is_boot and BOOT_LINE.search(line):
                ctx = "\n".join(f"{j}: {lines[j - 1]}" for j in range(max(1, i - 2), min(len(lines), i + 2) + 1))
                boot_lines.append({"relative_path": rp, "line_number": i, "line": s, "context": ctx})
    return suggest, consts, argp, tab_funcs, tab_los, boot_funcs, boot_lines, scripts


# ----------------------------------------------------------------------------- C/D. params, trials, duplicates
SIG_KEYS = ["grad_clip", "weight_decay", "patience", "scheduler"]
PARAM_FILES = re.compile(r"(best_params|metadata|selected_config|final_config|hparams|hyperparam)", re.I)
INFO_KEYS = re.compile(r"(n_optuna_trials|n_trials|best_trial|best_value|optuna|seed|selection_metric|validation_selection_score|"
                       r"sampler|pruner|version|torch|python|rocm|hip|cuda|device|gpu|los_.*transform|loss|objective|"
                       r"tabular|n_features|input_dim|feature|summar)", re.I)


def norm(v: Any) -> str:
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float)):
        return f"{float(v):.6g}"
    return str(v).strip().lower()


def scan_params(root: Path):
    params, info, sigs = [], [], []
    for p in walk(root):
        if p.suffix.lower() != ".json" or not PARAM_FILES.search(p.name):
            continue
        obj = load_json(p, root)
        if obj is None:
            continue
        rp, md, stg = rel(p, root), model_dir_of(p), stage_of(p)
        leaves: dict[str, Any] = {}
        for k, v in flatten(obj, max_list=50):
            leaf = re.sub(r"\[\d+\]$", "", k.split(".")[-1])
            row = {"relative_path": rp, "model_dir": md, "stage": stg, "json_key": k, "value": v}
            if "best_params" in p.name.lower() or re.search(r"(param|config|hyper)", k, re.I):
                params.append(row)
                leaves.setdefault(leaf.lower(), v)
            if INFO_KEYS.search(k):
                info.append(row)
        if all(k in leaves for k in SIG_KEYS):
            sigs.append({"relative_path": rp, "model_dir": md, "stage": stg,
                         **{k: leaves[k] for k in SIG_KEYS},
                         "batch_size": leaves.get("batch_size", ""), "lr": leaves.get("lr", leaves.get("learning_rate", "")),
                         "signature": " | ".join(norm(leaves[k]) for k in SIG_KEYS)})
    groups = defaultdict(list)
    for s in sigs:
        if s["stage"] != "fixed":          # fixed runs share settings by design
            groups[s["signature"]].append(s)
    dup = []
    for sig, items in groups.items():
        models = sorted({x["model_dir"] for x in items})
        if len(models) >= 2:
            for x in items:
                dup.append({"models_sharing_settings": ", ".join(models), **x})
    return params, info, sigs, dup


def scan_trials(root: Path):
    """Per-trial tables saved as CSV + Optuna storages (read with optuna when available)."""
    rows, storages = [], []
    for p in walk(root):
        n = p.name.lower()
        if p.suffix.lower() == ".csv" and "trial" in n:
            got = small_safe_csv(p, root)
            if got:
                for i, r in enumerate(got[1]):
                    rows.append({"relative_path": rel(p, root), "model_dir": model_dir_of(p), "row": i + 1,
                                 "values_json": json.dumps(r, ensure_ascii=False)})
        if p.suffix.lower() in {".db", ".sqlite", ".sqlite3"} or ("journal" in n and "optuna" in str(p).lower()):
            storages.append(p)
    for p in storages:
        entry = {"relative_path": rel(p, root), "model_dir": model_dir_of(p), "row": "", "values_json": ""}
        try:
            import optuna  # noqa: PLC0415
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            url = f"sqlite:///{p}"
            for summ in optuna.get_all_study_summaries(storage=url):
                study = optuna.load_study(study_name=summ.study_name, storage=url)
                for t in study.trials:
                    rows.append({"relative_path": rel(p, root), "model_dir": model_dir_of(p), "row": t.number,
                                 "values_json": json.dumps({"study": summ.study_name, "number": t.number, "state": t.state.name,
                                                            "value": t.value, **t.params}, default=str)})
        except Exception as e:  # noqa: BLE001
            entry["values_json"] = f"storage found but not read: {e}"
            rows.append(entry)
    return rows


# ----------------------------------------------------------------------------- E. ensembles
ENS_DIR = re.compile(r"(model_y|model_z|z4|ensemble|fusion_family|proposed_model|external_eicu)", re.I)


def scan_ensembles(root: Path):
    files, js, tabs = [], [], []
    for p in walk(root):
        rp = rel(p, root)
        if not ENS_DIR.search(rp):
            continue
        try:
            st = p.stat()
        except OSError:
            continue
        files.append({"relative_path": rp, "size_bytes": st.st_size,
                      "modified_time": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds")})
        ext = p.suffix.lower()
        if ext == ".json":
            obj = load_json(p, root)
            if obj is not None:
                for k, v in flatten(obj):
                    js.append({"relative_path": rp, "json_key": k, "value": v})
        elif ext in {".csv", ".tsv"} and "prediction" not in p.name.lower():
            got = small_safe_csv(p, root)
            if got:
                for i, r in enumerate(got[1]):
                    tabs.append({"relative_path": rp, "row": i + 1, "values_json": json.dumps(r, ensure_ascii=False)})
        elif ext in {".md", ".txt", ".log"} and st.st_size <= 300 * 1024:
            t = read_text(p, 300 * 1024, root, "ensemble text")
            for i, line in enumerate((t or "").splitlines(), 1):
                if re.search(r"(select|chosen|gate|weight|member|source|stream|\bK\s*=|top[_ -]?k|locked)", line, re.I):
                    tabs.append({"relative_path": rp, "row": i, "values_json": line.strip()})
    return files, js, tabs


# ----------------------------------------------------------------------------- G. event counts
EVENT_NUMS = re.compile(r"(20[, ]?332[, ]?911|15[, ]?272[, ]?985|2[, ]?819[, ]?531|18[, ]?092[, ]?516)")
EVENT_FILES = re.compile(r"^(a[0-6]_|feature_quality|invalid_measurement|labevents_reference|fio2_unit|"
                         r"v2_freeze_manifest|preprocessing_metadata)", re.I)


def scan_events(root: Path):
    rows = []
    for p in walk(root):
        rp, ext = rel(p, root), p.suffix.lower()
        if ENS_DIR.search(rp):
            continue
        if ext == ".json" and EVENT_FILES.search(p.name):
            obj = load_json(p, root)
            if obj is not None:
                for k, v in flatten(obj, max_list=60):
                    if isinstance(v, (int, float)) or re.search(r"(row|event|count|candidate|valid|reject|drop|chart|lab)", k, re.I):
                        rows.append({"relative_path": rp, "where": k, "value": v})
        elif ext == ".csv" and EVENT_FILES.search(p.name):
            got = small_safe_csv(p, root)
            if got:
                for i, r in enumerate(got[1]):
                    rows.append({"relative_path": rp, "where": f"row {i + 1}", "value": json.dumps(r, ensure_ascii=False)})
        elif ext in {".log", ".txt", ".md", ".py", ".json"}:
            try:                                   # stream: logs can be large
                with p.open("r", encoding="utf-8", errors="replace") as f:
                    for i, line in enumerate(f, 1):
                        if EVENT_NUMS.search(line):
                            rows.append({"relative_path": rp, "where": f"line {i}", "value": line.strip()[:500]})
            except OSError:
                pass
    return rows


# ----------------------------------------------------------------------------- H. software
def software_versions() -> list[dict[str, Any]]:
    rows = [{"item": "python", "value": sys.version.replace("\n", " ")},
            {"item": "python_executable", "value": sys.executable},
            {"item": "platform", "value": platform.platform()}]
    try:
        from importlib import metadata  # noqa: PLC0415
        for pkg in ["torch", "optuna", "numpy", "pandas", "scikit-learn", "scipy", "xgboost", "lightgbm",
                    "transformers", "mamba-ssm", "pyarrow"]:
            try:
                rows.append({"item": pkg, "value": metadata.version(pkg)})
            except metadata.PackageNotFoundError:
                rows.append({"item": pkg, "value": "not installed in this Python"})
    except Exception as e:  # noqa: BLE001
        rows.append({"item": "importlib.metadata", "value": f"error: {e}"})
    try:
        import torch  # noqa: PLC0415
        rows.append({"item": "torch.version.hip (ROCm)", "value": getattr(torch.version, "hip", None)})
        rows.append({"item": "torch.version.cuda", "value": getattr(torch.version, "cuda", None)})
        if torch.cuda.is_available():
            rows.append({"item": "gpu", "value": torch.cuda.get_device_name(0)})
    except Exception as e:  # noqa: BLE001
        rows.append({"item": "torch import", "value": f"not available: {e}"})
    return rows


# ----------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description="Targeted provenance extraction for the thesis workbook.")
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--out", default="")
    ap.add_argument("--no-zip", action="store_true")
    a = ap.parse_args()
    root = Path(a.root).expanduser().resolve()
    if not root.is_dir():
        print(f"ERROR: project root not found: {root}", file=sys.stderr)
        return 2
    out = Path(a.out).expanduser().resolve() if a.out else root / OUT_NAME
    out.mkdir(parents=True, exist_ok=True)
    print(f"Project root: {root}\nOutput dir  : {out}\n")

    suggest, consts, argp, tab_funcs, tab_los, boot_funcs, boot_lines, scripts = scan_python(root)
    params, info, sigs, dup = scan_params(root)
    trials = scan_trials(root)
    ens_files, ens_json, ens_tabs = scan_ensembles(root)
    events = scan_events(root)
    soft = software_versions()

    write_csv(out / "A1_search_space_suggest_calls.csv", suggest)
    write_csv(out / "A2_script_constants.csv", consts)
    write_csv(out / "A3_script_cli_arguments.csv", argp)
    write_csv(out / "A4_scripts_inventory_sha256.csv", scripts)
    write_csv(out / "B1_tabular_feature_functions.csv", tab_funcs)
    write_csv(out / "B2_tabular_los_objective_lines.csv", tab_los)
    write_csv(out / "C1_selected_params.csv", params)
    write_csv(out / "C2_run_info_trials_versions.csv", info)
    write_csv(out / "C3_optuna_trials.csv", trials)
    write_csv(out / "D1_training_settings_all_models.csv", sigs)
    write_csv(out / "D2_identical_training_settings.csv", dup)
    write_csv(out / "E1_ensemble_files.csv", ens_files)
    write_csv(out / "E2_ensemble_json.csv", ens_json)
    write_csv(out / "E3_ensemble_small_tables.csv", ens_tabs)
    write_csv(out / "F1_bootstrap_functions.csv", boot_funcs)
    write_csv(out / "F2_bootstrap_lines.csv", boot_lines)
    write_csv(out / "G1_event_count_evidence.csv", events)
    write_csv(out / "H1_software_versions_now.csv", soft)
    write_csv(out / "Z_skipped_files.csv", SKIPPED)

    fam = lambda rows, pat: sum(1 for r in rows if re.search(pat, r["relative_path"], re.I))  # noqa: E731
    summary = [
        ("A. suggest_* calls found (all scripts)", len(suggest)),
        ("A. suggest_* calls in Family 01 scripts", fam(suggest, r"(family[_-]?0?1[_-]|tabular)")),
        ("A. suggest_* calls in Family 03 scripts", fam(suggest, r"(family[_-]?0?3[_-]|tcn)")),
        ("B. tabular feature functions", len(tab_funcs)),
        ("B. tabular LOS objective lines", len(tab_los)),
        ("C. selected-parameter records", len(params)),
        ("C. Optuna trial rows", len(trials)),
        ("D. models with full training settings", len({s["model_dir"] for s in sigs})),
        ("D. models in an identical-settings group", len({d["model_dir"] for d in dup})),
        ("E. ensemble files listed", len(ens_files)),
        ("E. ensemble JSON records", len(ens_json)),
        ("E. ensemble small-table rows", len(ens_tabs)),
        ("F. bootstrap functions", len(boot_funcs)),
        ("F. bootstrap lines", len(boot_lines)),
        ("G. event-count records", len(events)),
        ("H. software version rows", len(soft)),
        ("Skipped files (see Z_skipped_files.csv)", len(SKIPPED)),
    ]
    status = lambda k, v: ("OK" if not v else "REVIEW") if k.startswith("Skipped") else ("FOUND" if v else "NOT FOUND")  # noqa: E731
    write_csv(out / "00_summary.csv", [{"item": k, "count": v, "status": status(k, v)} for k, v in summary])
    (out / "README.txt").write_text(
        "Targeted provenance extraction (v2). Code, JSON configs and small aggregate tables only.\n"
        "No rows were copied from files with stay_id / subject_id / hadm_id. Raw data and the Python environment were not scanned.\n"
        f"Root: {root}\nRun: {datetime.now().isoformat(timespec='seconds')}\n\n"
        "A1 = search ranges per model (column 'branch' = which model the range applies to)\n"
        "B1/B2 = tensor -> tabular features and LOS objective of Models 00-05\n"
        "C1-C3 = selected parameters, trial counts, best trial number, per-trial tables\n"
        "D2 = models whose tuned grad_clip / weight_decay / patience / scheduler are identical\n"
        "E1-E3 = what A2 / A3 / Z1 / Z2 / Z3 / Z4 selected\n"
        "F1/F2 = how the bootstrap resamples (rows vs subjects)\n"
        "G1 = A1 event counts\nH1 = software versions of the Python used to run this script\n", encoding="utf-8")
    if not a.no_zip:
        z = out.with_name(f"{out.name}_{datetime.now():%Y%m%d_%H%M%S}.zip")
        with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(out.iterdir()):
                if p.is_file():
                    zf.write(p, arcname=p.name)
        print(f"ZIP: {z}")
    for k, v in summary:
        print(f"  {k:48s} {v:>7}  {status(k, v)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
