#!/usr/bin/env python3
"""
MIMIC-III CareVue external validation, step 3 of 4: radiology text embeddings.

This is a port of pipeline_a_v2_a2.py (Pipeline A V2, stage A2) to MIMIC-III
NOTEEVENTS. Nothing is fitted. The same rules are applied:

    text source        radiology reports only
    link               subject_id + hadm_id
    availability time  storetime when present, otherwise charttime
    window             [intime, intime + 24h)
    multi-stay notes   latest qualifying ICU intime, then largest stay id
    cleaning           lower-case, de-identification markers, whitespace;
                       notes with <= 20 characters removed
    encoder            emilyalsentzer/Bio_ClinicalBERT, [CLS], max_length 512
    pooling            deterministic norm-softmax pooling over a stay's notes

Two things differ in MIMIC-III and are recorded in the report:

  * STORETIME is empty for radiology reports, so the charttime fallback of the
    same rule is what applies. Charttime is the time of the examination; the
    report text may have been finalised later.
  * MIMIC-III writes de-identified spans as [** ... **]; MIMIC-IV-Note writes
    them as ___. By default the MIMIC-III markers are rewritten to ___ so that
    the encoder sees the placeholder style it saw for MIMIC-IV
    (--deid remove applies the A2 cleaning regex literally instead).
  * NOTEEVENTS rows flagged ISERROR = 1 are excluded (MIMIC-IV has no such flag).

Outputs (row order = mimic3_index_24h.csv):
    mimic3_text_embeddings.npy      [N, 768] float32, zeros where no text
    mimic3_text_missing.npy         [N] float32, 1 = no text
    mimic3_text_embedding_report.json

Check the port against MIMIC-IV first (re-embeds a sample of MIMIC-IV stays and
compares them with the frozen clinical_note_embeddings.parquet):

    python build_mimic3_text_embeddings.py --verify-on-mimic4

Then build:

    python build_mimic3_text_embeddings.py \
        --data-dir external_validation_data/mimic-iii-clinical-database-1.4

No patient rows are printed.
"""
from __future__ import annotations

import argparse
import gc
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
HIDDEN = 768
MAX_LENGTH = 512
LANDMARK_HOURS = 24
MIMIC4_TEXT_SHARE_PRIMARY = 46968 / 74572      # a5_alignment_audit.json
MIMIC4_STAYS_WITH_TEXT_A0 = 55843               # a2_metadata.json
DEID = re.compile(r"\[\*\*.*?\*\*\]")


# ----------------------------------------------------------------------------
# A2 rules (verbatim logic)
# ----------------------------------------------------------------------------
def clean_note(text) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"\[\*\*.*?\*\*\]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def select_notes(notes: pd.DataFrame, stays: pd.DataFrame, audit: dict) -> pd.DataFrame:
    """notes: note_id, subject_id, hadm_id, charttime, storetime, text.  stays: stay_id, subject_id, hadm_id, intime."""
    audit["note_rows_loaded"] = int(len(notes))
    usable = notes[["subject_id", "hadm_id", "note_id", "charttime"]].notna().all(axis=1)
    notes = notes[usable].copy()
    notes["subject_id"] = notes["subject_id"].astype("int64")
    notes["hadm_id"] = notes["hadm_id"].astype("int64")
    notes["note_id"] = notes["note_id"].astype(str)
    if notes["note_id"].duplicated().any():
        raise AssertionError("duplicate note ids")
    audit["note_rows_with_usable_ids_and_charttime"] = int(len(notes))

    win = stays[["subject_id", "hadm_id", "stay_id", "intime"]].copy()
    win["landmark_end"] = win["intime"] + pd.Timedelta(hours=LANDMARK_HOURS)
    m = notes.merge(win, on=["subject_id", "hadm_id"], how="inner")
    audit["matched_note_rows_before_time_filter"] = int(len(m))
    m["availability_time"] = m["storetime"].fillna(m["charttime"])
    m["availability_source"] = np.where(m["storetime"].notna(), "storetime", "charttime")
    audit["notes_before_icu_intime"] = int((m["availability_time"] < m["intime"]).sum())
    audit["notes_at_or_after_24h_landmark"] = int((m["availability_time"] >= m["landmark_end"]).sum())
    m = m[(m["availability_time"] >= m["intime"]) & (m["availability_time"] < m["landmark_end"])].copy()
    audit["candidate_notes_in_24h_window"] = int(len(m))
    m = m.sort_values(["note_id", "intime", "stay_id"], kind="mergesort")
    audit["notes_matching_multiple_icu_stays"] = int(m.groupby("note_id", sort=False)["stay_id"].nunique().gt(1).sum())
    m = m.drop_duplicates(subset=["note_id"], keep="last").reset_index(drop=True)

    m["text_clean"] = m["text"].map(clean_note)
    audit["short_notes_20_or_fewer_characters_removed"] = int((m["text_clean"].str.len() <= 20).sum())
    m = m[m["text_clean"].str.len() > 20].copy()
    audit["usable_note_rows"] = int(len(m))
    audit["stays_with_text"] = int(m["stay_id"].nunique())
    audit["notes_using_storetime"] = int((m["availability_source"] == "storetime").sum())
    audit["notes_using_charttime_fallback"] = int((m["availability_source"] == "charttime").sum())
    return m[["note_id", "stay_id", "availability_time", "intime", "text_clean"]].reset_index(drop=True)


def norm_softmax_pooling(emb: np.ndarray) -> np.ndarray:
    scores = np.linalg.norm(emb, axis=1).astype(np.float64)
    scores -= np.max(scores)
    w = np.exp(scores)
    w /= float(w.sum())
    return (w[:, None] * emb).sum(axis=0).astype(np.float32)


def load_encoder():
    import torch
    from transformers import AutoModel, AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
        model = AutoModel.from_pretrained(MODEL_NAME, local_files_only=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: no GPU visible; encoding on CPU will be slow.")
    model = model.to(device).eval()
    assert int(model.config.hidden_size) == HIDDEN
    return tok, model, device


def encode_texts(texts: list[str], tok, model, device, batch_size: int) -> np.ndarray:
    import torch
    while batch_size >= 1:
        try:
            out = []
            with torch.inference_mode():
                for s in range(0, len(texts), batch_size):
                    enc = tok(texts[s:s + batch_size], padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
                    enc = {k: v.to(device) for k, v in enc.items()}
                    out.append(model(**enc).last_hidden_state[:, 0, :].detach().cpu().float().numpy())
                    if (s // batch_size) % 50 == 0:
                        print(f"  encoded {min(s + batch_size, len(texts)):,} / {len(texts):,}", end="\r", flush=True)
            print()
            return np.vstack(out).astype(np.float32, copy=False)
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or batch_size == 1:
                raise
            batch_size = max(1, batch_size // 2)
            gc.collect()
            torch.cuda.empty_cache()
            print(f"\n  GPU out of memory; retrying with batch size {batch_size}")
    raise RuntimeError("encoding failed")


def pool_by_stay(notes: pd.DataFrame, emb: np.ndarray) -> dict[int, np.ndarray]:
    order = np.argsort(notes["stay_id"].to_numpy(np.int64), kind="mergesort")
    ns = notes.iloc[order].reset_index(drop=True)
    es = emb[order]
    return {int(sid): norm_softmax_pooling(es[g.index.to_numpy(np.int64)]) for sid, g in ns.groupby("stay_id", sort=False)}


# ----------------------------------------------------------------------------
# Readers
# ----------------------------------------------------------------------------
def find_table(root: Path, stem: str) -> Path:
    names = {f"{stem}.csv", f"{stem}.csv.gz", f"{stem.lower()}.csv", f"{stem.lower()}.csv.gz"}
    hits = [p for p in root.rglob("*") if p.is_file() and p.name in names]
    if not hits:
        raise FileNotFoundError(f"Could not find {stem} under {root}")
    return sorted(hits, key=lambda p: len(p.parts))[0]


def read_mimic3_radiology(path: Path, hadm_ids: set[int], deid: str, audit: dict) -> pd.DataFrame:
    parts, n_read, n_rad, n_err = [], 0, 0, 0
    for chunk in pd.read_csv(path, chunksize=100_000, low_memory=False):
        chunk.columns = [c.lower() for c in chunk.columns]
        n_read += len(chunk)
        chunk = chunk[chunk["category"].astype(str).str.strip().str.lower().eq("radiology")]
        n_rad += len(chunk)
        err = pd.to_numeric(chunk["iserror"], errors="coerce").fillna(0).gt(0) if "iserror" in chunk.columns else pd.Series(False, index=chunk.index)
        n_err += int(err.sum())
        chunk = chunk[~err]
        chunk = chunk[pd.to_numeric(chunk["hadm_id"], errors="coerce").isin(hadm_ids)]
        if len(chunk):
            parts.append(chunk[["row_id", "subject_id", "hadm_id", "charttime", "storetime", "description", "text"]])
        print(f"  NOTEEVENTS: {n_read:,} rows read, {sum(len(p) for p in parts):,} kept", end="\r", flush=True)
    print()
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["row_id", "subject_id", "hadm_id", "charttime", "storetime", "description", "text"])
    audit.update({"noteevents_rows_read": n_read, "radiology_rows": n_rad, "radiology_rows_flagged_iserror_excluded": n_err})
    df = df.rename(columns={"row_id": "note_id"})
    df["charttime"] = pd.to_datetime(df["charttime"], errors="coerce")
    df["storetime"] = pd.to_datetime(df["storetime"], errors="coerce")
    audit["radiology_rows_with_storetime"] = int(df["storetime"].notna().sum())
    audit["radiology_rows_without_charttime"] = int(df["charttime"].isna().sum())
    if deid == "mimic4_placeholder":
        df["text"] = df["text"].map(lambda t: DEID.sub("___", t) if isinstance(t, str) else t)
    return df


# ----------------------------------------------------------------------------
# Verification against the frozen MIMIC-IV embeddings
# ----------------------------------------------------------------------------
def verify_on_mimic4(v2: Path, note_path: Path | None, n_stays: int, batch_size: int) -> bool:
    meta = json.loads((v2 / "a2_metadata.json").read_text())
    path = note_path or Path(meta["mimic_note_radiology_table"])
    print(f"MIMIC-IV radiology table: {path}")
    cohort = pd.read_csv(v2 / "cohort_master.csv", usecols=["subject_id", "hadm_id", "stay_id", "intime"], parse_dates=["intime"])
    hadm = set(pd.to_numeric(cohort["hadm_id"], errors="coerce").dropna().astype("int64"))
    parts = []
    for chunk in pd.read_csv(path, usecols=["note_id", "subject_id", "hadm_id", "charttime", "storetime", "text"], chunksize=200_000, low_memory=False):
        parts.append(chunk[pd.to_numeric(chunk["hadm_id"], errors="coerce").isin(hadm)])
    rad = pd.concat(parts, ignore_index=True)
    rad["charttime"] = pd.to_datetime(rad["charttime"], errors="coerce")
    rad["storetime"] = pd.to_datetime(rad["storetime"], errors="coerce")
    audit: dict = {}
    notes = select_notes(rad, cohort, audit)
    del rad
    ok_count = audit["stays_with_text"] == MIMIC4_STAYS_WITH_TEXT_A0
    print(f"Selection check: stays with text {audit['stays_with_text']:,} | frozen A2 value {MIMIC4_STAYS_WITH_TEXT_A0:,} | usable notes {audit['usable_note_rows']:,} (frozen 112,520)")

    ref = pd.read_parquet(v2 / "clinical_note_embeddings.parquet")
    rng = np.random.default_rng(20260923)
    sample = rng.choice(np.sort(notes["stay_id"].unique()), size=min(n_stays, notes["stay_id"].nunique()), replace=False)
    sub = notes[notes["stay_id"].isin(sample)].reset_index(drop=True)
    tok, model, device = load_encoder()
    pooled = pool_by_stay(sub, encode_texts(sub["text_clean"].tolist(), tok, model, device, batch_size))
    ref = ref[ref["stay_id"].isin(sample)]
    diffs = [float(np.max(np.abs(np.asarray(v, dtype=np.float32) - pooled[int(s)]))) for s, v in zip(ref["stay_id"], ref["text_embedding"])]
    worst = max(diffs) if diffs else float("nan")
    ok_emb = len(diffs) == len(sample) and worst < 5e-3
    print(f"Embedding check on {len(diffs)} MIMIC-IV stays: max abs difference {worst:.3e}")
    ok = ok_count and ok_emb
    print("PASS: same note selection and same embeddings" if ok else "FAIL: do not use the MIMIC-III embeddings until this is resolved")
    return ok


# ----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, help="MIMIC-III folder (searched recursively for NOTEEVENTS)")
    ap.add_argument("--m3-dir", type=Path, default=Path("outputs/external_mimic3_tempC"), help="folder with mimic3_index_24h.csv; outputs are written here")
    ap.add_argument("--v2-dir", type=Path, default=Path("outputs/pipeline_a_v2"))
    ap.add_argument("--deid", choices=["mimic4_placeholder", "remove"], default="mimic4_placeholder")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--verify-on-mimic4", action="store_true")
    ap.add_argument("--verify-stays", type=int, default=300)
    ap.add_argument("--mimic4-radiology", type=Path, default=None)
    a = ap.parse_args()

    if a.verify_on_mimic4:
        verify_on_mimic4(a.v2_dir, a.mimic4_radiology, a.verify_stays, a.batch_size)
        return
    if a.data_dir is None:
        ap.error("--data-dir is required")

    idx = pd.read_csv(a.m3_dir / "mimic3_index_24h.csv", parse_dates=["intime"])
    assert (idx["row_index"].to_numpy() == np.arange(len(idx))).all(), "index is not in tensor row order"
    stays = idx[["subject_id", "hadm_id", "icustay_id", "intime"]].rename(columns={"icustay_id": "stay_id"})
    print(f"Cohort: {len(stays):,} stays")

    audit: dict = {"deid_handling": a.deid}
    rad = read_mimic3_radiology(find_table(a.data_dir, "NOTEEVENTS"), set(stays["hadm_id"].astype("int64")), a.deid, audit)
    by_type = rad["description"].astype(str).str.strip().str.upper().str[:40]
    notes = select_notes(rad.drop(columns=["description"]), stays, audit)
    audit["hours_from_intime_median"] = round(float(((notes["availability_time"] - notes["intime"]).dt.total_seconds() / 3600).median()), 2)
    audit["notes_per_stay_with_text_median"] = float(notes.groupby("stay_id").size().median())
    audit["distinct_report_descriptions_in_cohort_admissions"] = int(by_type.nunique())
    del rad

    tok, model, device = load_encoder()
    pooled = pool_by_stay(notes, encode_texts(notes["text_clean"].tolist(), tok, model, device, a.batch_size))

    text = np.zeros((len(idx), HIDDEN), dtype=np.float32)
    missing = np.ones(len(idx), dtype=np.float32)
    row_of = pd.Series(idx["row_index"].to_numpy(), index=idx["icustay_id"].to_numpy())
    for sid, vec in pooled.items():
        r = int(row_of.loc[sid])
        text[r] = vec
        missing[r] = 0.0
    assert np.isfinite(text).all()
    np.save(a.m3_dir / "mimic3_text_embeddings.npy", text, allow_pickle=False)
    np.save(a.m3_dir / "mimic3_text_missing.npy", missing, allow_pickle=False)

    share = float((missing == 0).mean())
    audit.update({"cohort_stays": int(len(idx)), "stays_with_text_share": round(share, 4), "mimic4_primary_cohort_text_share": round(MIMIC4_TEXT_SHARE_PRIMARY, 4),
                  "encoder": MODEL_NAME, "max_length": MAX_LENGTH, "pooling": "deterministic_norm_softmax_pooling", "window": "[intime, intime + 24h)",
                  "availability_time_rule": "storetime when present, otherwise charttime", "fitted_on_mimic3": False})
    (a.m3_dir / "mimic3_text_embedding_report.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))
    print(f"\nStays with radiology text: {share * 100:.1f}%  (MIMIC-IV primary cohort: {MIMIC4_TEXT_SHARE_PRIMARY * 100:.1f}%)")
    print(f"wrote {a.m3_dir / 'mimic3_text_embeddings.npy'}  shape {text.shape}")


if __name__ == "__main__":
    main()
