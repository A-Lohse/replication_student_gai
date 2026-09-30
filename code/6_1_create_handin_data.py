#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compute Writing Quality sub-indices + overall WQ index for the ENGLISH dataset.

Changes:
- Uses `dale_chall` instead of `lix` in the readability block.
- Works off a single metadata table (`english_metadata_clean.parquet`).
- Constructs `clause_count_per_sentence` and uses it in the syntactic index
  instead of raw `clause_count`.
- Removes observations beyond ±5 SD on any index ingredient before
  standardization and index construction.
- Standardizes underlying metrics before constructing subindices.
- Standardizes the three subindices separately before constructing WQ_index,
  matching the Danish-data procedure.
- Applies a final ±5 SD outlier removal on constructed indices before
  min-max normalization.
- Keeps steps defensive/optional if certain columns are not present
  (e.g., year, timestamp).

Outputs the pre-similarity analysis Parquet with merged metrics, subindices,
WQ_index, normalized variants, and LLM rates per 100 words:
common_per_100, rare_per_100, llm_per_100.
"""

import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# =============================
# ======== CONFIG =============
# =============================

DATA_PATH = Path("/work/data_assigments/DATA/")
META_PARQUET = "english_metadata_clean.parquet"
WQ_JSONL = "bid2WritingQualityMeasuresEnglish.jsonl"
LLM_JSONL = "bid2LLMWordMeasuresEnglish.jsonl"
OUTPUT_PARQUET = "english_handin_analysis_pre_similarity.parquet"
SAMPLE_FLOW_OUTPUT = DATA_PATH / "sample_flow.csv"

BID_COL = "bid"
DATETIME_COL = "AfleveringsTidspunkt"  # if not present, semester labels are skipped
YEAR_COL = "year"  # optional, used for year_norm if present

# Metrics columns
READABILITY_VARS: List[str] = ["ARI", "flesch_kincaid", "dale_chall"]
LEXICAL_VARS: List[str] = ["MTLD", "CTTR"]

# Use clause count per sentence, derived below, rather than raw clause_count.
SYNTACTIC_VARS: List[str] = ["max_tree_depth", "clause_count_per_sentence"]

EXTRA_VARS: List[str] = ["num_words", "num_chars", "num_sentences"]

# Raw variables needed to construct derived index variables.
DERIVED_SOURCE_VARS: List[str] = ["clause_count", "num_sentences"]

INDEX_VARS: List[str] = READABILITY_VARS + LEXICAL_VARS + SYNTACTIC_VARS

# Text outcomes required in the pre-similarity analysis sample.
# Grade is retained in the data but is not used to restrict the sample here.
ANALYSIS_OUTCOME_VARS: List[str] = [
    "rare_per_100",
    "WQ_index_norm",
    "readability_index_norm",
    "lexical_index_norm",
    "syntactic_index_norm",
]
REQUIRED_ANALYSIS_VARS: List[str] = ANALYSIS_OUTCOME_VARS

# Variables to load from the precomputed WQ metrics file.
# Includes final index variables and raw ingredients needed for derived variables.
SELECTED_VARS: List[str] = (
    READABILITY_VARS
    + LEXICAL_VARS
    + ["max_tree_depth"]
    + DERIVED_SOURCE_VARS
    + EXTRA_VARS
)

MIN_WORDS = 100
OUTLIER_SD = 5.0

# =============================
# ======== HELPERS ============
# =============================

def log(msg: str) -> None:
    print(msg, flush=True)


sample_flow_records = []


def append_sample_flow(stage, data, before_n=None):
    """Store one script-6 sample-construction checkpoint for the SI flow figure."""
    record = {
        "script": "6_1_create_handin_data",
        "stage": stage,
        "n_submissions": len(data),
        "n_removed": (before_n - len(data)) if before_n is not None else np.nan,
        "n_students": (
            data["studerendeid"].nunique(dropna=True)
            if "studerendeid" in data.columns
            else np.nan
        ),
        "n_courses": (
            data["coursecode_clean"].nunique(dropna=True)
            if "coursecode_clean" in data.columns
            else np.nan
        ),
        "n_exams": (
            data["exam_code"].nunique(dropna=True)
            if "exam_code" in data.columns
            else np.nan
        ),
        "n_pre": (
            int((data["after"] == 0).sum())
            if "after" in data.columns
            else np.nan
        ),
        "n_post": (
            int((data["after"] == 1).sum())
            if "after" in data.columns
            else np.nan
        ),
    }
    sample_flow_records.append(record)


def save_sample_flow():
    """Append script-6 checkpoints to the cumulative sample-flow CSV."""
    current = pd.DataFrame(sample_flow_records)

    if SAMPLE_FLOW_OUTPUT.exists():
        existing = pd.read_csv(SAMPLE_FLOW_OUTPUT)
        if "script" in existing.columns:
            existing = existing.loc[
                existing["script"] != "6_1_create_handin_data"
            ].copy()
        current = pd.concat([existing, current], ignore_index=True)

    current.to_csv(SAMPLE_FLOW_OUTPUT, index=False)
    log(f"📊 Updated sample-flow file: {SAMPLE_FLOW_OUTPUT}")


def load_jsonl_dict(path: Path) -> dict:
    out = {}

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            out.update(obj)

    return out


def load_LLM_dict_to_df(path: Path) -> pd.DataFrame:
    data = {}

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)

            # Drop heavy keys if not needed.
            obj.pop("rare_freqs", None)
            obj.pop("common_freqs", None)

            # Append each value into a list per key.
            for k, v in obj.items():
                if k not in data:
                    data[k] = []
                data[k].append(v)

    return pd.DataFrame(data)


def remove_outliers_sd(series: pd.Series, n_sd: float = 5.0) -> pd.Series:
    """
    Remove values more than n_sd standard deviations from the mean.

    Values outside mean ± n_sd * SD are set to NaN.
    Rows containing these NaNs are dropped before index construction.
    """
    mean = series.mean(skipna=True)
    sd = series.std(skipna=True)

    if pd.isna(mean) or pd.isna(sd) or sd == 0:
        return series

    lower = mean - n_sd * sd
    upper = mean + n_sd * sd

    return series.where(series.between(lower, upper))


def get_semester_info(dt: pd.Series) -> pd.DataFrame:
    """Map timestamps to (term, term_year, semester_label)."""
    yr = dt.dt.year
    mo = dt.dt.month

    term = pd.Series("Fall", index=dt.index)
    term[(mo >= 2) & (mo <= 8)] = "Spring"   # Feb–Aug
    term[mo == 1] = "Fall"                   # Jan belongs to Fall of previous year

    term_year = yr.where(mo != 1, yr - 1)
    semester_label = term + " " + term_year.astype(str)

    return pd.DataFrame(
        {
            "term": term,
            "term_year": term_year,
            "semester_label": semester_label,
        },
        index=dt.index,
    )


def minmax_norm(s: pd.Series) -> pd.Series:
    mn, mx = s.min(), s.max()

    if pd.notnull(mn) and pd.notnull(mx) and mx != mn:
        return (s - mn) / (mx - mn)

    return s


# =============================
# =========== MAIN ============
# =============================

if __name__ == "__main__":
    log("✅ Libraries imported successfully.")
    log(f"📂 Loading data from: {DATA_PATH}")

    # -----------------------------
    # Base metadata
    # -----------------------------
    meta = pd.read_parquet(DATA_PATH / META_PARQUET)
    log(f"📄 Loaded english metadata: {len(meta)} rows")
    append_sample_flow("Input to text-measure construction", meta)

    # -----------------------------
    # Precomputed WQ metrics
    # -----------------------------
    WQ_dict = load_jsonl_dict(DATA_PATH / WQ_JSONL)
    log(f"✏️ Loaded writing quality measures for {len(WQ_dict)} assignments.")

    metrics_df = pd.DataFrame.from_dict(WQ_dict, orient="index").reset_index()
    metrics_df = metrics_df.rename(columns={"index": BID_COL})

    # Retain only expected metric columns that actually exist.
    # Remove duplicates while preserving order.
    keep_cols = [BID_COL] + [c for c in SELECTED_VARS if c in metrics_df.columns]
    keep_cols = list(dict.fromkeys(keep_cols))
    metrics_df = metrics_df[keep_cols]

    # -----------------------------
    # Load LLM aggregate counts
    # -----------------------------
    llm_df = load_LLM_dict_to_df(DATA_PATH / LLM_JSONL)
    log(f"✏️ Loaded LLM word count measures for {llm_df.shape[0]} assignments.")

    llm_keep = [BID_COL, "common_count", "rare_count", "common_unique", "rare_unique"]
    llm_keep = [c for c in llm_keep if c in llm_df.columns]
    llm_df = llm_df[llm_keep]

    if {"common_count", "rare_count"}.issubset(llm_df.columns):
        llm_df["llm_words"] = llm_df["common_count"] + llm_df["rare_count"]
    else:
        llm_df["llm_words"] = np.nan
        log("⚠️ Could not construct llm_words; missing common_count and/or rare_count.")

    if {"common_unique", "rare_unique"}.issubset(llm_df.columns):
        llm_df["llm_unique"] = llm_df["common_unique"] + llm_df["rare_unique"]
    else:
        llm_df["llm_unique"] = np.nan
        log("⚠️ Could not construct llm_unique; missing common_unique and/or rare_unique.")

    # -----------------------------
    # Merge metadata + WQ metrics
    # -----------------------------
    df = meta.merge(metrics_df, on=BID_COL, how="left")
    log("🧠 Merged metadata with writing quality metrics.")

    # -----------------------------
    # Derived syntactic measure
    # -----------------------------
    # Clause count per sentence is used in the syntactic index instead of raw clause_count.
    if {"clause_count", "num_sentences"}.issubset(df.columns):
        df["clause_count_per_sentence"] = np.where(
            df["num_sentences"] > 0,
            df["clause_count"] / df["num_sentences"],
            np.nan,
        ).astype(float)

        log("🧩 Constructed clause_count_per_sentence.")
    else:
        missing = {"clause_count", "num_sentences"} - set(df.columns)
        log(f"⚠️ Could not construct clause_count_per_sentence; missing columns: {missing}")

    # -----------------------------
    # Merge LLM aggregates
    # -----------------------------
    df = df.merge(llm_df, on=BID_COL, how="left")
    log("🔗 Merged LLM marker aggregates: counts + uniques.")

    # -----------------------------
    # LLM rates per 100 words
    # -----------------------------
    if "num_words" in df.columns:
        denom = df["num_words"] / 100

        if "common_count" in df.columns:
            df["common_per_100"] = (df["common_count"] / denom).astype(float)
        else:
            df["common_per_100"] = np.nan

        if "rare_count" in df.columns:
            df["rare_per_100"] = (df["rare_count"] / denom).astype(float)
        else:
            df["rare_per_100"] = np.nan

        if "llm_words" in df.columns:
            df["llm_per_100"] = (df["llm_words"] / denom).astype(float)
        else:
            df["llm_per_100"] = np.nan

        log("📐 Constructed LLM rates per 100 words.")
    else:
        df["common_per_100"] = np.nan
        df["rare_per_100"] = np.nan
        df["llm_per_100"] = np.nan
        log("⚠️ 'num_words' not found; skipping LLM rates per 100 words.")

    # -----------------------------
    # Filter by minimum word count
    # -----------------------------
    if "num_words" in df.columns:
        before = len(df)
        df = df.loc[df["num_words"] >= MIN_WORDS].copy()
        after = len(df)

        log(
            f"🧠 Removed assignments with fewer than {MIN_WORDS} words. "
            f"Remaining: {after} (dropped {before - after})."
        )
        append_sample_flow("Keep submissions with at least 100 words", df, before_n=before)
    else:
        log("⚠️ 'num_words' not found; skipping word-count filter.")

    # -----------------------------
    # Outlier removal + z-scoring of underlying metrics
    # -----------------------------
    existing_index_vars = [c for c in INDEX_VARS if c in df.columns]

    missing_index_vars = [c for c in INDEX_VARS if c not in df.columns]
    if missing_index_vars:
        log(f"⚠️ These index variables are missing and will be skipped: {missing_index_vars}")

    log(f"📌 Underlying index variables used: {existing_index_vars}")

    # Remove observations beyond ±5 SD for any underlying index ingredient.
    log(f"🧹 Removing outliers beyond ±{OUTLIER_SD:g} SD on underlying index ingredients...")

    for var in existing_index_vars:
        df[var] = remove_outliers_sd(df[var], n_sd=OUTLIER_SD)

    # Drop rows missing any underlying index ingredient after outlier removal.
    before = len(df)

    if existing_index_vars:
        df = df.dropna(subset=existing_index_vars).copy()

    after = len(df)

    log(
        f"🧼 Removed rows with missing values or ±{OUTLIER_SD:g} SD outliers "
        f"in underlying text metrics. Remaining: {after} (dropped {before - after})."
    )
    append_sample_flow("Valid linguistic component measures", df, before_n=before)

    # Standardize underlying index ingredients.
    if existing_index_vars:
        metric_scaler = StandardScaler()

        z = pd.DataFrame(
            metric_scaler.fit_transform(df[existing_index_vars]),
            columns=existing_index_vars,
            index=df.index,
        )
    else:
        z = pd.DataFrame(index=df.index)
        log("⚠️ No index variables found; index construction will produce missing values.")

    # -----------------------------
    # Construct subindices from standardized underlying metrics
    # -----------------------------
    read_vars_present = [c for c in READABILITY_VARS if c in z.columns]
    lex_vars_present = [c for c in LEXICAL_VARS if c in z.columns]
    syn_vars_present = [c for c in SYNTACTIC_VARS if c in z.columns]

    df["readability_index"] = (
        z[read_vars_present].mean(axis=1) if read_vars_present else np.nan
    )

    df["lexical_index"] = (
        z[lex_vars_present].mean(axis=1) if lex_vars_present else np.nan
    )

    df["syntactic_index"] = (
        z[syn_vars_present].mean(axis=1) if syn_vars_present else np.nan
    )

    log("📊 Subindices constructed from standardized underlying metrics.")

    # -----------------------------
    # Standardize subindices before constructing WQ index
    # -----------------------------
    subindex_vars = ["readability_index", "lexical_index", "syntactic_index"]

    before = len(df)
    df = df.dropna(subset=subindex_vars).copy()
    after = len(df)

    log(
        f"🧼 Removed rows with missing subindices before WQ construction. "
        f"Remaining: {after} (dropped {before - after})."
    )

    subindex_scaler = StandardScaler()

    subindex_z = pd.DataFrame(
        subindex_scaler.fit_transform(df[subindex_vars]),
        columns=[col + "_z" for col in subindex_vars],
        index=df.index,
    )

    df = pd.concat([df, subindex_z], axis=1)

    # -----------------------------
    # Overall WQ index
    # -----------------------------
    # This matches the Danish procedure:
    # WQ_index is the equal-weighted average of the standardized subindices.
    df["WQ_index"] = df[
        ["readability_index_z", "lexical_index_z", "syntactic_index_z"]
    ].mean(axis=1)

    log("📊 Writing Quality index calculated from standardized subindices.")

    # -----------------------------
    # Final outlier check on constructed indices
    # -----------------------------
    constructed_index_vars = [
        "WQ_index",
        "readability_index",
        "lexical_index",
        "syntactic_index",
    ]

    before = len(df)

    for var in constructed_index_vars:
        if var in df.columns:
            df[var] = remove_outliers_sd(df[var], n_sd=OUTLIER_SD)

    df = df.dropna(subset=constructed_index_vars).copy()

    after = len(df)

    log(
        f"🧼 Final ±{OUTLIER_SD:g} SD outlier removal on constructed indices. "
        f"Remaining: {after} (dropped {before - after})."
    )
    append_sample_flow("Valid linguistic complexity indices", df, before_n=before)

    # -----------------------------
    # Normalize indices to [0, 1]
    # -----------------------------
    for col in constructed_index_vars:
        if col in df.columns:
            df[col + "_norm"] = minmax_norm(df[col])

    log("📏 Normalized WQ index and subindices to [0, 1].")

    # -----------------------------
    # Validate analysis outcomes
    # -----------------------------
    missing_required = [c for c in REQUIRED_ANALYSIS_VARS if c not in df.columns]
    if missing_required:
        raise KeyError(
            "Required analysis variables are missing: "
            f"{missing_required}"
        )

    # Ensure analysis outcomes are numeric. Non-numeric values become missing.
    for col in REQUIRED_ANALYSIS_VARS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    log("📋 Missing values in required GAI/linguistic outcomes before complete-case filtering:")
    for col in REQUIRED_ANALYSIS_VARS:
        n_missing = int(df[col].isna().sum())
        pct_missing = 100 * n_missing / len(df) if len(df) else np.nan
        log(f"   {col}: {n_missing:,} ({pct_missing:.2f}%)")

    before = len(df)
    df = df.dropna(subset=REQUIRED_ANALYSIS_VARS).copy()
    after = len(df)

    log(
        "🧼 Restricted to complete cases on GAI and linguistic outcomes. "
        f"Remaining: {after} (dropped {before - after})."
    )
    append_sample_flow("Complete GAI and linguistic outcomes", df, before_n=before)

    # -----------------------------
    # Convenience columns
    # -----------------------------
    if YEAR_COL in df.columns:
        df["year_norm"] = df[YEAR_COL] - df[YEAR_COL].min()

    if "num_words" in df.columns:
        df["word_count_percent"] = df["num_words"].rank(pct=True)
        df["log_num_words"] = np.log(df["num_words"])

    # -----------------------------
    # Semester info if timestamp present
    # -----------------------------
    if DATETIME_COL in df.columns:
        df[DATETIME_COL] = pd.to_datetime(df[DATETIME_COL], errors="coerce")
        sem = get_semester_info(df[DATETIME_COL])
        df = pd.concat([df, sem], axis=1)

    # -----------------------------
    # Save
    # -----------------------------
    save_sample_flow()

    out_path = DATA_PATH / OUTPUT_PARQUET
    df.to_parquet(out_path)

    log(f"💾 Saved cleaned & enriched ENGLISH indices to {out_path}")