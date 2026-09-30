#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Finalize the common analytical sample for the English GAI analysis.

This script:
1. Loads the pre-similarity analytical sample produced by script 6_1.
2. Merges the semantic-similarity measures produced by script 7_2.
3. Restricts to observations with a valid full-embedding similarity measure.
4. Restricts to observations with a valid numerical grade on the 0--6 scale.
5. Validates that the GAI and linguistic outcomes required upstream are complete.
6. Validates the final analysis identifiers required downstream.
7. Re-applies the final course-support requirements after the similarity and grade restrictions.
8. Merges the individual rare LLM-word frequencies and word counts required for the downstream adoption analysis.
9. Saves the complete internal analysis-ready dataset used by all downstream analyses.
10. Creates and saves a public analysis dataset by removing identifying or unnecessary metadata and replacing submission and course identifiers with anonymized IDs while preserving the analytical structure.
11. Appends the final sample-construction checkpoints to sample_flow.csv.

No linguistic measures, LLM-word measures, or similarity measures are recalculated here.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ================================================================
# CONFIG
# ================================================================

DATA_PATH = Path("/work/data_assigments/DATA/")

ANALYSIS_PATH = DATA_PATH / "english_handin_analysis_pre_similarity.parquet"
SIMILARITY_PATH = DATA_PATH / "embedding_similarity_english.parquet"
LLM_WORD_MEASURES_PATH = DATA_PATH / "bid2LLMWordMeasuresEnglish.jsonl"

OUTPUT_PATH = DATA_PATH / "english_handin_analysis_final.parquet"
PUBLIC_OUTPUT_PATH = DATA_PATH / "analysis_data_public.parquet"
SAMPLE_FLOW_OUTPUT = DATA_PATH / "sample_flow.csv"

ID_COL = "bid"
GRADE_COL = "karakter_0_6"
SIMILARITY_COL = "cosine_similarity_full"
COURSE_COL = "coursecode_clean"
PERIOD_COL = "period"
PERIOD_START_COL = "period_start"
AFTER_COL = "after"

# These should already be complete after script 6_1.
# They are validated here but are not used to make additional exclusions.
UPSTREAM_REQUIRED_OUTCOMES = [
    "rare_per_100",
    "WQ_index_norm",
    "readability_index_norm",
    "lexical_index_norm",
    "syntactic_index_norm",
]

SIMILARITY_COLS = [
    "n_course_period_handins",
    "cosine_similarity_full",
    "cosine_similarity_pca_100",
]


# ================================================================
# HELPERS
# ================================================================

sample_flow_records = []


def print_sample_stats(label, data):
    """Print the main sample-size statistics for the current dataframe."""
    print(f"\n{label}")
    print(f"  Hand-ins: {len(data):,}")

    if "exam_code" in data.columns:
        print(f"  Exams: {data['exam_code'].nunique(dropna=True):,}")
    if "coursecode_clean" in data.columns:
        print(f"  Courses: {data['coursecode_clean'].nunique(dropna=True):,}")
    if "studerendeid" in data.columns:
        print(f"  Students: {data['studerendeid'].nunique(dropna=True):,}")
    if "after" in data.columns:
        print(f"  Pre-ChatGPT:  {(data['after'] == 0).sum():,}")
        print(f"  Post-ChatGPT: {(data['after'] == 1).sum():,}")


def append_sample_flow(stage, data, before_n=None):
    """Store one script-7_3 sample-construction checkpoint for the SI flow figure."""
    record = {
        "script": "7_3_finalize_analysis_sample",
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


def report_restriction(label, before_n, data, record_flow=True):
    """Print and optionally record the number of observations removed by a restriction."""
    after_n = len(data)
    removed_n = before_n - after_n
    removed_pct = 100 * removed_n / before_n if before_n else 0

    print(f"\n🔹 {label}")
    print(f"  Before:  {before_n:,}")
    print(f"  After:   {after_n:,}")
    print(f"  Removed: {removed_n:,} ({removed_pct:.2f}%)")

    if "exam_code" in data.columns:
        print(f"  Exams remaining: {data['exam_code'].nunique(dropna=True):,}")
    if "coursecode_clean" in data.columns:
        print(f"  Courses remaining: {data['coursecode_clean'].nunique(dropna=True):,}")
    if "studerendeid" in data.columns:
        print(f"  Students remaining: {data['studerendeid'].nunique(dropna=True):,}")

    if record_flow:
        append_sample_flow(label, data, before_n=before_n)


def save_sample_flow():
    """Append script-7_3 checkpoints to the cumulative sample-flow CSV."""
    current = pd.DataFrame(sample_flow_records)

    if SAMPLE_FLOW_OUTPUT.exists():
        existing = pd.read_csv(SAMPLE_FLOW_OUTPUT)

        if "script" in existing.columns:
            existing = existing.loc[
                existing["script"] != "7_3_finalize_analysis_sample"
            ].copy()

        current = pd.concat(
            [existing, current],
            ignore_index=True,
        )

    current.to_csv(SAMPLE_FLOW_OUTPUT, index=False)
    print(f"\n📊 Updated sample-flow file: {SAMPLE_FLOW_OUTPUT}")


def make_random_ids(series, prefix, seed=42):
    """
    Replace unique values with reproducible random anonymous IDs.
    The original-to-public mapping is not saved.
    """
    values = pd.Series(series.dropna().unique())

    rng = np.random.default_rng(seed)
    shuffled_values = rng.permutation(values.to_numpy())

    width = max(4, len(str(len(shuffled_values))))

    mapping = {
        original: f"{prefix}_{i:0{width}d}"
        for i, original in enumerate(shuffled_values, start=1)
    }

    return series.map(mapping)


# ================================================================
# MAIN
# ================================================================


def main():

    # ------------------------------------------------------------
    # 1. Load pre-similarity analytical data
    # ------------------------------------------------------------

    print("Loading pre-similarity analytical data...")

    analysis_data = pd.read_parquet(
        ANALYSIS_PATH
    ).copy()

    analysis_data[ID_COL] = analysis_data[ID_COL].astype(str)

    if analysis_data[ID_COL].duplicated().any():
        duplicate_bids = analysis_data.loc[
            analysis_data[ID_COL].duplicated(keep=False),
            ID_COL,
        ].unique()

        raise ValueError(
            "Duplicate bids found in pre-similarity analytical data. "
            f"Examples: {duplicate_bids[:10]}"
        )

    print_sample_stats(
        "INPUT PRE-SIMILARITY SAMPLE",
        analysis_data,
    )

    append_sample_flow(
        "Input to final sample construction",
        analysis_data,
    )


    # ------------------------------------------------------------
    # 2. Validate upstream variables
    # ------------------------------------------------------------

    required_analysis_cols = [
        ID_COL,
        GRADE_COL,
        *UPSTREAM_REQUIRED_OUTCOMES,
    ]

    missing_analysis_cols = [
        col
        for col in required_analysis_cols
        if col not in analysis_data.columns
    ]

    if missing_analysis_cols:
        raise KeyError(
            "Required columns are missing from the pre-similarity data: "
            f"{missing_analysis_cols}"
        )

    # Script 6_1 should already guarantee completeness on these outcomes.
    upstream_missing = analysis_data[
        UPSTREAM_REQUIRED_OUTCOMES
    ].isna().sum()

    if upstream_missing.any():
        raise ValueError(
            "Pre-similarity data contain missing values in outcomes that "
            "should already be complete after script 6_1:\n"
            + upstream_missing.loc[upstream_missing > 0].to_string()
        )

    # The upstream sample should already be restricted to take-home exams.
    if "take_home" in analysis_data.columns:
        non_take_home = int((analysis_data["take_home"] != 1).sum())

        if non_take_home > 0:
            raise ValueError(
                "Pre-similarity data contain non-take-home observations: "
                f"{non_take_home:,}"
            )


    # ------------------------------------------------------------
    # 3. Load and validate similarity data
    # ------------------------------------------------------------

    print("\nLoading embedding-similarity data...")

    similarity_data = pd.read_parquet(
        SIMILARITY_PATH
    ).copy()

    similarity_data[ID_COL] = similarity_data[ID_COL].astype(str)

    if similarity_data[ID_COL].duplicated().any():
        duplicate_bids = similarity_data.loc[
            similarity_data[ID_COL].duplicated(keep=False),
            ID_COL,
        ].unique()

        raise ValueError(
            "Duplicate bids found in similarity data. "
            f"Examples: {duplicate_bids[:10]}"
        )

    available_similarity_cols = [
        col
        for col in SIMILARITY_COLS
        if col in similarity_data.columns
    ]

    if SIMILARITY_COL not in available_similarity_cols:
        raise KeyError(
            f"Required similarity column is missing: {SIMILARITY_COL}"
        )

    print(
        f"Similarity file contains "
        f"{len(similarity_data):,} hand-ins."
    )


    # ------------------------------------------------------------
    # 4. Merge similarity onto the full pre-similarity sample
    # ------------------------------------------------------------

    # Remove stale versions of similarity columns if present so that the
    # dedicated similarity file is the only source of these measures.
    analysis_data = analysis_data.drop(
        columns=[
            col
            for col in SIMILARITY_COLS
            if col in analysis_data.columns
        ],
        errors="ignore",
    )

    analysis_data = analysis_data.merge(
        similarity_data[
            [ID_COL, *available_similarity_cols]
        ],
        on=ID_COL,
        how="left",
        validate="one_to_one",
    )

    n_matched_similarity_rows = int(
        analysis_data[ID_COL].isin(
            set(similarity_data[ID_COL])
        ).sum()
    )

    print(
        f"Hand-ins matched to similarity file: "
        f"{n_matched_similarity_rows:,} / {len(analysis_data):,}"
    )

    print(
        f"Hand-ins with valid {SIMILARITY_COL}: "
        f"{analysis_data[SIMILARITY_COL].notna().sum():,}"
    )


    # ------------------------------------------------------------
    # 5. Require valid semantic similarity
    # ------------------------------------------------------------

    before_n = len(analysis_data)

    analysis_data = analysis_data.loc[
        analysis_data[SIMILARITY_COL].notna()
    ].copy()

    report_restriction(
        "Valid semantic similarity",
        before_n,
        analysis_data,
    )


    # ------------------------------------------------------------
    # 6. Require valid numerical grade
    # ------------------------------------------------------------

    # The grade was constructed upstream on the ordered 0--6 scale.
    # Non-numeric values and values outside this scale are treated as invalid.
    analysis_data[GRADE_COL] = pd.to_numeric(
        analysis_data[GRADE_COL],
        errors="coerce",
    )

    analysis_data[GRADE_COL] = analysis_data[GRADE_COL].where(
        analysis_data[GRADE_COL].isin(range(7))
    )

    n_missing_grade = int(
        analysis_data[GRADE_COL].isna().sum()
    )

    print(
        f"\nHand-ins without a valid numerical grade before final grade "
        f"restriction: {n_missing_grade:,}"
    )

    before_n = len(analysis_data)

    analysis_data = analysis_data.loc[
        analysis_data[GRADE_COL].notna()
    ].copy()

    report_restriction(
        "Valid numerical grade",
        before_n,
        analysis_data,
    )


    # ------------------------------------------------------------
    # 7. Require complete downstream analysis identifiers
    # ------------------------------------------------------------

    required_identifiers = [
        COURSE_COL,
        PERIOD_COL,
        PERIOD_START_COL,
        AFTER_COL,
    ]

    missing_identifier_cols = [
        col
        for col in required_identifiers
        if col not in analysis_data.columns
    ]

    if missing_identifier_cols:
        raise KeyError(
            "Required downstream analysis identifiers are missing: "
            f"{missing_identifier_cols}"
        )

    before_n = len(analysis_data)

    analysis_data = analysis_data.dropna(
        subset=required_identifiers
    ).copy()

    report_restriction(
        "Complete downstream analysis identifiers",
        before_n,
        analysis_data,
    )


    # ------------------------------------------------------------
    # 8. Re-apply recurring-course support after final restrictions
    # ------------------------------------------------------------
    #
    # Similarity and grade restrictions can remove observations after
    # the recurring-course restriction was imposed upstream. Re-check
    # the final sample so every retained course is represented both
    # before and after ChatGPT. This also guarantees at least two
    # submissions per retained course and prevents singleton fixed
    # effects from being silently removed in downstream regressions.

    course_period_support = (
        analysis_data
        .groupby(COURSE_COL)[AFTER_COL]
        .agg(
            lambda x: set(
                x.dropna().astype(int)
            )
        )
    )

    spanning_courses = (
        course_period_support.loc[
            course_period_support.apply(
                lambda values: {0, 1}.issubset(values)
            )
        ]
        .index
    )

    before_n = len(analysis_data)

    analysis_data = analysis_data.loc[
        analysis_data[COURSE_COL].isin(
            spanning_courses
        )
    ].copy()

    report_restriction(
        "Courses represented both before and after ChatGPT "
        "after final outcome restrictions",
        before_n,
        analysis_data,
    )

    # ------------------------------------------------------------
    # 9. Merge individual rare LLM-word counts
    # ------------------------------------------------------------
    # These variables are required for the individual-word analysis
    # in 8_1. They are merged only after the final analytical sample
    # has been constructed.
    
    print("\nLoading individual LLM-word measures...")
    
    llm_word_data = pd.read_json(
        LLM_WORD_MEASURES_PATH,
        lines=True,
    )
    
    required_llm_cols = [
        ID_COL,
        "rare_freqs",
        "n_words",
    ]
    
    missing_llm_cols = [
        col
        for col in required_llm_cols
        if col not in llm_word_data.columns
    ]
    
    if missing_llm_cols:
        raise KeyError(
            "LLM-word measures file is missing required columns: "
            f"{missing_llm_cols}"
        )
    
    llm_word_data = (
        llm_word_data[
            required_llm_cols
        ]
        .rename(
            columns={
                "n_words": "llm_n_words",
            }
        )
    )
    
    # Each submission should appear only once in the word-measures file.
    if llm_word_data[ID_COL].duplicated().any():
        raise ValueError(
            "Duplicate bids found in LLM-word measures file."
        )
    
    analysis_data = analysis_data.merge(
        llm_word_data,
        on=ID_COL,
        how="left",
        validate="one_to_one",
    )
    
    # Both variables are required for reproducing Figure 1B.
    missing_rare_freqs = analysis_data["rare_freqs"].isna().sum()
    missing_llm_n_words = analysis_data["llm_n_words"].isna().sum()
    
    print(
        f"  Missing rare-word dictionaries: {missing_rare_freqs:,}"
    )
    print(
        f"  Missing LLM word counts: {missing_llm_n_words:,}"
    )
    
    if missing_rare_freqs > 0 or missing_llm_n_words > 0:
        raise ValueError(
            "Some observations in the final analytical sample are missing "
            "individual LLM-word measures."
        )
     # ------------------------------------------------------------
    # 10. Merge individual rare LLM-word counts
    # ------------------------------------------------------------
    final_required = [
        COURSE_COL,
        PERIOD_COL,
        PERIOD_START_COL,
        AFTER_COL,
        GRADE_COL,
        SIMILARITY_COL,
        "rare_freqs",
        "llm_n_words",
        *UPSTREAM_REQUIRED_OUTCOMES,
    ]

    missing_final = analysis_data[
        final_required
    ].isna().sum()

    if missing_final.any():
        raise ValueError(
            "Final analytical sample still contains missing required values:\n"
            + missing_final.loc[missing_final > 0].to_string()
        )

    if analysis_data[ID_COL].duplicated().any():
        raise ValueError(
            "Duplicate bids found in final analytical sample."
        )

    invalid_after = ~analysis_data[AFTER_COL].isin([0, 1])

    if invalid_after.any():
        raise ValueError(
            "Final analytical sample contains observations outside the "
            "defined pre/post periods."
        )

    final_course_sizes = (
        analysis_data
        .groupby(COURSE_COL)
        .size()
    )

    if (final_course_sizes < 2).any():
        raise ValueError(
            "Final analytical sample contains singleton courses."
        )

    final_course_support = (
        analysis_data
        .groupby(COURSE_COL)[AFTER_COL]
        .nunique()
    )

    if (final_course_support < 2).any():
        raise ValueError(
            "Final analytical sample contains courses that are not "
            "represented both before and after ChatGPT."
        )

    append_sample_flow(
        "Final common analytical sample",
        analysis_data,
        before_n=len(analysis_data),
    )

    print_sample_stats(
        "FINAL COMMON ANALYTICAL SAMPLE",
        analysis_data,
    )

    if "AfleveringsTidspunkt" in analysis_data.columns:
        dates = pd.to_datetime(
            analysis_data["AfleveringsTidspunkt"],
            errors="coerce",
        )

        print(
            "  Date range: "
            f"{dates.min()} to {dates.max()}"
        )

# ================================================================
    # 11 CREATE PUBLIC ANALYSIS DATASET
    # ================================================================
    
    public_data = analysis_data.copy()
    
    cols_to_del = [
        "Kursusnavn",
        "censorid",
        "eksaminatorid",
        "BedoemtTidspunkt",
        "AfleveringsTidspunkt",
        "Sex",
        "eth_single_lastname",
        "eth_single",
        "Kursuskode",
        "coursecode",
        "coursecode_noE",
        "exam_code",
        "exam",
        "studerendeid",
    ]
    
    # Drop identifying / unused variables
    public_data = public_data.drop(
        columns=cols_to_del,
        errors="ignore",
    )
    
    # Replace original identifiers with anonymous reproducible IDs
    public_data["coursecode_clean"] = make_random_ids(
        public_data["coursecode_clean"],
        prefix="course",
        seed=42,
    )
    
    public_data["bid"] = make_random_ids(
        public_data["bid"],
        prefix="submission",
        seed=43,
    )
    
    assert len(public_data) == len(analysis_data)
    assert public_data["coursecode_clean"].nunique() == analysis_data["coursecode_clean"].nunique()
    assert public_data["bid"].nunique() == len(public_data)
    
   
    

    # ------------------------------------------------------------
    # 11. Save final analytical dataset and sample-flow table
    # ------------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    
    analysis_data.to_parquet(
        OUTPUT_PATH,
        index=False,
    )
    
    public_data.to_parquet(
        PUBLIC_OUTPUT_PATH,
        index=False,
    )
    
    save_sample_flow()

    print(
        f"\n💾 Saved final common analytical sample to: "
        f"{OUTPUT_PATH}"
    )

    print(
        f"\n💾 Saved public analytical dataset to: "
        f"{PUBLIC_OUTPUT_PATH}"
    )
# ================================================================
# RUN
# ================================================================

if __name__ == "__main__":
    main()
