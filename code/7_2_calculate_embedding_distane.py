# calculate_embedding_similarity.py
#
# Calculate mean within-course × period cosine similarity for each hand-in.
#
# Similarity is calculated in two embedding spaces:
#   1. Full MPNet embeddings
#   2. PCA-reduced embeddings
#
# Important:
#   - The input is the pre-similarity analytical sample produced by script 6_1.
#   - This upstream sample is already restricted to take-home exams.
#   - Only embeddings corresponding to bids in the analytical sample are loaded.
#   - PCA is fitted ONLY on embeddings from the analytical sample.
#   - Loaded full embeddings are not re-normalized before PCA.
#   - Cosine normalization is performed internally when cosine similarity
#     is calculated.
#   - PCA and similarity calculations run on CPU.
#
# Output:
#   One row per analytical hand-in with:
#       bid
#       coursecode_clean
#       period
#       n_course_period_handins
#       cosine_similarity_full
#       cosine_similarity_pca_100


from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from tqdm.auto import tqdm


# ================================================================
# CONFIG
# ================================================================

DATA_PATH = Path("/work/data_assigments/DATA/")

# Pre-similarity analytical sample produced by script 6_1
ANALYSIS_PATH = DATA_PATH / "english_handin_analysis_pre_similarity.parquet"

EMBEDDING_PATH = DATA_PATH / "embeddings_english.jsonl"

OUTPUT_PATH = DATA_PATH / "embedding_similarity_english.parquet"


# Column names
ID_COL = "bid"
COURSE_COL = "coursecode_clean"
PERIOD_COL = "period"


# PCA settings
N_COMPONENTS = 100
PCA_RANDOM_STATE = 42


# ================================================================
# HELPERS
# ================================================================

def count_lines(path: Path) -> int:
    """
    Count lines in a JSONL file.

    Used only to give tqdm a total.
    """
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def load_selected_embeddings(
    path: Path,
    keep_bids: set,
):
    """
    Load embeddings only for bids contained in keep_bids.

    Embeddings are loaded exactly as stored in the JSONL file.
    No normalization or other transformation is performed here.

    Returns
    -------
    bids : list[str]
        bid corresponding to each row of X.

    X : np.ndarray
        Embedding matrix with shape:
        [n_handins, embedding_dimension]
    """

    bids = []
    vectors = []

    total_lines = count_lines(path)

    seen_bids = set()

    with path.open("r", encoding="utf-8") as f:

        for line in tqdm(
            f,
            total=total_lines,
            desc="Loading analytical embeddings",
        ):

            if not line.strip():
                continue

            row = json.loads(line)

            bid = str(row["bid"])

            # Ignore embeddings outside analytical sample
            if bid not in keep_bids:
                continue

            # Detect duplicate embedding records
            if bid in seen_bids:
                raise ValueError(
                    f"Duplicate embedding found for bid: {bid}"
                )

            seen_bids.add(bid)

            vector = np.asarray(
                row["embedding"],
                dtype=np.float32,
            )

            bids.append(bid)
            vectors.append(vector)

    if not vectors:
        raise ValueError(
            "No embeddings matched bids in the analytical sample."
        )

    # Check that all embeddings have the same dimension
    embedding_dims = {
        len(vector)
        for vector in vectors
    }

    if len(embedding_dims) != 1:
        raise ValueError(
            "Embeddings do not all have the same dimension."
        )

    X = np.vstack(vectors)

    return bids, X


def mean_peer_cosine(
    X: np.ndarray,
) -> np.ndarray:
    """
    Calculate each row's mean cosine similarity to every OTHER
    row in X.

    For hand-in i:

        mean_similarity_i
            = mean cosine similarity between i and all j != i

    This implementation does NOT construct the complete n × n
    pairwise similarity matrix.

    Instead, after obtaining unit vectors z_i:

        sum_j cosine(i, j)
            = z_i @ sum_j(z_j)

    Because self-similarity is 1:

        sum_{j != i} cosine(i, j)
            = z_i @ sum_j(z_j) - 1

    Therefore:

        mean_{j != i} cosine(i, j)
            = [z_i @ sum_j(z_j) - 1] / (n - 1)

    The supplied X matrix is NOT modified.

    Returns
    -------
    np.ndarray
        Mean peer cosine similarity for each row.
        Returns NaN for groups containing only one hand-in.
    """

    n = X.shape[0]

    # No peers to compare against
    if n < 2:
        return np.full(
            n,
            np.nan,
            dtype=np.float32,
        )

    # Vector lengths needed for cosine similarity
    norms = np.linalg.norm(
        X,
        axis=1,
        keepdims=True,
    )

    if np.any(norms == 0):
        raise ValueError(
            "Encountered a zero-length embedding."
        )

    # Unit vectors are created only as a temporary representation
    # for computing cosine similarity.
    #
    # X itself is not changed.
    X_unit = X / norms

    # Sum of all vectors in this course-period group
    group_sum = X_unit.sum(
        axis=0
    )

    # Sum of similarities between each hand-in and every hand-in
    # in the group, including itself
    similarity_sum = (
        X_unit @ group_sum
    )

    # Remove self-similarity (= 1)
    similarity_sum = (
        similarity_sum - 1.0
    )

    # Mean similarity to the n - 1 other hand-ins
    mean_similarity = (
        similarity_sum
        / (n - 1)
    )

    return mean_similarity.astype(
        np.float32
    )


# ================================================================
# MAIN
# ================================================================

def main():

    # ------------------------------------------------------------
    # 1. Load analytical data
    # ------------------------------------------------------------

    print("Loading analytical data...")

    analysis_data = pd.read_parquet(
        ANALYSIS_PATH
    )

    print(
        f"Input to similarity calculation: "
        f"{len(analysis_data):,} hand-ins"
    )


    # ------------------------------------------------------------
    # 2. Validate required columns
    # ------------------------------------------------------------

    required_cols = [
        ID_COL,
        COURSE_COL,
        PERIOD_COL,
        "take_home",
    ]

    missing_cols = [
        col
        for col in required_cols
        if col not in analysis_data.columns
    ]

    if missing_cols:
        raise ValueError(
            "Missing required columns in analysis_data: "
            f"{missing_cols}"
        )


    # ------------------------------------------------------------
    # 4. Clean IDs and required grouping variables
    # ------------------------------------------------------------

    analysis_data[ID_COL] = (
        analysis_data[ID_COL]
        .astype(str)
    )

    analysis_data = analysis_data.dropna(
        subset=[
            COURSE_COL,
            PERIOD_COL,
        ]
    ).copy()


    # Each bid should occur once in the analytical sample
    duplicated_bids = (
        analysis_data[ID_COL]
        .duplicated()
    )

    if duplicated_bids.any():

        duplicate_examples = (
            analysis_data.loc[
                duplicated_bids,
                ID_COL,
            ]
            .head(10)
            .tolist()
        )

        raise ValueError(
            "Duplicate bids found in analytical sample. "
            f"Examples: {duplicate_examples}"
        )


    analysis_data = (
        analysis_data
        .reset_index(drop=True)
    )


    print(
        f"Analytical take-home hand-ins: "
        f"{len(analysis_data):,}"
    )

    print(
        f"Unique courses: "
        f"{analysis_data[COURSE_COL].nunique():,}"
    )

    print(
        f"Unique course-period groups: "
        f"{analysis_data.groupby([COURSE_COL, PERIOD_COL]).ngroups:,}"
    )


    # ------------------------------------------------------------
    # 5. Define bids used in the analysis
    # ------------------------------------------------------------

    analysis_bids = set(
        analysis_data[ID_COL]
    )

    print(
        f"\nBids requiring embeddings: "
        f"{len(analysis_bids):,}"
    )


    # ------------------------------------------------------------
    # 6. Load ONLY embeddings in analytical sample
    # ------------------------------------------------------------

    embedding_bids, X_loaded = (
        load_selected_embeddings(
            EMBEDDING_PATH,
            analysis_bids,
        )
    )


    print(
        f"\nMatched embeddings: "
        f"{len(embedding_bids):,}"
    )

    print(
        f"Embedding dimension: "
        f"{X_loaded.shape[1]:,}"
    )


    # ------------------------------------------------------------
    # 7. Check embedding coverage
    # ------------------------------------------------------------

    matched_bids = set(
        embedding_bids
    )

    missing_embedding_bids = (
        analysis_bids
        - matched_bids
    )

    print(
        f"Analytical hand-ins without embeddings: "
        f"{len(missing_embedding_bids):,}"
    )


    if missing_embedding_bids:

        print(
            "Hand-ins without embeddings will be excluded."
        )

        analysis_data = analysis_data.loc[
            analysis_data[ID_COL].isin(
                matched_bids
            )
        ].copy()

        analysis_data = (
            analysis_data
            .reset_index(drop=True)
        )

    print(
        f"Hand-ins retained after embedding match: "
        f"{len(analysis_data):,}"
    )


    if len(analysis_data) == 0:
        raise ValueError(
            "No analytical observations remain after "
            "matching embeddings."
        )


    # ------------------------------------------------------------
    # 8. Align embedding matrix with analysis_data
    # ------------------------------------------------------------

    # X_loaded follows the order in embeddings_english.jsonl.
    # We explicitly reorder it to follow analysis_data.

    embedding_lookup = {
        bid: i
        for i, bid
        in enumerate(embedding_bids)
    }

    embedding_indices = np.array(
        [
            embedding_lookup[bid]
            for bid in analysis_data[ID_COL]
        ],
        dtype=np.int64,
    )

    X_full = X_loaded[
        embedding_indices
    ]


    # We no longer need the JSONL ordering
    del X_loaded


    # ------------------------------------------------------------
    # 9. Diagnostic: inspect existing embedding norms
    # ------------------------------------------------------------

    # IMPORTANT:
    # We do NOT normalize X_full here.
    #
    # This diagnostic simply checks what was stored by the
    # embedding-generation script.

    full_norms = np.linalg.norm(
        X_full,
        axis=1,
    )

    print("\nStored embedding norm diagnostic:")

    print(
        f"  Mean: {full_norms.mean():.6f}"
    )

    print(
        f"  SD:   {full_norms.std():.6f}"
    )

    print(
        f"  Min:  {full_norms.min():.6f}"
    )

    print(
        f"  Max:  {full_norms.max():.6f}"
    )


    if np.any(full_norms == 0):
        raise ValueError(
            "Zero-length embeddings found."
        )


    # ------------------------------------------------------------
    # 10. Fit PCA on analytical embeddings ONLY
    # ------------------------------------------------------------

    n_obs, embedding_dim = (
        X_full.shape
    )

    max_components = min(
        n_obs,
        embedding_dim,
    )


    if N_COMPONENTS >= max_components:

        raise ValueError(
            f"N_COMPONENTS={N_COMPONENTS} is too large. "
            f"It must be smaller than min(n observations, "
            f"embedding dimensions) = {max_components}."
        )


    print(
        "\nFitting PCA on analytical sample only..."
    )

    print(
        f"PCA observations: "
        f"{n_obs:,}"
    )

    print(
        f"Dimensions: "
        f"{embedding_dim} -> {N_COMPONENTS}"
    )


    # Randomized PCA is considerably faster than a full SVD here
    # and runs entirely on CPU.
    pca = PCA(
        n_components=N_COMPONENTS,
        svd_solver="randomized",
        random_state=PCA_RANDOM_STATE,
    )


    X_pca = pca.fit_transform(
        X_full
    ).astype(
        np.float32
    )


    explained_variance = (
        pca
        .explained_variance_ratio_
        .sum()
    )


    print(
        f"Explained variance retained: "
        f"{explained_variance:.2%}"
    )


    # ------------------------------------------------------------
    # 11. Prepare output arrays
    # ------------------------------------------------------------

    similarity_full = np.full(
        len(analysis_data),
        np.nan,
        dtype=np.float32,
    )

    similarity_pca = np.full(
        len(analysis_data),
        np.nan,
        dtype=np.float32,
    )

    group_size = np.zeros(
        len(analysis_data),
        dtype=np.int32,
    )


    # ------------------------------------------------------------
    # 12. Create course × period groups
    # ------------------------------------------------------------

    grouped_indices = (
        analysis_data
        .groupby(
            [
                COURSE_COL,
                PERIOD_COL,
            ],
            sort=False,
        )
        .indices
    )


    print(
        f"\nCalculating similarity within "
        f"{len(grouped_indices):,} course-period groups..."
    )


    # ------------------------------------------------------------
    # 13. Calculate similarity
    # ------------------------------------------------------------

    for (
        group_key,
        idx,
    ) in tqdm(
        grouped_indices.items(),
        total=len(grouped_indices),
        desc="Course-period similarity",
    ):

        idx = np.asarray(
            idx,
            dtype=np.int64,
        )

        n = len(idx)

        group_size[idx] = n


        # --------------------------------------------------------
        # Full embedding cosine similarity
        # --------------------------------------------------------

        similarity_full[idx] = (
            mean_peer_cosine(
                X_full[idx]
            )
        )


        # --------------------------------------------------------
        # PCA embedding cosine similarity
        # --------------------------------------------------------

        similarity_pca[idx] = (
            mean_peer_cosine(
                X_pca[idx]
            )
        )


    # ------------------------------------------------------------
    # 14. Add results to analytical data
    # ------------------------------------------------------------

    analysis_data[
        "n_course_period_handins"
    ] = group_size

    analysis_data[
        "cosine_similarity_full"
    ] = similarity_full

    pca_similarity_col = (
        f"cosine_similarity_pca_{N_COMPONENTS}"
    )

    analysis_data[
        pca_similarity_col
    ] = similarity_pca


    # ------------------------------------------------------------
    # 15. Validation
    # ------------------------------------------------------------

    print("\nGroup size summary:")

    print(
        analysis_data[
            "n_course_period_handins"
        ]
        .describe()
    )


    n_singletons = (
        analysis_data[
            "n_course_period_handins"
        ] == 1
    ).sum()


    print(
        f"\nHand-ins in singleton course-period groups: "
        f"{n_singletons:,}"
    )


    # Singleton groups must have undefined similarity
    assert (
        analysis_data.loc[
            analysis_data[
                "n_course_period_handins"
            ] == 1,
            "cosine_similarity_full",
        ]
        .isna()
        .all()
    )

    assert (
        analysis_data.loc[
            analysis_data[
                "n_course_period_handins"
            ] == 1,
            pca_similarity_col,
        ]
        .isna()
        .all()
    )


    # ------------------------------------------------------------
    # 16. Similarity diagnostics
    # ------------------------------------------------------------

    print("\nSimilarity summary:")

    print(
        analysis_data[
            [
                "cosine_similarity_full",
                pca_similarity_col,
            ]
        ]
        .describe()
    )

    n_valid_similarity = int(
        analysis_data["cosine_similarity_full"]
        .notna()
        .sum()
    )
    n_missing_similarity = (
        len(analysis_data) - n_valid_similarity
    )

    print(
        f"\nHand-ins with valid full-embedding similarity: "
        f"{n_valid_similarity:,}"
    )
    print(
        f"Hand-ins without valid full-embedding similarity: "
        f"{n_missing_similarity:,}"
    )


    # Cosine similarity is bounded above by 1.
    # Small numerical tolerance is allowed.
    full_max = (
        analysis_data[
            "cosine_similarity_full"
        ]
        .dropna()
        .max()
    )

    pca_max = (
        analysis_data[
            pca_similarity_col
        ]
        .dropna()
        .max()
    )


    if pd.notna(full_max):
        assert full_max <= 1.00001

    if pd.notna(pca_max):
        assert pca_max <= 1.00001


    # Cosine similarity is bounded below by -1.
    full_min = (
        analysis_data[
            "cosine_similarity_full"
        ]
        .dropna()
        .min()
    )

    pca_min = (
        analysis_data[
            pca_similarity_col
        ]
        .dropna()
        .min()
    )


    if pd.notna(full_min):
        assert full_min >= -1.00001

    if pd.notna(pca_min):
        assert pca_min >= -1.00001


    # ------------------------------------------------------------
    # 17. Compare full and PCA similarities
    # ------------------------------------------------------------

    valid_similarity = (
        analysis_data[
            [
                "cosine_similarity_full",
                pca_similarity_col,
            ]
        ]
        .dropna()
    )


    if len(valid_similarity) > 1:

        similarity_correlation = (
            valid_similarity[
                "cosine_similarity_full"
            ]
            .corr(
                valid_similarity[
                    pca_similarity_col
                ]
            )
        )

        print(
            f"\nCorrelation between full and "
            f"PCA similarities: "
            f"{similarity_correlation:.4f}"
        )


    # ------------------------------------------------------------
    # 18. Create output
    # ------------------------------------------------------------

    output_cols = [
        ID_COL,
        COURSE_COL,
        PERIOD_COL,
        "n_course_period_handins",
        "cosine_similarity_full",
        pca_similarity_col,
    ]


    # Keep period_start too if available
    if "period_start" in analysis_data.columns:

        output_cols.insert(
            3,
            "period_start",
        )


    output = analysis_data[
        output_cols
    ].copy()


    # ------------------------------------------------------------
    # 19. Save
    # ------------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    output.to_parquet(
        OUTPUT_PATH,
        index=False,
    )


    print(
        "\nDone."
    )

    print(
        f"Saved {len(output):,} hand-ins to:"
    )

    print(
        OUTPUT_PATH
    )


# ================================================================
# RUN
# ================================================================

if __name__ == "__main__":
    main()