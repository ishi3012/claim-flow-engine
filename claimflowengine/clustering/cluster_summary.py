"""
Module: cluster_summary.py — Cluster Labeling and Explanation

This module computes representative labels for each cluster using either:
- Mode of denial_reason (simple majority)
- TF-IDF vectorization and top keyword scoring

Features:
- Cluster-wise label generation
- Attachable label map to original DataFrame
- Compatible with human-in-the-loop auditing and downstream routing logic

Author: ClaimFlowEngine Team (2025)
"""

from typing import Dict, cast

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from claimflowengine.utils.logger import get_logger

# Initialize logger
logger = get_logger("cluster")


def generate_cluster_labels(
    clustered_df: pd.DataFrame,
    text_col: str = "denial_reason",
    cluster_col: str = "denial_cluster_id",
    method: str = "mode",
    top_n: int = 1,
) -> Dict[int, str]:
    """
    Generate interpretable labels for each cluster.

    Args:
        clustered_df: DataFrame with cluster assignments
        text_col: Column with raw denial text
        cluster_col: Column with cluster IDs
        method: "mode" or "tfidf" (default: mode)
        top_n: number of keywords to keep if using TF-IDF

    Returns:
        Dict mapping cluster_id -> label string
    """

    labels = {}

    if method == "mode":
        for cid, group in clustered_df.groupby(cluster_col):
            mode_text = group[text_col].mode(dropna=True)
            labels[cid] = mode_text.iloc[0] if not mode_text.empty else "Unknown"
    elif method == "tfidf":
        for cid, group in clustered_df.groupby(cluster_col):
            texts = group[text_col].dropna().astype(str).tolist()
            if not texts:
                labels[cid] = "Unknown"
                continue
            vectorizer = TfidfVectorizer(stop_words="english", max_features=50)
            tfidf_matrix = vectorizer.fit_transform(texts)
            tfidf_mean = tfidf_matrix.mean(axis=0).A1
            feature_names = vectorizer.get_feature_names_out()
            valid_idx = np.argsort(tfidf_mean)[::-1]
            top_terms = [feature_names[i] for i in valid_idx if tfidf_mean[i] > 0][
                :top_n
            ]

            labels[cid] = ", ".join(top_terms) if top_terms else "Unknown"
            logger.debug(f"Cluster {cid}: {labels[cid]}")
    else:
        logger.error("Invalid method. Choose 'mode' or 'tfidf'.")
        raise ValueError("Invalid method. Choose 'mode' or 'tfidf'.")
    return cast(dict[int, str], labels)


def attach_cluster_labels(
    df: pd.DataFrame,
    label_map: Dict[int, str],
    cluster_col: str = "denial_cluster_id",
    label_col: str = "cluster_label",
) -> pd.DataFrame:
    """
    Attach cluster label strings to the original DataFrame.

    Args:
        df: DataFrame with cluster assignments
        label_map: Dict mapping cluster_id -> label
        cluster_col: Cluster ID column
        label_col: New label column name

    Returns:
        DataFrame with new label column
    """
    df[label_col] = df[cluster_col].map(label_map)
    return df
