"""
Module: claimflowengine.inference.merger
Combines model prediction output and root cause cluster
output into a routing-ready DataFrame.
Features:
- Merges on `claim_id`
- Validates schema
- Ensures readiness for routing pipeline
Author: ClaimFlowEngine (2025)
"""

from pathlib import Path
from typing import Union

import pandas as pd

from claimflowengine.configs.paths import (
    CLUSTERING_CLAIMS_LABELED_PATH,
    DENIAL_PREDICTION_OUTPUT_PATH,
)

REQUIRED_COLUMNS = {
    "claim_id",
    "denial_prediction",
    "denial_probability",
    "denial_cluster_id",
}


def merge_predictions_and_clusters(
    pred_df: pd.DataFrame, cluster_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge predictions and cluster labels on `claim_id`.

    Args:
        pred_df (pd.DataFrame): Output of denial prediction model
        cluster_df (pd.DataFrame): Output of root cause clustering

    Returns:
        pd.DataFrame: Merged routing-ready claims
    """
    if not pred_df["claim_id"].is_unique or not cluster_df["claim_id"].is_unique:
        raise ValueError(
            "`claim_id` must be unique in both"
            + " prediction and clustering dataframes."
        )

    merged = pd.merge(pred_df, cluster_df, on="claim_id", how="inner")

    if not REQUIRED_COLUMNS.issubset(set(merged.columns)):
        missing = REQUIRED_COLUMNS - set(merged.columns)
        raise ValueError(f"Missing required columns: {missing}")

    return merged


def merge_predictions_and_clusters_from_files(
    pred_path: Union[str, Path] = DENIAL_PREDICTION_OUTPUT_PATH,
    cluster_path: Union[str, Path] = CLUSTERING_CLAIMS_LABELED_PATH,
) -> pd.DataFrame:
    """
    Loads predictions and clusters from CSVs and merges them.
    Args:
        pred_path (str): Path to prediction output CSV
        cluster_path (str): Path to clustering output CSV
    Returns:
        pd.DataFrame: Routing-ready claims
    """
    pred_df = pd.read_csv(pred_path)
    cluster_df = pd.read_csv(cluster_path)
    return merge_predictions_and_clusters(pred_df, cluster_df)
