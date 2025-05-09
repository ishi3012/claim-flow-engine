"""
label_clusters.py

Generates post-hoc labels for denial reason clusters using TF-IDF-based summarization.
Enables human analysts and clinical workflows to interpret root cause clusters
in denied claims. Designed for integration within ClaimFlowEngine’s clustering module.

    This module:
        - Groups denial_reason text by cluster ID
        - Computes TF-IDF weights within each cluster
        - Extracts representative terms as summary labels
        - Optionally saves cluster summaries to CSV

    Intended Use:
        - As a final interpretability step after HDBSCAN clustering
        - Supports CHW escalation, denial management, and pattern recognition

    Inputs:
        - `pd.DataFrame` of denied claims with `denial_reason` and `cluster` columns
        - `Dict` config with cluster summary toggles and output paths

    Outputs:
        - Logged cluster summaries with top denial terms
        - Optional CSV summary saved to config["clustering"]["cluster_summary_path"]

    Functions:
        - summarize_clusters: Computes and optionally saves per-cluster keyword summaries

    Args:
        df (pd.DataFrame): Denied claims with denial_reason and cluster columns
        config (Dict): YAML-derived settings including I/O paths and toggles

    Example:
        from claimflowengine.clustering.label_clusters import summarize_clusters
        summarize_clusters(df_denied, config)

    Author: ClaimFlowEngine Project
"""

from __future__ import annotations

import pandas as pd
from typing import Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path
from claimflowengine.utils.log import get_logger

logger = get_logger(__name__)


def summarize_clusters(df: pd.DataFrame, config: Dict[str, Any]) -> None:
    """
    Summarizes denial_reason text for each cluster using TF-IDF keyword extraction.

     Args:
        df (pd.DataFrame): DataFrame containing denied claims with cluster labels
        config (dict): Loaded configuration dictionary
    """

    logger.info("Generating TF-IDF summaries for each cluster...")
    summaries = []

    cluster_ids = sorted(df["cluster"].unique())
    cluster_ids = [cid for cid in cluster_ids if cid != -1]

    for cluster_id in cluster_ids:
        cluster_texts = (
            df[df["cluster"] == cluster_id]["denial_reason"].dropna().tolist()
        )

        if not cluster_texts:
            logger.warning(f"No text available for cluster {cluster_id}")
            continue

        vectorizer = TfidfVectorizer(stop_words="english", max_features=10)
        # X = vectorizer.fit_transform(cluster_texts)
        top_terms = vectorizer.get_feature_names_out()

        logger.info(
            f"Cluster {cluster_id} ({len(cluster_texts)} samples): {', '.join(top_terms)}"
        )
        summaries.append(
            {
                "cluster_id": cluster_id,
                "top_terms": ", ".join(top_terms),
                "count": len(cluster_texts),
            }
        )

    if config["clustering"].get("save_cluster_summary", False):
        out_path = Path(
            config["clustering"].get(
                "cluster_summary_path", "outputs/cluster_summary.csv"
            )
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if summaries:
            pd.DataFrame(summaries).to_csv(out_path, index=False)
        else:
            logger.warning("No clusters were summarized — writing empty header.")
            pd.DataFrame(columns=["cluster_id", "top_terms", "count"]).to_csv(
                out_path, index=False
            )

        logger.info(f"Cluster summary saved to: {out_path}")
