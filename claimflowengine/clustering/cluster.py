"""
cluster.py

Performs dimensionality reduction and unsupervised clustering on denial reason embeddings
to reveal latent root cause structures. Built for scalability and interpretability in healthcare
claim pipelines. Designed for use within ClaimFlowEngine’s clustering and routing modules.

    This module:
        - Reduces high-dimensional SBERT embeddings using UMAP
        - Applies HDBSCAN to identify density-based clusters without pre-specifying cluster count
        - Optionally visualizes 2D cluster layout using matplotlib
        - Returns trained clustering models for serialization and reuse

    Intended Use:
        - As a core component of the Root Cause Clustering Engine.
        - Enables visual and structural segmentation of denied claims.
        - Supports downstream workflow routing and CHW escalation logic.

    Inputs:
        - `np.ndarray` of denial_reason embeddings (from embed_denial_reasons)
        - Optional: filtered denied claims DataFrame (for plotting purposes)
        - Clustering hyperparameters and toggles from config.yaml

    Outputs:
        - Cluster assignments per claim (np.ndarray)
        - Trained UMAP reducer and HDBSCAN clusterer
        - Optional 2D plot saved to config["clustering"]["cluster_plot_path"]

    Functions:
        - run_clustering: Executes UMAP and HDBSCAN on embeddings
        - plot_clusters (optional): Saves a 2D cluster visualization

    Args:
        embeddings (np.ndarray): Sentence-level vector embeddings
        df (pd.DataFrame): Denied claims metadata (used for logging/plot context)
        config (Dict): YAML-derived settings controlling all behavior

    Example:
        from claimflowengine.clustering.cluster import run_clustering
        cluster_labels, reducer, clusterer = run_clustering(embeddings, df, config)

    Author: ClaimFlowEngine Project
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import umap
import hdbscan
from typing import Tuple, Dict, Any
from pathlib import Path
from claimflowengine.utils.log import get_logger


# initialize logger
logger = get_logger(__name__)

# clustering execution


def run_clustering(
    embeddings: np.ndarray, df: pd.DataFrame, config: Dict[str, Any]
) -> Tuple[np.ndarray, umap.UMAP, hdbscan.HDBSCAN]:
    """
    Run UMAP + HDBSCAN clustering pipeline.

    Args:
        - embedding (np.ndarray) : Embedded denial_reason vectors
        - df (pd.DataFrame)      : Original claims DataFrame (denied claims)
        - config (Dict)          : Configuration parameters loaded from YAML

    Returns:
        Tuple:
            - cluster_labels (np.ndarray)   : Cluster assignments per claim
            - reducer (umap.UMAP)           : Trained UMAP reducer
            - clusterer (hdbscan.HDBSCAN)   : Trained HDBSCAN model
    """
    # Reduces high-dimensional SBERT embeddings using UMAP
    logger.info(" Running UMAP dimensionality reduction ...")

    reducer = umap.UMAP(
        n_neighbors=config["clustering"].get("n_neighbors", 30),
        min_dist=config["clustering"].get("min_dist", 0.0),
        n_components=config["clustering"].get("n_components", 10),
        metric=config["clustering"].get("umap_metric", "cosine"),
        random_state=config["clustering"].get("random_state", 42),
    )
    X_umap = reducer.fit_transform(embeddings)
    logger.info(f"UMAP complete. Output shape: {X_umap.shape}")

    # Apply HDBSCAN to identify density-based clusters
    logger.info(" Running HDBSCAN clustering...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=config["clustering"].get("hdbscan_min_cluster_size", 15),
        metric=config["clustering"].get("hdbscan_metric", "euclidean"),
        prediction_data=True,
    )

    cluster_labels = clusterer.fit_predict(X_umap)
    logger.info(
        f"HDBSCAN complete. Found {len(np.unique(cluster_labels)) - (1 if -1 in cluster_labels else 0)} clusters."
    )

    # Optional 2D cluster plot
    if config["clustering"].get("plot_clusters", False) and X_umap.shape[1] <= 2:
        plot_clusters(X_umap, cluster_labels, config)

    return cluster_labels, reducer, clusterer


def plot_clusters(
    X_umap: np.ndarray, labels: np.ndarray, config: Dict[str, Any]
) -> None:
    """
    Plot and save 2D UMAP cluster visualization.

    Args:
        X_umap (np.ndarray): 2D UMAP-reduced data
        labels (np.ndarray): Cluster labels
        config (dict): YAML config with output path
    """
    logger.info("Plotting cluster visualization...")

    plt.figure(figsize=(10, 6))

    colors = cm.tab20(labels.astype(int) % 20)

    plt.scatter(
        X_umap[:, 0],
        X_umap[:, 1],
        c=colors,
        s=10,
        alpha=0.8,
        edgecolor="k",
        linewidth=0.1,
    )

    plt.title("HDBSCAN Clusters (2D UMAP)")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.grid(True)

    out_path = Path(
        config["clustering"].get("cluster_plot_path", "outputs/cluster_plot.png")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    logger.info(f"Cluster plot saved to: {out_path}")
