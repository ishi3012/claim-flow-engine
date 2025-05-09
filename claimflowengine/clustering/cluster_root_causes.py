"""
cluster_root_causes.py

Clusters denied healthcare claims into latent root causes using a hybrid of NLP and unsupervised learning.
Designed to assist revenue cycle analysts in understanding systemic denial patterns beyond surface-level codes.
Can be used independently or as part of the ClaimFlowEngine pipeline:
Preprocessing → Denial Prediction → Root Cause Clustering → Routing

    This script:
        - Embeds denial reason text using SBERT (or ClinicalBERT via config)
        - Optionally appends structured features (CPT, payer, etc.)
        - Applies UMAP for dimensionality reduction
        - Performs unsupervised clustering with HDBSCAN
        - Saves cluster labels, models, and optional summaries for inspection
    Intended Use:
        - A post-denial-analysis step in the ClaimFlowEngine pipeline.
        - Supports downstream routing logic and CHW decision tooling.
    Inputs:
        - data/processed/balanced_claims.csv (from config["data"]["balanced_path"]) or
        - data/processed/claims.csv (from config["data"]["default_path"])

    Outputs:
        - Clustered denial claims CSV (to config["clustering"]["output_path"])
        - Serialized UMAP and HDBSCAN models to MODEL_DIR
        - Optional: Cluster summaries for interpretability
    Args:
        --config (str): Path to the YAML config file containing embedding and clustering settings.
    Key Components:
        - embed_denial_reasons: SBERT-based sentence embedding.
        - run_clustering: Dimensionality reduction + density-based clustering.
        - summarize_clusters (optional): Heuristic-based text summarization per cluster.
    Example:
        $ python claimflowengine/clustering/cluster_root_causes.py --config config/config.yaml

        Author: ClaimFlowEngine Project
"""

from __future__ import annotations

import pandas as pd
import joblib
import argparse
import sys
import traceback
from typing import Any, Dict


from claimflowengine.utils.paths import CONFIG_PATH, DATA_DIR, MODEL_DIR
from claimflowengine.utils.config_loader import load_config
from claimflowengine.clustering.embed import embed_denial_reasons
from claimflowengine.clustering.cluster import run_clustering
from claimflowengine.clustering.label_clusters import summarize_clusters
from claimflowengine.utils.log import get_logger

# Initialize Logger
logger = get_logger(__name__)


# main
def main(config: Dict[str, Any]) -> None:
    """
    Main clustering workflow to identify latent root cause clusters in denied claims.

    Steps:
    1. Load preprocessed claims dataset
    2. Filter denied claims only
    3. Embed denial reasons via SBERT (or ClinicalBERT)
    4. Reduce dimensionality using UMAP
    5. Cluster using HDBSCAN
    6. Save results and optionally summarize clusters

    Args:
        config (dict): Configuration dictionary loaded from YAML
    """
    # Load dataset
    logger.info(" Loading dataset...")

    path_key = "balanced_path" if "balanced_path" in config["data"] else "default_path"
    df = pd.read_csv(DATA_DIR / config["data"][path_key])

    denial_col = "denial_flag" if "denial_flag" in df.columns else "denied"
    df_denied = df[df[denial_col] == 1].copy()

    logger.info("Embedding denial reasons...")
    embeddings = embed_denial_reasons(df_denied["denial_reason"].tolist(), config)

    logger.info("Running clustering pipeline...")
    cluster_labels, reducer, clusterer = run_clustering(embeddings, df_denied, config)

    logger.info("Saving cluster results...")
    df_denied["cluster"] = cluster_labels
    output_path = DATA_DIR / config["clustering"]["output_path"]
    df_denied.to_csv(output_path, index=False)

    joblib.dump(reducer, MODEL_DIR / "umap_reducer.pkl")
    joblib.dump(clusterer, MODEL_DIR / "hdbscan_model.pkl")

    logger.info(f" Clustering complete. Saved to: {output_path}")

    if config["clustering"].get("summarize_clusters", False):
        logger.info("Summarizing clusters...")
        summarize_clusters(df_denied, config)


if __name__ == "__main__":
    """
    CLI Entrypoint:
    Parses config path and runs the clustering pipeline using the loaded config.
    Designed for use in batch mode or via orchestrated pipelines (e.g., Vertex AI).
    """

    parser = argparse.ArgumentParser(description="Cluster denied claims by root cause.")
    parser.add_argument(
        "--config", type=str, default=CONFIG_PATH, help="Path to YAML config"
    )
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        print(f"[CONFIG ERROR] {e}")
        logger.exception(f"[CONFIG ERROR] {e}")
        sys.exit(1)
    except Exception as e:
        print("Unexpected error during clustering pipeline:")
        logger.exception(f"[CONFIG ERROR] {e}")
        traceback.print_exc()
        sys.exit(1)
