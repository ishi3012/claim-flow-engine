"""
Module: clustering_pipeline.py — Denial Clustering Orchestration Logic

This module exposes a callable `cluster_claims()` function that:
- Loads transformers
- Applies full preprocessing + embedding
- Clusters claims into latent root cause groups
- Saves clustered output and artifacts

Intended Use:
- Called by batch script or FastAPI route
- Compatible with `predictor.py` structure

Author: ClaimFlowEngine Team (2025)
"""

from pathlib import Path
from typing import Optional, Union

import pandas as pd

from claimflowengine.clustering.root_cause_cluster import run_clustering_pipeline
from claimflowengine.configs.paths import (
    CLUSTER_MODEL_PATH,
    CLUSTERING_OUTPUT_PATH,
    REDUCER_MODEL_PATH,
    SENTENCE_TRANSFORMER_MODEL_NAME,
)
from claimflowengine.utils.logger import get_logger

logger = get_logger("cluster")


def cluster_claims(
    raw_data: pd.DataFrame,
    text_col: str = "denial_reason",
    notes_col: Optional[str] = "followup_notes",
    use_notes: bool = True,
    id_col: str = "claim_id",
    model_name: str = SENTENCE_TRANSFORMER_MODEL_NAME,
    output_path: Union[str, Path] = CLUSTERING_OUTPUT_PATH,
    cluster_model_path: Union[str, Path] = CLUSTER_MODEL_PATH,
    reducer_model_path: Union[str, Path] = REDUCER_MODEL_PATH,
) -> pd.DataFrame:
    """
    Wrapper to run denial root cause clustering on a raw DataFrame.

    Returns:
        pd.DataFrame with added 'denial_cluster_id' column.
    """

    logger.info(f"Received data for clustering: {raw_data.shape} rows.")

    return run_clustering_pipeline(
        df=raw_data,
        text_col=text_col,
        notes_col=notes_col,
        use_notes=use_notes,
        id_col=id_col,
        model_name=model_name,
        output_path=output_path,
        cluster_model_path=cluster_model_path,
        reducer_model_path=reducer_model_path,
    )
