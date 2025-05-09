"""
embed.py

Embeds denial_reason text using SBERT or ClinicalBERT to produce semantically rich
sentence-level vector representations. Caches results to disk for performance
and reproducibility. Designed for reuse across clustering and retrieval pipelines.

    This module:
        - Loads a sentence-transformer model from HuggingFace (default: MiniLM)
        - Encodes a list of denial_reason strings into dense vectors
        - Optionally caches embeddings to disk using joblib
        - Supports plug-and-play via config["clustering"]["embedding_model"]

    Intended Use:
        - As the first step in the Root Cause Clustering pipeline
        - Can also support downstream semantic search or retrieval tasks

    Inputs:
        - List of denial_reason texts (List[str])
        - Config dict containing model name and cache path

    Outputs:
        - Numpy array of embeddings (n_samples x dim)
        - Optionally saved `.pkl` file of cached vectors

    Functions:
        - embed_denial_reasons: Embeds text with model and applies optional caching

    Args:
        texts (List[str]): Input denial_reason strings to encode
        config (Dict): YAML-derived config including model + cache settings

    Example:
        from claimflowengine.clustering.embed import embed_denial_reasons
        embeddings = embed_denial_reasons(df["denial_reason"].tolist(), config)

    Author: ClaimFlowEngine Project
"""

from __future__ import annotations

import os
import joblib
import numpy as np
from typing import List, Dict, Any
from pathlib import Path
from sentence_transformers import SentenceTransformer

from claimflowengine.utils.log import get_logger

logger = get_logger(__name__)


def embed_denial_reasons(texts: List[str], config: Dict[str, Any]) -> np.ndarray:
    """
    Embed a list of denial_reason strings using a sentence transformer model.

    Args:
        texts (List[str]): Denial reason text strings
        config (Dict): Configuration with model and cache parameters

    Returns:
        np.ndarray: Embeddings array (n_samples x embedding_dim)
    """
    model_name = config["clustering"].get("embedding_model", "all-MiniLM-L6-v2")
    cache_path = Path(
        config["clustering"].get("embedding_cache", "data/embeddings.pkl")
    )

    if cache_path.exists():
        logger.info(f"Loading cached embeddings from: {cache_path}")
        return joblib.load(cache_path)

    logger.info(f"Loading sentence transformer model {model_name}")
    model = SentenceTransformer(model_name)

    logger.info(f"Encoding {len(texts)} denial_reason texts...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)

    logger.info(f"Caching embeddings to: {cache_path}")
    os.makedirs(cache_path.parent, exist_ok=True)
    joblib.dump(embeddings, cache_path)

    return embeddings
