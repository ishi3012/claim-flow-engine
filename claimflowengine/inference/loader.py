"""
Module: loader.py — Model Loader for Denial Prediction Inference

This module provides functions to load the trained model
and optional transformer (preprocessor) used during training.
Designed for use in both script and API-based
prediction workflows.

Features:
- Loads `model.pkl` and `transformer.pkl` from disk
- Supports customizable model and transformer paths
- Ensures modularity for unit testing and reuse

Intended Use:
- Import into predict_denials.py or FastAPI for inference
- Call `load_model()` once during initialization

Inputs:
- model_path (optional):
    Path to saved model file (default: models/model.pkl)
- transformer_path (optional):
    Path to transformer file (default: models/transformer.pkl)

Outputs:
- model: Trained ML model
- transformer: Optional preprocessing transformer (or None)

Author: ClaimFlowEngine Project (2025)
"""

import os
from pathlib import Path
from typing import Any, Tuple, Union

import joblib

from claimflowengine.configs.paths import (
    NUMERICAL_TRANSFORMER_PATH,
    PREDICTION_MODEL_PATH,
    TARGET_ENCODER_PATH,
)
from claimflowengine.utils.logger import get_logger

# Initialize Logging
logger = get_logger("inference")


def load_model(
    model_path: Union[str, Path] = PREDICTION_MODEL_PATH,
    target_encoder_path: Union[str, Path] = TARGET_ENCODER_PATH,
    numerical_transformer_path: Union[str, Path] = NUMERICAL_TRANSFORMER_PATH,
) -> Tuple[Any, Any, Any]:
    """
    Load the denial model and required transformers.

    Returns:
        Tuple containing:
        - Trained model
        - TargetEncoder for categoricals
        - Numeric/Boolean ColumnTransformer
    """

    if not os.path.exists(model_path):
        logger.error(f"FileNotFoundError : {model_path}")
        raise FileNotFoundError(f"Model file not found at {model_path}")
    if not os.path.exists(target_encoder_path):
        logger.error(f"FileNotFoundError : {target_encoder_path}")
        raise FileNotFoundError(f"Target encoder not found at {target_encoder_path}")
    if not os.path.exists(numerical_transformer_path):
        logger.error(f"FileNotFoundError : {numerical_transformer_path}")
        raise FileNotFoundError(
            f"Numeric transformer not found at {numerical_transformer_path}"
        )

    logger.info("Loading trained models...")
    model = joblib.load(model_path)
    target_encoder = joblib.load(target_encoder_path)
    numeric_tranformer = joblib.load(numerical_transformer_path)

    return model, target_encoder, numeric_tranformer
