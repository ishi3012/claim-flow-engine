"""
Module: predictor.py — Denial Prediction Inference Logic

This module implements the core business logic to:
- Accept raw claim data in various formats
- Apply full preprocessing pipeline
    (text cleaning, feature engineering, encoding)
- Generate predictions:
    denial flag, denial probability, top 3 reasons (optional)

Features:
- Input flexibility: dict, list[dict], or pd.DataFrame
- PII-safe, reusable business logic
- Designed for use in batch scripts or FastAPI

Author: ClaimFlowEngine Team
"""

from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer

from claimflowengine.inference.preprocessor import preprocess_for_inference
from claimflowengine.utils.logger import get_logger

# Initialize Logging
logger = get_logger("inference")


def predict_claims(
    raw_data: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame],
    model: BaseEstimator,
    target_encoder: TargetEncoder,
    numeric_transformer: ColumnTransformer,
) -> List[Dict[str, Any]]:
    """
    Generate denial predictions from raw claim data.

    Args:
        raw_data: Input data (dict, list of dicts, or DataFrame)
        model: Trained denial prediction model
        target_encoder: Fitted target encoder
        numeric_transformer: Fitted numeric/boolean transformer

    Returns:
        List of prediction dicts:
        - denied (bool)
        - denial_probability (float)
        - top_3_denial_reasons (optional, if multiclass)
    """

    # Create Dataframe from the input
    logger.info(f"Create Dataframe from the input: {raw_data}... ")

    if isinstance(raw_data, dict):
        df_input = pd.DataFrame([raw_data])
    elif isinstance(raw_data, list):
        df_input = pd.DataFrame(raw_data)
    elif isinstance(raw_data, pd.DataFrame):
        df_input = raw_data.copy()
    else:
        logger.error(f"# Create Dataframe from the input: {raw_data}... ")
        raise ValueError("Input must be a dict, list of dicts, or DataFrame.")

    # Preprocess features
    logger.info("Preprocess features... ")
    processed_df = preprocess_for_inference(
        df_input, target_encoder, numeric_transformer
    )

    # Predict
    logger.info("Predict denials... ")
    probabilities = model.predict_proba(processed_df)
    is_multiclass = len(probabilities.shape) == 2 and probabilities.shape[1] > 2

    results = []

    for idx, row in df_input.iterrows():
        if is_multiclass:
            prob_array = probabilities[idx]
            top_3_idx = np.argsort(prob_array)[::-1][:3]
            denial_probability = float(np.max(prob_array))
            denied = bool(top_3_idx[0] != 0)
            result = {
                "denied": denied,
                "denial_probability": denial_probability,
                "top_3_denial_reasons": top_3_idx.tolist(),
            }
        else:
            # Binary classification
            denial_probability = float(probabilities[idx][1])
            denied = bool(denial_probability > 0.5)
            result = {
                "denied": denied,
                "denial_probability": denial_probability,
            }
        results.append(result)

    return results
