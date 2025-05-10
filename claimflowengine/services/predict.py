"""
Module: predict.py

Description:
    Denial prediction service logic. Loads the trained model and computes
    denial probability and top 3 predicted reasons.

Inputs:
    - claim_features (dict): Dictionary of preprocessed features

Outputs:
    - dict: Contains:
        - denied (bool)
        - denial_prob (float)
        - top_reasons (List of (reason, prob))

Author: ClaimFlowEngine Team
"""

from __future__ import annotations

import joblib
import numpy as np

from claimflowengine.utils.config_loader import load_config
from claimflowengine.utils.log import get_logger

# initialize logger
logger = get_logger(__name__)

# load config YAML
config = load_config()

_model = None


def load_model():
    global _model
    if _model is None:
        model_path = config["models"].get("model_path", "models/denial_model.joblib")
        logger.info(f"Loading model from {model_path}")
        _model = joblib.load(model_path)
    return _model


def predict_denial(claim_features: dict) -> dict:
    logger.info("Running denial prediction")
    model = load_model()
    input_vec = np.array([list(claim_features.values())])
    proba = model.predict_proba(input_vec)[0]
    classes = model.classes_

    top_indices = np.argsort(proba)[::-1][:3]
    top_reasons = [(classes[i], float(proba[i])) for i in top_indices]

    return {
        "denied": bool(np.argmax(proba)),
        "denial_prob": float(max(proba)),
        "top_reasons": top_reasons,
    }
