"""
Configuration module for denial prediction model training.

Defines constants such as file paths and model save locations.
"""

from pathlib import Path

# DATA
RAW_DATA_PATH = Path("data/processed_claims.csv")
PROCESSED_DATA_PATH = Path("data/processed_claims.csv")
INFERENCE_INPUT_PATH = Path("data/inference_input.csv")
INFERENCE_OUTPUT_PATH = Path("data/inference_output.csv")

# Target column
TARGET_COL = "denied"

# Prediction model path
PREDICTION_MODEL_PATH = Path("models/denial_prediction_model.joblib")

# Transformers  path
NUMERICAL_TRANSFORMER_PATH = Path("models/numeric_transformer.joblib")
TARGET_ENCODER_PATH = Path("models/target_encoder.joblib")
