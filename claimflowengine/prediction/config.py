"""
Configuration module for denial prediction model training.

Defines constants such as file paths and model save locations.
"""

from pathlib import Path

# Path to processed dataset
DATA_PATH = Path("data/processed_claims.csv")

# Target column
TARGET_COL = "denied"

# Output model path
MODEL_SAVE_PATH = Path("models/denial_model.joblib")

# Transformers  path
TRANSFORMER_SAVE_PATH = Path("models/numeric_transformer.joblib")
