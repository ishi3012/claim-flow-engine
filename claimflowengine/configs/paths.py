"""
Configuration module for denial prediction model training.

Defines constants such as file paths and model save locations.
"""

from pathlib import Path

# DATA
RAW_DATA_PATH = Path("data/processed_claims.csv")
PROCESSED_DATA_PATH = Path("data/processed_claims.csv")
INFERENCE_INPUT_PATH = Path("data/inference_input.csv")
DENIAL_PREDICTION_OUTPUT_PATH = Path("data/denial_prediction_output.csv")
CLUSTERING_OUTPUT_PATH = Path("data/clustering_output.csv")
CLUSTERING_CLAIMS_LABELED_PATH = Path("data/clustered_claims_labeled.csv")
CLUSTER_LABELS_JSON_PATH = Path("data/cluster_labels.json")
# Target column
TARGET_COL = "denied"

# Sentence Transformer Model Name
SENTENCE_TRANSFORMER_MODEL_NAME = "all-MiniLM-L6-v2"

# Prediction model path
PREDICTION_MODEL_PATH = Path("models/denial_prediction_model.joblib")

# Transformers  path
NUMERICAL_TRANSFORMER_PATH = Path("models/numeric_transformer.joblib")
TARGET_ENCODER_PATH = Path("models/target_encoder.joblib")

# Cluster model path
CLUSTER_MODEL_PATH = Path("models/cluster_model.joblib")
REDUCER_MODEL_PATH = Path("models/reducer_model.joblib")
