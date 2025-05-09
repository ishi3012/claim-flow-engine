"""
train.py

Train a claim denial prediction model using config-driven settings and
a modular feature pipeline. Supports multiple models and outputs metrics,
trained model artifact, and experiment logs.

Intended to be wrapped later as a @kfp.component for Vertex AI Pipelines.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from pathlib import Path
from typing import Dict, Tuple, Union

from claimflowengine.utils.paths import MODEL_DIR, LOG_DIR, DATA_DIR
from claimflowengine.preprocessing.features import (
    replace_none_with_nan,
    preprocess_dates,
    build_feature_pipeline,
)
from claimflowengine.utils.log import get_logger

# Initialize logger
logger = get_logger(__name__)


def load_and_preprocess(data_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load a CSV file and apply preprocessing steps:
    - Replace None with np.nan
    - Convert datetime columns to 'days_since_*'
    - Log shape and missing values

    Args:
        data_path (Union[str, Path]): Path to input CSV file

    Returns:
        pd.DataFrame: Preprocessed DataFrame
    """
    data_path = Path(data_path)

    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        logger.exception(f"Data file not found: {data_path}")
        raise

    logger.info(f"Loaded data from: {data_path}")
    logger.info(f"Shape: {df.shape}")

    df = replace_none_with_nan(df)
    df = preprocess_dates(df)

    nulls = df.isna().sum()[df.isna().sum() > 0]

    if not nulls.empty:
        logger.warning(f"Missing values:\n{nulls}")

    return df


def train_model(config: Dict) -> Tuple[object, Dict]:
    """
    Train and evaluate a model using the provided config.

    Args:
        config (Dict) : Configuration dictionary loaded from config.YAML file.
    Returns:
        Tuple: (trained model object, metrics dictionary)
    """

    # Read Data paths from Config file
    default_path = (DATA_DIR / config["data"]["default_path"]).resolve()
    balanced_path = (DATA_DIR / config["data"]["balanced_path"]).resolve()
    data_path = balanced_path if balanced_path.exists() else default_path

    # Load and process data file
    df = load_and_preprocess(data_path)
    target_col = config["data"]["target_col"]
    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in dataset. Columns: {df.columns.tolist()}"
        )
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["train"].get("test_size", 0.2),
        random_state=config["train"].get("random_state", 42),
    )
    logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # Feature pipeline
    feature_pipeline = build_feature_pipeline(X_train)
    X_train_transformed = feature_pipeline.fit_transform(X_train)
    X_test_transformed = feature_pipeline.transform(X_test)
    X_train_transformed = np.nan_to_num(X_train_transformed)
    X_test_transformed = np.nan_to_num(X_test_transformed)

    # Model Selection
    model_name = config["train"]["model"]

    if model_name == "logreg":
        model = LogisticRegression(max_iter=1000)

    elif model_name == "xgb":
        model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")

    elif model_name == "lgbm":
        model = LGBMClassifier()
    else:
        logger.exception(f"Unsupported model: {model_name}")
        raise ValueError(f"Unsupported model: {model_name}")

    # Fit the model
    model.fit(X_train_transformed, y_train)

    # Evaluation
    y_pred = model.predict(X_test_transformed)
    y_proba = model.predict_proba(X_test_transformed)[:, 1]

    metrics = {
        "model": model_name,
        "auc": roc_auc_score(y_test, y_proba),
        "f1": f1_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    logger.info(f"AUC: {metrics['auc']:.4f}, F1: {metrics['f1']:.4f}")
    logger.debug(f"Confusion matrix:\n{metrics['confusion_matrix']}")

    # Save model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump((feature_pipeline, model), MODEL_DIR / "model.pkl")
    logger.info(f"Model saved to {MODEL_DIR / 'model.pkl'}")

    # Log metrics
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_df = pd.DataFrame([metrics])
    log_path = LOG_DIR / "experiments.csv"

    if log_path.exists():
        log_df.to_csv(log_path, mode="a", header=False, index=False)
    else:
        log_df.to_csv(log_path, index=False)

    logger.info(f"Metrics logged to {log_path}")

    return model, metrics
