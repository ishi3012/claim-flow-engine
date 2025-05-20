"""
Module: train_denial_model.py

Description:
This script trains and evaluates multiple machine learning models
(Logistic Regression, XGBoost, LightGBM) on engineered healthcare
claims data to predict claim denials.

It performs cross-validation, logs evaluation metrics,
selects the best-performing model based on a composite score,
and serializes the trained model to disk for downstream use
in real-time prediction APIs or batch workflows.

Features:
- Loads data/processed_claims.csv from prior feature pipeline
- Supports multiple model architectures via modular interface
- Evaluates using Stratified K-Fold cross-validation
- Computes and logs AUC, F1, Recall, and Accuracy for each model
- Selects best model using composite scoring heuristic
- Saves best model to models/denial_model.joblib
- Can be run as a script or imported as a module

Intended Use:
- Local experimentation and benchmarking
- Model registration and deployment pipeline entry point
- Integration with Vertex AI Pipelines or FastAPI agent

Usage:
    $ python -m claimflowengine.prediction.train_denial_model

Inputs:
    data/processed_claims.csv  — Features generated in Day 5

Outputs:
    models/denial_model.joblib — Trained and serialized best model

Functions:
    - load_data(): Prepares feature matrix and labels
    - evaluate_model(): Runs cross-validation and scores performance
    - composite_score(): Aggregates metric scores into ranking value
    - main(): Orchestrates training pipeline

Usage:
    python -m claimflowengine.prediction.train_denial_model \
    --data data/processed_claims.csv \
    --transformer models/transformer.joblib \
    --model models/denial_model.joblib
Dependencies:
    - pandas, scikit-learn, xgboost, lightgbm, joblib, logging

Author: ClaimFlowEngine Team
"""

import argparse
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from claimflowengine.prediction.config import TARGET_COL

Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    handlers=[
        logging.FileHandler("logs/train.log", mode="a"),  # or logs/preprocess.log
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def load_data(data_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    logger.info(f"Loading transformed features from: {data_path}")
    df = pd.read_csv(data_path)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset")

    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])

    logger.info(f"Loaded dataset: X shape = {X.shape}, y shape = {y.shape}")
    return X, y


def evaluate_model(
    model: Any, X: pd.DataFrame, y: pd.Series, n_splits: int = 5
) -> Dict[str, float]:
    class_counts = Counter(y)
    n_splits = min(n_splits, min(class_counts.values()))

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs, f1s, recalls, accs = [], [], [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        aucs.append(roc_auc_score(y_test, y_proba))
        f1s.append(f1_score(y_test, y_pred))
        recalls.append(recall_score(y_test, y_pred))
        accs.append(accuracy_score(y_test, y_pred))

        logger.info(
            f"Fold {fold}: AUC={aucs[-1]:.4f}, F1={f1s[-1]:.4f}, "
            f"Recall={recalls[-1]:.4f}, Accuracy={accs[-1]:.4f}"
        )

    return {
        "AUC": np.mean(aucs),
        "F1_Score": np.mean(f1s),
        "Recall": np.mean(recalls),
        "Accuracy": np.mean(accs),
    }


def composite_score(metrics: Dict[str, float]) -> float:
    return (
        0.4 * metrics["AUC"]
        + 0.3 * metrics["F1_Score"]
        + 0.2 * metrics["Recall"]
        + 0.1 * metrics["Accuracy"]
    )


def train_and_save(data_path: str, transformer_path: str, model_path: str) -> None:
    X, y = load_data(data_path)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "XGBoost": XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            max_depth=3,
            n_estimators=50,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=1.0,
        ),
        "LightGBM": LGBMClassifier(
            max_depth=3,
            n_estimators=50,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=1.0,
        ),
    }

    best_model = None
    best_score = -1.0
    best_model_name = ""
    results = {}

    for name, model in models.items():
        logger.info(f"Evaluating model: {name}")
        metrics = evaluate_model(model, X, y)
        score = composite_score(metrics)
        results[name] = metrics
        logger.info(f"{name} — Composite Score: {score:.4f} — {metrics}")

        if score > best_score:
            best_score = score
            best_model = model
            best_model_name = name

    logger.info(f"Best Model: {best_model_name} (Score: {best_score:.4f})")

    if best_model is None:
        raise RuntimeError("Training failed — no model selected.")

    best_model.fit(X, y)

    # Save model
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, model_path)
    logger.info(f"Trained model saved to {model_path}")

    # Confirm transformer exists
    if not Path(transformer_path).exists():
        logger.warning(f"Transformer file not found at {transformer_path}")
    else:
        logger.info(f"Transformer found at {transformer_path}")


# ---------------------- CLI ----------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="claimflowengine/data/processed_claims.csv",
        help="Path to input data CSV",
    )
    parser.add_argument(
        "--transformer",
        default="claimflowengine/models/transformer.joblib",
        help="Path to transformer.pkl",
    )
    parser.add_argument(
        "--model",
        default="claimflowengine/models/denial_model.joblib",
        help="Path to save model.pkl",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_and_save(
        data_path=args.data, transformer_path=args.transformer, model_path=args.model
    )
