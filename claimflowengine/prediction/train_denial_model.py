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

Dependencies:
    - pandas, scikit-learn, xgboost, lightgbm, joblib, logging

Author: ClaimFlowEngine Team
"""

import logging
from collections import Counter
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from claimflowengine.prediction.config import DATA_PATH, MODEL_SAVE_PATH, TARGET_COL

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format="%(acstime)s — %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)


def load_data() -> Tuple[pd.DataFrame, pd.Series]:
    logger.info(f"Loading data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    DROP_COLS = [
        "claim_id",
        "patient_id",
        "denial_date",
        "denial_reason",
        "followup_notes",
        "appeal_outcome",
        "denial_code",
    ]

    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL] + DROP_COLS, errors="ignore")

    logger.info(f"Loaded dataset shape: X={X.shape}, y={y.shape}")

    return X, y


def evaluate_model(
    model: Any, X: pd.DataFrame, y: pd.Series, n_splits: int = 5
) -> Dict[str, float]:
    # Dynamically adjust folds to class counts
    class_counts = Counter(y)
    min_class_count = min(class_counts.values())
    n_splits = min(n_splits, min_class_count)

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs, f1_scores, recalls, accuracies = [], [], [], []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X, y), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        aucs.append(roc_auc_score(y_test, y_proba))
        f1_scores.append(f1_score(y_test, y_pred))
        recalls.append(recall_score(y_test, y_pred))
        accuracies.append(accuracy_score(y_test, y_pred))

        logger.info(
            f"Fold {fold}: AUC={aucs[-1]:.4f}, F1={f1_scores[-1]:.4f}, "
            f"Recall={recalls[-1]:.4f}, Accuracy={accuracies[-1]:.4f}"
        )

    return {
        "AUC": np.mean(aucs),
        "F1_Score": np.mean(f1_scores),
        "Recall": np.mean(recalls),
        "Accuracy": np.mean(accuracies),
    }


def composite_score(metrics: Dict[str, float]) -> float:
    return (
        0.4 * metrics["AUC"]
        + 0.3 * metrics["F1_Score"]
        + 0.2 * metrics["Recall"]
        + 0.1 * metrics["Accuracy"]
    )


def main() -> None:
    X, y = load_data()

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
        "LightGBM": LGBMClassifier(),
    }

    best_model_name = ""
    best_score = -1.0
    best_model = None

    for name, model in models.items():
        logger.info(f"Training and evaluating model: {name}")
        metrics = evaluate_model(model, X, y)
        score = composite_score(metrics)
        logger.info(f"{name} — Composite Score: {score:.4f}, Metrics: {metrics}")

        if score > best_score:
            best_score = score
            best_model_name = name
            best_model = model

    logger.info(f"Best model: {best_model_name} with score {best_score:.4f}")

    if best_model is None:
        raise RuntimeError("No valid model was selected during training.")

    best_model.fit(X, y)
    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_SAVE_PATH)

    logger.info(f"Final model saved to:{MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
