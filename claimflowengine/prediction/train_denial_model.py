import argparse
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from interpret.glassbox import ExplainableBoostingClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from claimflowengine.configs.paths import (
    NUMERICAL_TRANSFORMER_PATH,
    PREDICTION_MODEL_PATH,
    RAW_DATA_PATH,
    TARGET_COL,
)
from claimflowengine.utils.logger import get_logger

# Initialize Logging
logger = get_logger("training")


def load_data(data_path: str) -> Tuple[pd.DataFrame, "pd.Series[Any]"]:
    logger.info(f"Loading transformed features from: {data_path}")
    df = pd.read_csv(data_path)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset")

    y = df[TARGET_COL]
    df = df.drop(columns=[TARGET_COL])
    logger.info(f"Loaded dataset: X shape = {df.shape}, y shape = {y.shape}")
    return df, y


def evaluate_model(
    model: Any, X: pd.DataFrame, y: "pd.Series[Any]", n_splits: int = 5
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

        logger.info(f"Predicted labels: {np.unique(y_pred, return_counts=True)}")
        logger.info(f"Actual labels: {np.unique(y_test, return_counts=True)}")

        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            raw_scores = model.decision_function(X_test)
            y_proba = (raw_scores - raw_scores.min()) / (
                raw_scores.max() - raw_scores.min()
            )
        else:
            y_proba = np.array(y_pred)

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
        0.5 * metrics["AUC"]
        + 0.3 * metrics["F1_Score"]
        + 0.1 * metrics["Recall"]
        + 0.1 * metrics["Accuracy"]
    )


def train_and_save(data_path: str, transformer_path: str, model_path: str) -> None:
    X, y = load_data(data_path)

    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, class_weight="balanced"
        ),
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
        "CatBoost": CatBoostClassifier(
            depth=4, learning_rate=0.05, iterations=100, verbose=0
        ),
        "EBM": ExplainableBoostingClassifier(random_state=42, interactions=0),
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

    for metric in ["AUC", "F1_Score", "Recall", "Accuracy"]:
        sorted_models = sorted(
            results.items(), key=lambda x: x[1][metric], reverse=True
        )
        top_models = [m[0] for m in sorted_models[:3]]
        logger.info(f"Top models by {metric}: {top_models}")

    logger.info(f"Best Model: {best_model_name} (Score: {best_score:.4f})")

    if best_model is None:
        raise RuntimeError("Training failed — no model selected.")

    best_model.fit(X, y)

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, model_path)
    logger.info(f"Trained model saved to {model_path}")

    if not Path(transformer_path).exists():
        logger.warning(f"Transformer file not found at {transformer_path}")
    else:
        logger.info(f"Transformer found at {transformer_path}")


# ---------------------- CLI ----------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=RAW_DATA_PATH,
        help="Path to input data CSV",
    )
    parser.add_argument(
        "--transformer",
        default=NUMERICAL_TRANSFORMER_PATH,
        help="Path to transformer.pkl",
    )
    parser.add_argument(
        "--model",
        default=PREDICTION_MODEL_PATH,
        help="Path to save model.pkl",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_and_save(
        data_path=args.data, transformer_path=args.transformer, model_path=args.model
    )
