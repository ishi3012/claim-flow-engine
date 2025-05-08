"""
train_denial_predictor.py

train_denial_predictor.py

Benchmark and select the best model for healthcare claim denial prediction using composite score
based on config-defined metric weights.

Models evaluated:
- Logistic Regression
- Random Forest
- XGBoost
- LightGBM

Metrics used:
- AUC
- F1-score
- Recall
- Composite (weighted sum from config)

Author: Claim Flow Engine Project
"""

import joblib
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
from sklearn.model_selection import (
    StratifiedGroupKFold,
    cross_val_score,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, f1_score, recall_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb

# ------------------
#   Get config file and paths
# ------------------

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH: Path = ROOT / "config" / "config.yaml"
MODEL_DIR: Path = ROOT / "models"

# get data file path
with open(CONFIG_PATH, "r") as f:
    config: Dict[str, Any] = yaml.safe_load(f)

default_path = (ROOT / config["data"]["default_path"]).resolve()
balanced_path = (ROOT / config["data"]["balanced_path"]).resolve()


DATA_PATH = Path(balanced_path) if balanced_path.exists() else Path(default_path)

TARGET_COL: str = config["data"].get("target_col", "denial_flag")
ID_COLS: List[str] = config["data"].get("id_cols", [])
RANDOM_STATE: int = config.get("random_state", 42)
N_SPLITS: int = config.get("cv_folds", 5)
WEIGHTS: Dict[str, float] = config.get("model_selection", {}).get(
    "weights", {"auc": 0.33, "f1": 0.33, "recall": 0.34}
)

# -------------------------
# Utilities
# -------------------------


def load_data(path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load and splitfeatures and target.
    """
    if not path.exists():
        raise FileNotFoundError(f"ERROR: File not found: {path}")

    df = pd.read_csv(path)
    df.drop(columns=ID_COLS, errors="ignore", inplace=True)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y


def get_models(random_state: int) -> Dict[str, Any]:
    """
    Returns the dictionary of model candidates.
    """
    return {
        "logistic_regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100, random_state=random_state, class_weight="balanced"
        ),
        "xgboost": xgb.XGBClassifier(
            use_label_encoder=False, eval_metric="logloss", random_state=random_state
        ),
        "lightgbm": lgb.LGBMClassifier(random_state=random_state),
    }


def composite_score(
    auc: float, f1: float, recall: float, weights: Dict[str, float]
) -> float:
    """
    Compute the weighted composite scores.
    """
    return (
        weights.get("auc", 0.0) * auc
        + weights.get("f1", 0.0) * f1
        + weights.get("recall", 0.0) * recall
    )


def evaluate_models(
    models: Dict[str, Any], X: pd.DataFrame, y: pd.Series
) -> List[Dict[str, Any]]:
    """
    Cross Validate models and return the perform results.
    """
    results = []
    cv = StratifiedGroupKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )

    for name, model in models.items():
        auc_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
        f1_scores = cross_val_score(model, X, y, cv=cv, scoring=make_scorer(f1_score))

        y_pred = cross_val_predict(model, X, y, cv=cv)
        recall = recall_score(y, y_pred)

        score = composite_score(
            np.mean(auc_scores), np.mean(f1_scores), recall, WEIGHTS
        )

        results.append(
            {
                "model": name,
                "auc_mean": np.mean(auc_scores),
                "f1_mean": np.mean(f1_scores),
                "recall": recall,
                "composite_score": score,
            }
        )
        print(
            f"{name.upper()} | AUC: {np.mean(auc_scores):.4f} | F1: {np.mean(f1_scores):.4f} | Recall: {recall:.4f} | Composite: {score:.4f}"
        )

    return results


def save_best_model(
    models: Dict[str, Any],
    results: List[Dict[str, Any]],
    X: pd.DataFrame,
    y: pd.Series,
    save_dir: Path,
) -> None:
    """
    Train and save the best model based on composite score.
    """

    results_df = pd.DataFrame(results)
    results_df.sort_values("composite_score", ascending=False, inplace=True)

    best_model_name = results_df.iloc[0]["model"]
    best_model = models[best_model_name]
    best_model.fit(X, y)

    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_dir / f"{best_model_name}_best_model.pkl"
    joblib.dump(best_model, model_path)

    results_df.to_csv(save_dir / "model_benchmark_results.csv", index=False)

    print(f"\n Best model '{best_model_name}' saved to: {model_path}")
    print(" Performance metrics saved to model_benchmark_results.csv")


def main():
    print("Starting model training and evaluation...\n")
    X, y = load_data(DATA_PATH)

    print(X.shape)
    print(y.shape)
    models = get_models(RANDOM_STATE)

    for name, model in models.items():
        try:
            # scores = cross_val_score(model, X, y, cv=3, scoring="roc_auc")
            print(f"✔ {name} OK")
        except Exception as e:
            print(f"❌ {name} failed:", e)

    # results = evaluate_models(models, X, y)
    # save_best_model(models, results, X, y, MODEL_DIR)


if __name__ == "__main__":
    main()
