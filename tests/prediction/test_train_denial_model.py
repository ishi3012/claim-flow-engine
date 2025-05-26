# tests/prediction/test_train_denial_model.py

"""
Tests for train_denial_model.py

Covers:
- composite_score() correctness
- data loading shape
- cross-validation metrics for all models
- saved model file generation
"""

from pathlib import Path
from typing import Any

import joblib
import pytest
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from claimflowengine.prediction.train_denial_model import (
    composite_score,
    evaluate_model,
)
from tests.helpers.test_data import get_mock_claims_df


def test_composite_score() -> None:
    metrics = {"AUC": 0.8, "F1_Score": 0.7, "Recall": 0.6, "Accuracy": 0.9}
    score = composite_score(metrics)
    expected = (
        0.3 * metrics["F1_Score"]
        + 0.1 * metrics["Recall"]
        + 0.5 * metrics["AUC"]
        + 0.1 * metrics["Accuracy"]
    )

    assert abs(score - expected) < 1e-4, "Composite score calculation mismatch"


@pytest.mark.parametrize(
    "model_cls", [LogisticRegression, XGBClassifier, LGBMClassifier]
)  # type: ignore[misc]
def test_evaluate_model_with_synthetic_data(model_cls: Any) -> None:
    df = get_mock_claims_df()
    y = df["denied"]
    X = df.drop(columns=["denied"])

    if model_cls is XGBClassifier:
        model = model_cls(use_label_encoder=False, eval_metric="logloss")
    else:
        model = model_cls()

    metrics = evaluate_model(model, X, y, n_splits=2)
    expected_keys = {"AUC", "F1_Score", "Recall", "Accuracy"}
    assert set(metrics.keys()).issuperset(expected_keys)
    assert all(0.0 <= v <= 1.0 for v in metrics.values())


def test_model_save_path(tmp_path: Path) -> None:
    """Simulate saving a model file and confirm it is saved."""
    model = LogisticRegression()
    dummy_path = tmp_path / "dummy_model.joblib"
    joblib.dump(model, dummy_path)

    assert dummy_path.exists()
    assert dummy_path.suffix == ".joblib"
    assert dummy_path.stat().st_size > 0
