"""
claimflowengine.preprocessing

This module contains reusable, production-ready preprocessing functions for:
- Feature typing (numeric vs categorical)
- Imputation
- Date conversions (e.g., days_since_X)
- Pipeline construction for sklearn/XGBoost/Vertex compatibility

Used across both local development and Vertex AI pipeline deployments.
"""

from .features import (
    build_feature_pipeline,
    preprocess_dates,
    replace_none_with_nan,
    get_feature_types,
    describe_features,
)

__all__ = [
    "build_feature_pipeline",
    "preprocess_dates",
    "replace_none_with_nan",
    "get_feature_types",
    "describe_features",
]
