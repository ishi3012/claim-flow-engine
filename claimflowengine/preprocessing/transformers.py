"""
Module: transformers.py

Description:
    Builds a reusable scikit-learn ColumnTransformer pipeline
    for transforming numeric and boolean features during model training and inference.

Features:
- StandardScaler for numeric features
- Imputation and integer coercion for boolean features
- Fully compatible with get_feature_names_out
- Target encoding is handled externally

Functions:
- get_transformer_pipeline(df: pd.DataFrame) -> ColumnTransformer

Author: ClaimFlowEngine Team
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from claimflowengine.utils.functions import convert_to_int
from claimflowengine.utils.logger import get_logger

# Initialize Logging
logger = get_logger("preprocessing")


def get_transformer_pipeline(df: pd.DataFrame) -> ColumnTransformer:
    """
    Creates a column transformer for numeric and boolean input columns only.
    Assumes categorical features have already been target-encoded externally.

    Args:
        df (pd.DataFrame): DataFrame with pre-engineered features

    Returns:
        ColumnTransformer: A pipeline-ready transformer for numeric and boolean columns
    """
    numeric_features = [
        "claim_age_days",
        "note_length",
        "patient_age",
        "total_charge_amount",
        "days_to_submission",
    ]

    boolean_features = [
        "is_resubmission",
        "contains_auth_term",
        "prior_authorization",
        "accident_indicator",
    ]

    numeric_features = [f for f in numeric_features if f in df.columns]
    boolean_features = [f for f in boolean_features if f in df.columns]
    logger.info("Creating preprocessing pipeline...")
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="mean")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "bool",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "to_int",
                            FunctionTransformer(
                                convert_to_int,
                                validate=False,
                                feature_names_out="one-to-one",
                            ),
                        ),
                    ]
                ),
                boolean_features,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    return preprocessor
