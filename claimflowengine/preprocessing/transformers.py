"""
    Module: transformers.py

    Description:
        Builds a reusable scikit-learn ColumnTransformer pipeline for modeling.

    Features:
    - OneHotEncoder for categoricals
    - Passthrough for numeric and boolean features
    - Handles unknown categories in inference

    Functions:
    - get_transformer_pipeline(df: pd.DataFrame) -> ColumnTransformer

    Author: ClaimFlowEngine Team
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def get_transformer_pipeline(df: pd.DataFrame) -> ColumnTransformer:
    """
    Creates a column transformer for model input.

    Args:
        df (pd.DataFrame): DataFrame with all engineered features.

    Returns:
        ColumnTransformer: A pipeline-ready transformer.
    """
    NUMERIC_FEATURES_ALL = [
        "claim_age_days",
        "note_length",
        "patient_age",
        "total_charge_amount",
        "days_to_submission",
    ]

    BOOLEAN_FEATURES_ALL = [
        "is_resubmission",
        "prior_denials_flag",
        "contains_auth_term",
        "prior_authorization",
        "accident_indicator",
    ]

    CATEGORICAL_FEATURES_ALL = [
        "payer_id",
        "provider_type",
    ]
    categorical_features = [f for f in NUMERIC_FEATURES_ALL if f in df.columns]
    numerical_features = [f for f in BOOLEAN_FEATURES_ALL if f in df.columns]
    boolean_features = [f for f in CATEGORICAL_FEATURES_ALL if f in df.columns]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="mean")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numerical_features,
            ),
            (
                "boolean",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                    ]
                ),
                boolean_features,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    return preprocessor
