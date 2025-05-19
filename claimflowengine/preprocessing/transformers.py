"""
Module: transformers.py

Description:
    Builds a reusable scikit-learn ColumnTransformer pipeline
    for model training and inference.

Features:
- OneHotEncoder for categorical features (payer_id, provider_type, etc.)
- StandardScaler for numeric features (age, amount, etc.)
- Passthrough for boolean flags (prior_denials_flag, etc.)
- Handles missing values with SimpleImputer
- Ensures all output is dense (sparse_threshold=0.0)

Functions:
- get_transformer_pipeline(df: pd.DataFrame) -> ColumnTransformer

Author: ClaimFlowEngine Team
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


def get_transformer_pipeline(df: pd.DataFrame) -> ColumnTransformer:
    """
    Creates a column transformer for model input.

    Args:
        df (pd.DataFrame): DataFrame with all engineered features.

    Returns:
        ColumnTransformer: A pipeline-ready transformer
        with dense numeric output.
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
        "prior_denials_flag",
        "contains_auth_term",
        "prior_authorization",
        "accident_indicator",
    ]

    categorical_features = [
        "payer_id",
        "provider_type",
        "plan_type",
        "claim_type",
        "billing_provider_specialty",
        "facility_code",
        "diagnosis_code",
        "procedure_code",
    ]

    numeric_features = [f for f in numeric_features if f in df.columns]
    boolean_features = [f for f in boolean_features if f in df.columns]
    categorical_features = [f for f in categorical_features if f in df.columns]

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
                                lambda x: x.astype(int),
                                validate=False,
                                feature_names_out="one-to-one",
                            ),
                        ),
                    ]
                ),
                boolean_features,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )

    return preprocessor
