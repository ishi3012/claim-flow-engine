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
from sklearn.preprocessing import OneHotEncoder


def get_transformer_pipeline(df: pd.DataFrame) -> ColumnTransformer:
    """
    Creates a column transformer for model input.

    Args:
        df (pd.DataFrame): DataFrame with all engineered features.

    Returns:
        ColumnTransformer: A pipeline-ready transformer.
    """
    categorical_cols = ["payer_id", "provider_type"]
    numerical_cols = ["claim_age_days", "note_length"]
    boolean_cols = ["is_resubmission", "prior_denials_flag", "contains_auth_term"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("numerical", SimpleImputer(strategy="mean"), numerical_cols),
            ("boolean", "passthrough", boolean_cols),
        ]
    )

    return preprocessor
