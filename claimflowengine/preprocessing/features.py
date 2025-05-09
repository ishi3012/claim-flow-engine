"""
features.py

Feature engineering module for the ClaimFlowEngine project.

This script provides preprocessing utilities tailored for structured healthcare claims data,
preparing features for use in downstream denial prediction models. It handles:
    • Date normalization into 'days since' format.
    • Missing value imputation.
    • Categorical encoding and numeric scaling via a unified scikit-learn pipeline.
    • Automatic feature type detection (numeric vs categorical).

Key Functions:
    - replace_none_with_nan: Ensures compatibility with sklearn by replacing None with np.nan.
    - preprocess_dates: Converts datetime columns into elapsed time features.
    - get_feature_types: Splits DataFrame columns into numeric and categorical.
    - build_feature_pipeline: Constructs a ColumnTransformer for ML-ready preprocessing.
    - describe_features: Prints summary statistics on feature types and missing values.

Intended Usage:
    - Called during preprocessing of ML training and inference pipelines.
    - Integrated with `train_denial_predictor.py` and related model components.

Example:
    >>> X = replace_none_with_nan(df)
    >>> X = preprocess_dates(X)
    >>> pipeline = build_feature_pipeline(X)
    >>> X_transformed = pipeline.fit_transform(X)

Author: ClaimFlowEngine Project
"""

import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def replace_none_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace None values in a DataFrame with np.nan for sklearn compatibility.

    Args:
        df (pd.DataFrame): Input DataFrame possibly containing None values.

    Returns:
        pd.DataFrame: DataFrame with None values replaced by np.nan.
    """
    return df.replace({None: np.nan})


def preprocess_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert known datetime columns into 'days since' features.
    Only applies to columns that are likely dates.
    """
    today = pd.Timestamp.today()

    # List of columns you *know* are dates
    candidate_date_cols = [
        col for col in df.columns if "date" in col.lower() or "dob" in col.lower()
    ]

    for col in candidate_date_cols:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col], errors="raise")
                df[f"days_since_{col}"] = (today - df[col]).dt.days
                df.drop(columns=[col], inplace=True)
            except Exception:
                continue  # If conversion fails, move on

    return df


def get_feature_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Identify numeric and categorical feature columns in a DataFrame.

    Args:
        X (pd.DataFrame): Input DataFrame of features.

    Returns:
        Tuple[List[str], List[str]]: A tuple containing two lists:
            - List of numeric column names (int64, float64).
            - List of categorical column names (object, category).
    """
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    return numeric_cols, categorical_cols


def build_feature_pipeline(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build a preprocessing pipeline for numeric and categorical features.

    Args:
        X (pd.DataFrame): The input feature matrix.
        numeric_cols (List[str], optional): List of numeric columns. Auto-inferred if None.
        categorical_cols (List[str], optional): List of categorical columns. Auto-inferred if None.

    Returns:
        ColumnTransformer: A scikit-learn transformer for preprocessing.
    """
    numeric_cols, categorical_cols = get_feature_types(X)

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
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )
    return preprocessor


def describe_features(X: pd.DataFrame) -> None:
    """
    Prints a summary of feature types and missing values.
    """
    numeric_cols, categorical_cols = get_feature_types(X)
    print(f"Numeric features: {numeric_cols}")
    print(f"Categorical features: {categorical_cols}")
    print("Missing values per column:\n", X.isna().sum())
