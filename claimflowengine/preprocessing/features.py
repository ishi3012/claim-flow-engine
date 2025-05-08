import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from pandas.api.types import is_datetime64_any_dtype


def replace_none_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replaces Python None values with np.nan to ensure compatibility with scikit-learn transformers.
    """
    return df.replace({None: np.nan})


def preprocess_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts any column that can be parsed as datetime into days-since features.
    Drops the original datetime columns.
    """
    today = pd.Timestamp.today()

    for col in df.columns:
        if not is_datetime64_any_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col], errors="raise")
            except Exception:
                continue  # Skip non-date columns

        if is_datetime64_any_dtype(df[col]):
            df[f"days_since_{col}"] = (today - df[col]).dt.days
            df.drop(columns=[col], inplace=True)

    return df


def get_feature_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Returns lists of numeric and categorical feature names.
    """
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    return numeric_cols, categorical_cols


def build_feature_pipeline(X: pd.DataFrame) -> ColumnTransformer:
    """
    Builds a preprocessing pipeline with separate handling for numeric and categorical features.
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
