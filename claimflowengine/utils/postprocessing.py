"""
Postprocessing utilities for standardizing model outputs.

This module ensures consistency across the pipeline by renaming ambiguous
columns like `denied` into clearly defined names:
    - `denial_label` → historical ground truth
    - `denial_prediction` → model's binary output (bool)
    - `denial_probability` → model's raw score

Example usage:
    from claimflowengine.utils.postprocessing import standardize_prediction_columns
    df = standardize_prediction_columns(df)

Author: ClaimFlowEngine Project (2025)
"""

import pandas as pd
from pandas import Index


def standardize_prediction_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names in prediction output DataFrame.

    Handles:
    - Ambiguous 'denied' columns
    - Preserves existing 'denial_probability'

    Args:
        df (pd.DataFrame): Input dataframe with model outputs

    Returns:
        pd.DataFrame: Cleaned dataframe with renamed columns
    """

    df = df.copy()
    denied_cols = [col for col in df.columns if col.lower() == "denied"]

    # Case: two denied columns (e.g. label + prediction)
    if len(denied_cols) == 2:
        # Get column indices where name == "denied"
        denied_idx = [i for i, col in enumerate(df.columns) if col.lower() == "denied"]

        # Fetch dtype using iloc
        dtype1 = df.iloc[:, denied_idx[0]].dtype
        dtype2 = df.iloc[:, denied_idx[1]].dtype

        new_cols = list(df.columns)

        if dtype1 in [int, float] and dtype2 is bool:
            new_cols[denied_idx[0]] = "denial_label"
            new_cols[denied_idx[1]] = "denial_prediction"
        elif dtype2 in [int, float] and dtype1 is bool:
            new_cols[denied_idx[0]] = "denial_prediction"
            new_cols[denied_idx[1]] = "denial_label"
        else:
            new_cols[denied_idx[0]] = "denial_label"
            new_cols[denied_idx[1]] = "denial_prediction"

        df.columns = Index(new_cols)

    elif len(denied_cols) == 1:
        col = denied_cols[0]
        if df[col].dtype == bool:
            df.rename(columns={col: "denial_prediction"}, inplace=True)
        else:
            df.rename(columns={col: "denial_label"}, inplace=True)
    return df
