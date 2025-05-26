import pandas as pd
from pandas import Index

from claimflowengine.utils.postprocessing import standardize_prediction_columns


def test_single_denied_column_label() -> None:
    df = pd.DataFrame({"denied": [0, 1, 0]})
    df_out = standardize_prediction_columns(df)
    assert "denial_label" in df_out.columns
    assert "denied" not in df_out.columns


def test_single_denied_column_prediction() -> None:
    df = pd.DataFrame({"denied": [True, False, True]})
    df_out = standardize_prediction_columns(df)
    assert "denial_prediction" in df_out.columns
    assert "denied" not in df_out.columns


def test_two_denied_columns() -> None:
    df = pd.DataFrame(
        {
            "denied": [0, 1, 0],
            "denied.1": [True, False, False],
            "denial_probability": [0.9, 0.2, 0.3],
        }
    )
    df.columns = Index(["denied", "denied", "denial_probability"])  # simulate duplicate
    df_out = standardize_prediction_columns(df)
    assert "denial_label" in df_out.columns
    assert "denial_prediction" in df_out.columns
    assert "denied" not in df_out.columns


def test_preserves_probability() -> None:
    df = pd.DataFrame({"denied": [0, 1], "denial_probability": [0.7, 0.9]})
    df_out = standardize_prediction_columns(df)
    assert "denial_probability" in df_out.columns
