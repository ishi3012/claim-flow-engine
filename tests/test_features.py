import pandas as pd

from claimflowengine.preprocessing import (
    replace_none_with_nan,
    preprocess_dates,
    build_feature_pipeline,
    get_feature_types,
)


def test_replace_none_with_nan():
    df = pd.DataFrame({"a": [1, None, 3], "b": ["x", None, "z"]})
    df_clean = replace_none_with_nan(df)
    assert df_clean.isna().sum().sum() == 2
    assert df_clean.isna().equals(
        pd.DataFrame({"a": [False, True, False], "b": [False, True, False]})
    )


def test_preprocess_dates_generic():
    df = pd.DataFrame({"procedure_date": ["2020-01-01", "2021-06-15", None]})

    df_out = preprocess_dates(df.copy())
    assert "days_since_procedure_date" in df_out.columns
    assert "procedure_date" not in df_out.columns
    assert df_out["days_since_procedure_date"].notna().sum() == 2


def test_build_feature_pipeline_runs():
    df = pd.DataFrame(
        {
            "amount": [100.0, 200.0, None],
            "category": ["A", "B", None],
            "days_since_procedure_date": [123, 456, 789],
        }
    )
    df = replace_none_with_nan(df)
    pipeline = build_feature_pipeline(df)
    output = pipeline.fit_transform(df)
    assert output.shape[0] == df.shape[0]  # rows match
    assert output.shape[1] > 0  # some features exist


def test_get_feature_types_separates_correctly():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [1.1, 2.2, 3.3]})
    num, cat = get_feature_types(df)
    assert "a" in num and "c" in num
    assert "b" in cat
