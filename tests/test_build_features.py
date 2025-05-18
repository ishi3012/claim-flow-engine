# tests/test_build_features.py

from pathlib import Path

import pandas as pd

from claimflowengine.preprocessing.build_features import preprocess_and_save
from claimflowengine.preprocessing.features import engineer_features
from claimflowengine.preprocessing.text_cleaning import clean_text_fields
from claimflowengine.preprocessing.transformers import get_transformer_pipeline


def test_preprocess_and_save_runs(tmp_path: Path) -> None:
    """E2E test for build_features.py pipeline"""
    dummy_data = pd.DataFrame(
        {
            "claim_id": [1],
            "submission_date": ["2023-01-01"],
            "denial_date": ["2023-01-05"],
            "denial_reason": ["Authorization pending"],
            "payer_id": ["P123"],
            "provider_type": ["Clinic"],
            "resubmission": [1],
            "followup_notes": ["Call made to payer"],
        }
    )
    raw_path = tmp_path / "raw.csv"
    out_path = tmp_path / "processed.csv"
    dummy_data.to_csv(raw_path, index=False)

    preprocess_and_save(str(raw_path), str(out_path))
    processed = pd.read_csv(out_path)

    assert not processed.empty
    assert "numerical__claim_age_days" in processed.columns
    assert "boolean__contains_auth_term" in processed.columns


def test_text_cleaning_simple_case() -> None:
    df = pd.DataFrame(
        {
            "denial_reason": ["Authorization pending", "MEDICAL NECESSITY"],
            "followup_notes": ["Call made. Followed up!", "Refile due to error."],
        }
    )
    cleaned = clean_text_fields(df)
    assert "denial_reason_clean" in cleaned.columns
    assert cleaned["denial_reason_clean"].iloc[0] == "auth pending"


def test_engineer_features_completes() -> None:
    df = pd.DataFrame(
        {
            "submission_date": ["2023-01-01"],
            "denial_date": ["2023-01-03"],
            "denial_reason": ["Authorization required"],
            "resubmission": [1],
            "followup_notes_clean": ["call made to payer"],
            "denial_reason_clean": ["authorization required"],
        }
    )
    df = clean_text_fields(df)
    result = engineer_features(df)
    assert "claim_age_days" in result.columns
    assert result["contains_auth_term"].iloc[0]


def test_transformer_pipeline_output_shape() -> None:
    df = pd.DataFrame(
        {
            "payer_id": ["P123", "P234"],
            "provider_type": ["Clinic", "Hospital"],
            "claim_age_days": [5, 10],
            "note_length": [12, 18],
            "is_resubmission": [True, False],
            "prior_denials_flag": [True, True],
            "contains_auth_term": [True, False],
        }
    )
    pipeline = get_transformer_pipeline(df)
    transformed = pipeline.fit_transform(df)
    assert transformed.shape[0] == 2
