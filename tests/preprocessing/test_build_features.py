from pathlib import Path

import pandas as pd

from app.schemas import ClaimInput
from claimflowengine.preprocessing.build_features import preprocess_and_save
from claimflowengine.preprocessing.feature_engineering import engineer_features
from claimflowengine.preprocessing.text_cleaning import clean_text_fields
from claimflowengine.preprocessing.transformers import get_transformer_pipeline


def test_preprocess_and_save_runs(tmp_path: Path) -> None:
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
            "service_date": ["2022-12-28"],
            "total_charge_amount": [1000.0],
            "patient_dob": ["1980-01-01"],
            "denied": [1],
        }
    )
    dummy_data["resubmission"] = dummy_data["resubmission"].astype(int)
    dummy_data["prior_authorization"] = 1
    dummy_data["accident_indicator"] = 0

    raw_path = tmp_path / "raw.csv"
    out_path = tmp_path / "processed.csv"
    transformer_path = tmp_path / "transformer.joblib"
    dummy_data.to_csv(raw_path, index=False)

    preprocess_and_save(str(raw_path), str(out_path), str(transformer_path))
    processed = pd.read_csv(out_path)

    assert not processed.empty
    assert any("claim_age_days" in col for col in processed.columns)
    assert any("contains_auth_term" in col for col in processed.columns)


def test_preprocess_and_save_with_edi_schema(tmp_path: Path) -> None:
    edi_data = pd.DataFrame(
        {
            "claim_id": [2],
            "patient_gender": ["F"],
            "patient_dob": ["1980-05-20"],
            "service_date": ["2023-04-01"],
            "billing_provider_specialty": ["Cardiology"],
            "facility_code": ["123"],
            "provider_type": ["Clinic"],
            "claim_type": ["Institutional"],
            "total_charge_amount": [1200.0],
            "prior_authorization": ["Y"],
            "accident_indicator": ["N"],
            "diagnosis_code_primary": ["I10"],
            "procedure_code": ["99213"],
            "payer_name": ["BlueCross"],
            "payer_id": ["BC001"],
            "submission_date": ["2023-03-25"],
            "denial_date": ["2023-04-05"],
            "denial_reason": ["Auth required"],
            "resubmission": [0],
            "followup_notes": ["Submitted twice"],
            "denied": [0],
        }
    )

    raw_path = tmp_path / "edi_raw.csv"
    out_path = tmp_path / "edi_processed.csv"
    edi_data.to_csv(raw_path, index=False)
    transformer_path = tmp_path / "transformer.joblib"

    preprocess_and_save(str(raw_path), str(out_path), str(transformer_path))
    processed = pd.read_csv(out_path)

    assert not processed.empty
    assert any("patient_age" in col for col in processed.columns)
    assert any("prior_authorization" in col for col in processed.columns)
    assert any("claim_age_days" in col for col in processed.columns)
    assert processed.columns.is_unique


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
            "denial_reason": ["auth required"],
            "resubmission": [1],
            "followup_notes_clean": ["call made to payer"],
            "denial_reason_clean": ["auth required"],
            "service_date": ["2022-12-30"],
            "denied": [1],
        }
    )
    result = engineer_features(df)
    assert "claim_age_days" in result.columns
    assert "followup_intensity_score" in result.columns

    assert result["contains_auth_term"].iloc[0]


def test_transformer_pipeline_output_shape() -> None:
    df = pd.DataFrame(
        {
            "payer_id": ["P123", "P234"],
            "provider_type": ["Clinic", "Hospital"],
            "claim_age_days": [5, 10],
            "note_length": [12, 18],
            "patient_age": [43, 51],
            "total_charge_amount": [1200.0, 900.0],
            "days_to_submission": [3, 5],
            "is_resubmission": [1, 0],
            "contains_auth_term": [1, 0],
            "prior_authorization": [1, 0],
            "accident_indicator": [0, 1],
        }
    )

    pipeline = get_transformer_pipeline(df)
    transformed = pipeline.fit_transform(df)
    assert transformed.shape[0] == 2


def test_claim_input_schema_validates() -> None:
    sample = {
        "patient_age": 45,
        "gender": "F",
        "provider_type": "Clinic",
        "billing_provider_specialty": "Oncology",
        "claim_type": "institutional",
        "diagnosis_code": "I10",
        "procedure_code": "99213",
        "facility_code": "123",
        "claim_age_days": 20,
        "days_to_submission": 5,
        "total_charge_amount": 1500.0,
        "payer_id": "PAYER123",
        "plan_type": "medicare",
        "place_of_service": "11",
        "contains_auth_term": False,
        "note_length": 120,
        "prior_authorization": 1,
        "accident_indicator": 0,
        "prior_denials_flag": 1,
        "is_resubmission": 1,
    }
    validated = ClaimInput(**sample)
    assert validated.gender == "F"


def test_text_cleaning_handles_nulls() -> None:
    df = pd.DataFrame({"denial_reason": [None], "followup_notes": [None]})
    cleaned = clean_text_fields(df)
    assert "denial_reason_clean" in cleaned.columns
    assert cleaned["denial_reason_clean"].iloc[0] == ""
