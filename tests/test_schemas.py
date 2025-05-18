# tests/test_schemas.py

from app.schemas import ClaimInput, ClaimPredictionResponse


def test_claim_input_schema_instantiation() -> None:
    sample = ClaimInput(
        claim_id="CLM123",
        patient_age=45,
        patient_gender="F",
        diagnosis_codes=["I10", "E11.9"],
        procedure_codes=["99213", "80050"],
        provider_npi="1234567890",
        payer_id="PAYER001",
    )
    assert sample.claim_id == "CLM123"
    assert sample.patient_age == 45


def test_claim_prediction_response_schema() -> None:
    response = ClaimPredictionResponse(
        denial_probability=0.85,
        top_denial_reasons=["coding error", "prior auth missing"],
        model_version="v1.0",
    )
    assert response.denial_probability == 0.85
    assert len(response.top_denial_reasons) == 2
