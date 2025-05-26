# tests/test_schemas.py

from app.schemas import ClaimInput, ClaimPredictionResponse


def test_claim_input_schema_instantiation() -> None:
    sample = ClaimInput(
        patient_age=45,
        gender="F",
        provider_type="Hospital",
        billing_provider_specialty="Oncology",
        claim_type="inpatient",
        diagnosis_code="I10",
        procedure_code="99213",
        facility_code="01",
        claim_age_days=20,
        days_to_submission=3,
        total_charge_amount=1500.0,
        payer_id="PAYER001",
        plan_type="medicare",
        prior_authorization=1,
        accident_indicator=0,
        prior_denials_flag=1,
        is_resubmission=0,
        contains_auth_term=True,
        note_length=120,
        place_of_service="11",
    )

    assert sample.patient_age == 45


def test_claim_prediction_response_schema() -> None:
    response = ClaimPredictionResponse(
        denial_probability=0.85,
        top_denial_reasons=["coding error", "prior auth missing"],
        model_version="v1.0",
        routing_cluster_id="auth_required",
        explainability_scores={"feature1": 0.5},
    )
    assert response.denial_probability == 0.85
    assert len(response.top_denial_reasons) == 2
