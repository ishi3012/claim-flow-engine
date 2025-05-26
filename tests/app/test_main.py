# tests/test_main.py

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_predict_endpoint_mock() -> None:
    payload = {
        "patient_age": 34,
        "gender": "F",
        "provider_type": "Clinic",
        "billing_provider_specialty": "Internal Medicine",
        "claim_type": "professional",
        "diagnosis_code": "E11.9",
        "procedure_code": "99213",
        "facility_code": "11",
        "claim_age_days": 15,
        "days_to_submission": 5,
        "total_charge_amount": 150.0,
        "payer_id": "PAYER001",
        "plan_type": "commercial",
        "place_of_service": "11",
        "contains_auth_term": True,
        "note_length": 80,
        "prior_authorization": 1,
        "accident_indicator": 0,
        "prior_denials_flag": 0,
        "is_resubmission": 0,
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "denial_probability" in body
    assert "top_denial_reasons" in body
    assert "model_version" in body
    assert isinstance(body["denial_probability"], float)
    assert isinstance(body["top_denial_reasons"], list)
    assert isinstance(body["model_version"], str)
