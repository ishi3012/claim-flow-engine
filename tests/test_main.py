# tests/test_main.py

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_predict_endpoint_mock() -> None:
    payload = {
        "claim_id": "CLM123",
        "patient_age": 34,
        "patient_gender": "F",
        "diagnosis_codes": ["E11.9", "I10"],
        "procedure_codes": ["99213", "81001"],
        "provider_npi": "1234567890",
        "payer_id": "PAYER001",
        "service_location": "IL",
        "previous_denial": False,
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
