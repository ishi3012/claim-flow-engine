"""
Test suite for FastAPI health check endpoint (/ping).

This test ensures that the FastAPI app is running and the
/ping endpoint returns a 200 status with expected JSON payload.

Author: ClaimFlowEngine Team
"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_ping_returns_200_and_message() -> None:
    response = client.get("/ping")
    assert response.status_code == 200
    assert response.json() == {"message": "pong"}
