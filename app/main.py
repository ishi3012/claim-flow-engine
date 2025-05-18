"""
    Module: app.main

    Main FastAPI application module for ClaimFlowEngine.

    This module initializes the FastAPI app and provides a health check endpoint.

    Features:
    - Initializes FastAPI app instance
    - Defines a root /ping endpoint for health monitoring

    Intended Use:
    - Importable by ASGI servers (e.g., Uvicorn)
    - Used as the entrypoint for running the microservice

    Example:
        uvicorn app.main:app --reload

    Author: ClaimFlowEngine Team
"""

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.schemas import ClaimInput, ClaimPredictionResponse

app: FastAPI = FastAPI(title="ClaimFlowEngine", version="0.1.0")


@app.get("/ping", response_class=JSONResponse)  # type: ignore[misc]
def ping() -> dict[str, str]:
    """
    Health check endpoint.

    Returns:
        dict[str, str]: JSON with {"message": "pong"}
    """
    return {"message": "pong"}


@app.post("/predict", response_model=ClaimPredictionResponse)  # type: ignore[misc]
def predict_claim(input_claim: ClaimInput) -> ClaimPredictionResponse:
    """
    Replace this with actual model inference later.
    """
    return ClaimPredictionResponse(
        denial_probability=0.75,
        top_denial_reasons=[
            "missing authorization",
            "invalid diagnosis",
            "policy exclusion",
        ],
        model_version="v0.1-dummy",
    )
