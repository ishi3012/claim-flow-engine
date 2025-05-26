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

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from app.router.route_router import router as route_router
from app.schemas import ClaimInput, ClaimPredictionResponse
from claimflowengine.inference.loader import load_model
from claimflowengine.inference.predictor import predict_claims
from claimflowengine.utils.git import get_model_version_tag
from claimflowengine.utils.logger import get_logger

MODEL_VERSION = get_model_version_tag()

app: FastAPI = FastAPI(title="ClaimFlowEngine", version="0.1.0")
app.include_router(route_router, prefix="/api")

# Initialize Logger
logger = get_logger("inference")

# Load model and transformers once on startup
model, target_encoder, numeric_transformer = load_model()


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
    try:
        logger.info("Received claim for prediction")

        # Convert Pydantic model to DataFrame
        df_input = pd.DataFrame([input_claim.dict()])

        # Run Prediction
        results = predict_claims(
            raw_data=df_input,
            model=model,
            target_encoder=target_encoder,
            numeric_transformer=numeric_transformer,
        )

        result = results[0]

        return ClaimPredictionResponse(
            denial_probability=result["denial_probability"],
            top_denial_reasons=result.get("top_3_denial_reasons", []),
            model_version=MODEL_VERSION,
            routing_cluster_id=None,
            explainability_scores=None,
        )
    except Exception as e:
        logger.exception("Prediction failed.")
        raise HTTPException(status_code=500, detail=str(e))
