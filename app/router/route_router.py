"""
Route Router

Handles the routing endpoint to score claims and assign A/R work queues.

Features:
- Accepts CSV uploads of claims
- Routes each claim with a priority score and recommended queue
- Returns list of routed claim dicts

Author: ClaimFlowEngine Team
"""

from io import BytesIO
from typing import Any, Dict, List, cast

import numpy as np
import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile

from claimflowengine.clustering.cluster_summary import (
    attach_cluster_labels,
    generate_cluster_labels,
)
from claimflowengine.clustering.clustering_pipeline import cluster_claims
from claimflowengine.inference.loader import load_model
from claimflowengine.inference.predictor import predict_claims
from claimflowengine.routing.policy import PolicyEngine
from claimflowengine.utils.logger import get_logger
from claimflowengine.utils.postprocessing import standardize_prediction_columns

# Initialize logger
logger = get_logger("routing")

router: APIRouter = APIRouter()
policy_engine = PolicyEngine()


@router.post("/route", response_model=List[Dict[str, Any]])  # type: ignore[misc]
async def route_claims(file: UploadFile = File(...)) -> List[Dict[str, Any]]:
    """
    Accepts a CSV file with claims and returns routed priorities.
    """
    try:
        content = await file.read()
        df = pd.read_csv(BytesIO(content))
        routed = policy_engine.route_all(df)
        return cast(List[Dict[str, Any]], routed.to_dict(orient="records"))
    except Exception as e:
        logger.exception("Routing failed.")
        raise HTTPException(status_code=500, detail=f"Routing failed: {e}")


@router.post("/fullroute", response_model=List[Dict[str, Any]])  # type: ignore[misc]
async def full_route_claims(file: UploadFile = File(...)) -> List[Dict[str, Any]]:
    """
    Accepts a CSV file, performs:
    - denial prediction
    - root cause clustering
    - routing priority scoring

    Returns:
        List of routed claims as dicts.
    """
    try:
        logger.info("Reading uploaded CSV...")
        content = await file.read()
        df_raw = pd.read_csv(BytesIO(content))

        # Step 1: Denial Prediction
        logger.info("Step 1: Denial Prediction")
        model, target_encoder, numeric_transformer = load_model()
        pred_results = predict_claims(
            raw_data=df_raw,
            model=model,
            target_encoder=target_encoder,
            numeric_transformer=numeric_transformer,
        )

        df_pred = pd.concat(
            [df_raw.reset_index(drop=True), pd.DataFrame(pred_results)], axis=1
        )
        df_pred = standardize_prediction_columns(df_pred)

        # Step 2: Root cause clustering
        logger.info("Step 2: Root cause clustering")
        clustered_df = cluster_claims(df_pred)
        label_map = generate_cluster_labels(clustered_df)
        labeled_df = attach_cluster_labels(clustered_df, label_map)

        # Step 3: Routing
        logger.info("Step 3: Routing")
        policy_engine = PolicyEngine()
        routed_df = policy_engine.route_all(labeled_df)

        logger.info("Routing complete. Returning routed claim data.")
        routed_df = routed_df.replace([np.inf, -np.inf], np.nan).fillna("NA")

        return cast(List[Dict[str, Any]], routed_df.to_dict(orient="records"))

    except Exception as e:
        logger.exception("Full routing pipeline failed.")
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {str(e)}")
