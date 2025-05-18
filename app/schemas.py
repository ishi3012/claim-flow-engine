"""
Module: schemas.py

    Module Description:
        Defines Pydantic schemas for request and response handling in ClaimFlowEngine's FastAPI app.

    Features:
        - Input schema for receiving claim data in POST requests.
        - Output schema for structured prediction responses.
        - Fully typed and FastAPI-compatible.

    Intended Use:
        Used by FastAPI routes to validate incoming claim payloads and format outgoing prediction responses.

    Inputs:
        - Claim features (demographics, codes, payer info).

    Outputs:
        - Denial probability and reasons.

    Functions:
        - ClaimInput: input schema for prediction.
        - ClaimPredictionResponse: output schema.

    Author: ClaimFlowEngine team
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class ClaimInput(BaseModel):
    claim_id: str = Field(..., description="Unique Identifier for the claim.")
    patient_age: int = Field(..., ge=0, le=120, description="Age of the patient.")
    patient_gender: str = Field(..., description="Gender of the patient (M/F/0)")
    diagnosis_codes: List[str] = Field(
        ..., description="List of ICD-10 diagnosis codes."
    )
    procedure_codes: List[str] = Field(
        ..., description="List of CPT/HCPCS procedure codes."
    )
    provider_npi: str = Field(..., description="National Provider Indentifier.")
    payer_id: str = Field(..., description="Unique identifier of the payer.")
    service_location: Optional[str] = Field(
        None, description="Facility or place of service."
    )
    previous_denial: Optional[bool] = Field(
        False, description="Was the claim previously denied?"
    )


class ClaimPredictionResponse(BaseModel):
    denial_probability: float = Field(
        ..., ge=0.0, le=1.0, description="Predicted probability of denial."
    )
    top_denial_reasons: List[str] = Field(
        ..., description="Top 3 predicted reasons for denial."
    )
    model_version: Optional[str] = Field(
        None, description="Version of the deployed model used for prdiction."
    )
