# mypy: disable-error-code=misc
"""
Module: schemas.py

Module Description:
    - Defines typed Pydantic schemas for secure input/output
    handling in ClaimFlowEngine's FastAPI denial prediction microservice.
    - Ensures HIPAA-safe data flow for ML model inference.

Features:
    - PII/PHI-safe input schema (`ClaimInput`)
    - Modular, type-enforced design with `Annotated` syntax
    - Clean output schema (`ClaimPredictionResponse`) for safe response exposure
    - Supports explainability and downstream clustering logic

Intended Use:
    Used in FastAPI POST endpoints and ML pipelines
    for denial prediction and workflow orchestration.

Inputs:
    - De-identified claim features (demographics, clinical codes, payer metadata, flags)

Outputs:
    - Denial risk prediction, potential denial reasons, model metadata

Author: ClaimFlowEngine team
"""

from typing import Annotated, List, Literal, Optional

from pydantic import BaseModel, Field

# ----------------------
# Input Schema
# ----------------------


class ClaimInput(BaseModel):
    """
    Schema for incoming claim data used for denial prediction.

    All fields are de-identified.
    Supports FastAPI request validation and typed model inputs.
    """

    # Abstracted Demographics
    patient_age: Annotated[int, Field(ge=0, le=120)]
    gender: Literal["M", "F", "U"]

    # Provider & Clinical Metadata
    provider_type: Annotated[str, Field(min_length=1)]
    billing_provider_specialty: Optional[str]
    claim_type: Literal["professional", "institutional", "inpatient", "outpatient"]
    diagnosis_code: str
    procedure_code: str
    facility_code: Optional[str]
    place_of_service: Optional[str]

    # Time & Financials
    claim_age_days: Annotated[int, Field(ge=0)]
    days_to_submission: Optional[Annotated[int, Field(ge=0)]]
    total_charge_amount: Annotated[float, Field(ge=0.0)]

    # Payer Metadata
    payer_id: Annotated[str, Field(min_length=3, max_length=10)]
    plan_type: Optional[Literal["medicare", "medicaid", "commercial", "other"]]

    # Binary Flags & Derived Features
    prior_authorization: Optional[Literal[0, 1]]
    accident_indicator: Optional[Literal[0, 1]]
    prior_denials_flag: Optional[Literal[0, 1]]
    is_resubmission: Optional[Literal[0, 1]]
    contains_auth_term: Optional[bool]
    note_length: Optional[Annotated[int, Field(ge=0)]]

    class Config:
        schema_extra = {
            "example": {
                "patient_age": 67,
                "gender": "F",
                "provider_type": "Hospital",
                "billing_provider_specialty": "Orthopedics",
                "claim_type": "inpatient",
                "diagnosis_code": "M17.11",
                "procedure_code": "27447",
                "facility_code": "01",
                "place_of_service": "21",
                "claim_age_days": 30,
                "days_to_submission": 10,
                "total_charge_amount": 12000.0,
                "payer_id": "12345",
                "plan_type": "medicare",
                "prior_authorization": 1,
                "accident_indicator": 0,
                "prior_denials_flag": 1,
                "is_resubmission": 0,
                "contains_auth_term": True,
                "note_length": 128,
            }
        }


# ----------------------
# Output Schema
# ----------------------


class ClaimPredictionResponse(BaseModel):
    """
    Schema for model prediction output sent back to client or UI layer.

    Devoid of any PII. Supports downstream explainability and routing.
    """

    denial_probability: Annotated[float, Field(ge=0.0, le=1.0)]
    top_denial_reasons: List[str]
    model_version: Optional[str]
    routing_cluster_id: Optional[str]
    explainability_scores: Optional[dict[str, float]]

    class Config:
        schema_extra = {
            "example": {
                "denial_probability": 0.84,
                "top_denial_reasons": ["Authorization missing", "Service not covered"],
                "model_version": "v1.2.1",
                "routing_cluster_id": "CL-04",
                "explainability_scores": {
                    "prior_authorization": 0.23,
                    "claim_type": 0.18,
                    "payer_id": 0.15,
                },
            }
        }
