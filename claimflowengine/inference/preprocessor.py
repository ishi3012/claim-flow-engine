"""
Module: preprocessor.py
    — Inference-Time Feature Engineering for Claim Denial Prediction

This module reproduces the exact feature engineering pipeline used during training,
ensuring consistent input format for the denial prediction model.

Features:
- Text cleaning for 'denial_reason' and 'followup_notes'
- EDI-specific feature generation (if sufficient schema coverage)
- Structured feature engineering (e.g., claim_age_days, payer_deny_rate)
- Target encoding for categorical features
- Numeric + boolean transformation using fitted ColumnTransformer

Intended Use:
- Called by `predict_claims()` inside predictor.py
- Used in both batch and real-time inference flows

Inputs:
- raw_input: pd.DataFrame (or converted from dict/list)
- target_encoder: fitted TargetEncoder
- numeric_transformer: fitted ColumnTransformer

Output:
- pd.DataFrame of fully preprocessed features

Author: ClaimFlowEngine Team (2025)
"""

import pandas as pd
import yaml
from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer

from claimflowengine.configs.paths import FEATURE_CONFIG_PATH
from claimflowengine.preprocessing.feature_engineering import (
    engineer_edi_features,
    engineer_features,
)
from claimflowengine.preprocessing.text_cleaning import clean_text_fields
from claimflowengine.utils.logger import get_logger

# Initialize Logging
logger = get_logger("inference")


def preprocess_for_inference(
    raw_input: pd.DataFrame,
    target_encoder: TargetEncoder,
    numeric_transformer: ColumnTransformer,
) -> pd.DataFrame:
    """
    Preprocess raw claim data for model inference.

    Args:
        raw_input (pd.DataFrame):
            Raw claim data.
        target_encoder (TargetEncoder):
            Fitted encoder for categorical features.
        numeric_transformer (ColumnTransformer):
            Fitted transformer for numeric + boolean.

    Returns:
        pd.DataFrame: Preprocessed feature matrix.

    """

    logger.info(f"Loading claims input file: {raw_input}... ")
    df = raw_input.copy()

    # Step 1: Clean denial_reason + followup_notes
    logger.info("Step 1: Cleaning text fields... ")
    df = clean_text_fields(df)

    # Step 2: Apply EDI features if sufficient coverage
    edi_fields = [
        "patient_dob",
        "service_date",
        "patient_gender",
        "facility_code",
        "billing_provider_specialty",
        "claim_type",
        "prior_authorization",
        "accident_indicator",
    ]

    edi_coverage = len([col for col in edi_fields if col in df.columns]) / len(
        edi_fields
    )
    if edi_coverage >= 0.5:
        logger.info("Step 2: Engineer EDI features... ")
        df = engineer_edi_features(df)

    # Step 3: Dummy label for compatibility with engineer_features
    df["denied"] = 0  # dummy label
    logger.info("Step 3:Engineer general features... ")
    df = engineer_features(df, df["denied"])

    #  Drop label again
    X = df.drop(columns=["denied"])

    # Step 4: Encode categorical features
    logger.info("Step 4: Encode categorical features... ")
    categorical_cols = [
        "payer_id",
        "provider_type",
        "plan_type",
        "claim_type",
        "billing_provider_specialty",
        "facility_code",
        "diagnosis_code",
        "procedure_code",
        # "prior_authorization",
    ]

    categorical_cols = [col for col in categorical_cols if col in X.columns]
    encoded_cats = target_encoder.transform(X[categorical_cols])

    # Step 6: Transform Numeric + boolean
    logger.info("Step 6: Transform Numeric + boolean... ")
    numeric_features = [
        "claim_age_days",
        "patient_age",
        "total_charge_amount",
        "days_to_submission",
        "payer_deny_rate",
        "provider_deny_rate",
        "resubmission_rate_by_payer",
        "followup_intensity_score",
    ]
    boolean_features = [
        "is_resubmission",
        "contains_auth_term",
        "prior_authorization",
        "accident_indicator",
        "high_charge_flag",
    ]

    numeric_features = [f for f in numeric_features if f in X.columns]
    boolean_features = [f for f in boolean_features if f in X.columns]

    transformed_array = numeric_transformer.transform(
        X[numeric_features + boolean_features]
    )
    transformed_df = pd.DataFrame(
        transformed_array, columns=numeric_transformer.get_feature_names_out()
    )
    logger.info("Step 7: Merge final dataframe... ")
    final_df = pd.concat(
        [encoded_cats.reset_index(drop=True), transformed_df.reset_index(drop=True)],
        axis=1,
    )
    with open(FEATURE_CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    final_df = final_df[config["features"]]
    return final_df
