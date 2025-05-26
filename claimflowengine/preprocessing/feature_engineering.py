"""
Module: feature_engineering.py

Description:
    Generates structured features from raw or cleaned
    claims data for modeling and clustering.

Features:
- claim_age_days (safe: submission - service)
- is_resubmission
- note_length / followup_intensity_score
- contains_auth_term
- patient_age
- days_to_submission
- prior_authorization, accident_indicator (binary mapped)
- payer_deny_rate
- provider_deny_rate
- resubmission_rate_by_payer
- high_charge_flag

Functions:
- engineer_features(df: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame
- engineer_edi_features(df: pd.DataFrame) -> pd.DataFrame

Author: ClaimFlowEngine Team
"""

from typing import Any, Optional

import pandas as pd

from claimflowengine.utils.logger import get_logger

# Initialize Logging
logger = get_logger("preprocessing")


def engineer_features(df: pd.DataFrame, y: Optional[Any] = None) -> pd.DataFrame:
    """
    Computes new features for denial modeling and clustering.

    Args:
        df (pd.DataFrame): Input DataFrame with raw/cleaned fields.
        y (pd.Series | None): Optional binary target (0 = paid, 1 = denied).

    Returns:
        pd.DataFrame: DataFrame with new feature columns added.
    """
    df = df.copy()
    logger.info("Engineer features...")

    # --- Time-based Features ---
    if "submission_date" in df.columns and "service_date" in df.columns:
        df["submission_date"] = pd.to_datetime(df["submission_date"], errors="coerce")
        df["service_date"] = pd.to_datetime(df["service_date"], errors="coerce")
        df["claim_age_days"] = (df["submission_date"] - df["service_date"]).dt.days
        df["claim_age_days"] = df["claim_age_days"].astype(float)
    elif "claim_age_days" not in df.columns:
        df["claim_age_days"] = 0.0  # Fallback default

    # --- Resubmission Flag ---
    if "resubmission" in df.columns:
        df["is_resubmission"] = df["resubmission"].astype(bool)
    else:
        df["is_resubmission"] = False

    # --- Follow-up Intensity Score ---
    if "followup_notes" in df.columns:
        df["followup_intensity_score"] = df["followup_notes"].apply(
            lambda x: min(len(str(x).split()), 100) / 100.0
        )
    else:
        df["followup_intensity_score"] = 0.0

    # --- Keyword Match in Denial Reason ---
    if "denial_reason" in df.columns:
        df["contains_auth_term"] = df["denial_reason"].str.contains(
            r"\bauth\b", case=False, na=False
        )
    else:
        df["contains_auth_term"] = False

    # --- Ensure Dummy Label Exists ---
    if "denied" not in df.columns:
        df["denied"] = 0  # Dummy for inference mode

    # --- Domain-Informed Statistical Features (require label) ---
    if y is not None and len(df) == len(y):
        df["denied"] = y.values  # Temporarily merge target

        # payer_deny_rate
        payer_rate = df.groupby("payer_id")["denied"].mean().rename("payer_deny_rate")
        df = df.merge(payer_rate, how="left", on="payer_id")

        # provider_deny_rate
        provider_rate = (
            df.groupby("provider_type")["denied"].mean().rename("provider_deny_rate")
        )
        df = df.merge(provider_rate, how="left", on="provider_type")

        # resubmission_rate_by_payer
        # resub_rate = None
        if "resubmission" in df.columns:
            resub_rate = (
                df.groupby("payer_id")["resubmission"]
                .mean()
                .rename("resubmission_rate_by_payer")
            )
            df = df.merge(resub_rate, how="left", on="payer_id")
        else:
            df["resubmission_rate_by_payer"] = 0.0

    else:
        df["payer_deny_rate"] = 0.5
        df["provider_deny_rate"] = 0.5
        df["resubmission_rate_by_payer"] = 0.0

    # --- High Charge Flag ---
    if "total_charge_amount" in df.columns:
        df["total_charge_amount"] = df["total_charge_amount"].astype(float)
        threshold = df["total_charge_amount"].median(skipna=True)
        df["high_charge_flag"] = (df["total_charge_amount"] > threshold).astype(int)
    else:
        df["high_charge_flag"] = 0

    return df


def engineer_edi_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features from raw EDI 837 schema fields.

    Args:
        df (pd.DataFrame): Raw DataFrame with EDI 837 features.

    Returns:
        pd.DataFrame: DataFrame with engineered EDI-specific fields.
    """
    df = df.copy()

    # Patient age
    if "patient_dob" in df.columns and "service_date" in df.columns:
        df["patient_age"] = (
            pd.to_datetime(df["service_date"], errors="coerce")
            - pd.to_datetime(df["patient_dob"], errors="coerce")
        ).dt.days // 365
        df["patient_age"] = df["patient_age"].astype(float)

    # Binary flag mapping
    for col in ["prior_authorization", "accident_indicator"]:
        if col in df.columns:
            df[col] = df[col].map({"Y": 1, "N": 0}).fillna(0).astype(int)

    # Diagnosis code override
    if "diagnosis_code_primary" in df.columns:
        df["diagnosis_code"] = df["diagnosis_code_primary"]

    # Days from service to submission
    if "service_date" in df.columns and "submission_date" in df.columns:
        df["days_to_submission"] = (
            pd.to_datetime(df["submission_date"], errors="coerce")
            - pd.to_datetime(df["service_date"], errors="coerce")
        ).dt.days
        df["days_to_submission"] = df["days_to_submission"].astype(float)

    # Normalize key categorical strings
    for col in [
        "patient_gender",
        "billing_provider_specialty",
        "facility_code",
        "provider_type",
        "claim_type",
    ]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().str.strip()

    # Ensure numeric for total_charge_amount
    if "total_charge_amount" in df.columns:
        df["total_charge_amount"] = df["total_charge_amount"].astype(float)

    return df
