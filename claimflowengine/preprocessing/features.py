"""
Module: features.py

Description:
    Generates structured features from raw or cleaned
    claims data for modeling and clustering.

Features:
- claim_age_days
- is_resubmission
- prior_denials_flag
- note_length
- contains_auth_term
- patient_age
- days_to_submission
- prior_authorization, accident_indicator (binary mapped)

Functions:
- engineer_features(df: pd.DataFrame) -> pd.DataFrame
- engineer_edi_features(df: pd.DataFrame) -> pd.DataFrame

Author: ClaimFlowEngine Team
"""

import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes new features for denial modeling and clustering.

    Args:
        df (pd.DataFrame): Input DataFrame with raw/cleaned fields.

    Returns:
        pd.DataFrame: DataFrame with new feature columns added.
    """
    df = df.copy()

    # Calculate claim age in days
    df["submission_date"] = pd.to_datetime(df["submission_date"], errors="coerce")
    df["denial_date"] = pd.to_datetime(df["denial_date"], errors="coerce")
    df["claim_age_days"] = (df["denial_date"] - df["submission_date"]).dt.days
    df["claim_age_days"] = df["claim_age_days"].astype(float)

    # Flag for prior denial
    df["prior_denials_flag"] = df["denial_reason"].notnull() & df[
        "denial_reason"
    ].str.strip().ne("")

    # Resubmission flag (assume field is 0/1 or bool)
    df["is_resubmission"] = df["resubmission"].astype(bool)

    # Note length
    if "followup_notes_clean" in df.columns:
        df["note_length"] = df["followup_notes_clean"].apply(
            lambda x: len(str(x).split())
        )
    else:
        df["note_length"] = 0
    df["note_length"] = df["note_length"].astype(float)

    # Contains 'auth' keyword in denial reason
    if "denial_reason_clean" in df.columns:
        df["contains_auth_term"] = df["denial_reason_clean"].str.contains(
            r"\bauth\b", case=False, na=False
        )
    else:
        df["contains_auth_term"] = False

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

    # Clean up string fields
    for col in [
        "patient_gender",
        "billing_provider_specialty",
        "facility_code",
        "provider_type",
        "claim_type",
    ]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().str.strip()

    # Total charge amount — ensure float
    if "total_charge_amount" in df.columns:
        df["total_charge_amount"] = df["total_charge_amount"].astype(float)

    return df
