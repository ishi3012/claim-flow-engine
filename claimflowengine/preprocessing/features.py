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

Functions:
- engineer_features(df: pd.DataFrame) -> pd.DataFrame

Author: ClaimFlowEngine Team
"""

import pandas as pd

# import numpy as np


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

    # Flag for prior denial
    df["prior_denials_flag"] = df["denial_reason"].notnull() & df[
        "denial_reason"
    ].str.strip().ne("")

    # Resubmission (assumption: the field is numerical)
    df["is_resubmission"] = df["resubmission"].astype(bool)

    # Note length
    if "followup_notes_clean" in df.columns:
        df["note_length"] = df["followup_notes_clean"].apply(
            lambda x: len(str(x).split())
        )
    else:
        df["note_length"] = 0

    # Contains 'auth' term in cleaned denial_reason
    if "denial_reason_clean" in df.columns:
        df["contains_auth_term"] = df["denial_reason_clean"].str.contains(
            r"\bauth\b", case=False, na=False
        )
    else:
        df["contains_auth_term"] = False

    return df
