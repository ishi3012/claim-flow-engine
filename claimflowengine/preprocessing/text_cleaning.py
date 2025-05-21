"""
Module: text_cleaning.py

Description:
    Text normalization utilities for cleaning denial_reason and followup_notes.
    Standardizes casing, punctuation, and common term variants.

Intended Use:
    Used during preprocessing pipeline to clean free-text fields
    prior to modeling or clustering.

Functions:
- clean_text_fields(df: pd.DataFrame) -> pd.DataFrame

Author: ClaimFlowEngine Team
"""

import re

import pandas as pd

from claimflowengine.utils.logger import get_logger

# Initialize Logging
logger = get_logger("preprocessing")


def _normalize_text(text: str) -> str:
    """
    Lowercases, strips punctuation, and replaces domain-specific phrases
    with canonical forms.

    Args:
        text (str): Raw input text (e.g., denial_reason or notes).

    Returns:
        str: Cleaned and normalized text suitable for modeling or clustering.
    """
    if not isinstance(text, str):
        return ""

    REPLACEMENTS = {
        "authorization": "auth",
        "medical necessity": "med necessity",
        "not covered by plan": "not covered",
        "timely filing": "late filing",
        "duplicate claim": "duplicate",
        "invalid diagnosis": "dx error",
        "coding issue": "coding error",
    }

    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)

    for phrase, replacement in REPLACEMENTS.items():
        text = text.replace(phrase, replacement)

    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_text_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and standardizes free-text fields in a claims dataframe.

    Args:
        df (pd.DataFrame): Raw claims dataframe with 'denial_reason'
                           and 'followup_notes'.

    Returns:
        pd.DataFrame: DataFrame with cleaned text fields.
    """
    df = df.copy()
    logger.info("Clean text fields...")

    if "denial_reason" in df.columns:
        logger.info("Cleaning denial_reason column")
        df["denial_reason_clean"] = df["denial_reason"].apply(_normalize_text)

    if "followup_notes" in df.columns:
        logger.info("Cleaning followup_notes column")
        df["followup_notes_clean"] = df["followup_notes"].apply(_normalize_text)

    return df
