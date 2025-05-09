"""
enrich_routing_features.py

Computes routing-relevant features from raw claim metadata to support hybrid
scoring and assignment in the Routing Policy Engine. Designed as a pre-processing
step within ClaimFlowEngine’s workflow orchestration module.

    This module:
        - Calculates claim age in days based on submission date
        - Maps denial cluster labels or reasons to severity scores using config
        - Maps payer names to turnaround complexity scores from config

    Intended Use:
        - As a preparatory step before routing claims using policy_engine.py
        - Ensures all priority scoring fields are present and standardized
        - Keeps routing logic stateless and decoupled from feature computation

    Inputs:
        - `pd.DataFrame` with columns like 'claim_submission_date', 'denial_cluster_label', 'payer'
        - Routing section of YAML config (with severity_map, payer_turnaround)

    Outputs:
        - `pd.DataFrame` with three new columns: 'claim_age_days', 'denial_severity', 'payer_complexity'

    Functions:
        - enrich_routing_features: Main enrichment function for routing pipeline

    Args:
        df (pd.DataFrame): Raw or pre-processed claims data
        config (Dict): Loaded 'routing' section from YAML config

    Example:
        from claimflowengine.routing.enrich_routing_features import enrich_routing_features
        enriched_df = enrich_routing_features(df, config)

    Author: ClaimFlowEngine Project
"""

from __future__ import annotations

import pandas as pd
import datetime
from typing import Dict, Any

from claimflowengine.utils.log import get_logger

# initialize logger
logger = get_logger(__name__)


def enrich_routing_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:

    logger.info("Enriching claim DataFrame with routing features.")

    today = pd.to_datetime(datetime.datetime.today())

    # Claim age in days
    try:
        df["claim_submission_date"] = pd.to_datetime(df["claim_submission_date"])
        df["claim_age_days"] = (today - df["claim_submission_date"]).dt.days
        logger.debug("claim_age_days added.")
    except Exception as e:
        logger.error(f"Failed to compute claim_age_days: {e}")
        df["claim_age_days"] = 0

    # Denials Severity mapping
    severity_map = config.get("severity_map", {})

    if "denial_cluster_label" in df.columns:
        df["denial_severity"] = df["denial_cluster_label"].map(severity_map).fillna(0.1)
    elif "denial_reason" in df.columns:
        df["denial_severity"] = df["denial_reason"].map(severity_map).fillna(0.1)
    else:
        logger.warning(
            "No 'denial_cluster_label' or 'denial_reason' found for severity mapping."
        )
        df["denial_severity"] = 0.1
    logger.debug("denial_severity mapped.")

    # Payer complexity mapping
    turnaround_map = config.get("payer_turnaround", {})

    df["payer_complexity"] = df["payer"].map(turnaround_map).fillna(10)
    logger.debug("payer_complexity mapped.")

    logger.info("Routing features enrichment complete.")

    return df
