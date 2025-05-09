"""
policy_engine.py

Implements the Routing Policy Engine that assigns denied claims to optimal
A/R teams or workflow queues based on a hybrid scoring model. Designed to
prioritize claims using claim metadata, denial patterns, and payer history
for efficient recovery and reduced rework.

    This module:
        - Computes a priority score using weighted claim age, denial severity, and payer complexity
        - Assigns each claim to a team using denial cluster and payer-based mappings
        - Leverages centralized configuration for weights, mappings, and default fallbacks
        - Supports integration with enrich_routing_features for feature prep

    Intended Use:
        - As the core routing logic within ClaimFlowEngine’s workflow engine
        - Enables ML-informed and rules-backed prioritization of denied claims
        - Plugs into FastAPI apps, batch pipelines, or agent-based execution flows

    Inputs:
        - `pd.DataFrame` of claims with enriched routing features
        - YAML config file path containing routing configuration block

    Outputs:
        - `pd.DataFrame` with two additional columns: 'priority_score', 'assigned_team'

    Functions:
        - load_routing_config: Parses YAML config and returns only the routing section
        - score_claim: Computes weighted score for a single claim row
        - assign_team: Determines team assignment based on cluster or payer
        - route_claims: Main entry point to score and assign a full claim dataset

    Args:
        df (pd.DataFrame): Enriched claims with all necessary routing features
        config_path (str): Path to YAML config file containing routing block

    Example:
        from claimflowengine.routing.policy_engine import route_claims
        routed_df = route_claims(df, config_path="config/config.yaml")

    Author: ClaimFlowEngine Project
"""

from __future__ import annotations

import pandas as pd
from typing import Dict, Any, Union

from claimflowengine.utils.log import get_logger
from claimflowengine.utils.config_loader import load_config
from claimflowengine.routing.enrich_routing_features import enrich_routing_features

# initialize logger
logger = get_logger(__name__)


def load_routing_config(path: str) -> Dict[str, Any]:

    logger.info(f"Loading routing config from {path}")

    try:
        config = load_config(path)
        routing_config = config.get("routing", {})
    except FileNotFoundError as e:
        logger.exception(f"[CONFIG ERROR] {e}")
    except Exception as e:
        logger.exception(f"[CONFIG ERROR] {e}")

    return routing_config


def score_claim(claim: pd.Series, config: Dict[str, Any]) -> float:
    weights = config["weights"]

    try:
        score = (
            weights["claim_age_days"] * claim["claim_age_days"]
            + weights["denial_severity"] * claim["denial_severity"]
            + weights["payer_complexity"] * claim["payer_complexity"]
        )
        logger.debug(f"Score for claim {claim['claim_id']}: {score}")
        return score
    except Exception as e:
        logger.error(f"Failed to score claim {claim['claim_id']}: {e}")
        return 0.0


def assign_team(claim: pd.Series, config: Dict[str, Any]) -> str:
    try:
        if claim["denial_cluster_label"] in config["team_mapping"]:
            team = config["team_mapping"][claim["denial_cluster_label"]]
        elif claim["payer"] in config["payer_team_map"]:
            team = config["payer_team_map"][claim["payer"]]
        else:
            team = config["default_team"]

        logger.debug(f"Claim {claim['claim_id']} routed to {team}")
        return team

    except Exception as e:
        logger.error(f"Failed to assign team for claim {claim['claim_id']}: {e}")
        return config["default_team"]


def route_claims(df: pd.DataFrame, config: Union[str, Dict[str, Any]]) -> pd.DataFrame:
    logger.info("Starting routing of claims")

    if isinstance(config, str):
        config = load_routing_config(config)

    df = enrich_routing_features(df, config)

    df["priority_score"] = df.apply(score_claim, axis=1, args=(config,))
    df["assigned_team"] = df.apply(assign_team, axis=1, args=(config,))

    logger.info(f"Completed routing {len(df)} claims.")
    return df
