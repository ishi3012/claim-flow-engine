"""
Routing Policy Engine Module

This module assigns priority scores and
recommended work queues to healthcare claims
based on denial status, root cause cluster,
claim metadata, and payer/provider behavior.

Features:
- Modular scoring logic via `score_claim()`
- Queue assignment logic via `assign_queue()`
- Rule-based fallback engine, extensible for RL upgrades
    (contextual bandit, Q-learning)
- Supports mock logic for claim complexity,
    slow payers, and denial clusters
- Ready for integration into routing microservice or batch pipeline

Intended Use:
    from claimflowengine.routing.policy_engine import PolicyEngine
    policy_engine = PolicyEngine()
    routed_df = policy_engine.route_all(claims_df)

Inputs:
- claims_df (pd.DataFrame) with at minimum:
    - denial_prediction (bool or int)
    - denial_cluster_id (str or int)
    - claim_submission_date (datetime)
    - last_followup_date (datetime)
    - payer_id (str)
    - CPT_codes (list or str)
    - claim_type, billing_provider_specialty

Outputs:
- DataFrame with:
    - priority_score (float)
    - recommended_queue (str)
    - debug_notes (list of scoring decisions)

Author: ClaimFlowEngine Project (2025)
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union, cast

import pandas as pd
import yaml

from claimflowengine.configs.paths import ROUTING_CONFIG_PATH
from claimflowengine.utils.logger import get_logger

# Initilialize logger
logger = get_logger("routing")

# Fallback configs
MOCK_WEIGHTS = {
    "denial_penalty": 2.0,
    "cluster_weights": {
        "auth_required": 1.5,
        "coding_error": 1.2,
        "expired_coverage": 1.0,
        "duplicate_claim": 0.8,
    },
    "slow_payers": {"P001": 1.5, "P002": 1.2},
    "complex_cpt": {"99285": 2.0, "99284": 1.8},
    "claim_age_weight": 0.02,
    "default_score": 1.0,
}

MOCK_TEAM_RULES = {
    "TeamA": ["auth_required", "expired_coverage"],
    "TeamB": ["coding_error"],
    "TeamC": ["duplicate_claim"],
}


def load_routing_config(path: Union[str, Path] = ROUTING_CONFIG_PATH) -> dict[Any, Any]:
    config_path = Path(path) if path else ROUTING_CONFIG_PATH
    try:
        with open(config_path, "r") as f:
            return cast(dict[Any, Any], yaml.safe_load(f))
    except Exception as e:
        msg = (
            f"[PolicyEngine] ! Warning: Failed to load config from {config_path}. "
            f"Using defaults. Reason: {e}"
        )
        print(msg)
        logger.error(msg)
        return {}


def score_claim(
    claim: Optional[Any], config: Dict[str, Any]
) -> Tuple[float, list[str]]:
    score = config.get("default_score", 1.0)
    notes = []
    logger.info("Calculate score and write notes...")
    if claim is None:
        return 0.0, []
    if claim.get("denial_prediction", 0):
        score += config.get("denial_penalty", 0)
        notes.append(f"Denied: +{config.get('denial_penalty', 0): .1f}")
    cluster_id = str(claim.get("denial_cluster_id", "unknown"))
    cluster_score = config.get("cluster_weights", {}).get(cluster_id, 0)
    score += cluster_score
    notes.append(f"Cluster '{cluster_id}: +{cluster_score:.1f}'")

    try:
        age_days = (
            pd.to_datetime(claim["last_followup_date"])
            - pd.to_datetime(claim["claim_submission_date"])
        ).days
    except Exception:
        age_days = 0
    age_score = age_days * config.get("claim_age_weight", 0)
    score += age_score
    notes.append(f"Claim age {age_days}d: +{age_score:.2f}")

    payer = claim.get("payer_id", "")
    payer_score = config.get("slow_payers", {}).get(payer, 0)
    score += payer_score
    notes.append(f"Payer '{payer}': +{payer_score:.1f}")

    cpts = claim.get("CPT_codes", [])
    if isinstance(cpts, str):
        cpts = cpts.split(",")

    for cpt in cpts:
        if cpt in config.get("complex_cpt", {}):
            cpt_score = config["complex_cpt"][cpt]
            score += cpt_score
            notes.append(f"CPT {cpt}: +{cpt_score:.1f}")

    return score, notes


def assign_queue(claim: Optional[Any], config: Dict[str, list[str]]) -> str:
    logger.info("Assign team...")
    if claim is None:
        return "default_queue"
    cluster_id = (
        str(claim.get("denial_cluster_id", "unknown"))
        if claim is not None
        else "unknown"
    )
    for team, clusters in config.items():
        if cluster_id in clusters:
            return team
    return "DefaultQueue"


class PolicyEngine:
    def __init__(self, config_path: Union[str, Path] = ROUTING_CONFIG_PATH) -> None:
        config = load_routing_config(config_path)
        self.weights = config.get("weights", MOCK_WEIGHTS)
        self.team_rules = config.get("team_rules", MOCK_TEAM_RULES)

        if "weights" not in config or "team_rules" not in config:
            msg = (
                "[PolicyEngine] ! Using default mock weights and team rules. "
                "Please verify your YAML config."
            )
            print(msg)
            logger.error(msg)

    def route_all(self, df: pd.DataFrame) -> pd.DataFrame:
        scores, queues, notes = [], [], []

        for _, row in df.iterrows():
            score, debug = score_claim(row, self.weights)
            queue = assign_queue(row, self.team_rules)

            scores.append(round(score, 2))
            queues.append(queue)
            notes.append(debug)

        df = df.copy()
        df["priority_score"] = scores
        df["recommended_queue"] = queues
        df["debug_notes"] = notes

        return df


if __name__ == "__main__":
    load_routing_config()
