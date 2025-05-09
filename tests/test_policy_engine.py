"""
test_policy_engine.py

Unit tests for the Routing Policy Engine in ClaimFlowEngine. Validates
priority score computation and team assignment logic based on enriched
claim metadata and configuration values.

Author: ClaimFlowEngine Project
"""

from __future__ import annotations

import pandas as pd
import pytest

from claimflowengine.routing.policy_engine import route_claims


@pytest.fixture
def mock_config():
    return {
        "weights": {
            "claim_age_days": 0.3,
            "denial_severity": 0.5,
            "payer_complexity": 0.2,
        },
        "team_mapping": {"auth required": "Team A", "coding error": "Team B"},
        "payer_team_map": {"UHC": "Team C", "Cigna": "Team C"},
        "default_team": "Fallback Team",
    }


@pytest.fixture
def mock_claims():
    return pd.DataFrame(
        {
            "claim_id": [1, 2, 3],
            "claim_submission_date": pd.to_datetime(
                ["2025-01-01", "2025-01-10", "2025-01-20"]
            ),
            "denial_cluster_label": ["auth required", "unknown", "coding error"],
            "payer": ["UHC", "Cigna", "OtherPayer"],
        }
    )


def test_routing_output_columns(mock_claims, mock_config):
    routed_df = route_claims(mock_claims.copy(), config=mock_config)
    assert "priority_score" in routed_df.columns
    assert "assigned_team" in routed_df.columns


def test_priority_score_logic(mock_claims, mock_config):
    routed_df = route_claims(mock_claims.copy(), config=mock_config)

    weights = mock_config["weights"]

    expected_scores = (
        weights["claim_age_days"] * routed_df["claim_age_days"]
        + weights["denial_severity"] * routed_df["denial_severity"]
        + weights["payer_complexity"] * routed_df["payer_complexity"]
    )

    assert all(
        abs(a - b) < 1e-3 for a, b in zip(routed_df["priority_score"], expected_scores)
    )


def test_team_assignment(mock_claims, mock_config):
    routed_df = route_claims(mock_claims.copy(), config=mock_config)
    expected_teams = ["Team A", "Team C", "Team B"]
    assert list(routed_df["assigned_team"]) == expected_teams
