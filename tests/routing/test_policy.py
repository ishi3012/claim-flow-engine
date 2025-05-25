import pandas as pd

from claimflowengine.routing.policy import score_claim


def test_score_claim_basic() -> None:
    claim = pd.Series(
        {
            "claim_id": 123,
            "denial_prediction": 1,
            "denial_probability": 0.88,
            "denial_cluster_id": 0,
            "payer_id": "PAYER123",
            "claim_submission_date": "2024-01-01",
            "last_followup_date": "2024-01-10",
            "CPT_codes": "99213,85025",
        }
    )

    mock_config = {
        "default_score": 1.0,
        "denial_penalty": 0.5,
        "cluster_weights": {"0": 0.2},
        "claim_age_weight": 0.01,
        "slow_payers": {"PAYER123": 0.1},
        "complex_cpt": {"99213": 0.3},
    }

    score, notes = score_claim(claim, mock_config)

    assert abs(score - 2.19) < 1e-6
    assert any("Denied" in n for n in notes)
    assert any("CPT 99213" in n for n in notes)
    assert any("Payer" in n for n in notes)
