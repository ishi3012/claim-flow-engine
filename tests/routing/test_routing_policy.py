import pandas as pd

from claimflowengine.routing.policy import PolicyEngine


def mock_input_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "denial_prediction": [True, False, True],
            "denial_cluster_id": ["auth_required", "coding_error", "expired_coverage"],
            "claim_submission_date": pd.to_datetime(
                ["2024-01-01", "2024-01-05", "2024-01-10"]
            ),
            "last_followup_date": pd.to_datetime(
                ["2024-01-15", "2024-01-25", "2024-02-01"]
            ),
            "payer_id": ["P001", "P002", "P003"],
            "CPT_codes": ["99285", "85025", "93000"],
        }
    )


def test_policy_engine_outputs_expected_columns() -> None:
    df = mock_input_df()
    engine = PolicyEngine()
    result_df = engine.route_all(df)

    assert "priority_score" in result_df.columns
    assert "recommended_queue" in result_df.columns
    assert "debug_notes" in result_df.columns
    assert isinstance(result_df["priority_score"].iloc[0], float)
    assert isinstance(result_df["recommended_queue"].iloc[0], str)
