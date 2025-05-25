import pandas as pd

from claimflowengine.inference.merger import merge_predictions_and_clusters


def test_merge_predictions_and_clusters() -> None:
    df_pred = pd.DataFrame(
        {
            "claim_id": [1, 2, 3],
            "denial_prediction": [1, 0, 1],
            "denial_probability": [0.91, 0.22, 0.76],
        }
    )

    df_cluster = pd.DataFrame(
        {
            "claim_id": [1, 2, 3],
            "denial_cluster_id": [0, 1, 2],
            "cluster_label": ["Auth Missing", "Not Covered", "Duplicate"],
        }
    )

    merged = merge_predictions_and_clusters(df_pred, df_cluster)

    assert merged.shape[0] == 3
    assert "denial_cluster_id" in merged.columns
    assert "cluster_label" in merged.columns
    assert merged.loc[0, "cluster_label"] == "Auth Missing"
