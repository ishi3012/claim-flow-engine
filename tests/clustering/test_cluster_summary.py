import pandas as pd

from claimflowengine.clustering.cluster_summary import generate_cluster_labels


def test_generate_cluster_labels_mode() -> None:
    df = pd.DataFrame(
        {
            "claim_id": [1, 2, 3, 4, 5],
            "denial_reason": [
                "Missing Auth",
                "Missing Auth",
                "Invalid Code",
                "Invalid Code",
                "Service Not Covered",
            ],
            "denial_cluster_id": [0, 0, 1, 1, 2],
        }
    )

    label_map = generate_cluster_labels(df, method="mode")

    assert label_map[0] == "Missing Auth"
    assert label_map[1] == "Invalid Code"
    assert label_map[2] == "Service Not Covered"
