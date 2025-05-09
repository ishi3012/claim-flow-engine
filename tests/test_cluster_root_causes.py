"""
test_cluster_root_causes.py

End-to-end test for the cluster_root_causes.py pipeline.
Validates that embeddings, clustering, and outputs are created as expected.

Author: ClaimFlowEngine Project
"""

import tempfile
import shutil
import pandas as pd
from pathlib import Path

from claimflowengine.clustering.cluster_root_causes import main as cluster_main


def test_cluster_pipeline_end_to_end():
    """Test full root cause clustering pipeline on toy data."""

    temp_dir = Path(tempfile.mkdtemp())
    temp_data_path = temp_dir / "toy_claims.csv"
    temp_output_path = temp_dir / "clustered.csv"
    temp_embed_cache = temp_dir / "embed.pkl"
    temp_summary = temp_dir / "summary.csv"
    temp_model_dir = temp_dir / "models"
    temp_model_dir.mkdir(exist_ok=True)

    # create toy sample dataset

    df = pd.DataFrame(
        {
            "claim_id": range(5),
            "denial_reason": [
                "Missing prior auth",
                "Invalid CPT code",
                "No supporting documents",
                "Prior auth required",
                "Duplicate billing",
            ],
            "denied": [1, 1, 1, 1, 1],
        }
    )
    df.to_csv(temp_data_path, index=False)

    # Build fake config dict
    config = {
        "data": {"balanced_path": str(temp_data_path)},
        "clustering": {
            "embedding_model": "all-MiniLM-L6-v2",
            "embedding_cache": str(temp_embed_cache),
            "output_path": str(temp_output_path),
            "n_neighbors": 5,
            "min_dist": 0.0,
            "n_components": 2,
            "hdbscan_min_cluster_size": 2,
            "plot_clusters": False,
            "summarize_clusters": True,
            "save_cluster_summary": True,
            "cluster_summary_path": str(temp_summary),
        },
    }

    # Run pipeline
    cluster_main(config)

    # Check output files
    assert Path(temp_output_path).exists()
    assert Path(temp_embed_cache).exists()
    assert Path(temp_summary).exists()

    # Load and Validate output
    clustered_df = pd.read_csv(temp_output_path)
    assert "cluster" in clustered_df.columns
    assert clustered_df.shape[0] == 5

    if Path(temp_summary).stat().st_size > 0:
        summary_df = pd.read_csv(temp_summary)
        assert "cluster_id" in summary_df.columns
    else:
        print("⚠️ No clusters found — summary file is empty as expected.")

    assert "cluster_id" in summary_df.columns

    # cleanup
    shutil.rmtree(temp_dir)
