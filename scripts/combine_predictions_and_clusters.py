"""
CLI Tool: Combine Prediction and Clustering Outputs

This script merges denial prediction output and root cause clustering output into
a single routing-ready file using claim_id as the key.

Usage:
    python scripts/combine_predictions_and_clusters.py \
        --pred outputs/predictions.csv \
        --cluster outputs/clusters.csv \
        --output outputs/merged_for_routing.csv

Author: ClaimFlowEngine Team (2025)
"""

import argparse

from claimflowengine.configs.paths import (
    CLUSTERING_OUTPUT_PATH,
    DENIAL_PREDICTION_OUTPUT_PATH,
    MERGED_ROUTING_PATH,
)
from claimflowengine.inference.merger import merge_predictions_and_clusters_from_files


def main(pred_path: str, cluster_path: str, output_path: str) -> None:
    df = merge_predictions_and_clusters_from_files(pred_path, cluster_path)
    df.to_csv(output_path, index=False)
    print(f"Merged file written to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge prediction and clustering results."
    )
    parser.add_argument(
        "--pred", default=DENIAL_PREDICTION_OUTPUT_PATH, help="Path to predictions CSV"
    )
    parser.add_argument(
        "--cluster", default=CLUSTERING_OUTPUT_PATH, help="Path to clustering CSV"
    )
    parser.add_argument(
        "--output", default=MERGED_ROUTING_PATH, help="Path to save merged routing file"
    )
    args = parser.parse_args()

    main(args.pred, args.cluster, args.output)
