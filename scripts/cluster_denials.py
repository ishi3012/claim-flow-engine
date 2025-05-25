import argparse
import json

import pandas as pd

from claimflowengine.clustering.cluster_summary import (
    attach_cluster_labels,
    generate_cluster_labels,
)
from claimflowengine.clustering.clustering_pipeline import cluster_claims
from claimflowengine.configs.paths import (
    CLUSTER_LABELS_JSON_PATH,
    CLUSTERING_OUTPUT_PATH,
    INFERENCE_INPUT_PATH,
)
from claimflowengine.utils.logger import get_logger

logger = get_logger("cluster")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=str, default=INFERENCE_INPUT_PATH, help="Path to input CSV"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=CLUSTERING_OUTPUT_PATH,
        help="Path to save clustered claims",
    )
    parser.add_argument(
        "--label_method",
        type=str,
        default="mode",
        choices=["mode", "tfidf"],
        help="Labeling strategy",
    )
    parser.add_argument(
        "--label_json",
        type=str,
        default=CLUSTER_LABELS_JSON_PATH,
        help="Where to save label JSON",
    )

    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if df is None or df.empty:
        logger.error("Input file is empty.")
        exit(1)

    # Run clustering
    clustered_df = cluster_claims(df, output_path=args.output)

    # Generate human-readable cluster labels
    label_map = generate_cluster_labels(clustered_df, method=args.label_method)
    logger.info(f"Generated labels for {len(label_map)} clusters.")

    # Save JSON label map
    with open(args.label_json, "w") as f:
        json.dump(label_map, f, indent=2)
        logger.info(f"Saved cluster label map to {args.label_json}")

    # Optionally attach labels and overwrite clustered output
    labeled_df = attach_cluster_labels(clustered_df, label_map)
    labeled_df.to_csv(args.output, index=False)
    logger.info(f"Updated clustered claims with labels saved to {args.output}")
