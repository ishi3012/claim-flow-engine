import argparse

import pandas as pd

from claimflowengine.clustering.clustering_pipeline import cluster_claims
from claimflowengine.configs.paths import (
    CLUSTERING_OUTPUT_PATH,
    INFERENCE_INPUT_PATH,
)
from claimflowengine.utils.logger import get_logger

logger = get_logger("cluster")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default=INFERENCE_INPUT_PATH,
        help="Path to input CSV"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=CLUSTERING_OUTPUT_PATH,
        help="Path to input CSV"
    )

    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if df is None or df.empty:
        logger.error("Input file is empty.")
    else:
        clustered = cluster_claims(df, output_path=args.output)
        print(f"Clustered claims saved to {args.output}")

