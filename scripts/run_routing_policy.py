"""
CLI: Run Routing Policy Engine

Takes a merged CSV file (with denial predictions + clusters),
computes priority scores and routing queues, and saves output.

Usage:
    python scripts/run_routing_policy.py \
        --input outputs/merged_for_routing.csv \
        --output outputs/routed_claims.csv

Author: ClaimFlowEngine Team (2025)
"""

import argparse

import pandas as pd

from claimflowengine.configs.paths import (
    CLAIMS_ROUTING_OUTPUT_PATH,
    MERGED_ROUTING_PATH,
)
from claimflowengine.routing.policy import PolicyEngine


def main(input_path: str, output_path: str) -> None:
    df = pd.read_csv(input_path)

    engine = PolicyEngine()
    routed_df = engine.route_all(df)

    routed_df.to_csv(output_path, index=False)
    print(f"Routed claims saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run routing policy engine on merged claims."
    )
    parser.add_argument(
        "--input", default=MERGED_ROUTING_PATH, help="Path to merged input CSV"
    )
    parser.add_argument(
        "--output",
        default=CLAIMS_ROUTING_OUTPUT_PATH,
        help="Path to save routed output",
    )
    args = parser.parse_args()

    main(args.input, args.output)
