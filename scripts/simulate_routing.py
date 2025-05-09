"""
simulate_routing.py

Command-line utility to simulate routing of denied claims using the
Routing Policy Engine. Useful for development, testing, and quick
inspection of routing outcomes on sample datasets.

    This script:
        - Loads a CSV of processed claim data
        - Loads routing configuration from YAML
        - Applies feature enrichment (claim age, denial severity, etc.)
        - Computes priority scores and assigns teams using policy engine
        - Saves or prints the routed output

    Intended Use:
        - Run locally to test or demo the full routing logic
        - Validate routing outcomes before API or pipeline integration

    Example:
        python scripts/simulate_routing.py --input data/processed/claims.csv \
                                           --config config/config.yaml \
                                           --output outputs/routed_claims.csv

    Author: ClaimFlowEngine Project
"""

from __future__ import annotations

import argparse
import pandas as pd

from claimflowengine.routing.policy_engine import route_claims
from claimflowengine.utils.log import get_logger
from claimflowengine.utils.config_loader import load_config

# initialize logger
logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Simulate routing of denied claims.")
    parser.add_argument(
        "--input", type=str, required=False, help="Path to input CSV with claims."
    )
    parser.add_argument("--config", required=True, help="Path to config YAML file.")
    parser.add_argument(
        "--output", type=str, default=None, help="Optional path to save routed claims."
    )

    # Parse the arguments
    args = parser.parse_args()

    # Load config
    full_config = load_config(args.config)

    # Get input path from CLI or config fallback
    input_path = args.input if args.input else full_config["data"]["default_path"]

    logger.info(f"Loading claim data from {input_path}...")
    df = pd.read_csv(input_path)

    logger.info("Running routing logic...")
    routed_df = route_claims(df, config_path=args.config)

    if args.output:
        logger.info(f"Saving routed claims to {args.output}")
        routed_df.to_csv(args.output, index=False)
    else:
        print(routed_df[["claim_id", "priority_score", "assigned_team"]].head())


if __name__ == "__main__":
    main()
