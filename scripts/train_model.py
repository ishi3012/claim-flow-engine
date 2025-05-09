"""
train_model.py

Command-line script to train a healthcare claim denial prediction model.

Loads a YAML config file, invokes the core training pipeline defined in
claimflowengine.pipelines.train, and logs results. Accepts runtime options
for config path and log level.

Intended for use in local development, CI/CD, or integration with orchestration tools
like Airflow, Vertex Pipelines, or bash scripts.

Usage:
    python scripts/train_model.py --config config/config.yaml --log_level info

Author: Claim Flow Engine Project
"""

import argparse
import yaml
import sys
from pathlib import Path
from typing import Any, Dict

from claimflowengine.pipelines.train import train_model
from claimflowengine.utils.log import get_logger
from claimflowengine.utils.paths import CONFIG_PATH


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="train denial prediction model")
    parser.add_argument(
        "--config",
        type=str,
        default=str(CONFIG_PATH),
        help="Path to YAML config file (default: config/config.yaml)",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="info",
        choices=["debug", "info", "warning"],
        help="Logging level (default: info)",
    )
    return parser.parse_args()


def main() -> None:
    """
    Main entry point for the CLI script.
    Loads config, sets logging level, and trains model.
    """

    args = parse_args()

    logger = get_logger(__name__)
    logger.setLevel(args.log_level.upper())
    config_path: Path = Path(args.config)

    try:
        with open(config_path, "r") as f:
            config: Dict[str, Any] = yaml.safe_load(f)
    except Exception:
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    logger.info(f"Config loaded from: {config_path}")
    model, metric = train_model(config)

    logger.info("Training Complete")
    logger.info(f"AUC: {metric['auc']:.4f}, F1: {metric['f1']:.4f}")


if __name__ == "__main__":
    main()
