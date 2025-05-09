#!/usr/bin/env python

"""
train_denial_benchmark.py

Run model benchmarking for claim denial prediction using multiple classifiers.
Evaluates models based on a composite score (AUC, F1, Recall) defined in config.
Selects the best-performing model, saves it, and logs performance.

Intended for local experimentation or model selection prior to deployment.

Usage:
    python scripts/train_denial_benchmark.py --config config/config.yaml --log_level info
"""

import argparse
import yaml
import sys
from pathlib import Path
from typing import Dict, Any

from claimflowengine.utils.log import get_logger
from claimflowengine.utils.paths import CONFIG_PATH, MODEL_DIR
from scripts.train_denial_predictor import (
    load_data,
    get_models,
    evaluate_models,
    save_best_model,
)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Benchmark models for denial prediction"
    )
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
    Run benchmarking and model selection.
    Loads data, evaluates candidate models, and saves the best.
    """
    args = parse_args()

    logger = get_logger(__name__)
    logger.setLevel(args.log_level.upper())

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        config: Dict[str, Any] = yaml.safe_load(f)

    logger.info(f"Config loaded from: {config_path}")

    # Load data
    try:
        X, y = load_data(Path(config["data"]["balanced_path"]))
    except Exception:
        logger.exception("Failed to load training data.")
        sys.exit(1)

    logger.info(f"Data shape: {X.shape}, Labels shape: {y.shape}")

    # Get candidate models
    models = get_models(config.get("random_state", 42))

    # Evaluate
    results = evaluate_models(models, X, y)

    # Define results path
    results_path = MODEL_DIR / "model_benchmark_results.csv"

    # Save best model and full benchmark
    save_best_model(models, results, X, y, MODEL_DIR, results_path=results_path)

    logger.info("Benchmarking complete. Best model saved.")


if __name__ == "__main__":
    main()
