"""
balance_claims.py

This script performs class balancing on healthcare claims data used for denial prediction modeling
in the ClaimFlowEngine pipeline. It up-samples the minority class (`denial_flag == 0`) to mitigate
class imbalance issues during model training.

Usage:
    python balance_claims.py --factor 2.0 --config config/config.yaml

Core Functionality:
    • Loads the ML-ready claims dataset defined in the config file.
    • Identifies the minority and majority classes based on the `denial_flag` column.
    • Upsamples the minority class by a configurable factor using bootstrapped resampling.
    • Outputs a balanced dataset to the path specified in the config.

Args:
    --factor (float): Upsampling multiplier for the minority class (default: 2.0).
    --config (str): Path to YAML config file with input and output data paths.

Inputs:
    - data/processed/claims.csv (from config['data']['default_path'])

Outputs:
    - data/processed/balanced_claims.csv (to config['data']['balanced_path'])

Notes:
    - The upsampling is done using scikit-learn’s `resample` function with replacement.
    - Output is shuffled to maintain random distribution before saving.
    - The script is part of the preprocessing pipeline and should be run prior to model training.

Example:
    $ python balance_claims.py --factor 3.0 --config config/config.yaml

Author: Claim Flow Engine Project
"""

import argparse
import pandas as pd
from sklearn.utils import resample
import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r") as file:
        return yaml.safe_load(file)


def balance_by_upsampling(
    df: pd.DataFrame, target: str = "denial_flag", factor: float = 2.0
) -> pd.DataFrame:
    """
    Randomly upsample the minority class by a specified factor.
    """
    majority_class = df[df[target] == 1]
    minority_class = df[df[target] == 0]

    print(
        f"Upsampling {len(minority_class)} → {int(len(minority_class) * factor)} rows (factor={factor})"
    )

    upsampled = resample(
        minority_class,
        replace=True,
        n_samples=int(len(minority_class) * factor),
        random_state=42,
    )

    df_balanced = pd.concat([majority_class, upsampled])
    return df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--factor", type=float, default=2.0, help="Upsample factor for minority class"
    )
    parser.add_argument(
        "--config", default="config/config.yaml", help="Path to config YAML"
    )
    args = parser.parse_args()

    # Get project root
    ROOT = Path(__file__).resolve().parent.parent

    # Load config and resolve paths
    config = load_config(ROOT / args.config)
    default_path = (ROOT / config["data"]["default_path"]).resolve()
    balanced_path = (ROOT / config["data"]["balanced_path"]).resolve()

    if not default_path.exists():
        raise FileNotFoundError(f"Input file not found: {default_path}")

    print(f"Loading input data from: {default_path}")
    df = pd.read_csv(default_path)

    # Perform balancing
    df_balanced = balance_by_upsampling(df, factor=args.factor)

    # Save to balanced path
    balanced_path.parent.mkdir(parents=True, exist_ok=True)
    df_balanced.to_csv(balanced_path, index=False)

    print(f"Balanced dataset saved to: {balanced_path}")
    print("Class distribution after balancing:")
    print(df_balanced["denial_flag"].value_counts(normalize=True).rename("proportion"))
