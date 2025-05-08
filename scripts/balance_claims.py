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
