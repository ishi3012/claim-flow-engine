"""
Module: build_features.py

Description:
This script performs preprocessing and feature
engineering on raw healthcare claims data
to produce a clean dataset for downstream
denial prediction modeling.

Features:
- Loads data/raw_claims.csv
- Applies text cleaning to denial_reason and notes
- Supports legacy and EDI 837 schemas
- Computes structured features (claim age, denial history, patient age, etc.)
- Applies transformers (encoding, imputation)
- Saves data/processed_claims.csv
- Can be run as a script or imported as a module

Usage:
    $ python -m claimflowengine.preprocessing.build_features

Inputs:
    data/raw_claims.csv

Outputs:
    data/processed_claims.csv

Author: ClaimFlowEngine Team
"""

import logging
from pathlib import Path

import pandas as pd

from claimflowengine.preprocessing.feature_engineering import (
    engineer_edi_features,
    engineer_features,
)
from claimflowengine.preprocessing.text_cleaning import clean_text_fields
from claimflowengine.preprocessing.transformers import get_transformer_pipeline

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def preprocess_and_save(raw_path: str, output_path: str) -> None:
    """
    Loads raw claim data, processes it, and saves the output.
    Args:
        raw_path (str): Path to raw claims CSV.
        output_path (str): Path to save processed claims CSV.
    """
    try:
        logger.info(f"Loading raw data from {raw_path}")
        df = pd.read_csv(raw_path)
    except FileNotFoundError:
        logger.error(f"ERROR: File not found: {raw_path}")
        raise
    except pd.errors.ParserError:
        logger.error(f"ERROR: Failed to parse csv: {raw_path}")
        raise
    except Exception:
        logger.error(f"ERROR: Unexpected error loading data: {raw_path}")

    try:
        logger.info("Cleaning text fields...")
        df = clean_text_fields(df)

        # Detect is EDI 837 fields are available in the dataset.
        edi_critical_fields = [
            "patient_dob",
            "service_date",
            "patient_gender",
            "facility_code",
            "billing_provider_specialty",
            "claim_type",
            "prior_authorization",
            "accident_indicator",
        ]

        edi_available = [col for col in edi_critical_fields if col in df.columns]
        edi_coverage = len(edi_available) / len(edi_critical_fields)
        if edi_coverage >= 0.5:  # ✅ at least 50% of EDI features present
            logger.info(
                f"Detected partial EDI schema ({edi_coverage:.0%} coverage). "
                + "Applying EDI feature engineering..."
            )
            df = engineer_edi_features(df)
        else:
            logger.info(
                f"EDI schema not detected (only {edi_coverage:.0%} coverage)."
                + "Skipping EDI features."
            )

        logger.info("Engineering common structured features...")
        df = engineer_features(df)

        logger.info("Applying transformers...")
        transformer = get_transformer_pipeline(df)
        transformed = transformer.fit_transform(df)
        transformed_df = pd.DataFrame(
            transformed, columns=transformer.get_feature_names_out()
        )
        if "denied" in df.columns:
            assert len(df) == len(
                transformed_df
            ), "Row count mismatch before and after transformation"
            transformed_df["denied"] = df["denied"].values

        logger.info(f"Saving processed data to {output_path}")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        transformed_df.to_csv(output_path, index=False)

        logger.info("Preprocessing complete.")
    except Exception as e:
        logger.info(f" Preprocessing failed: {e}")
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess raw claims csv.")
    parser.add_argument(
        "--input",
        default="claimflowengine/data/raw_claims.csv",
        help="Path to raw claims CSV "
        + "(default: claimflowengine/data/raw_claims.csv)",
    )
    parser.add_argument(
        "--output",
        default="claimflowengine/data/processed_claims.csv",
        help="Path to save processed CSV "
        + "(default: claimflowengine/data/processed_claims.csv)",
    )
    args = parser.parse_args()

    # Ensure input exists
    assert Path(args.input).exists(), f"Input file {args.input} not found"

    # Create output directory if needed
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    preprocess_and_save(args.input, args.output)
