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

from claimflowengine.preprocessing.features import (
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
        if (
            "patient_gender" in df.columns
            and "billing_provider_specialty" in df.columns
        ):
            logger.info("Detected EDI 837 schema. Applying EDI feature engineering...")
            df = engineer_edi_features(df)
        else:
            logger.info(
                "Legacy schema detected. Skipping EDI-specific feature engineering..."
            )

        logger.info("Engineering common structured features...")
        df = engineer_features(df)

        logger.info("Applying transformers...")
        transformer = get_transformer_pipeline(df)
        transformed = transformer.fit_transform(df)
        transformed_df = pd.DataFrame(
            transformed, columns=transformer.get_feature_names_out()
        )

        logger.info(f"Saving processed data to {output_path}")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        transformed_df.to_csv(output_path, index=False)

        logger.info("Preprocessing complete.")
    except Exception as e:
        logger.info(f" Preprocessing failed: {e}")
        raise


if __name__ == "__main__":
    preprocess_and_save("data/raw_claims.csv", "data/processed_claims.csv")
