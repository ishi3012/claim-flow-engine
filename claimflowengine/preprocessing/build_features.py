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

Author: ClaimFlowEngine Team
"""

from pathlib import Path

import pandas as pd
from category_encoders import TargetEncoder
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from claimflowengine.preprocessing.feature_engineering import (
    engineer_edi_features,
    engineer_features,
)
from claimflowengine.preprocessing.text_cleaning import clean_text_fields
from claimflowengine.utils.functions import convert_to_int
from claimflowengine.utils.logger import get_logger

# Initialize Logging
logger = get_logger("preprocessing")


# ------------------------- Core Logic -------------------------
def preprocess_and_save(raw_path: str, output_path: str, transformer_dir: str) -> None:
    logger.info(f"Loading raw data from {raw_path}")
    df = pd.read_csv(raw_path)

    # Validate required label column
    assert (
        "denied" in df.columns
    ), "'denied' column must be present in raw data for supervised learning."

    logger.info("Cleaning text fields...")
    df = clean_text_fields(df)

    # Apply EDI-specific features if schema coverage is sufficient
    edi_fields = [
        "patient_dob",
        "service_date",
        "patient_gender",
        "facility_code",
        "billing_provider_specialty",
        "claim_type",
        "prior_authorization",
        "accident_indicator",
    ]
    edi_coverage = len([col for col in edi_fields if col in df.columns]) / len(
        edi_fields
    )
    if edi_coverage >= 0.5:
        logger.info(
            f"Detected EDI schema ({edi_coverage:.0%} coverage) — applying EDI features"
        )
        df = engineer_edi_features(df)

    # Extract label and apply full feature engineering
    y = df["denied"]
    logger.info("Engineering structured features...")
    df = engineer_features(df, y)

    logger.info("Splitting features and target...")
    X = df.drop(columns=["denied"])

    # ------------------------- Target Encode Categoricals -------------------------
    categorical_cols = [
        "payer_id",
        "provider_type",
        "plan_type",
        "claim_type",
        "billing_provider_specialty",
        "facility_code",
        "diagnosis_code",
        "procedure_code",
    ]
    categorical_cols = [col for col in categorical_cols if col in X.columns]

    logger.info("Applying TargetEncoder to categorical features...")
    target_encoder = TargetEncoder()
    X[categorical_cols] = target_encoder.fit_transform(X[categorical_cols], y)

    # ------------------------- Numeric + Boolean -------------------------
    numeric_features = [
        "claim_age_days",
        "patient_age",
        "total_charge_amount",
        "days_to_submission",
        "payer_deny_rate",
        "provider_deny_rate",
        "resubmission_rate_by_payer",
        "followup_intensity_score",
    ]
    boolean_features = [
        "is_resubmission",
        "contains_auth_term",
        "prior_authorization",
        "accident_indicator",
        "high_charge_flag",
    ]
    numeric_features = [f for f in numeric_features if f in X.columns]
    boolean_features = [f for f in boolean_features if f in X.columns]

    logger.info("Building preprocessing pipeline...")
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="mean")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "bool",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "to_int",
                            FunctionTransformer(
                                convert_to_int,
                                validate=False,
                                feature_names_out="one-to-one",
                            ),
                        ),
                    ]
                ),
                boolean_features,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    logger.info("Fitting numeric+boolean transformer...")
    transformed_array = preprocessor.fit_transform(X)
    transformed_df = pd.DataFrame(
        transformed_array, columns=preprocessor.get_feature_names_out()
    )

    logger.info("Reconstructing final training matrix...")
    encoded_cat_df = X[categorical_cols].reset_index(drop=True)
    final_df = pd.concat([encoded_cat_df, transformed_df], axis=1)
    final_df["denied"] = y.values

    logger.info(f"Saving processed features to: {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False)

    logger.info("Saving transformers...")
    Path(transformer_dir).mkdir(parents=True, exist_ok=True)
    dump(preprocessor, f"{transformer_dir}/numeric_transformer.joblib")
    dump(target_encoder, f"{transformer_dir}/target_encoder.joblib")

    logger.info("Preprocessing complete.")


# ------------------------- CLI -------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw_claims.csv")
    parser.add_argument("--output", default="data/processed_claims.csv")
    parser.add_argument("--transformer_dir", default="models")

    args = parser.parse_args()
    preprocess_and_save(args.input, args.output, args.transformer_dir)
