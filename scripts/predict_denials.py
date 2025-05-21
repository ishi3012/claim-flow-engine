"""
predict_denials.py — CLI Script to Predict Claim Denials

This script:
- Loads a trained denial prediction model and transformers
- Reads raw claim data from a CSV
- Applies full preprocessing and prediction
- Saves results to an output CSV

Usage:
python scripts/predict_denials.py \
    --input data/sample_input_for_inference.csv \
    --output output/predictions.csv

Author: ClaimFlowEngine Team
"""

import argparse
from pathlib import Path

import pandas as pd

from claimflowengine.configs.paths import INFERENCE_INPUT_PATH, INFERENCE_OUTPUT_PATH
from claimflowengine.inference.loader import load_model
from claimflowengine.inference.predictor import predict_claims
from claimflowengine.utils.logger import get_logger

# Initialize Logging
logger = get_logger("inference")


# ------------------------- Main Function -------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict claim denials from raw input data."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=INFERENCE_INPUT_PATH,
        help="Path to raw claim CSV file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=INFERENCE_OUTPUT_PATH,
        help="Path to save predictions.",
    )

    args = parser.parse_args()

    logger.info("Reading input file...")
    df_input = pd.read_csv(args.input)

    logger.info(
        f"Input File data: {df_input.shape}" + " rows x {df_input.shape[1]} columns"
    )

    logger.info("Loading model and transformers...")
    model, target_encoder, numeric_transformer = load_model()

    logger.info("Running predictions...")
    predictions = predict_claims(
        raw_data=df_input,
        model=model,
        target_encoder=target_encoder,
        numeric_transformer=numeric_transformer,
    )

    # Merge predictions back to original input
    df_output = df_input.copy()
    df_pred = pd.DataFrame(predictions)
    df_output = pd.concat([df_output.reset_index(drop=True), df_pred], axis=1)

    logger.info(f"Saving output to: {args.output}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df_output.to_csv(args.output, index=False)

    logger.info("Inference completed successfully.")


if __name__ == "__main__":
    main()
