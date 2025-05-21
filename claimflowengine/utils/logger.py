"""
Module: logger.py — Central Logging Configuration

Creates and configures stage-aware loggers for each pipeline module
(e.g., training, inference, clustering, etc.).

Usage:
from claimflowengine.utils.logger import get_logger
logger = get_logger("inference")

Author: ClaimFlowEngine Team
"""

import logging
from pathlib import Path


def get_logger(stage: str = "default") -> logging.Logger:
    """
    Creates and returns a logger that writes to logs/{stage}.log
    Args:
        stage (str): Name of pipeline stage (e.g., "inference", "training")
    Returns:
        logging.Logger: Configured logger instance
    """
    Path("logs").mkdir(exist_ok=True)
    logger = logging.getLogger(stage)

    # Prevent duplicate handlers
    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s — %(levelname)s — %(message)s")

        # File handler
        file_handler = logging.FileHandler(f"logs/{stage}.log", mode="a")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Stream handler
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger
