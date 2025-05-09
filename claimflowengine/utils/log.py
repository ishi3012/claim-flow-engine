"""
log.py

Provides a reusable, configurable logger that outputs to both console and file.
Log file is saved to Logs/train.log by default.

"""

import logging
from pathlib import Path


def get_logger(
    name: str = __name__, log_file: str = "logs/train.log"
) -> logging.Logger:
    """
    Creates a logger with dual output: console + file.

    Args:
        name (str) : Name of the logger (usually __name__)
        log_file (str) : Path to the log file.

    Returns:
        logging.Logger : Configured logger instance.
    """

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Console handler

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # File Handler
    file_handler = logging.FileHandler(log_path, mode="a")
    file_handler.setFormatter(formatter)

    # Setup Logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger
