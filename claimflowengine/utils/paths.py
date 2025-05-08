"""
paths.py

Defines and manages key project-level paths for data, models, logs, and configuration.

This module should be used across scripts, notebooks, and modules to ensure consistent
and portable filesystem access, particularly for transitioning between local and cloud (e.g., GCP Vertex AI) environments.

Example Usage:
    from claimflowengine.utils.paths import DATA_DIR, CONFIG_PATH

    df = pd.read_csv(DATA_DIR / "processed" / "claims.csv")
"""

from pathlib import Path

# Project root
ROOT: Path = Path(__file__).resolve().parent.parent.parent

# Configuration file path
CONFIG_PATH: Path = ROOT / "config" / "config.yaml"

# Directory for saving trained models
MODEL_DIR: Path = ROOT / "models"

# Directory for storing experiment logs
LOG_DIR: Path = ROOT / "logs"

# Directory containing input and processed data
DATA_DIR: Path = ROOT / "data"


def _ensure_directories(paths: list[Path]) -> None:
    """
    Create directories if they do not already exist.

    """
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


# Ensure required directories exist during import
_ensure_directories([MODEL_DIR, LOG_DIR, DATA_DIR])
