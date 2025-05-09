import yaml
from pathlib import Path
from typing import Dict, Any, Union


def load_config(path: Union[str, Path] = "config/config.yaml") -> Dict[str, Any]:
    """
    Load YAML configuration file from a relative or absolute path.

    Args:
        path (str or Path): Path to YAML config file

    Returns:
        dict: Parsed YAML config as a dictionary
    """
    config_path = Path(path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    with config_path.open("r") as f:
        return yaml.safe_load(f)
