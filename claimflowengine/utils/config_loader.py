# utils/config_loader.py

import yaml

def load_config(path='config/config.yaml'):
    """
    Load YAML configuration file from given path.
    """
    with open(path, 'r') as file:
        return yaml.safe_load(file)
