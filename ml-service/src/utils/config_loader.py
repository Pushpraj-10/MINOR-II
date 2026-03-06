"""Configuration loading and validation utilities."""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file.

    Args:
        config_path: Path to configuration file

    Returns:
        Dictionary with configuration parameters

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If file format is unsupported
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    suffix = path.suffix.lower()

    if suffix in [".yaml", ".yml"]:
        with open(path, "r") as f:
            config = yaml.safe_load(f)
    elif suffix == ".json":
        with open(path, "r") as f:
            config = json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {suffix}")

    logger.info(f"Loaded config from {config_path}")
    return config


def validate_config(config: Dict, required_keys: list) -> None:
    """
    Validate that config contains all required keys.

    Args:
        config: Configuration dictionary
        required_keys: List of required top-level keys

    Raises:
        KeyError: If a required key is missing
    """
    missing = [k for k in required_keys if k not in config]
    if missing:
        raise KeyError(f"Missing required config keys: {missing}")
    logger.debug(f"Config validated: {len(required_keys)} required keys present")


def merge_configs(base: Dict, override: Dict) -> Dict:
    """
    Deep merge two configuration dictionaries.

    Args:
        base: Base configuration
        override: Override values (takes priority)

    Returns:
        Merged configuration dictionary
    """
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged
