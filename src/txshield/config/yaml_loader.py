"""Small, explicit helpers for loading command-line YAML configuration."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping, returning an empty mapping for an empty file."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as source:
        loaded = yaml.safe_load(source)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Configuration root must be a mapping: {config_path}")
    return loaded


def cli_or_config(cli_value: Any, config: dict[str, Any], key: str, default: Any = None) -> Any:
    """Return an explicit CLI value before a YAML value and then a default."""
    if cli_value is not None:
        return cli_value
    return config.get(key, default)
