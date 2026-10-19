"""Configuration loader for the Customer Behavior Prediction Platform."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "config" / "config.yaml"

load_dotenv(PROJECT_ROOT / ".env")


@lru_cache(maxsize=4)
def load_config(path: str | None = None) -> dict[str, Any]:
    """Load the YAML configuration file.

    Resolution order: explicit ``path`` argument, the ``CONFIG_PATH``
    environment variable, then ``config/config.yaml``.
    """
    config_path = Path(path or os.getenv("CONFIG_PATH") or DEFAULT_CONFIG_PATH)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Config file is empty or invalid: {config_path}")
    return config


def get_config_value(key: str, default: Any = None, path: str | None = None) -> Any:
    """Read a nested value using dot notation, e.g. ``"features.cutoff_date"``."""
    value: Any = load_config(path)
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def resolve_path(relative_path: str | os.PathLike, create: bool = False) -> Path:
    """Convert a project-relative path to an absolute one.

    If ``create`` is True, the directory (or the file's parent directory)
    is created when missing.
    """
    path = Path(relative_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if create:
        target = path if not path.suffix else path.parent
        target.mkdir(parents=True, exist_ok=True)
    return path


def get_seed() -> int:
    """Return the global random seed from config (default 42)."""
    return int(get_config_value("project.seed", 42))