"""DyGenePT configuration loader."""

import os
from pathlib import Path
from omegaconf import OmegaConf, DictConfig


def load_config(config_path: str = "configs/default.yaml") -> DictConfig:
    """Load configuration from YAML file and resolve environment variables.

    Environment variables in YAML should use OmegaConf's native syntax:
        ${oc.env:VAR_NAME}       — raises error if not set
        ${oc.env:VAR_NAME,}      — empty string default if not set

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Resolved OmegaConf DictConfig object.
    """
    cfg = OmegaConf.load(config_path)
    # OmegaConf's built-in oc.env resolver handles ${oc.env:VAR} automatically.
    # Force full resolution so downstream code gets plain strings.
    OmegaConf.resolve(cfg)
    return cfg


def get_project_root() -> Path:
    """Return the project root directory (parent of src/)."""
    return Path(__file__).resolve().parent.parent
