from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML experiment configuration and resolve its data paths."""
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")
    project_root = config_path.parent.parent
    config["_config_path"] = config_path
    config["_project_root"] = project_root
    return config


def data_path(config: dict[str, Any], relative_path: str | Path) -> Path:
    """Resolve a data path relative to the repository root, never the caller cwd."""
    path = Path(relative_path).expanduser()
    if path.is_absolute():
        return path
    return Path(config["_project_root"]) / path


def seed_everything(seed: int) -> None:
    """Seed Python and NumPy; optional frameworks are seeded by their callers."""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
