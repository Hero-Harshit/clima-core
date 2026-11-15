"""
Configuration loader for Clima-Core.
Reads and validates project and data sources YAML configurations.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import yaml
from climacore.common.exceptions import ConfigurationError


def load_yaml(file_path: Path | str) -> Dict[str, Any]:
    """Safely load a YAML configuration file."""
    path = Path(file_path)
    if not path.is_file():
        raise ConfigurationError(f"Config file does not exist: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return data or {}
    except Exception as exc:
        raise ConfigurationError(f"Failed to parse YAML file {path}: {exc}") from exc


def get_project_root() -> Path:
    """Return the repository root directory."""
    return Path(__file__).resolve().parent.parent.parent.parent


def get_data_sources_config(config_dir: Optional[Path | str] = None) -> Dict[str, Any]:
    """Load configs/data_sources.yaml."""
    base_dir = Path(config_dir) if config_dir else get_project_root() / "configs"
    return load_yaml(base_dir / "data_sources.yaml")


def get_project_config(config_dir: Optional[Path | str] = None) -> Dict[str, Any]:
    """Load configs/project.yaml."""
    base_dir = Path(config_dir) if config_dir else get_project_root() / "configs"
    return load_yaml(base_dir / "project.yaml")
