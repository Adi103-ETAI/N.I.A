"""Model provider/catalog presets and metadata."""
from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict

from src.core.logger import setup_logger

logger = setup_logger("Models")

# Config data lives at src/core/config/defaults/
_CONFIG_DATA = Path(__file__).resolve().parents[1] / "core" / "config" / "defaults"

def _load_general_config() -> dict:
    """Load NIA general config (default provider, valid providers).

    Reads from src/core/config/data/nia/general.json.
    Returns an empty dict on any failure so module loading never crashes.
    """
    config_path = _CONFIG_DATA / "nia" / "general.json"
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load general.json: {e}")
        return {}

_GENERAL_CONFIG = _load_general_config()

# Default provider when no override is set
DEFAULT_PROVIDER = _GENERAL_CONFIG.get("DEFAULT_PROVIDER", "nvidia")

# Frozenset of valid provider identifiers for runtime validation
VALID_PROVIDERS = frozenset(_GENERAL_CONFIG.get("VALID_PROVIDERS", ["nvidia", "openai", "groq", "ollama"]))

class Provider(str, Enum):
    """Supported LLM providers."""
    NVIDIA = "nvidia"
    OPENAI = "openai"
    OLLAMA = "ollama"
    GROQ = "groq"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"

@dataclass
class ModelSpec:
    """Specification for an LLM model."""
    provider: Provider
    model_name: str
    display_name: str
    context_window: int = 4096
    supports_vision: bool = False
    supports_function_calling: bool = False
    is_local: bool = False
    cost_tier: str = "medium"  # 'free', 'low', 'medium', 'high'
    speed_tier: str = "medium"  # 'fast', 'medium', 'slow'

def _load_catalog() -> Dict[str, ModelSpec]:
    """Load model catalog from external JSON file.
    
    Returns:
        Dictionary mapping model keys to ModelSpec objects.
        
    Raises:
        FileNotFoundError: If catalog.json is missing.
        json.JSONDecodeError: If catalog.json has invalid JSON.
    """
    catalog_path = _CONFIG_DATA / "nia" / "models.json"
    
    if not catalog_path.exists():
        logger.warning("models.json not found, using empty catalog")
        return {}
    
    with open(catalog_path, "r", encoding="utf-8") as f:
        raw_catalog = json.load(f)
    
    # Convert raw dicts to ModelSpec objects
    catalog = {}
    for key, spec_dict in raw_catalog.items():
        catalog[key] = _spec_from_dict(spec_dict)
    
    logger.debug("Loaded %d models from src.models.json", len(catalog))
    return catalog

def _spec_from_dict(data: dict) -> ModelSpec:
    """Convert a dictionary to a ModelSpec object.
    
    Args:
        data: Dictionary with model specification fields.
        
    Returns:
        ModelSpec instance.
    """
    return ModelSpec(
        provider=Provider(data["provider"]),
        model_name=data["model_name"],
        display_name=data["display_name"],
        context_window=data.get("context_window", 4096),
        supports_vision=data.get("supports_vision", False),
        supports_function_calling=data.get("supports_function_calling", False),
        is_local=data.get("is_local", False),
        cost_tier=data.get("cost_tier", "medium"),
        speed_tier=data.get("speed_tier", "medium"),
    )

# Load catalog at module level (cached)
MODEL_CATALOG: Dict[str, ModelSpec] = _load_catalog()

__all__ = [
    "DEFAULT_PROVIDER",
    "VALID_PROVIDERS",
    "Provider",
    "ModelSpec",
    "MODEL_CATALOG",
]
