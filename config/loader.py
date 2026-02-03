"""
Configuration loader with validation.

Loads YAML configuration files and constructs validated Settings object.
Supports environment variable substitution using ${VAR_NAME} syntax.
"""

import os
import re
import yaml
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from config.base.settings import (
    Settings,
    NIAConfig,
    TARAConfig,
    IRISConfig,
    NOLAConfig,
    ModelConfig,
    ModelProviderConfig,
    DesktopConfig,
    UIAConfig,
    GatekeeperConfig,
    GraphConfig,
    MemoryConfig,
    RoutingConfig,
    RoutingKeywords,
    CommandCategories,
    SentryConfig,
    VisionTriggersConfig,
    SpeechConfig,
    HearingConfig,
)

logger = logging.getLogger(__name__)


def _substitute_env_vars(content: str) -> str:
    """
    Replace ${VAR_NAME} patterns with environment variable values.
    
    Args:
        content: String content with potential ${VAR} patterns
        
    Returns:
        Content with environment variables substituted
    """
    def replace_match(match):
        var_name = match.group(1)
        value = os.getenv(var_name)
        if value is None:
            # Keep original if env var not set
            return match.group(0)
        return value
    
    return re.sub(r'\$\{([^}]+)\}', replace_match, content)


def _load_yaml_file(file_path: Path) -> Dict[str, Any]:
    """
    Load a YAML file with environment variable substitution.
    
    Args:
        file_path: Path to YAML file
        
    Returns:
        Parsed YAML content as dict
    """
    if not file_path.exists():
        logger.warning(f"Config file not found: {file_path}")
        return {}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Substitute environment variables
        content = _substitute_env_vars(content)
        
        # Parse YAML
        data = yaml.safe_load(content)
        return data if data else {}
        
    except yaml.YAMLError as e:
        logger.error(f"YAML parse error in {file_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        raise


def _get_config_dir() -> Path:
    """Get the config directory path."""
    # Try relative to current working directory first
    config_dir = Path("config")
    if config_dir.exists():
        return config_dir
    
    # Try relative to this file's location
    config_dir = Path(__file__).parent
    if config_dir.exists():
        return config_dir
    
    raise FileNotFoundError("Config directory not found")


def load_settings() -> Settings:
    """
    Load and validate application settings from YAML files.
    
    Loads configuration from:
    - config/agents/nia.yaml
    - config/agents/tara.yaml
    - config/agents/iris.yaml
    - config/agents/nola.yaml
    - config/models.yaml
    - config/capabilities/desktop.yaml
    
    Returns:
        Validated Settings object
    """
    config_dir = _get_config_dir()
    
    logger.info(f"Loading configuration from: {config_dir}")
    
    # Load agent configs
    nia_data = _load_yaml_file(config_dir / "agents" / "nia.yaml")
    tara_data = _load_yaml_file(config_dir / "agents" / "tara.yaml")
    iris_data = _load_yaml_file(config_dir / "agents" / "iris.yaml")
    nola_data = _load_yaml_file(config_dir / "agents" / "nola.yaml")
    
    # Load model config
    models_data = _load_yaml_file(config_dir / "models.yaml")
    
    # Load capability configs
    desktop_data = _load_yaml_file(config_dir / "capabilities" / "desktop.yaml")
    
    # Build Settings object
    try:
        # Parse NIA config
        nia_config = NIAConfig(
            name=nia_data.get("name", "NIA"),
            version=nia_data.get("version", "4.0.0"),
            debug_mode=nia_data.get("debug_mode", False),
            log_level=nia_data.get("log_level", "INFO"),
            routing_mode=nia_data.get("routing_mode", "hybrid"),
            confidence_threshold=nia_data.get("confidence_threshold", 0.7),
            gatekeeper=GatekeeperConfig(**nia_data.get("gatekeeper", {})) if nia_data.get("gatekeeper") else GatekeeperConfig(),
            graph=GraphConfig(**nia_data.get("graph", {})) if nia_data.get("graph") else GraphConfig(),
            memory=MemoryConfig(**nia_data.get("memory", {})) if nia_data.get("memory") else MemoryConfig(),
            routing=RoutingConfig(
                tara_keywords=RoutingKeywords(**nia_data.get("routing", {}).get("tara_keywords", {})) if nia_data.get("routing", {}).get("tara_keywords") else RoutingKeywords(),
                iris_keywords=RoutingKeywords(**nia_data.get("routing", {}).get("iris_keywords", {})) if nia_data.get("routing", {}).get("iris_keywords") else RoutingKeywords(),
                general_fallback=nia_data.get("routing", {}).get("general_fallback", "general"),
            ) if nia_data.get("routing") else RoutingConfig(),
        )
        
        # Parse TARA config
        tara_config = TARAConfig(
            name=tara_data.get("name", "TARA"),
            version=tara_data.get("version", "4.0.0"),
            debug_mode=tara_data.get("debug_mode", False),
            log_level=tara_data.get("log_level", "INFO"),
            max_tool_retries=tara_data.get("max_tool_retries", 3),
            tool_timeout=tara_data.get("tool_timeout", 30),
            commands=CommandCategories(**tara_data.get("commands", {})) if tara_data.get("commands") else CommandCategories(),
        )
        
        # Parse IRIS config
        iris_config = IRISConfig(
            name=iris_data.get("name", "IRIS"),
            version=iris_data.get("version", "4.0.0"),
            debug_mode=iris_data.get("debug_mode", False),
            log_level=iris_data.get("log_level", "INFO"),
            vision_model=iris_data.get("vision_model", "meta/llama-3.2-90b-vision-instruct"),
            confidence_threshold=iris_data.get("confidence_threshold", 0.6),
            sentry=SentryConfig(**iris_data.get("sentry", {})) if iris_data.get("sentry") else SentryConfig(),
            triggers=VisionTriggersConfig(**iris_data.get("triggers", {})) if iris_data.get("triggers") else VisionTriggersConfig(),
        )
        
        # Parse NOLA config
        nola_config = NOLAConfig(
            name=nola_data.get("name", "NOLA"),
            version=nola_data.get("version", "4.0.0"),
            debug_mode=nola_data.get("debug_mode", False),
            log_level=nola_data.get("log_level", "INFO"),
            wake_word=nola_data.get("wake_word", "hey nia"),
            tts_voice=nola_data.get("tts_voice", "af_sarah"),
            stt_model=nola_data.get("stt_model", "whisper-large-v3"),
            speech=SpeechConfig(**nola_data.get("speech", {})) if nola_data.get("speech") else SpeechConfig(),
            hearing=HearingConfig(**nola_data.get("hearing", {})) if nola_data.get("hearing") else HearingConfig(),
        )
        
        # Parse model configs
        models = {}
        for model_id, model_data in models_data.get("models", {}).items():
            models[model_id] = ModelConfig(**model_data)
        
        providers = {}
        for provider_id, provider_data in models_data.get("providers", {}).items():
            providers[provider_id] = ModelProviderConfig(**provider_data)
        
        # Parse desktop config
        desktop_config = DesktopConfig(
            system_apps=desktop_data.get("system_apps", []),
            custom_aliases=desktop_data.get("custom_aliases", {}),
        )
        
        uia_config = UIAConfig(**desktop_data.get("uia", {})) if desktop_data.get("uia") else UIAConfig()
        
        # Create Settings object
        settings = Settings(
            nia=nia_config,
            tara=tara_config,
            iris=iris_config,
            nola=nola_config,
            default_provider=models_data.get("default_provider", "nvidia"),
            valid_providers=models_data.get("valid_providers", ["nvidia", "openai", "groq", "ollama"]),
            models=models,
            providers=providers,
            fallback_chain=models_data.get("fallback_chain", ["nvidia", "openai", "ollama"]),
            desktop=desktop_config,
            uia=uia_config,
        )
        
        logger.info("✓ Configuration loaded and validated successfully")
        return settings
        
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        raise


def reload_settings() -> Settings:
    """Force reload settings from disk."""
    from config.base.settings import reset_settings
    reset_settings()
    return load_settings()


# Convenience function
def get_config() -> Settings:
    """Get configuration (alias for get_settings)."""
    from config.base.settings import get_settings
    return get_settings()
