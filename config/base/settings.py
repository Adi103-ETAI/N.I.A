"""
Configuration schemas using Pydantic.

Provides type-safe configuration models for all N.I.A. components.
Supports YAML config files with environment variable overrides.
"""

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal, Optional, Dict, List, Any
from pathlib import Path


# ============================================================================
# Model Configuration
# ============================================================================

class ModelConfig(BaseModel):
    """Configuration for a single LLM model."""
    
    provider: str
    model_name: str
    display_name: str
    context_window: int = 128000
    supports_vision: bool = False
    supports_function_calling: bool = True
    is_local: bool = False
    cost_tier: Literal["free", "low", "high"] = "free"
    speed_tier: Literal["slow", "medium", "fast"] = "medium"


class ModelProviderConfig(BaseModel):
    """LLM provider configuration."""
    
    api_key: Optional[str] = None
    model: str
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(2000, ge=1)
    base_url: Optional[str] = None


# ============================================================================
# Agent Configurations
# ============================================================================

class GatekeeperConfig(BaseModel):
    """Gatekeeper (routing) configuration."""
    
    enabled: bool = True
    fallback_agent: str = "chat"


class GraphConfig(BaseModel):
    """LangGraph configuration."""
    
    max_iterations: int = Field(10, ge=1)
    timeout_seconds: int = Field(30, ge=1)


class MemoryConfig(BaseModel):
    """Memory system configuration."""
    
    enabled: bool = True
    max_conversation_length: int = Field(50, ge=1)


class RoutingKeywords(BaseModel):
    """Keywords for routing queries to agents."""
    
    categories: Dict[str, List[str]] = Field(default_factory=dict)


class RoutingConfig(BaseModel):
    """Routing configuration for NIA."""
    
    tara_keywords: RoutingKeywords = Field(default_factory=RoutingKeywords)
    iris_keywords: RoutingKeywords = Field(default_factory=RoutingKeywords)
    general_fallback: str = "general"


class NIAConfig(BaseModel):
    """NIA supervisor agent configuration."""
    
    name: str = "NIA"
    version: str = "4.0.0"
    debug_mode: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    
    # Routing
    routing_mode: Literal["llm", "rules", "hybrid"] = "hybrid"
    confidence_threshold: float = Field(0.7, ge=0.0, le=1.0)
    
    # Sub-configs
    gatekeeper: GatekeeperConfig = Field(default_factory=GatekeeperConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)


class CommandCategories(BaseModel):
    """Command categories for TARA."""
    
    mic_control: Dict[str, List[str]] = Field(default_factory=dict)
    iris_control: Dict[str, List[str]] = Field(default_factory=dict)
    speaker_control: Dict[str, List[str]] = Field(default_factory=dict)
    tts_control: Dict[str, List[str]] = Field(default_factory=dict)
    system_control: Dict[str, List[str]] = Field(default_factory=dict)


class TARAConfig(BaseModel):
    """TARA tool agent configuration."""
    
    name: str = "TARA"
    version: str = "4.0.0"
    debug_mode: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    
    max_tool_retries: int = Field(3, ge=1)
    tool_timeout: int = Field(30, ge=1)
    
    # Command patterns
    commands: CommandCategories = Field(default_factory=CommandCategories)


class SentryConfig(BaseModel):
    """IRIS Sentry mode configuration."""
    
    enabled: bool = True
    scan_interval: int = Field(8, ge=1)
    triggers: Dict[str, List[str]] = Field(default_factory=dict)
    ignore_patterns: List[str] = Field(default_factory=list)


class VisionTriggersConfig(BaseModel):
    """Vision trigger keywords."""
    
    screen: List[str] = Field(default_factory=list)
    camera: List[str] = Field(default_factory=list)
    actions: List[str] = Field(default_factory=list)


class IRISConfig(BaseModel):
    """IRIS vision agent configuration."""
    
    name: str = "IRIS"
    version: str = "4.0.0"
    debug_mode: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    
    vision_model: str = "meta/llama-3.2-90b-vision-instruct"
    confidence_threshold: float = Field(0.6, ge=0.0, le=1.0)
    
    sentry: SentryConfig = Field(default_factory=SentryConfig)
    triggers: VisionTriggersConfig = Field(default_factory=VisionTriggersConfig)


class SpeechConfig(BaseModel):
    """Speech (TTS) settings."""
    
    playback_poll_interval: float = 0.1
    piper_timeout_sec: int = 30


class HearingConfig(BaseModel):
    """Hearing (STT) settings."""
    
    model_dir_name: str = "vosk_model"


class NOLAConfig(BaseModel):
    """NOLA voice agent configuration."""
    
    name: str = "NOLA"
    version: str = "4.0.0"
    debug_mode: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    
    wake_word: str = "hey nia"
    tts_voice: str = "af_sarah"
    stt_model: str = "whisper-large-v3"
    
    speech: SpeechConfig = Field(default_factory=SpeechConfig)
    hearing: HearingConfig = Field(default_factory=HearingConfig)


# ============================================================================
# Capability Configurations
# ============================================================================

class DesktopConfig(BaseModel):
    """Desktop automation capability config."""
    
    system_apps: List[str] = Field(default_factory=lambda: ["notepad", "calc", "cmd", "explorer"])
    custom_aliases: Dict[str, str] = Field(default_factory=dict)


class UIAConfig(BaseModel):
    """UI Automation configuration."""
    
    actionable_types: List[str] = Field(default_factory=list)
    skip_types: List[str] = Field(default_factory=list)
    max_elements: int = Field(100, ge=1)


# ============================================================================
# Main Settings Class
# ============================================================================

class Settings(BaseSettings):
    """
    Application settings.
    
    Loads from YAML files with environment variable overrides.
    Environment variables use NIA_ prefix (e.g., NIA_DEBUG=true).
    """
    
    model_config = SettingsConfigDict(
        env_prefix="NIA_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    # Agent configs
    nia: NIAConfig = Field(default_factory=NIAConfig)
    tara: TARAConfig = Field(default_factory=TARAConfig)
    iris: IRISConfig = Field(default_factory=IRISConfig)
    nola: NOLAConfig = Field(default_factory=NOLAConfig)
    
    # Model configs
    default_provider: str = "nvidia"
    valid_providers: List[str] = Field(default_factory=lambda: ["nvidia", "openai", "groq", "ollama"])
    models: Dict[str, ModelConfig] = Field(default_factory=dict)
    providers: Dict[str, ModelProviderConfig] = Field(default_factory=dict)
    fallback_chain: List[str] = Field(default_factory=lambda: ["nvidia", "openai", "ollama"])
    
    # Capability configs
    desktop: DesktopConfig = Field(default_factory=DesktopConfig)
    uia: UIAConfig = Field(default_factory=UIAConfig)


# ============================================================================
# Singleton Access
# ============================================================================

_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get settings singleton (lazy load)."""
    global _settings
    if _settings is None:
        from config.loader import load_settings
        _settings = load_settings()
    return _settings


def reset_settings() -> None:
    """Reset settings singleton (for testing)."""
    global _settings
    _settings = None
