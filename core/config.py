"""N.I.A. Centralized Configuration Module.

Uses pydantic-settings to provide type-safe configuration with automatic
.env file loading. All configuration values should be accessed via the
global `settings` instance.

Usage:
    from core.config import settings
    
    # Access configuration values
    api_key = settings.NVIDIA_API_KEY.get_secret_value()
    if settings.DEBUG:
        print("Debug mode enabled")
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Union

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# =============================================================================
# Base Directory Detection
# =============================================================================

def _get_base_dir() -> Path:
    """Determine the project base directory."""
    # Start from this file's location and go up to find the project root
    current = Path(__file__).resolve().parent.parent
    return current


# =============================================================================
# Settings Class
# =============================================================================

class Settings(BaseSettings):
    """Centralized configuration for N.I.A.
    
    All settings can be overridden via environment variables or .env file.
    Environment variables take precedence over .env values.
    
    Example .env file:
        NVIDIA_API_KEY=nvapi-xxx
        DEBUG=true
        WAKE_WORDS=nia,jarvis,hey nia
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # =========================================================================
    # Paths
    # =========================================================================
    
    BASE_DIR: Path = Field(
        default_factory=_get_base_dir,
        description="Project root directory",
    )
    
    LOG_DIR: Path = Field(
        default=Path("logs"),
        description="Directory for log files",
    )
    
    MODEL_DIR: Path = Field(
        default=Path("models"),
        description="Directory for ML models",
    )
    
    DATA_DIR: Path = Field(
        default=Path("data"),
        description="Directory for data files (memory, state, cache)",
    )
    
    SOUNDS_DIR: Path = Field(
        default=Path("sounds"),
        description="Directory for audio files",
    )
    
    # =========================================================================
    # Voice Settings
    # =========================================================================
    
    # Using Union[str, List[str]] to allow comma-separated string from .env
    # The validator will convert string to list
    WAKE_WORDS: Union[str, List[str]] = Field(
        default=["nia", "jarvis", "hey nia"],
        description="Wake words to activate voice mode",
    )
    
    TTS_VOICE: str = Field(
        default="en-US-AriaNeural",
        description="Edge TTS voice name",
    )
    
    VOICE_ENABLED: bool = Field(
        default=True,
        description="Enable voice mode by default",
    )
    
    WAKE_WORD_TIMEOUT: float = Field(
        default=30.0,
        description="Seconds before returning to sleep after wake word",
    )
    
    # =========================================================================
    # AI Provider Settings
    # =========================================================================
    
    NVIDIA_API_KEY: SecretStr = Field(
        default=SecretStr(""),
        description="NVIDIA NIM API key (required)",
    )
    
    OPENAI_API_KEY: Optional[SecretStr] = Field(
        default=None,
        description="OpenAI API key (optional fallback)",
    )
    
    HUGGINGFACE_API_KEY: Optional[SecretStr] = Field(
        default=None,
        description="HuggingFace API key (optional)",
    )
    
    OLLAMA_HOST: str = Field(
        default="http://localhost:11434",
        description="Ollama server URL for local models",
    )
    
    # =========================================================================
    # LLM Model Selection
    # =========================================================================
    
    LLM_MODEL: str = Field(
        default="meta/llama-3.1-70b-instruct",
        description="Primary LLM model for NIA brain",
    )
    
    LLM_MODEL_SMART: str = Field(
        default="meta/llama-3.1-405b-instruct",
        description="High-quality LLM for complex reasoning",
    )
    
    LLM_MODEL_FAST: str = Field(
        default="meta/llama-3.1-8b-instruct",
        description="Fast LLM for simple tasks",
    )
    
    LLM_MODEL_VISION: str = Field(
        default="meta/llama-3.2-90b-vision-instruct",
        description="Vision-capable LLM for IRIS",
    )
    
    LLM_TEMPERATURE: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="LLM temperature for response generation",
    )
    
    # =========================================================================
    # System Settings
    # =========================================================================
    
    DEBUG: bool = Field(
        default=False,
        description="Enable debug mode with verbose logging",
    )
    
    MEMORY_RETENTION_DAYS: int = Field(
        default=7,
        ge=1,
        le=365,
        description="Days to retain conversation history",
    )
    
    MEMORY_DB_PATH: Path = Field(
        default=Path("data/memory.db"),
        description="Path to SQLite memory database",
    )
    
    STATE_DB_PATH: Path = Field(
        default=Path("data/state.db"),
        description="Path to LangGraph state database",
    )
    
    # =========================================================================
    # Version Info
    # =========================================================================
    
    VERSION: str = Field(
        default="2.1.0",
        description="N.I.A. version string",
    )
    
    # =========================================================================
    # Validators
    # =========================================================================
    
    @field_validator("WAKE_WORDS", mode="before")
    @classmethod
    def parse_wake_words(cls, v):
        """Parse comma-separated wake words from env string."""
        if v is None:
            return ["nia", "jarvis", "hey nia"]
        if isinstance(v, str):
            # Try to parse as JSON first (for arrays), fallback to comma-separated
            v = v.strip()
            if v.startswith("["):
                try:
                    import json
                    parsed = json.loads(v)
                    return [w.strip().lower() for w in parsed if w.strip()]
                except (json.JSONDecodeError, TypeError):
                    pass
            # Fallback: comma-separated string
            return [w.strip().lower() for w in v.split(",") if w.strip()]
        if isinstance(v, list):
            return [str(w).lower() for w in v]
        return ["nia", "jarvis", "hey nia"]
    
    @field_validator("LOG_DIR", "MODEL_DIR", "DATA_DIR", "SOUNDS_DIR", mode="before")
    @classmethod
    def ensure_path(cls, v):
        """Convert string to Path."""
        return Path(v) if isinstance(v, str) else v
    
    # =========================================================================
    # Computed Properties
    # =========================================================================
    
    @property
    def log_file(self) -> Path:
        """Full path to the main log file."""
        return self.LOG_DIR / "nia.log"
    
    @property
    def has_nvidia_key(self) -> bool:
        """Check if NVIDIA API key is configured."""
        key = self.NVIDIA_API_KEY.get_secret_value()
        return bool(key and key.startswith("nvapi-"))
    
    @property
    def has_openai_key(self) -> bool:
        """Check if OpenAI API key is configured."""
        if self.OPENAI_API_KEY is None:
            return False
        key = self.OPENAI_API_KEY.get_secret_value()
        return bool(key and key.startswith("sk-"))
    
    def ensure_directories(self) -> None:
        """Create all required directories if they don't exist."""
        for dir_path in [self.LOG_DIR, self.MODEL_DIR, self.DATA_DIR, self.SOUNDS_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Global Settings Instance (with fallback)
# =============================================================================

def _create_settings() -> Settings:
    """Create settings with graceful fallback on .env parsing errors."""
    import warnings
    
    try:
        return Settings()
    except Exception as e:
        # .env file has parsing issues - try without it
        warnings.warn(f"Failed to load .env file: {e}. Using defaults.")
        
        # Create settings class that ignores .env
        class FallbackSettings(Settings):
            model_config = SettingsConfigDict(
                env_file=None,  # Skip problematic .env
                extra="ignore",
            )
        
        return FallbackSettings()

settings = _create_settings()

# Ensure directories exist on import
settings.ensure_directories()


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "Settings",
    "settings",
]
