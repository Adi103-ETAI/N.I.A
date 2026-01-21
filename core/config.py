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
from functools import lru_cache
from pathlib import Path
from typing import Any, List, Optional, Union

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
    
    ACTIVE_LLM_PROVIDER: str = Field(
        default="nvidia",
        description="Active LLM provider for runtime switching (nvidia, openai, groq, ollama)",
    )
    
    NVIDIA_API_KEY: SecretStr = Field(
        default=SecretStr(""),
        description="NVIDIA NIM API key (required)",
    )
    
    OPENAI_API_KEY: Optional[SecretStr] = Field(
        default=None,
        description="OpenAI API key (optional fallback)",
    )
    
    GROQ_API_KEY: Optional[SecretStr] = Field(
        default=None,
        description="Groq API key (optional, for fast inference)",
    )
    
    HUGGINGFACE_API_KEY: Optional[SecretStr] = Field(
        default=None,
        description="HuggingFace API key (optional)",
    )
    
    OLLAMA_HOST: str = Field(
        default="http://localhost:11434",
        description="Ollama server URL for local models",
    )
    
    NVIDIA_BASE_URL: str = Field(
        default="https://integrate.api.nvidia.com/v1",
        description="NVIDIA NIM API endpoint URL",
    )
    
    # =========================================================================
    # LLM Model Selection
    # =========================================================================
    
    LLM_MODEL: str = Field(
        default="meta/llama-3.1-70b-instruct",
        description="Primary LLM model for NIA brain",
    )
    
    LLM_MODEL_SMART: str = Field(
        default="meta/llama-3.1-70b-instruct",  # Changed from 405b for speed
        description="High-quality LLM for complex reasoning",
    )
    
    LLM_MODEL_FAST: str = Field(
        default="meta/llama-3.1-70b-instruct",
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
    # Conversation Management
    # =========================================================================
    
    MAX_HISTORY: int = Field(
        default=50,
        ge=10,
        le=200,
        description="Maximum conversation messages before pruning",
    )
    
    PRUNE_COUNT: int = Field(
        default=15,
        ge=5,
        le=50,
        description="Number of messages to summarize when pruning",
    )
    
    MAX_RETRIES: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum retries for LLM routing validation",
    )
    
    MAX_ITERATIONS: int = Field(
        default=10,
        ge=3,
        le=30,
        description="Maximum reasoning iterations for TARA tool loop",
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
        default="2.5.2",
        description="N.I.A. version string",
    )
    
    FORCE_LOCAL_EMBEDDINGS: bool = Field(
        default=True,
        description="Force local embeddings (ignore OpenAI key). Set to False to use OpenAI.",
    )
    
    # =========================================================================
    # Desktop Automation Settings (Phase 2)
    # =========================================================================
    
    UIA_DEFAULT_TIMEOUT: float = Field(
        default=5.0,
        ge=1.0,
        le=30.0,
        description="Default timeout (seconds) for UI element waits",
    )
    
    UIA_POLL_INTERVAL: float = Field(
        default=0.5,
        ge=0.1,
        le=2.0,
        description="Polling interval (seconds) for element wait retries",
    )
    
    PYAUTOGUI_PAUSE: float = Field(
        default=0.5,
        ge=0.1,
        le=2.0,
        description="Pause between pyautogui actions (stability delay)",
    )
    
    PYAUTOGUI_FAILSAFE: bool = Field(
        default=True,
        description="Enable pyautogui failsafe (move mouse to corner to abort)",
    )
    
    SCROLL_STEPS: int = Field(
        default=200,
        ge=50,
        le=1000,
        description="Default scroll distance in pixels",
    )
    
    LAUNCH_MAX_RETRIES: int = Field(
        default=20,
        ge=5,
        le=60,
        description="Max retries when waiting for app window to appear",
    )
    
    LAUNCH_POLL_INTERVAL: float = Field(
        default=0.5,
        ge=0.1,
        le=2.0,
        description="Polling interval (seconds) for launch window verification",
    )
    
    # =========================================================================
    # TARA 2.0 Configuration
    # =========================================================================
    
    # File System Paths
    SCREENSHOT_DIR: Path = Field(
        default=Path("data/screenshots"),
        description="Directory for screenshot captures",
    )
    
    BROWSER_DOWNLOAD_DIR: Path = Field(
        default=Path("data/downloads"),
        description="Directory for browser downloads",
    )
    
    # Browser (Playwright)
    BROWSER_HEADLESS: bool = Field(
        default=False,
        description="Run Playwright browser in headless mode",
    )
    
    BROWSER_DEFAULT_TIMEOUT: int = Field(
        default=30000,
        ge=5000,
        le=120000,
        description="Default browser operation timeout (milliseconds)",
    )
    
    BROWSER_VIEWPORT_WIDTH: int = Field(
        default=1280,
        description="Browser viewport width in pixels",
    )
    
    BROWSER_VIEWPORT_HEIGHT: int = Field(
        default=800,
        description="Browser viewport height in pixels",
    )
    
    BROWSER_EXECUTABLE_PATH: Optional[str] = Field(
        default=r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe",
        description="Custom browser executable path (e.g., Brave, Edge). Set to None for default Chromium.",
    )
    
    # Safety Limits
    MAX_FILE_READ_CHARS: int = Field(
        default=5000,
        ge=1000,
        le=50000,
        description="Maximum characters to read from files (token budget)",
    )
    
    SAFE_DELETE_CONFIRM: bool = Field(
        default=True,
        description="Require confirm=True for delete operations",
    )
    
    MAX_SEARCH_RESULTS: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum file search results to return",
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
    
    @field_validator(
        "LOG_DIR", "MODEL_DIR", "DATA_DIR", "SOUNDS_DIR",
        "SCREENSHOT_DIR", "BROWSER_DOWNLOAD_DIR",
        mode="before"
    )
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
        for dir_path in [
            self.LOG_DIR,
            self.MODEL_DIR,
            self.DATA_DIR,
            self.SOUNDS_DIR,
            self.SCREENSHOT_DIR,
            self.BROWSER_DOWNLOAD_DIR,
        ]:
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


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings singleton.
    
    Uses lru_cache for efficient caching across the application.
    """
    return _create_settings()


# Create module-level settings for backward compatibility
settings = get_settings()

# Ensure directories exist on import
settings.ensure_directories()


# =============================================================================
# Embedding Function Factory
# =============================================================================

def get_embedding_function() -> Any:
    """Get the embedding function for Vector DB.
    
    FORCE LOCAL to match the existing database and save costs.
    Returns None which triggers ChromaDB's default SentenceTransformer (Local/Free).
    """
    # Return None = ChromaDB uses default local embeddings (all-MiniLM-L6-v2)
    return None


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "Settings",
    "settings",
    "get_settings",
    "get_embedding_function",
]
