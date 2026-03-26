"""Model provider runtime configuration."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Union

from dotenv import load_dotenv

from src.core.config import settings
from .presets import Provider

load_dotenv()

@dataclass
class ModelConfig:
    """Configuration for model providers and settings.
    
    Loads API keys from centralized settings (secure SecretStr).
    """
    # API Keys (Securely fetched from settings)
    nvidia_api_key: Optional[str] = field(default_factory=lambda: settings.NVIDIA_API_KEY.get_secret_value() if settings.NVIDIA_API_KEY else None)
    openai_api_key: Optional[str] = field(default_factory=lambda: settings.OPENAI_API_KEY.get_secret_value() if settings.OPENAI_API_KEY else None)
    groq_api_key: Optional[str] = field(default_factory=lambda: settings.GROQ_API_KEY.get_secret_value() if settings.GROQ_API_KEY else None)
    huggingface_api_key: Optional[str] = field(default_factory=lambda: settings.HUGGINGFACE_API_KEY.get_secret_value() if settings.HUGGINGFACE_API_KEY else None)
    
    # Endpoints
    ollama_base_url: str = field(default_factory=lambda: os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    nvidia_base_url: str = field(default_factory=lambda: os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"))
    
    # Default model settings
    default_temperature: float = 0.7
    default_max_tokens: int = 2048
    default_timeout: int = 60
    
    # Provider preferences (ordered by priority)
    preferred_providers: List[str] = field(default_factory=lambda: ["nvidia", "openai", "ollama"])
    
    # Preset model selections
    smart_model: str = "nvidia/llama-3.1-70b"
    fast_model: str = "nvidia/llama-3.1-8b"
    vision_model: str = "nvidia/llama-3.2-11b-vision"
    embedding_model: str = "openai/text-embedding-3-small"
    
    def get_api_key(self, provider: Union[str, Provider]) -> Optional[str]:
        """Get API key for a provider."""
        provider_str = provider.value if isinstance(provider, Provider) else provider.lower()
        return {
            "nvidia": self.nvidia_api_key,
            "openai": self.openai_api_key,
            "groq": self.groq_api_key,
            "huggingface": self.huggingface_api_key,
        }.get(provider_str)
    
    def has_api_key(self, provider: Union[str, Provider]) -> bool:
        """Check if API key is available for provider."""
        return bool(self.get_api_key(provider))
    
    def get_available_providers(self) -> List[str]:
        """Get list of providers with valid API keys."""
        available = []
        for provider in ["nvidia", "openai", "groq"]:
            if self.has_api_key(provider):
                available.append(provider)
        # Ollama doesn't need API key
        available.append("ollama")
        return available

__all__ = ["ModelConfig"]
