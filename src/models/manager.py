"""Model Manager for NIA - Multi-Provider LLM Factory.

This module provides a clean, decoupled interface for working with multiple
LLM providers (NVIDIA NIM, OpenAI, Ollama, Groq) through a unified API.

v2.5.2 "Velocity" - Key Features:
    - Hot-Swap Provider Switching: Change active provider at runtime via
      `set_active_provider()`. All agents automatically use the new provider.
    - SafeLLM Circuit Breaker: All models are wrapped with automatic retry
      and fallback logic. If Provider A fails (429), switches to Provider B.
    - Dynamic Access Pattern: Agents use `@property` to fetch models on each
      access, enabling seamless hot-swap without restart.

Data Flow:
    User -> Supervisor -> SafeLLM -> ModelManager -> [NVIDIA|OpenAI|Groq|Ollama]
                            ^
                            |__ Circuit Breaker: Auto-fallback on 429/503

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                       ModelManager                              │
    │                                                                 │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
    │  │ ModelConfig │  │ ModelFactory│  │      Model Presets      │  │
    │  │ (API Keys)  │  │ (Providers) │  │ smart/fast/vision/embed │  │
    │  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
    │         │                │                     │                │
    │         ▼                ▼                     ▼                │
    │  ┌───────────────────────────────────────────────────────────┐  │
    │  │               SafeLLM Wrapped Chat Models                 │  │
    │  │  ChatNVIDIA  │  ChatOpenAI  │  ChatOllama  │  ChatGroq    │  │
    │  └───────────────────────────────────────────────────────────┘  │
    └─────────────────────────────────────────────────────────────────┘

Usage:
    from src.models import ModelManager
    
    manager = ModelManager()
    
    # Use presets (all wrapped with SafeLLM)
    smart = manager.get_smart_model()   # Best quality
    fast = manager.get_fast_model()     # Fastest response
    vision = manager.get_vision_model() # Image understanding
    
    # Hot-swap provider at runtime
    manager.set_active_provider("openai")  # All agents now use OpenAI
    
    # Or get specific provider/model
    model = manager.get_chat_model("nvidia", "meta/llama-3.1-70b-instruct")
    response = model.invoke("Hello!")

Version: 2.5.2
"""
from __future__ import annotations

import base64
import json
import mimetypes
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
from src.core.logger import setup_logger
from src.core.config import settings

# v2.5.2: SafeLLM import (deferred to avoid circular import)
# SafeLLM is imported lazily in _wrap_with_safety()

# Load environment variables
load_dotenv()

# Configure module logger
logger = setup_logger("Models")

def _load_general_config() -> dict:
    config_path = Path(__file__).resolve().parents[2] / "config" / "nia" / "general.json"
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load general.json: {e}")
        return {}

_GENERAL_CONFIG = _load_general_config()

# Default fallback provider (NVIDIA supremacy)
DEFAULT_PROVIDER = _GENERAL_CONFIG.get("DEFAULT_PROVIDER", "nvidia")

# Valid providers for runtime switching
VALID_PROVIDERS = frozenset(_GENERAL_CONFIG.get("VALID_PROVIDERS", ["nvidia", "openai", "groq", "ollama"]))

# v2.5.2: Enable/disable SafeLLM wrapping (for testing/debugging)
ENABLE_SAFE_LLM = True


# =============================================================================
# Provider Enum
# =============================================================================

class Provider(str, Enum):
    """Supported LLM providers."""
    NVIDIA = "nvidia"
    OPENAI = "openai"
    OLLAMA = "ollama"
    GROQ = "groq"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"


# =============================================================================
# Model Definitions
# =============================================================================

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
    

# =============================================================================
# Catalog Loader (Dynamic from JSON)
# =============================================================================

def _load_catalog() -> Dict[str, ModelSpec]:
    """Load model catalog from external JSON file.
    
    Returns:
        Dictionary mapping model keys to ModelSpec objects.
        
    Raises:
        FileNotFoundError: If catalog.json is missing.
        json.JSONDecodeError: If catalog.json has invalid JSON.
    """
    catalog_path = Path(__file__).resolve().parents[2] / "config" / "nia" / "models.json"
    
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


# =============================================================================
# Model Configuration
# =============================================================================

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


# =============================================================================
# Model Factory
# =============================================================================

class ModelFactory:
    """Factory for creating LangChain chat models.
    
    Supports multiple providers with automatic fallback.
    
    Example:
        factory = ModelFactory()
        model = factory.get_chat_model("nvidia", "meta/llama-3.1-70b-instruct")
        response = model.invoke("Hello!")
    """
    
    # Track available providers
    _available_providers: Dict[str, bool] = {}
    
    def __init__(self, config: Optional[ModelConfig] = None) -> None:
        """Initialize the factory.
        
        Args:
            config: Model configuration. Uses defaults if not provided.
        """
        self.config = config or ModelConfig()
        self._check_providers()
    
    def _check_providers(self) -> None:
        """Check which providers are available."""
        # Check NVIDIA
        try:
            self._available_providers["nvidia"] = True
        except ImportError:
            self._available_providers["nvidia"] = False
            logger.debug("langchain-nvidia-ai-endpoints not installed")
        
        # Check OpenAI
        try:
            self._available_providers["openai"] = True
        except ImportError:
            self._available_providers["openai"] = False
            logger.debug("langchain-openai not installed")
        
        # Check Ollama
        try:
            self._available_providers["ollama"] = True
        except ImportError:
            # Try alternative import
            try:
                self._available_providers["ollama"] = True
            except ImportError:
                self._available_providers["ollama"] = False
                logger.debug("langchain-ollama not installed")
        
        # Check Groq
        try:
            self._available_providers["groq"] = True
        except ImportError:
            self._available_providers["groq"] = False
            logger.debug("langchain-groq not installed")
    
    def is_provider_available(self, provider: str) -> bool:
        """Check if a provider is available."""
        return self._available_providers.get(provider.lower(), False)
    
    def get_available_providers(self) -> List[str]:
        """Get list of installed providers."""
        return [p for p, available in self._available_providers.items() if available]
    
    def get_chat_model(
        self,
        provider: str,
        model_name: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> Any:
        """Create a LangChain chat model for the specified provider.
        
        Args:
            provider: Provider name ('nvidia', 'openai', 'ollama', 'groq').
            model_name: Model identifier for the provider.
            temperature: Sampling temperature (0.0-2.0).
            max_tokens: Maximum tokens in response.
            **kwargs: Additional provider-specific arguments.
            
        Returns:
            LangChain chat model instance.
            
        Raises:
            ImportError: If provider's package is not installed.
            ValueError: If API key is missing for cloud provider.
        """
        provider = provider.lower()
        
        if provider == "nvidia":
            return self._create_nvidia_model(model_name, temperature, max_tokens, **kwargs)
        elif provider == "openai":
            return self._create_openai_model(model_name, temperature, max_tokens, **kwargs)
        elif provider == "ollama":
            return self._create_ollama_model(model_name, temperature, max_tokens, **kwargs)
        elif provider == "groq":
            return self._create_groq_model(model_name, temperature, max_tokens, **kwargs)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def _create_nvidia_model(
        self,
        model_name: str,
        temperature: float,
        max_tokens: Optional[int],
        **kwargs,
    ) -> Any:
        """Create NVIDIA NIM chat model."""
        if not self._available_providers.get("nvidia"):
            raise ImportError(
                "langchain-nvidia-ai-endpoints not installed. "
                "Install with: pip install langchain-nvidia-ai-endpoints"
            )
        
        api_key = self.config.nvidia_api_key
        if not api_key:
            raise ValueError("NVIDIA_API_KEY not set in environment")
        
        from langchain_nvidia_ai_endpoints import ChatNVIDIA
        
        return ChatNVIDIA(
            model=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens or self.config.default_max_tokens,
            base_url=self.config.nvidia_base_url,
            timeout=5,  # Fast fail on startup - prevents 60s hang
            **kwargs,
        )
    
    def _create_openai_model(
        self,
        model_name: str,
        temperature: float,
        max_tokens: Optional[int],
        **kwargs,
    ) -> Any:
        """Create OpenAI chat model."""
        if not self._available_providers.get("openai"):
            raise ImportError(
                "langchain-openai not installed. "
                "Install with: pip install langchain-openai"
            )
        
        api_key = self.config.openai_api_key
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set in environment")
        
        from langchain_openai import ChatOpenAI
        
        return ChatOpenAI(
            model=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens or self.config.default_max_tokens,
            request_timeout=5,  # Fast fail on startup
            **kwargs,
        )
    
    def _create_ollama_model(
        self,
        model_name: str,
        temperature: float,
        max_tokens: Optional[int],
        **kwargs,
    ) -> Any:
        """Create Ollama chat model (local)."""
        if not self._available_providers.get("ollama"):
            raise ImportError(
                "langchain-ollama not installed. "
                "Install with: pip install langchain-ollama"
            )
        
        # Try modern import first
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            from langchain_community.chat_models import ChatOllama
        
        return ChatOllama(
            model=model_name,
            base_url=self.config.ollama_base_url,
            temperature=temperature,
            num_predict=max_tokens or self.config.default_max_tokens,
            timeout=5,  # Fast fail if Ollama not running
            **kwargs,
        )
    
    def _create_groq_model(
        self,
        model_name: str,
        temperature: float,
        max_tokens: Optional[int],
        **kwargs,
    ) -> Any:
        """Create Groq chat model."""
        if not self._available_providers.get("groq"):
            raise ImportError(
                "langchain-groq not installed. "
                "Install with: pip install langchain-groq"
            )
        
        api_key = self.config.groq_api_key
        if not api_key:
            raise ValueError("GROQ_API_KEY not set in environment")
        
        from langchain_groq import ChatGroq
        
        return ChatGroq(
            model=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens or self.config.default_max_tokens,
            request_timeout=5,  # Fast fail on startup
            **kwargs,
        )
    
    def get_model_from_spec(
        self,
        spec_key: str,
        temperature: float = 0.7,
        **kwargs,
    ) -> Any:
        """Create a model from a catalog specification.
        
        Args:
            spec_key: Key in MODEL_CATALOG (e.g., 'nvidia/llama-3.1-70b').
            temperature: Sampling temperature.
            **kwargs: Additional arguments.
            
        Returns:
            LangChain chat model.
        """
        if spec_key not in MODEL_CATALOG:
            raise ValueError(f"Unknown model spec: {spec_key}")
        
        spec = MODEL_CATALOG[spec_key]
        return self.get_chat_model(
            provider=spec.provider.value,
            model_name=spec.model_name,
            temperature=temperature,
            **kwargs,
        )


# =============================================================================
# Model Manager (Main Interface)
# =============================================================================

class ModelManager:
    """Unified model manager with preset models and provider management.
    
    This is the main interface for NIA to interact with LLMs. It provides:
    - Preset models (smart, fast, vision, embedding)
    - Automatic fallback across providers
    - Simple invoke() method for text generation
    
    Example:
        manager = ModelManager()
        
        # Use the smart model
        response = manager.invoke("What is the capital of France?")
        
        # Or get specific model types
        smart = manager.get_smart_model()
        fast = manager.get_fast_model()
        vision = manager.get_vision_model()
    """
    
    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Initialize the model manager.
        
        Args:
            config: Model configuration.
            provider: Default provider (overrides config).
            model_name: Default model name (overrides config).
            **kwargs: Passed to ModelConfig for backwards compatibility.
        """
        # Handle legacy config dict
        legacy_config = kwargs.pop("config", None)
        if isinstance(legacy_config, dict):
            # Extract relevant fields for backwards compatibility
            kwargs.update({
                k: v for k, v in legacy_config.items()
                if k in ["temperature", "max_tokens", "persona_prompt"]
            })
        
        self.config = config or ModelConfig()
        self.factory = ModelFactory(self.config)
        
        # Override defaults if provided
        if provider:
            self.config.preferred_providers = [provider] + [
                p for p in self.config.preferred_providers if p != provider
            ]
        
        # Store for legacy compatibility
        self.provider = provider or self.config.preferred_providers[0]
        self.model_name = model_name
        self._temperature = kwargs.get("temperature", self.config.default_temperature)
        self._persona_prompt = kwargs.get("persona_prompt", "")
        
        # v3.0: Active provider for runtime switching (defaults to settings or "nvidia")
        self.active_provider: str = getattr(settings, "ACTIVE_LLM_PROVIDER", DEFAULT_PROVIDER).lower()
        self._default_provider: str = DEFAULT_PROVIDER
        
        # Cached models
        self._smart_model = None
        self._fast_model = None
        self._vision_model = None
        self._current_model = None
        
        # Logger - use module level logger
        self.logger = logger  # Reference module-level setup_logger("Models")
        self.logger.info(
            "ModelManager initialized (active_provider: %s, available: %s)",
            self.active_provider,
            self.factory.get_available_providers()
        )
    
    # =========================================================================
    # v3.0: Dynamic Provider Switching
    # =========================================================================
    
    def set_active_provider(self, provider: str) -> None:
        """Hot-swap the active LLM provider at runtime.
        
        This allows switching between providers (nvidia, openai, groq, ollama)
        without restarting the application. Cached models are cleared to force
        rebuild on next access.
        
        Args:
            provider: Provider name ('nvidia', 'openai', 'groq', 'ollama').
            
        Raises:
            ValueError: If provider is invalid or missing API key.
            
        Example:
            manager.set_active_provider("openai")  # Switch to OpenAI
            manager.set_active_provider("nvidia")  # Switch back to NVIDIA
        """
        provider = provider.lower().strip()
        
        # Validate provider name
        if provider not in VALID_PROVIDERS:
            raise ValueError(
                f"Unsupported provider: '{provider}'. "
                f"Valid options: {', '.join(sorted(VALID_PROVIDERS))}"
            )
        
        # Check API key exists (for cloud providers, not ollama)
        if provider != "ollama" and not self.config.has_api_key(provider):
            raise ValueError(
                f"Missing API key for provider '{provider}'. "
                f"Set the corresponding environment variable (e.g., OPENAI_API_KEY)."
            )
        
        # Check if provider package is installed
        if not self.factory.is_provider_available(provider):
            raise ValueError(
                f"Provider '{provider}' is not installed. "
                f"Install with: pip install langchain-{provider}-ai-endpoints"
            )
        
        # Clear cached models (force rebuild on next access)
        self._clear_model_cache()
        
        # Switch provider
        old_provider = self.active_provider
        self.active_provider = provider
        self.logger.info(
            "Switched active provider: %s -> %s",
            old_provider,
            provider
        )
    
    def get_active_provider(self) -> str:
        """Get the current active provider name.
        
        Returns:
            Active provider string (e.g., 'nvidia', 'openai').
        """
        return self.active_provider
    
    def _clear_model_cache(self) -> None:
        """Clear all cached model instances.
        
        Called when switching providers to ensure fresh model creation.
        """
        self._smart_model = None
        self._fast_model = None
        self._vision_model = None
        self._current_model = None
        self.logger.debug("Model cache cleared")
    
    # =========================================================================
    # Preset Models
    # =========================================================================
    
    def get_smart_model(self, temperature: float = 0.7) -> Any:
        """Get the highest quality model available.
        
        This returns the best model for complex reasoning, coding,
        and nuanced conversation. May be slower than fast model.
        
        Default: NVIDIA Llama 3.1 405B (most powerful) or fallbacks
        
        Returns:
            LangChain chat model.
        """
        if self._smart_model is None:
            self._smart_model = self._get_best_available_model(
                preferred_specs=["nvidia/llama-3.1-70b", "nvidia/nemotron", "openai/gpt-4o"],
                temperature=temperature,
            )
        return self._smart_model
    
    def get_fast_model(self, temperature: float = 0.7) -> Any:
        """Get the fastest model available.
        
        This returns a smaller, faster model for quick responses.
        Best for simple queries and low-latency requirements.
        
        Default: NVIDIA Llama 3.1 8B or Groq
        
        Returns:
            LangChain chat model.
        """
        if self._fast_model is None:
            self._fast_model = self._get_best_available_model(
                preferred_specs=[
                    "nvidia/llama-3.1-8b",
                    "nvidia/mistral-nemo",
                    "groq/llama-3.1-8b",
                    "ollama/llama3.1",
                ],
                temperature=temperature,
            )
        return self._fast_model
    
    def get_vision_model(self, temperature: float = 0.7) -> Any:
        """Get a vision-capable model.
        
        This returns a model that can understand images.
        
        Default: NVIDIA Llama 3.2 Vision or OpenAI GPT-4o
        
        Returns:
            LangChain chat model with vision support.
        """
        if self._vision_model is None:
            self._vision_model = self._get_best_available_model(
                preferred_specs=[
                    "nvidia/llama-3.2-11b-vision",
                    "nvidia/llama-3.2-vision",
                    "openai/gpt-4o",
                    "ollama/llava",
                ],
                temperature=temperature,
            )
        return self._vision_model
    
    def get_default_model(self, temperature: float = 0.7) -> Any:
        """Get the default model based on configuration.
        
        Returns:
            LangChain chat model.
        """
        if self._current_model is None:
            self._current_model = self.get_smart_model(temperature)
        return self._current_model
    
    def _get_best_available_model(
        self,
        preferred_specs: List[str],
        temperature: float = 0.7,
    ) -> Any:
        """Get the first available model from preferred list.
        
        v3.0: Prioritizes models from self.active_provider, then falls back
        to other providers in the preferred order.
        
        Args:
            preferred_specs: Ordered list of model spec keys.
            temperature: Sampling temperature.
            
        Returns:
            First available LangChain chat model.
            
        Raises:
            RuntimeError: If no models are available.
        """
        # v3.0: Reorder specs to prioritize active provider
        active = self.active_provider
        prioritized = [s for s in preferred_specs if s.startswith(f"{active}/")]
        others = [s for s in preferred_specs if not s.startswith(f"{active}/")]
        reordered_specs = prioritized + others
        
        self.logger.debug(
            "Model selection order (active=%s): %s",
            active,
            reordered_specs
        )
        
        errors = []
        
        for spec_key in reordered_specs:
            if spec_key not in MODEL_CATALOG:
                continue
            
            spec = MODEL_CATALOG[spec_key]
            provider = spec.provider.value
            
            # Check if provider is available
            if not self.factory.is_provider_available(provider):
                errors.append(f"{spec_key}: provider {provider} not installed")
                continue
            
            # Check if API key is available (for cloud providers)
            if not spec.is_local and not self.config.has_api_key(provider):
                errors.append(f"{spec_key}: missing API key for {provider}")
                continue
            
            try:
                model = self.factory.get_model_from_spec(spec_key, temperature)
                self.logger.info("Using model: %s", spec.display_name)
                
                # v2.5.2: Wrap with SafeLLM circuit breaker
                return self._wrap_with_safety(model)
            except Exception as exc:
                errors.append(f"{spec_key}: {exc}")
                continue
        
        # No model available
        error_summary = "\n".join(f"  - {e}" for e in errors)
        raise RuntimeError(
            f"No models available. Tried:\n{error_summary}\n\n"
            f"Install providers with: pip install langchain-nvidia-ai-endpoints langchain-openai"
        )
    
    def _wrap_with_safety(self, model: Any) -> Any:
        """Wrap a model with SafeLLM circuit breaker.
        
        v2.5.2: All models are wrapped for automatic 429 handling
        and provider fallback.
        
        Args:
            model: Raw LangChain chat model.
            
        Returns:
            SafeLLM wrapper (or raw model if disabled).
        """
        if not ENABLE_SAFE_LLM:
            return model
        
        try:
            from src.models.safe_llm import SafeLLM
            
            # Determine fallback based on current provider
            # If we're on NVIDIA, fallback to OpenAI; otherwise fallback to NVIDIA
            if self.active_provider == "nvidia":
                fallback = "openai" if self.config.has_api_key("openai") else "nvidia"
            else:
                fallback = "nvidia"
            
            wrapped = SafeLLM(
                primary_model=model,
                manager=self,
                fallback_provider=fallback,
                max_retries=2,
            )
            self.logger.debug(f"Model wrapped with SafeLLM (fallback={fallback})")
            return wrapped
            
        except ImportError as e:
            self.logger.warning(f"SafeLLM not available, using raw model: {e}")
            return model
    
    # =========================================================================
    # High-Level API
    # =========================================================================
    
    def invoke(
        self,
        prompt: str,
        model_type: str = "smart",
        temperature: Optional[float] = None,
    ) -> str:
        """Invoke the model with a prompt.
        
        Args:
            prompt: The prompt to send.
            model_type: 'smart', 'fast', or 'vision'.
            temperature: Override temperature.
            
        Returns:
            Model response as string.
        """
        temp = temperature or self._temperature
        
        if model_type == "fast":
            model = self.get_fast_model(temp)
        elif model_type == "vision":
            model = self.get_vision_model(temp)
        else:
            model = self.get_smart_model(temp)
        
        try:
            response = model.invoke(prompt)
            return self._extract_content(response)
        except Exception as exc:
            self.logger.exception("Model invocation failed: %s", exc)
            raise
    
    def _extract_content(self, response: Any) -> str:
        """Extract text content from model response."""
        if isinstance(response, str):
            return response
        if hasattr(response, "content"):
            return response.content
        if isinstance(response, dict):
            return response.get("content", str(response))
        return str(response)
    
    # =========================================================================
    # Image Encoding Helper
    # =========================================================================
    
    def _encode_image(self, image_path: str) -> Optional[str]:
        """Encode an image file to base64.
        
        Args:
            image_path: Path to the image file.
            
        Returns:
            Base64 encoded string, or None on error.
        """
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except FileNotFoundError:
            self.logger.error("Image file not found: %s", image_path)
            return None
        except Exception as exc:
            self.logger.error("Failed to encode image: %s", exc)
            return None
    
    def _get_mime_type(self, image_path: str) -> str:
        """Get MIME type from file path."""
        mime_type, _ = mimetypes.guess_type(image_path)
        return mime_type or "image/jpeg"
    
    # =========================================================================
    # Vision-Enabled Response Generation
    # =========================================================================
    
    def generate_response(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        model_type: str = "smart",
        temperature: Optional[float] = None,
    ) -> str:
        """Generate a response with optional image input.
        
        This method supports multimodal (text + image) inputs for vision models.
        
        Args:
            prompt: The text prompt to send.
            image_path: Optional path to an image file for vision queries.
            model_type: 'smart', 'fast', or 'vision'. Auto-selects 'vision' if image provided.
            temperature: Override temperature.
            
        Returns:
            Model response as string.
            
        Example:
            # Text only
            response = manager.generate_response("What is AI?")
            
            # With image
            response = manager.generate_response(
                "What's in this image?",
                image_path="screenshot.png"
            )
        """
        temp = temperature or self._temperature
        
        # Auto-select vision model if image is provided
        if image_path:
            model_type = "vision"
        
        # Get appropriate model
        if model_type == "fast":
            model = self.get_fast_model(temp)
        elif model_type == "vision":
            model = self.get_vision_model(temp)
        else:
            model = self.get_smart_model(temp)
        
        try:
            # Build message content
            if image_path:
                # Multimodal message (text + image)
                b64_image = self._encode_image(image_path)
                if not b64_image:
                    return f"Error: Failed to read image at {image_path}"
                
                mime_type = self._get_mime_type(image_path)
                
                # Import HumanMessage for multimodal
                try:
                    from langchain_core.messages import HumanMessage
                    
                    message = HumanMessage(content=[
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{b64_image}"}
                        }
                    ])
                    response = model.invoke([message])
                    
                except ImportError:
                    return "Error: langchain_core not available for vision queries"
            else:
                # Text-only message
                response = model.invoke(prompt)
            
            return self._extract_content(response)
            
        except Exception as exc:
            self.logger.exception("generate_response failed: %s", exc)
            raise
    
    # =========================================================================
    # Legacy Compatibility (for existing code)
    # =========================================================================
    
    def reason(self, prompt: str, mode: str = "default") -> str:
        """Legacy reasoning method for backwards compatibility.
        
        Args:
            prompt: The prompt to process.
            mode: Processing mode (ignored, for compatibility).
            
        Returns:
            Model response.
        """
        # Add persona if configured
        if self._persona_prompt:
            full_prompt = f"{self._persona_prompt}\n\n{prompt}"
        else:
            full_prompt = prompt
        
        return self.invoke(full_prompt, model_type="smart")
    
    def render_response(self, action_result: Dict[str, Any]) -> Optional[str]:
        """Legacy method to summarize action results.
        
        Args:
            action_result: Action result dictionary.
            
        Returns:
            Human-friendly summary or None.
        """
        summary_prompt = (
            "Summarize this action result for the user in a natural, "
            "conversational way. Do not mention technical details.\n\n"
            f"Result: {action_result}"
        )
        try:
            return self.invoke(summary_prompt, model_type="fast")
        except Exception as exc:
            self.logger.debug("render_response failed: %s", exc)
            return None


# =============================================================================
# Module-level Functions
# =============================================================================

_default_manager: Optional[ModelManager] = None


def get_model_manager(**kwargs) -> ModelManager:
    """Get or create the default ModelManager singleton."""
    global _default_manager
    if _default_manager is None:
        _default_manager = ModelManager(**kwargs)
    return _default_manager


def get_smart_model(**kwargs) -> Any:
    """Convenience function to get the smart model."""
    return get_model_manager().get_smart_model(**kwargs)


def get_fast_model(**kwargs) -> Any:
    """Convenience function to get the fast model."""
    return get_model_manager().get_fast_model(**kwargs)


def get_vision_model(**kwargs) -> Any:
    """Convenience function to get the vision model."""
    return get_model_manager().get_vision_model(**kwargs)


# =============================================================================
# Status Check
# =============================================================================

def print_status() -> None:
    """Print model system status."""
    config = ModelConfig()
    factory = ModelFactory(config)
    
    print("\n" + "=" * 50)
    print("  Model Manager Status")
    print("=" * 50)
    
    print("\n📦 Installed Providers:")
    for provider in ["nvidia", "openai", "ollama", "groq"]:
        installed = factory.is_provider_available(provider)
        has_key = config.has_api_key(provider) if provider != "ollama" else True
        status = "✓" if installed else "✗"
        key_status = "(key set)" if has_key else "(no key)"
        print(f"   {status} {provider:<10} {key_status if installed else ''}")
    
    print("\n📋 Available Model Presets:")
    
    try:
        manager = ModelManager(config)
        
        for preset, getter in [
            ("Smart", manager.get_smart_model),
            ("Fast", manager.get_fast_model),
            ("Vision", manager.get_vision_model),
        ]:
            try:
                model = getter()
                model_name = getattr(model, "model", "unknown")
                print(f"   ✓ {preset:<8}: {model_name}")
            except Exception as exc:
                print(f"   ✗ {preset:<8}: {exc}")
    except Exception as exc:
        print(f"   Error initializing manager: {exc}")
    
    print()


# Demo
if __name__ == "__main__":
    print_status()

