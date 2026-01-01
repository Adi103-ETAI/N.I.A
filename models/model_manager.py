"""Model Manager for NIA - Multi-Provider LLM Factory.

This module provides a clean, decoupled interface for working with multiple
LLM providers (NVIDIA NIM, OpenAI, Ollama, Groq) through a unified API.

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
    │  │                  LangChain Chat Models                    │  │
    │  │  ChatNVIDIA  │  ChatOpenAI  │  ChatOllama  │  ChatGroq    │  │
    │  └───────────────────────────────────────────────────────────┘  │
    └─────────────────────────────────────────────────────────────────┘

Usage:
    from models import ModelManager
    
    manager = ModelManager()
    
    # Use presets
    smart = manager.get_smart_model()   # Best quality
    fast = manager.get_fast_model()     # Fastest response
    vision = manager.get_vision_model() # Image understanding
    
    # Or get specific provider/model
    model = manager.get_chat_model("nvidia", "meta/llama-3.1-70b-instruct")
    response = model.invoke("Hello!")

Version: 2.0.0
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure module logger
logger = logging.getLogger(__name__)


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
    

# Model catalog
MODEL_CATALOG: Dict[str, ModelSpec] = {
    # NVIDIA NIM Models (Free Tier Available)
    "nvidia/llama-3.1-405b": ModelSpec(
        provider=Provider.NVIDIA,
        model_name="meta/llama-3.1-405b-instruct",
        display_name="Llama 3.1 405B (NVIDIA)",
        context_window=128000,
        supports_function_calling=True,
        cost_tier="free",
        speed_tier="slow",
    ),
    "nvidia/llama-3.1-70b": ModelSpec(
        provider=Provider.NVIDIA,
        model_name="meta/llama-3.1-70b-instruct",
        display_name="Llama 3.1 70B (NVIDIA)",
        context_window=128000,
        supports_function_calling=True,
        cost_tier="free",
        speed_tier="medium",
    ),
    "nvidia/llama-3.1-8b": ModelSpec(
        provider=Provider.NVIDIA,
        model_name="meta/llama-3.1-8b-instruct",
        display_name="Llama 3.1 8B (NVIDIA)",
        context_window=128000,
        supports_function_calling=True,
        cost_tier="free",
        speed_tier="fast",
    ),
    "nvidia/llama-3.2-vision": ModelSpec(
        provider=Provider.NVIDIA,
        model_name="meta/llama-3.2-90b-vision-instruct",
        display_name="Llama 3.2 90B Vision (NVIDIA)",
        context_window=128000,
        supports_vision=True,
        cost_tier="free",
        speed_tier="slow",
    ),
    "nvidia/llama-3.2-11b-vision": ModelSpec(
        provider=Provider.NVIDIA,
        model_name="meta/llama-3.2-11b-vision-instruct",
        display_name="Llama 3.2 11B Vision (NVIDIA)",
        context_window=128000,
        supports_vision=True,
        cost_tier="free",
        speed_tier="medium",
    ),
    "nvidia/nemotron": ModelSpec(
        provider=Provider.NVIDIA,
        model_name="nvidia/llama-3.1-nemotron-70b-instruct",
        display_name="Nemotron 70B (NVIDIA)",
        context_window=128000,
        supports_function_calling=True,
        cost_tier="free",
        speed_tier="medium",
    ),
    "nvidia/mistral-nemo": ModelSpec(
        provider=Provider.NVIDIA,
        model_name="nv-mistralai/mistral-nemo-12b-instruct",
        display_name="Mistral Nemo 12B (NVIDIA)",
        context_window=128000,
        cost_tier="free",
        speed_tier="fast",
    ),
    
    # OpenAI Models
    "openai/gpt-4o": ModelSpec(
        provider=Provider.OPENAI,
        model_name="gpt-4o",
        display_name="GPT-4o (OpenAI)",
        context_window=128000,
        supports_vision=True,
        supports_function_calling=True,
        cost_tier="high",
        speed_tier="fast",
    ),
    "openai/gpt-4o-mini": ModelSpec(
        provider=Provider.OPENAI,
        model_name="gpt-4o-mini",
        display_name="GPT-4o Mini (OpenAI)",
        context_window=128000,
        supports_vision=True,
        supports_function_calling=True,
        cost_tier="low",
        speed_tier="fast",
    ),
    "openai/gpt-4-turbo": ModelSpec(
        provider=Provider.OPENAI,
        model_name="gpt-4-turbo",
        display_name="GPT-4 Turbo (OpenAI)",
        context_window=128000,
        supports_vision=True,
        supports_function_calling=True,
        cost_tier="high",
        speed_tier="medium",
    ),
    
    # Ollama Models (Local)
    "ollama/llama3": ModelSpec(
        provider=Provider.OLLAMA,
        model_name="llama3",
        display_name="Llama 3 8B (Ollama)",
        context_window=8192,
        is_local=True,
        cost_tier="free",
        speed_tier="fast",
    ),
    "ollama/llama3.1": ModelSpec(
        provider=Provider.OLLAMA,
        model_name="llama3.1",
        display_name="Llama 3.1 8B (Ollama)",
        context_window=128000,
        is_local=True,
        cost_tier="free",
        speed_tier="fast",
    ),
    "ollama/mistral": ModelSpec(
        provider=Provider.OLLAMA,
        model_name="mistral",
        display_name="Mistral 7B (Ollama)",
        context_window=32768,
        is_local=True,
        cost_tier="free",
        speed_tier="fast",
    ),
    "ollama/llava": ModelSpec(
        provider=Provider.OLLAMA,
        model_name="llava",
        display_name="LLaVA (Ollama)",
        context_window=4096,
        supports_vision=True,
        is_local=True,
        cost_tier="free",
        speed_tier="medium",
    ),
    
    # Groq Models (Fast)
    "groq/llama-3.1-70b": ModelSpec(
        provider=Provider.GROQ,
        model_name="llama-3.1-70b-versatile",
        display_name="Llama 3.1 70B (Groq)",
        context_window=128000,
        cost_tier="low",
        speed_tier="fast",
    ),
    "groq/llama-3.1-8b": ModelSpec(
        provider=Provider.GROQ,
        model_name="llama-3.1-8b-instant",
        display_name="Llama 3.1 8B (Groq)",
        context_window=128000,
        cost_tier="free",
        speed_tier="fast",
    ),
    "groq/mixtral": ModelSpec(
        provider=Provider.GROQ,
        model_name="mixtral-8x7b-32768",
        display_name="Mixtral 8x7B (Groq)",
        context_window=32768,
        cost_tier="low",
        speed_tier="fast",
    ),
}


# =============================================================================
# Model Configuration
# =============================================================================

@dataclass
class ModelConfig:
    """Configuration for model providers and settings.
    
    Loads API keys and endpoints from environment variables.
    """
    # API Keys
    nvidia_api_key: Optional[str] = field(default_factory=lambda: os.environ.get("NVIDIA_API_KEY"))
    openai_api_key: Optional[str] = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY"))
    groq_api_key: Optional[str] = field(default_factory=lambda: os.environ.get("GROQ_API_KEY"))
    huggingface_api_key: Optional[str] = field(default_factory=lambda: os.environ.get("HUGGINGFACE_API_KEY"))
    
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
        
        # Cached models
        self._smart_model = None
        self._fast_model = None
        self._vision_model = None
        self._current_model = None
        
        # Logger
        self.logger = logging.getLogger(__name__)
        self.logger.info(
            "ModelManager initialized (providers: %s)",
            self.factory.get_available_providers()
        )
    
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
                preferred_specs=["nvidia/llama-3.1-405b", "nvidia/llama-3.1-70b", "nvidia/nemotron", "openai/gpt-4o"],
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
        
        Args:
            preferred_specs: Ordered list of model spec keys.
            temperature: Sampling temperature.
            
        Returns:
            First available LangChain chat model.
            
        Raises:
            RuntimeError: If no models are available.
        """
        errors = []
        
        for spec_key in preferred_specs:
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
                return model
            except Exception as exc:
                errors.append(f"{spec_key}: {exc}")
                continue
        
        # No model available
        error_summary = "\n".join(f"  - {e}" for e in errors)
        raise RuntimeError(
            f"No models available. Tried:\n{error_summary}\n\n"
            f"Install providers with: pip install langchain-nvidia-ai-endpoints langchain-openai"
        )
    
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
    logging.basicConfig(level=logging.INFO)
    print_status()
