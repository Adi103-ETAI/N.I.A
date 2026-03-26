"""Factory for creating chat models across providers."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.core.logger import setup_logger

from .config import ModelConfig
from .presets import MODEL_CATALOG

logger = setup_logger("Models")

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
                "Install with: uv add langchain-nvidia-ai-endpoints"
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
                "Install with: uv add langchain-openai"
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
                "Install with: uv add langchain-ollama"
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
                "Install with: uv add langchain-groq"
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

__all__ = ["ModelFactory"]
