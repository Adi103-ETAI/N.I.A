"""Model manager compatibility facade and orchestration API."""
from __future__ import annotations

import base64
import mimetypes
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from src.core.config import settings
from src.core.logger import setup_logger
from .config import ModelConfig
from .embeddings import get_embedding_function
from .factory import ModelFactory
from .presets import (
    DEFAULT_PROVIDER,
    MODEL_CATALOG,
    VALID_PROVIDERS,
    ModelSpec,
    Provider,
)

load_dotenv()
logger = setup_logger("Models")

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
                f"Install with: uv add langchain-{provider}-ai-endpoints"
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
            f"Install providers with: uv add langchain-nvidia-ai-endpoints langchain-openai"
        )
    
    def _wrap_with_safety(self, model: Any) -> Any:
        """Wrap a model with SafeLLM circuit breaker.
        
        v4.0.0 Update: SafeLLM was deprecated. Models are now returned directly.
        This method is kept as a pass-through for API compatibility.
        
        Args:
            model: Raw LangChain chat model.
            
        Returns:
            The raw model.
        """
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

def switch_llm_provider(provider: str) -> str:
    """Switch the active AI model provider at runtime.

    Allows changing between LLM providers (nvidia, openai, groq, ollama)
    without restarting the application. All agents will use the new provider.

    Args:
        provider: The provider to switch to. Valid: 'nvidia', 'openai', 'groq', 'ollama'.

    Returns:
        Success message or error description.
    """
    import logging
    _log = logging.getLogger("Models.LLMOps")
    provider_clean = provider.lower().strip()
    _log.info(f"Attempting to switch LLM provider to: {provider_clean}")
    try:
        mm = get_model_manager()
        old = mm.get_active_provider()
        mm.set_active_provider(provider_clean)
        _log.info(f"Switched: {old} → {provider_clean}")
        return f"✅ Switched active LLM provider from '{old}' to '{provider_clean}'."
    except ValueError as e:
        return f"❌ Cannot switch to '{provider_clean}': {e}"
    except Exception as e:
        return f"❌ Unexpected error: {e}"

def get_current_provider() -> str:
    """Get the name of the currently active LLM provider."""
    try:
        return f"🧠 Current provider: '{get_model_manager().get_active_provider()}'"
    except Exception as e:
        return f"❌ Error: {e}"

def list_available_providers() -> str:
    """List all available LLM providers."""
    try:
        mm = get_model_manager()
        available = ", ".join(sorted(mm.factory.get_available_providers()))
        current = mm.get_active_provider()
        return f"📋 Available: {available}\n🧠 Using: '{current}'"
    except Exception as e:
        return f"❌ Error: {e}"

if __name__ == "__main__":
    print_status()

__all__ = [
    "ModelManager",
    "ModelFactory",
    "ModelConfig",
    "Provider",
    "ModelSpec",
    "MODEL_CATALOG",
    "DEFAULT_PROVIDER",
    "VALID_PROVIDERS",
    "get_model_manager",
    "get_smart_model",
    "get_fast_model",
    "get_vision_model",
    "get_embedding_function",
    "switch_llm_provider",
    "get_current_provider",
    "list_available_providers",
    "print_status",
]
