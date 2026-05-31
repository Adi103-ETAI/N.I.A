"""Provider Registry - Discovers and manages LLM providers.

Central registry that:
- Registers available providers
- Discovers providers from config
- Routes requests to the active provider
"""

from __future__ import annotations

import logging
from typing import Any

from agents.nia.providers.base import LLMProvider
from agents.nia.providers.types import ModelInfo, ProviderInfo
from agents.nia.config import ConfigManager, NIAConfig

logger = logging.getLogger(__name__)


class ProviderRegistry:
    """Registry of all available LLM providers.

    Usage:
        registry = ProviderRegistry(config_manager)
        await registry.initialize()

        # List providers
        providers = registry.list_providers()

        # Switch provider/model
        registry.set_active("openai", "gpt-4o")

        # Get the active LLM
        llm = registry.get_active_provider()
        response = await llm.complete(request)
    """

    def __init__(self, config_manager: ConfigManager) -> None:
        self._config_manager = config_manager
        self._providers: dict[str, LLMProvider] = {}
        self._active_provider_id: str = ""
        self._active_model: str = ""

    async def initialize(self) -> None:
        """Initialize all configured providers."""
        config = self._config_manager.config

        # Import and register all built-in providers
        await self._register_builtin_providers()

        # Configure providers from config
        for provider_id, pc in config.providers.items():
            if provider_id in self._providers:
                provider = self._providers[provider_id]
                provider.configure(
                    api_key=pc.api_key,
                    base_url=pc.base_url,
                )
                logger.info(f"Configured provider: {provider_id}")

        # Fetch models from configured providers
        for provider_id, provider in self._providers.items():
            if provider.is_configured():
                try:
                    models = await provider.fetch_models()
                    logger.info(f"Fetched {len(models)} models from {provider_id}")
                except Exception as e:
                    logger.debug(f"Failed to fetch models from {provider_id}: {e}")

        # Set active provider
        self._active_provider_id = config.active_provider
        self._active_model = config.active_model

        if self._active_provider_id and self._active_provider_id not in self._providers:
            logger.warning(f"Active provider '{self._active_provider_id}' not found, falling back")
            self._active_provider_id = self._find_first_configured()

        logger.info(f"Registry initialized. Active: {self._active_provider_id}/{self._active_model}")

    def _find_first_configured(self) -> str:
        """Find the first configured provider."""
        for pid, provider in self._providers.items():
            if provider.is_configured():
                return pid
        return ""

    async def _register_builtin_providers(self) -> None:
        """Register all built-in providers."""
        # Anthropic
        from agents.nia.providers.anthropic import AnthropicProvider
        self._providers["anthropic"] = AnthropicProvider()

        # OpenAI
        from agents.nia.providers.openai import OpenAIProvider
        self._providers["openai"] = OpenAIProvider()

        # Ollama
        from agents.nia.providers.ollama import OllamaProvider
        self._providers["ollama"] = OllamaProvider()

        # Groq
        from agents.nia.providers.groq import GroqProvider
        self._providers["groq"] = GroqProvider()

        # Together
        from agents.nia.providers.together import TogetherProvider
        self._providers["together"] = TogetherProvider()

        # DeepSeek
        from agents.nia.providers.deepseek import DeepSeekProvider
        self._providers["deepseek"] = DeepSeekProvider()

        # Google
        from agents.nia.providers.google import GoogleProvider
        self._providers["google"] = GoogleProvider()

        # NVIDIA
        from agents.nia.providers.nvidia import NVIDIAProvider
        self._providers["nvidia"] = NVIDIAProvider()

        # Cerebras
        from agents.nia.providers.cerebras import CerebrasProvider
        self._providers["cerebras"] = CerebrasProvider()

        # Fireworks
        from agents.nia.providers.fireworks import FireworksProvider
        self._providers["fireworks"] = FireworksProvider()

        # OpenRouter
        from agents.nia.providers.openrouter import OpenRouterProvider
        self._providers["openrouter"] = OpenRouterProvider()

        logger.info(f"Registered {len(self._providers)} providers")

    def list_providers(self) -> list[ProviderInfo]:
        """List all registered providers."""
        return [p.get_info() for p in self._providers.values()]

    def get_provider(self, provider_id: str) -> LLMProvider | None:
        """Get a specific provider."""
        return self._providers.get(provider_id)

    def get_active_provider(self) -> LLMProvider | None:
        """Get the currently active provider."""
        if self._active_provider_id:
            return self._providers.get(self._active_provider_id)
        return None

    def get_active_model(self) -> str:
        """Get the currently active model ID."""
        return self._active_model

    def set_active(self, provider_id: str, model: str | None = None) -> bool:
        """Set the active provider and model."""
        if provider_id not in self._providers:
            logger.error(f"Provider not found: {provider_id}")
            return False

        self._active_provider_id = provider_id
        if model:
            self._active_model = model

        # Persist to config
        self._config_manager.set_active_provider(provider_id, model)
        logger.info(f"Active provider set to: {provider_id}/{self._active_model}")
        return True

    def get_all_models(self) -> list[ModelInfo]:
        """Get all models from all providers."""
        models = []
        for provider in self._providers.values():
            if provider.is_configured():
                models.extend(provider.list_models())
        return models

    def search_providers(self, query: str) -> list[ProviderInfo]:
        """Search providers by name or ID."""
        query_lower = query.lower()
        return [
            p.get_info() for p in self._providers.values()
            if query_lower in p.id.lower() or query_lower in p.name.lower()
        ]

    def is_configured(self, provider_id: str) -> bool:
        """Check if a provider is configured."""
        provider = self._providers.get(provider_id)
        return provider is not None and provider.is_configured()

    def get_status(self) -> dict[str, Any]:
        """Get registry status."""
        return {
            "total_providers": len(self._providers),
            "configured": sum(1 for p in self._providers.values() if p.is_configured()),
            "active_provider": self._active_provider_id,
            "active_model": self._active_model,
        }
