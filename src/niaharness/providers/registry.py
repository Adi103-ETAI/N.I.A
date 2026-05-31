"""Unified Provider Registry with switching capabilities.

Single source of truth for all LLM providers.
Supports dynamic switching between providers and models.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from niaharness.providers.base import LLMProvider, ProviderModel

logger = logging.getLogger(__name__)

CONFIG_DIR = Path.home() / ".nia"
CONFIG_FILE = CONFIG_DIR / "providers.json"


@dataclass
class ProviderState:
    """State of a provider in the registry."""
    name: str
    configured: bool = False
    api_key: str | None = None
    base_url: str | None = None
    active_model: str | None = None


class ProviderRegistry:
    """Unified registry for all LLM providers.

    Features:
    - Register multiple providers
    - Switch between providers dynamically
    - Switch between models within a provider
    - Persist configuration to disk
    - Auto-detect from environment variables

    Usage:
        registry = ProviderRegistry()
        await registry.initialize()

        # List providers
        providers = registry.list_providers()

        # Switch provider
        registry.set_active("openai", "gpt-4o")

        # Get active provider
        provider = registry.get_active_provider()
        client = provider.get_client()
    """

    def __init__(self) -> None:
        self._providers: dict[str, LLMProvider] = {}
        self._states: dict[str, ProviderState] = {}
        self._active_provider_id: str = ""
        self._active_model: str = ""
        self._config_path: Path = CONFIG_FILE

    async def initialize(self) -> None:
        """Initialize registry with all providers and load config."""
        # Register all built-in providers
        self._register_builtin_providers()

        # Load saved config
        self._load_config()

        # Auto-detect from environment
        self._auto_detect_providers()

        # Fetch models from configured providers
        for name, state in self._states.items():
            if state.configured and name in self._providers:
                try:
                    await self._providers[name].fetch_models()
                except Exception as e:
                    logger.debug(f"Failed to fetch models from {name}: {e}")

        # Set active provider from config
        if self._active_provider_id and self._active_provider_id in self._providers:
            pass  # Already set
        elif self._providers:
            # Find first configured provider
            for name, state in self._states.items():
                if state.configured:
                    self._active_provider_id = name
                    break

        logger.info(f"Registry initialized. Active: {self._active_provider_id}/{self._active_model}")

    def _register_builtin_providers(self) -> None:
        """Register all built-in providers."""
        from niaharness.providers.anthropic import AnthropicProvider
        from niaharness.providers.openai import (
            OpenAIProvider,
            OllamaProvider,
            OpenRouterProvider,
            GroqProvider,
            TogetherProvider,
            DeepSeekProvider,
            GoogleProvider,
            NVIDIAProvider,
            CerebrasProvider,
            FireworksProvider,
        )
        from niaharness.providers.bedrock import BedrockProvider
        from niaharness.providers.vertex import VertexProvider
        from niaharness.providers.azure import AzureOpenAIProvider
        from niaharness.providers.mistral import MistralProvider

        providers = {
            "anthropic": AnthropicProvider(),
            "openai": OpenAIProvider(),
            "ollama": OllamaProvider(),
            "openrouter": OpenRouterProvider(),
            "groq": GroqProvider(),
            "together": TogetherProvider(),
            "deepseek": DeepSeekProvider(),
            "google": GoogleProvider(),
            "nvidia": NVIDIAProvider(),
            "cerebras": CerebrasProvider(),
            "fireworks": FireworksProvider(),
            "bedrock": BedrockProvider(),
            "vertex": VertexProvider(),
            "azure": AzureOpenAIProvider(),
            "mistral": MistralProvider(),
        }

        for name, provider in providers.items():
            self._providers[name] = provider
            self._states[name] = ProviderState(name=name)

        logger.info(f"Registered {len(providers)} providers")

    def _load_config(self) -> None:
        """Load provider config from disk."""
        if not self._config_path.exists():
            return

        try:
            data = json.loads(self._config_path.read_text(encoding="utf-8"))
            self._active_provider_id = data.get("active_provider", "")
            self._active_model = data.get("active_model", "")

            for name, pdata in data.get("providers", {}).items():
                if name in self._states:
                    state = self._states[name]
                    state.api_key = pdata.get("api_key")
                    state.base_url = pdata.get("base_url")
                    state.active_model = pdata.get("active_model")
                    state.configured = bool(state.api_key or state.base_url)

                    # Configure the provider
                    if name in self._providers:
                        self._providers[name].configure(
                            api_key=state.api_key,
                            base_url=state.base_url,
                        )

            logger.info(f"Loaded config from {self._config_path}")
        except Exception as e:
            logger.warning(f"Failed to load config: {e}")

    def save_config(self) -> None:
        """Save provider config to disk."""
        self._config_path.parent.mkdir(parents=True, exist_ok=True)

        providers_data = {}
        for name, state in self._states.items():
            if state.configured or state.api_key:
                providers_data[name] = {
                    "api_key": state.api_key,
                    "base_url": state.base_url,
                    "active_model": state.active_model,
                }

        data = {
            "active_provider": self._active_provider_id,
            "active_model": self._active_model,
            "providers": providers_data,
        }

        self._config_path.write_text(
            json.dumps(data, indent=2) + "\n",
            encoding="utf-8",
        )
        logger.info(f"Saved config to {self._config_path}")

    def _auto_detect_providers(self) -> None:
        """Auto-detect providers from environment variables."""
        import os

        env_map = {
            "anthropic": {"api_key_env": "ANTHROPIC_API_KEY", "base_url_env": "ANTHROPIC_BASE_URL"},
            "openai": {"api_key_env": "OPENAI_API_KEY", "base_url_env": "OPENAI_BASE_URL"},
            "groq": {"api_key_env": "GROQ_API_KEY"},
            "together": {"api_key_env": "TOGETHER_API_KEY"},
            "deepseek": {"api_key_env": "DEEPSEEK_API_KEY"},
            "google": {"api_key_env": "GOOGLE_API_KEY"},
            "nvidia": {"api_key_env": "NVIDIA_API_KEY"},
            "openrouter": {"api_key_env": "OPENROUTER_API_KEY"},
        }

        for provider_id, env_config in env_map.items():
            api_key = os.environ.get(env_config.get("api_key_env", ""), "")
            base_url = os.environ.get(env_config.get("base_url_env", ""), "")

            if api_key or base_url:
                if provider_id in self._states:
                    state = self._states[provider_id]
                    if api_key:
                        state.api_key = api_key
                    if base_url:
                        state.base_url = base_url
                    state.configured = True

                    # Configure the provider
                    if provider_id in self._providers:
                        self._providers[provider_id].configure(
                            api_key=state.api_key,
                            base_url=state.base_url,
                        )

    def list_providers(self) -> list[dict[str, Any]]:
        """List all providers with their status."""
        result = []
        for name, provider in self._providers.items():
            state = self._states.get(name, ProviderState(name=name))
            config = provider.config
            result.append({
                "id": name,
                "name": config.label,
                "configured": state.configured,
                "active": name == self._active_provider_id,
                "models": len(config.models),
                "category": config.category.value,
            })
        return result

    def get_provider(self, name: str) -> LLMProvider | None:
        """Get a provider by name."""
        return self._providers.get(name)

    def get_active_provider(self) -> LLMProvider | None:
        """Get the currently active provider."""
        return self._providers.get(self._active_provider_id)

    def get_active_provider_id(self) -> str:
        """Get the currently active provider ID."""
        return self._active_provider_id

    def get_active_model(self) -> str:
        """Get the currently active model."""
        return self._active_model

    def set_active(self, provider_id: str, model: str | None = None) -> bool:
        """Set the active provider and optionally model.

        Args:
            provider_id: Provider ID (e.g., "openai", "anthropic")
            model: Model ID (e.g., "gpt-4o", "claude-sonnet-4")

        Returns:
            True if successful, False if provider not found
        """
        if provider_id not in self._providers:
            logger.error(f"Provider not found: {provider_id}")
            return False

        self._active_provider_id = provider_id
        if model:
            self._active_model = model

        # Update state
        if provider_id in self._states:
            self._states[provider_id].active_model = model

        # Save config
        self.save_config()

        logger.info(f"Active provider: {provider_id}/{self._active_model}")
        return True

    def set_provider_config(
        self,
        provider_id: str,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> bool:
        """Configure a provider with credentials.

        Args:
            provider_id: Provider ID
            api_key: API key
            base_url: Custom base URL

        Returns:
            True if successful
        """
        if provider_id not in self._providers:
            logger.error(f"Provider not found: {provider_id}")
            return False

        state = self._states[provider_id]
        if api_key:
            state.api_key = api_key
        if base_url:
            state.base_url = base_url
        state.configured = True

        # Configure the provider
        self._providers[provider_id].configure(
            api_key=state.api_key,
            base_url=state.base_url,
        )

        # Save config
        self.save_config()

        logger.info(f"Configured provider: {provider_id}")
        return True

    async def configure_and_fetch_models(
        self,
        provider_id: str,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> list[dict[str, Any]]:
        """Configure provider and fetch available models.

        This is the method to call when user provides an API key.
        It configures the provider and returns the list of available models.

        Args:
            provider_id: Provider ID
            api_key: API key
            base_url: Custom base URL

        Returns:
            List of available models
        """
        self.set_provider_config(provider_id, api_key, base_url)

        # Fetch models from the provider
        provider = self._providers.get(provider_id)
        if provider:
            try:
                models = await provider.fetch_models()
                return [
                    {
                        "id": m.id,
                        "label": m.label,
                        "context_window": m.context_window,
                        "max_output": m.max_output_tokens,
                    }
                    for m in models
                ]
            except Exception as e:
                logger.error(f"Failed to fetch models from {provider_id}: {e}")

        return []

    def get_all_models(self) -> list[dict[str, Any]]:
        """Get all models from all configured providers."""
        models = []
        for name, provider in self._providers.items():
            state = self._states.get(name, ProviderState(name=name))
            if state.configured or name == "ollama":  # Ollama is always available
                for model in provider.list_models():
                    models.append({
                        "id": model.id,
                        "label": model.label,
                        "provider": name,
                        "context_window": model.context_window,
                        "max_output": model.max_output_tokens,
                        "active": (name == self._active_provider_id and model.id == self._active_model),
                    })
        return models

    def search_providers(self, query: str) -> list[dict[str, Any]]:
        """Search providers by name."""
        query_lower = query.lower()
        return [
            p for p in self.list_providers()
            if query_lower in p["id"].lower() or query_lower in p["name"].lower()
        ]

    def get_status(self) -> dict[str, Any]:
        """Get registry status."""
        configured = sum(1 for s in self._states.values() if s.configured)
        return {
            "total_providers": len(self._providers),
            "configured": configured,
            "active_provider": self._active_provider_id,
            "active_model": self._active_model,
        }
