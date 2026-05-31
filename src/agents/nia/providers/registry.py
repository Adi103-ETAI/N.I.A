"""N.I.A Provider Registry - Uses niaharness unified registry.

This is a thin wrapper around niaharness.providers.registry
that adds NIA-specific features like config persistence.
"""

from __future__ import annotations

import logging
from typing import Any

from niaharness.providers.registry import ProviderRegistry as NiaHarnessRegistry
from niaharness.providers.base import LLMProvider, ProviderModel

logger = logging.getLogger(__name__)


class ProviderRegistry:
    """N.I.A's provider registry.

    Wraps niaharness.providers.registry.ProviderRegistry
    and adds NIA-specific features.
    """

    def __init__(self, config_manager: Any = None) -> None:
        self._config_manager = config_manager
        self._registry = NiaHarnessRegistry()

    async def initialize(self) -> None:
        """Initialize all providers."""
        await self._registry.initialize()
        logger.info(f"NIA registry initialized: {self._registry.get_status()}")

    def list_providers(self) -> list[dict[str, Any]]:
        """List all providers with status."""
        return self._registry.list_providers()

    def get_provider(self, provider_id: str) -> LLMProvider | None:
        """Get a provider by ID."""
        return self._registry.get_provider(provider_id)

    def get_active_provider(self) -> LLMProvider | None:
        """Get the currently active provider."""
        return self._registry.get_active_provider()

    def get_active_provider_id(self) -> str:
        """Get the active provider ID."""
        return self._registry.get_active_provider_id()

    def get_active_model(self) -> str:
        """Get the active model."""
        return self._registry.get_active_model()

    def set_active(self, provider_id: str, model: str | None = None) -> bool:
        """Set the active provider and model."""
        return self._registry.set_active(provider_id, model)

    def set_provider_config(
        self,
        provider_id: str,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> bool:
        """Configure a provider."""
        return self._registry.set_provider_config(provider_id, api_key, base_url)

    async def configure_and_fetch_models(
        self,
        provider_id: str,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> list[dict[str, Any]]:
        """Configure provider and fetch available models."""
        return await self._registry.configure_and_fetch_models(provider_id, api_key, base_url)

    def get_all_models(self) -> list[dict[str, Any]]:
        """Get all models from all providers."""
        return self._registry.get_all_models()

    def search_providers(self, query: str) -> list[dict[str, Any]]:
        """Search providers."""
        return self._registry.search_providers(query)

    def get_status(self) -> dict[str, Any]:
        """Get registry status."""
        return self._registry.get_status()
