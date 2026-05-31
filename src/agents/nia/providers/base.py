"""Abstract base class for LLM providers.

Every provider (Anthropic, OpenAI, Ollama, etc.) implements this interface.
"""

from __future__ import annotations

import abc
import logging
from typing import Any, AsyncIterator

from agents.nia.providers.types import (
    LLMRequest,
    LLMResponse,
    ModelInfo,
    ProviderInfo,
)

logger = logging.getLogger(__name__)


class LLMProvider(abc.ABC):
    """Abstract base class for LLM providers.

    To add a new provider:
    1. Subclass LLMProvider
    2. Implement all abstract methods
    3. Register in the provider registry

    Model listing:
    - Override `list_models()` to return hardcoded models
    - Override `fetch_models()` to fetch from API (cached automatically)
    - Or both: `list_models()` returns cached, `fetch_models()` refreshes
    """

    @property
    @abc.abstractmethod
    def id(self) -> str:
        """Unique provider identifier (e.g., 'anthropic', 'openai')."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable provider name."""

    @abc.abstractmethod
    def get_info(self) -> ProviderInfo:
        """Get provider metadata and available models."""

    @abc.abstractmethod
    def list_models(self) -> list[ModelInfo]:
        """List all available models from this provider."""

    @abc.abstractmethod
    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Send a completion request and return the response."""

    @abc.abstractmethod
    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        """Stream a completion response token by token."""

    def configure(self, api_key: str | None = None, base_url: str | None = None, **kwargs: Any) -> None:
        """Configure provider credentials. Override if needed."""
        if api_key is not None:
            self._api_key = api_key
        if base_url is not None:
            self._base_url = base_url

    def is_configured(self) -> bool:
        """Check if the provider has valid credentials."""
        return hasattr(self, '_api_key') and bool(self._api_key)

    async def fetch_models(self) -> list[ModelInfo]:
        """Fetch models from the provider's API.

        Override this method for providers that support dynamic model listing.
        Default implementation returns the result of list_models() (hardcoded).
        Returns cached models if available, otherwise fetches fresh.
        """
        return self.list_models()

    def _build_headers(self) -> dict[str, str]:
        """Build HTTP headers for API calls."""
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if hasattr(self, '_api_key') and self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers
