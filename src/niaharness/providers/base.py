"""Base classes for LLM provider implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator


class ProviderCategory(Enum):
    """Provider classification."""

    NATIVE = "native"
    HOSTED = "hosted"
    LOCAL = "local"


class AuthMode(Enum):
    """Authentication mode for providers."""

    API_KEY = "api-key"
    OAUTH = "oauth"
    AWS_ADC = "aws-adc"
    GCP_ADC = "gcp-adc"
    AZURE_KEY = "azure-key"


@dataclass(frozen=True)
class ProviderCapabilities:
    """Capabilities supported by a provider."""

    supports_vision: bool = False
    supports_streaming: bool = True
    supports_function_calling: bool = True
    supports_json_mode: bool = True
    supports_reasoning: bool = False
    supports_precise_token_count: bool = False
    supports_usage: bool = True
    supports_thinking: bool = False


@dataclass(frozen=True)
class ProviderModel:
    """Model definition for a provider."""

    id: str
    label: str
    context_window: int
    max_output_tokens: int
    capabilities: ProviderCapabilities = field(default_factory=ProviderCapabilities)
    api_name: str | None = None

    @property
    def effective_api_name(self) -> str:
        """Return the API model name (defaults to id if not set)."""
        return self.api_name or self.id


@dataclass(frozen=True)
class ProviderAuthConfig:
    """Authentication configuration for a provider."""

    mode: AuthMode
    api_key_env_vars: list[str] = field(default_factory=list)
    base_url_env_vars: list[str] = field(default_factory=list)
    model_env_vars: list[str] = field(default_factory=list)
    default_base_url: str = ""
    default_model: str = ""


@dataclass
class ProviderConfig:
    """Complete provider configuration."""

    name: str
    label: str
    category: ProviderCategory
    auth: ProviderAuthConfig
    transport_kind: str = "openai-compatible"
    models: list[ProviderModel] = field(default_factory=list)
    is_first_party: bool = False
    supports_model_routing: bool = True


class LLMProvider(ABC):
    """Abstract base class for LLM providers.

    All provider implementations must subclass this and implement:
    - config: Provider configuration
    - get_client: Create an API client for this provider
    - fetch_models: Dynamically fetch models from provider API
    """

    def __init__(self) -> None:
        self._fetched_models: list[ProviderModel] | None = None

    @property
    @abstractmethod
    def config(self) -> ProviderConfig:
        """Return the provider configuration."""

    @abstractmethod
    def get_client(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Create an API client for this provider.

        Args:
            api_key: API key (falls back to env var if None).
            base_url: Override base URL.
            **kwargs: Provider-specific options.

        Returns:
            An async client suitable for streaming.
        """

    def configure(self, api_key: str | None = None, base_url: str | None = None) -> None:
        """Configure provider with credentials."""
        if api_key:
            self._api_key = api_key
        if base_url:
            self._base_url = base_url
        self._fetched_models = None  # Reset cache

    def resolve_api_key(self, api_key: str | None = None) -> str | None:
        """Resolve API key from argument or environment variables."""
        import os

        if api_key:
            return api_key

        for env_var in self.config.auth.api_key_env_vars:
            value = os.environ.get(env_var)
            if value:
                return value

        return None

    def resolve_base_url(self, base_url: str | None = None) -> str:
        """Resolve base URL from argument or environment variables."""
        import os

        if base_url:
            return base_url

        for env_var in self.config.auth.base_url_env_vars:
            value = os.environ.get(env_var)
            if value:
                return value

        return self.config.auth.default_base_url

    def resolve_model(self, model: str | None = None) -> str:
        """Resolve model from argument or environment variables."""
        import os

        if model:
            return model

        for env_var in self.config.auth.model_env_vars:
            value = os.environ.get(env_var)
            if value:
                return value

        return self.config.auth.default_model

    def get_model_info(self, model_id: str) -> ProviderModel | None:
        """Get model information by ID."""
        models = self.list_models()
        for model in models:
            if model.id == model_id or model.effective_api_name == model_id:
                return model
        return None

    def list_models(self) -> list[ProviderModel]:
        """List all available models.

        Returns fetched models if available, otherwise hardcoded defaults.
        """
        if self._fetched_models is not None:
            return self._fetched_models
        return self.config.models

    async def fetch_models(self) -> list[ProviderModel]:
        """Dynamically fetch models from the provider's API.

        Override this method in subclasses to fetch models from the API.
        Falls back to hardcoded models if API call fails.

        Returns:
            List of available models.
        """
        # Default: return hardcoded models
        return self.config.models
