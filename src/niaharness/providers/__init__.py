"""LLM provider implementations for NiaHarness.

Provides provider classes for:
- Anthropic (native + OAuth)
- AWS Bedrock
- Google Vertex AI
- Azure OpenAI
- Mistral AI
"""

from typing import Any

from niaharness.providers.base import (
    LLMProvider,
    ProviderModel,
    ProviderCapabilities,
    ProviderConfig,
    ProviderCategory,
    AuthMode,
    ProviderAuthConfig,
)
from niaharness.providers.anthropic import AnthropicProvider
from niaharness.providers.bedrock import BedrockProvider
from niaharness.providers.vertex import VertexProvider
from niaharness.providers.azure import AzureOpenAIProvider
from niaharness.providers.mistral import MistralProvider

__all__ = [
    # Base classes
    "LLMProvider",
    "ProviderModel",
    "ProviderCapabilities",
    "ProviderConfig",
    "ProviderCategory",
    "AuthMode",
    "ProviderAuthConfig",
    # Provider implementations
    "AnthropicProvider",
    "BedrockProvider",
    "VertexProvider",
    "AzureOpenAIProvider",
    "MistralProvider",
]

PROVIDER_REGISTRY: dict[str, type[LLMProvider]] = {
    "anthropic": AnthropicProvider,
    "bedrock": BedrockProvider,
    "vertex": VertexProvider,
    "azure": AzureOpenAIProvider,
    "mistral": MistralProvider,
}


def get_provider_class(name: str) -> type[LLMProvider] | None:
    """Get a provider class by name."""
    return PROVIDER_REGISTRY.get(name)


def list_providers() -> list[str]:
    """List all registered provider names."""
    return list(PROVIDER_REGISTRY.keys())


def create_provider(name: str, **kwargs: Any) -> LLMProvider:
    """Create a provider instance by name.

    Args:
        name: Provider name (e.g., "anthropic", "bedrock", "vertex").
        **kwargs: Additional keyword arguments passed to provider constructor.

    Returns:
        Provider instance.

    Raises:
        ValueError: If provider name is not registered.
    """
    from typing import Any as _Any

    cls = PROVIDER_REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown provider: {name!r}. "
            f"Available providers: {', '.join(PROVIDER_REGISTRY.keys())}"
        )
    return cls(**kwargs)
