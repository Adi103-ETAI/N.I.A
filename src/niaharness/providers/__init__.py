"""Unified LLM Provider Registry for NiaHarness.

Single source of truth for all LLM providers with dynamic switching.

Usage:
    from niaharness.providers import ProviderRegistry

    registry = ProviderRegistry()
    await registry.initialize()

    # List providers
    providers = registry.list_providers()

    # Switch provider and model
    registry.set_active("openai", "gpt-4o")

    # Get active provider
    provider = registry.get_active_provider()
    client = provider.get_client()
"""

from niaharness.providers.base import (
    LLMProvider,
    ProviderModel,
    ProviderCapabilities,
    ProviderConfig,
    ProviderCategory,
    AuthMode,
    ProviderAuthConfig,
)
from niaharness.providers.registry import ProviderRegistry

__all__ = [
    # Base classes
    "LLMProvider",
    "ProviderModel",
    "ProviderCapabilities",
    "ProviderConfig",
    "ProviderCategory",
    "AuthMode",
    "ProviderAuthConfig",
    # Registry
    "ProviderRegistry",
]


def get_provider_class(name: str) -> type[LLMProvider] | None:
    """Get a provider class by name."""
    from niaharness.providers import registry as _reg
    # Import all provider classes
    from niaharness.providers.anthropic import AnthropicProvider
    from niaharness.providers.openai import (
        OpenAIProvider, OllamaProvider, OpenRouterProvider,
        GroqProvider, TogetherProvider, DeepSeekProvider,
        GoogleProvider, NVIDIAProvider, CerebrasProvider, FireworksProvider,
    )
    from niaharness.providers.bedrock import BedrockProvider
    from niaharness.providers.vertex import VertexProvider
    from niaharness.providers.azure import AzureOpenAIProvider
    from niaharness.providers.mistral import MistralProvider

    classes = {
        "anthropic": AnthropicProvider,
        "openai": OpenAIProvider,
        "ollama": OllamaProvider,
        "openrouter": OpenRouterProvider,
        "groq": GroqProvider,
        "together": TogetherProvider,
        "deepseek": DeepSeekProvider,
        "google": GoogleProvider,
        "nvidia": NVIDIAProvider,
        "cerebras": CerebrasProvider,
        "fireworks": FireworksProvider,
        "bedrock": BedrockProvider,
        "vertex": VertexProvider,
        "azure": AzureOpenAIProvider,
        "mistral": MistralProvider,
    }
    return classes.get(name)
