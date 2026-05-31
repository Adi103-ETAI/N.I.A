"""Azure OpenAI provider.

Ported from OpenClaude's azure-openai gateway with support for:
- Azure API key authentication
- Azure AD token authentication
- Deployment-based model routing
- Azure-specific API format
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from niaharness.providers.base import (
    AuthMode,
    LLMProvider,
    ProviderAuthConfig,
    ProviderCapabilities,
    ProviderCategory,
    ProviderConfig,
    ProviderModel,
)


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

AZURE_CAPABILITIES = ProviderCapabilities(
    supports_vision=True,
    supports_streaming=True,
    supports_function_calling=True,
    supports_json_mode=True,
    supports_reasoning=False,
    supports_precise_token_count=True,
    supports_usage=True,
    supports_thinking=False,
)

AZURE_MODELS = [
    ProviderModel(
        id="azure-gpt-4o",
        label="GPT-4o (Azure)",
        context_window=128_000,
        max_output_tokens=16_384,
        capabilities=AZURE_CAPABILITIES,
    ),
    ProviderModel(
        id="azure-gpt-4o-mini",
        label="GPT-4o Mini (Azure)",
        context_window=128_000,
        max_output_tokens=16_384,
        capabilities=AZURE_CAPABILITIES,
    ),
    ProviderModel(
        id="azure-gpt-4-turbo",
        label="GPT-4 Turbo (Azure)",
        context_window=128_000,
        max_output_tokens=4_096,
        capabilities=AZURE_CAPABILITIES,
    ),
    ProviderModel(
        id="azure-gpt-4",
        label="GPT-4 (Azure)",
        context_window=8_192,
        max_output_tokens=4_096,
        capabilities=AZURE_CAPABILITIES,
    ),
    ProviderModel(
        id="azure-deployment",
        label="Azure Deployment",
        context_window=128_000,
        max_output_tokens=16_384,
        capabilities=AZURE_CAPABILITIES,
    ),
]


# ---------------------------------------------------------------------------
# Provider implementation
# ---------------------------------------------------------------------------

class AzureOpenAIProvider(LLMProvider):
    """Azure OpenAI provider.

    Supports:
    - Azure API key authentication
    - Azure AD (Entra ID) token authentication
    - Deployment-based routing
    - Azure-specific API versioning

    Environment variables:
    - AZURE_OPENAI_API_KEY: Azure OpenAI API key
    - AZURE_OPENAI_ENDPOINT: Azure OpenAI endpoint URL
    - AZURE_OPENAI_API_VERSION: API version (default: 2024-12-01-preview)
    - AZURE_OPENAI_DEPLOYMENT: Default deployment name
    """

    @property
    def config(self) -> ProviderConfig:
        return ProviderConfig(
            name="azure-openai",
            label="Azure OpenAI",
            category=ProviderCategory.HOSTED,
            auth=ProviderAuthConfig(
                mode=AuthMode.AZURE_KEY,
                api_key_env_vars=["AZURE_OPENAI_API_KEY"],
                base_url_env_vars=["AZURE_OPENAI_ENDPOINT"],
                model_env_vars=["AZURE_OPENAI_DEPLOYMENT"],
                default_base_url="",
                default_model="YOUR-DEPLOYMENT-NAME",
            ),
            transport_kind="openai-compatible",
            models=AZURE_MODELS,
        )

    def get_client(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        api_version: str | None = None,
        azure_ad_token: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Create an Azure OpenAI client.

        Args:
            api_key: Azure OpenAI API key. Falls back to AZURE_OPENAI_API_KEY.
            base_url: Azure OpenAI endpoint. Falls back to AZURE_OPENAI_ENDPOINT.
            api_version: API version. Falls back to AZURE_OPENAI_API_VERSION.
            azure_ad_token: Azure AD token for token-based auth.
            **kwargs: Additional options.

        Returns:
            AsyncAzureOpenAI client instance.
        """
        from openai import AsyncAzureOpenAI

        resolved_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        resolved_endpoint = base_url or os.environ.get("AZURE_OPENAI_ENDPOINT")
        resolved_version = api_version or os.environ.get(
            "AZURE_OPENAI_API_VERSION", "2024-12-01-preview"
        )

        if not resolved_endpoint:
            raise ValueError(
                "Azure OpenAI endpoint required. Set AZURE_OPENAI_ENDPOINT "
                "environment variable or pass base_url parameter."
            )

        client_kwargs: dict[str, Any] = {
            "api_version": resolved_version,
        }

        if resolved_key:
            client_kwargs["api_key"] = resolved_key
        elif azure_ad_token:
            client_kwargs["azure_ad_token"] = azure_ad_token
        else:
            raise ValueError(
                "Either api_key or azure_ad_token required. "
                "Set AZURE_OPENAI_API_KEY or provide azure_ad_token."
            )

        if resolved_endpoint:
            client_kwargs["azure_endpoint"] = resolved_endpoint

        return AsyncAzureOpenAI(**client_kwargs)

    def get_deployment_url(self, deployment: str, api_version: str | None = None) -> str:
        """Construct the full deployment URL.

        Args:
            deployment: Deployment name.
            api_version: API version (defaults to 2024-12-01-preview).

        Returns:
            Full URL for the deployment endpoint.
        """
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
        version = api_version or os.environ.get(
            "AZURE_OPENAI_API_VERSION", "2024-12-01-preview"
        )
        return f"{endpoint}/openai/deployments/{deployment}/chat/completions?api-version={version}"

    def resolve_api_key(self, api_key: str | None = None) -> str | None:
        """Resolve Azure API key."""
        if api_key:
            return api_key
        return os.environ.get("AZURE_OPENAI_API_KEY")
