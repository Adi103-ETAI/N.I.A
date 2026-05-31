"""Mistral AI provider.

Ported from OpenClaude's mistral gateway with support for:
- Mistral API key authentication
- OpenAI-compatible API format
- Mistral-specific model routing
- Devstral and other Mistral models
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
# Model definitions (ported from OpenClaude mistral models)
# ---------------------------------------------------------------------------

MISTRAL_CAPABILITIES = ProviderCapabilities(
    supports_vision=False,
    supports_streaming=True,
    supports_function_calling=True,
    supports_json_mode=True,
    supports_reasoning=False,
    supports_precise_token_count=False,
    supports_usage=True,
    supports_thinking=False,
)

MISTRAL_MODELS = [
    ProviderModel(
        id="mistral-large-latest",
        label="Mistral Large Latest",
        context_window=256_000,
        max_output_tokens=32_768,
        capabilities=MISTRAL_CAPABILITIES,
    ),
    ProviderModel(
        id="mistral-small-latest",
        label="Mistral Small Latest",
        context_window=256_000,
        max_output_tokens=32_768,
        capabilities=MISTRAL_CAPABILITIES,
    ),
    ProviderModel(
        id="devstral-latest",
        label="Devstral Latest",
        context_window=256_000,
        max_output_tokens=32_768,
        capabilities=MISTRAL_CAPABILITIES,
    ),
    ProviderModel(
        id="ministral-3b-latest",
        label="Ministral 3B Latest",
        context_window=256_000,
        max_output_tokens=32_768,
        capabilities=MISTRAL_CAPABILITIES,
    ),
    ProviderModel(
        id="mixtral-8x7b-32768",
        label="Mixtral 8x7B 32768",
        context_window=32_768,
        max_output_tokens=32_768,
        capabilities=MISTRAL_CAPABILITIES,
    ),
    ProviderModel(
        id="codestral",
        label="Codestral",
        context_window=32_768,
        max_output_tokens=8_192,
        capabilities=MISTRAL_CAPABILITIES,
    ),
]


# ---------------------------------------------------------------------------
# Provider implementation
# ---------------------------------------------------------------------------

class MistralProvider(LLMProvider):
    """Mistral AI provider.

    Uses the OpenAI-compatible API with Mistral-specific handling.
    Requires MISTRAL_API_KEY environment variable.

    Environment variables:
    - MISTRAL_API_KEY: Mistral API key
    - MISTRAL_MODEL: Default model to use
    """

    @property
    def config(self) -> ProviderConfig:
        return ProviderConfig(
            name="mistral",
            label="Mistral AI",
            category=ProviderCategory.HOSTED,
            auth=ProviderAuthConfig(
                mode=AuthMode.API_KEY,
                api_key_env_vars=["MISTRAL_API_KEY"],
                base_url_env_vars=["MISTRAL_BASE_URL"],
                model_env_vars=["MISTRAL_MODEL"],
                default_base_url="https://api.mistral.ai/v1",
                default_model="mistral-large-latest",
            ),
            transport_kind="openai-compatible",
            models=MISTRAL_MODELS,
        )

    def get_client(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Create a Mistral-compatible OpenAI client.

        Args:
            api_key: Mistral API key. Falls back to MISTRAL_API_KEY.
            base_url: Override base URL. Falls back to MISTRAL_BASE_URL.
            **kwargs: Additional options.

        Returns:
            AsyncOpenAI client configured for Mistral.
        """
        from openai import AsyncOpenAI

        resolved_key = self.resolve_api_key(api_key)
        if not resolved_key:
            raise ValueError(
                "No API key found. Set MISTRAL_API_KEY environment variable "
                "or provide api_key parameter."
            )

        resolved_url = self.resolve_base_url(base_url)

        client_kwargs: dict[str, Any] = {
            "api_key": resolved_key,
            "base_url": resolved_url,
        }

        return AsyncOpenAI(**client_kwargs)

    def get_model_id(self, model: str) -> str:
        """Get the Mistral model ID.

        Handles model aliasing and ensures proper Mistral model names.
        """
        # Check if it's a known model
        model_info = self.get_model_info(model)
        if model_info:
            return model_info.id

        # Return as-is if already a valid model name
        return model

    def supports_thinking(self, model: str) -> bool:
        """Check if a model supports extended thinking.

        Mistral models currently don't support extended thinking.
        """
        return False
