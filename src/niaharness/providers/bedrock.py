"""AWS Bedrock provider.

Ported from OpenClaude's bedrock gateway with support for:
- AWS ADC (Ambient Default Credentials) authentication
- Claude models via Bedrock
- Bedrock-specific API format
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
# Model definitions (ported from OpenClaude bedrock gateway)
# ---------------------------------------------------------------------------

BEDROCK_CAPABILITIES = ProviderCapabilities(
    supports_vision=True,
    supports_streaming=True,
    supports_function_calling=True,
    supports_json_mode=True,
    supports_reasoning=True,
    supports_precise_token_count=False,
    supports_usage=False,
    supports_thinking=True,
)

BEDROCK_MODELS = [
    ProviderModel(
        id="bedrock-claude-opus",
        label="Claude Opus (Bedrock)",
        context_window=200_000,
        max_output_tokens=8192,
        capabilities=BEDROCK_CAPABILITIES,
        api_name="us.anthropic.claude-opus-4-6-v1",
    ),
    ProviderModel(
        id="bedrock-claude-sonnet",
        label="Claude Sonnet (Bedrock)",
        context_window=200_000,
        max_output_tokens=8192,
        capabilities=BEDROCK_CAPABILITIES,
        api_name="us.anthropic.claude-sonnet-4-20250514-v1:0",
    ),
    ProviderModel(
        id="bedrock-claude-haiku",
        label="Claude Haiku (Bedrock)",
        context_window=144_000,
        max_output_tokens=8192,
        capabilities=ProviderCapabilities(
            supports_vision=True,
            supports_streaming=True,
            supports_function_calling=True,
            supports_json_mode=True,
            supports_reasoning=False,
            supports_precise_token_count=False,
            supports_usage=False,
            supports_thinking=False,
        ),
        api_name="us.anthropic.claude-haiku-4-5-20250514-v1:0",
    ),
]


# ---------------------------------------------------------------------------
# Provider implementation
# ---------------------------------------------------------------------------

class BedrockProvider(LLMProvider):
    """AWS Bedrock provider.

    Uses AWS ambient default credentials (ADC) for authentication.
    Requires boto3 to be installed and AWS credentials configured.

    Environment variables:
    - AWS_REGION: AWS region (default: us-east-1)
    - AWS_PROFILE: AWS profile to use
    - AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY: Explicit credentials
    """

    @property
    def config(self) -> ProviderConfig:
        return ProviderConfig(
            name="bedrock",
            label="AWS Bedrock",
            category=ProviderCategory.HOSTED,
            auth=ProviderAuthConfig(
                mode=AuthMode.AWS_ADC,
                api_key_env_vars=[],
                base_url_env_vars=["AWS_BEDROCK_ENDPOINT"],
                model_env_vars=["BEDROCK_MODEL"],
                default_base_url="",
                default_model="us.anthropic.claude-opus-4-6-v1",
            ),
            transport_kind="bedrock",
            models=BEDROCK_MODELS,
            supports_model_routing=True,
        )

    def get_client(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        region: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Create a Bedrock-compatible client.

        Uses boto3 for AWS authentication and the anthropic-bedrock SDK
        for Bedrock-specific API format.

        Args:
            api_key: Not used for Bedrock (uses AWS ADC).
            base_url: Optional custom endpoint URL.
            region: AWS region (falls back to AWS_REGION env var).
            **kwargs: Additional options.

        Returns:
            AsyncAnthropicBedrock client instance.
        """
        try:
            from anthropic import AsyncAnthropicBedrock
        except ImportError:
            raise ImportError(
                "anthropic package with bedrock support required: "
                "pip install anthropic[bedrock]"
            )

        resolved_region = region or os.environ.get("AWS_REGION", "us-east-1")
        resolved_url = self.resolve_base_url(base_url)

        client_kwargs: dict[str, Any] = {"region": resolved_region}
        if resolved_url:
            client_kwargs["base_url"] = resolved_url

        return AsyncAnthropicBedrock(**client_kwargs)

    def get_model_id(self, model: str) -> str:
        """Map a model ID to its Bedrock API name.

        Handles both friendly names and full ARN-style names.
        """
        # Check if it's a known model
        model_info = self.get_model_info(model)
        if model_info:
            return model_info.effective_api_name

        # If it looks like a full Bedrock model ID already, return as-is
        if "." in model and ("anthropic" in model or "claude" in model):
            return model

        # Default fallback
        return model

    def resolve_api_key(self, api_key: str | None = None) -> str | None:
        """Bedrock doesn't use API keys; returns None."""
        return None
