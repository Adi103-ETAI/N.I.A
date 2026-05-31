"""Google Vertex AI provider.

Ported from OpenClaude's vertex gateway with support for:
- GCP ADC (Application Default Credentials) authentication
- Claude models via Vertex AI
- Gemini models via Vertex AI
- Vertex-specific API format
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
# Model definitions (ported from OpenClaude vertex gateway + gemini models)
# ---------------------------------------------------------------------------

VERTEX_CLAUDE_CAPABILITIES = ProviderCapabilities(
    supports_vision=True,
    supports_streaming=True,
    supports_function_calling=True,
    supports_json_mode=True,
    supports_reasoning=True,
    supports_precise_token_count=False,
    supports_usage=False,
    supports_thinking=True,
)

VERTEX_GEMINI_CAPABILITIES = ProviderCapabilities(
    supports_vision=True,
    supports_streaming=True,
    supports_function_calling=True,
    supports_json_mode=True,
    supports_reasoning=True,
    supports_precise_token_count=False,
    supports_usage=False,
    supports_thinking=True,
)

VERTEX_MODELS = [
    # Claude models via Vertex AI
    ProviderModel(
        id="vertex-claude-opus",
        label="Claude Opus (Vertex)",
        context_window=200_000,
        max_output_tokens=8192,
        capabilities=VERTEX_CLAUDE_CAPABILITIES,
        api_name="claude-opus-4-6",
    ),
    ProviderModel(
        id="vertex-claude-sonnet",
        label="Claude Sonnet (Vertex)",
        context_window=200_000,
        max_output_tokens=8192,
        capabilities=VERTEX_CLAUDE_CAPABILITIES,
        api_name="claude-sonnet-4-6",
    ),
    # Gemini models via Vertex AI
    ProviderModel(
        id="vertex-gemini-2.5-pro",
        label="Gemini 2.5 Pro (Vertex)",
        context_window=1_048_576,
        max_output_tokens=65_536,
        capabilities=VERTEX_GEMINI_CAPABILITIES,
        api_name="gemini-2.5-pro",
    ),
    ProviderModel(
        id="vertex-gemini-2.5-flash",
        label="Gemini 2.5 Flash (Vertex)",
        context_window=1_048_576,
        max_output_tokens=65_536,
        capabilities=VERTEX_GEMINI_CAPABILITIES,
        api_name="gemini-2.5-flash",
    ),
    ProviderModel(
        id="vertex-gemini-2.0-flash",
        label="Gemini 2.0 Flash (Vertex)",
        context_window=1_048_576,
        max_output_tokens=8_192,
        capabilities=VERTEX_GEMINI_CAPABILITIES,
        api_name="gemini-2.0-flash",
    ),
]


# ---------------------------------------------------------------------------
# Provider implementation
# ---------------------------------------------------------------------------

class VertexProvider(LLMProvider):
    """Google Vertex AI provider.

    Uses GCP Application Default Credentials for authentication.
    Requires google-auth and google-cloud-aiplatform packages.

    Environment variables:
    - GCLOUD_PROJECT: GCP project ID
    - GCLOUD_REGION: GCP region (default: us-east5)
    - GOOGLE_APPLICATION_CREDENTIALS: Path to service account key
    """

    @property
    def config(self) -> ProviderConfig:
        return ProviderConfig(
            name="vertex",
            label="Google Vertex AI",
            category=ProviderCategory.HOSTED,
            auth=ProviderAuthConfig(
                mode=AuthMode.GCP_ADC,
                api_key_env_vars=[],
                base_url_env_vars=["VERTEX_ENDPOINT"],
                model_env_vars=["VERTEX_MODEL"],
                default_base_url="",
                default_model="gemini-2.5-pro",
            ),
            transport_kind="vertex",
            models=VERTEX_MODELS,
            supports_model_routing=True,
        )

    def get_client(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        project: str | None = None,
        region: str | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Create a Vertex AI client.

        Uses google-auth for GCP ADC and the anthropic SDK for
        Claude models on Vertex.

        Args:
            api_key: Not used for Vertex (uses GCP ADC).
            base_url: Optional custom endpoint URL.
            project: GCP project ID (falls back to GCLOUD_PROJECT env var).
            region: GCP region (falls back to GCLOUD_REGION env var).
            model: Model to use.
            **kwargs: Additional options.

        Returns:
            Client instance suitable for the specified model type.
        """
        resolved_project = project or os.environ.get("GCLOUD_PROJECT")
        resolved_region = region or os.environ.get("GCLOUD_REGION", "us-east5")
        resolved_model = self.resolve_model(model)

        # Determine if this is a Claude or Gemini model
        if resolved_model.startswith("claude"):
            return self._get_claude_client(
                project=resolved_project,
                region=resolved_region,
                model=resolved_model,
            )
        else:
            return self._get_gemini_client(
                project=resolved_project,
                region=resolved_region,
                model=resolved_model,
                base_url=base_url,
            )

    def _get_claude_client(
        self,
        project: str | None = None,
        region: str | None = None,
        model: str | None = None,
    ) -> Any:
        """Create a Vertex AI client for Claude models."""
        try:
            from anthropic import AsyncAnthropicVertex
        except ImportError:
            raise ImportError(
                "anthropic package with vertex support required: "
                "pip install anthropic[bedrock]"
            )

        if not project:
            raise ValueError(
                "GCP project ID required for Vertex AI. "
                "Set GCLOUD_PROJECT environment variable or pass project parameter."
            )

        client_kwargs: dict[str, Any] = {
            "project_id": project,
            "region": region or "us-east5",
        }

        return AsyncAnthropicVertex(**client_kwargs)

    def _get_gemini_client(
        self,
        project: str | None = None,
        region: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ) -> Any:
        """Create a Vertex AI client for Gemini models.

        Uses the OpenAI-compatible endpoint for Gemini on Vertex.
        """
        from openai import AsyncOpenAI

        resolved_url = self.resolve_base_url(base_url)
        if not resolved_url and project and region:
            resolved_url = (
                f"https://{region}-aiplatform.googleapis.com/v1/projects/"
                f"{project}/locations/{region}/publishers/google/models"
            )

        # For Gemini on Vertex, use service account or ADC
        api_key = self._get_vertex_api_key()

        client_kwargs: dict[str, Any] = {"api_key": api_key or "adc-auth"}
        if resolved_url:
            client_kwargs["base_url"] = resolved_url

        return AsyncOpenAI(**client_kwargs)

    def _get_vertex_api_key(self) -> str | None:
        """Get API key for Vertex (uses ADC or API key auth)."""
        # Try API key first
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if api_key:
            return api_key

        # For ADC, return None (client handles it)
        return None

    def resolve_api_key(self, api_key: str | None = None) -> str | None:
        """Vertex uses ADC, but also supports API key auth."""
        if api_key:
            return api_key

        # Check for API key auth
        for env_var in self.config.auth.api_key_env_vars:
            value = os.environ.get(env_var)
            if value:
                return value

        # Check standard Google API key env vars
        for env_var in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
            value = os.environ.get(env_var)
            if value:
                return value

        return None
