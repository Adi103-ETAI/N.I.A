"""Anthropic provider with OAuth support and thinking models.

Ported from OpenClaude's anthropic vendor with support for:
- Native Anthropic API (x-api-key auth)
- OAuth authentication flow
- Extended thinking / reasoning models
- Full Claude model lineup
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
# Model definitions (ported from OpenClaude claude.ts)
# ---------------------------------------------------------------------------

ANTHROPIC_CAPABILITIES = ProviderCapabilities(
    supports_vision=True,
    supports_streaming=True,
    supports_function_calling=True,
    supports_json_mode=True,
    supports_reasoning=True,
    supports_precise_token_count=False,
    supports_usage=True,
    supports_thinking=True,
)

ANTHROPIC_MODELS = [
    ProviderModel(
        id="claude-sonnet-4-6",
        label="Claude Sonnet 4.6",
        context_window=200_000,
        max_output_tokens=8192,
        capabilities=ANTHROPIC_CAPABILITIES,
    ),
    ProviderModel(
        id="claude-opus-4-7",
        label="Claude Opus 4.7",
        context_window=200_000,
        max_output_tokens=8192,
        capabilities=ANTHROPIC_CAPABILITIES,
    ),
    ProviderModel(
        id="claude-opus-4-6",
        label="Claude Opus 4.6",
        context_window=200_000,
        max_output_tokens=8192,
        capabilities=ANTHROPIC_CAPABILITIES,
    ),
    ProviderModel(
        id="claude-haiku-4-5",
        label="Claude Haiku 4.5",
        context_window=144_000,
        max_output_tokens=8192,
        capabilities=ProviderCapabilities(
            supports_vision=True,
            supports_streaming=True,
            supports_function_calling=True,
            supports_json_mode=True,
            supports_reasoning=False,
            supports_precise_token_count=False,
            supports_usage=True,
            supports_thinking=False,
        ),
    ),
]


# ---------------------------------------------------------------------------
# OAuth token management
# ---------------------------------------------------------------------------

@dataclass
class OAuthTokens:
    """OAuth token pair."""

    access_token: str
    refresh_token: str | None = None
    expires_at: float | None = None


class OAuthTokenManager:
    """Manages OAuth tokens for Anthropic API access.

    Supports:
    - Token refresh via refresh_token grant
    - Token persistence to disk
    - Automatic expiry detection
    """

    def __init__(self, token_path: str | None = None) -> None:
        self._token_path = token_path or os.path.expanduser(
            "~/.config/niaharness/anthropic-oauth.json"
        )
        self._tokens: OAuthTokens | None = None

    def get_valid_token(self) -> str | None:
        """Get a valid access token, refreshing if necessary."""
        import time

        tokens = self._load_tokens()
        if tokens is None:
            return None

        # Check expiry
        if tokens.expires_at and tokens.expires_at < time.time():
            if tokens.refresh_token:
                tokens = self._refresh_token(tokens.refresh_token)
            else:
                return None

        return tokens.access_token if tokens else None

    def save_tokens(self, tokens: OAuthTokens) -> None:
        """Persist tokens to disk."""
        import json

        os.makedirs(os.path.dirname(self._token_path), exist_ok=True)
        data = {
            "access_token": tokens.access_token,
            "refresh_token": tokens.refresh_token,
            "expires_at": tokens.expires_at,
        }
        with open(self._token_path, "w") as f:
            json.dump(data, f)
        self._tokens = tokens

    def _load_tokens(self) -> OAuthTokens | None:
        """Load tokens from disk."""
        import json
        import time

        if self._tokens:
            return self._tokens

        if not os.path.exists(self._token_path):
            return None

        try:
            with open(self._token_path) as f:
                data = json.load(f)
            self._tokens = OAuthTokens(
                access_token=data["access_token"],
                refresh_token=data.get("refresh_token"),
                expires_at=data.get("expires_at"),
            )
            return self._tokens
        except (json.JSONDecodeError, KeyError):
            return None

    def _refresh_token(self, refresh_token: str) -> OAuthTokens | None:
        """Refresh the access token using a refresh token.

        In production, this would call the Anthropic OAuth token endpoint.
        Returns None if refresh fails.
        """
        # Placeholder: In production, POST to https://api.anthropic.com/oauth/token
        # with grant_type=refresh_token
        return None

    def clear(self) -> None:
        """Clear stored tokens."""
        self._tokens = None
        if os.path.exists(self._token_path):
            os.remove(self._token_path)


# ---------------------------------------------------------------------------
# Provider implementation
# ---------------------------------------------------------------------------

class AnthropicProvider(LLMProvider):
    """Anthropic API provider with OAuth support.

    Supports:
    - Native Anthropic Messages API
    - API key authentication
    - OAuth authentication (beta)
    - Extended thinking / reasoning models
    """

    @property
    def config(self) -> ProviderConfig:
        return ProviderConfig(
            name="anthropic",
            label="Anthropic",
            category=ProviderCategory.NATIVE,
            auth=ProviderAuthConfig(
                mode=AuthMode.API_KEY,
                api_key_env_vars=["ANTHROPIC_API_KEY"],
                base_url_env_vars=["ANTHROPIC_BASE_URL"],
                model_env_vars=["ANTHROPIC_MODEL"],
                default_base_url="https://api.anthropic.com",
                default_model="claude-sonnet-4-6",
            ),
            transport_kind="anthropic-native",
            models=ANTHROPIC_MODELS,
            is_first_party=True,
        )

    def get_client(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Create an Anthropic async client.

        Args:
            api_key: Anthropic API key. Falls back to ANTHROPIC_API_KEY env var.
            base_url: Override base URL.
            **kwargs: Additional options (e.g., use_oauth=True).

        Returns:
            AsyncAnthropic client instance.
        """
        from anthropic import AsyncAnthropic

        resolved_key = self.resolve_api_key(api_key)
        if not resolved_key:
            # Try OAuth if enabled
            if kwargs.get("use_oauth"):
                token_manager = OAuthTokenManager()
                resolved_key = token_manager.get_valid_token()
            if not resolved_key:
                raise ValueError(
                    "No API key found. Set ANTHROPIC_API_KEY environment variable "
                    "or provide api_key parameter."
                )

        resolved_url = self.resolve_base_url(base_url)

        client_kwargs: dict[str, Any] = {"api_key": resolved_key}
        if resolved_url:
            client_kwargs["base_url"] = resolved_url

        return AsyncAnthropic(**client_kwargs)

    def get_thinking_client(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        budget_tokens: int = 10_000,
        **kwargs: Any,
    ) -> tuple[Any, dict[str, Any]]:
        """Create a client configured for extended thinking.

        Returns:
            Tuple of (client, thinking_config) where thinking_config
            can be passed to message creation.
        """
        client = self.get_client(api_key, base_url=base_url, **kwargs)

        thinking_config = {
            "type": "enabled",
            "budget_tokens": budget_tokens,
        }

        return client, thinking_config
