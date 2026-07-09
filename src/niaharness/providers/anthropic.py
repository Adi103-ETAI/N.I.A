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

    P2 fix: ported from Hermes Agent's agent/anthropic_adapter.py — real
    PKCE OAuth flow for Claude Pro/Max subscriptions, real token refresh,
    token persistence to ~/.nia/anthropic-oauth.json.

    Supports:
    - PKCE OAuth login (browser flow + code paste)
    - Token refresh via refresh_token grant (real HTTP call)
    - Token persistence to disk
    - Automatic expiry detection
    - Cross-process token file sync (re-reads on each get_valid_token)
    """

    # OAuth constants (ported from Hermes — same client ID Claude Code uses)
    _OAUTH_CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
    _OAUTH_TOKEN_URLS = [
        "https://platform.claude.com/v1/oauth/token",
        "https://console.anthropic.com/v1/oauth/token",
    ]
    _OAUTH_TOKEN_USER_AGENT = "axios/1.7.9"  # Non-claude-code UA to avoid 429
    _OAUTH_REDIRECT_URI = "https://console.anthropic.com/oauth/code/callback"
    _OAUTH_SCOPES = "org:create_api_key user:profile user:inference"

    def __init__(self, token_path: str | None = None) -> None:
        # P2 fix: store under ~/.nia/ (NIA's home) for profile-scoped access.
        if token_path is None:
            try:
                from niaharness.prompts.soul import get_nia_home

                token_path = str(get_nia_home() / "anthropic-oauth.json")
            except Exception:
                token_path = os.path.expanduser("~/.nia/anthropic-oauth.json")
        self._token_path = token_path
        self._tokens: OAuthTokens | None = None

    def get_valid_token(self) -> str | None:
        """Get a valid access token, refreshing if necessary.

        Re-reads the token file on every call so another process (or the
        CLI's own login command) can refresh tokens and this process picks
        them up without a restart.
        """
        import time

        # P2 fix: always re-read from disk (another process may have refreshed).
        self._tokens = None
        tokens = self._load_tokens()
        if tokens is None:
            return None

        # Check expiry (5-minute skew to avoid using a token that's about to expire)
        if tokens.expires_at and tokens.expires_at < time.time() + 300:
            if tokens.refresh_token:
                refreshed = self._refresh_token(tokens.refresh_token)
                if refreshed is not None:
                    tokens = refreshed
                    self.save_tokens(tokens)
                else:
                    return None
            else:
                return None

        return tokens.access_token if tokens else None

    def login(self) -> OAuthTokens | None:
        """Run the PKCE OAuth login flow (browser + code paste).

        Ported from Hermes's run_hermes_oauth_login_pure(). Opens the
        Anthropic authorization URL in the browser, prompts for the
        authorization code, exchanges it for tokens.

        Returns OAuthTokens on success, None on failure.
        """
        import base64
        import hashlib
        import json
        import secrets
        import time
        import urllib.request
        from urllib.parse import urlencode

        # Generate PKCE verifier + challenge (S256).
        verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b"=").decode()
        challenge = base64.urlsafe_b64encode(
            hashlib.sha256(verifier.encode()).digest()
        ).rstrip(b"=").decode()
        oauth_state = secrets.token_urlsafe(32)

        # Build the authorization URL.
        params = {
            "code": "true",
            "client_id": self._OAUTH_CLIENT_ID,
            "response_type": "code",
            "redirect_uri": self._OAUTH_REDIRECT_URI,
            "scope": self._OAUTH_SCOPES,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "state": oauth_state,
        }
        auth_url = f"https://claude.ai/oauth/authorize?{urlencode(params)}"

        print()
        print("Authorize N.I.A with your Claude Pro/Max subscription.")
        print()
        print("Open this link in your browser:")
        print()
        print(f"  {auth_url}")
        print()

        # Try to open the browser automatically.
        try:
            import webbrowser

            webbrowser.open(auth_url)
            print("(Browser opened automatically)")
        except Exception:
            pass

        print()
        print("After authorizing, you'll see a code. Paste it below.")
        print()
        try:
            auth_code = input("Authorization code: ").strip()
        except (KeyboardInterrupt, EOFError):
            return None

        if not auth_code:
            print("No code entered.")
            return None

        # Parse code + state (Anthropic returns code#state).
        splits = auth_code.split("#")
        code = splits[0]
        received_state = splits[1] if len(splits) > 1 else ""

        # Validate state to prevent CSRF (RFC 6749 §10.12).
        if received_state != oauth_state:
            print("OAuth state mismatch — possible CSRF, aborting")
            return None

        # Exchange the authorization code for tokens.
        exchange_data = json.dumps({
            "grant_type": "authorization_code",
            "client_id": self._OAUTH_CLIENT_ID,
            "code": code,
            "state": received_state,
            "redirect_uri": self._OAUTH_REDIRECT_URI,
            "code_verifier": verifier,
        }).encode()

        result = None
        last_error = None
        for endpoint in self._OAUTH_TOKEN_URLS:
            req = urllib.request.Request(
                endpoint,
                data=exchange_data,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": self._OAUTH_TOKEN_USER_AGENT,
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=15) as resp:
                    result = json.loads(resp.read().decode())
                break
            except Exception as exc:
                last_error = exc
                continue

        if result is None:
            print(f"Token exchange failed: {last_error}")
            return None

        access_token = result.get("access_token", "")
        refresh_token = result.get("refresh_token", "")
        expires_in = result.get("expires_in", 3600)

        if not access_token:
            print("No access token in response.")
            return None

        tokens = OAuthTokens(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_at=time.time() + expires_in,
        )
        self.save_tokens(tokens)
        return tokens

    def save_tokens(self, tokens: OAuthTokens) -> None:
        """Persist tokens to disk."""
        import json

        os.makedirs(os.path.dirname(self._token_path) or ".", exist_ok=True)
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

        P2 fix: real HTTP call to the Anthropic OAuth token endpoint.
        Ported from Hermes's refresh_anthropic_oauth_pure().
        """
        import json
        import time
        import urllib.parse
        import urllib.request

        if not refresh_token:
            return None

        data = urllib.parse.urlencode({
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": self._OAUTH_CLIENT_ID,
        }).encode()

        last_error = None
        for endpoint in self._OAUTH_TOKEN_URLS:
            req = urllib.request.Request(
                endpoint,
                data=data,
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "User-Agent": self._OAUTH_TOKEN_USER_AGENT,
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=10) as resp:
                    result = json.loads(resp.read().decode())
            except Exception as exc:
                last_error = exc
                continue

            access_token = result.get("access_token", "")
            if not access_token:
                continue

            next_refresh = result.get("refresh_token", refresh_token)
            expires_in = result.get("expires_in", 3600)
            return OAuthTokens(
                access_token=access_token,
                refresh_token=next_refresh,
                expires_at=time.time() + expires_in,
            )

        # Refresh failed — could be revoked, expired, or network error.
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

        Delegates to :func:`niaharness.providers.anthropic_transport.build_anthropic_client`,
        which auto-detects auth mode (``api_key`` vs OAuth bearer vs third-party
        bearer) based on the key shape and base URL, and applies the correct
        ``anthropic-beta`` header set per endpoint.

        Args:
            api_key: Anthropic API key OR OAuth token. Falls back to
                ``ANTHROPIC_API_KEY`` env var, then to OAuth (when
                ``use_oauth=True``), then to ``resolve_anthropic_token()``.
            base_url: Override base URL (auto-resolved from env vars).
            **kwargs: Additional options. Recognized keys:
                - ``use_oauth``: when True and no api_key, resolve via
                  :func:`resolve_anthropic_token` (OAuth → credential pool → env).
                - ``timeout``: read timeout in seconds (default 900).
                - ``drop_context_1m_beta``: strip the 1M-context beta.

        Returns:
            ``anthropic.AsyncAnthropic`` client configured for the
            resolved endpoint and auth mode.
        """
        from niaharness.providers.anthropic_transport import (
            build_anthropic_client,
            resolve_anthropic_token,
        )

        use_oauth = bool(kwargs.pop("use_oauth", False))
        timeout = kwargs.pop("timeout", 900.0)
        drop_context_1m_beta = bool(kwargs.pop("drop_context_1m_beta", False))

        resolved_key = api_key or self.resolve_api_key()

        # If no explicit key, try OAuth / credential pool / env chain.
        if not resolved_key and use_oauth:
            resolved_key = resolve_anthropic_token()

        # If still no key but use_oauth is False, try OAuth manager directly
        # as a last resort (preserves the old behavior).
        if not resolved_key:
            token_manager = OAuthTokenManager()
            resolved_key = token_manager.get_valid_token()

        if not resolved_key:
            raise ValueError(
                "No API key found. Set ANTHROPIC_API_KEY environment variable, "
                "run `nia auth login` to set up OAuth, or provide api_key parameter."
            )

        resolved_url = self.resolve_base_url(base_url)

        return build_anthropic_client(
            api_key=resolved_key,
            base_url=resolved_url,
            timeout=timeout,
            drop_context_1m_beta=drop_context_1m_beta,
            **kwargs,
        )

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
