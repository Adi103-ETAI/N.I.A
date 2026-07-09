"""Anthropic transport layer — message conversion, repair, caching, thinking.

Ported from Hermes Agent's ``agent/anthropic_adapter.py`` (2,787 LOC).
This is the central wire-format adapter that sits between NIA's
:class:`ConversationMessage` model and the Anthropic Messages SDK.

Responsibilities:

  - **Message conversion** — NIA ``ConversationMessage`` → Anthropic wire-format
    dicts (text / thinking / tool_use / tool_result blocks).  Also accepts
    OpenAI-format message dicts for compatibility with code paths that still
    build them (e.g. the auxiliary chain).
  - **Message repair** — strip orphaned tool blocks whose counterpart was lost
    to compression/truncation, merge consecutive same-role messages to enforce
    Anthropic's strict alternation rule, evict old screenshots.
  - **Thinking-signature management** — strip / preserve / downgrade
    ``thinking`` and ``redacted_thinking`` blocks based on endpoint type
    (third-party / Kimi / DeepSeek / direct Anthropic) and turn-mutation flags.
  - **Prompt caching** — apply ``cache_control`` markers on the last cacheable
    block of each assistant turn + on the last tool definition, so Anthropic's
    automatic prompt caching kicks in (~10x cost reduction on long sessions).
  - **Model capability detection** — runtime feature gating based on model
    name: adaptive thinking, xhigh effort, sampling-param rejection, fast-mode.
  - **Client construction + token resolution** — pick the right auth mode
    (``api_key`` vs OAuth bearer vs third-party bearer), resolve the token
    from env / OAuth store / credential pool.

Public entry points:

  - :func:`build_anthropic_kwargs` — turn ``(model, messages, system, tools,
    max_tokens, reasoning_effort, base_url, ...)`` into the final kwargs dict
    consumed by ``client.messages.create(**kwargs)`` /
    ``client.messages.stream(**kwargs)``.
  - :func:`build_anthropic_client` — auto-detect OAuth / Bearer / x-api-key /
    third-party auth and return a configured ``AsyncAnthropic`` client.
  - :func:`resolve_anthropic_token` — token-resolution priority chain.
  - :func:`convert_messages_to_anthropic` — OpenAI-format → Anthropic message
    list conversion (the lower-level primitive used by
    :func:`build_anthropic_kwargs`).
  - :func:`convert_tools_to_anthropic` — OpenAI tool def → Anthropic tool def.
  - :func:`convert_conversation_messages_to_anthropic` — NIA-native
    ``ConversationMessage`` → Anthropic wire format (used by
    ``AnthropicApiClient._stream_once``).
"""

from __future__ import annotations

import copy
import json
import logging
import math
import os
import re
from typing import Any, Optional, Tuple
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level constants & globals
# ---------------------------------------------------------------------------

# Hermes effort → manual-thinking budget_tokens map (legacy path for
# Claude models that don't support the adaptive effort config).
THINKING_BUDGET: dict[str, int] = {
    "xhigh": 32_000,
    "high": 16_000,
    "medium": 8_000,
    "low": 4_000,
}

# Hermes effort → Anthropic adaptive-thinking ``output_config.effort``.
ADAPTIVE_EFFORT_MAP: dict[str, str] = {
    "max": "max",
    "xhigh": "xhigh",
    "high": "high",
    "medium": "medium",
    "low": "low",
    "minimal": "low",
}

# Older Claude families that still use manual budget-based thinking.
_LEGACY_MANUAL_THINKING_CLAUDE_SUBSTRINGS: tuple[str, ...] = (
    "claude-3",
    "claude-opus-4-0",
    "claude-opus-4-1",
    "claude-opus-4-5",
    "claude-sonnet-4-0",
    "claude-sonnet-4-1",
    "claude-sonnet-4-5",
    "claude-haiku-4-5",
    "claude-opus-4-2025",
    "claude-sonnet-4-2025",
    "claude-haiku-4-2025",
)

# Adaptive Claude models that don't accept ``xhigh`` effort.
_NO_XHIGH_CLAUDE_SUBSTRINGS: tuple[str, ...] = (
    "claude-opus-4-6",
    "claude-opus-4.6",
    "claude-sonnet-4-6",
    "claude-sonnet-4.6",
    "claude-haiku-4-6",
    "claude-haiku-4.6",
)

# Models that accept ``speed: "fast"``.
_FAST_MODE_SUPPORTED_SUBSTRINGS: tuple[str, ...] = (
    "opus-4-6",
    "opus-4.6",
)

# Per-model max output tokens (longest-prefix match wins).
_ANTHROPIC_OUTPUT_LIMITS: dict[str, int] = {
    "claude-opus-4-8": 128_000,
    "claude-opus-4-7": 128_000,
    "claude-opus-4-6": 128_000,
    "claude-opus-4-5": 16_384,
    "claude-opus-4-1": 32_000,
    "claude-opus-4-0": 4_096,
    "claude-sonnet-4-6": 64_000,
    "claude-sonnet-4-5": 16_384,
    "claude-sonnet-4-1": 64_000,
    "claude-sonnet-4-0": 64_000,
    "claude-3-7-sonnet": 64_000,
    "claude-3-5-sonnet": 8_192,
    "claude-3-5-haiku": 8_192,
    "claude-3-opus": 4_096,
    "claude-3-haiku": 4_096,
    "claude-haiku-4-5": 8_192,
}
_ANTHROPIC_DEFAULT_OUTPUT_LIMIT = 128_000

# Betas sent on every Anthropic request.
_COMMON_BETAS: list[str] = [
    "interleaved-thinking-2025-05-14",
    "fine-grained-tool-streaming-2025-05-14",
]
_TOOL_STREAMING_BETA = "fine-grained-tool-streaming-2025-05-14"
_CONTEXT_1M_BETA = "context-1m-2025-08-07"
_FAST_MODE_BETA = "fast-mode-2026-02-01"
_OAUTH_ONLY_BETAS: list[str] = [
    "claude-code-20250219",
    "oauth-2025-04-20",
]

# Claude Code system prefix prepended on OAuth path.
_CLAUDE_CODE_SYSTEM_PREFIX = "You are Claude Code, Anthropic's official CLI for Claude."

# Tool-name prefix for OAuth wire format.
_MCP_TOOL_PREFIX = "mcp__"

# Kimi / Moonshot family model-name prefixes.
_KIMI_FAMILY_MODEL_PREFIXES: tuple[str, ...] = (
    "kimi-", "kimi_",
    "moonshot-", "moonshot_",
    "k1.", "k1-",
    "k2.", "k2-",
    "k25", "k2.5",
)

# Anthropic OAuth client ID (same one Claude Code uses — required for the
# ``claude-code`` beta header on the messages endpoint).
_OAUTH_CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"

# Claude Code version fallback for the OAuth user-agent.
_CLAUDE_CODE_VERSION_FALLBACK = "2.1.74"
_claude_code_version_cache: Optional[str] = None

# Responses-API-only keys that must be stripped before a Messages SDK call.
_RESPONSES_ONLY_KWARGS = frozenset(
    {"instructions", "input", "store", "parallel_tool_calls"}
)

# Maximum number of computer-use screenshots to retain in a single turn
# history.  Each base64 image costs ~1,465 tokens and they accumulate
# quickly across tool calls.
_MAX_KEEP_SCREENSHOTS = 3


# ---------------------------------------------------------------------------
# Endpoint classifiers (ported from Hermes — adapted to NIA)
# ---------------------------------------------------------------------------

def _normalize_base_url_text(base_url: Any) -> str:
    """Coerce ``httpx.URL`` / ``None`` / str → plain stripped string."""
    if base_url is None:
        return ""
    try:
        return str(base_url).strip()
    except Exception:
        return ""


def base_url_host_matches(base_url: Any, host: str) -> bool:
    """Return True if *base_url*'s host equals (or ends with) *host*."""
    url = _normalize_base_url_text(base_url)
    if not url:
        return False
    try:
        parsed = urlparse(url if "://" in url else f"https://{url}")
        url_host = (parsed.hostname or "").lower()
    except Exception:
        return False
    if not url_host:
        return False
    host = host.lower()
    return url_host == host or url_host.endswith("." + host)


def _is_third_party_anthropic_endpoint(base_url: Any) -> bool:
    """True for any non-empty base URL that does NOT point at ``anthropic.com``.

    Third-party endpoints (MiniMax, Azure AI Foundry, AWS Bedrock, self-hosted
    proxies) cannot validate Anthropic-proprietary thinking signatures and
    will reject them with HTTP 400.
    """
    url = _normalize_base_url_text(base_url)
    if not url:
        return False  # No base URL = direct Anthropic.
    return "anthropic.com" not in url.lower()


def _is_kimi_coding_endpoint(base_url: Any) -> bool:
    url = _normalize_base_url_text(base_url).lower().rstrip("/")
    return url.startswith("https://api.kimi.com/coding")


def _model_name_is_kimi_family(model: Any) -> bool:
    if not isinstance(model, str) or not model:
        return False
    # Strip vendor prefix (everything before the last "/").
    bare = model.rsplit("/", 1)[-1].lower()
    return any(bare.startswith(p) for p in _KIMI_FAMILY_MODEL_PREFIXES)


def _is_kimi_family_endpoint(base_url: Any, model: Any = None) -> bool:
    """True for Kimi/Moonshot hosts OR Kimi-family model names."""
    if _is_kimi_coding_endpoint(base_url):
        return True
    if base_url_host_matches(base_url, "api.kimi.com"):
        return True
    if base_url_host_matches(base_url, "moonshot.ai"):
        return True
    if base_url_host_matches(base_url, "moonshot.cn"):
        return True
    if _model_name_is_kimi_family(model):
        return True
    return False


def _is_deepseek_anthropic_endpoint(base_url: Any) -> bool:
    """True iff host is ``api.deepseek.com`` AND path contains ``/anthropic``."""
    url = _normalize_base_url_text(base_url).lower()
    if not url:
        return False
    try:
        parsed = urlparse(url if "://" in url else f"https://{url}")
        host = (parsed.hostname or "").lower()
        path = (parsed.path or "").lower()
    except Exception:
        return False
    return host == "api.deepseek.com" and "/anthropic" in path


def _is_minimax_anthropic_endpoint(base_url: Any) -> bool:
    url = _normalize_base_url_text(base_url).lower()
    return (
        url.startswith("https://api.minimax.io/anthropic")
        or url.startswith("https://api.minimaxi.com/anthropic")
    )


def _is_azure_anthropic_endpoint(base_url: Any) -> bool:
    """True for Azure AI Foundry / OpenAI-Azure endpoints with ``/anthropic`` path."""
    url = _normalize_base_url_text(base_url).lower()
    if not url:
        return False
    try:
        parsed = urlparse(url if "://" in url else f"https://{url}")
        host = (parsed.hostname or "").lower()
        path = (parsed.path or "").lower()
    except Exception:
        return False
    return (
        (".services.ai.azure." in f".{host}." or ".openai.azure." in f".{host}.")
        and "/anthropic" in path
    )


def _requires_bearer_auth(base_url: Any) -> bool:
    """True for endpoints that take ``Authorization: Bearer`` instead of ``x-api-key``."""
    return _is_minimax_anthropic_endpoint(base_url) or "azure.com" in _normalize_base_url_text(base_url).lower()


def _base_url_needs_context_1m_beta(base_url: Any) -> bool:
    return "azure.com" in _normalize_base_url_text(base_url).lower()


def _common_betas_for_base_url(
    base_url: Any,
    *,
    drop_context_1m_beta: bool = False,
) -> list[str]:
    """Build the ``anthropic-beta`` header list per endpoint."""
    betas = list(_COMMON_BETAS)
    if _base_url_needs_context_1m_beta(base_url) and not drop_context_1m_beta:
        betas.append(_CONTEXT_1M_BETA)
    if _is_minimax_anthropic_endpoint(base_url):
        betas = [b for b in betas if b not in {_TOOL_STREAMING_BETA, _CONTEXT_1M_BETA}]
    if drop_context_1m_beta:
        betas = [b for b in betas if b != _CONTEXT_1M_BETA]
    return betas


# ---------------------------------------------------------------------------
# Token / OAuth helpers
# ---------------------------------------------------------------------------

def _is_oauth_token(key: Any) -> bool:
    """Identify Anthropic OAuth / setup tokens.

    Claude Code OAuth tokens start with ``sk-ant-`` (non-api), JWTs start with
    ``eyJ``, and Claude Code access tokens start with ``cc-``.  Regular API
    keys start with ``sk-ant-api`` and are NOT OAuth tokens.
    """
    if not isinstance(key, str) or not key:
        return False
    if key.startswith("sk-ant-api"):
        return False
    if key.startswith("sk-ant-"):
        return True
    if key.startswith("eyJ"):
        return True
    if key.startswith("cc-"):
        return True
    return False


def _detect_claude_code_version() -> str:
    """Run ``claude --version`` (or ``claude-code --version``) and cache it."""
    global _claude_code_version_cache
    if _claude_code_version_cache is not None:
        return _claude_code_version_cache

    import subprocess

    for cmd in (["claude", "--version"], ["claude-code", "--version"]):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            for token in (result.stdout or "").split():
                if token and token[0].isdigit():
                    _claude_code_version_cache = token
                    return token
        except Exception:
            continue

    _claude_code_version_cache = _CLAUDE_CODE_VERSION_FALLBACK
    return _claude_code_version_cache


def resolve_anthropic_token() -> Optional[str]:
    """Token-resolution priority chain for the Anthropic Messages API.

    Order:

      1. ``ANTHROPIC_TOKEN`` env var (highest priority — explicit override).
      2. ``CLAUDE_CODE_OAUTH_TOKEN`` env var (Claude Code export).
      3. NIA's :class:`OAuthTokenManager` (reads ``~/.nia/anthropic-oauth.json``,
         refreshes on expiry).
      4. NIA's :class:`CredentialPool` ``anthropic`` entries (rotated creds).
      5. ``ANTHROPIC_API_KEY`` env var (legacy API-key fallback).

    Returns the first non-empty token, or ``None`` if no auth source is
    available.
    """
    # 1. ANTHROPIC_TOKEN (explicit override).
    explicit = os.environ.get("ANTHROPIC_TOKEN", "").strip()
    if explicit:
        return explicit

    # 2. CLAUDE_CODE_OAUTH_TOKEN.
    cc_oauth = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", "").strip()
    if cc_oauth:
        return cc_oauth

    # 3. NIA's OAuthTokenManager.
    try:
        from niaharness.providers.anthropic import OAuthTokenManager

        token = OAuthTokenManager().get_valid_token()
        if token:
            return token
    except Exception as exc:  # pragma: no cover — defensive
        logger.debug("resolve_anthropic_token: OAuthTokenManager failed: %s", exc)

    # 4. NIA's CredentialPool.
    try:
        from niaharness.api.credential_pool import load_pool

        pool = load_pool("anthropic")
        if pool is not None:
            for entry in pool.available_entries():
                token = entry.api_key or entry.access_token
                if token:
                    return token
    except Exception as exc:  # pragma: no cover — defensive
        logger.debug("resolve_anthropic_token: credential pool failed: %s", exc)

    # 5. ANTHROPIC_API_KEY (legacy fallback).
    legacy = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if legacy:
        return legacy

    return None


def build_anthropic_client(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    *,
    timeout: float = 900.0,
    drop_context_1m_beta: bool = False,
    use_oauth: bool = False,
    **kwargs: Any,
) -> Any:
    """Build a configured ``AsyncAnthropic`` client.

    Auto-detects auth mode based on ``api_key`` shape and ``base_url``:

      - **Kimi ``/coding`` endpoint** → x-api-key + ``claude-code/0.1.0`` UA +
        tool-streaming + 1M-context betas.
      - **Bearer-auth endpoints** (MiniMax / Azure) → ``auth_token=api_key`` +
        endpoint betas.
      - **Third-party Anthropic-compatible** (Foundry / Bedrock / proxy) →
        x-api-key + betas, OAuth detection skipped.
      - **OAuth token** (``sk-ant-`` non-api / ``eyJ`` JWT / ``cc-``) →
        ``auth_token=api_key`` + ``claude-code/<version>`` UA + OAuth betas
        + ``x-app: cli``.
      - **Regular API key** (``sk-ant-api``) → ``api_key=api_key`` + betas.

    Args:
        api_key: API key or OAuth token. If ``None``, falls back to
            :func:`resolve_anthropic_token` (only when ``use_oauth=True``).
        base_url: Override base URL.
        timeout: Read timeout in seconds (default 900s = 15 min for long
            thinking turns).
        drop_context_1m_beta: Strip the 1M-context beta (used for endpoints
            that don't support it).
        use_oauth: When ``api_key`` is None and ``use_oauth`` is True, resolve
            a token via :func:`resolve_anthropic_token`.
        **kwargs: Additional SDK kwargs passed through verbatim.

    Returns:
        Configured ``anthropic.AsyncAnthropic`` client.
    """
    from anthropic import AsyncAnthropic
    try:
        from httpx import Timeout
    except ImportError:  # pragma: no cover — httpx is an SDK dep
        Timeout = None  # type: ignore[assignment]

    resolved_url = _normalize_base_url_text(base_url).rstrip("/")
    # Strip trailing /v1 — the SDK adds its own /v1 path.
    resolved_url = re.sub(r"/v1/?$", "", resolved_url)

    if api_key is None and use_oauth:
        api_key = resolve_anthropic_token()

    if not api_key:
        raise ValueError(
            "No Anthropic API key or OAuth token available. Set the "
            "ANTHROPIC_API_KEY environment variable, run `nia auth login` "
            "to set up OAuth, or pass api_key explicitly."
        )

    client_kwargs: dict[str, Any] = {"max_retries": 0}
    if Timeout is not None:
        client_kwargs["timeout"] = Timeout(timeout, connect=10.0)
    if resolved_url:
        client_kwargs["base_url"] = resolved_url
    client_kwargs.update(kwargs)

    common_betas = _common_betas_for_base_url(
        resolved_url, drop_context_1m_beta=drop_context_1m_beta
    )

    # Branch 1: Kimi /coding endpoint.
    if _is_kimi_coding_endpoint(resolved_url):
        client_kwargs["api_key"] = api_key
        client_kwargs["default_headers"] = {
            "User-Agent": "claude-code/0.1.0",
            "anthropic-beta": ",".join(common_betas),
        }
        return AsyncAnthropic(**client_kwargs)

    # Branch 2: MiniMax / Azure bearer-auth endpoints.
    if _requires_bearer_auth(resolved_url):
        client_kwargs["auth_token"] = api_key
        client_kwargs["default_headers"] = {
            "anthropic-beta": ",".join(common_betas),
        }
        if _is_azure_anthropic_endpoint(resolved_url):
            client_kwargs["default_query"] = {"api-version": "2025-04-15"}
        return AsyncAnthropic(**client_kwargs)

    # Branch 3: Third-party Anthropic-compatible endpoints.
    if _is_third_party_anthropic_endpoint(resolved_url):
        client_kwargs["api_key"] = api_key
        client_kwargs["default_headers"] = {
            "anthropic-beta": ",".join(common_betas),
        }
        return AsyncAnthropic(**client_kwargs)

    # Branch 4: OAuth token (Claude Code / Hermes / NIA-managed).
    if _is_oauth_token(api_key):
        oauth_betas = list(common_betas) + list(_OAUTH_ONLY_BETAS)
        client_kwargs["auth_token"] = api_key
        client_kwargs["default_headers"] = {
            "User-Agent": f"claude-code/{_detect_claude_code_version()} (external, cli)",
            "x-app": "cli",
            "anthropic-beta": ",".join(oauth_betas),
        }
        return AsyncAnthropic(**client_kwargs)

    # Branch 5: Regular API key.
    client_kwargs["api_key"] = api_key
    if common_betas:
        client_kwargs["default_headers"] = {
            "anthropic-beta": ",".join(common_betas),
        }
    return AsyncAnthropic(**client_kwargs)


# ---------------------------------------------------------------------------
# Model-name normalization + capability detection
# ---------------------------------------------------------------------------

def _is_bedrock_model_id(model: str) -> bool:
    """Detect AWS Bedrock model IDs (dots are namespace separators)."""
    lower = (model or "").lower()
    if any(lower.startswith(p) for p in ("global.", "us.", "eu.", "ap.", "jp.")):
        return True
    if lower.startswith("anthropic."):
        return True
    return False


def normalize_model_name(model: str, preserve_dots: bool = False) -> str:
    """Normalize a model name for the Anthropic API.

    - Strips ``anthropic/`` prefix (OpenRouter format).
    - Converts dots to hyphens in version numbers (OpenRouter uses dots,
      Anthropic uses hyphens: ``claude-opus-4.6`` → ``claude-opus-4-6``),
      unless ``preserve_dots=True`` (for Alibaba/DashScope) or the ID is a
      Bedrock regional inference profile.
    """
    if not model:
        return model
    lower = model.lower()
    if lower.startswith("anthropic/"):
        model = model[len("anthropic/"):]
    if not preserve_dots:
        if _is_bedrock_model_id(model):
            return model
        _lower = model.lower()
        if _lower.startswith("claude-") or _lower.startswith("anthropic/"):
            model = model.replace(".", "-")
    return model


def _is_claude_model(model: Optional[str]) -> bool:
    return "claude" in (model or "").lower()


def _get_anthropic_max_output(model: str) -> int:
    """Per-model max output tokens (longest-prefix match wins)."""
    normalized = (model or "").lower().replace(".", "-")
    best_match: Optional[str] = None
    for key in _ANTHROPIC_OUTPUT_LIMITS:
        if key in normalized:
            if best_match is None or len(key) > len(best_match):
                best_match = key
    if best_match is not None:
        return _ANTHROPIC_OUTPUT_LIMITS[best_match]
    return _ANTHROPIC_DEFAULT_OUTPUT_LIMIT


def _resolve_positive_max_tokens(value: Any) -> Optional[int]:
    """Floor ``value`` to a positive int (reject bool / NaN / non-finite / ≤0)."""
    if isinstance(value, bool):
        return None
    if not isinstance(value, (int, float)):
        return None
    try:
        if not math.isfinite(float(value)):
            return None
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    return int(value)


def _resolve_anthropic_messages_max_tokens(
    requested: Any,
    model: str,
    context_length: Optional[int] = None,
) -> int:
    """Resolve final ``max_tokens``: prefer ``requested`` if positive finite,
    else fall back to the model's output ceiling, else raise.
    """
    pos = _resolve_positive_max_tokens(requested)
    if pos is not None:
        return pos
    fallback = _get_anthropic_max_output(model)
    if fallback > 0:
        return fallback
    raise ValueError(
        f"Anthropic Messages adapter requires a positive max_tokens value "
        f"for model {model!r}; got {requested!r} and no model default resolved."
    )


def _supports_adaptive_thinking(model: str) -> bool:
    """True for Claude models that accept the adaptive effort config.

    Older Claude families (4-0, 4-1, 4-5, 3-x) still use manual budget-based
    thinking; everything else (4-6, 4-7, 4-8, future) defaults to adaptive.
    """
    if not _is_claude_model(model):
        return False
    return not any(
        v in model.lower() for v in _LEGACY_MANUAL_THINKING_CLAUDE_SUBSTRINGS
    )


def _supports_xhigh_effort(model: str) -> bool:
    """True for adaptive Claude models that accept ``xhigh`` effort."""
    if not _supports_adaptive_thinking(model):
        return False
    return not any(v in model.lower() for v in _NO_XHIGH_CLAUDE_SUBSTRINGS)


def _forbids_sampling_params(model: str) -> bool:
    """True for Claude models that 400 on non-default temperature/top_p/top_k.

    Claude 4.6 and legacy models accept sampling params; 4.7+ and unknown
    Claude models reject them.  Defaults unknown Claude to ``True`` (the
    modern contract).
    """
    if not _is_claude_model(model):
        return False
    if any(v in model.lower() for v in _NO_XHIGH_CLAUDE_SUBSTRINGS):
        return False
    return not any(
        v in model.lower() for v in _LEGACY_MANUAL_THINKING_CLAUDE_SUBSTRINGS
    )


def _supports_fast_mode(model: str) -> bool:
    return any(v in model for v in _FAST_MODE_SUPPORTED_SUBSTRINGS)


# ---------------------------------------------------------------------------
# Tool-schema sanitization
# ---------------------------------------------------------------------------

def _strip_nullable_unions(schema: Any, *, keep_nullable_hint: bool = False) -> Any:
    """Collapse ``anyOf: [T, null]`` → T, optionally preserving ``nullable: true``.

    Anthropic's tool-schema validator rejects nullable unions that Pydantic/MCP
    commonly emit for optional fields.  Tool optionality is represented by the
    parent ``required`` array, so we collapse to the non-null branch.

    Ported from Hermes's ``tools/schema_sanitizer.strip_nullable_unions``.
    """
    if not isinstance(schema, dict):
        return schema

    if "anyOf" in schema and isinstance(schema["anyOf"], list):
        non_null = [
            s for s in schema["anyOf"]
            if not (isinstance(s, dict) and s.get("type") == "null")
        ]
        if non_null and len(non_null) < len(schema["anyOf"]):
            # Collapse to the single non-null branch (or merge if multiple).
            if len(non_null) == 1:
                collapsed = dict(non_null[0]) if isinstance(non_null[0], dict) else {}
            else:
                collapsed = {"anyOf": non_null}
            for k, v in schema.items():
                if k == "anyOf":
                    continue
                collapsed[k] = v
            if not keep_nullable_hint:
                collapsed.pop("nullable", None)
            schema = collapsed

    if "oneOf" in schema and isinstance(schema["oneOf"], list):
        non_null = [
            s for s in schema["oneOf"]
            if not (isinstance(s, dict) and s.get("type") == "null")
        ]
        if non_null and len(non_null) < len(schema["oneOf"]):
            if len(non_null) == 1:
                collapsed = dict(non_null[0]) if isinstance(non_null[0], dict) else {}
            else:
                collapsed = {"oneOf": non_null}
            for k, v in schema.items():
                if k == "oneOf":
                    continue
                collapsed[k] = v
            if not keep_nullable_hint:
                collapsed.pop("nullable", None)
            schema = collapsed

    # Recurse into properties.
    if isinstance(schema.get("properties"), dict):
        schema = dict(schema)
        schema["properties"] = {
            k: _strip_nullable_unions(v, keep_nullable_hint=keep_nullable_hint)
            for k, v in schema["properties"].items()
        }

    # Recurse into items.
    if isinstance(schema.get("items"), dict):
        schema = dict(schema)
        schema["items"] = _strip_nullable_unions(
            schema["items"], keep_nullable_hint=keep_nullable_hint
        )

    return schema


def _normalize_tool_input_schema(schema: Any) -> dict[str, Any]:
    """Normalize a tool input schema before sending it to Anthropic.

    - Collapse nullable unions (``anyOf: [T, null]`` → T).
    - Strip top-level ``oneOf`` / ``allOf`` / ``anyOf`` (Anthropic rejects them).
    - Ensure ``type: object`` schemas have a dict ``properties`` field.
    """
    if not schema:
        return {"type": "object", "properties": {}}

    normalized = _strip_nullable_unions(schema, keep_nullable_hint=False)
    if not isinstance(normalized, dict):
        return {"type": "object", "properties": {}}

    banned = {"oneOf", "allOf", "anyOf"}
    if banned & normalized.keys():
        normalized = {k: v for k, v in normalized.items() if k not in banned}
        if "type" not in normalized:
            normalized["type"] = "object"

    if normalized.get("type") == "object" and not isinstance(
        normalized.get("properties"), dict
    ):
        normalized = {**normalized, "properties": {}}
    return normalized


# ---------------------------------------------------------------------------
# Tool-ID sanitization
# ---------------------------------------------------------------------------

_TOOL_ID_RE = re.compile(r"[^a-zA-Z0-9_-]")


def _sanitize_tool_id(tool_id: str) -> str:
    """Sanitize a tool call ID for the Anthropic API.

    Anthropic requires IDs matching ``[a-zA-Z0-9_-]``.  Replace invalid
    characters with underscores and ensure non-empty.
    """
    if not tool_id:
        return "tool_0"
    sanitized = _TOOL_ID_RE.sub("_", str(tool_id))
    return sanitized or "tool_0"


# ---------------------------------------------------------------------------
# Tool conversion (OpenAI → Anthropic)
# ---------------------------------------------------------------------------

def convert_tools_to_anthropic(tools: Optional[list[dict]]) -> list[dict]:
    """Convert OpenAI tool definitions to Anthropic format.

    Accepts both OpenAI-format tools (``{"function": {"name", "description",
    "parameters"}}``) and already-Anthropic-format tools (``{"name",
    "description", "input_schema"}``).

    Deduplicates by name (Anthropic rejects duplicate tool names with 400).
    Forwards ``cache_control`` markers when present.
    """
    if not tools:
        return []
    result: list[dict] = []
    seen_names: set[str] = set()
    for t in tools:
        if not isinstance(t, dict):
            continue
        # Auto-detect format.
        if "function" in t:
            fn = t.get("function", {}) or {}
            name = fn.get("name", "")
            description = fn.get("description", "")
            input_schema = fn.get("parameters", {"type": "object", "properties": {}})
        else:
            name = t.get("name", "")
            description = t.get("description", "")
            input_schema = t.get("input_schema", t.get("parameters", {"type": "object", "properties": {}}))

        if not name:
            continue
        if name in seen_names:
            logger.warning(
                "convert_tools_to_anthropic: duplicate tool name %r — "
                "dropping second occurrence", name,
            )
            continue
        seen_names.add(name)

        anthropic_tool: dict[str, Any] = {
            "name": name,
            "description": description,
            "input_schema": _normalize_tool_input_schema(input_schema),
        }
        cache_control = t.get("cache_control")
        if isinstance(cache_control, dict):
            anthropic_tool["cache_control"] = dict(cache_control)
        result.append(anthropic_tool)
    return result


# ---------------------------------------------------------------------------
# Image-source conversion
# ---------------------------------------------------------------------------

def _image_source_from_openai_url(url: str) -> dict[str, str]:
    """Convert an OpenAI image URL / data URI into an Anthropic image source."""
    url = str(url or "").strip()
    if not url:
        return {"type": "url", "url": ""}

    if url.startswith("data:"):
        header, _, data = url.partition(",")
        media_type = "image/jpeg"
        if header.startswith("data:"):
            mime_part = header[len("data:"):].split(";", 1)[0].strip()
            if mime_part.startswith("image/"):
                media_type = mime_part
        return {
            "type": "base64",
            "media_type": media_type,
            "data": data,
        }
    return {"type": "url", "url": url}


def _convert_content_part_to_anthropic(part: Any) -> Optional[dict[str, Any]]:
    """Convert a single OpenAI-style content part to Anthropic format."""
    if part is None:
        return None
    if isinstance(part, str):
        return {"type": "text", "text": part}
    if not isinstance(part, dict):
        return {"type": "text", "text": str(part)}

    ptype = part.get("type")
    if ptype == "input_text":
        block: dict[str, Any] = {"type": "text", "text": part.get("text", "")}
    elif ptype == "text":
        # Stored Anthropic text block — rebuild from whitelisted fields only.
        # SDK response text blocks carry output-only siblings (parsed_output,
        # citations=None) that the Messages INPUT schema rejects with HTTP 400
        # "Extra inputs are not permitted".
        block = {"type": "text", "text": part.get("text", "")}
        cits = part.get("citations")
        if isinstance(cits, list) and cits:
            block["citations"] = cits
    elif ptype in {"image_url", "input_image"}:
        image_value = part.get("image_url", {})
        url = (
            image_value.get("url", "")
            if isinstance(image_value, dict)
            else str(image_value or "")
        )
        block = {"type": "image", "source": _image_source_from_openai_url(url)}
    else:
        block = dict(part)

    if isinstance(part.get("cache_control"), dict) and "cache_control" not in block:
        block["cache_control"] = dict(part["cache_control"])
    return block


def _convert_content_to_anthropic(content: Any) -> Any:
    """Convert an OpenAI multimodal content array → list of Anthropic blocks."""
    if not isinstance(content, list):
        return content
    converted = []
    for part in content:
        block = _convert_content_part_to_anthropic(part)
        if block is not None:
            converted.append(block)
    return converted


def _content_parts_to_anthropic_blocks(parts: Any) -> list[dict[str, Any]]:
    """Convert OpenAI tool-message content parts → Anthropic tool_result inner blocks."""
    if not isinstance(parts, list):
        return []
    out: list[dict[str, Any]] = []
    for part in parts:
        block = _convert_content_part_to_anthropic(part)
        if not block:
            continue
        btype = block.get("type")
        if btype == "text":
            text_val = block.get("text")
            if isinstance(text_val, str) and text_val:
                out.append({"type": "text", "text": text_val})
        elif btype == "image":
            src = block.get("source")
            if isinstance(src, dict) and src:
                out.append({"type": "image", "source": src})
    return out


# ---------------------------------------------------------------------------
# SDK-object → plain-data conversion (cycle-safe)
# ---------------------------------------------------------------------------

def _to_plain_data(
    value: Any,
    *,
    _depth: int = 0,
    _path: Optional[set] = None,
) -> Any:
    """Recursively convert SDK objects to plain Python data structures.

    Guards against circular references via ``id()`` path tracking and runaway
    depth (capped at 20 levels).  Uses path-based tracking so shared (but
    non-cyclic) objects referenced by multiple siblings convert correctly.
    """
    _MAX_DEPTH = 20
    if _depth > _MAX_DEPTH:
        return str(value)
    if _path is None:
        _path = set()

    obj_id = id(value)
    if obj_id in _path:
        return str(value)

    if hasattr(value, "model_dump"):
        _path.add(obj_id)
        result = _to_plain_data(value.model_dump(), _depth=_depth + 1, _path=_path)
        _path.discard(obj_id)
        return result
    if isinstance(value, dict):
        _path.add(obj_id)
        result = {
            k: _to_plain_data(v, _depth=_depth + 1, _path=_path)
            for k, v in value.items()
        }
        _path.discard(obj_id)
        return result
    if isinstance(value, (list, tuple)):
        _path.add(obj_id)
        result = [_to_plain_data(v, _depth=_depth + 1, _path=_path) for v in value]
        _path.discard(obj_id)
        return result
    if hasattr(value, "__dict__"):
        _path.add(obj_id)
        result = {
            k: _to_plain_data(v, _depth=_depth + 1, _path=_path)
            for k, v in vars(value).items()
            if not k.startswith("_")
        }
        _path.discard(obj_id)
        return result
    return value


def _extract_preserved_thinking_blocks(message: dict[str, Any]) -> list[dict[str, Any]]:
    """Return Anthropic thinking blocks previously preserved on the message."""
    raw_details = message.get("reasoning_details")
    if not isinstance(raw_details, list):
        return []
    preserved: list[dict[str, Any]] = []
    for detail in raw_details:
        if not isinstance(detail, dict):
            continue
        block_type = str(detail.get("type", "") or "").strip().lower()
        if block_type not in {"thinking", "redacted_thinking"}:
            continue
        preserved.append(copy.deepcopy(detail))
    return preserved


# ---------------------------------------------------------------------------
# Prompt caching
# ---------------------------------------------------------------------------

def _apply_assistant_cache_control_to_last_cacheable_block(
    blocks: list[dict[str, Any]],
    cache_control: Any,
) -> None:
    """Apply ``cache_control`` to the last ``text`` or ``tool_use`` block in *blocks*.

    Walks in reverse so the *last* cacheable block (closest to the next user
    message) gets the marker — Anthropic's automatic prompt caching uses the
    cache_control marker as a content-prefix breakpoint.
    """
    if not isinstance(cache_control, dict):
        return
    for block in reversed(blocks):
        if isinstance(block, dict) and block.get("type") in {"text", "tool_use"}:
            block.setdefault("cache_control", dict(cache_control))
            break


def apply_cache_control_to_last_tool(tools: list[dict[str, Any]]) -> None:
    """Mark the last tool definition with ``cache_control: {"type": "ephemeral"}``.

    Anthropic caches the entire tool-schema array cross-session when the last
    tool has a cache_control marker — saves ~1k tokens of input per turn on
    tool-heavy sessions.
    """
    if not tools:
        return
    last = tools[-1]
    if isinstance(last, dict):
        last.setdefault("cache_control", {"type": "ephemeral"})


def apply_cache_control_to_system(system: Any) -> Any:
    """Mark the last system block with ``cache_control: {"type": "ephemeral"}``.

    Accepts a string (returns a list of one cacheable block) or a list of
    blocks (mutates the last in place).
    """
    if not system:
        return system
    if isinstance(system, str):
        return [
            {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}},
        ]
    if isinstance(system, list) and system:
        last = system[-1]
        if isinstance(last, dict):
            last.setdefault("cache_control", {"type": "ephemeral"})
    return system


# ---------------------------------------------------------------------------
# Block-replay sanitization
# ---------------------------------------------------------------------------

def _sanitize_replay_block(b: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Whitelist-strip output-only fields from a stored Anthropic block so it
    is valid as REQUEST input on replay.

    The SDK response objects carry output-only attributes that the Messages
    *input* schema forbids ("Extra inputs are not permitted"): text blocks get
    ``parsed_output`` / ``citations=None``, tool_use blocks get ``caller``,
    etc.  This is a whitelist per type (NOT a blacklist) so future SDK
    output-only fields can't reintroduce the bug.
    """
    if not isinstance(b, dict):
        return None
    btype = b.get("type")
    if btype == "text":
        out: dict[str, Any] = {"type": "text", "text": b.get("text", "")}
        cits = b.get("citations")
        if isinstance(cits, list) and cits:
            out["citations"] = cits
        if isinstance(b.get("cache_control"), dict):
            out["cache_control"] = b["cache_control"]
        return out
    if btype == "thinking":
        out = {"type": "thinking", "thinking": b.get("thinking", "")}
        if b.get("signature"):
            out["signature"] = b["signature"]
        return out
    if btype == "redacted_thinking":
        return (
            {"type": "redacted_thinking", "data": b["data"]}
            if b.get("data")
            else None
        )
    if btype == "tool_use":
        out = {
            "type": "tool_use",
            "id": _sanitize_tool_id(b.get("id", "")),
            "name": b.get("name", ""),
            "input": b.get("input", {}),
        }
        if isinstance(b.get("cache_control"), dict):
            out["cache_control"] = b["cache_control"]
        return out
    if btype == "image":
        src = b.get("source")
        return {"type": "image", "source": src} if isinstance(src, dict) else None
    return None


# ---------------------------------------------------------------------------
# Per-role message converters (OpenAI-format input → Anthropic output)
# ---------------------------------------------------------------------------

def _convert_assistant_message(m: dict[str, Any]) -> dict[str, Any]:
    """Convert an assistant message to Anthropic content blocks.

    Handles thinking blocks, regular content, tool calls, and
    reasoning_content injection for Kimi/DeepSeek endpoints.
    """
    content = m.get("content", "")

    # Anthropic interleaved-thinking fast path: replay the verbatim
    # block list (set by normalize_response when a turn interleaves
    # SIGNED thinking with tool_use).  Each block is run through
    # _sanitize_replay_block to strip output-only SDK fields that the
    # Messages INPUT schema forbids.
    ordered_blocks = m.get("anthropic_content_blocks")
    if isinstance(ordered_blocks, list) and ordered_blocks:
        # Re-source each tool_use input from the stored tool_calls map
        # rather than the captured block.  The ordered-blocks list captures
        # tool_use input from the RAW API response (which is NOT
        # credential-redacted); tool_calls[].function.arguments IS redacted
        # at storage time.  Replaying the raw block input would resurrect
        # a secret the model inlined into a tool call.
        redacted_input_by_id: dict[str, Any] = {}
        for tc in m.get("tool_calls", []) or []:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function", {}) or {}
            raw_args = fn.get("arguments", "{}")
            try:
                parsed_args = (
                    json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                )
            except (json.JSONDecodeError, ValueError):
                parsed_args = {}
            redacted_input_by_id[_sanitize_tool_id(tc.get("id", ""))] = parsed_args
        replayed: list[dict[str, Any]] = []
        for b in ordered_blocks:
            clean = _sanitize_replay_block(b)
            if clean is None:
                continue
            if clean.get("type") == "tool_use":
                redacted = redacted_input_by_id.get(clean.get("id", ""))
                if redacted is not None:
                    clean["input"] = redacted
            replayed.append(clean)
        if replayed:
            _apply_assistant_cache_control_to_last_cacheable_block(
                replayed, m.get("cache_control")
            )
            return {"role": "assistant", "content": replayed}

    blocks = _extract_preserved_thinking_blocks(m)
    if content:
        if isinstance(content, list):
            converted_content = _convert_content_to_anthropic(content)
            if isinstance(converted_content, list):
                blocks.extend(converted_content)
        else:
            blocks.append({"type": "text", "text": str(content)})

    for tc in m.get("tool_calls", []):
        if not tc or not isinstance(tc, dict):
            continue
        fn = tc.get("function", {})
        args = fn.get("arguments", "{}")
        try:
            parsed_args = json.loads(args) if isinstance(args, str) else args
        except (json.JSONDecodeError, ValueError):
            parsed_args = {}
        blocks.append({
            "type": "tool_use",
            "id": _sanitize_tool_id(tc.get("id", "")),
            "name": fn.get("name", ""),
            "input": parsed_args,
        })

    _apply_assistant_cache_control_to_last_cacheable_block(blocks, m.get("cache_control"))

    # Kimi's /coding endpoint requires assistant tool-call messages to carry
    # reasoning_content when thinking is enabled server-side.  Preserve it
    # as a thinking block so Kimi can validate the message history.
    reasoning_content = m.get("reasoning_content")
    _already_has_thinking = any(
        isinstance(b, dict) and b.get("type") in {"thinking", "redacted_thinking"}
        for b in blocks
    )
    if isinstance(reasoning_content, str) and not _already_has_thinking:
        blocks.insert(0, {"type": "thinking", "thinking": reasoning_content})

    effective = blocks or content
    if not effective or effective == "":
        effective = [{"type": "text", "text": "(empty)"}]
    return {"role": "assistant", "content": effective}


def _convert_tool_message_to_result(
    result: list[dict[str, Any]],
    m: dict[str, Any],
) -> None:
    """Convert an OpenAI tool message → Anthropic tool_result block.

    Mutates *result* in place — either appends a new user message or extends
    the trailing user message's tool_result list (merging consecutive tool
    results into one user message, which Anthropic requires).
    """
    content = m.get("content", "")
    multimodal_blocks: Optional[list[dict[str, Any]]] = None
    if isinstance(content, dict) and content.get("_multimodal"):
        multimodal_blocks = _content_parts_to_anthropic_blocks(
            content.get("content") or []
        )
        if not multimodal_blocks and content.get("text_summary"):
            multimodal_blocks = [
                {"type": "text", "text": str(content["text_summary"])}
            ]
    elif isinstance(content, list):
        converted = _content_parts_to_anthropic_blocks(content)
        if any(b.get("type") == "image" for b in converted):
            multimodal_blocks = converted

    if multimodal_blocks is None:
        stashed = m.get("_anthropic_content_blocks")
        if isinstance(stashed, list) and stashed:
            text_content = (
                content if isinstance(content, str) and content.strip() else None
            )
            multimodal_blocks = (
                [{"type": "text", "text": text_content}] + stashed
                if text_content
                else list(stashed)
            )

    if multimodal_blocks:
        result_content: Any = multimodal_blocks
    elif isinstance(content, str):
        result_content = content
    else:
        result_content = json.dumps(content) if content else "(no output)"
    if not result_content:
        result_content = "(no output)"

    tool_result = {
        "type": "tool_result",
        "tool_use_id": _sanitize_tool_id(m.get("tool_call_id", "")),
        "content": result_content,
    }
    if isinstance(m.get("cache_control"), dict):
        tool_result["cache_control"] = dict(m["cache_control"])

    if (
        result
        and result[-1]["role"] == "user"
        and isinstance(result[-1]["content"], list)
        and result[-1]["content"]
        and result[-1]["content"][0].get("type") == "tool_result"
    ):
        result[-1]["content"].append(tool_result)
    else:
        result.append({"role": "user", "content": [tool_result]})


def _convert_user_message(content: Any) -> dict[str, Any]:
    """Validate and convert a user message to Anthropic format."""
    if isinstance(content, list):
        converted_blocks = _convert_content_to_anthropic(content)
        if not converted_blocks or all(
            b.get("text", "").strip() == ""
            for b in converted_blocks
            if isinstance(b, dict) and b.get("type") == "text"
        ):
            converted_blocks = [{"type": "text", "text": "(empty message)"}]
        return {"role": "user", "content": converted_blocks}
    if not content or (isinstance(content, str) and not content.strip()):
        content = "(empty message)"
    return {"role": "user", "content": content}


# ---------------------------------------------------------------------------
# Repair pipeline: orphan-strip → merge → manage-thinking-signatures → evict
# ---------------------------------------------------------------------------

def _strip_orphaned_tool_blocks(result: list[dict[str, Any]]) -> None:
    """Strip tool_use blocks with no matching tool_result, and vice versa.

    Context compression or session truncation can remove either side of a
    tool-call pair, or insert messages between a tool_use and its result.
    Anthropic requires each tool_use to have a matching tool_result in the
    IMMEDIATELY FOLLOWING user message — a global ID match is not enough.

    Mutates *result* in place.
    """
    # Pass 1: strip tool_use blocks whose ID has no adjacent tool_result.
    for i, m in enumerate(result):
        if m.get("role") != "assistant" or not isinstance(m.get("content"), list):
            continue
        tool_use_ids_in_turn = {
            b.get("id")
            for b in m["content"]
            if isinstance(b, dict) and b.get("type") == "tool_use"
        }
        if not tool_use_ids_in_turn:
            continue

        adjacent_result_ids: set = set()
        if i + 1 < len(result):
            nxt = result[i + 1]
            if nxt.get("role") == "user" and isinstance(nxt.get("content"), list):
                for block in nxt["content"]:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        adjacent_result_ids.add(block.get("tool_use_id"))

        orphaned = tool_use_ids_in_turn - adjacent_result_ids
        if not orphaned:
            continue

        kept = [
            b
            for b in m["content"]
            if not (
                isinstance(b, dict)
                and b.get("type") == "tool_use"
                and b.get("id") in orphaned
            )
        ]
        # If stripping an orphaned tool_use mutated a turn that also carries
        # a signed thinking block, that block's signature was computed
        # against the ORIGINAL (un-stripped) turn content and is now invalid.
        if len(kept) != len(m["content"]) and any(
            isinstance(b, dict) and b.get("type") in {"thinking", "redacted_thinking"}
            for b in m["content"]
        ):
            m["_thinking_signature_invalidated"] = True
        m["content"] = kept if kept else [{"type": "text", "text": "(tool call removed)"}]

    # Pass 2: rebuild the set of surviving tool_use IDs, then strip
    # tool_result blocks that no longer have any matching tool_use.
    surviving_tool_use_ids: set = set()
    for m in result:
        if m.get("role") == "assistant" and isinstance(m.get("content"), list):
            for block in m["content"]:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    surviving_tool_use_ids.add(block.get("id"))

    for m in result:
        if m.get("role") != "user" or not isinstance(m.get("content"), list):
            continue
        new_content = [
            b
            for b in m["content"]
            if not (isinstance(b, dict) and b.get("type") == "tool_result")
            or b.get("tool_use_id") in surviving_tool_use_ids
        ]
        if len(new_content) != len(m["content"]):
            m["content"] = (
                new_content if new_content else [{"type": "text", "text": "(tool result removed)"}]
            )


def _merge_consecutive_roles(result: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge consecutive same-role messages to enforce Anthropic alternation.

    Returns a new list (caller must rebind ``result``).
    """
    fixed: list[dict[str, Any]] = []
    for m in result:
        if fixed and fixed[-1]["role"] == m["role"]:
            if m["role"] == "user":
                prev_content = fixed[-1]["content"]
                curr_content = m["content"]
                if isinstance(prev_content, str) and isinstance(curr_content, str):
                    fixed[-1]["content"] = prev_content + "\n" + curr_content
                elif isinstance(prev_content, list) and isinstance(curr_content, list):
                    fixed[-1]["content"] = prev_content + curr_content
                else:
                    if isinstance(prev_content, str):
                        prev_content = [{"type": "text", "text": prev_content}]
                    if isinstance(curr_content, str):
                        curr_content = [{"type": "text", "text": curr_content}]
                    fixed[-1]["content"] = prev_content + curr_content
            else:
                # Consecutive assistant messages — merge text content.
                # Propagate the orphan-strip signature-invalidation flag
                # onto the surviving (prev) dict so
                # _manage_thinking_signatures still sees it.
                if m.get("_thinking_signature_invalidated"):
                    fixed[-1]["_thinking_signature_invalidated"] = True
                # Drop thinking blocks from the SECOND message: their
                # signature was computed against a different turn boundary
                # and becomes invalid once merged.
                if isinstance(m["content"], list):
                    m["content"] = [
                        b
                        for b in m["content"]
                        if not (
                            isinstance(b, dict)
                            and b.get("type") in {"thinking", "redacted_thinking"}
                        )
                    ]
                prev_blocks = fixed[-1]["content"]
                curr_blocks = m["content"]
                if isinstance(prev_blocks, list) and isinstance(curr_blocks, list):
                    fixed[-1]["content"] = prev_blocks + curr_blocks
                elif isinstance(prev_blocks, str) and isinstance(curr_blocks, str):
                    fixed[-1]["content"] = prev_blocks + "\n" + curr_blocks
                else:
                    if isinstance(prev_blocks, str):
                        prev_blocks = [{"type": "text", "text": prev_blocks}]
                    if isinstance(curr_blocks, str):
                        curr_blocks = [{"type": "text", "text": curr_blocks}]
                    fixed[-1]["content"] = prev_blocks + curr_blocks
        else:
            fixed.append(m)
    return fixed


def _manage_thinking_signatures(
    result: list[dict[str, Any]],
    base_url: Optional[str],
    model: Optional[str],
) -> None:
    """Strip or preserve thinking blocks based on endpoint type.

    Anthropic signs thinking blocks against the full turn content.  Any
    upstream mutation (compression, truncation, orphan stripping, message
    merging) invalidates the signature, causing HTTP 400 "Invalid signature
    in thinking block".

    Signatures are Anthropic-proprietary.  Third-party endpoints cannot
    validate them and reject them outright.  Kimi's ``/coding`` and
    DeepSeek's ``/anthropic`` endpoints speak the Anthropic protocol
    upstream but require unsigned thinking blocks to round-trip on replayed
    assistant tool-call messages.

    Mutates *result* in place.
    """
    _THINKING_TYPES = frozenset(("thinking", "redacted_thinking"))
    _is_third_party = _is_third_party_anthropic_endpoint(base_url)
    _preserve_unsigned_thinking = (
        _is_kimi_family_endpoint(base_url, model)
        or _is_deepseek_anthropic_endpoint(base_url)
    )

    last_assistant_idx: Optional[int] = None
    for i in range(len(result) - 1, -1, -1):
        if result[i].get("role") == "assistant":
            last_assistant_idx = i
            break

    for idx, m in enumerate(result):
        if m.get("role") != "assistant" or not isinstance(m.get("content"), list):
            continue

        if _preserve_unsigned_thinking:
            # Kimi / DeepSeek: strip signed, preserve unsigned.
            new_content = []
            for b in m["content"]:
                if not isinstance(b, dict) or b.get("type") not in _THINKING_TYPES:
                    new_content.append(b)
                    continue
                if b.get("signature") or b.get("data"):
                    continue  # Signed or redacted-with-data — strip.
                new_content.append(b)
            m["content"] = new_content or [{"type": "text", "text": "(empty)"}]
        elif _is_third_party or idx != last_assistant_idx:
            # Third-party: strip ALL thinking blocks.
            # Direct Anthropic: strip from non-latest assistant messages only.
            stripped = [
                b
                for b in m["content"]
                if not (isinstance(b, dict) and b.get("type") in _THINKING_TYPES)
            ]
            m["content"] = stripped or [{"type": "text", "text": "(thinking elided)"}]
        else:
            # Latest assistant on direct Anthropic: keep signed, downgrade
            # unsigned to text so the reasoning isn't lost.
            signature_dead = bool(m.get("_thinking_signature_invalidated"))
            new_content = []
            for b in m["content"]:
                if not isinstance(b, dict) or b.get("type") not in _THINKING_TYPES:
                    new_content.append(b)
                    continue
                if signature_dead:
                    thinking_text = b.get("thinking", "")
                    if thinking_text:
                        new_content.append({"type": "text", "text": thinking_text})
                    continue
                if b.get("type") == "redacted_thinking":
                    if b.get("data"):
                        new_content.append(b)
                elif b.get("signature"):
                    new_content.append(b)
                else:
                    thinking_text = b.get("thinking", "")
                    if thinking_text:
                        new_content.append({"type": "text", "text": thinking_text})
            m["content"] = new_content or [{"type": "text", "text": "(empty)"}]

        # Strip cache_control from any remaining thinking blocks.
        for b in m["content"]:
            if isinstance(b, dict) and b.get("type") in _THINKING_TYPES:
                b.pop("cache_control", None)

        # Drop the internal bookkeeping flag.
        m.pop("_thinking_signature_invalidated", None)


def _evict_old_screenshots(result: list[dict[str, Any]]) -> None:
    """Keep only the most recent ``_MAX_KEEP_SCREENSHOTS`` computer-use screenshots.

    Base64 images cost ~1,465 tokens each and accumulate across tool calls.
    Walk backward, keep the most recent N, replace older ones with a
    placeholder.  Mutates *result* in place.
    """
    _image_count = 0
    for msg in reversed(result):
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            inner = block.get("content")
            if not isinstance(inner, list):
                continue
            has_image = any(
                isinstance(b, dict) and b.get("type") == "image" for b in inner
            )
            if not has_image:
                continue
            _image_count += 1
            if _image_count > _MAX_KEEP_SCREENSHOTS:
                block["content"] = [
                    b if b.get("type") != "image"
                    else {"type": "text", "text": "[screenshot removed to save context]"}
                    for b in inner
                ]


# ---------------------------------------------------------------------------
# Top-level OpenAI → Anthropic conversion (entry point)
# ---------------------------------------------------------------------------

def convert_messages_to_anthropic(
    messages: list[dict],
    base_url: Optional[str] = None,
    model: Optional[str] = None,
) -> Tuple[Optional[Any], list[dict]]:
    """Convert OpenAI-format messages to Anthropic format.

    Returns ``(system_prompt, anthropic_messages)``.  System messages are
    extracted since Anthropic takes them as a separate param.  ``system_prompt``
    is a string or list of content blocks (when ``cache_control`` is present).

    When *base_url* points to a third-party Anthropic-compatible endpoint,
    all thinking block signatures are stripped.  When *model* matches the
    Kimi/Moonshot family (or *base_url* is a Kimi/Moonshot host), unsigned
    thinking blocks synthesised from ``reasoning_content`` are preserved on
    replayed assistant tool-call messages.
    """
    system: Any = None
    result: list[dict[str, Any]] = []

    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role", "user")
        content = m.get("content", "")

        if role == "system":
            if isinstance(content, list):
                has_cache = any(
                    p.get("cache_control") for p in content if isinstance(p, dict)
                )
                if has_cache:
                    system = [p for p in content if isinstance(p, dict)]
                else:
                    system = "\n".join(
                        p["text"]
                        for p in content
                        if isinstance(p, dict) and p.get("type") == "text"
                    )
            else:
                system = content
            continue

        if role == "assistant":
            result.append(_convert_assistant_message(m))
            continue

        if role == "tool":
            _convert_tool_message_to_result(result, m)
            continue

        # Regular user message.
        result.append(_convert_user_message(content))

    _strip_orphaned_tool_blocks(result)
    result = _merge_consecutive_roles(result)
    _manage_thinking_signatures(result, base_url, model)
    _evict_old_screenshots(result)

    return system, result


# ---------------------------------------------------------------------------
# NIA ConversationMessage → Anthropic wire-format conversion
# ---------------------------------------------------------------------------

def _convert_nia_block_to_anthropic(block: Any) -> Optional[dict[str, Any]]:
    """Convert a NIA content block (Pydantic) to an Anthropic wire-format dict.

    NIA uses four block types — ``TextBlock``, ``ThinkingBlock``,
    ``ToolUseBlock``, ``ToolResultBlock`` — all of which are already close
    to Anthropic's wire format.  This function strips Pydantic wrappers and
    applies the same whitelist rules as ``_sanitize_replay_block``.
    """
    # Pydantic v2 BaseModel.
    if hasattr(block, "model_dump"):
        block_dict = block.model_dump()
    elif isinstance(block, dict):
        block_dict = block
    else:
        return None

    return _sanitize_replay_block(block_dict)


def convert_conversation_messages_to_anthropic(
    messages: list[Any],
    *,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> Tuple[Optional[Any], list[dict[str, Any]]]:
    """Convert NIA :class:`ConversationMessage` list → Anthropic wire format.

    This is the bridge used by ``AnthropicApiClient._stream_once``.  Each
    NIA message becomes an Anthropic-format dict; the conversion preserves
    text/thinking/tool_use/tool_result blocks, applies tool-id sanitization,
    and runs the same repair pipeline as the OpenAI path.

    Args:
        messages: list of ``ConversationMessage`` Pydantic objects (or
            dicts already in NIA format).
        base_url: Base URL — used for thinking-signature policy.
        model: Model name — used for thinking-signature policy.
        system_prompt: Optional system prompt string.  If provided, returned
            as the first tuple element (string or cacheable block list).

    Returns:
        ``(system, anthropic_messages)`` — ready to splat into
        ``client.messages.create(**kwargs)``.
    """
    # Build OpenAI-format dicts from NIA messages so we can reuse the
    # existing convert_messages_to_anthropic repair pipeline.  We use a
    # lossless mapping: each NIA block becomes the corresponding OpenAI
    # block shape, then convert_messages_to_anthropic normalizes it back
    # to Anthropic wire format (with repair + thinking-signature policy).
    openai_format: list[dict[str, Any]] = []

    for msg in messages:
        # Pull the content list out of either a Pydantic model or a dict.
        if hasattr(msg, "model_dump"):
            msg_dict = msg.model_dump()
        elif isinstance(msg, dict):
            msg_dict = msg
        else:
            continue
        role = msg_dict.get("role", "user")
        blocks = msg_dict.get("content", []) or []

        if role == "user":
            # NIA user messages may contain tool_result blocks (when the
            # user is feeding back tool output) or text blocks.
            tool_results = [
                b for b in blocks
                if isinstance(b, dict) and b.get("type") == "tool_result"
            ]
            other_blocks = [
                b for b in blocks
                if not (isinstance(b, dict) and b.get("type") == "tool_result")
            ]
            for tr in tool_results:
                openai_format.append({
                    "role": "tool",
                    "tool_call_id": tr.get("tool_use_id", ""),
                    "content": tr.get("content", ""),
                    "cache_control": tr.get("cache_control"),
                })
            if other_blocks:
                openai_format.append({"role": "user", "content": other_blocks})
            elif not tool_results and not other_blocks:
                openai_format.append({"role": "user", "content": ""})
        elif role == "assistant":
            # Reconstruct an OpenAI-format assistant message with tool_calls
            # array for any tool_use blocks; remaining blocks (text/thinking)
            # become the message "content".
            tool_calls = []
            content_blocks = []
            preserved_thinking = []
            for b in blocks:
                if not isinstance(b, dict):
                    continue
                btype = b.get("type")
                if btype == "tool_use":
                    tool_calls.append({
                        "id": b.get("id", ""),
                        "function": {
                            "name": b.get("name", ""),
                            "arguments": json.dumps(b.get("input", {})),
                        },
                    })
                elif btype in {"thinking", "redacted_thinking"}:
                    preserved_thinking.append(copy.deepcopy(b))
                else:
                    content_blocks.append(b)
            entry: dict[str, Any] = {
                "role": "assistant",
                "content": content_blocks or "",
            }
            if tool_calls:
                entry["tool_calls"] = tool_calls
            if preserved_thinking:
                entry["reasoning_details"] = preserved_thinking
            openai_format.append(entry)

    # If a system prompt was provided, prepend it as a system message so
    # convert_messages_to_anthropic can extract + cache it.
    if system_prompt:
        openai_format.insert(0, {"role": "system", "content": system_prompt})

    return convert_messages_to_anthropic(openai_format, base_url=base_url, model=model)


# ---------------------------------------------------------------------------
# Responses-API-only kwarg stripping (defensive guard)
# ---------------------------------------------------------------------------

def sanitize_anthropic_kwargs(api_kwargs: Any, *, log_prefix: str = "") -> Any:
    """Drop Responses-API-only keys before an Anthropic Messages SDK call.

    Defensive boundary guard: under rare api_mode-flip races, a
    Responses-shaped payload carrying ``instructions=`` can reach
    ``messages.stream()``.  The Anthropic SDK rejects it with a
    non-retryable ``TypeError`` that nukes the whole turn.

    Mutates *api_kwargs* in place and returns it.
    """
    if not isinstance(api_kwargs, dict):
        return api_kwargs
    leaked = _RESPONSES_ONLY_KWARGS.intersection(api_kwargs)
    if leaked:
        for _key in leaked:
            api_kwargs.pop(_key, None)
        logger.warning(
            "%sStripped Responses-only kwarg(s) %s from an Anthropic Messages "
            "call (api_mode flip race). The call will proceed.",
            log_prefix,
            sorted(leaked),
        )
    return api_kwargs


# ---------------------------------------------------------------------------
# THE CENTRAL ENTRY POINT — build_anthropic_kwargs
# ---------------------------------------------------------------------------

def build_anthropic_kwargs(
    model: str,
    messages: list[Any],
    *,
    system_prompt: Optional[str] = None,
    tools: Optional[list[dict]] = None,
    max_tokens: Optional[int] = None,
    reasoning_effort: Optional[str] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    tool_choice: Optional[str] = None,
    base_url: Optional[str] = None,
    is_oauth: bool = False,
    preserve_dots: bool = False,
    context_length: Optional[int] = None,
    fast_mode: bool = False,
    drop_context_1m_beta: bool = False,
    enable_caching: bool = True,
    stop_sequences: Optional[list[str]] = None,
    **extra_kwargs: Any,
) -> dict[str, Any]:
    """Build the kwargs dict consumed by ``client.messages.create(**kwargs)``.

    This is the central entry point that orchestrates:

      1. Message conversion (NIA ``ConversationMessage`` or OpenAI-format
         dicts → Anthropic wire format via
         :func:`convert_conversation_messages_to_anthropic`).
      2. Tool conversion + cache_control on the last tool.
      3. Model-name normalization (``claude-opus-4.6`` → ``claude-opus-4-6``).
      4. ``max_tokens`` resolution (caller-provided → model ceiling → raise).
      5. OAuth transforms (system prefix, name sanitization, ``mcp__`` tool
         prefixing) when ``is_oauth=True``.
      6. Reasoning-config → thinking mapping (adaptive vs manual budget).
      7. Sampling-param strip when the model forbids them.
      8. Fast-mode injection when supported.
      9. Prompt-caching markers on system / last tool / last assistant block.

    Args:
        model: Model name (will be normalized).
        messages: List of NIA ``ConversationMessage`` objects OR OpenAI-format
            message dicts.  Auto-detected per element.
        system_prompt: Optional system prompt string.
        tools: List of tool definitions (OpenAI-format or Anthropic-format).
        max_tokens: Max output tokens. If ``None`` or non-positive, falls
            back to the model's output ceiling.
        reasoning_effort: One of ``"max" | "xhigh" | "high" | "medium" |
            "low" | "minimal"``.  ``None`` disables thinking.
        temperature: Sampling temperature.
        top_p: Nucleus sampling probability.
        tool_choice: Tool-choice mode (``"auto" | "any" | "none"``).
        base_url: Base URL — used for endpoint-type detection (thinking
            signature policy, beta headers, OAuth transforms).
        is_oauth: When True, apply OAuth-path transforms (system prefix,
            name sanitization, ``mcp__`` tool prefixing).
        preserve_dots: Preserve dots in model name (for non-Claude models).
        context_length: Context window size (for max_tokens clamping).
        fast_mode: Enable fast-mode beta (Opus 4.6 only).
        drop_context_1m_beta: Strip the 1M-context beta.
        enable_caching: Apply prompt-caching markers (default True).
        stop_sequences: Optional stop sequences.
        **extra_kwargs: Additional SDK kwargs (e.g. ``user_id``).

    Returns:
        Dict suitable for ``client.messages.create(**result)`` or
        ``client.messages.stream(**result)``.
    """
    # 1. Normalize model name.
    normalized_model = normalize_model_name(model, preserve_dots=preserve_dots)

    # 2. Convert messages → Anthropic wire format.  Auto-detect input shape:
    #    if the first message is a Pydantic model or has a "content" key
    #    whose items have NIA block types, use the NIA path; otherwise use
    #    the OpenAI path.
    is_nia_messages = bool(messages) and (
        hasattr(messages[0], "model_dump")
        or (
            isinstance(messages[0], dict)
            and isinstance(messages[0].get("content"), list)
            and any(
                isinstance(b, dict) and b.get("type") in {
                    "text", "thinking", "tool_use", "tool_result"
                }
                for b in messages[0]["content"]
            )
            and messages[0].get("role") in {"user", "assistant"}
            # Exclude OpenAI-format: OpenAI assistant messages have "tool_calls"
            # and OpenAI user messages have content parts with "input_text" /
            # "image_url" types.
            and "tool_calls" not in messages[0]
        )
    )

    if is_nia_messages:
        system, anthropic_messages = convert_conversation_messages_to_anthropic(
            messages, base_url=base_url, model=normalized_model,
            system_prompt=system_prompt,
        )
    else:
        # OpenAI-format input: prepend system as a system message.
        openai_messages = list(messages)
        if system_prompt:
            openai_messages.insert(0, {"role": "system", "content": system_prompt})
        system, anthropic_messages = convert_messages_to_anthropic(
            openai_messages, base_url=base_url, model=normalized_model,
        )

    # 3. Convert tools + apply cache_control on the last tool.
    anthropic_tools = convert_tools_to_anthropic(tools)
    if enable_caching and anthropic_tools:
        apply_cache_control_to_last_tool(anthropic_tools)

    # 4. max_tokens resolution.
    resolved_max_tokens = _resolve_anthropic_messages_max_tokens(
        max_tokens, normalized_model, context_length=context_length
    )

    # 5. Build the kwargs dict.
    kwargs: dict[str, Any] = {
        "model": normalized_model,
        "messages": anthropic_messages,
        "max_tokens": resolved_max_tokens,
    }
    if system:
        if enable_caching:
            system = apply_cache_control_to_system(system)
        kwargs["system"] = system
    if anthropic_tools:
        kwargs["tools"] = anthropic_tools
    if tool_choice:
        kwargs["tool_choice"] = {"type": tool_choice}

    # 6. OAuth transforms.
    if is_oauth:
        # Prepend the Claude Code system prefix.
        if "system" in kwargs:
            existing = kwargs["system"]
            if isinstance(existing, str):
                kwargs["system"] = _CLAUDE_CODE_SYSTEM_PREFIX + "\n\n" + existing
            elif isinstance(existing, list):
                kwargs["system"] = [
                    {"type": "text", "text": _CLAUDE_CODE_SYSTEM_PREFIX},
                    *existing,
                ]
        else:
            kwargs["system"] = _CLAUDE_CODE_SYSTEM_PREFIX
        # Prefix tool names with mcp__ (OAuth wire format).
        if "tools" in kwargs:
            for t in kwargs["tools"]:
                if not t["name"].startswith(_MCP_TOOL_PREFIX):
                    t["name"] = _MCP_TOOL_PREFIX + t["name"]
        # Prefix tool_use IDs in messages (so they match the prefixed tool names).
        for msg in kwargs["messages"]:
            if not isinstance(msg.get("content"), list):
                continue
            for b in msg["content"]:
                if isinstance(b, dict):
                    if b.get("type") == "tool_use" and not b["name"].startswith(_MCP_TOOL_PREFIX):
                        b["name"] = _MCP_TOOL_PREFIX + b["name"]

    # 7. Reasoning-config → thinking mapping.
    if reasoning_effort:
        effort = (reasoning_effort or "").lower().strip()
        if _supports_adaptive_thinking(normalized_model):
            # Adaptive thinking path (Claude 4-6+).
            if effort in ADAPTIVE_EFFORT_MAP:
                if not _supports_xhigh_effort(normalized_model) and ADAPTIVE_EFFORT_MAP[effort] == "xhigh":
                    # Downgrade xhigh → high on models that don't support it.
                    kwargs["thinking"] = {
                        "type": "enabled",
                        "output_config": {"effort": "high"},
                    }
                else:
                    kwargs["thinking"] = {
                        "type": "enabled",
                        "output_config": {"effort": ADAPTIVE_EFFORT_MAP[effort]},
                    }
        elif effort in THINKING_BUDGET:
            # Legacy manual budget path (Claude 4-0, 4-1, 4-5, 3-x).
            budget = THINKING_BUDGET[effort]
            # Clamp budget to max_tokens - 1 (Anthropic requires budget < max_tokens).
            budget = min(budget, resolved_max_tokens - 1)
            if budget > 0:
                kwargs["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": budget,
                }

    # 8. Sampling params — strip if the model forbids them.
    if temperature is not None and not _forbids_sampling_params(normalized_model):
        kwargs["temperature"] = temperature
    if top_p is not None and not _forbids_sampling_params(normalized_model):
        kwargs["top_p"] = top_p

    # 9. Stop sequences.
    if stop_sequences:
        kwargs["stop_sequences"] = list(stop_sequences)

    # 10. Fast-mode beta (Opus 4.6 only).
    if fast_mode and _supports_fast_mode(normalized_model):
        betas = kwargs.setdefault("_extra_headers", {}).setdefault("anthropic-beta", "")
        existing_betas = [b for b in betas.split(",") if b] if betas else []
        if _FAST_MODE_BETA not in existing_betas:
            existing_betas.append(_FAST_MODE_BETA)
        kwargs["_extra_headers"]["anthropic-beta"] = ",".join(existing_betas)

    # 11. Merge extra kwargs (e.g. user_id, metadata).
    for k, v in extra_kwargs.items():
        if v is not None and k not in kwargs:
            kwargs[k] = v

    return sanitize_anthropic_kwargs(kwargs)


# ---------------------------------------------------------------------------
# SDK invocation wrapper (stream-with-create-fallback)
# ---------------------------------------------------------------------------

def _is_stream_unavailable_error(exc: Exception) -> bool:
    """Detect "stream not supported" / Bedrock access-denied errors."""
    msg = str(exc).lower()
    if "stream" in msg and ("not supported" in msg or "unavailable" in msg):
        return True
    if "invokemodelwithresponsestream" in msg and "access" in msg:
        return True
    return False


async def create_anthropic_message(
    client: Any,
    api_kwargs: dict[str, Any],
    *,
    log_prefix: str = "",
    prefer_stream: bool = True,
) -> Any:
    """Execute ``client.messages.stream(...).get_final_message()`` with fallback.

    Prefers streaming (required for SSE-only gateways like Kimi /coding).
    Falls back to ``messages.create()`` when streaming is unavailable.

    Args:
        client: ``AsyncAnthropic`` (or compatible) client.
        api_kwargs: Kwargs dict from :func:`build_anthropic_kwargs`.
        log_prefix: Prefix for log messages.
        prefer_stream: When True (default), try streaming first.

    Returns:
        The final message object from the SDK.
    """
    # Strip kwargs that the SDK doesn't accept (internal helpers).
    kwargs = {k: v for k, v in api_kwargs.items() if not k.startswith("_")}
    # Merge _extra_headers into the SDK call.
    extra_headers = api_kwargs.get("_extra_headers")
    if extra_headers:
        kwargs["extra_headers"] = extra_headers

    kwargs = sanitize_anthropic_kwargs(kwargs, log_prefix=log_prefix)

    if prefer_stream:
        try:
            async with client.messages.stream(**kwargs) as stream:
                return await stream.get_final_message()
        except Exception as exc:
            if _is_stream_unavailable_error(exc):
                logger.info(
                    "%sStreaming unavailable (%s); falling back to messages.create()",
                    log_prefix, exc,
                )
                return await client.messages.create(**kwargs)
            raise

    return await client.messages.create(**kwargs)


__all__ = [
    # Constants
    "THINKING_BUDGET",
    "ADAPTIVE_EFFORT_MAP",
    "_COMMON_BETAS",
    "_OAUTH_CLIENT_ID",
    # Endpoint classifiers
    "base_url_host_matches",
    "_is_third_party_anthropic_endpoint",
    "_is_kimi_coding_endpoint",
    "_is_kimi_family_endpoint",
    "_is_deepseek_anthropic_endpoint",
    "_is_minimax_anthropic_endpoint",
    "_is_azure_anthropic_endpoint",
    "_requires_bearer_auth",
    "_common_betas_for_base_url",
    # Token / client construction
    "resolve_anthropic_token",
    "build_anthropic_client",
    # Model-name normalization + capability detection
    "normalize_model_name",
    "_is_claude_model",
    "_get_anthropic_max_output",
    "_resolve_positive_max_tokens",
    "_resolve_anthropic_messages_max_tokens",
    "_supports_adaptive_thinking",
    "_supports_xhigh_effort",
    "_forbids_sampling_params",
    "_supports_fast_mode",
    # Tool-schema sanitization
    "_strip_nullable_unions",
    "_normalize_tool_input_schema",
    "_sanitize_tool_id",
    "convert_tools_to_anthropic",
    # Image / content conversion
    "_image_source_from_openai_url",
    "_convert_content_part_to_anthropic",
    "_convert_content_to_anthropic",
    "_content_parts_to_anthropic_blocks",
    "_to_plain_data",
    "_extract_preserved_thinking_blocks",
    "_sanitize_replay_block",
    # Prompt caching
    "_apply_assistant_cache_control_to_last_cacheable_block",
    "apply_cache_control_to_last_tool",
    "apply_cache_control_to_system",
    # Per-role converters
    "_convert_assistant_message",
    "_convert_tool_message_to_result",
    "_convert_user_message",
    # Repair pipeline
    "_strip_orphaned_tool_blocks",
    "_merge_consecutive_roles",
    "_manage_thinking_signatures",
    "_evict_old_screenshots",
    # Top-level conversion entry points
    "convert_messages_to_anthropic",
    "convert_conversation_messages_to_anthropic",
    # Defensive guard
    "sanitize_anthropic_kwargs",
    # Central entry point
    "build_anthropic_kwargs",
    # SDK invocation
    "create_anthropic_message",
    "_is_stream_unavailable_error",
]
