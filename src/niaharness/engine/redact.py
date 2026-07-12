"""Regex-based secret redaction for logs and tool output.

Ported from hermes-agent/agent/redact.py (812 LOC).

Applies pattern matching to mask API keys, tokens, and credentials
before they reach log files, verbose output, or gateway logs.

Short tokens (< 18 chars) are fully masked. Longer tokens preserve
the first 6 and last 4 characters for debuggability.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)

# Snapshot at import time so runtime env mutations cannot disable redaction
# mid-session. ON by default — secure default.
_REDACT_ENABLED = os.getenv(
    "NIA_REDACT_SECRETS", os.getenv("HERMES_REDACT_SECRETS", "true")
).lower() in {"1", "true", "yes", "on"}

# Known API key prefixes — match the prefix + contiguous token chars
_PREFIX_PATTERNS = [
    r"sk-[A-Za-z0-9_-]{10,}",           # OpenAI / OpenRouter / Anthropic (sk-ant-*)
    r"ghp_[A-Za-z0-9]{10,}",            # GitHub PAT (classic)
    r"github_pat_[A-Za-z0-9_]{10,}",    # GitHub PAT (fine-grained)
    r"gho_[A-Za-z0-9]{10,}",            # GitHub OAuth access token
    r"ghu_[A-Za-z0-9]{10,}",            # GitHub user-to-server token
    r"ghs_[A-Za-z0-9]{10,}",            # GitHub server-to-server token
    r"ghr_[A-Za-z0-9]{10,}",            # GitHub refresh token
    r"xox[baprs]-[A-Za-z0-9-]{10,}",    # Slack bot/app/user tokens
    r"AIza[A-Za-z0-9_-]{30,}",          # Google API keys
    r"AKIA[A-Z0-9]{16}",                # AWS Access Key ID
    r"sk_live_[A-Za-z0-9]{10,}",        # Stripe secret key (live)
    r"sk_test_[A-Za-z0-9]{10,}",        # Stripe secret key (test)
    r"SG\.[A-Za-z0-9_-]{10,}",          # SendGrid API key
    r"hf_[A-Za-z0-9]{10,}",             # HuggingFace token
    r"npm_[A-Za-z0-9]{10,}",            # npm access token
    r"pypi-[A-Za-z0-9_-]{10,}",         # PyPI API token
    r"xai-[A-Za-z0-9]{30,}",            # xAI (Grok) API key
    r"fw-[A-Za-z0-9]{30,}",             # Fireworks AI API key
    r"gsk_[A-Za-z0-9]{10,}",            # Groq Cloud API key
    r"tvly-[A-Za-z0-9]{10,}",           # Tavily search API key
    r"exa_[A-Za-z0-9]{10,}",            # Exa search API key
    r"pplx-[A-Za-z0-9]{10,}",           # Perplexity
]

# ENV assignment patterns: KEY=value where KEY contains a secret-like name.
_SECRET_ENV_NAMES = r"(?:API_?KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH)"
_ENV_ASSIGN_RE = re.compile(
    rf"([A-Z0-9_]{{0,50}}{_SECRET_ENV_NAMES}[A-Z0-9_]{{0,50}})\s*=\s*(['\"]?)(\S+)\2",
)

# Lowercase / dotted / hyphenated config keys from config files
_SECRET_CFG_NAMES = r"(?:api[ _.\-]?key|token|secret|passwd|password|credential|auth)"
_CFG_VALUE = r"(['\"]?)([^\s&]+?)\2(?=[\s&]|$)"
_CFG_DOTTED_RE = re.compile(
    rf"((?:[A-Za-z0-9_\-]+\.)+[A-Za-z0-9_.\-]*{_SECRET_CFG_NAMES}[A-Za-z0-9_.\-]*"
    rf"|[A-Za-z0-9_.\-]*{_SECRET_CFG_NAMES}[A-Za-z0-9_.\-]*\.[A-Za-z0-9_.\-]+)"
    rf"={_CFG_VALUE}",
    re.IGNORECASE,
)
_CFG_ANCHORED_RE = re.compile(
    rf"(^[ \t]*(?:export[ \t]+)?[A-Za-z0-9_\-]*{_SECRET_CFG_NAMES}[A-Za-z0-9_\-]*)={_CFG_VALUE}",
    re.IGNORECASE | re.MULTILINE,
)

# Unquoted YAML / colon config
_YAML_CFG_NAMES = r"(?:api[ _.\-]?key|token|secret|passwd|password|credential)"
_YAML_ASSIGN_RE = re.compile(
    rf"(^[ \t]*[A-Za-z0-9_.\-]*{_YAML_CFG_NAMES}[A-Za-z0-9_.\-]*)(:[ \t]*)(?!['\"])([^\s&]+)",
    re.IGNORECASE | re.MULTILINE,
)

# JSON field patterns
_JSON_KEY_NAMES = (
    r"(?:api_?[Kk]ey|token|secret|password|access_token|refresh_token|"
    r"auth_token|bearer|secret_value|raw_secret|secret_input|key_material)"
)
_JSON_FIELD_RE = re.compile(
    rf'("{_JSON_KEY_NAMES}")\s*:\s*"([^"]+)"',
    re.IGNORECASE,
)

# Authorization headers — any scheme (Bearer, Basic, Token, Digest, …)
_AUTH_HEADER_RE = re.compile(
    r"((?:Proxy-)?Authorization:\s*)([A-Za-z][\w.+-]*\s+)?([^\s\"']+)",
    re.IGNORECASE,
)

# API-key style auth headers carrying a single opaque value
_SECRET_HEADER_NAMES = (
    r"(?:x-api-key|x-goog-api-key|api-key|apikey|x-api-token|x-auth-token|x-access-token)"
)
_SECRET_HEADER_RE = re.compile(
    rf"({_SECRET_HEADER_NAMES}\s*:\s*)(\S+)",
    re.IGNORECASE,
)

# Telegram bot tokens
_TELEGRAM_RE = re.compile(r"(bot)?(\d{8,}):([-A-Za-z0-9_]{30,})")

# Private key blocks
_PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN[A-Z ]*PRIVATE KEY-----[\s\S]*?-----END[A-Z ]*PRIVATE KEY-----"
)

# Database connection strings: protocol://user:PASSWORD@host
_DB_CONNSTR_RE = re.compile(
    r"((?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|amqp)://[^:\s]+:)([^@\s]+)(@)",
    re.IGNORECASE,
)

# Bare-token credential in a web/transport URL: ``scheme://TOKEN@host``
_URL_BARE_TOKEN_RE = re.compile(
    r"((?:https?|wss?|git|ssh|ftp|ftps|sftp)://)"
    r"([^\s:@/]{8,})"
    r"(@[^\s]+)",
    re.IGNORECASE,
)

# JWT tokens
_JWT_RE = re.compile(
    r"eyJ[A-Za-z0-9_-]{10,}(?:\.[A-Za-z0-9_=-]{4,}){0,2}"
)

# Compile known prefix patterns into one alternation
_PREFIX_RE = re.compile(
    r"(?<![A-Za-z0-9_-])(" + "|".join(_PREFIX_PATTERNS) + r")(?![A-Za-z0-9_-])"
)

# Programmatic env lookups — skip redaction (code snippets, not secrets)
_ENV_LOOKUP_VALUE_RE = re.compile(
    r"^(?:os\.(?:getenv|environ)|process\.env|\$ENV\{)"
)

# Sensitive query-string parameter names
_SENSITIVE_QUERY_PARAMS = frozenset({
    "access_token", "refresh_token", "id_token", "token", "api_key", "apikey",
    "client_secret", "password", "auth", "jwt", "session", "secret", "key",
    "code", "signature", "x-amz-signature",
})


# ---------------------------------------------------------------------------
# Masking helpers
# ---------------------------------------------------------------------------


def mask_secret(
    value: str,
    *,
    head: int = 4,
    tail: int = 4,
    floor: int = 12,
    placeholder: str = "***",
    empty: str = "",
) -> str:
    """Mask a secret for display, preserving ``head`` and ``tail`` characters.

    Ported from hermes-agent/agent/redact.py line 309.

    Args:
        value:       The secret to mask. ``None``/empty returns ``empty``.
        head:        Leading characters to preserve. Default 4.
        tail:        Trailing characters to preserve. Default 4.
        floor:       Values shorter than ``head + tail + floor_margin`` are
                     fully masked (returns ``placeholder``). Default 12.
        placeholder: Value returned for too-short inputs. Default ``"***"``.
        empty:       Value returned when ``value`` is falsy.

    Examples:
        >>> mask_secret("sk-proj-abcdef1234567890")
        'sk-p...7890'
        >>> mask_secret("short")
        '***'
    """
    if not value:
        return empty
    if len(value) < floor:
        return placeholder
    return f"{value[:head]}...{value[-tail:]}"


def _mask_token(token: str) -> str:
    """Mask a log token — conservative 18-char floor, preserves 6 prefix / 4 suffix."""
    if not token:
        return "***"
    return mask_secret(token, head=6, tail=4, floor=18)


def _mask_token_nonreusable(token: str) -> str:
    """Replace a prefix-matched credential with a non-reusable sentinel.

    Used for file-read content so the mask can't be mistaken for a usable key
    and written back as one (issue #35519).
    """
    if not token:
        return "[REDACTED]"
    # Include the prefix in the sentinel so the source is debuggable.
    prefix = token.split("-")[0].split("_")[0]
    return f"«redacted:{prefix}…»"


def _has_known_prefix_substring(text: str) -> bool:
    """Cheap substring pre-check before running the prefix regex."""
    for prefix in ("sk-", "ghp_", "github_pat_", "gho_", "ghu_", "ghs_", "ghr_",
                   "xox", "AIza", "AKIA", "sk_live_", "sk_test_", "SG.", "hf_",
                   "npm_", "pypi-", "xai-", "fw-", "gsk_", "tvly-", "exa_", "pplx-"):
        if prefix in text:
            return True
    return False


# ---------------------------------------------------------------------------
# Query-string / form-body redaction
# ---------------------------------------------------------------------------


def _redact_query_string(query: str) -> str:
    """Redact sensitive parameter values in a URL query string."""
    if not query:
        return query
    parts = []
    for pair in query.split("&"):
        if "=" not in pair:
            parts.append(pair)
            continue
        key, _, value = pair.partition("=")
        if key.lower() in _SENSITIVE_QUERY_PARAMS:
            parts.append(f"{key}=***")
        else:
            parts.append(pair)
    return "&".join(parts)


def _redact_form_body(text: str) -> str:
    """Redact sensitive values in a form-urlencoded body (k=v&k=v)."""
    # Only triggers on clean k=v&k=v inputs.
    if "\n" in text:
        return text
    form_re = re.compile(
        r"^[A-Za-z_][A-Za-z0-9_.-]*=[^&\s]*(?:&[A-Za-z_][A-Za-z0-9_.-]*=[^&\s]*)+$"
    )
    if not form_re.match(text):
        return text
    return _redact_query_string(text)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def redact_sensitive_text(
    text: str,
    *,
    force: bool = False,
    code_file: bool = False,
    file_read: bool = False,
) -> str:
    """Apply all redaction patterns to a block of text.

    Ported from hermes-agent/agent/redact.py line 491.

    Safe to call on any string — non-matching text passes through unchanged.
    Enabled by default. Disable via security.redact_secrets: false in config.yaml.
    Set force=True for safety boundaries that must never return raw secrets.
    Set code_file=True to skip ENV/JSON patterns when the text is source code.
    Set file_read=True for file content returned to the agent (uses non-reusable
    sentinels instead of head/tail-preserving masks).
    """
    if text is None:
        return None  # type: ignore[return-value]
    if not isinstance(text, str):
        text = str(text)
    if not text:
        return text
    if not (force or _REDACT_ENABLED):
        return text

    if file_read:
        code_file = True

    # Known prefixes (sk-, ghp_, etc.)
    if _has_known_prefix_substring(text):
        _prefix_sub = _mask_token_nonreusable if file_read else _mask_token
        text = _PREFIX_RE.sub(lambda m: _prefix_sub(m.group(1)), text)

    # ENV assignments: OPENAI_API_KEY=***
    if not code_file:
        if "=" in text:
            def _redact_env(m: re.Match) -> str:
                name, quote, value = m.group(1), m.group(2), m.group(3)
                if _ENV_LOOKUP_VALUE_RE.match(value):
                    return m.group(0)
                return f"{name}={quote}{_mask_token(value)}{quote}"
            text = _ENV_ASSIGN_RE.sub(_redact_env, text)
            if "://" not in text:
                text = _CFG_DOTTED_RE.sub(_redact_env, text)
                text = _CFG_ANCHORED_RE.sub(_redact_env, text)

        # JSON fields: "apiKey": "***"
        if ":" in text and '"' in text:
            def _redact_json(m: re.Match) -> str:
                key, value = m.group(1), m.group(2)
                if _ENV_LOOKUP_VALUE_RE.match(value):
                    return m.group(0)
                return f'{key}: "{_mask_token(value)}"'
            text = _JSON_FIELD_RE.sub(_redact_json, text)

        # Unquoted YAML / colon config: password: ***
        if ":" in text and "://" not in text:
            def _redact_yaml(m: re.Match) -> str:
                key, sep, value = m.group(1), m.group(2), m.group(3)
                if _ENV_LOOKUP_VALUE_RE.match(value):
                    return m.group(0)
                return f"{key}{sep}{_mask_token(value)}"
            text = _YAML_ASSIGN_RE.sub(_redact_yaml, text)

    # Authorization headers
    if "uthorization" in text or "UTHORIZATION" in text:
        text = _AUTH_HEADER_RE.sub(
            lambda m: m.group(1) + (m.group(2) or "") + _mask_token(m.group(3)),
            text,
        )

    # API-key style headers
    if ":" in text:
        text = _SECRET_HEADER_RE.sub(
            lambda m: m.group(1) + _mask_token(m.group(2)),
            text,
        )

    # Telegram bot tokens
    if ":" in text:
        def _redact_telegram(m: re.Match) -> str:
            prefix = m.group(1) or ""
            digits = m.group(2)
            return f"{prefix}{digits}:***"
        text = _TELEGRAM_RE.sub(_redact_telegram, text)

    # Private key blocks
    if "BEGIN" in text and "-----" in text:
        text = _PRIVATE_KEY_RE.sub("[REDACTED PRIVATE KEY]", text)

    # Database connection string passwords
    if "://" in text:
        if code_file:
            def _redact_db(m: re.Match) -> str:
                pw = m.group(2)
                if pw.startswith("{") and pw.endswith("}"):
                    return m.group(0)
                return f"{m.group(1)}***{m.group(3)}"
            text = _DB_CONNSTR_RE.sub(_redact_db, text)
        else:
            text = _DB_CONNSTR_RE.sub(lambda m: f"{m.group(1)}***{m.group(3)}", text)

        # Bare-token userinfo: scheme://TOKEN@host
        text = _URL_BARE_TOKEN_RE.sub(
            lambda m: f"{m.group(1)}{_mask_token(m.group(2))}{m.group(3)}",
            text,
        )

    # JWT tokens
    if "eyJ" in text:
        text = _JWT_RE.sub(lambda m: _mask_token(m.group(0)), text)

    # Form-urlencoded bodies
    if "&" in text and "=" in text:
        text = _redact_form_body(text)

    return text


__all__ = [
    "redact_sensitive_text",
    "mask_secret",
]
