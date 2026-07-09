"""Secret redaction for logs, streamed output, and trajectories.

Ported from Hermes Agent's agent/redact.py (811 LOC), scoped to what NIA
needs: redact API keys, tokens, and other secrets from text before it
reaches the UI or logs.

Usage::

    from niaharness.utils.redact import redact_secrets

    safe = redact_secrets("My key is sk-ant-abc123...")
    # → "My key is [REDACTED:api_key]..."
"""

from __future__ import annotations

import re
from typing import Optional

# Patterns for common secret formats. Each pattern is (regex, label).
# The regex matches the secret value; the label is used in the redaction
# placeholder: [REDACTED:label].
_SECRET_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Anthropic API keys: sk-ant-api03-...
    (re.compile(r"sk-ant-api03-[A-Za-z0-9\-_]{20,}"), "anthropic_key"),
    (re.compile(r"sk-ant-[A-Za-z0-9\-_]{20,}"), "anthropic_key"),
    # OpenAI API keys: sk-proj-... or sk-...
    (re.compile(r"sk-proj-[A-Za-z0-9\-_]{20,}"), "openai_key"),
    (re.compile(r"sk-[A-Za-z0-9]{40,}"), "openai_key"),
    # Generic API keys (long hex/base64 strings after key= or api_key=)
    (re.compile(r"(?:api[_-]?key|apikey)\s*[=:]\s*[\"']?([A-Za-z0-9\-_]{32,})[\"']?", re.IGNORECASE), "api_key"),
    # Bearer tokens
    (re.compile(r"Bearer\s+([A-Za-z0-9\-_\.]{20,})", re.IGNORECASE), "bearer_token"),
    # AWS access keys
    (re.compile(r"AKIA[0-9A-Z]{16}"), "aws_key"),
    # AWS secret keys (40-char base64 after secret)
    (re.compile(r"(?:secret|aws_secret_access_key)\s*[=:]\s*[\"']?([A-Za-z0-9/+=]{40})[\"']?", re.IGNORECASE), "aws_secret"),
    # GitHub tokens
    (re.compile(r"gh[ps]_[A-Za-z0-9]{36,}"), "github_token"),
    (re.compile(r"github_pat_[A-Za-z0-9_]{82}"), "github_token"),
    # Generic long hex tokens (64+ chars, likely SHA-256 hashes or tokens)
    (re.compile(r"\b[A-Fa-f0-9]{64}\b"), "hex_token"),
    # JWT tokens (eyJ...)
    (re.compile(r"eyJ[A-Za-z0-9\-_]+\.eyJ[A-Za-z0-9\-_]+\.[A-Za-z0-9\-_]+"), "jwt"),
    # Private key blocks
    (re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----[\s\S]*?-----END (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"), "private_key"),
]


def redact_secrets(text: str, *, replacement: str = "[REDACTED:{label}]") -> str:
    """Redact secrets from text.

    Replaces API keys, tokens, JWTs, private keys, and other secret patterns
    with a placeholder like ``[REDACTED:api_key]``.

    Args:
        text: The text to redact.
        replacement: Template for the replacement. ``{label}`` is replaced
            with the secret type (e.g. "api_key", "bearer_token").

    Returns:
        Redacted text.
    """
    if not text:
        return text

    result = text
    for pattern, label in _SECRET_PATTERNS:
        repl = replacement.format(label=label)
        result = pattern.sub(repl, result)

    return result


def redact_secrets_in_dict(data: dict, *, replacement: str = "[REDACTED:{label}]") -> dict:
    """Redact secrets in all string values of a dict (recursive).

    Args:
        data: The dict to redact.
        replacement: Template for the replacement.

    Returns:
        New dict with secrets redacted.
    """
    if not isinstance(data, dict):
        return data

    redacted: dict = {}
    for key, value in data.items():
        if isinstance(value, str):
            redacted[key] = redact_secrets(value, replacement=replacement)
        elif isinstance(value, dict):
            redacted[key] = redact_secrets_in_dict(value, replacement=replacement)
        elif isinstance(value, list):
            redacted[key] = [
                redact_secrets(item, replacement=replacement) if isinstance(item, str)
                else redact_secrets_in_dict(item, replacement=replacement) if isinstance(item, dict)
                else item
                for item in value
            ]
        else:
            redacted[key] = value
    return redacted


__all__ = [
    "redact_secrets",
    "redact_secrets_in_dict",
]
