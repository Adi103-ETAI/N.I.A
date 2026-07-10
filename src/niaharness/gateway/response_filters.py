"""P1 Gateway response filters — intentional silence detection.

Ported from Hermes Agent's ``gateway/response_filters.py`` (107 LOC).

The gateway boundary decides whether a completed agent turn should be
delivered to the chat. Some agent turns intentionally produce no chat
output — the model emits a silence marker (``NO_REPLY``, ``[SILENT]``)
when it decides not to respond. This module detects those markers
reliably, both for complete responses and for streaming partial output.

Why this matters:
  - Without silence detection, the gateway would deliver "NO_REPLY" as
    a literal chat message — confusing and ugly.
  - Without partial-silence detection, a streaming gateway would show
    "NO" then "NO_" then "NO_REPLY" as the marker streams in, then
    belatedly retract it. The partial-silence check holds back the
    buffer until it's clear the response is (or isn't) a silence marker.

Usage::

    from niaharness.gateway.response_filters import (
        is_intentional_silence_response,
        is_partial_silence_marker,
    )

    # Complete response:
    if is_intentional_silence_response(reply):
        return  # Don't deliver — agent chose silence.

    # Streaming:
    for delta in stream:
        buffer += delta
        if is_partial_silence_marker(buffer):
            continue  # Hold back — might still resolve to a silence marker.
        emit(buffer)
        buffer = ""
"""

from __future__ import annotations

import unicodedata
from typing import Any

# Canonical model-emitted control token for intentional silence.
SILENT_REPLY_TOKEN = "NO_REPLY"

# Exact whole-response markers that mean "the agent intentionally chose not
# to reply". Keep this list small and explicit; arbitrary empty output
# remains an error/empty-response path, not silence.
LIVE_GATEWAY_SILENT_MARKERS = frozenset({
    "[SILENT]",
    "SILENT",
    "NO_REPLY",
    "NO REPLY",
})


def _canonical_silence_candidate(text: str) -> str:
    """Normalize whitespace + uppercase for comparison."""
    return " ".join(text.strip().upper().split())


def _strip_edge_silence_punctuation(text: str) -> str:
    """Strip stray edge punctuation without erasing marker structure.

    Models sometimes emit ``.NO_REPLY`` or ``*NO_REPLY*`` instead of the
    exact marker. Keep square brackets structural so malformed ``[SILENT``
    does not become ``SILENT``.
    """
    start = 0
    end = len(text)
    while start < end and text[start] not in "[]" and unicodedata.category(text[start]).startswith("P"):
        start += 1
    while end > start and text[end - 1] not in "[]" and unicodedata.category(text[end - 1]).startswith("P"):
        end -= 1
    return text[start:end].strip()


def _canonical_silence_candidates(text: str) -> tuple[str, ...]:
    """Return all canonical forms of *text* to check against the marker set."""
    exact = _canonical_silence_candidate(text)
    stripped = _strip_edge_silence_punctuation(text.strip())
    if stripped == text.strip():
        return (exact,)
    fallback = _canonical_silence_candidate(stripped)
    return (exact, fallback)


def is_intentional_silence_response(response: Any) -> bool:
    """Return True only when *response* is exactly a silence marker.

    Substantive prose that merely mentions ``NO_REPLY`` or ``[SILENT]``
    must be delivered normally. A blank response is also not silence;
    blank output is handled by the empty-response failure path.
    """
    if not isinstance(response, str):
        return False
    stripped = response.strip()
    if not stripped:
        return False
    # Responses longer than 64 chars are definitely not silence markers.
    if len(stripped) > 64:
        return False
    return any(
        candidate in LIVE_GATEWAY_SILENT_MARKERS
        for candidate in _canonical_silence_candidates(stripped)
    )


def is_intentional_silence_agent_result(
    agent_result: dict | None,
    response: Any,
) -> bool:
    """Silence markers suppress delivery only for successful agent turns.

    A failed agent turn that happened to emit a silence marker before
    crashing should NOT be silenced — the user needs to see the error.
    """
    if not isinstance(agent_result, dict):
        return False
    if agent_result.get("failed"):
        return False
    return is_intentional_silence_response(response)


def is_partial_silence_marker(text: Any) -> bool:
    """Return True while *text* could still resolve to a silence marker.

    The streaming path accumulates the reply delta-by-delta and must
    decide, before the whole response is known, whether to show what it
    has so far. A buffer whose canonical form is a non-empty *prefix*
    of a silence marker (e.g. ``"NO"`` on the way to ``"NO_REPLY"``)
    is held back so a raw marker is never edited onto the screen and
    then belatedly retracted.

    Anything that has already diverged from every marker (ordinary
    prose) — and anything longer than the marker cap — returns False so
    normal streaming resumes immediately.
    """
    if not isinstance(text, str):
        return False
    stripped = text.strip()
    if not stripped or len(stripped) > 64:
        return False
    for candidate in _canonical_silence_candidates(stripped):
        if candidate and any(
            marker.startswith(candidate) for marker in LIVE_GATEWAY_SILENT_MARKERS
        ):
            return True
    return False


# ---------------------------------------------------------------------------
# Secret redaction — strip API keys / tokens / passwords from responses
# ---------------------------------------------------------------------------


import re

# Patterns for common secret formats that should never be delivered to chat.
_GITHUB_TOKEN_RE = re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{8,}\b")
_OPENAI_KEY_RE = re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")
_ANTHROPIC_KEY_RE = re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b")
_BEARER_TOKEN_RE = re.compile(r"\bBearer\s+[A-Za-z0-9_.\-]{20,}", re.IGNORECASE)
_AWS_KEY_RE = re.compile(r"\bAKIA[A-Z0-9]{16}\b")
_PASSWORD_ASSIGNMENT_RE = re.compile(
    r"(?i)(password|passwd|pwd|secret|token|api_key|apikey)\s*[=:]\s*\S+"
)


def redact_secrets(text: str) -> str:
    """Redact API keys, tokens, and passwords from *text*.

    Replaces matches with ``[REDACTED]``. Applied to every gateway
    response before delivery so the agent never leaks secrets into chat.
    """
    if not text or not isinstance(text, str):
        return text or ""

    result = text
    result = _GITHUB_TOKEN_RE.sub("[REDACTED]", result)
    result = _OPENAI_KEY_RE.sub("[REDACTED]", result)
    result = _ANTHROPIC_KEY_RE.sub("[REDACTED]", result)
    result = _BEARER_TOKEN_RE.sub("Bearer [REDACTED]", result)
    result = _AWS_KEY_RE.sub("[REDACTED]", result)
    result = _PASSWORD_ASSIGNMENT_RE.sub(
        lambda m: m.group(0).split("=")[0] + "= [REDACTED]"
        if "=" in m.group(0)
        else m.group(0).split(":")[0] + ": [REDACTED]",
        result,
    )
    return result


__all__ = [
    "LIVE_GATEWAY_SILENT_MARKERS",
    "SILENT_REPLY_TOKEN",
    "is_intentional_silence_agent_result",
    "is_intentional_silence_response",
    "is_partial_silence_marker",
    "redact_secrets",
]
