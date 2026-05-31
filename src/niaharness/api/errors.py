"""Comprehensive API error classification for NiaHarness.

Ported from OpenClaude's openaiErrorClassification.ts and errors.ts
with support for OpenAI-compatible providers.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Optional


class OpenAICategory(Enum):
    """Classification of OpenAI-compatible API failures."""

    CONNECTION_REFUSED = "connection_refused"
    LOCALHOST_RESOLUTION_FAILED = "localhost_resolution_failed"
    REQUEST_TIMEOUT = "request_timeout"
    NETWORK_ERROR = "network_error"
    AUTH_INVALID = "auth_invalid"
    RATE_LIMITED = "rate_limited"
    MODEL_NOT_FOUND = "model_not_found"
    ENDPOINT_NOT_FOUND = "endpoint_not_found"
    CONTEXT_OVERFLOW = "context_overflow"
    TOOL_CALL_INCOMPATIBLE = "tool_call_incompatible"
    MALFORMED_PROVIDER_RESPONSE = "malformed_provider_response"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    UNKNOWN = "unknown"


LOCALHOST_HOSTNAMES = frozenset({"localhost", "127.0.0.1", "::1"})


def _get_error_code(error: BaseException) -> Optional[str]:
    """Extract error code from nested exception chain."""
    current: BaseException | None = error
    for _ in range(5):
        if current is None:
            break
        code = getattr(current, "code", None)
        if isinstance(code, str):
            return code
        cause = getattr(current, "__cause__", None)
        if cause is current or cause is None:
            break
        current = cause
    return None


def _get_hostname(url: str) -> Optional[str]:
    """Extract hostname from URL string."""
    try:
        from urllib.parse import urlparse
        return urlparse(url).hostname.lower()
    except Exception:
        return None


def is_localhost_like_host(host: str | None) -> bool:
    """Check if a host string refers to localhost."""
    if not host:
        return False
    hostname = host.lower()
    if hostname in LOCALHOST_HOSTNAMES:
        return True
    return bool(re.match(r"^127\.", hostname))


def _is_context_overflow_message(body: str) -> bool:
    lower = body.lower()
    return any(
        phrase in lower
        for phrase in [
            "too many tokens",
            "request too large",
            "context length",
            "maximum context",
            "input length",
            "payload too large",
            "prompt is too long",
        ]
    )


def _is_tool_compatibility_message(body: str) -> bool:
    lower = body.lower()
    return any(
        phrase in lower
        for phrase in [
            "tool_calls",
            "tool_call",
            "tool_use",
            "tool_result",
            "function calling",
            "function call",
        ]
    )


def _is_malformed_provider_response(body: str) -> bool:
    lower = body.lower()
    return any(
        phrase in lower
        for phrase in [
            "<!doctype html",
            "<html",
            "invalid json",
            "malformed",
            "unexpected token",
            "cannot parse",
            "not valid json",
        ]
    )


def _is_model_not_found_message(body: str) -> bool:
    lower = body.lower()
    if "model" not in lower:
        return False
    return any(
        phrase in lower
        for phrase in [
            "not found",
            "does not exist",
            "unknown model",
            "unavailable model",
        ]
    )


def format_category_marker(category: OpenAICategory, host: Optional[str] = None) -> str:
    """Format an OpenAI category marker string."""
    if host and re.match(r"^[A-Za-z0-9.\-:]+$", host):
        return f"[openai_category={category.value},host={host}]"
    return f"[openai_category={category.value}]"


def extract_category_marker(message: str) -> Optional[OpenAICategory]:
    """Extract OpenAI category marker from error message."""
    match = re.search(r"\[openai_category=([a-z_]+)(?:,host=[^\]]+)?\]", message)
    if not match:
        return None
    try:
        return OpenAICategory(match.group(1))
    except ValueError:
        return None


def extract_category_host(message: str) -> Optional[str]:
    """Extract host from OpenAI category marker."""
    match = re.search(r"\[openai_category=[a-z_]+,host=([A-Za-z0-9.\-:]+)\]", message)
    return match.group(1) if match else None


def build_error_message(base_message: str, category: OpenAICategory, hint: Optional[str] = None, url: Optional[str] = None) -> str:
    """Build an error message with category marker and optional hint."""
    host = _get_hostname(url) if url else None
    marker = format_category_marker(category, host)
    hint_str = f" Hint: {hint}" if hint else ""
    return f"{base_message} {marker}{hint_str}"


def classify_network_error(error: BaseException, url: str) -> tuple[OpenAICategory, bool, str]:
    """Classify a network-level error.

    Returns:
        Tuple of (category, retryable, message)
    """
    message = str(error)
    lower_message = message.lower()
    code = _get_error_code(error)
    host = _get_hostname(url)
    is_local = is_localhost_like_host(host)

    if code in {"ECONNREFUSED", "ECONNRESET"} or "connection refused" in lower_message:
        if is_local:
            return (
                OpenAICategory.LOCALHOST_RESOLUTION_FAILED,
                True,
                "Could not connect to the local provider. Ensure the server is running.",
            )
        return (
            OpenAICategory.CONNECTION_REFUSED,
            True,
            f"Could not connect to provider at {host}. Verify the endpoint is reachable.",
        )

    if code in {"ECONNRESET", "EPIPE"}:
        return (
            OpenAICategory.NETWORK_ERROR,
            True,
            "Connection was reset. The server may have closed the connection.",
        )

    if "timeout" in lower_message or code == "ETIMEDOUT":
        return (
            OpenAICategory.REQUEST_TIMEOUT,
            True,
            "Request timed out. The provider may be overloaded.",
        )

    if "enotfound" in lower_message or code == "ENOTFOUND":
        if is_local:
            return (
                OpenAICategory.LOCALHOST_RESOLUTION_FAILED,
                True,
                "Could not resolve local hostname. Check your network configuration.",
            )
        return (
            OpenAICategory.NETWORK_ERROR,
            True,
            f"Could not resolve hostname {host}.",
        )

    return (
        OpenAICategory.NETWORK_ERROR,
        True,
        f"Network error: {message[:200]}",
    )


def classify_http_error(status_code: int, body: str, url: Optional[str] = None) -> tuple[OpenAICategory, bool, str]:
    """Classify an HTTP error response.

    Returns:
        Tuple of (category, retryable, message)
    """
    host = _get_hostname(url) if url else None

    if status_code == 401 or status_code == 403:
        return (
            OpenAICategory.AUTH_INVALID,
            False,
            "Authentication failed. Verify your API key and permissions.",
        )

    if status_code == 404:
        if is_localhost_like_host(host):
            return (
                OpenAICategory.ENDPOINT_NOT_FOUND,
                False,
                "Provider endpoint not found. Confirm the base URL targets an OpenAI-compatible /v1 endpoint.",
            )
        return (
            OpenAICategory.ENDPOINT_NOT_FOUND,
            False,
            f"Endpoint not found at {host}. Verify the base URL and model availability.",
        )

    if status_code == 429:
        return (
            OpenAICategory.RATE_LIMITED,
            True,
            "Rate limit reached. Retry after a short delay.",
        )

    if status_code == 402:
        return (
            OpenAICategory.RATE_LIMITED,
            False,
            "Credits exhausted or max_tokens exceeds affordable limit.",
        )

    if status_code in {500, 502, 503}:
        return (
            OpenAICategory.PROVIDER_UNAVAILABLE,
            True,
            f"Provider temporarily unavailable (HTTP {status_code}).",
        )

    if status_code == 400:
        if _is_context_overflow_message(body):
            return (
                OpenAICategory.CONTEXT_OVERFLOW,
                False,
                "Request exceeds provider context limit.",
            )
        if _is_tool_compatibility_message(body):
            return (
                OpenAICategory.TOOL_CALL_INCOMPATIBLE,
                False,
                "Provider rejected tool-calling payload.",
            )
        if _is_model_not_found_message(body):
            return (
                OpenAICategory.MODEL_NOT_FOUND,
                False,
                "Model not found on this provider.",
            )
        if _is_malformed_provider_response(body):
            return (
                OpenAICategory.MALFORMED_PROVIDER_RESPONSE,
                False,
                "Provider returned a malformed response.",
            )

    return (
        OpenAICategory.UNKNOWN,
        status_code >= 500,
        f"HTTP {status_code}: {body[:200]}",
    )


# ---------------------------------------------------------------------------
# Base error types
# ---------------------------------------------------------------------------

class NiaHarnessApiError(RuntimeError):
    """Base class for upstream API failures."""


class AuthenticationFailure(NiaHarnessApiError):
    """Raised when the upstream service rejects the provided credentials."""


class RateLimitFailure(NiaHarnessApiError):
    """Raised when the upstream service rejects the request due to rate limits."""


class RequestFailure(NiaHarnessApiError):
    """Raised for generic request or transport failures."""


class ContextOverflowFailure(NiaHarnessApiError):
    """Raised when the request exceeds the provider context limit."""


class ModelNotFoundFailure(NiaHarnessApiError):
    """Raised when the requested model is not available."""


class ProviderUnavailableFailure(NiaHarnessApiError):
    """Raised when the provider is temporarily unavailable."""


class ConnectionFailure(NiaHarnessApiError):
    """Raised when unable to connect to the provider."""


class ToolCallIncompatibleFailure(NiaHarnessApiError):
    """Raised when the provider rejects tool-calling payloads."""


class RetryableError(NiaHarnessApiError):
    """Wrapper indicating an error is retryable."""

    def __init__(self, original: BaseException, category: OpenAICategory, message: str):
        self.original = original
        self.category = category
        super().__init__(message)


def translate_api_error(
    status_code: int,
    message: str,
    url: Optional[str] = None,
) -> NiaHarnessApiError:
    """Translate an HTTP status code to the appropriate error type."""
    category, _, _ = classify_http_error(status_code, message, url)

    if category == OpenAICategory.AUTH_INVALID:
        return AuthenticationFailure(message)
    if category == OpenAICategory.RATE_LIMITED:
        return RateLimitFailure(message)
    if category == OpenAICategory.CONTEXT_OVERFLOW:
        return ContextOverflowFailure(message)
    if category == OpenAICategory.MODEL_NOT_FOUND:
        return ModelNotFoundFailure(message)
    if category in {OpenAICategory.PROVIDER_UNAVAILABLE, OpenAICategory.CONNECTION_REFUSED, OpenAICategory.LOCALHOST_RESOLUTION_FAILED}:
        return ProviderUnavailableFailure(message)
    if category == OpenAICategory.TOOL_CALL_INCOMPATIBLE:
        return ToolCallIncompatibleFailure(message)

    return RequestFailure(message)
