"""Anthropic API client wrapper with comprehensive retry logic.

Enhanced with retry patterns from OpenClaude's withRetry.ts including:
- Exponential backoff with jitter
- Retry-After header support
- Consecutive error tracking
- 529 overload handling
- Provider-specific error classification
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Protocol

from anthropic import APIError, APIStatusError, AsyncAnthropic

from niaharness.api.errors import (
    AuthenticationFailure,
    ConnectionFailure,
    ContextOverflowFailure,
    ModelNotFoundFailure,
    NiaHarnessApiError,
    ProviderUnavailableFailure,
    RateLimitFailure,
    RequestFailure,
    RetryableError,
    ToolCallIncompatibleFailure,
    classify_http_error,
    classify_network_error,
)
from niaharness.api.usage import UsageSnapshot
from niaharness.engine.messages import ConversationMessage, assistant_message_from_api

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Retry configuration (ported from OpenClaude withRetry.ts)
# ---------------------------------------------------------------------------

MAX_RETRIES = 10
MAX_CONFIGURABLE_RETRIES = 100
BASE_DELAY_MS = 500  # 500ms base delay
MAX_RETRY_DELAY_BASE_MS = 60_000  # 60 seconds max delay
FLOOR_OUTPUT_TOKENS = 3000
MAX_529_RETRIES = 3
DEFAULT_RETRY_DELAY_MS = 500

# Retryable status codes
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 529})

# Persistent retry config for unattended sessions
PERSISTENT_MAX_BACKOFF_MS = 5 * 60 * 1000  # 5 minutes
PERSISTENT_RESET_CAP_MS = 6 * 60 * 60 * 1000  # 6 hours
HEARTBEAT_INTERVAL_MS = 30_000  # 30 seconds


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ApiMessageRequest:
    """Input parameters for a model invocation."""

    model: str
    messages: list[ConversationMessage]
    system_prompt: str | None = None
    max_tokens: int = 4096
    tools: list[dict[str, Any]] = field(default_factory=list)
    temperature: float | None = None
    top_p: float | None = None
    reasoning_effort: str | None = None
    stream: bool = True


@dataclass(frozen=True)
class ApiTextDeltaEvent:
    """Incremental text produced by the model."""

    text: str


@dataclass(frozen=True)
class ApiThinkingDeltaEvent:
    """Incremental thinking/reasoning produced by the model."""

    thinking: str


@dataclass(frozen=True)
class ApiToolCallDeltaEvent:
    """Partial tool-call argument JSON produced by the model."""

    partial_json: str


@dataclass(frozen=True)
class ApiMessageCompleteEvent:
    """Terminal event containing the full assistant message."""

    message: ConversationMessage
    usage: UsageSnapshot
    stop_reason: str | None = None


@dataclass(frozen=True)
class ApiRetryEvent:
    """Event emitted when a retry is attempted."""

    attempt: int
    max_retries: int
    delay_seconds: float
    error: str
    status_code: int | None = None


ApiStreamEvent = (
    ApiTextDeltaEvent
    | ApiThinkingDeltaEvent
    | ApiToolCallDeltaEvent
    | ApiMessageCompleteEvent
    | ApiRetryEvent
)


class SupportsStreamingMessages(Protocol):
    """Protocol used by the query engine in tests and production."""

    async def stream_message(self, request: ApiMessageRequest) -> AsyncIterator[ApiStreamEvent]:
        """Yield streamed events for the request."""


# ---------------------------------------------------------------------------
# Retry utilities (ported from OpenClaude withRetry.ts)
# ---------------------------------------------------------------------------

def get_retry_after_ms(exc: Exception) -> float | None:
    """Extract Retry-After header value in milliseconds."""
    headers = getattr(exc, "headers", None)
    if headers is None:
        return None
    retry_after = None
    if hasattr(headers, "get"):
        retry_after = headers.get("retry-after")
    if retry_after is None:
        return None
    try:
        seconds = int(retry_after)
        return seconds * 1000.0
    except (ValueError, TypeError):
        return None


def get_retry_delay(
    attempt: int,
    retry_after_header: str | None = None,
    max_delay_ms: float = MAX_RETRY_DELAY_BASE_MS,
) -> float:
    """Calculate retry delay with exponential backoff and jitter.

    Ported from OpenClaude's getRetryDelay function.
    """
    if retry_after_header:
        try:
            seconds = int(retry_after_header)
            if seconds > 0:
                return seconds * 1000.0
        except (ValueError, TypeError):
            pass

    base_delay = min(
        BASE_DELAY_MS * (2 ** (attempt - 1)),
        max_delay_ms,
    )
    jitter = random.random() * 0.25 * base_delay
    return base_delay + jitter


def _is_retryable(exc: Exception) -> bool:
    """Check if an exception is retryable."""
    if isinstance(exc, APIStatusError):
        return exc.status_code in RETRYABLE_STATUS_CODES
    if isinstance(exc, APIError):
        return True  # Network errors are retryable
    if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return True
    if isinstance(exc, RetryableError):
        return True
    return False


def _is_quota_exhausted(exc: Exception) -> bool:
    """Check if error indicates quota exhaustion."""
    msg = str(exc).lower()
    status = getattr(exc, "status_code", None)
    return status == 429 and ("limit: 0" in msg or "exceeded your current quota" in msg)


def _is_transient_capacity_error(exc: Exception) -> bool:
    """Check if error is a transient capacity issue (429/529)."""
    if isinstance(exc, APIStatusError):
        return exc.status_code in {429, 529}
    return False


def _is_529_error(exc: Exception) -> bool:
    """Check if error is a 529 overload."""
    return isinstance(exc, APIStatusError) and exc.status_code == 529


# ---------------------------------------------------------------------------
# Error classification (ported from OpenClaude openaiErrorClassification.ts)
# ---------------------------------------------------------------------------

def _translate_api_error(exc: APIError) -> NiaHarnessApiError:
    """Translate Anthropic SDK errors to NiaHarness errors."""
    name = exc.__class__.__name__
    status = getattr(exc, "status_code", None)

    if name in {"AuthenticationError", "PermissionDeniedError"}:
        return AuthenticationFailure(str(exc))
    if name == "RateLimitError" or status == 429:
        return RateLimitFailure(str(exc))
    if status == 400:
        msg = str(exc).lower()
        if "too many tokens" in msg or "context length" in msg or "prompt is too long" in msg:
            return ContextOverflowFailure(str(exc))
        if "model" in msg and ("not found" in msg or "does not exist" in msg):
            return ModelNotFoundFailure(str(exc))
        if "tool" in msg:
            return ToolCallIncompatibleFailure(str(exc))
    if status in {502, 503}:
        return ProviderUnavailableFailure(str(exc))
    if status in {500,}:
        return ProviderUnavailableFailure(str(exc))

    return RequestFailure(str(exc))


def _classify_error(exc: Exception) -> tuple[str, bool]:
    """Classify error into category and retryability.

    Returns:
        Tuple of (category_name, is_retryable)
    """
    if isinstance(exc, APIStatusError):
        status = exc.status_code
        msg = str(exc)

        if status == 401 or status == 403:
            return "auth_invalid", False
        if status == 404:
            return "endpoint_not_found", False
        if status == 429:
            return "rate_limited", True
        if status == 529:
            return "provider_unavailable", True
        if status in {500, 502, 503}:
            return "provider_unavailable", True
        if status == 400:
            msg_lower = msg.lower()
            if "too many tokens" in msg_lower or "context length" in msg_lower:
                return "context_overflow", False
            if "model" in msg_lower and "not found" in msg_lower:
                return "model_not_found", False
            if "tool" in msg_lower:
                return "tool_call_incompatible", False
        return "unknown", status >= 500

    if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return "connection_refused", True

    return "unknown", False


# ---------------------------------------------------------------------------
# Client implementations
# ---------------------------------------------------------------------------

class AnthropicApiClient:
    """Thin wrapper around the Anthropic async SDK with comprehensive retry logic.

    Supports:
    - Exponential backoff with jitter
    - Retry-After header parsing
    - Consecutive 529 error tracking with fallback
    - Provider-specific error classification
    """

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str | None = None,
        max_retries: int = MAX_RETRIES,
    ) -> None:
        # Delegate client construction to the transport so the client gets
        # the right auth mode (api_key vs OAuth bearer vs third-party bearer)
        # and the correct ``anthropic-beta`` header set per endpoint.  This
        # enables prompt caching, extended thinking, and the 1M-context beta
        # without callers needing to configure them manually.
        from niaharness.providers.anthropic_transport import build_anthropic_client

        self._client = build_anthropic_client(
            api_key=api_key,
            base_url=base_url,
            timeout=900.0,
        )
        self._max_retries = min(max_retries, MAX_CONFIGURABLE_RETRIES)

    async def stream_message(self, request: ApiMessageRequest) -> AsyncIterator[ApiStreamEvent]:
        """Yield text deltas and the final assistant message with retry on transient errors."""
        last_error: Exception | None = None
        consecutive_529 = 0

        for attempt in range(self._max_retries + 1):
            try:
                async for event in self._stream_once(request):
                    yield event
                return  # Success
            except NiaHarnessApiError:
                raise  # Auth errors are not retried
            except Exception as exc:
                last_error = exc
                category, retryable = _classify_error(exc)

                # Track consecutive 529 errors
                if _is_529_error(exc):
                    consecutive_529 += 1
                    if consecutive_529 >= MAX_529_RETRIES:
                        log.error(
                            "Consecutive 529 errors (%d) exceeded limit, giving up",
                            consecutive_529,
                        )
                        raise _translate_api_error(exc) from exc
                else:
                    consecutive_529 = 0

                # Check if we should retry
                if attempt >= self._max_retries or not retryable:
                    if isinstance(exc, APIError):
                        raise _translate_api_error(exc) from exc
                    raise RequestFailure(str(exc)) from exc

                # Calculate delay
                retry_after_ms = get_retry_after_ms(exc)
                if retry_after_ms is not None:
                    delay_s = min(retry_after_ms / 1000.0, 60.0)
                else:
                    delay_ms = get_retry_delay(attempt + 1)
                    delay_s = delay_ms / 1000.0

                status = getattr(exc, "status_code", "?")
                log.warning(
                    "API request failed (attempt %d/%d, status=%s, category=%s), retrying in %.1fs: %s",
                    attempt + 1, self._max_retries + 1, status, category, delay_s, exc,
                )

                yield ApiRetryEvent(
                    attempt=attempt + 1,
                    max_retries=self._max_retries,
                    delay_seconds=delay_s,
                    error=str(exc),
                    status_code=status if isinstance(status, int) else None,
                )

                await asyncio.sleep(delay_s)

        if last_error is not None:
            if isinstance(last_error, APIError):
                raise _translate_api_error(last_error) from last_error
            raise RequestFailure(str(last_error)) from last_error

    async def _stream_once(self, request: ApiMessageRequest) -> AsyncIterator[ApiStreamEvent]:
        """Single attempt at streaming a message.

        Routes through :func:`niaharness.providers.anthropic_transport.build_anthropic_kwargs`
        so the request benefits from:

          - **Message conversion + repair** — NIA ``ConversationMessage`` →
            Anthropic wire format, with orphan-strip / role-merge /
            thinking-signature management.
          - **Prompt caching** — ``cache_control`` markers applied to the
            system prompt, last tool definition, and last assistant block
            (~10x cost reduction on long sessions).
          - **Model capability detection** — adaptive thinking config,
            sampling-param strip, ``max_tokens`` resolution from the model
            ceiling when not provided.
          - **Tool-schema sanitization** — nullable-union collapse +
            top-level ``oneOf``/``allOf``/``anyOf`` strip (Anthropic rejects
            them with 400).
        """
        from niaharness.providers.anthropic_transport import build_anthropic_kwargs

        # Base URL is read from the underlying SDK client (set at construction).
        base_url = getattr(self._client, "base_url", None)
        base_url_str = str(base_url) if base_url else None

        # Build the full kwargs via the transport.  We pass the raw
        # ConversationMessage objects — the transport handles conversion +
        # repair + caching.
        messages = list(request.messages)
        params = build_anthropic_kwargs(
            model=request.model,
            messages=messages,
            system_prompt=request.system_prompt,
            tools=request.tools or None,
            max_tokens=request.max_tokens,
            reasoning_effort=request.reasoning_effort,
            temperature=request.temperature,
            top_p=request.top_p,
            base_url=base_url_str,
            enable_caching=True,
        )

        # Strip internal helper keys the SDK doesn't accept.
        sdk_params = {k: v for k, v in params.items() if not k.startswith("_")}
        extra_headers = params.get("_extra_headers")
        if extra_headers:
            sdk_params["extra_headers"] = extra_headers

        try:
            async with self._client.messages.stream(**sdk_params) as stream:
                async for event in stream:
                    event_type = getattr(event, "type", None)

                    # Stream text deltas.
                    if event_type == "content_block_delta":
                        delta = getattr(event, "delta", None)
                        delta_type = getattr(delta, "type", None)

                        if delta_type == "text_delta":
                            text = getattr(delta, "text", "")
                            if text:
                                yield ApiTextDeltaEvent(text=text)

                        elif delta_type == "thinking_delta":
                            # Stream thinking/reasoning deltas.
                            thinking = getattr(delta, "thinking", "")
                            if thinking:
                                yield ApiThinkingDeltaEvent(thinking=thinking)

                        elif delta_type == "input_json_delta":
                            # Stream partial tool-call argument JSON.
                            partial_json = getattr(delta, "partial_json", "")
                            if partial_json:
                                yield ApiToolCallDeltaEvent(partial_json=partial_json)

                    # Also forward retry events so the UI can show "retrying...".
                    elif event_type == "message_delta":
                        # Contains stop_reason, usage updates.
                        pass

                final_message = await stream.get_final_message()
        except APIError as exc:
            if isinstance(exc, APIStatusError) and exc.status_code in RETRYABLE_STATUS_CODES:
                raise  # Let retry logic handle it
            raise _translate_api_error(exc) from exc

        usage = getattr(final_message, "usage", None)
        yield ApiMessageCompleteEvent(
            message=assistant_message_from_api(final_message),
            usage=UsageSnapshot(
                input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
                output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            ),
            stop_reason=getattr(final_message, "stop_reason", None),
        )
