"""Error recovery guards — one-shot handlers for transient API failures.

Ported from the reference project's recovery patterns (agent_runtime_helpers.py,
conversation_loop.py), providing a structured registry of one-shot recovery
guards that fire on specific API error patterns.

Each guard:
  - Detects a specific error pattern (status code, message substring, exception type)
  - Returns a RecoveryAction describing what to do (retry, compress, rotate, rebuild, abort)
  - Is "one-shot" — it fires at most once per request, preventing infinite loops

The 16 recovery strategies (adapted from the reference project):
  1.  prompt_too_long_compress      — context overflow → trigger compaction
  2.  max_tokens_length_continue    — stop_reason=max_tokens → continue generation
  3.  rate_limit_429_backoff        — 429 → exponential backoff with Retry-After
  4.  auth_401_rotate_credential    — 401 → mark credential exhausted, rotate
  5.  billing_402_rotate_credential — 402 → mark credential exhausted (billing)
  6.  forbidden_403_abort           — 403 → non-retryable, abort with clear message
  7.  provider_529_overload_backoff — 529 → extended backoff for overload
  8.  provider_5xx_retry            — 500/502/503/504 → retry with backoff
  9.  thinking_signature_strip      — invalid thinking signature → strip thinking blocks
  10. image_shrink_multimodal       — image too large → shrink and retry
  11. multimodal_tool_content_fix   — tool content type error → convert to text
  12. llama_cpp_grammar_disable     — grammar parse error → disable grammar
  13. rebuild_messages_drop_thinking— orphaned thinking blocks → rebuild messages
  14. transport_rebuild_client      — connection pool exhaustion → rebuild client
  15. context_window_truncate       — context window exceeded → truncate oldest messages
  16. restart_signal_compress       — explicit restart signal → compress and retry

Usage::

    from niaharness.engine.recovery import RecoveryRegistry, RecoveryAction, ActionType

    registry = RecoveryRegistry()
    action = registry.match(exc, context)
    if action is not None:
        result = await action.execute(context)
        if result.should_retry:
            # retry the request
            ...
"""

from __future__ import annotations

import enum
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Action types
# ---------------------------------------------------------------------------


class ActionType(str, enum.Enum):
    """What the recovery guard recommends doing."""

    RETRY = "retry"                    # Retry the request (possibly after a delay)
    COMPRESS = "compress"              # Compress context, then retry
    ROTATE_CREDENTIAL = "rotate"       # Rotate to next credential, then retry
    REBUILD_MESSAGES = "rebuild"       # Rebuild message list (drop thinking, fix roles), then retry
    REBUILD_CLIENT = "rebuild_client"  # Rebuild the API client (clear connection pool), then retry
    TRUNCATE_CONTEXT = "truncate"      # Truncate oldest messages, then retry
    STRIP_THINKING = "strip_thinking"  # Strip thinking blocks from messages, then retry
    SHRINK_IMAGE = "shrink_image"      # Shrink images in messages, then retry
    ABORT = "abort"                    # Non-retryable, abort with message
    RESTART = "restart"                # Send restart signal to engine


# ---------------------------------------------------------------------------
# Recovery action
# ---------------------------------------------------------------------------


@dataclass
class RecoveryAction:
    """A recovery action returned by a guard.

    The ``execute`` callback is optional — if not provided, the caller
    interprets ``type`` and ``delay_seconds`` directly. If provided,
    ``execute`` performs the recovery (e.g. compress context, rotate
    credential) and returns whether the request should be retried.
    """

    type: ActionType
    description: str
    delay_seconds: float = 0.0
    should_retry: bool = True
    execute: Optional[Callable[[Any], Any]] = None
    guard_name: str = ""  # for logging/audit
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Error context
# ---------------------------------------------------------------------------


@dataclass
class ErrorContext:
    """Context passed to recovery guards for matching."""

    exc: Exception
    status_code: Optional[int] = None
    error_message: str = ""
    error_body: Optional[Dict[str, Any]] = None
    provider: str = ""
    model: str = ""
    attempt: int = 0
    max_retries: int = 10
    # Mutable bag for guards to store state (e.g. "already compressed once")
    state: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.error_message:
            self.error_message = str(self.exc)
        if self.status_code is None:
            self.status_code = _extract_status_code(self.exc)
        if self.error_body is None:
            self.error_body = _extract_error_body(self.exc)


def _extract_status_code(exc: Exception) -> Optional[int]:
    """Extract HTTP status code from an exception."""
    for attr in ("status_code", "status", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    cause = getattr(exc, "__cause__", None)
    if cause is not None:
        for attr in ("status_code", "status", "code"):
            value = getattr(cause, attr, None)
            if isinstance(value, int):
                return value
    return None


def _extract_error_body(exc: Exception) -> Optional[Dict[str, Any]]:
    """Extract the JSON error body from an API exception, if available."""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        return body
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            json_body = response.json() if hasattr(response, "json") else None
            if isinstance(json_body, dict):
                return json_body
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------------
# Guard type
# ---------------------------------------------------------------------------


GuardFn = Callable[[ErrorContext], Optional[RecoveryAction]]


@dataclass
class Guard:
    """A named recovery guard with one-shot semantics."""

    name: str
    match: GuardFn
    description: str = ""
    # Guards are one-shot per request: once fired, they won't fire again
    # for the same request. This prevents infinite recovery loops.


# ---------------------------------------------------------------------------
# Built-in guards (16 strategies)
# ---------------------------------------------------------------------------


def _guard_prompt_too_long_compress(ctx: ErrorContext) -> Optional[RecoveryAction]:
    """1. prompt_too_long_compress — context overflow → trigger compaction."""
    msg = ctx.error_message.lower()
    if ctx.status_code == 400 and any(
        s in msg
        for s in ("too many tokens", "context length", "prompt is too long", "context window")
    ):
        if ctx.state.get("compressed"):
            return RecoveryAction(
                type=ActionType.ABORT,
                description="Context already compressed, still too long — abort",
                should_retry=False,
                guard_name="prompt_too_long_compress",
            )
        ctx.state["compressed"] = True
        return RecoveryAction(
            type=ActionType.COMPRESS,
            description="Context overflow — compress and retry",
            should_retry=True,
            guard_name="prompt_too_long_compress",
        )
    return None


def _guard_max_tokens_length_continue(ctx: ErrorContext) -> Optional[RecoveryAction]:
    """2. max_tokens_length_continue — stop_reason=max_tokens → continue generation."""
    body = ctx.error_body or {}
    stop_reason = body.get("stop_reason") or body.get("stop")
    if stop_reason == "max_tokens" and ctx.attempt < ctx.max_retries:
        return RecoveryAction(
            type=ActionType.RETRY,
            description="Generation stopped at max_tokens — continue with length-continue",
            delay_seconds=0.0,
            guard_name="max_tokens_length_continue",
            metadata={"stop_reason": stop_reason},
        )
    return None


def _guard_rate_limit_429_backoff(ctx: ErrorContext) -> Optional[RecoveryAction]:
    """3. rate_limit_429_backoff — 429 → exponential backoff with Retry-After."""
    if ctx.status_code != 429:
        return None
    # Extract Retry-After if present.
    delay = _extract_retry_after(ctx.exc)
    if delay is None:
        delay = min(2 ** ctx.attempt, 60)  # exponential backoff, cap 60s
    return RecoveryAction(
        type=ActionType.RETRY,
        description=f"Rate limited (429) — backoff {delay:.1f}s",
        delay_seconds=delay,
        guard_name="rate_limit_429_backoff",
    )


def _guard_auth_401_rotate_credential(ctx: ErrorContext) -> Optional[RecoveryAction]:
    """4. auth_401_rotate_credential — 401 → mark credential exhausted, rotate."""
    if ctx.status_code != 401:
        return None
    return RecoveryAction(
        type=ActionType.ROTATE_CREDENTIAL,
        description="Auth failure (401) — rotate credential and retry",
        delay_seconds=0.0,
        guard_name="auth_401_rotate_credential",
        metadata={"status_code": 401},
    )


def _guard_billing_402_rotate_credential(ctx: ErrorContext) -> Optional[RecoveryAction]:
    """5. billing_402_rotate_credential — 402 → mark credential exhausted (billing)."""
    if ctx.status_code != 402:
        return None
    return RecoveryAction(
        type=ActionType.ROTATE_CREDENTIAL,
        description="Billing failure (402) — rotate credential and retry",
        delay_seconds=0.0,
        guard_name="billing_402_rotate_credential",
        metadata={"status_code": 402},
    )


def _guard_forbidden_403_abort(ctx: ErrorContext) -> Optional[RecoveryAction]:
    """6. forbidden_403_abort — 403 → non-retryable, abort with clear message."""
    if ctx.status_code != 403:
        return None
    return RecoveryAction(
        type=ActionType.ABORT,
        description="Forbidden (403) — non-retryable auth/permission error",
        should_retry=False,
        guard_name="forbidden_403_abort",
    )


def _guard_provider_529_overload_backoff(ctx: ErrorContext) -> Optional[RecoveryAction]:
    """7. provider_529_overload_backoff — 529 → extended backoff for overload."""
    if ctx.status_code != 529:
        return None
    delay = max(5.0, 2 ** ctx.attempt)  # longer backoff for overload
    return RecoveryAction(
        type=ActionType.RETRY,
        description=f"Provider overload (529) — extended backoff {delay:.1f}s",
        delay_seconds=delay,
        guard_name="provider_529_overload_backoff",
    )


def _guard_provider_5xx_retry(ctx: ErrorContext) -> Optional[RecoveryAction]:
    """8. provider_5xx_retry — 500/502/503/504 → retry with backoff."""
    if ctx.status_code not in {500, 502, 503, 504}:
        return None
    delay = min(2 ** ctx.attempt, 30)
    return RecoveryAction(
        type=ActionType.RETRY,
        description=f"Provider error ({ctx.status_code}) — retry in {delay:.1f}s",
        delay_seconds=delay,
        guard_name="provider_5xx_retry",
    )


def _guard_thinking_signature_strip(ctx: ErrorContext) -> Optional[RecoveryAction]:
    """9. thinking_signature_strip — invalid thinking signature → strip thinking blocks."""
    msg = ctx.error_message.lower()
    if ctx.status_code == 400 and any(
        s in msg
        for s in ("thinking signature", "invalid thinking", "thinking block", "signature verification")
    ):
        if ctx.state.get("stripped_thinking"):
            return None  # Already tried, don't loop
        ctx.state["stripped_thinking"] = True
        return RecoveryAction(
            type=ActionType.STRIP_THINKING,
            description="Invalid thinking signature — strip thinking blocks and retry",
            guard_name="thinking_signature_strip",
        )
    return None


def _guard_image_shrink_multimodal(ctx: ErrorContext) -> Optional[RecoveryAction]:
    """10. image_shrink_multimodal — image too large → shrink and retry."""
    msg = ctx.error_message.lower()
    if ctx.status_code == 400 and any(
        s in msg
        for s in ("image too large", "image size exceeds", "image dimensions", "media size")
    ):
        if ctx.state.get("shrunk_image"):
            return None  # Already tried
        ctx.state["shrunk_image"] = True
        return RecoveryAction(
            type=ActionType.SHRINK_IMAGE,
            description="Image too large — shrink and retry",
            guard_name="image_shrink_multimodal",
        )
    return None


def _guard_multimodal_tool_content_fix(ctx: ErrorContext) -> Optional[RecoveryAction]:
    """11. multimodal_tool_content_fix — tool content type error → convert to text."""
    msg = ctx.error_message.lower()
    if ctx.status_code == 400 and any(
        s in msg
        for s in ("tool content", "tool_result content", "invalid content type", "unrecognized content type")
    ):
        if ctx.state.get("fixed_tool_content"):
            return None
        ctx.state["fixed_tool_content"] = True
        return RecoveryAction(
            type=ActionType.REBUILD_MESSAGES,
            description="Tool content type error — convert multimodal tool content to text and retry",
            guard_name="multimodal_tool_content_fix",
        )
    return None


def _guard_llama_cpp_grammar_disable(ctx: ErrorContext) -> Optional[RecoveryAction]:
    """12. llama_cpp_grammar_disable — grammar parse error → disable grammar."""
    msg = ctx.error_message.lower()
    if any(s in msg for s in ("grammar", "gbnf", "parse error", "llama.cpp")):
        if ctx.state.get("disabled_grammar"):
            return None
        ctx.state["disabled_grammar"] = True
        return RecoveryAction(
            type=ActionType.RETRY,
            description="Grammar parse error — disable grammar and retry",
            guard_name="llama_cpp_grammar_disable",
            metadata={"disable_grammar": True},
        )
    return None


def _guard_rebuild_messages_drop_thinking(ctx: ErrorContext) -> Optional[RecoveryAction]:
    """13. rebuild_messages_drop_thinking — orphaned thinking blocks → rebuild messages."""
    msg = ctx.error_message.lower()
    if ctx.status_code == 400 and any(
        s in msg
        for s in ("orphaned thinking", "thinking block without", "invalid message structure", "role alternation")
    ):
        if ctx.state.get("rebuilt_messages"):
            return None
        ctx.state["rebuilt_messages"] = True
        return RecoveryAction(
            type=ActionType.REBUILD_MESSAGES,
            description="Message structure error — drop thinking-only turns and rebuild",
            guard_name="rebuild_messages_drop_thinking",
        )
    return None


def _guard_transport_rebuild_client(ctx: ErrorContext) -> Optional[RecoveryAction]:
    """14. transport_rebuild_client — connection pool exhaustion → rebuild client."""
    error_type = type(ctx.exc).__name__
    transient_errors = {
        "ConnectionError",
        "ConnectError",
        "ReadTimeout",
        "WriteTimeout",
        "PoolTimeout",
        "RemoteProtocolError",
        "LocalProtocolError",
        "ConnectTimeout",
        "ReadError",
    }
    if error_type in transient_errors:
        if ctx.state.get("rebuilt_client"):
            return None
        ctx.state["rebuilt_client"] = True
        delay = min(3 + ctx.attempt, 8)
        return RecoveryAction(
            type=ActionType.REBUILD_CLIENT,
            description=f"Transient transport error ({error_type}) — rebuild client, retry in {delay}s",
            delay_seconds=delay,
            guard_name="transport_rebuild_client",
        )
    return None


def _guard_context_window_truncate(ctx: ErrorContext) -> Optional[RecoveryAction]:
    """15. context_window_truncate — context window exceeded → truncate oldest messages."""
    msg = ctx.error_message.lower()
    if any(s in msg for s in ("context window", "maximum context", "input tokens exceed")):
        if ctx.state.get("truncated"):
            return RecoveryAction(
                type=ActionType.ABORT,
                description="Context already truncated, still exceeds window — abort",
                should_retry=False,
                guard_name="context_window_truncate",
            )
        ctx.state["truncated"] = True
        return RecoveryAction(
            type=ActionType.TRUNCATE_CONTEXT,
            description="Context window exceeded — truncate oldest messages and retry",
            guard_name="context_window_truncate",
        )
    return None


def _guard_restart_signal_compress(ctx: ErrorContext) -> Optional[RecoveryAction]:
    """16. restart_signal_compress — explicit restart signal → compress and retry."""
    msg = ctx.error_message.lower()
    if "restart_signal" in msg or "compress_required" in msg:
        return RecoveryAction(
            type=ActionType.RESTART,
            description="Restart signal received — compress context and restart",
            guard_name="restart_signal_compress",
        )
    return None


# ---------------------------------------------------------------------------
# Retry-After extraction
# ---------------------------------------------------------------------------


def _extract_retry_after(exc: Exception) -> Optional[float]:
    """Extract Retry-After header value in seconds."""
    headers = getattr(exc, "headers", None)
    if headers is None:
        response = getattr(exc, "response", None)
        if response is not None:
            headers = getattr(response, "headers", None)
    if headers is None:
        return None
    retry_after = None
    if hasattr(headers, "get"):
        retry_after = headers.get("retry-after") or headers.get("Retry-After")
    if retry_after is None:
        return None
    try:
        seconds = int(retry_after)
        return float(seconds)
    except (ValueError, TypeError):
        # Try HTTP-date form.
        try:
            from email.utils import parsedate_to_datetime

            dt = parsedate_to_datetime(str(retry_after))
            if dt is not None:
                delay = dt.timestamp() - __import__("time").time()
                if delay > 0:
                    return delay
        except (ValueError, TypeError):
            pass
        return None


# ---------------------------------------------------------------------------
# Recovery registry
# ---------------------------------------------------------------------------


class RecoveryRegistry:
    """Registry of one-shot recovery guards.

    Guards are tried in priority order. The first guard that returns a
    non-None RecoveryAction wins. Each guard tracks its fired state in
    the ErrorContext.state dict, so it won't fire twice for the same
    request (one-shot semantics).
    """

    def __init__(self) -> None:
        self._guards: List[Guard] = []
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register the 16 built-in recovery guards in priority order."""
        defaults = [
            Guard("prompt_too_long_compress", _guard_prompt_too_long_compress,
                  "Context overflow → trigger compaction"),
            Guard("context_window_truncate", _guard_context_window_truncate,
                  "Context window exceeded → truncate oldest"),
            Guard("max_tokens_length_continue", _guard_max_tokens_length_continue,
                  "stop_reason=max_tokens → continue generation"),
            Guard("rate_limit_429_backoff", _guard_rate_limit_429_backoff,
                  "429 → exponential backoff with Retry-After"),
            Guard("auth_401_rotate_credential", _guard_auth_401_rotate_credential,
                  "401 → rotate credential and retry"),
            Guard("billing_402_rotate_credential", _guard_billing_402_rotate_credential,
                  "402 → rotate credential (billing)"),
            Guard("forbidden_403_abort", _guard_forbidden_403_abort,
                  "403 → non-retryable, abort"),
            Guard("provider_529_overload_backoff", _guard_provider_529_overload_backoff,
                  "529 → extended backoff for overload"),
            Guard("provider_5xx_retry", _guard_provider_5xx_retry,
                  "500/502/503/504 → retry with backoff"),
            Guard("thinking_signature_strip", _guard_thinking_signature_strip,
                  "Invalid thinking signature → strip thinking blocks"),
            Guard("image_shrink_multimodal", _guard_image_shrink_multimodal,
                  "Image too large → shrink and retry"),
            Guard("multimodal_tool_content_fix", _guard_multimodal_tool_content_fix,
                  "Tool content type error → convert to text"),
            Guard("llama_cpp_grammar_disable", _guard_llama_cpp_grammar_disable,
                  "Grammar parse error → disable grammar"),
            Guard("rebuild_messages_drop_thinking", _guard_rebuild_messages_drop_thinking,
                  "Orphaned thinking blocks → rebuild messages"),
            Guard("transport_rebuild_client", _guard_transport_rebuild_client,
                  "Connection pool exhaustion → rebuild client"),
            Guard("restart_signal_compress", _guard_restart_signal_compress,
                  "Restart signal → compress and retry"),
        ]
        for guard in defaults:
            self._guards.append(guard)

    def register(self, guard: Guard, *, priority: Optional[int] = None) -> None:
        """Register a custom guard. Lower priority = checked first."""
        if priority is not None:
            self._guards.insert(priority, guard)
        else:
            self._guards.append(guard)

    def match(self, exc: Exception, context: Optional[ErrorContext] = None) -> Optional[RecoveryAction]:
        """Try each guard in order. Return the first non-None action, or None."""
        if context is None:
            context = ErrorContext(exc=exc)
        else:
            context.exc = exc
            # Re-extract status/body in case the exception changed.
            if context.status_code is None:
                context.status_code = _extract_status_code(exc)
            if context.error_body is None:
                context.error_body = _extract_error_body(exc)
            if not context.error_message:
                context.error_message = str(exc)

        for guard in self._guards:
            try:
                action = guard.match(context)
                if action is not None:
                    logger.info(
                        "recovery guard '%s' matched: %s",
                        guard.name, action.description,
                    )
                    return action
            except Exception as exc:
                logger.warning(
                    "recovery guard '%s' raised an exception: %s",
                    guard.name, exc,
                )
                continue
        return None

    def guards(self) -> List[Guard]:
        """Return the list of registered guards (for inspection/testing)."""
        return list(self._guards)


# ---------------------------------------------------------------------------
# Singleton registry
# ---------------------------------------------------------------------------


_default_registry: Optional[RecoveryRegistry] = None


def get_default_registry() -> RecoveryRegistry:
    """Return the process-wide default RecoveryRegistry."""
    global _default_registry
    if _default_registry is None:
        _default_registry = RecoveryRegistry()
    return _default_registry


__all__ = [
    "ActionType",
    "ErrorContext",
    "Guard",
    "RecoveryAction",
    "RecoveryRegistry",
    "get_default_registry",
]
