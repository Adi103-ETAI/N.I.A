"""OpenAI-compatible API client with comprehensive provider support.

Enhanced port from OpenClaude with support for:
- All OpenAI-compatible providers (Ollama, OpenRouter, Groq, Together, etc.)
- Comprehensive retry logic with exponential backoff
- Error classification and provider-specific handling
- Streaming with tool call accumulation
- Thinking/reasoning content preservation
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from dataclasses import dataclass
from typing import Any, AsyncIterator

from openai import AsyncOpenAI

from niaharness.api.client import (
    ApiMessageCompleteEvent,
    ApiMessageRequest,
    ApiRetryEvent,
    ApiStreamEvent,
    ApiTextDeltaEvent,
    get_retry_delay,
)
from niaharness.api.errors import (
    AuthenticationFailure,
    ConnectionFailure,
    ContextOverflowFailure,
    ModelNotFoundFailure,
    NiaHarnessApiError,
    RateLimitFailure,
    RequestFailure,
    translate_api_error,
)
from niaharness.api.provider_config import (
    ProviderConfig,
    detect_provider_from_url,
    get_local_provider_retry_base_urls,
    is_azure_endpoint,
    is_local_provider_url,
    is_likely_ollama_endpoint,
    should_attempt_local_toolless_retry,
)
from niaharness.api.usage import UsageSnapshot
from niaharness.engine.messages import (
    ConversationMessage,
    ContentBlock,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)

log = logging.getLogger(__name__)

# Retry configuration
MAX_RETRIES = 3
BASE_DELAY = 1.0
MAX_DELAY = 30.0
RETRYABLE_STATUS_CODES = {429, 500, 502, 503}


# ---------------------------------------------------------------------------
# Message conversion helpers
# ---------------------------------------------------------------------------

def _convert_tools_to_openai(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Anthropic tool schemas to OpenAI function-calling format.

    Anthropic format:
        {"name": "...", "description": "...", "input_schema": {...}}
    OpenAI format:
        {"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}
    """
    result = []
    for tool in tools:
        result.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {}),
            },
        })
    return result


def _convert_messages_to_openai(
    messages: list[ConversationMessage],
    system_prompt: str | None,
) -> list[dict[str, Any]]:
    """Convert Anthropic-style messages to OpenAI chat format.

    Key differences:
    - Anthropic: system prompt is a separate parameter
    - OpenAI: system prompt is a message with role="system"
    - Anthropic: tool_use / tool_result are content blocks
    - OpenAI: tool_calls on assistant message, tool results are separate messages
    """
    openai_messages: list[dict[str, Any]] = []

    if system_prompt:
        openai_messages.append({"role": "system", "content": system_prompt})

    for msg in messages:
        if msg.role == "assistant":
            openai_msg = _convert_assistant_message(msg)
            openai_messages.append(openai_msg)
        elif msg.role == "user":
            # User messages may contain text or tool_result blocks
            tool_results = [b for b in msg.content if isinstance(b, ToolResultBlock)]
            text_blocks = [b for b in msg.content if isinstance(b, TextBlock)]

            if tool_results:
                # Each tool result becomes a separate message with role="tool"
                for tr in tool_results:
                    openai_messages.append({
                        "role": "tool",
                        "tool_call_id": tr.tool_use_id,
                        "content": tr.content,
                    })
            if text_blocks:
                text = "".join(b.text for b in text_blocks)
                if text.strip():
                    openai_messages.append({"role": "user", "content": text})
            if not tool_results and not text_blocks:
                # Empty user message (shouldn't happen, but handle gracefully)
                openai_messages.append({"role": "user", "content": ""})

    return openai_messages


def _convert_assistant_message(msg: ConversationMessage) -> dict[str, Any]:
    """Convert an assistant ConversationMessage to OpenAI format.

    Providers with thinking models (e.g. Kimi k2.5, DeepSeek) require a
    ``reasoning_content`` field on every assistant message that contains
    tool calls.
    """
    text_parts = [b.text for b in msg.content if isinstance(b, TextBlock)]
    tool_uses = [b for b in msg.content if isinstance(b, ToolUseBlock)]

    openai_msg: dict[str, Any] = {"role": "assistant"}

    content = "".join(text_parts)
    openai_msg["content"] = content if content else None

    # Replay reasoning_content for thinking models (stored by streaming parser)
    reasoning = getattr(msg, "_reasoning", None)
    if reasoning:
        openai_msg["reasoning_content"] = reasoning
    elif tool_uses:
        # Thinking models require this field even if empty
        openai_msg["reasoning_content"] = ""

    if tool_uses:
        openai_msg["tool_calls"] = [
            {
                "id": tu.id,
                "type": "function",
                "function": {
                    "name": tu.name,
                    "arguments": json.dumps(tu.input),
                },
            }
            for tu in tool_uses
        ]

    return openai_msg


def _parse_assistant_response(response: Any) -> ConversationMessage:
    """Parse an OpenAI ChatCompletion response into a ConversationMessage."""
    choice = response.choices[0]
    message = choice.message
    content: list[ContentBlock] = []

    if message.content:
        content.append(TextBlock(text=message.content))

    if message.tool_calls:
        for tc in message.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, TypeError):
                args = {}
            content.append(ToolUseBlock(
                id=tc.id,
                name=tc.function.name,
                input=args,
            ))

    return ConversationMessage(role="assistant", content=content)


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------

def _classify_openai_error(exc: Exception) -> tuple[str, bool]:
    """Classify OpenAI error into category and retryability."""
    status = getattr(exc, "status_code", None)
    msg = str(exc).lower()

    if status == 401 or status == 403:
        return "auth_invalid", False
    if status == 404:
        return "endpoint_not_found", False
    if status == 429:
        return "rate_limited", True
    if status == 402:
        return "rate_limited", False  # Credits exhausted
    if status in {500, 502, 503}:
        return "provider_unavailable", True
    if status == 400:
        if "too many tokens" in msg or "context length" in msg or "prompt is too long" in msg:
            return "context_overflow", False
        if "model" in msg and ("not found" in msg or "does not exist" in msg):
            return "model_not_found", False
        if "tool" in msg:
            return "tool_call_incompatible", False

    if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return "connection_refused", True

    return "unknown", False


# ---------------------------------------------------------------------------
# Client implementation
# ---------------------------------------------------------------------------

class OpenAICompatibleClient:
    """Client for OpenAI-compatible APIs with comprehensive provider support.

    Implements the same SupportsStreamingMessages protocol as AnthropicApiClient
    so it can be used as a drop-in replacement in the agent loop.

    Supports:
    - OpenAI, Azure OpenAI, Ollama, OpenRouter, Groq, Together AI
    - DeepSeek, Fireworks, NVIDIA NIM, Cerebras
    - AWS Bedrock, Google Vertex, Mistral
    - Comprehensive retry with exponential backoff
    - Error classification and provider-specific handling
    - Local provider optimizations (Ollama tool-less retry)
    """

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str | None = None,
        max_retries: int = MAX_RETRIES,
        provider_config: ProviderConfig | None = None,
    ) -> None:
        self._provider_config = provider_config or detect_provider_from_url(base_url or "")
        kwargs: dict[str, Any] = {"api_key": api_key}

        if base_url:
            kwargs["base_url"] = base_url

        # Azure uses different auth header
        if is_azure_endpoint(base_url or ""):
            kwargs["default_headers"] = {"api-key": api_key}

        self._client = AsyncOpenAI(**kwargs)
        self._max_retries = max_retries
        self._base_url = base_url or ""

    async def stream_message(self, request: ApiMessageRequest) -> AsyncIterator[ApiStreamEvent]:
        """Yield text deltas and the final message, matching the Anthropic client interface."""
        last_error: Exception | None = None
        should_retry_without_tools = False

        for attempt in range(self._max_retries + 1):
            try:
                # If we're retrying without tools (Ollama compatibility)
                current_request = request
                if should_retry_without_tools:
                    current_request = ApiMessageRequest(
                        model=request.model,
                        messages=request.messages,
                        system_prompt=request.system_prompt,
                        max_tokens=request.max_tokens,
                        tools=[],  # Empty tools
                        temperature=request.temperature,
                        top_p=request.top_p,
                        reasoning_effort=request.reasoning_effort,
                        stream=request.stream,
                    )
                    should_retry_without_tools = False

                async for event in self._stream_once(current_request):
                    yield event
                return
            except NiaHarnessApiError:
                raise
            except Exception as exc:
                last_error = exc
                category, retryable = _classify_openai_error(exc)

                # Check for local provider tool-less retry
                if (
                    not should_retry_without_tools
                    and is_likely_ollama_endpoint(self._base_url)
                    and should_attempt_local_toolless_retry(self._base_url, bool(request.tools))
                    and category == "tool_call_incompatible"
                ):
                    log.warning(
                        "Ollama rejected tool-calling, retrying without tools",
                    )
                    should_retry_without_tools = True
                    continue

                if attempt >= self._max_retries or not retryable:
                    raise self._translate_error(exc) from exc

                # Calculate delay
                delay = min(BASE_DELAY * (2 ** attempt), MAX_DELAY)

                # Check for Retry-After header
                retry_after = getattr(exc, "headers", {})
                if hasattr(retry_after, "get"):
                    val = retry_after.get("retry-after")
                    if val:
                        try:
                            delay = min(float(val), MAX_DELAY)
                        except (ValueError, TypeError):
                            pass

                log.warning(
                    "OpenAI API request failed (attempt %d/%d, category=%s), retrying in %.1fs: %s",
                    attempt + 1, self._max_retries + 1, category, delay, exc,
                )

                yield ApiRetryEvent(
                    attempt=attempt + 1,
                    max_retries=self._max_retries,
                    delay_seconds=delay,
                    error=str(exc),
                    status_code=getattr(exc, "status_code", None),
                )

                await asyncio.sleep(delay)

        if last_error is not None:
            raise self._translate_error(last_error) from last_error

    async def _stream_once(self, request: ApiMessageRequest) -> AsyncIterator[ApiStreamEvent]:
        """Single attempt: stream an OpenAI chat completion."""
        openai_messages = _convert_messages_to_openai(request.messages, request.system_prompt)
        openai_tools = _convert_tools_to_openai(request.tools) if request.tools else None

        params: dict[str, Any] = {
            "model": request.model,
            "messages": openai_messages,
            "max_tokens": request.max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if openai_tools:
            params["tools"] = openai_tools
            # Some providers (Kimi) error on empty reasoning_content in
            # tool-call follow-ups. Omit stream_options if tools are present.
            params.pop("stream_options", None)

        if request.temperature is not None:
            params["temperature"] = request.temperature
        if request.top_p is not None:
            params["top_p"] = request.top_p

        # Collect full response while streaming text deltas
        collected_content = ""
        collected_reasoning = ""
        collected_tool_calls: dict[int, dict[str, Any]] = {}
        finish_reason: str | None = None
        usage_data: dict[str, int] = {}

        stream = await self._client.chat.completions.create(**params)
        async for chunk in stream:
            if not chunk.choices:
                # Usage-only chunk (some providers send this at the end)
                if chunk.usage:
                    usage_data = {
                        "input_tokens": chunk.usage.prompt_tokens or 0,
                        "output_tokens": chunk.usage.completion_tokens or 0,
                    }
                continue

            delta = chunk.choices[0].delta
            chunk_finish = chunk.choices[0].finish_reason

            if chunk_finish:
                finish_reason = chunk_finish

            # Accumulate reasoning_content from thinking models (not shown to user)
            reasoning_piece = getattr(delta, "reasoning_content", None) or ""
            if reasoning_piece:
                collected_reasoning += reasoning_piece

            # Stream text content to user
            if delta.content:
                collected_content += delta.content
                yield ApiTextDeltaEvent(text=delta.content)

            # Accumulate tool calls
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in collected_tool_calls:
                        collected_tool_calls[idx] = {
                            "id": tc_delta.id or "",
                            "name": "",
                            "arguments": "",
                        }
                    entry = collected_tool_calls[idx]
                    if tc_delta.id:
                        entry["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            entry["name"] = tc_delta.function.name
                        if tc_delta.function.arguments:
                            entry["arguments"] += tc_delta.function.arguments

            # Usage in chunk (if provider sends it)
            if chunk.usage:
                usage_data = {
                    "input_tokens": chunk.usage.prompt_tokens or 0,
                    "output_tokens": chunk.usage.completion_tokens or 0,
                }

        # Build the final ConversationMessage
        content: list[ContentBlock] = []
        if collected_content:
            content.append(TextBlock(text=collected_content))

        for _idx in sorted(collected_tool_calls.keys()):
            tc = collected_tool_calls[_idx]
            # Skip phantom/empty tool calls that some providers send
            if not tc["name"]:
                continue
            try:
                args = json.loads(tc["arguments"])
            except (json.JSONDecodeError, TypeError):
                args = {}
            content.append(ToolUseBlock(
                id=tc["id"],
                name=tc["name"],
                input=args,
            ))

        final_message = ConversationMessage(role="assistant", content=content)

        # Stash reasoning for thinking models so _convert_assistant_message
        # can replay it when the message is sent back to the API
        if collected_reasoning:
            final_message._reasoning = collected_reasoning  # type: ignore[attr-defined]

        yield ApiMessageCompleteEvent(
            message=final_message,
            usage=UsageSnapshot(
                input_tokens=usage_data.get("input_tokens", 0),
                output_tokens=usage_data.get("output_tokens", 0),
            ),
            stop_reason=finish_reason,
        )

    @staticmethod
    def _translate_error(exc: Exception) -> NiaHarnessApiError:
        """Translate an exception to a NiaHarnessApiError."""
        status = getattr(exc, "status_code", None)
        msg = str(exc)

        if status == 401 or status == 403:
            return AuthenticationFailure(msg)
        if status == 429:
            return RateLimitFailure(msg)
        if status == 400:
            msg_lower = msg.lower()
            if "too many tokens" in msg_lower or "context length" in msg_lower:
                return ContextOverflowFailure(msg)
            if "model" in msg_lower and "not found" in msg_lower:
                return ModelNotFoundFailure(msg)
        if status in {502, 503}:
            return ConnectionFailure(msg)

        return RequestFailure(msg)
