"""OpenAI-compatible API shim for NiaHarness.

Full port of OpenClaude's openaiShim.ts with support for:
- Message conversion (Anthropic ↔ OpenAI)
- Tool conversion with schema normalization
- Streaming response transformation
- All supported providers (Ollama, OpenRouter, Groq, Together, DeepSeek, etc.)
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional

from niaharness.api.provider_config import (
    ProviderTransport,
    ReasoningEffort,
    ResolvedProviderRequest,
    detect_provider_from_url,
    get_local_fast_path_config,
    is_azure_endpoint,
    is_gemini_endpoint,
    is_local_provider_url,
    is_likely_ollama_endpoint,
    resolve_provider_request,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

@dataclass
class OpenAIMessage:
    """OpenAI chat completion message."""

    role: str  # "system", "user", "assistant", "tool"
    content: str | list[dict[str, Any]] | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None
    reasoning_content: Optional[str] = None


@dataclass
class OpenAITool:
    """OpenAI function tool definition."""

    type: str = "function"
    function: dict[str, Any] = field(default_factory=dict)


@dataclass
class OpenAIStreamChunk:
    """OpenAI streaming response chunk."""

    id: str = ""
    object: str = ""
    model: str = ""
    choices: list[dict[str, Any]] = field(default_factory=list)
    usage: Optional[dict[str, Any]] = None


@dataclass
class AnthropicStreamEvent:
    """Anthropic-format stream event for compatibility."""

    type: str
    index: Optional[int] = None
    content_block: Optional[dict[str, Any]] = None
    delta: Optional[dict[str, Any]] = None
    message: Optional[dict[str, Any]] = None
    usage: Optional[dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Message conversion: Anthropic → OpenAI
# ---------------------------------------------------------------------------

def _convert_system_prompt(system: Any) -> str:
    """Convert Anthropic system prompt to string."""
    if not system:
        return ""
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        parts = []
        for block in system:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
                if not text.startswith("x-anthropic-billing-header"):
                    parts.append(text)
        return "\n\n".join(parts)
    return str(system)


def _convert_tool_result_content(content: Any, is_error: bool = False) -> str | list[dict[str, Any]]:
    """Convert tool result content to OpenAI format."""
    if isinstance(content, str):
        return f"Error: {content}" if is_error else content
    if not isinstance(content, list):
        text = json.dumps(content if content is not None else "")
        return f"Error: {text}" if is_error else text

    parts: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            parts.append({"type": "text", "text": block["text"]})
        elif block.get("type") == "image":
            source = block.get("source", {})
            if source.get("type") == "url" and source.get("url"):
                parts.append({"type": "image_url", "image_url": {"url": source["url"]}})
            elif source.get("type") == "base64" and source.get("media_type") and source.get("data"):
                parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{source['media_type']};base64,{source['data']}"},
                })
        elif isinstance(block.get("text"), str):
            parts.append({"type": "text", "text": block["text"]})

    if not parts:
        return ""
    if len(parts) == 1 and parts[0].get("type") == "text":
        text = parts[0].get("text", "")
        return f"Error: {text}" if is_error else text

    # DeepSeek compatibility: collapse all-text arrays into a single string
    all_text = all(p.get("type") == "text" for p in parts)
    if all_text:
        text = "\n\n".join(p.get("text", "") for p in parts)
        return f"Error: {text}" if is_error else text

    if is_error and parts and parts[0].get("type") == "text":
        parts[0] = {**parts[0], "text": f"Error: {parts[0].get('text', '')}"}
    elif is_error:
        parts.insert(0, {"type": "text", "text": "Error:"})

    return parts


def _convert_content_blocks(content: Any) -> str | list[dict[str, Any]]:
    """Convert Anthropic content blocks to OpenAI format."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content if content is not None else "")

    parts: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")

        if block_type == "text":
            parts.append({"type": "text", "text": block.get("text", "")})
        elif block_type == "image":
            source = block.get("source", {})
            if source.get("type") == "base64":
                parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{source['media_type']};base64,{source['data']}"},
                })
            elif source.get("type") == "url":
                parts.append({"type": "image_url", "image_url": {"url": source["url"]}})
        elif block_type in ("tool_use", "tool_result", "thinking", "redacted_thinking"):
            # Skip Anthropic-specific types
            continue
        elif block.get("text"):
            parts.append({"type": "text", "text": block["text"]})

    if not parts:
        return ""
    if len(parts) == 1 and parts[0].get("type") == "text":
        return parts[0].get("text", "")

    # DeepSeek compatibility: collapse all-text arrays
    all_text = all(p.get("type") == "text" for p in parts)
    if all_text:
        return "\n\n".join(p.get("text", "") for p in parts)

    return parts


def _make_tool_call_id() -> str:
    """Generate a tool call ID in Anthropic format."""
    return f"toolu_{uuid.uuid4().hex[:24]}"


def _make_message_id() -> str:
    """Generate a message ID in Anthropic format."""
    return f"msg_{uuid.uuid4().hex}"


def convert_messages(
    messages: list[dict[str, Any]],
    system: Any = None,
    *,
    preserve_reasoning_content: bool = False,
    reasoning_content_fallback: Optional[str] = None,
) -> list[OpenAIMessage]:
    """Convert Anthropic messages to OpenAI format.

    Handles:
    - System message extraction
    - Tool use/result pairing
    - Content block conversion
    - Role alternation coalescing
    - Orphaned tool result dropping
    """
    result: list[OpenAIMessage] = []
    known_tool_call_ids: set[str] = set()

    # Pre-scan for all tool results
    tool_result_ids: set[str] = set()
    for msg in messages:
        inner = msg.get("message", msg)
        content = inner.get("content", msg.get("content"))
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    tool_result_id = block.get("tool_use_id")
                    if tool_result_id:
                        tool_result_ids.add(tool_result_id)

    # System message
    sys_text = _convert_system_prompt(system)
    if sys_text:
        result.append(OpenAIMessage(role="system", content=sys_text))

    for i, msg in enumerate(messages):
        is_last = i == len(messages) - 1
        inner = msg.get("message", msg)
        role = inner.get("role", msg.get("role", "user"))
        content = inner.get("content", msg.get("content"))

        if role == "user":
            if isinstance(content, list):
                tool_results = [b for b in content if isinstance(b, dict) and b.get("type") == "tool_result"]
                other_content = [b for b in content if not (isinstance(b, dict) and b.get("type") == "tool_result")]

                # Emit tool results
                for tr in tool_results:
                    tr_id = tr.get("tool_use_id", "unknown")
                    if tr_id in known_tool_call_ids:
                        result.append(OpenAIMessage(
                            role="tool",
                            tool_call_id=tr_id,
                            content=_convert_tool_result_content(tr.get("content"), tr.get("is_error")),
                        ))
                    else:
                        log.debug("Dropping orphan tool_result for ID: %s", tr_id)

                # Emit remaining user content
                if other_content:
                    result.append(OpenAIMessage(role="user", content=_convert_content_blocks(other_content)))
            else:
                result.append(OpenAIMessage(role="user", content=_convert_content_blocks(content)))

        elif role == "assistant":
            if isinstance(content, list):
                tool_uses = [b for b in content if isinstance(b, dict) and b.get("type") == "tool_use"]
                thinking_block = next((b for b in content if isinstance(b, dict) and b.get("type") == "thinking"), None)
                text_content = [b for b in content if isinstance(b, dict) and b.get("type") not in ("tool_use", "thinking")]

                text = _convert_content_blocks(text_content)
                if isinstance(text, list):
                    text = "".join(p.get("text", "") for p in text)

                assistant_msg = OpenAIMessage(role="assistant", content=text or "")

                # Preserve reasoning content for thinking models
                if preserve_reasoning_content and thinking_block:
                    thinking_text = thinking_block.get("thinking", "")
                    if isinstance(thinking_text, str) and thinking_text.strip():
                        assistant_msg.reasoning_content = thinking_text
                    elif tool_uses and reasoning_content_fallback == "":
                        assistant_msg.reasoning_content = ""

                # Convert tool calls
                if tool_uses:
                    mapped_tool_calls = []
                    for tu in tool_uses:
                        tu_id = tu.get("id") or f"call_{uuid.uuid4().hex}"
                        # Only keep tool calls with matching results or if last message
                        if tu_id not in tool_result_ids and not is_last:
                            continue
                        known_tool_call_ids.add(tu_id)
                        tu_input = tu.get("input", {})
                        mapped_tool_calls.append({
                            "id": tu_id,
                            "type": "function",
                            "function": {
                                "name": tu.get("name", "unknown"),
                                "arguments": tu_input if isinstance(tu_input, str) else json.dumps(tu_input),
                            },
                        })
                    if mapped_tool_calls:
                        assistant_msg.tool_calls = mapped_tool_calls

                # Only push if has content or tool calls
                if assistant_msg.content or assistant_msg.tool_calls:
                    result.append(assistant_msg)
            else:
                text = _convert_content_blocks(content)
                if isinstance(text, list):
                    text = "".join(p.get("text", "") for p in text)
                if text:
                    result.append(OpenAIMessage(role="assistant", content=text))

    # Coalescing pass: merge consecutive messages of the same role
    coalesced: list[OpenAIMessage] = []
    for msg in result:
        if not coalesced:
            coalesced.append(msg)
            continue

        prev = coalesced[-1]

        # Inject assistant message between tool and user for Mistral/Devstral
        if prev.role == "tool" and msg.role == "user":
            coalesced.append(OpenAIMessage(role="assistant", content="[Tool execution interrupted by user]"))

        last = coalesced[-1]
        if last.role == msg.role and msg.role not in ("tool", "system"):
            # Merge content
            if isinstance(last.content, str) and isinstance(msg.content, str):
                last.content = last.content + ("\n" if last.content and msg.content else "") + msg.content
            else:
                prev_parts = _to_list(last.content)
                cur_parts = _to_list(msg.content)
                last.content = prev_parts + cur_parts

            # Merge tool calls
            if msg.tool_calls:
                last.tool_calls = (last.tool_calls or []) + msg.tool_calls
        else:
            coalesced.append(msg)

    return coalesced


def _to_list(content: str | list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Convert content to a list of content blocks."""
    if not content:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    return content


# ---------------------------------------------------------------------------
# Tool conversion: Anthropic → OpenAI
# ---------------------------------------------------------------------------

def _normalize_schema_for_openai(schema: dict[str, Any], strict: bool = True) -> dict[str, Any]:
    """Normalize JSON schema for OpenAI compatibility.

    OpenAI requires every key in `properties` to also appear in `required`.
    """
    record = dict(schema)

    if record.get("type") == "object" and "properties" in record:
        properties = record["properties"]
        existing_required = record.get("required", [])

        # Recurse into each property
        normalized_props = {}
        for key, value in properties.items():
            normalized_props[key] = _normalize_schema_for_openai(value, strict)
        record["properties"] = normalized_props

        if strict:
            record["required"] = [k for k in existing_required if k in normalized_props]
            record["additionalProperties"] = False
        else:
            record["required"] = [k for k in existing_required if k in normalized_props]

    # Recurse into array items
    if "items" in record:
        items = record["items"]
        if isinstance(items, list):
            record["items"] = [_normalize_schema_for_openai(item, strict) for item in items]
        elif isinstance(items, dict):
            record["items"] = _normalize_schema_for_openai(items, strict)

    # Recurse into combinators
    for key in ("anyOf", "oneOf", "allOf"):
        if key in record and isinstance(record[key], list):
            record[key] = [_normalize_schema_for_openai(item, strict) for item in record[key]]

    return record


def convert_tools(
    tools: list[dict[str, Any]],
    *,
    skip_strict: bool = False,
) -> list[OpenAITool]:
    """Convert Anthropic tool schemas to OpenAI format.

    Anthropic format:
        {"name": "...", "description": "...", "input_schema": {...}}
    OpenAI format:
        {"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}
    """
    # Check if we should skip strict mode
    is_gemini = False  # Will be set by caller if needed
    strict = not is_gemini and not skip_strict

    result = []
    for tool in tools:
        schema = dict(tool.get("input_schema", {"type": "object", "properties": {}}))

        # For Agent tools, promote known sub-fields into required
        if tool.get("name") == "Agent" and "properties" in schema:
            props = schema["properties"]
            required = schema.get("required", [])
            for key in ("message", "subagent_type"):
                if key in props and key not in required:
                    required.append(key)
            schema["required"] = required

        result.append(OpenAITool(
            function={
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "parameters": _normalize_schema_for_openai(schema, strict),
            },
        ))

    return result


# ---------------------------------------------------------------------------
# Streaming: OpenAI SSE → Anthropic stream events
# ---------------------------------------------------------------------------

async def openai_stream_to_anthropic(
    response: Any,  # httpx.Response or similar
    model: str,
    signal: Any = None,
) -> AsyncIterator[AnthropicStreamEvent]:
    """Transform OpenAI SSE stream into Anthropic-format stream events."""
    message_id = _make_message_id()
    content_block_index = 0
    active_tool_calls: dict[int, dict[str, Any]] = {}
    has_emitted_content_start = False
    has_emitted_thinking_start = False
    has_closed_thinking = False
    last_stop_reason: Optional[str] = None
    has_emitted_final_usage = False
    has_processed_finish_reason = False

    # Emit message_start
    yield AnthropicStreamEvent(
        type="message_start",
        message={
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": model,
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    )

    async def close_active_content_block():
        nonlocal content_block_index, has_emitted_content_start
        if not has_emitted_content_start:
            return
        yield AnthropicStreamEvent(type="content_block_stop", index=content_block_index)
        content_block_index += 1
        has_emitted_content_start = False

    async def emit_text_delta(text: str):
        nonlocal content_block_index, has_emitted_content_start
        if not text:
            return
        if not has_emitted_content_start:
            yield AnthropicStreamEvent(
                type="content_block_start",
                index=content_block_index,
                content_block={"type": "text", "text": ""},
            )
            has_emitted_content_start = True
        yield AnthropicStreamEvent(
            type="content_block_delta",
            index=content_block_index,
            delta={"type": "text_delta", "text": text},
        )

    try:
        async for line in _read_sse_lines(response, signal):
            if not line or line == "data: [DONE]":
                continue
            if not line.startswith("data: "):
                continue

            try:
                chunk_data = json.loads(line[6:])
            except json.JSONDecodeError:
                continue

            # Handle in-stream errors
            error_data = chunk_data.get("error")
            if error_data and isinstance(error_data, dict):
                error_msg = error_data.get("message", "Provider returned an in-stream error")
                raise RuntimeError(f"Provider error: {error_msg}")

            # Process choices
            for choice in chunk_data.get("choices", []):
                delta = choice.get("delta", {})
                finish_reason = choice.get("finish_reason")

                # Reasoning content (thinking models)
                reasoning_content = delta.get("reasoning_content")
                if reasoning_content and reasoning_content != "":
                    if not has_emitted_thinking_start:
                        yield AnthropicStreamEvent(
                            type="content_block_start",
                            index=content_block_index,
                            content_block={"type": "thinking", "thinking": ""},
                        )
                        has_emitted_thinking_start = True
                    yield AnthropicStreamEvent(
                        type="content_block_delta",
                        index=content_block_index,
                        delta={"type": "thinking_delta", "thinking": reasoning_content},
                    )

                # Text content
                content = delta.get("content")
                if content is not None and content != "":
                    if has_emitted_thinking_start and not has_closed_thinking:
                        yield AnthropicStreamEvent(type="content_block_stop", index=content_block_index)
                        content_block_index += 1
                        has_closed_thinking = True
                    async for event in emit_text_delta(content):
                        yield event

                # Tool calls
                tool_calls = delta.get("tool_calls")
                if tool_calls:
                    for tc in tool_calls:
                        tc_id = tc.get("id")
                        func = tc.get("function", {})
                        func_name = func.get("name")

                        if tc_id and func_name:
                            # New tool call
                            if has_emitted_thinking_start and not has_closed_thinking:
                                yield AnthropicStreamEvent(type="content_block_stop", index=content_block_index)
                                content_block_index += 1
                                has_closed_thinking = True
                            if has_emitted_content_start:
                                async for event in close_active_content_block():
                                    yield event

                            tool_block_index = content_block_index
                            initial_arguments = func.get("arguments", "")

                            active_tool_calls[tc.get("index", 0)] = {
                                "id": tc_id,
                                "name": func_name,
                                "index": tool_block_index,
                                "json_buffer": initial_arguments,
                            }

                            yield AnthropicStreamEvent(
                                type="content_block_start",
                                index=tool_block_index,
                                content_block={
                                    "type": "tool_use",
                                    "id": tc_id,
                                    "name": func_name,
                                    "input": {},
                                },
                            )
                            content_block_index += 1

                            if initial_arguments:
                                yield AnthropicStreamEvent(
                                    type="content_block_delta",
                                    index=tool_block_index,
                                    delta={"type": "input_json_delta", "partial_json": initial_arguments},
                                )
                        elif func.get("arguments"):
                            # Continuation of existing tool call
                            active = active_tool_calls.get(tc.get("index", 0))
                            if active:
                                active["json_buffer"] += func["arguments"]
                                yield AnthropicStreamEvent(
                                    type="content_block_delta",
                                    index=active["index"],
                                    delta={"type": "input_json_delta", "partial_json": func["arguments"]},
                                )

                # Finish reason
                if finish_reason and not has_processed_finish_reason:
                    has_processed_finish_reason = True

                    if has_emitted_thinking_start and not has_closed_thinking:
                        yield AnthropicStreamEvent(type="content_block_stop", index=content_block_index)
                        content_block_index += 1
                        has_closed_thinking = True

                    if has_emitted_content_start:
                        async for event in close_active_content_block():
                            yield event

                    # Close active tool calls
                    for _idx, tc in active_tool_calls.items():
                        yield AnthropicStreamEvent(type="content_block_stop", index=tc["index"])

                    stop_reason = "tool_use" if finish_reason == "tool_calls" else (
                        "max_tokens" if finish_reason == "length" else "end_turn"
                    )

                    # Handle content filter / safety
                    if finish_reason in ("content_filter", "safety"):
                        if not has_emitted_content_start:
                            yield AnthropicStreamEvent(
                                type="content_block_start",
                                index=content_block_index,
                                content_block={"type": "text", "text": ""},
                            )
                            has_emitted_content_start = True
                        yield AnthropicStreamEvent(
                            type="content_block_delta",
                            index=content_block_index,
                            delta={"type": "text_delta", "text": "\n\n[Content blocked by provider safety filter]"},
                        )

                    last_stop_reason = stop_reason

                    chunk_usage = _convert_chunk_usage(chunk_data.get("usage"))
                    yield AnthropicStreamEvent(
                        type="message_delta",
                        delta={"stop_reason": stop_reason, "stop_sequence": None},
                        **({"usage": chunk_usage} if chunk_usage else {}),
                    )
                    if chunk_usage:
                        has_emitted_final_usage = True

            # Handle usage-only chunks
            chunk_usage = _convert_chunk_usage(chunk_data.get("usage"))
            if (
                not has_emitted_final_usage
                and chunk_usage
                and len(chunk_data.get("choices", [])) == 0
                and last_stop_reason is not None
            ):
                yield AnthropicStreamEvent(
                    type="message_delta",
                    delta={"stop_reason": last_stop_reason, "stop_sequence": None},
                    usage=chunk_usage,
                )
                has_emitted_final_usage = True

    finally:
        yield AnthropicStreamEvent(type="message_stop")


async def _read_sse_lines(response: Any, signal: Any = None) -> AsyncIterator[str]:
    """Read SSE lines from a response stream."""
    buffer = ""
    async for chunk in response.aiter_text():
        buffer += chunk
        lines = buffer.split("\n")
        buffer = lines.pop() or ""
        for line in lines:
            yield line.strip()


def _convert_chunk_usage(usage: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Convert OpenAI chunk usage to Anthropic format."""
    if not usage:
        return None
    return {
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
    }


# ---------------------------------------------------------------------------
# Non-streaming response conversion
# ---------------------------------------------------------------------------

def convert_non_streaming_response(data: dict[str, Any], model: str) -> dict[str, Any]:
    """Convert OpenAI non-streaming response to Anthropic format."""
    message_id = _make_message_id()
    choice = data.get("choices", [{}])[0] if data.get("choices") else {}
    message = choice.get("message", {})

    content_blocks = []

    # Text content
    if message.get("content"):
        content_blocks.append({"type": "text", "text": message["content"]})

    # Tool calls
    if message.get("tool_calls"):
        for tc in message["tool_calls"]:
            func = tc.get("function", {})
            try:
                args = json.loads(func.get("arguments", "{}"))
            except (json.JSONDecodeError, TypeError):
                args = {}
            content_blocks.append({
                "type": "tool_use",
                "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:24]}"),
                "name": func.get("name", ""),
                "input": args,
            })

    # Map stop reason
    finish_reason = choice.get("finish_reason")
    stop_reason = "tool_use" if finish_reason == "tool_calls" else (
        "max_tokens" if finish_reason == "length" else "end_turn"
    )

    usage_data = data.get("usage", {})

    return {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "content": content_blocks,
        "model": model,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage_data.get("prompt_tokens", 0),
            "output_tokens": usage_data.get("completion_tokens", 0),
        },
    }


# ---------------------------------------------------------------------------
# Gemini SSE → Anthropic stream events
# ---------------------------------------------------------------------------

async def gemini_sse_to_anthropic(
    response: Any,
    model: str,
    signal: Any = None,
) -> AsyncIterator[AnthropicStreamEvent]:
    """Transform Google AI SDK SSE stream into Anthropic-format stream events."""
    message_id = _make_message_id()
    content_block_index = 0
    has_emitted_start = False
    has_emitted_text_start = False
    has_emitted_current_tool = False
    usage: Optional[dict[str, Any]] = None
    finish_reason: Optional[str] = None

    async for line in _read_sse_lines(response, signal):
        if not line or line == "data: [DONE]":
            if has_emitted_text_start or has_emitted_current_tool:
                yield AnthropicStreamEvent(type="content_block_stop", index=content_block_index)
            yield AnthropicStreamEvent(
                type="message_delta",
                delta={"stop_reason": _map_gemini_finish_reason(finish_reason, has_emitted_current_tool)},
                usage=usage or {},
            )
            yield AnthropicStreamEvent(type="message_stop")
            return

        if not line.startswith("data: "):
            continue

        try:
            parsed = json.loads(line[6:])
        except json.JSONDecodeError:
            continue

        if not has_emitted_start:
            yield AnthropicStreamEvent(
                type="message_start",
                message={
                    "id": message_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": model,
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                },
            )
            has_emitted_start = True

        # Usage metadata
        usage_metadata = parsed.get("usageMetadata")
        if usage_metadata and isinstance(usage_metadata, dict):
            usage = {
                "input_tokens": usage_metadata.get("promptTokenCount", 0),
                "output_tokens": usage_metadata.get("candidatesTokenCount", 0) + usage_metadata.get("thoughtsTokenCount", 0),
            }

        candidates = parsed.get("candidates", [])
        if not candidates:
            continue
        candidate = candidates[0]

        if isinstance(candidate.get("finishReason"), str):
            finish_reason = candidate["finishReason"]

        content = candidate.get("content")
        if not content or not content.get("parts"):
            continue

        for part in content["parts"]:
            text = part.get("text")
            fc = part.get("functionCall")

            if text:
                if has_emitted_current_tool:
                    yield AnthropicStreamEvent(type="content_block_stop", index=content_block_index)
                    content_block_index += 1
                    has_emitted_current_tool = False
                if not has_emitted_text_start:
                    yield AnthropicStreamEvent(
                        type="content_block_start",
                        index=content_block_index,
                        content_block={"type": "text", "text": ""},
                    )
                    has_emitted_text_start = True
                yield AnthropicStreamEvent(
                    type="content_block_delta",
                    index=content_block_index,
                    delta={"type": "text_delta", "text": text},
                )
            elif fc and fc.get("name"):
                if has_emitted_text_start:
                    yield AnthropicStreamEvent(type="content_block_stop", index=content_block_index)
                    content_block_index += 1
                    has_emitted_text_start = False
                tool_id = f"toolu_{uuid.uuid4().hex[:24]}"
                yield AnthropicStreamEvent(
                    type="content_block_start",
                    index=content_block_index,
                    content_block={"type": "tool_use", "id": tool_id, "name": fc["name"], "input": {}},
                )
                has_emitted_current_tool = True
                args = fc.get("args", {})
                yield AnthropicStreamEvent(
                    type="content_block_delta",
                    index=content_block_index,
                    delta={
                        "type": "input_json_delta",
                        "partial_json": args if isinstance(args, str) else json.dumps(args),
                    },
                )

    # Stream ended without [DONE]
    if has_emitted_text_start or has_emitted_current_tool:
        yield AnthropicStreamEvent(type="content_block_stop", index=content_block_index)
    yield AnthropicStreamEvent(
        type="message_delta",
        delta={"stop_reason": _map_gemini_finish_reason(finish_reason, has_emitted_current_tool)},
        usage=usage or {},
    )
    yield AnthropicStreamEvent(type="message_stop")


def _map_gemini_finish_reason(reason: Optional[str], has_tool_use: bool) -> str:
    """Map Gemini finish reason to Anthropic stop reason."""
    if has_tool_use:
        return "tool_use"
    if reason == "MAX_TOKENS":
        return "max_tokens"
    return "end_turn"
