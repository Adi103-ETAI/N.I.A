"""Conversation message models used by the query engine.

Includes message sanitization (surrogate stripping, tool-call repair,
role-alternation repair) ported from Hermes Agent's
agent/message_sanitization.py + agent/message_content.py.

Without sanitization, NIA will crash on:
- Unicode surrogate pairs (U+D800–U+DFFF) from some providers
- Malformed/truncated tool_use blocks from weak models
- Role-alternation violations (consecutive user or assistant messages)
"""

from __future__ import annotations

import logging
import re
from typing import Any, Annotated, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class TextBlock(BaseModel):
    """Plain text content."""

    type: Literal["text"] = "text"
    text: str


class ThinkingBlock(BaseModel):
    """A thinking/reasoning block from extended-thinking models."""

    type: Literal["thinking"] = "thinking"
    thinking: str


class ToolUseBlock(BaseModel):
    """A request from the model to execute a named tool."""

    type: Literal["tool_use"] = "tool_use"
    id: str = Field(default_factory=lambda: f"toolu_{uuid4().hex}")
    name: str
    input: dict[str, Any] = Field(default_factory=dict)


class ToolResultBlock(BaseModel):
    """Tool result content sent back to the model."""

    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str
    is_error: bool = False


ContentBlock = Annotated[
    TextBlock | ThinkingBlock | ToolUseBlock | ToolResultBlock,
    Field(discriminator="type"),
]


class ConversationMessage(BaseModel):
    """A single assistant or user message."""

    role: Literal["user", "assistant"]
    content: list[ContentBlock] = Field(default_factory=list)

    @classmethod
    def from_user_text(cls, text: str) -> "ConversationMessage":
        """Construct a user message from raw text."""
        return cls(role="user", content=[TextBlock(text=text)])

    @property
    def text(self) -> str:
        """Return concatenated text blocks."""
        return "".join(
            block.text for block in self.content if isinstance(block, TextBlock)
        )

    @property
    def tool_uses(self) -> list[ToolUseBlock]:
        """Return all tool calls contained in the message."""
        return [block for block in self.content if isinstance(block, ToolUseBlock)]

    def to_api_param(self) -> dict[str, Any]:
        """Convert the message into Anthropic SDK message params."""
        return {
            "role": self.role,
            "content": [serialize_content_block(block) for block in self.content],
        }


def serialize_content_block(block: ContentBlock) -> dict[str, Any]:
    """Convert a local content block into the provider wire format."""
    if isinstance(block, TextBlock):
        return {"type": "text", "text": block.text}

    if isinstance(block, ThinkingBlock):
        return {"type": "thinking", "thinking": block.thinking}

    if isinstance(block, ToolUseBlock):
        return {
            "type": "tool_use",
            "id": block.id,
            "name": block.name,
            "input": block.input,
        }

    return {
        "type": "tool_result",
        "tool_use_id": block.tool_use_id,
        "content": block.content,
        "is_error": block.is_error,
    }


def assistant_message_from_api(raw_message: Any) -> ConversationMessage:
    """Convert an Anthropic SDK message object into a conversation message.

    Handles text, thinking, and tool_use blocks. Drops unknown block types
    gracefully.
    """
    content: list[ContentBlock] = []

    for raw_block in getattr(raw_message, "content", []):
        block_type = getattr(raw_block, "type", None)
        if block_type == "text":
            text = sanitize_text(getattr(raw_block, "text", ""))
            if text:
                content.append(TextBlock(text=text))
        elif block_type == "thinking":
            thinking = sanitize_text(getattr(raw_block, "thinking", ""))
            if thinking:
                content.append(ThinkingBlock(thinking=thinking))
        elif block_type == "tool_use":
            tool_id = getattr(raw_block, "id", f"toolu_{uuid4().hex}")
            tool_name = getattr(raw_block, "name", "")
            tool_input = getattr(raw_block, "input", {}) or {}
            if not isinstance(tool_input, dict):
                tool_input = {}
            content.append(
                ToolUseBlock(id=tool_id, name=tool_name, input=tool_input)
            )
        # Drop unknown block types silently.

    return ConversationMessage(role="assistant", content=content)


# ---------------------------------------------------------------------------
# Message Sanitization (ported from Hermes agent/message_sanitization.py)
# ---------------------------------------------------------------------------

# Regex matching Unicode surrogate code points (U+D800–U+DFFF).
# These are invalid in UTF-8 and cause JSON serialization errors / crashes
# when passed to the Anthropic or OpenAI APIs.
_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")

# Regex matching other problematic Unicode control characters
# (excluding tab \t, newline \n, carriage return \r).
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Regex matching null bytes.
_NULL_RE = re.compile(r"\x00")


def sanitize_text(text: str) -> str:
    """Sanitize text by removing surrogate pairs, null bytes, and control chars.

    Ported from Hermes Agent's message sanitization. Surrogate pairs
    (U+D800–U+DFFF) are invalid in UTF-8 and will cause:
    - JSON serialization errors (the Anthropic SDK uses json.dumps)
    - API rejection (HTTP 400 "invalid string")
    - Crashes in some terminal renderers

    This function strips them before the text reaches the API or UI.

    Args:
        text: The text to sanitize.

    Returns:
        Sanitized text with surrogates, nulls, and control chars removed.
    """
    if not text:
        return text

    # Strip null bytes first (they can break string operations).
    result = _NULL_RE.sub("", text)

    # Strip Unicode surrogates (U+D800–U+DFFF).
    result = _SURROGATE_RE.sub("", result)

    # Strip other control characters (except tab, newline, carriage return).
    result = _CONTROL_CHAR_RE.sub("", result)

    return result


def sanitize_messages(messages: list[ConversationMessage]) -> list[ConversationMessage]:
    """Sanitize a list of conversation messages.

    Applies sanitize_text to all TextBlock content, repairs malformed
    tool_use blocks, and fixes role-alternation violations.

    Ported from Hermes Agent's message_sanitization.py. This should be
    called before every API request to prevent crashes from:
    - Surrogate pairs in model output (some providers emit them)
    - Truncated tool_use blocks (weak models produce malformed JSON)
    - Consecutive same-role messages (API rejects them with 400)

    Args:
        messages: The message list to sanitize (modified in place).

    Returns:
        The sanitized message list (same list, modified in place).
    """
    if not messages:
        return messages

    # Phase 1: Sanitize text content in all blocks.
    for msg in messages:
        for i, block in enumerate(msg.content):
            if isinstance(block, TextBlock):
                sanitized = sanitize_text(block.text)
                if sanitized != block.text:
                    msg.content[i] = TextBlock(text=sanitized)
            elif isinstance(block, ThinkingBlock):
                sanitized = sanitize_text(block.thinking)
                if sanitized != block.thinking:
                    msg.content[i] = ThinkingBlock(thinking=sanitized)

    # Phase 2: Repair malformed tool_use blocks.
    # Drop tool_use blocks with empty names or missing IDs — the API
    # will reject them with 400 "invalid tool_use".
    for msg in messages:
        if msg.role != "assistant":
            continue
        repaired: list[ContentBlock] = []
        for block in msg.content:
            if isinstance(block, ToolUseBlock):
                if not block.name or not block.id:
                    logger.warning(
                        "Dropping malformed tool_use block: name=%r id=%r",
                        block.name, block.id,
                    )
                    continue
                # Ensure input is a dict (some models send a string).
                if not isinstance(block.input, dict):
                    block.input = {}
                repaired.append(block)
            else:
                repaired.append(block)
        msg.content = repaired

    # Phase 3: Drop empty messages (no content blocks at all).
    messages[:] = [msg for msg in messages if msg.content]

    # Phase 4: Fix role-alternation violations.
    # The Anthropic API requires strict user→assistant→user→... alternation.
    # Consecutive same-role messages cause HTTP 400.
    # Strategy: merge consecutive same-role messages.
    if len(messages) <= 1:
        return messages

    merged: list[ConversationMessage] = [messages[0]]
    for msg in messages[1:]:
        if msg.role == merged[-1].role:
            # Same role — merge content into the previous message.
            merged[-1].content.extend(msg.content)
            logger.debug(
                "Merged consecutive %s messages (%d + %d blocks)",
                msg.role, len(merged[-1].content) - len(msg.content), len(msg.content),
            )
        else:
            merged.append(msg)

    messages[:] = merged
    return messages


def strip_thinking_blocks(messages: list[ConversationMessage]) -> list[ConversationMessage]:
    """Strip thinking-only assistant turns and merge adjacent user messages.

    Ported from Hermes Agent's drop_thinking_only_and_merge_users.
    Used as a recovery action when the API returns "invalid thinking signature"
    errors after context compaction.

    Thinking-only assistant turns (containing only ThinkingBlock, no text or
    tool_use) cause "invalid signature" errors after compaction because the
    signature was computed against the full turn content. Dropping them and
    merging any newly-adjacent user messages preserves the role-alternation
    invariant.

    Args:
        messages: The message list to strip (modified in place).

    Returns:
        The stripped message list (same list, modified in place).
    """
    if not messages:
        return messages

    # Phase 1: Drop thinking-only assistant turns.
    kept = []
    for msg in messages:
        if msg.role == "assistant":
            has_text = any(
                isinstance(b, TextBlock) and b.text.strip()
                for b in msg.content
            )
            has_tool_use = any(
                isinstance(b, ToolUseBlock) for b in msg.content
            )
            if not has_text and not has_tool_use:
                # Thinking-only — drop it.
                continue
        kept.append(msg)

    # Phase 2: Merge any newly-adjacent user messages.
    if len(kept) <= 1:
        messages[:] = kept
        return messages

    merged: list[ConversationMessage] = [kept[0]]
    for msg in kept[1:]:
        if msg.role == merged[-1].role and msg.role == "user":
            # Merge adjacent user messages.
            merged[-1].content.extend(msg.content)
        else:
            merged.append(msg)

    messages[:] = merged
    return messages
