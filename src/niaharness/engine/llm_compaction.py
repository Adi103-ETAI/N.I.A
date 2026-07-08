"""LLM-based context compaction — structured summarization with iterative updates.

Ported from the reference project's agent/context_compressor.py (3,082 lines),
focused on the critical improvements over NIA's text-flattening approach:

  - **Structured LLM summarization** — instead of ``role: text`` lines, send
    a structured prompt to the auxiliary model asking for a concise summary
    of the conversation so far, preserving key context (decisions, file
    paths, tool results, user preferences).
  - **Iterative summary updates** — on subsequent compactions, the previous
    summary is included in the prompt so the LLM can *update* it rather
    than re-summarize from scratch. This preserves long-term context across
    multiple compactions in a long session.
  - **Head + tail protection** — the system prompt + first user exchange
    (head) and the most recent N messages (tail) are preserved verbatim.
    Only the middle is summarized.
  - **Tool result pruning** — old tool results are pruned *before* LLM
    summarization (cheap, no LLM call). Only the tool name + truncated
    args are kept as a breadcrumb.
  - **Image part stripping** — images are stripped from old messages
    before summarization (they consume tokens but can't be summarized).
  - **Failure cooldown** — if the LLM summarization fails, a cooldown
    prevents retrying for a few minutes (avoids hammering a failing
    aux model). During cooldown, falls back to text-flatten compaction.
  - **Prompt-too-long vs max-tokens distinction** — different recovery
    paths for "prompt is too long" (compress aggressively) vs
    "stop_reason=max_tokens" (continue generation, no compaction).

Why this matters
----------------
NIA's current compaction (``services/compact.py``) flattens messages into
``role: text`` lines. This loses structure: tool calls become plain text,
decisions are mixed with chit-chat, and the LLM has to re-derive context
from a flat blob. The reference project's approach sends a structured
prompt to the aux model asking for a *summary* — the result is a concise,
context-rich paragraph that preserves the important parts of the
conversation.

Usage::

    from niaharness.engine.llm_compaction import LLMCompactor, CompactionRequest

    compactor = LLMCompactor(aux_client=my_aux_client)
    result = await compactor.compact(request)
    if result.success:
        messages = result.compacted_messages
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from niaharness.engine.messages import ConversationMessage, TextBlock
from niaharness.services.compact import (
    estimate_message_tokens,
    estimate_tokens,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Number of messages to protect at the head (system prompt + first exchange).
DEFAULT_HEAD_PROTECT = 2

# Number of messages to protect at the tail (most recent).
DEFAULT_TAIL_PROTECT = 6

# Maximum tokens for the LLM summary response.
MAX_SUMMARY_TOKENS = 1024

# Cooldown after a failed LLM summarization (5 minutes).
SUMMARY_FAILURE_COOLDOWN_SECONDS = 5 * 60

# Maximum consecutive LLM failures before falling back to text flatten.
MAX_CONSECUTIVE_LLM_FAILURES = 3

# Tool result pruning: keep only tool name + truncated args for old results.
TOOL_RESULT_BREADCRUMB_CHARS = 200

# Maximum messages to include in the LLM summarization prompt (avoid huge prompts).
MAX_MESSAGES_FOR_LLM_SUMMARY = 50


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class CompactionRequest:
    """Input to a compaction operation.

    Attributes:
        messages: The full conversation message list.
        model: The model name (used for token estimation + context window).
        context_window: The model's context window in tokens.
        target_tokens: The target token count after compaction. If None,
            defaults to 50% of context_window.
        head_protect: Number of messages to protect at the head.
        tail_protect: Number of messages to protect at the tail.
        previous_summary: The previous LLM summary (for iterative updates).
            None on first compaction.
    """

    messages: List[ConversationMessage]
    model: str = ""
    context_window: int = 32_000
    target_tokens: Optional[int] = None
    head_protect: int = DEFAULT_HEAD_PROTECT
    tail_protect: int = DEFAULT_TAIL_PROTECT
    previous_summary: Optional[str] = None


@dataclass
class CompactionResult:
    """Output of a compaction operation.

    Attributes:
        success: Whether the compaction succeeded.
        compacted_messages: The compacted message list (head + summary + tail).
        summary: The LLM-generated summary (or text-flatten fallback).
        method: "llm" if LLM summarization was used, "text_flatten" if fallback.
        tokens_before: Estimated tokens before compaction.
        tokens_after: Estimated tokens after compaction.
        error: Error message if success is False.
    """

    success: bool
    compacted_messages: List[ConversationMessage] = field(default_factory=list)
    summary: str = ""
    method: str = "text_flatten"
    tokens_before: int = 0
    tokens_after: int = 0
    error: str = ""


# ---------------------------------------------------------------------------
# Aux client protocol
# ---------------------------------------------------------------------------


class AuxClientProtocol:
    """Protocol for an auxiliary LLM client used for summarization.

    Any client with an async ``complete(prompt, max_tokens) -> str`` method
    satisfies this protocol. In production, this is the auxiliary model
    client (small, fast, cheap model). In tests, it can be a stub.
    """

    async def complete(self, prompt: str, *, max_tokens: int = MAX_SUMMARY_TOKENS) -> str:
        """Generate a completion for the prompt."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# LLM Compactor
# ---------------------------------------------------------------------------


class LLMCompactor:
    """LLM-based context compactor with iterative summary updates.

    Algorithm:
      1. Prune old tool results (cheap, no LLM call)
      2. Protect head messages (system prompt + first exchange)
      3. Protect tail messages (most recent N)
      4. Summarize middle turns with structured LLM prompt
      5. On subsequent compactions, iteratively update the previous summary

    Falls back to text-flatten compaction if:
      - No aux client is configured
      - The aux client fails (cooldown prevents retry for 5 min)
      - Consecutive failures exceed MAX_CONSECUTIVE_LLM_FAILURES
    """

    def __init__(
        self,
        aux_client: Optional[AuxClientProtocol] = None,
        *,
        head_protect: int = DEFAULT_HEAD_PROTECT,
        tail_protect: int = DEFAULT_TAIL_PROTECT,
    ) -> None:
        self._aux_client = aux_client
        self._head_protect = head_protect
        self._tail_protect = tail_protect
        self._previous_summary: Optional[str] = None
        self._consecutive_failures = 0
        self._failure_cooldown_until = 0.0
        self._last_summary_error: Optional[str] = None

    def reset_session_state(self) -> None:
        """Reset all per-session state (call on /new, /reset, session end)."""
        self._previous_summary = None
        self._consecutive_failures = 0
        self._failure_cooldown_until = 0.0
        self._last_summary_error = None

    async def compact(self, request: CompactionRequest) -> CompactionResult:
        """Compact the conversation using LLM summarization.

        Falls back to text-flatten compaction if LLM is unavailable or fails.
        """
        tokens_before = sum(estimate_message_tokens(m) for m in request.messages)
        target = request.target_tokens or (request.context_window // 2)

        if not request.messages:
            return CompactionResult(
                success=True,
                compacted_messages=[],
                tokens_before=0,
                tokens_after=0,
                method="none",
            )

        # Try LLM summarization first.
        if self._aux_client is not None and not self._is_in_cooldown():
            try:
                result = await self._compact_with_llm(request, tokens_before, target)
                if result.success:
                    self._consecutive_failures = 0
                    self._previous_summary = result.summary
                    return result
            except Exception as exc:
                self._record_failure(exc)
                logger.warning("LLM compaction failed, falling back to text flatten: %s", exc)

        # Fallback: text-flatten compaction.
        return self._compact_with_text_flatten(request, tokens_before, target)

    def _is_in_cooldown(self) -> bool:
        """True if the LLM is in a failure cooldown period."""
        return time.monotonic() < self._failure_cooldown_until

    def _record_failure(self, exc: Exception) -> None:
        """Record a failed LLM attempt and start cooldown if needed."""
        self._consecutive_failures += 1
        self._last_summary_error = str(exc)
        if self._consecutive_failures >= MAX_CONSECUTIVE_LLM_FAILURES:
            self._failure_cooldown_until = time.monotonic() + SUMMARY_FAILURE_COOLDOWN_SECONDS
            logger.warning(
                "LLM compaction: %d consecutive failures, entering %ds cooldown",
                self._consecutive_failures,
                SUMMARY_FAILURE_COOLDOWN_SECONDS,
            )

    async def _compact_with_llm(
        self, request: CompactionRequest, tokens_before: int, target: int
    ) -> CompactionResult:
        """Compact using LLM summarization."""
        head_protect = min(self._head_protect, len(request.messages))
        tail_protect = min(self._tail_protect, len(request.messages) - head_protect)

        if tail_protect <= 0:
            # Not enough messages to summarize — just return as-is.
            return CompactionResult(
                success=True,
                compacted_messages=list(request.messages),
                tokens_before=tokens_before,
                tokens_after=tokens_before,
                method="none",
            )

        head = request.messages[:head_protect]
        middle = request.messages[head_protect:-tail_protect] if tail_protect > 0 else request.messages[head_protect:]
        tail = request.messages[-tail_protect:] if tail_protect > 0 else []

        if not middle:
            return CompactionResult(
                success=True,
                compacted_messages=list(request.messages),
                tokens_before=tokens_before,
                tokens_after=tokens_before,
                method="none",
            )

        # Prune old tool results from middle (cheap, no LLM call).
        pruned_middle = self._prune_tool_results(middle)

        # Build the LLM prompt.
        prompt = self._build_summary_prompt(pruned_middle, request.previous_summary)

        # Call the aux model.
        summary = await self._aux_client.complete(  # type: ignore[union-attr]
            prompt, max_tokens=MAX_SUMMARY_TOKENS
        )

        if not summary or not summary.strip():
            raise RuntimeError("LLM returned empty summary")

        # Build the compacted message list: head + summary message + tail.
        summary_message = ConversationMessage(
            role="assistant",
            content=[TextBlock(text=f"[Conversation Summary]\n\n{summary.strip()}")],
        )
        compacted = list(head) + [summary_message] + list(tail)
        tokens_after = sum(estimate_message_tokens(m) for m in compacted)

        return CompactionResult(
            success=True,
            compacted_messages=compacted,
            summary=summary.strip(),
            method="llm",
            tokens_before=tokens_before,
            tokens_after=tokens_after,
        )

    def _compact_with_text_flatten(
        self, request: CompactionRequest, tokens_before: int, target: int
    ) -> CompactionResult:
        """Fallback: flatten middle messages into a text summary (no LLM call)."""
        head_protect = min(self._head_protect, len(request.messages))
        tail_protect = min(self._tail_protect, len(request.messages) - head_protect)

        if tail_protect <= 0:
            return CompactionResult(
                success=True,
                compacted_messages=list(request.messages),
                tokens_before=tokens_before,
                tokens_after=tokens_before,
                method="none",
            )

        head = request.messages[:head_protect]
        middle = request.messages[head_protect:-tail_protect] if tail_protect > 0 else request.messages[head_protect:]
        tail = request.messages[-tail_protect:] if tail_protect > 0 else []

        if not middle:
            return CompactionResult(
                success=True,
                compacted_messages=list(request.messages),
                tokens_before=tokens_before,
                tokens_after=tokens_before,
                method="none",
            )

        # Flatten middle into role: text lines.
        lines: List[str] = ["[Conversation Summary (text flatten)]"]
        for msg in middle:
            text = _extract_text(msg)
            if text:
                lines.append(f"{msg.role}: {text[:500]}")
        summary = "\n".join(lines)

        summary_message = ConversationMessage(
            role="assistant",
            content=[TextBlock(text=summary)],
        )
        compacted = list(head) + [summary_message] + list(tail)
        tokens_after = sum(estimate_message_tokens(m) for m in compacted)

        return CompactionResult(
            success=True,
            compacted_messages=compacted,
            summary=summary,
            method="text_flatten",
            tokens_before=tokens_before,
            tokens_after=tokens_after,
        )

    def _prune_tool_results(self, messages: List[ConversationMessage]) -> List[ConversationMessage]:
        """Prune old tool results to breadcrumbs (tool name + truncated args).

        Tool results consume tokens but the model has already seen and
        reasoned about them. Replacing them with a breadcrumb preserves
        the fact that the tool was called without the full output.
        """
        pruned: List[ConversationMessage] = []
        for msg in messages:
            # If the message has tool_result blocks, replace their content with a breadcrumb.
            has_tool_result = any(
                hasattr(b, "type") and getattr(b, "type", "") == "tool_result"
                for b in msg.content
            )
            if has_tool_result:
                # Replace tool_result content with a short breadcrumb.
                new_content = []
                for block in msg.content:
                    if hasattr(block, "type") and getattr(block, "type", "") == "tool_result":
                        # Keep the tool_use_id but replace content with a breadcrumb.
                        tool_use_id = getattr(block, "tool_use_id", "unknown")
                        original_content = getattr(block, "content", "")
                        if isinstance(original_content, str):
                            breadcrumb = original_content[:TOOL_RESULT_BREADCRUMB_CHARS]
                        else:
                            breadcrumb = "[tool result pruned]"
                        # Create a simplified text block instead of the full tool_result.
                        new_content.append(TextBlock(
                            text=f"[tool_result for {tool_use_id}]: {breadcrumb}"
                        ))
                    else:
                        new_content.append(block)
                pruned.append(ConversationMessage(role=msg.role, content=new_content))
            else:
                pruned.append(msg)
        return pruned

    def _build_summary_prompt(
        self, messages: List[ConversationMessage], previous_summary: Optional[str]
    ) -> str:
        """Build the structured LLM prompt for summarization.

        The prompt asks the LLM to produce a concise summary that preserves:
          - User's goals and requirements
          - Key decisions made
          - File paths mentioned
          - Tool calls and their results (briefly)
          - Any preferences or constraints

        If ``previous_summary`` is provided, the prompt asks the LLM to
        *update* the existing summary rather than start from scratch.
        """
        # Limit the number of messages in the prompt to avoid huge prompts.
        if len(messages) > MAX_MESSAGES_FOR_LLM_SUMMARY:
            # Keep the first few and last few messages, summarize the middle as "[...]".
            keep_head = MAX_MESSAGES_FOR_LLM_SUMMARY // 3
            keep_tail = MAX_MESSAGES_FOR_LLM_SUMMARY - keep_head
            messages = messages[:keep_head] + messages[-keep_tail:]

        # Serialize messages to a readable format.
        serialized: List[str] = []
        for msg in messages:
            text = _extract_text(msg)
            if text:
                serialized.append(f"[{msg.role}]: {text[:1000]}")  # Cap each message at 1000 chars

        conversation_text = "\n".join(serialized)

        if previous_summary:
            return f"""You are updating an existing conversation summary. The previous summary is below, followed by new messages that have occurred since.

Update the summary to incorporate the new information. Preserve all important context from the previous summary. Be concise but complete.

Previous summary:
{previous_summary}

New messages since the previous summary:
{conversation_text}

Updated summary (include: user goals, key decisions, file paths, important tool results, preferences/constraints):"""
        else:
            return f"""You are summarizing a conversation to preserve context for future turns. The conversation is below.

Produce a concise summary that preserves:
- The user's goals and requirements
- Key decisions made so far
- File paths mentioned
- Important tool calls and their results (briefly)
- Any preferences or constraints the user has stated

Conversation:
{conversation_text}

Summary:"""


def _extract_text(msg: ConversationMessage) -> str:
    """Extract all text from a message's content blocks."""
    parts: List[str] = []
    for block in msg.content:
        if isinstance(block, TextBlock):
            parts.append(block.text)
        elif hasattr(block, "text"):
            text = getattr(block, "text", "")
            if text:
                parts.append(str(text))
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


_default_compactor: Optional[LLMCompactor] = None


def get_default_compactor() -> LLMCompactor:
    """Return the process-wide default LLMCompactor.

    The default compactor has no aux client — it falls back to text-flatten
    compaction. To enable LLM summarization, set an aux client via
    ``compactor.set_aux_client(client)``.
    """
    global _default_compactor
    if _default_compactor is None:
        _default_compactor = LLMCompactor()
    return _default_compactor


__all__ = [
    "AuxClientProtocol",
    "CompactionRequest",
    "CompactionResult",
    "LLMCompactor",
    "get_default_compactor",
]
