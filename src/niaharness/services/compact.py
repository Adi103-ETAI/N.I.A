"""Token estimation and conversation auto-compaction.

This module provides:
- Heuristic token counters (no tiktoken dependency required).
- A summarizer that flattens the recent message tail into a compact text blob.
- A `compact_messages` helper that replaces older messages with a single
  summary message and preserves the most recent N messages verbatim.
- An `AutoCompactState` / `auto_compact_if_needed` pair used by the query
  engine to keep conversations under the model's context window.

The implementation is intentionally simple and dependency-light.  When the
real model reports a `prompt is too long` error, the engine will already
surface it via the recovery path in `query.py`; here we only do opportunistic
compaction to keep the conversation comfortable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from niaharness.engine.messages import ConversationMessage, TextBlock

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Roughly 4 characters per token (matches OpenAI's published heuristic for
# English text).  This is deliberately conservative so we trigger compaction
# slightly before the model would actually reject the prompt.
_CHARS_PER_TOKEN = 4

# Per-message overhead (role tag + framing).  Conservative; keeps estimates
# aligned with the OpenAI "every message follows <im_start>{role}\n{content}<im_end>\n"
# framing.
_PER_MESSAGE_OVERHEAD = 4

# Default auto-compact thresholds.  Real models range from 8K to 200K context;
# we pick a single conservative threshold so the loop has a single source of
# truth.  When the model name is recognised (claude-3-opus, gpt-4, etc.) we
# bump the threshold accordingly — see `_threshold_for_model`.
_DEFAULT_AUTOCOMPACT_THRESHOLD = 32_000
_LARGE_MODEL_AUTOCOMPACT_THRESHOLD = 100_000

# How many recent messages to always preserve verbatim during full compaction.
_DEFAULT_PRESERVE_RECENT = 6

# Microcompact: strip tool_result content from messages older than this many
# turns back, since the model has already seen and reasoned about them.
_MICROCOMPACT_KEEP_RECENT = 4


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------


def estimate_tokens(text: str) -> int:
    """Return a rough token count for ``text``.

    Uses the 4-chars-per-token heuristic.  Empty strings return 0.
    """
    if not text:
        return 0
    return max(1, len(text) // _CHARS_PER_TOKEN)


def estimate_message_tokens(content: Sequence[str] | ConversationMessage) -> int:
    """Return a rough token count for either a sequence of strings or a message.

    Accepts:
    - A list/tuple of strings (sums each string's tokens, no overhead).
    - A ConversationMessage (sums all text blocks + per-message overhead).
    """
    if isinstance(content, ConversationMessage):
        texts = [block.text for block in content.content if isinstance(block, TextBlock)]
        if not texts:
            return 0
        return sum(estimate_tokens(t) for t in texts) + _PER_MESSAGE_OVERHEAD
    # Plain sequence of strings: no per-message overhead (legacy contract).
    return sum(estimate_tokens(t) for t in content)


def estimate_conversation_tokens(messages: Sequence[ConversationMessage]) -> int:
    """Return a rough token count for the whole conversation."""
    if not messages:
        return 0
    return sum(estimate_message_tokens(m) for m in messages)


# ---------------------------------------------------------------------------
# Summarisation
# ---------------------------------------------------------------------------


def summarize_messages(
    messages: Sequence[ConversationMessage],
    *,
    max_messages: int = 10,
) -> str:
    """Return a flat text summary of the most recent ``max_messages`` messages.

    The summary uses ``role: text`` lines so downstream prompts can search for
    specific utterances.  Tool-use blocks are rendered as ``[tool_use name=X]``
    placeholders to avoid bloating the summary with raw JSON.
    """
    if not messages:
        return ""
    tail = list(messages[-max_messages:]) if max_messages > 0 else list(messages)
    lines: list[str] = []
    for msg in tail:
        for block in msg.content:
            if isinstance(block, TextBlock):
                lines.append(f"{msg.role}: {block.text}")
            else:
                # Render tool_use / tool_result blocks compactly.
                lines.append(f"{msg.role}: [{block.type}]")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Compact
# ---------------------------------------------------------------------------


def compact_messages(
    messages: Sequence[ConversationMessage],
    *,
    preserve_recent: int = _DEFAULT_PRESERVE_RECENT,
) -> list[ConversationMessage]:
    """Replace older messages with a single summary message.

    Returns a new list of length ``min(len(messages), preserve_recent + 1)``:
    - The first element is a user message whose text is ``[conversation summary]\n<summary>``.
    - The remaining elements are the last ``preserve_recent`` original messages.
    """
    if not messages:
        return []
    if len(messages) <= preserve_recent:
        return list(messages)

    older = list(messages[:-preserve_recent])
    recent = list(messages[-preserve_recent:])

    summary_text = summarize_messages(older)
    summary_block = TextBlock(text=f"[conversation summary]\n{summary_text}")
    summary_message = ConversationMessage(role="user", content=[summary_block])
    return [summary_message, *recent]


def microcompact_messages(
    messages: Sequence[ConversationMessage],
    *,
    keep_recent: int = _MICROCOMPACT_KEEP_RECENT,
) -> list[ConversationMessage]:
    """Cheap compaction: clear old tool_result content to free tokens.

    Unlike ``compact_messages`` this preserves message structure and roles —
    it only shortens the ``content`` field of old messages by dropping tool
    result payloads the model has already seen.
    """
    if len(messages) <= keep_recent:
        return list(messages)
    out: list[ConversationMessage] = []
    cutoff = len(messages) - keep_recent
    for idx, msg in enumerate(messages):
        if idx >= cutoff:
            out.append(msg)
            continue
        # Drop ToolResultBlock content for older messages.
        new_content = []
        for block in msg.content:
            if block.__class__.__name__ == "ToolResultBlock":
                # Replace with a short placeholder; do not import here to avoid
                # circular imports if messages.py ever re-exports from us.
                new_content.append(TextBlock(text="[tool result elided by microcompact]"))
            else:
                new_content.append(block)
        out.append(ConversationMessage(role=msg.role, content=new_content))
    return out


# ---------------------------------------------------------------------------
# Auto-compact state machine
# ---------------------------------------------------------------------------


def _threshold_for_model(model: str | None) -> int:
    """Return the auto-compact threshold (in tokens) for ``model``.

    Larger context models tolerate more tokens before compaction kicks in.
    Unknown models fall back to the conservative default.
    """
    if not model:
        return _DEFAULT_AUTOCOMPACT_THRESHOLD
    lowered = model.lower()
    # 200K-class models
    if any(tag in lowered for tag in ("claude-3", "claude-sonnet", "claude-opus", "gpt-4o", "gpt-4-turbo", "gemini-1.5", "qwen2.5", "deepseek-v3")):
        return _LARGE_MODEL_AUTOCOMPACT_THRESHOLD
    return _DEFAULT_AUTOCOMPACT_THRESHOLD


@dataclass
class AutoCompactState:
    """Tracks compaction history across turns in a single query run.

    Fields:
    - compacted_count: number of full compactions performed.
    - microcompacted_count: number of microcompactions performed.
    - last_compacted_at_turn: turn index of the last compaction (or -1).
    - threshold_tokens: the threshold currently in effect for the model.
    """

    compacted_count: int = 0
    microcompacted_count: int = 0
    last_compacted_at_turn: int = -1
    threshold_tokens: int = _DEFAULT_AUTOCOMPACT_THRESHOLD
    _model_set: bool = field(default=False, repr=False)

    def maybe_set_threshold(self, model: str | None) -> None:
        """Set the threshold once based on the model name."""
        if self._model_set:
            return
        self.threshold_tokens = _threshold_for_model(model)
        self._model_set = True


async def auto_compact_if_needed(
    messages: Sequence[ConversationMessage],
    *,
    model: str | None,
    state: AutoCompactState,
    current_turn: int | None = None,
) -> tuple[list[ConversationMessage], bool, AutoCompactState]:
    """Run auto-compaction on the conversation if needed.

    Strategy:
    1. Estimate tokens.
    2. If over threshold, try microcompact first (cheap).
    3. If still over threshold, do a full compact.

    Returns ``(new_messages, was_compacted, state)``.  ``was_compacted`` is
    True if either micro- or full-compaction ran this call.
    """
    state.maybe_set_threshold(model)
    threshold = state.threshold_tokens

    current_tokens = estimate_conversation_tokens(messages)
    if current_tokens <= threshold:
        return list(messages), False, state

    # Try microcompact first.
    new_messages = microcompact_messages(messages)
    if estimate_conversation_tokens(new_messages) <= threshold:
        state.microcompacted_count += 1
        state.last_compacted_at_turn = current_turn if current_turn is not None else -1
        return new_messages, True, state

    # Full compact.
    new_messages = compact_messages(new_messages)
    state.compacted_count += 1
    state.last_compacted_at_turn = current_turn if current_turn is not None else -1
    return new_messages, True, state
