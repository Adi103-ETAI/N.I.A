"""Self-improving learning loop — background memory review.

The audit (P0 Task 5) flagged that NIA has no self-improving learning loop.
This is Hermes Agent's signature feature: after every turn, fork the agent
and ask "should any memory be saved?".

How it works
------------
1. After each turn, :meth:`QueryEngine.submit_message` calls
   :func:`maybe_spawn_background_review`.
2. The review runs in a daemon thread (doesn't block the main conversation).
3. The thread makes a separate LLM call with the conversation snapshot + a
   review prompt asking "is anything worth saving to memory?".
4. If the LLM responds with structured memory writes, they're applied to
   NIA's memory system.
5. The main conversation is never touched — the review runs entirely
   out-of-band.

Configuration
-------------
- ``NIA_BACKGROUND_REVIEW`` env var: ``"0"`` / ``"false"`` / ``"off"`` disables.
- ``NIA_BACKGROUND_REVIEW_MODEL`` env var: override the model used for review
  (default: inherit the main agent's model).
- ``NIA_BACKGROUND_REVIEW_INTERVAL`` env var: minimum seconds between reviews
  (default: 30 — prevents review spam on rapid turns).

Memory-only for now
-------------------
This initial implementation handles MEMORY only (no skill creation). The
LLM is asked to identify:
- User preferences (how they want NIA to behave)
- User facts (who they are, what they're working on)
- Patterns (recurring tasks, workflows)

Skill creation will be added in a follow-up — it requires the LLM to call
``skill_manage`` which needs a more sophisticated tool-calling setup.

Reference: Hermes Agent's ``agent/background_review.py``. Hermes forks the
full AIAgent for the review; NIA's version is simpler — a direct LLM call
with the conversation snapshot, no fork. The review prompt is adapted from
Hermes's ``_MEMORY_REVIEW_PROMPT``.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Review prompt (adapted from Hermes's _MEMORY_REVIEW_PROMPT)
# ---------------------------------------------------------------------------

MEMORY_REVIEW_PROMPT = """\
Review the conversation above and consider saving to memory if appropriate.

Focus on:
1. Has the user revealed things about themselves — their persona, desires, \
preferences, or personal details worth remembering?
2. Has the user expressed expectations about how you should behave, your work \
style, or ways they want you to operate?
3. Are there recurring patterns in what the user asks for, or how they \
phrase their requests?

If something stands out, respond with a JSON object matching this schema:

{
  "memories": [
    {
      "category": "preference" | "fact" | "pattern",
      "content": "The memory text to save (concise, factual)",
      "key": "Optional key for preferences (e.g. 'tone', 'verbosity')"
    }
  ],
  "summary": "One-line summary of what you saved, or 'Nothing to save.'"
}

Rules:
- Only save DURABLE information — things that will matter in future sessions.
- Do NOT save transient details (the current file being edited, this \
session's bug) unless they reveal a pattern.
- Do NOT save things the user can trivially rediscover (file paths, \
command syntax).
- Preferences need a 'key' so they can be updated (e.g. \
{"category":"preference","key":"verbosity","content":"User prefers \
concise answers without preamble"}).
- Facts and patterns don't need a key.
- If nothing is worth saving, respond with {"memories":[],"summary":\
"Nothing to save."}.

Respond with ONLY the JSON object, no other text."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def is_background_review_enabled() -> bool:
    """Return True if background review is enabled (default: True)."""
    val = os.environ.get("NIA_BACKGROUND_REVIEW", "").strip().lower()
    return val not in ("0", "false", "off", "no", "disabled")


def get_review_model() -> str | None:
    """Return the model to use for review, or None to inherit the main model."""
    return os.environ.get("NIA_BACKGROUND_REVIEW_MODEL") or None


def get_review_interval() -> float:
    """Return the minimum seconds between reviews (default: 30)."""
    try:
        return float(os.environ.get("NIA_BACKGROUND_REVIEW_INTERVAL", "30"))
    except ValueError:
        return 30.0


# ---------------------------------------------------------------------------
# Review state (process-wide)
# ---------------------------------------------------------------------------


class _ReviewState:
    """Tracks review timing to prevent spam."""

    def __init__(self) -> None:
        self._last_review_time: float = 0.0
        self._lock = threading.Lock()
        self._active_threads: list[threading.Thread] = []

    def should_review(self) -> bool:
        """Return True if enough time has passed since the last review."""
        interval = get_review_interval()
        with self._lock:
            if time.monotonic() - self._last_review_time < interval:
                return False
            self._last_review_time = time.monotonic()
            return True

    def register_thread(self, thread: threading.Thread) -> None:
        with self._lock:
            self._active_threads.append(thread)
            # Clean up dead threads.
            self._active_threads = [t for t in self._active_threads if t.is_alive()]

    def active_count(self) -> int:
        with self._lock:
            return sum(1 for t in self._active_threads if t.is_alive())


_review_state = _ReviewState()


def get_review_state() -> _ReviewState:
    """Return the process-wide review state."""
    return _review_state


# ---------------------------------------------------------------------------
# Conversation snapshot
# ---------------------------------------------------------------------------


def _snapshot_messages(messages: list) -> list[dict[str, Any]]:
    """Convert ConversationMessage list to a serializable snapshot.

    Truncates long tool results to keep the snapshot small. Limits to the
    last 20 messages to bound the review cost.
    """
    snapshot: list[dict[str, Any]] = []
    # Take last 20 messages to bound cost.
    for msg in messages[-20:]:
        role = getattr(msg, "role", "unknown")
        content_blocks = []
        for block in getattr(msg, "content", []):
            cls = block.__class__.__name__
            if cls == "TextBlock":
                content_blocks.append({"type": "text", "text": block.text})
            elif cls == "ToolUseBlock":
                content_blocks.append(
                    {"type": "tool_use", "name": block.name, "input": block.input}
                )
            elif cls == "ToolResultBlock":
                # Truncate long tool results.
                content = block.content
                if len(content) > 500:
                    content = content[:500] + "... [truncated]"
                content_blocks.append(
                    {"type": "tool_result", "content": content, "is_error": block.is_error}
                )
        snapshot.append({"role": role, "content": content_blocks})
    return snapshot


# ---------------------------------------------------------------------------
# Review execution
# ---------------------------------------------------------------------------


def _parse_review_response(text: str) -> dict[str, Any]:
    """Parse the LLM's review response.

    Expects a JSON object with 'memories' and 'summary' fields. Tolerates
    markdown code fences and leading/trailing whitespace.
    """
    text = text.strip()
    # Strip markdown code fences if present.
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Try to find a JSON object in the text.
    import re

    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    return {"memories": [], "summary": f"Could not parse review response: {text[:200]}"}


def _apply_memory_writes(
    review_result: dict[str, Any],
    memory: Any,
) -> int:
    """Apply memory writes from the review result. Returns count applied."""
    memories = review_result.get("memories", [])
    if not isinstance(memories, list):
        return 0

    count = 0
    for mem in memories:
        if not isinstance(mem, dict):
            continue
        category = mem.get("category", "fact")
        content = mem.get("content", "")
        key = mem.get("key")

        if not content:
            continue

        try:
            if category == "preference" and key:
                if hasattr(memory, "add_preference"):
                    memory.add_preference(key, content)
                    count += 1
            elif category == "pattern":
                if hasattr(memory, "add_pattern"):
                    memory.add_pattern(content)
                    count += 1
            else:  # fact or unknown
                if hasattr(memory, "add_fact"):
                    memory.add_fact(content)
                    count += 1
        except Exception as exc:
            logger.warning("Failed to apply memory write %r: %s", mem, exc)

    # Persist memory if the memory object has a save method.
    if count > 0 and hasattr(memory, "save"):
        try:
            memory.save()
        except Exception as exc:
            logger.warning("Failed to persist memory after review: %s", exc)

    return count


def _run_review(
    messages_snapshot: list[dict[str, Any]],
    api_client: Any,
    model: str,
    system_prompt: str,
    memory: Any,
) -> dict[str, Any]:
    """Run a single review and apply any memory writes.

    Returns a dict with: 'applied' (count), 'summary' (str), 'error' (str|None).
    """
    # Build the review conversation: system prompt + conversation snapshot + review prompt.
    review_messages = []

    # Convert the snapshot into a single user message containing the transcript.
    transcript_lines: list[str] = []
    for msg in messages_snapshot:
        role = msg.get("role", "unknown")
        for block in msg.get("content", []):
            btype = block.get("type", "")
            if btype == "text":
                transcript_lines.append(f"{role}: {block.get('text', '')}")
            elif btype == "tool_use":
                transcript_lines.append(
                    f"{role}: [called tool {block.get('name', '?')}]"
                )
            elif btype == "tool_result":
                content = block.get("content", "")
                transcript_lines.append(f"[tool result] {content[:200]}")
    transcript = "\n".join(transcript_lines)

    review_messages.append(
        {
            "role": "user",
            "content": f"Here is the conversation to review:\n\n{transcript}\n\n{MEMORY_REVIEW_PROMPT}",
        }
    )

    # Make the LLM call.
    try:
        # Use the streaming API and collect the full text.
        import asyncio

        async def _do_call():
            from niaharness.api.client import ApiMessageRequest

            full_text = ""
            async for event in api_client.stream_message(
                ApiMessageRequest(
                    model=model,
                    messages=[
                        # Reconstruct minimal ConversationMessage list.
                        # The api_client expects ConversationMessage objects,
                        # but we built plain dicts. Build them properly.
                        _make_review_user_message(transcript),
                    ],
                    system_prompt=system_prompt,
                    max_tokens=1024,
                    tools=[],  # no tools for the review
                )
            ):
                from niaharness.api.client import ApiTextDeltaEvent, ApiMessageCompleteEvent

                if isinstance(event, ApiTextDeltaEvent):
                    full_text += event.text
                elif isinstance(event, ApiMessageCompleteEvent):
                    if not full_text and event.message:
                        full_text = event.message.text
            return full_text

        # Run in a new event loop since we're in a thread.
        loop = asyncio.new_event_loop()
        try:
            full_text = loop.run_until_complete(_do_call())
        finally:
            loop.close()

    except Exception as exc:
        logger.warning("Background review LLM call failed: %s", exc)
        return {"applied": 0, "summary": "", "error": str(exc)}

    # Parse and apply.
    review_result = _parse_review_response(full_text)
    applied = _apply_memory_writes(review_result, memory)
    return {
        "applied": applied,
        "summary": review_result.get("summary", ""),
        "error": None,
    }


def _make_review_user_message(transcript: str):
    """Build a ConversationMessage for the review call."""
    from niaharness.engine.messages import ConversationMessage, TextBlock

    return ConversationMessage(
        role="user",
        content=[TextBlock(text=f"Here is the conversation to review:\n\n{transcript}\n\n{MEMORY_REVIEW_PROMPT}")],
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def maybe_spawn_background_review(
    messages: list,
    api_client: Any,
    model: str,
    system_prompt: str,
    memory: Any,
) -> None:
    """Maybe spawn a background review thread after a turn.

    Non-blocking: returns immediately after spawning (or deciding not to).
    Never raises — review failures are logged but don't affect the main
    conversation.

    Args:
        messages: The current conversation messages (ConversationMessage list).
        api_client: The API client to use for the review call.
        model: The model to use for the review.
        system_prompt: The system prompt for context.
        memory: The Memory instance to write to (must have add_preference,
            add_pattern, add_fact, save methods).
    """
    if not is_background_review_enabled():
        return
    if memory is None:
        return
    if not _review_state.should_review():
        return

    # Snapshot the messages now (the live list will keep mutating).
    snapshot = _snapshot_messages(messages)
    if len(snapshot) < 2:
        # Not enough conversation to review.
        return

    review_model = get_review_model() or model

    def _target():
        try:
            result = _run_review(
                snapshot, api_client, review_model, system_prompt, memory
            )
            if result.get("error"):
                logger.debug("Background review error: %s", result["error"])
            elif result.get("applied", 0) > 0:
                logger.info(
                    "Background review saved %d memory item(s): %s",
                    result["applied"],
                    result.get("summary", ""),
                )
            else:
                logger.debug("Background review: %s", result.get("summary", "nothing saved"))
        except Exception as exc:
            logger.warning("Background review thread failed: %s", exc)

    thread = threading.Thread(target=_target, daemon=True, name="nia-background-review")
    _review_state.register_thread(thread)
    thread.start()


def wait_for_reviews(timeout: float = 5.0) -> None:
    """Wait for active review threads to finish (for tests).

    Blocks up to ``timeout`` seconds. Useful in tests to ensure the review
    has completed before asserting on memory state.
    """
    import time as _time

    deadline = _time.monotonic() + timeout
    while _time.monotonic() < deadline:
        if _review_state.active_count() == 0:
            return
        _time.sleep(0.05)


def get_review_stats() -> dict[str, Any]:
    """Return review system statistics (for debugging/UI)."""
    return {
        "enabled": is_background_review_enabled(),
        "model": get_review_model(),
        "interval_seconds": get_review_interval(),
        "active_threads": _review_state.active_count(),
        "last_review_time": datetime.fromtimestamp(
            _review_state._last_review_time, tz=timezone.utc
        ).isoformat()
        if _review_state._last_review_time > 0
        else None,
    }
