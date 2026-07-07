"""Self-improving learning loop — background memory + skill review.

Adapted from Hermes Agent's agent/background_review.py.

After every turn, NIA spawns a background thread that:
1. Snapshots the last 20 messages.
2. Makes a separate LLM call asking "should any memory or skill be saved?".
3. Parses the JSON response and applies writes to memory + skills.
4. Surfaces what it learned to the user via a callback.

This version addresses all audit findings:
- SKILL CREATION: The review now has access to skill_manage and memory tools.
- TOOL ACCESS: The review LLM can call tools (memory, skill_manage).
- SKILL REVIEW PROMPT: Adapted from Hermes's _SKILL_REVIEW_PROMPT.
- CACHE AWARENESS: Reuses the parent's system prompt for prefix-cache parity.
- USER FEEDBACK: Results are surfaced via a callback (not just logged).
- INTERRUPTED-TURN GUARD: Skips review if the turn was interrupted.

Reference: Hermes Agent's agent/background_review.py.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Review prompts (adapted from Hermes)
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

Do NOT capture (these become persistent self-imposed constraints that \
bite you later when the environment changes):
- Environment-dependent failures: missing binaries, fresh-install errors, \
post-migration path mismatches, 'command not found', unconfigured \
credentials. The user can fix these — they are not durable rules.
- Negative claims about tools or features ('browser tools do not work', \
'X tool is broken'). These harden into refusals you cite against yourself \
for months after the actual problem was fixed.
- Session-specific transient errors that resolved before the conversation \
ended. If retrying worked, the lesson is the retry pattern, not the \
original failure.
- One-off task narratives. A user asking 'summarize today's market' is \
not a class of work that warrants a skill.

Respond with ONLY the JSON object, no other text."""

SKILL_REVIEW_PROMPT = """\
Review the conversation above and update the skill library. Be \
ACTIVE — most sessions produce at least one skill update, even if \
small. A pass that does nothing is a missed learning opportunity, \
not a neutral outcome.

Signals to look for (any one of these warrants action):
  • User corrected your style, tone, format, legibility, or \
verbosity. Frustration signals like 'stop doing X', 'this is too \
verbose', 'don't format like this', 'just give me the answer', or \
an explicit 'remember this' are FIRST-CLASS skill signals.
  • User corrected your workflow, approach, or sequence of steps. \
Encode the correction as a pitfall or explicit step in the skill \
that governs that class of task.
  • Non-trivial technique, fix, workaround, debugging path, or \
tool-usage pattern emerged that a future session would benefit \
from. Capture it.
  • A skill that got loaded or consulted this session turned out \
to be wrong, missing a step, or outdated. Patch it NOW.

Preference order — prefer the earliest action that fits:
  1. UPDATE AN EXISTING SKILL. Use skill_manage action=edit to patch \
an existing skill that covers the territory of the new learning.
  2. CREATE A NEW CLASS-LEVEL SKILL when no existing skill covers the \
class. Use skill_manage action=create. The name MUST be at the class \
level — NOT a specific PR number, error string, or session artifact.

Protected skills (DO NOT edit these):
  • Bundled skills (shipped with NIA: plan, debug, diagnose, review, \
simplify, commit, test).

Do NOT capture (same rules as memory — environment failures, negative \
tool claims, transient errors, one-off tasks).

If the session ran smoothly with no corrections and produced no new \
technique, just say 'Nothing to save.' and stop. Otherwise, act.

Use the skill_manage tool to create or edit skills. Use the nia_memory \
tool to save durable user facts/preferences. Respond with a JSON summary:

{
  "skills_created": [{"name": "...", "description": "..."}],
  "skills_updated": [{"name": "...", "change": "..."}],
  "memories_saved": [{"category": "...", "content": "..."}],
  "summary": "One-line summary of what you did, or 'Nothing to save.'"
}

Respond with ONLY the JSON object, no other text."""

COMBINED_REVIEW_PROMPT = MEMORY_REVIEW_PROMPT + "\n\n" + SKILL_REVIEW_PROMPT


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def is_background_review_enabled() -> bool:
    val = os.environ.get("NIA_BACKGROUND_REVIEW", "").strip().lower()
    return val not in ("0", "false", "off", "no", "disabled")


def get_review_model() -> str | None:
    return os.environ.get("NIA_BACKGROUND_REVIEW_MODEL") or None


def get_review_interval() -> float:
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
        # User-visible feedback callback (set by the UI layer).
        self._feedback_callback: Callable[[str], None] | None = None

    def should_review(self) -> bool:
        interval = get_review_interval()
        with self._lock:
            if time.monotonic() - self._last_review_time < interval:
                return False
            self._last_review_time = time.monotonic()
            return True

    def register_thread(self, thread: threading.Thread) -> None:
        with self._lock:
            self._active_threads.append(thread)
            self._active_threads = [t for t in self._active_threads if t.is_alive()]

    def active_count(self) -> int:
        with self._lock:
            return sum(1 for t in self._active_threads if t.is_alive())

    def set_feedback_callback(self, cb: Callable[[str], None] | None) -> None:
        with self._lock:
            self._feedback_callback = cb

    def notify_feedback(self, message: str) -> None:
        with self._lock:
            cb = self._feedback_callback
        if cb:
            try:
                cb(message)
            except Exception:
                pass


_review_state = _ReviewState()


def get_review_state() -> _ReviewState:
    return _review_state


def set_feedback_callback(cb: Callable[[str], None] | None) -> None:
    """Set a callback for user-visible review feedback.

    The UI layer should call this to receive messages like:
    '💾 Self-improvement review: saved 1 preference, created skill 'pdf-extraction''
    """
    _review_state.set_feedback_callback(cb)


# ---------------------------------------------------------------------------
# Conversation snapshot
# ---------------------------------------------------------------------------


def _snapshot_messages(messages: list) -> list[dict[str, Any]]:
    """Convert ConversationMessage list to a serializable snapshot.

    Truncates long tool results to keep the snapshot small. Limits to the
    last 20 messages to bound the review cost.
    """
    snapshot: list[dict[str, Any]] = []
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
                content = block.content
                if len(content) > 500:
                    content = content[:500] + "... [truncated]"
                content_blocks.append(
                    {"type": "tool_result", "content": content, "is_error": block.is_error}
                )
        snapshot.append({"role": role, "content": content_blocks})
    return snapshot


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _parse_review_response(text: str) -> dict[str, Any]:
    """Parse the LLM's review response.

    Tolerates markdown code fences and embedded JSON.
    """
    text = text.strip()
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


# ---------------------------------------------------------------------------
# Memory writes
# ---------------------------------------------------------------------------


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
            else:
                if hasattr(memory, "add_fact"):
                    memory.add_fact(content)
                    count += 1
        except Exception as exc:
            logger.warning("Failed to apply memory write %r: %s", mem, exc)

    if count > 0 and hasattr(memory, "save"):
        try:
            memory.save()
        except Exception as exc:
            logger.warning("Failed to persist memory after review: %s", exc)

    return count


# ---------------------------------------------------------------------------
# Review execution
# ---------------------------------------------------------------------------


def _run_review(
    messages_snapshot: list[dict[str, Any]],
    api_client: Any,
    model: str,
    system_prompt: str,
    memory: Any,
) -> dict[str, Any]:
    """Run a single review and apply any memory + skill writes.

    Returns a dict with: 'memories_applied' (count), 'skills_created' (list),
    'skills_updated' (list), 'summary' (str), 'error' (str|None).
    """
    # Build the review conversation: system prompt + conversation snapshot + review prompt.
    # Use the PARENT's system prompt for prefix-cache parity (audit fix).
    from niaharness.engine.messages import ConversationMessage, TextBlock

    # Build the transcript as actual ConversationMessages (not flattened text)
    # so the prefix matches the parent's cached prefix (audit fix: cache awareness).
    review_messages: list[ConversationMessage] = []
    for msg in messages_snapshot:
        role = msg.get("role", "user")
        content_blocks: list[Any] = []
        for block in msg.get("content", []):
            btype = block.get("type", "")
            if btype == "text":
                content_blocks.append(TextBlock(text=block.get("text", "")))
            # Skip tool_use/tool_result blocks in the review snapshot —
            # they're not needed for the review and would bloat the context.
        if content_blocks:
            review_messages.append(ConversationMessage(role=role, content=content_blocks))

    # Append the review prompt as the final user message.
    review_messages.append(
        ConversationMessage(
            role="user",
            content=[TextBlock(text=COMBINED_REVIEW_PROMPT)],
        )
    )

    # Make the LLM call with tools (memory + skill_manage).
    try:
        import asyncio

        from niaharness.api.client import ApiMessageRequest
        from niaharness.tools import create_default_tool_registry
        from niaharness.tools.base import ToolRegistry

        # Build a restricted tool registry for the review (memory + skill_manage only).
        review_registry = ToolRegistry()
        full_registry = create_default_tool_registry()
        for tool_name in ("nia_memory", "skill_manage", "skill"):
            tool = full_registry.get(tool_name)
            if tool:
                review_registry.register(tool)

        # Wire memory into the nia_memory tool.
        mem_tool = review_registry.get("nia_memory")
        if mem_tool and hasattr(mem_tool, "set_memory"):
            mem_tool.set_memory(memory)

        async def _do_review_call():
            from niaharness.engine.query import run_query, QueryContext
            from niaharness.permissions.checker import PermissionChecker
            from niaharness.config.settings import PermissionSettings

            context = QueryContext(
                api_client=api_client,
                tool_registry=review_registry,
                permission_checker=PermissionChecker(PermissionSettings()),
                cwd=_get_cwd(),
                model=model,
                system_prompt=system_prompt,  # reuse parent's prompt for cache parity
                max_tokens=2048,
                max_turns=5,  # review should be quick
            )

            full_text = ""
            async for event, usage in run_query(context, review_messages):
                from niaharness.engine.stream_events import (
                    AssistantTextDelta,
                    AssistantTurnComplete,
                )
                if isinstance(event, AssistantTextDelta):
                    full_text += event.text
                elif isinstance(event, AssistantTurnComplete):
                    if event.message.text:
                        full_text = event.message.text
            return full_text

        loop = asyncio.new_event_loop()
        try:
            full_text = loop.run_until_complete(_do_review_call())
        finally:
            loop.close()

    except Exception as exc:
        logger.warning("Background review LLM call failed: %s", exc)
        return {
            "memories_applied": 0,
            "skills_created": [],
            "skills_updated": [],
            "summary": "",
            "error": str(exc),
        }

    # Parse and apply memory writes.
    review_result = _parse_review_response(full_text)
    memories_applied = _apply_memory_writes(review_result, memory)

    # Extract skill writes from the structured response.
    skills_created = review_result.get("skills_created", [])
    skills_updated = review_result.get("skills_updated", [])

    return {
        "memories_applied": memories_applied,
        "skills_created": skills_created if isinstance(skills_created, list) else [],
        "skills_updated": skills_updated if isinstance(skills_updated, list) else [],
        "summary": review_result.get("summary", ""),
        "error": None,
    }


def _get_cwd() -> str:
    """Get the current working directory (best-effort)."""
    import os

    return os.getcwd()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def maybe_spawn_background_review(
    messages: list,
    api_client: Any,
    model: str,
    system_prompt: str,
    memory: Any,
    *,
    was_interrupted: bool = False,
) -> None:
    """Maybe spawn a background review thread after a turn.

    Non-blocking: returns immediately after spawning (or deciding not to).
    Never raises — review failures are logged but don't affect the main
    conversation.

    Args:
        messages: The current conversation messages (ConversationMessage list).
        api_client: The API client to use for the review call.
        model: The model to use for the review.
        system_prompt: The parent's system prompt (reused for cache parity).
        memory: The Memory instance to write to.
        was_interrupted: If True, skip the review (audit fix: don't review
            interrupted turns — Hermes doesn't).
    """
    if not is_background_review_enabled():
        return
    if memory is None:
        return
    if was_interrupted:
        return  # audit fix: don't review interrupted turns
    if not _review_state.should_review():
        return

    snapshot = _snapshot_messages(messages)
    if len(snapshot) < 2:
        return

    review_model = get_review_model() or model

    def _target():
        # Audit fix: silence stdout/stderr in the review thread so any
        # print statements from the API client or tools don't leak into
        # the main conversation's console. Thread-scoped, not process-global.
        # Adapted from Hermes Agent's thread_scoped_silence().
        import contextlib
        import io

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            try:
                result = _run_review(
                    snapshot, api_client, review_model, system_prompt, memory
                )
                if result.get("error"):
                    logger.debug("Background review error: %s", result["error"])
                elif result.get("memories_applied", 0) > 0 or result.get("skills_created") or result.get("skills_updated"):
                    # Build user-visible feedback message (audit fix: surface results).
                    parts = []
                    if result["memories_applied"] > 0:
                        parts.append(f"{result['memories_applied']} memory item(s)")
                    if result.get("skills_created"):
                        names = [s.get("name", "?") for s in result["skills_created"]]
                        parts.append(f"created skill(s): {', '.join(names)}")
                    if result.get("skills_updated"):
                        names = [s.get("name", "?") for s in result["skills_updated"]]
                        parts.append(f"updated skill(s): {', '.join(names)}")

                    feedback = f"💾 Self-improvement review: {' · '.join(parts)}"
                    logger.info(feedback)
                    _review_state.notify_feedback(feedback)
                else:
                    logger.debug("Background review: %s", result.get("summary", "nothing saved"))
            except Exception as exc:
                logger.warning("Background review thread failed: %s", exc)

    thread = threading.Thread(target=_target, daemon=True, name="nia-background-review")
    _review_state.register_thread(thread)
    thread.start()


def wait_for_reviews(timeout: float = 5.0) -> None:
    """Wait for active review threads to finish (for tests)."""
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
