"""Background memory/skill review — autonomous self-improvement loop.

Ported from Hermes Agent's ``agent/background_review.py`` (960 LOC),
adapted to NIA's architecture. After a turn that involved ≥3 tool calls,
NIA spawns a background daemon thread that:

  1. Snapshots the conversation messages.
  2. Forks a restricted :class:`QueryEngine` with only ``memory`` +
     ``skill_manage`` tools (the review LLM can't run bash, edit files,
     etc.).
  3. Sends one of three review prompts (memory-only, skill-only, or
     combined) asking the LLM to identify reusable patterns, user
     preferences, or workflow corrections worth saving.
  4. The forked engine executes any resulting ``skill_manage`` /
     ``memory`` tool calls against the real skill/memory stores.
  5. Surfaces a compact action summary to the user via a callback.

Key safety features:
  - **Skill provenance** — the review fork can only patch a skill file
    it has actually read via ``skill_view`` in the current review turn
    (prevents the LLM from guessing at content it hasn't seen).
  - **Tool whitelist** — the fork is restricted to ``memory`` +
    ``skill_manage`` + ``skill_view`` + ``skills_list``. All other tools
    are denied at dispatch.
  - **Persistence isolation** — the fork does NOT write to the session
    DB (prevents the curator-takeover bug where the review's harness
    prompt leaks into the user's real session).
  - **Auto-deny approval callback** — the fork never blocks on
    interactive approval (prevents deadlocks against the parent's TUI).
  - **Thread-scoped silence** — the fork's stdout/stderr is silenced
    only for the review thread, not process-wide (other threads' output
    survives).

Gating:
  - ``engine.background_review.enabled`` config flag (default off).
  - Minimum 3 tool calls in the turn (Hermes uses a nudge-interval
    counter; NIA uses a simpler per-turn threshold).
  - Turn not interrupted.

Usage::

    from niaharness.engine.background_review import maybe_spawn_background_review

    maybe_spawn_background_review(
        messages=engine.messages,
        api_client=engine._api_client,
        model=engine._model,
        system_prompt=engine._system_prompt,
        memory=engine._memory,
        tool_call_count=turn_tool_call_count,
    )
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum tool calls in a turn to trigger a background review.
MIN_TOOL_CALLS_FOR_REVIEW = 3

# Maximum iterations the review fork can run (keeps it quick).
MAX_REVIEW_ITERATIONS = 16

# Maximum messages to snapshot for the review.
MAX_SNAPSHOT_MESSAGES = 40

# Config flag name.
CONFIG_FLAG = "engine.background_review.enabled"


# ---------------------------------------------------------------------------
# Review prompts (ported verbatim from Hermes agent/background_review.py)
# ---------------------------------------------------------------------------

MEMORY_REVIEW_PROMPT = (
    "Review the conversation above and consider saving to memory if appropriate.\n\n"
    "Focus on:\n"
    "1. Has the user revealed things about themselves — their persona, desires, "
    "preferences, or personal details worth remembering?\n"
    "2. Has the user expressed expectations about how you should behave, their work "
    "style, or ways they want you to operate?\n\n"
    "If something stands out, save it using the memory tool. "
    "If nothing is worth saving, just say 'Nothing to save.' and stop."
)

SKILL_REVIEW_PROMPT = (
    "Review the conversation above and update the skill library. Be "
    "ACTIVE — most sessions produce at least one skill update, even if "
    "small. A pass that does nothing is a missed learning opportunity, "
    "not a neutral outcome.\n\n"
    "Target shape of the library: CLASS-LEVEL skills, each with a rich "
    "SKILL.md and a `references/` directory for session-specific detail. "
    "Not a long flat list of narrow one-session-one-skill entries. This "
    "shapes HOW you update, not WHETHER you update.\n\n"
    "Signals to look for (any one of these warrants action):\n"
    "  • User corrected your style, tone, format, legibility, or "
    "verbosity. Frustration signals like 'stop doing X', 'this is too "
    "verbose', 'don't format like this', 'why are you explaining', "
    "'just give me the answer', 'you always do Y and I hate it', or an "
    "explicit 'remember this' are FIRST-CLASS skill signals, not just "
    "memory signals. Update the relevant skill(s) to embed the "
    "preference so the next session starts already knowing.\n"
    "  • User corrected your workflow, approach, or sequence of steps. "
    "Encode the correction as a pitfall or explicit step in the skill "
    "that governs that class of task.\n"
    "  • Non-trivial technique, fix, workaround, debugging path, or "
    "tool-usage pattern emerged that a future session would benefit "
    "from. Capture it.\n"
    "  • A skill that got loaded or consulted this session turned out "
    "to be wrong, missing a step, or outdated. Patch it NOW.\n\n"
    "Preference order — prefer the earliest action that fits, but do "
    "pick one when a signal above fired:\n"
    "  1. UPDATE A CURRENTLY-LOADED SKILL. Look back through the "
    "conversation for skills the user loaded via /skill-name or you "
    "read via skill_view. If any of them covers the territory of the "
    "new learning, PATCH that one first. It is the skill that was in "
    "play, so it's the right one to extend.\n"
    "  2. UPDATE AN EXISTING UMBRELLA (via skills_list + skill_view). "
    "If no loaded skill fits but an existing class-level skill does, "
    "patch it. Add a subsection, a pitfall, or broaden a trigger.\n"
    "  3. ADD A SUPPORT FILE under an existing umbrella. Skills can be "
    "packaged with three kinds of support files — use the right "
    "directory per kind:\n"
    "     • `references/<topic>.md` — session-specific detail (error "
    "transcripts, reproduction recipes, provider quirks) AND "
    "condensed knowledge banks: quoted research, API docs, external "
    "authoritative excerpts, or domain notes you found while working "
    "on the problem. Write it concise and for the value of the task, "
    "not as a full mirror of upstream docs.\n"
    "     • `templates/<name>.<ext>` — starter files meant to be "
    "copied and modified (boilerplate configs, scaffolding, a "
    "known-good example the agent can `reproduce with modifications`).\n"
    "     • `scripts/<name>.<ext>` — statically re-runnable actions "
    "the skill can invoke directly (verification scripts, fixture "
    "generators, deterministic probes, anything the agent should run "
    "rather than hand-type each time).\n"
    "     Add support files via skill_manage action=write_file with "
    "file_path starting 'references/', 'templates/', or 'scripts/'. "
    "The umbrella's SKILL.md should gain a one-line pointer to any "
    "new support file so future agents know it exists.\n"
    "  4. CREATE A NEW CLASS-LEVEL UMBRELLA SKILL when no existing "
    "skill covers the class. The name MUST be at the class level. "
    "The name MUST NOT be a specific PR number, error string, feature "
    "codename, library-alone name, or 'fix-X / debug-Y / audit-Z-today' "
    "session artifact. If the proposed name only makes sense for "
    "today's task, it's wrong — fall back to (1), (2), or (3).\n\n"
    "User-preference embedding (important): when the user expressed a "
    "style/format/workflow preference, the update belongs in the "
    "SKILL.md body, not just in memory. Memory captures 'who the user "
    "is and what the current situation and state of your operations "
    "are'; skills capture 'how to do this class of task for this "
    "user'. When they complain about how you handled a task, the "
    "skill that governs that task needs to carry the lesson.\n\n"
    "If you notice two existing skills that overlap, note it in your "
    "reply — the background curator handles consolidation at scale.\n\n"
    "Protected skills (DO NOT edit these):\n"
    "  • Bundled skills (shipped with NIA).\n"
    "  • Hub-installed skills (installed via the skills hub).\n"
    "If the only skills that need updating are protected, say\n"
    "'Nothing to save.' and stop.\n\n"
    "Do NOT capture (these become persistent self-imposed constraints "
    "that bite you later when the environment changes):\n"
    "  • Environment-dependent failures: missing binaries, fresh-install "
    "errors, post-migration path mismatches, 'command not found', "
    "unconfigured credentials, uninstalled packages. The user can fix "
    "these — they are not durable rules.\n"
    "  • Negative claims about tools or features ('browser tools do not "
    "work', 'X tool is broken', 'cannot use Y from execute_code'). These "
    "harden into refusals the agent cites against itself for months "
    "after the actual problem was fixed.\n"
    "  • Session-specific transient errors that resolved before the "
    "conversation ended. If retrying worked, the lesson is the retry "
    "pattern, not the original failure.\n"
    "  • One-off task narratives. A user asking 'summarize today's "
    "market' or 'analyze this PR' is not a class of work that warrants "
    "a skill.\n\n"
    "If a tool failed because of setup state, capture the FIX (install "
    "command, config step, env var to set) under an existing setup or "
    "troubleshooting skill — never 'this tool does not work' as a "
    "standalone constraint.\n\n"
    "'Nothing to save.' is a real option but should NOT be the "
    "default. If the session ran smoothly with no corrections and "
    "produced no new technique, just say 'Nothing to save.' and stop. "
    "Otherwise, act."
)

COMBINED_REVIEW_PROMPT = (
    "Review the conversation above and update two things:\n\n"
    "**Memory**: who the user is. Did the user reveal persona, "
    "desires, preferences, personal details, or expectations about "
    "how you should behave? Save facts about the user and durable "
    "preferences with the memory tool.\n\n"
    "**Skills**: how to do this class of task. Be ACTIVE — most "
    "sessions produce at least one skill update. A pass that does "
    "nothing is a missed learning opportunity, not a neutral outcome.\n\n"
    "Target shape of the skill library: CLASS-LEVEL skills with a rich "
    "SKILL.md and a `references/` directory for session-specific detail. "
    "Not a long flat list of narrow one-session-one-skill entries.\n\n"
    "Signals that warrant a skill update (any one is enough):\n"
    "  • User corrected your style, tone, format, legibility, "
    "verbosity, or approach. Frustration is a FIRST-CLASS skill "
    "signal, not just a memory signal. 'stop doing X', 'don't format "
    "like this', 'I hate when you Y' — embed the lesson in the skill "
    "that governs that task so the next session starts fixed.\n"
    "  • Non-trivial technique, fix, workaround, or debugging path "
    "emerged.\n"
    "  • A skill that was loaded or consulted turned out wrong, "
    "missing, or outdated — patch it now.\n\n"
    "Preference order for skills — pick the earliest that fits:\n"
    "  1. UPDATE A CURRENTLY-LOADED SKILL. Check what skills were "
    "loaded via /skill-name or skill_view in the conversation. If one "
    "of them covers the learning, PATCH it first. It was in play; "
    "it's the right place.\n"
    "  2. UPDATE AN EXISTING UMBRELLA (skills_list + skill_view to "
    "find the right one). Patch it.\n"
    "  3. ADD A SUPPORT FILE under an existing umbrella via "
    "skill_manage action=write_file. Three kinds: "
    "`references/<topic>.md` for session-specific detail OR condensed "
    "knowledge banks (quoted research, API docs excerpts, domain "
    "notes) written concise and task-focused; `templates/<name>.<ext>` "
    "for starter files meant to be copied and modified; "
    "`scripts/<name>.<ext>` for statically re-runnable actions "
    "(verification, fixture generators, probes). Add a one-line "
    "pointer in SKILL.md so future agents find them.\n"
    "  4. CREATE A NEW CLASS-LEVEL UMBRELLA when nothing exists. "
    "Name at the class level — NOT a PR number, error string, "
    "codename, library-alone name, or 'fix-X / debug-Y' session "
    "artifact. If the name only fits today's task, fall back to (1), "
    "(2), or (3).\n\n"
    "User-preference embedding: when the user complains about how "
    "you handled a task, update the skill that governs that task — "
    "memory alone isn't enough. Memory says 'who the user is and "
    "what the current situation and state of your operations are'; "
    "skills say 'how to do this class of task for this user'. Both "
    "should carry user-preference lessons when relevant.\n\n"
    "If you notice overlapping existing skills, mention it — the "
    "background curator handles consolidation.\n\n"
    "Protected skills (DO NOT edit these):\n"
    "  • Bundled skills (shipped with NIA).\n"
    "  • Hub-installed skills (installed via the skills hub).\n"
    "If the only skills that need updating are protected, say\n"
    "'Nothing to save.' and stop.\n\n"
    "Do NOT capture as skills (these become persistent self-imposed "
    "constraints that bite you later when the environment changes):\n"
    "  • Environment-dependent failures: missing binaries, fresh-install "
    "errors, post-migration path mismatches, 'command not found', "
    "unconfigured credentials, uninstalled packages. The user can fix "
    "these — they are not durable rules.\n"
    "  • Negative claims about tools or features ('browser tools do not "
    "work', 'X tool is broken', 'cannot use Y from execute_code'). These "
    "harden into refusals the agent cites against itself for months "
    "after the actual problem was fixed.\n"
    "  • Session-specific transient errors that resolved before the "
    "conversation ended. If retrying worked, the lesson is the retry "
    "pattern, not the original failure.\n"
    "  • One-off task narratives. A user asking 'summarize today's "
    "market' or 'analyze this PR' is not a class of work that warrants "
    "a skill.\n\n"
    "If a tool failed because of setup state, capture the FIX (install "
    "command, config step, env var to set) under an existing setup or "
    "troubleshooting skill — never 'this tool does not work' as a "
    "standalone constraint.\n\n"
    "Act on whichever of the two dimensions has real signal. If "
    "genuinely nothing stands out on either, say 'Nothing to save.' "
    "and stop — but don't reach for that conclusion as a default."
)

# Suffix appended to the review prompt at call time.
_TOOL_RESTRICTION_SUFFIX = (
    "\n\nYou can only call memory and skill management tools. Other "
    "tools will be denied at runtime — do not attempt them."
)


# ---------------------------------------------------------------------------
# Config + state
# ---------------------------------------------------------------------------


def _is_background_review_enabled() -> bool:
    """Check if background review is enabled via config or env var.

    Resolution order:
      1. ``NIA_BACKGROUND_REVIEW`` env var (1/true/yes/on = enabled).
      2. ``engine.background_review.enabled`` in config.yaml/settings.
      3. Default: off.
    """
    env_value = os.environ.get("NIA_BACKGROUND_REVIEW", "").strip().lower()
    if env_value in {"1", "true", "yes", "on"}:
        return True
    if env_value in {"0", "false", "no", "off"}:
        return False
    # Try config.
    try:
        from niaharness.config.settings import load_settings

        settings = load_settings()
        engine_cfg = getattr(settings, "engine", None) or {}
        if isinstance(engine_cfg, dict):
            bg_cfg = engine_cfg.get("background_review", {})
            if isinstance(bg_cfg, dict):
                return bool(bg_cfg.get("enabled", False))
    except Exception:
        pass
    return False


class _ReviewState:
    """Tracks active review threads + feedback callbacks."""

    def __init__(self) -> None:
        self._threads: list[threading.Thread] = []
        self._feedback_callback: Optional[Callable[[str], None]] = None
        self._lock = threading.Lock()

    def register_thread(self, thread: threading.Thread) -> None:
        with self._lock:
            self._threads.append(thread)

    def active_count(self) -> int:
        with self._lock:
            return sum(1 for t in self._threads if t.is_alive())

    def set_feedback_callback(self, cb: Optional[Callable[[str], None]]) -> None:
        with self._lock:
            self._feedback_callback = cb

    def notify_feedback(self, message: str) -> None:
        with self._lock:
            cb = self._feedback_callback
        if cb:
            try:
                cb(message)
            except Exception as exc:
                logger.debug("Feedback callback failed: %s", exc)


_review_state = _ReviewState()


def set_feedback_callback(cb: Optional[Callable[[str], None]]) -> None:
    """Set the user-visible feedback callback (e.g. for TUI status line)."""
    _review_state.set_feedback_callback(cb)


def get_review_stats() -> dict[str, Any]:
    """Return review system statistics (for debugging/UI)."""
    return {
        "active_threads": _review_state.active_count(),
        "enabled": _is_background_review_enabled(),
        "min_tool_calls": MIN_TOOL_CALLS_FOR_REVIEW,
    }


# ---------------------------------------------------------------------------
# Message snapshotting
# ---------------------------------------------------------------------------


def _snapshot_messages(messages: list) -> list[dict[str, Any]]:
    """Snapshot conversation messages for the review.

    Converts NIA ``ConversationMessage`` objects to plain dicts (the
    forked engine rehydrates them). Caps at ``MAX_SNAPSHOT_MESSAGES``
    (keeps head + tail, drops middle).
    """
    snapshot: list[dict[str, Any]] = []
    for msg in messages:
        if hasattr(msg, "model_dump"):
            msg_dict = msg.model_dump()
        elif isinstance(msg, dict):
            msg_dict = msg
        else:
            continue
        snapshot.append(msg_dict)

    if len(snapshot) > MAX_SNAPSHOT_MESSAGES:
        # Keep head (system prompt + first exchange) + tail (most recent).
        head_count = 4
        tail_count = MAX_SNAPSHOT_MESSAGES - head_count
        snapshot = snapshot[:head_count] + snapshot[-tail_count:]
    return snapshot


# ---------------------------------------------------------------------------
# Action summarization
# ---------------------------------------------------------------------------


def summarize_background_review_actions(
    review_messages: List[Dict[str, Any]],
    prior_snapshot: List[Dict[str, Any]],
    *,
    notification_mode: str = "on",
) -> List[str]:
    """Build the human-facing action summary for a background review pass.

    Walks the review agent's session messages and collects successful
    memory and skill-management actions to surface to the user. Tool
    messages already present in ``prior_snapshot`` are skipped so stale
    inherited results are not re-surfaced as fresh background work.

    Args:
        review_messages: The forked engine's message list after the review.
        prior_snapshot: The snapshot passed into the review (to skip stale).
        notification_mode: ``"off"`` = empty, ``"on"`` = generic messages,
            ``"verbose"`` = include content previews.

    Returns:
        List of human-readable action strings.
    """
    mode = str(notification_mode or "on").lower()
    if mode == "off":
        return []
    verbose = mode == "verbose"

    # Collect existing tool_call_ids from the prior snapshot to skip them.
    existing_tool_call_ids: set = set()
    for prior in prior_snapshot or []:
        if not isinstance(prior, dict) or prior.get("role") != "tool":
            continue
        tcid = prior.get("tool_call_id")
        if tcid:
            existing_tool_call_ids.add(tcid)

    # Map tool_call_ids to call details (name + arguments).
    notify_tools = {"nia_memory", "skill_manage", "memory", "skill_view", "skills_list"}
    call_details: dict[str, dict[str, Any]] = {}
    for msg in review_messages or []:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls", []) or []:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function", {}) or {}
            fn_name = fn.get("name", "")
            tcid = tc.get("id")
            if fn_name not in notify_tools or not tcid:
                continue
            try:
                args = json.loads(fn.get("arguments", "{}"))
            except (json.JSONDecodeError, TypeError):
                args = {}
            call_details[tcid] = {
                "tool": fn_name,
                "action": args.get("action", ""),
                "name": args.get("name", ""),
                "content": args.get("content", ""),
            }

    # Walk tool messages and collect actions.
    actions: List[str] = []
    for msg in review_messages or []:
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            continue
        tcid = msg.get("tool_call_id")
        if tcid and tcid in existing_tool_call_ids:
            continue  # Skip stale inherited results.
        if tcid and tcid not in call_details:
            continue

        try:
            data = json.loads(msg.get("content", "{}"))
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(data, dict) or not data.get("success"):
            continue

        message = data.get("message", "")
        detail = call_details.get(tcid, {})
        is_skill = detail.get("tool") in {"skill_manage", "skill_view"}

        message_lower = message.lower()
        if not verbose:
            if "created" in message_lower:
                actions.append(message)
            elif "updated" in message_lower:
                actions.append(message)
            elif is_skill and "patched" in message_lower:
                actions.append(message)
            continue

        # Verbose mode: include content previews.
        label = "Skill" if is_skill else "Memory"
        action = detail.get("action", "")
        content = detail.get("content", "")
        skill_name = detail.get("name", "")
        max_preview = 120

        if is_skill and skill_name:
            if action == "create":
                actions.append(f"📝 Skill '{skill_name}' created")
            elif action in {"edit", "patch"}:
                actions.append(f"📝 Skill '{skill_name}' patched")
            else:
                actions.append(f"📝 Skill '{skill_name}' {action}")
        elif content:
            preview = content[:max_preview] + ("…" if len(content) > max_preview else "")
            actions.append(f"{label} ➕ {preview}")
        elif message:
            actions.append(f"{label}: {message}")

    return actions


# ---------------------------------------------------------------------------
# Fork execution
# ---------------------------------------------------------------------------


def _run_review_in_thread(
    messages_snapshot: list[dict[str, Any]],
    api_client: Any,
    model: str,
    system_prompt: str,
    memory: Any,
    prompt: str,
) -> None:
    """Worker function executed in the background-review daemon thread.

    Forks a restricted :class:`QueryEngine` with only memory + skill
    tools, runs the review prompt, and surfaces a compact action summary.
    """
    import contextlib
    import io

    from niaharness.tools.skill_provenance import (
        reset_background_review_read_marks,
        set_current_write_origin,
        reset_current_write_origin,
    )

    # Bind the write-origin ContextVar so skill_manage knows it's in
    # background-review mode (enables the read-before-write gate).
    origin_token = set_current_write_origin("background_review")
    # Clear any stale read marks from a prior review.
    reset_background_review_read_marks()

    try:
        # Thread-scoped silence: only this thread's stdout/stderr is
        # redirected, not process-wide (other threads' output survives).
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            try:
                review_messages = _execute_review_fork(
                    messages_snapshot, api_client, model, system_prompt, memory, prompt,
                )
            except Exception as exc:
                logger.warning("Background review fork failed: %s", exc)
                return

        # Surface actions to the user.
        try:
            actions = summarize_background_review_actions(
                review_messages, messages_snapshot, notification_mode="on",
            )
        except Exception as exc:
            logger.warning("summarize_background_review_actions failed: %s", exc)
            actions = []

        if actions:
            summary = " · ".join(dict.fromkeys(actions))
            feedback = f"💾 Self-improvement review: {summary}"
            logger.info(feedback)
            _review_state.notify_feedback(feedback)
        else:
            logger.debug("Background review: nothing to save")
    except Exception as exc:
        logger.warning("Background memory/skill review failed: %s", exc)
    finally:
        reset_current_write_origin(origin_token)


def _execute_review_fork(
    messages_snapshot: list[dict[str, Any]],
    api_client: Any,
    model: str,
    system_prompt: str,
    memory: Any,
    prompt: str,
) -> list[dict[str, Any]]:
    """Execute the review in a forked QueryEngine with restricted tools.

    Returns the forked engine's message list (for action summarization).
    """
    import asyncio

    from niaharness.engine.messages import ConversationMessage, TextBlock
    from niaharness.tools import create_default_tool_registry
    from niaharness.tools.base import ToolRegistry

    # Build a restricted tool registry: memory + skill tools only.
    review_registry = ToolRegistry()
    full_registry = create_default_tool_registry()
    allowed_tools = ("nia_memory", "skill_manage", "skill", "skills_list", "skill_view")
    for tool_name in allowed_tools:
        tool = full_registry.get(tool_name)
        if tool:
            review_registry.register(tool)

    # Wire memory into the nia_memory tool.
    mem_tool = review_registry.get("nia_memory")
    if mem_tool and hasattr(mem_tool, "set_memory"):
        mem_tool.set_memory(memory)

    # Build the review conversation: system prompt + snapshot + review prompt.
    review_messages: list[ConversationMessage] = []
    for msg_dict in messages_snapshot:
        role = msg_dict.get("role", "user")
        if role == "system":
            continue  # System prompt is passed separately.
        content_blocks: list[Any] = []
        for block in msg_dict.get("content", []):
            btype = block.get("type", "") if isinstance(block, dict) else ""
            if btype == "text":
                content_blocks.append(TextBlock(text=block.get("text", "")))
        if content_blocks:
            review_messages.append(ConversationMessage(role=role, content=content_blocks))

    # Append the review prompt + tool-restriction suffix as the final user message.
    review_messages.append(
        ConversationMessage(
            role="user",
            content=[TextBlock(text=prompt + _TOOL_RESTRICTION_SUFFIX)],
        )
    )

    # Run the forked engine in its own event loop.
    async def _run_fork():
        from niaharness.engine.query import run_query, QueryContext
        from niaharness.permissions.checker import PermissionChecker
        from niaharness.config.settings import PermissionSettings, PermissionMode

        # FULL_AUTO so the fork doesn't block on approvals (it can only
        # call memory/skill tools anyway, which are safe).
        settings = PermissionSettings(mode=PermissionMode.FULL_AUTO)
        context = QueryContext(
            api_client=api_client,
            tool_registry=review_registry,
            permission_checker=PermissionChecker(settings),
            cwd=os.getcwd(),
            model=model,
            system_prompt=system_prompt,  # reuse parent's prompt for cache parity
            max_tokens=2048,
            max_turns=MAX_REVIEW_ITERATIONS,
        )

        # Track tool calls for the action summary.
        fork_messages: list[dict[str, Any]] = []
        async for event, usage in run_query(context, review_messages):
            from niaharness.engine.stream_events import (
                AssistantTurnComplete,
                ToolExecutionCompleted,
            )
            if isinstance(event, ToolExecutionCompleted):
                # Record the tool call + result for summarization.
                fork_messages.append({
                    "role": "assistant",
                    "tool_calls": [{"id": getattr(event, "tool_call_id", ""), "function": {"name": event.tool_name, "arguments": getattr(event, "arguments", "{}")}}],
                })
                fork_messages.append({
                    "role": "tool",
                    "tool_call_id": getattr(event, "tool_call_id", ""),
                    "content": getattr(event, "output", ""),
                })
            elif isinstance(event, AssistantTurnComplete):
                if event.message.text:
                    fork_messages.append({
                        "role": "assistant",
                        "content": [{"type": "text", "text": event.message.text}],
                    })
        return fork_messages

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_run_fork())
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Public API — spawn + wait
# ---------------------------------------------------------------------------


def maybe_spawn_background_review(
    messages: list,
    api_client: Any,
    model: str,
    system_prompt: str,
    memory: Any,
    *,
    tool_call_count: int = 0,
    was_interrupted: bool = False,
    review_memory: bool = True,
    review_skills: bool = True,
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
        tool_call_count: Number of tool calls in the just-completed turn.
            Must be ≥ ``MIN_TOOL_CALLS_FOR_REVIEW`` (3) to trigger.
        was_interrupted: If True, skip the review (don't review interrupted turns).
        review_memory: If True, include memory review in the prompt.
        review_skills: If True, include skill review in the prompt.
    """
    if not _is_background_review_enabled():
        return
    if memory is None:
        return
    if was_interrupted:
        return
    if tool_call_count < MIN_TOOL_CALLS_FOR_REVIEW:
        return
    if not review_memory and not review_skills:
        return

    snapshot = _snapshot_messages(messages)
    if len(snapshot) < 2:
        return

    # Pick the prompt based on which triggers fired.
    if review_memory and review_skills:
        prompt = COMBINED_REVIEW_PROMPT
    elif review_memory:
        prompt = MEMORY_REVIEW_PROMPT
    else:
        prompt = SKILL_REVIEW_PROMPT

    review_model = model  # Inherit parent's model for cache parity.

    def _target():
        _run_review_in_thread(
            snapshot, api_client, review_model, system_prompt, memory, prompt,
        )

    thread = threading.Thread(target=_target, daemon=True, name="nia-bg-review")
    _review_state.register_thread(thread)
    thread.start()


def wait_for_reviews(timeout: float = 5.0) -> None:
    """Wait for active review threads to finish (for tests)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _review_state.active_count() == 0:
            return
        time.sleep(0.05)


__all__ = [
    "COMBINED_REVIEW_PROMPT",
    "CONFIG_FLAG",
    "MAX_REVIEW_ITERATIONS",
    "MEMORY_REVIEW_PROMPT",
    "MIN_TOOL_CALLS_FOR_REVIEW",
    "SKILL_REVIEW_PROMPT",
    "get_review_stats",
    "maybe_spawn_background_review",
    "set_feedback_callback",
    "summarize_background_review_actions",
    "wait_for_reviews",
]
