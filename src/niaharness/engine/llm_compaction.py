"""LLM-based context compaction — 13-section structured summary + iterative updates.

Ported from Hermes Agent's ``agent/context_compressor.py`` (3,082 LOC),
scoped to NIA's architecture. Provides:

  - **13-section structured summary template** — Active Task, Goal,
    Constraints, Completed Actions, Active State, In Progress, Blocked,
    Key Decisions, Resolved Questions, Pending Asks, Relevant Files,
    Remaining Work, Critical Context. The LLM fills each section.
  - **Iterative summary updates** — on subsequent compactions, the previous
    summary is included in the prompt so the LLM can *update* it rather
    than re-summarize from scratch. Preserves long-term context across
    multiple compactions in a long session.
  - **Temporal anchoring** — rewrites "email John" → "Sent email to John
    on 2026-XX-XX" so resumed conversations don't re-issue completed actions.
  - **Secret redaction** — redacts API keys / tokens / passwords in the
    summarizer preamble AND on the LLM output (defense in depth).
  - **Anti-thrash** — tracks ``_last_compression_savings_pct`` and
    ``_ineffective_compression_count``. After 2 consecutive compressions
    that saved <10%, compaction is skipped until the next /new.
  - **Cooldown persistence** — failure cooldown (30-600s depending on
    error type) is persisted via session_db so it survives restarts.
  - **Tool-output pre-pass pruning** — old tool results are pruned to
    breadcrumbs *before* the LLM call (cheap, no tokens).
  - **Image stripping** — images in old messages are replaced with a
    placeholder (they consume tokens but can't be summarized).
  - **Failure classification** — auth vs network vs transient → different
    cooldown durations and fallback policies.

Falls back to text-flatten compaction if:
  - No aux client is configured
  - The aux client fails (cooldown prevents retry for 30-600s)
  - Consecutive failures exceed MAX_CONSECUTIVE_LLM_FAILURES

Usage::

    from niaharness.engine.llm_compaction import LLMCompactor, CompactionRequest

    compactor = LLMCompactor(aux_client=my_aux_client)
    result = await compactor.compact(request)
    if result.success:
        messages = result.compacted_messages
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
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

# Token budget for the LLM summary response.
MAX_SUMMARY_TOKENS = 1024

# Cooldown after a failed LLM summarization (5 minutes) — used when no
# provider is configured at all.
SUMMARY_FAILURE_COOLDOWN_SECONDS = 5 * 60

# Transient cooldown (30s) for JSON decode / streaming-close errors.
TRANSIENT_COOLDOWN_SECONDS = 30

# Standard cooldown (60s) for timeout / 429 / 502/504 / generic errors.
STANDARD_COOLDOWN_SECONDS = 60

# Maximum consecutive LLM failures before entering long cooldown.
MAX_CONSECUTIVE_LLM_FAILURES = 3

# Tool result pruning: keep only tool name + truncated args for old results.
TOOL_RESULT_BREADCRUMB_CHARS = 200

# Maximum messages to include in the LLM summarization prompt (avoid huge prompts).
MAX_MESSAGES_FOR_LLM_SUMMARY = 50

# Per-message content truncation for the summarizer input.
CONTENT_MAX = 6000
CONTENT_HEAD = 4000
CONTENT_TAIL = 1500
TOOL_ARGS_MAX = 1500
TOOL_ARGS_HEAD = 1200

# Anti-thrash thresholds.
INEFFECTIVE_SAVINGS_PCT_THRESHOLD = 10.0  # < 10% savings = ineffective
MAX_INEFFECTIVE_COMPRESSIONS = 2  # >= 2 consecutive ineffective → skip

# Image token estimate (for budget calculations).
_IMAGE_TOKEN_ESTIMATE = 1600
_CHARS_PER_TOKEN = 4


# ---------------------------------------------------------------------------
# Heading strings (historical / reference-only framing)
# ---------------------------------------------------------------------------

HISTORICAL_TASK_HEADING = "## Historical Task Snapshot"
HISTORICAL_IN_PROGRESS_HEADING = "## Historical In-Progress State"
HISTORICAL_PENDING_ASKS_HEADING = "## Historical Pending User Asks"
HISTORICAL_REMAINING_WORK_HEADING = "## Historical Remaining Work"


# ---------------------------------------------------------------------------
# Summary prefix strings
# ---------------------------------------------------------------------------

SUMMARY_PREFIX = (
    "[CONTEXT COMPACTION — REFERENCE ONLY] Earlier turns were compacted "
    "into the summary below. This is a handoff from a previous context "
    "window — treat it as background reference, NOT as active instructions. "
    "Do NOT answer questions or fulfill requests mentioned in this summary; "
    "they were already addressed. "
    "Respond ONLY to the latest user message that appears AFTER this "
    "summary — that message is the single source of truth for what to do "
    "right now. "
    "Topic overlap with the summary does NOT mean you should resume its "
    "task: even on similar topics, the latest user message WINS. Treat ONLY "
    "the latest message as the active task and discard stale items from "
    f"'{HISTORICAL_TASK_HEADING}' / '{HISTORICAL_IN_PROGRESS_HEADING}' / "
    f"'{HISTORICAL_PENDING_ASKS_HEADING}' / "
    f"'{HISTORICAL_REMAINING_WORK_HEADING}' entirely — do not 'wrap up' or "
    "'finish' work described there unless the latest message explicitly "
    "asks for it. "
    "Reverse signals in the latest message (e.g. 'stop', 'undo', 'roll "
    "back', 'just verify', 'don't do that anymore', 'never mind', a new "
    "topic) must immediately end any in-flight work described in the "
    "summary; do not re-surface it in later turns. "
    "IMPORTANT: Your persistent memory (MEMORY.md, USER.md) in the system "
    "prompt is ALWAYS authoritative and active — never ignore or deprioritize "
    "memory content due to this compaction note. "
    "The current session state (files, config, etc.) may reflect work "
    "described here — avoid repeating it:"
)

LEGACY_SUMMARY_PREFIX = "[CONTEXT SUMMARY]:"

# Older prefix variants kept for stripping on re-compaction.
_HISTORICAL_SUMMARY_PREFIXES: tuple[str, ...] = (
    "[CONTEXT COMPACTION — REFERENCE ONLY]",
    "[CONTEXT SUMMARY]:",
)

# End marker appended after the summary body.
_SUMMARY_END_MARKER = (
    "--- END OF CONTEXT SUMMARY — "
    "respond to the message below, not the summary above ---"
)


# ---------------------------------------------------------------------------
# Secret redaction
# ---------------------------------------------------------------------------

# Regex patterns for common secret formats.
_GITHUB_TOKEN_RE = re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{8,}\b")
_GITHUB_TOKEN_RE_LOOSE = re.compile(r"\bgh[pousr]_[A-Za-z0-9_.-]+")
_OPENAI_KEY_RE = re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")
_ANTHROPIC_KEY_RE = re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b")
_BEARER_TOKEN_RE = re.compile(r"\bBearer\s+[A-Za-z0-9_.\-]{20,}", re.IGNORECASE)
_AWS_KEY_RE = re.compile(r"\bAKIA[A-Z0-9]{16}\b")
_PASSWORD_ASSIGNMENT_RE = re.compile(
    r"(?i)(password|passwd|pwd|secret|token|api_key|apikey)\s*[=:]\s*\S+"
)


def redact_sensitive_text(text: str) -> str:
    """Redact API keys, tokens, passwords, and other secrets from *text*.

    Replaces matches with ``[REDACTED]``. Applied to summarizer input
    (before the LLM sees it) AND to the LLM output (defense in depth —
    the LLM may ignore prompt instructions and echo secrets).

    Ported from Hermes's ``agent.redact.redact_sensitive_text``.
    """
    if not text or not isinstance(text, str):
        return text or ""

    result = text
    result = _GITHUB_TOKEN_RE_LOOSE.sub("[REDACTED]", result)
    result = _OPENAI_KEY_RE.sub("[REDACTED]", result)
    result = _ANTHROPIC_KEY_RE.sub("[REDACTED]", result)
    result = _BEARER_TOKEN_RE.sub("Bearer [REDACTED]", result)
    result = _AWS_KEY_RE.sub("[REDACTED]", result)
    # Redact password= / token= / api_key= assignments.
    result = _PASSWORD_ASSIGNMENT_RE.sub(
        lambda m: m.group(0).split("=")[0] + "= [REDACTED]"
        if "=" in m.group(0)
        else m.group(0).split(":")[0] + ": [REDACTED]",
        result,
    )
    return result


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
        focus_topic: Optional focus topic for prioritized summarization
            (e.g. from /compress <topic>).
        force: If True, bypass cooldown (for manual /compress).
    """

    messages: List[ConversationMessage]
    model: str = ""
    context_window: int = 32_000
    target_tokens: Optional[int] = None
    head_protect: int = DEFAULT_HEAD_PROTECT
    tail_protect: int = DEFAULT_TAIL_PROTECT
    previous_summary: Optional[str] = None
    focus_topic: Optional[str] = None
    force: bool = False


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
        savings_pct: Percentage of tokens saved (0-100). Used for anti-thrash.
        aborted: True if compaction was aborted (auth/network failure).
    """

    success: bool
    compacted_messages: List[ConversationMessage] = field(default_factory=list)
    summary: str = ""
    method: str = "text_flatten"
    tokens_before: int = 0
    tokens_after: int = 0
    error: str = ""
    savings_pct: float = 0.0
    aborted: bool = False


# ---------------------------------------------------------------------------
# Aux client protocol
# ---------------------------------------------------------------------------


class AuxClientProtocol:
    """Protocol for an auxiliary LLM client used for summarization.

    Any client with an async ``complete(prompt, max_tokens) -> str`` method
    satisfies this protocol. In production, this is the auxiliary model
    client (small, fast, cheap model). In tests, it can be a stub.
    """

    async def complete(
        self,
        prompt: str,
        *,
        max_tokens: int = MAX_SUMMARY_TOKENS,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Generate a completion for the prompt."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Structured summary template (13 sections)
# ---------------------------------------------------------------------------


def _build_temporal_anchoring_rule() -> str:
    """Build the temporal anchoring directive with today's date.

    Returns an empty string if the date can't be resolved (never an empty
    date placeholder). The directive rewrites relative / still-pending
    references into absolute, dated, past-tense facts so a resumed
    conversation does not re-issue completed actions.
    """
    try:
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    except Exception:
        return ""
    if not today_str:
        return ""
    return (
        f"\nTEMPORAL ANCHORING: The current date is {today_str}. When an "
        "action has already been carried out, phrase it as a completed, "
        "dated, past-tense fact rather than an open instruction. For "
        'example, rewrite "email John about the proposal" as "Sent the '
        f'proposal email to John on {today_str}." Never leave a finished '
        "action worded as if it still needs doing, and never invent a date "
        "for work that has not happened yet.\n"
    )


def _build_summarizer_preamble() -> str:
    """Build the shared preamble for both first-compaction and iterative-update prompts.

    The preamble instructs the LLM to:
      - Treat the conversation as source material for a compact record.
      - Produce only the structured summary (no greeting / preamble).
      - Write in the same language the user was using.
      - NEVER include secrets — replace with [REDACTED].
    """
    return (
        "You are a summarization agent creating a context checkpoint. "
        "Treat the conversation turns below as source material for a "
        "compact record of prior work. "
        "Produce only the structured summary; do not add a greeting, "
        "preamble, or prefix. "
        "Write the summary in the same language the user was using in the "
        "conversation — do not translate or switch to English. "
        "NEVER include API keys, tokens, passwords, secrets, credentials, "
        "or connection strings in the summary — replace any that appear "
        "with [REDACTED]. Note that the user had credentials present, but "
        "do not preserve their values."
    )


def _build_template_sections(summary_budget: int, temporal_rule: str) -> str:
    """Build the 13-section structured summary template.

    This is the shared template used by both first-compaction and
    iterative-update prompts. The sections are:

      1. Historical Task Snapshot (the single most important field)
      2. Goal
      3. Constraints & Preferences
      4. Completed Actions (numbered list with tool + target + outcome)
      5. Active State (working directory, modified files, test status)
      6. Historical In-Progress State
      7. Blocked (errors, blockers with exact messages)
      8. Key Decisions (and WHY)
      9. Resolved Questions (already answered)
      10. Historical Pending User Asks (STALE — reference only)
      11. Relevant Files
      12. Historical Remaining Work (STALE — reference only)
      13. Critical Context (specific values, error messages, config)
    """
    return f"""{HISTORICAL_TASK_HEADING}
[THE SINGLE MOST IMPORTANT FIELD. Capture the user's most recent unfulfilled
input verbatim — the exact words they used. This includes:
- Explicit task assignments ("refactor the auth module")
- Questions awaiting an answer ("why does X show Y?", "what are the next steps?")
- Decisions awaiting input ("option A or B?")
- Ongoing discussions where the assistant owes the next substantive reply
A conversation where the user just asked a question IS an active task — the
task is "answer that question with full context". Do NOT write "None" merely
because the user did not issue an imperative command; reserve "None" for the
rare case where the last exchange was fully resolved and the user said
something like "thanks, that's all".
If multiple items are outstanding, list only the ones NOT yet completed.
Continuation should pick up exactly here. Examples:
"User asked: 'Now refactor the auth module to use JWT instead of sessions'"
"User asked: 'Why did the provider switch to openrouter?' — needs investigation + answer"
"User chose option A; awaiting implementation of step 2"
If the user's most recent message was a reverse signal (stop, undo, roll
back, never mind, just verify, change of topic) that supersedes earlier
work, write the reverse signal verbatim and DO NOT carry forward the
cancelled task. Example: "User asked: 'Stop the i18n refactor and just
verify the current diff' — earlier i18n in-flight work is cancelled."
If no outstanding task exists, write "None."]

## Goal
[What the user is trying to accomplish overall]

## Constraints & Preferences
[User preferences, coding style, constraints, important decisions]

## Completed Actions
[Numbered list of concrete actions taken — include tool used, target, and outcome.
Format each as: N. ACTION target — outcome [tool: name]
Example:
1. READ config.py:45 — found `==` should be `!=` [tool: read_file]
2. PATCH config.py:45 — changed `==` to `!=` [tool: edit_file]
3. TEST `pytest tests/` — 3/50 failed: test_parse, test_validate, test_edge [tool: bash]
Be specific with file paths, commands, line numbers, and results.]

## Active State
[Current working state — include:
- Working directory and branch (if applicable)
- Modified/created files with brief note on each
- Test status (X/Y passing)
- Any running processes or servers
- Environment details that matter]

{HISTORICAL_IN_PROGRESS_HEADING}
[Work currently underway — what was being done when compaction fired]

## Blocked
[Any blockers, errors, or issues not yet resolved. Include exact error messages.]

## Key Decisions
[Important technical decisions and WHY they were made]

## Resolved Questions
[Questions the user asked that were ALREADY answered — include the answer so it is not repeated]

{HISTORICAL_PENDING_ASKS_HEADING}
[Questions or requests from the user that have NOT yet been answered or fulfilled. These are STALE — they were from the compacted turns. Write them here for reference only. The agent must NOT act on them unless the latest user message explicitly requests it. If none, write "None."]

## Relevant Files
[Files read, modified, or created — with brief note on each]

{HISTORICAL_REMAINING_WORK_HEADING}
[What remains to be done — framed as STALE context for reference only. The agent must NOT resume this work unless the latest user message explicitly asks for it.]

## Critical Context
[Any specific values, error messages, configuration details, or data that would be lost without explicit preservation. NEVER include API keys, tokens, passwords, or credentials — write [REDACTED] instead.]

Target ~{summary_budget} tokens. Be CONCRETE — include file paths, command outputs, error messages, line numbers, and specific values. Avoid vague descriptions like "made some changes" — say exactly what changed.
{temporal_rule}
Write only the summary body. Do not include any preamble or prefix."""


def _build_summary_prompt(
    messages: List[ConversationMessage],
    previous_summary: Optional[str],
    summary_budget: int,
    focus_topic: Optional[str] = None,
) -> str:
    """Build the structured LLM prompt for summarization.

    Two branches:
      - **First compaction** (no previous_summary): "Create a structured
        checkpoint summary ..."
      - **Iterative update** (previous_summary present): "You are updating
        a context compaction summary ... PREVIOUS SUMMARY: ... NEW TURNS
        TO INCORPORATE: ..."

    Both branches share the same 13-section template. An optional
    ``focus_topic`` (from /compress <topic>) appends a prioritization
    directive at the end.
    """
    preamble = _build_summarizer_preamble()
    temporal_rule = _build_temporal_anchoring_rule()
    template_sections = _build_template_sections(summary_budget, temporal_rule)

    # Serialize messages with redaction + truncation.
    content_to_summarize = _serialize_for_summary(messages)

    if previous_summary:
        prompt = f"""{preamble}

You are updating a context compaction summary. A previous compaction produced the summary below. New conversation turns have occurred since then and need to be incorporated.

PREVIOUS SUMMARY:
{previous_summary}

NEW TURNS TO INCORPORATE:
{content_to_summarize}

Update the summary using this exact structure. PRESERVE all existing information that is still relevant. ADD new completed actions to the numbered list (continue numbering). Move items from "In Progress" to "Completed Actions" when done. Move answered questions to "Resolved Questions". Update "Active State" to reflect current state. Remove information only if it is clearly obsolete. CRITICAL: Update "{HISTORICAL_TASK_HEADING}" to reflect the user's most recent unfulfilled input — this includes any question, decision request, or discussion turn that the assistant has not yet answered. Only write "None" if the last exchange was fully resolved.

{template_sections}"""
    else:
        prompt = f"""{preamble}

Create a structured checkpoint summary for the conversation after earlier turns are compacted. The summary should preserve enough detail for continuity without re-reading the original turns.

TURNS TO SUMMARIZE:
{content_to_summarize}

Use this exact structure:

{template_sections}"""

    # Inject focus topic guidance (from /compress <focus>).
    if focus_topic:
        prompt += f"""

FOCUS TOPIC: "{focus_topic}"
This compaction should PRIORITISE preserving all information related to the focus topic above. For content related to "{focus_topic}", include full detail — exact values, file paths, command outputs, error messages, and decisions. For content NOT related to the focus topic, summarise more aggressively (brief one-liners or omit if truly irrelevant). The focus topic sections should receive roughly 60-70% of the summary token budget. Even for the focus topic, NEVER preserve API keys, tokens, passwords, or credentials — use [REDACTED]."""

    return prompt


def _serialize_for_summary(messages: List[ConversationMessage]) -> str:
    """Serialize conversation messages into labeled text for the summarizer.

    Applies secret redaction to every message's content and every tool-call's
    arguments BEFORE serialization. Truncates long content to CONTENT_MAX chars
    (keeping CONTENT_HEAD from the start + CONTENT_TAIL from the end).
    """
    if not messages:
        return "(no messages)"

    lines: List[str] = []
    for msg in messages:
        role = msg.role if hasattr(msg, "role") else "unknown"
        text = _extract_text(msg)
        # Redact secrets before the LLM sees them.
        text = redact_sensitive_text(text)
        # Truncate long content.
        if len(text) > CONTENT_MAX:
            text = text[:CONTENT_HEAD] + "\n...[truncated]...\n" + text[-CONTENT_TAIL:]
        if text:
            lines.append(f"[{role}]: {text}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM Compactor
# ---------------------------------------------------------------------------


class LLMCompactor:
    """LLM-based context compactor with iterative summary updates.

    Algorithm:
      1. Prune old tool results (cheap, no LLM call)
      2. Protect head messages (system prompt + first exchange)
      3. Protect tail messages (most recent N)
      4. Summarize middle turns with the 13-section structured LLM prompt
      5. On subsequent compactions, iteratively update the previous summary
      6. Apply secret redaction to input AND output
      7. Track anti-thrash (skip after 2 consecutive <10% savings)
      8. Persist failure cooldown via session_db

    Falls back to text-flatten compaction if:
      - No aux client is configured
      - The aux client fails (cooldown prevents retry for 30-600s)
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
        # Iterative summary state.
        self._previous_summary: Optional[str] = None
        # Failure tracking.
        self._consecutive_failures = 0
        self._failure_cooldown_until = 0.0  # time.monotonic() units
        self._last_summary_error: Optional[str] = None
        self._last_summary_auth_failure = False
        self._last_summary_network_failure = False
        # Anti-thrash tracking.
        self._last_compression_savings_pct: float = 100.0
        self._ineffective_compression_count: int = 0
        # Session DB binding for cooldown persistence.
        self._session_db: Any = None
        self._session_id: str = ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset_session_state(self) -> None:
        """Reset all per-session state (call on /new, /reset, session end)."""
        self._previous_summary = None
        self._consecutive_failures = 0
        self._failure_cooldown_until = 0.0
        self._last_summary_error = None
        self._last_summary_auth_failure = False
        self._last_summary_network_failure = False
        self._last_compression_savings_pct = 100.0
        self._ineffective_compression_count = 0
        # Clear durable cooldown too.
        self._clear_compression_failure_cooldown()

    def set_aux_client(self, client: Optional[AuxClientProtocol]) -> None:
        """Set or replace the auxiliary LLM client.

        Reset failure state so a new client gets a fresh start.
        """
        self._aux_client = client
        self._consecutive_failures = 0
        self._failure_cooldown_until = 0.0

    def bind_session_state(
        self,
        session_db: Any = None,
        session_id: str = "",
    ) -> None:
        """Bind the current session row so durable cooldowns can round-trip.

        Called by the context engine on session start. The session_db is
        duck-typed — if it doesn't have the cooldown methods, persistence
        silently degrades to in-memory-only.
        """
        self._session_db = session_db
        self._session_id = session_id or ""
        # Rehydrate any durable cooldown from the DB.
        self.get_active_compression_failure_cooldown()

    def get_active_compression_failure_cooldown(self) -> Optional[Dict[str, Any]]:
        """Return the live cooldown dict (in-memory + DB lookup) or None.

        Checks in-memory first, then falls back to session_db. If the DB
        has a non-expired cooldown, rehydrates the in-memory mirror.
        """
        # In-memory check (monotonic clock).
        remaining = self._failure_cooldown_until - time.monotonic()
        if remaining > 0:
            return {
                "cooldown_until": time.time() + remaining,
                "remaining_seconds": remaining,
                "error": self._last_summary_error,
            }

        # DB lookup (wall-clock).
        if self._session_db is not None and self._session_id:
            get_fn = getattr(self._session_db, "get_compression_failure_cooldown", None)
            if get_fn is not None:
                try:
                    state = get_fn(self._session_id)
                    if state:
                        db_remaining = float(state.get("remaining_seconds") or 0)
                        if db_remaining > 0:
                            # Rehydrate in-memory mirror.
                            self._failure_cooldown_until = time.monotonic() + db_remaining
                            self._last_summary_error = state.get("error")
                            return {
                                "cooldown_until": float(state.get("cooldown_until") or 0),
                                "remaining_seconds": db_remaining,
                                "error": self._last_summary_error,
                            }
                except Exception as exc:
                    logger.debug("Cooldown DB lookup failed: %s", exc)
        return None

    def should_compress(self, prompt_tokens: Optional[int] = None) -> bool:
        """Return True if compaction should proceed (cooldown + anti-thrash gate).

        Checks:
          1. Failure cooldown — if active, return False.
          2. Anti-thrash — if ≥2 consecutive ineffective compressions, return False.
        """
        # Cooldown check.
        cooldown = self.get_active_compression_failure_cooldown()
        if cooldown is not None:
            logger.debug(
                "Compression deferred — summary LLM in cooldown for %.0fs more",
                cooldown["remaining_seconds"],
            )
            return False
        # Anti-thrash check.
        if self._ineffective_compression_count >= MAX_INEFFECTIVE_COMPRESSIONS:
            logger.warning(
                "Compression skipped — last %d compressions saved <%.0f%% each. "
                "Consider /new to start a fresh session, or /compress <topic> "
                "for focused compression.",
                self._ineffective_compression_count,
                INEFFECTIVE_SAVINGS_PCT_THRESHOLD,
            )
            return False
        return True

    async def compact(self, request: CompactionRequest) -> CompactionResult:
        """Compact the conversation using LLM summarization.

        Falls back to text-flatten compaction if LLM is unavailable or fails.
        Aborts (returns original messages) on auth/network failures unless
        ``abort_on_summary_failure`` is False (then falls back to text-flatten).
        """
        # Reset per-call failure flags.
        self._last_summary_auth_failure = False
        self._last_summary_network_failure = False

        # Force bypasses cooldown.
        if request.force:
            self._clear_compression_failure_cooldown()

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

        # Try LLM summarization first (if not in cooldown and aux client available).
        if self._aux_client is not None and not self._is_in_cooldown():
            try:
                result = await self._compact_with_llm(request, tokens_before, target)
                if result.success:
                    self._consecutive_failures = 0
                    self._previous_summary = result.summary
                    self._update_anti_thrash(tokens_before, result.tokens_after)
                    return result
            except Exception as exc:
                self._record_failure(exc)
                logger.warning("LLM compaction failed, falling back to text flatten: %s", exc)

        # Abort branch: auth or network failure → preserve session unchanged.
        if self._last_summary_auth_failure or self._last_summary_network_failure:
            logger.warning(
                "Compaction aborted due to %s failure — preserving session unchanged",
                "auth" if self._last_summary_auth_failure else "network",
            )
            return CompactionResult(
                success=False,
                compacted_messages=list(request.messages),
                tokens_before=tokens_before,
                tokens_after=tokens_before,
                error=self._last_summary_error or "aborted",
                method="aborted",
                aborted=True,
            )

        # Fallback: text-flatten compaction.
        result = self._compact_with_text_flatten(request, tokens_before, target)
        self._update_anti_thrash(tokens_before, result.tokens_after)
        return result

    # ------------------------------------------------------------------
    # Anti-thrash
    # ------------------------------------------------------------------

    def _update_anti_thrash(self, tokens_before: int, tokens_after: int) -> None:
        """Track compression effectiveness for anti-thrash.

        After each compaction, compute the savings percentage. If <10%,
        increment the ineffective counter. If ≥10%, reset it. After 2
        consecutive ineffective compressions, :meth:`should_compress`
        returns False until the next /new.
        """
        if tokens_before <= 0:
            return
        saved = tokens_before - tokens_after
        savings_pct = (saved / tokens_before * 100) if tokens_before > 0 else 0.0
        self._last_compression_savings_pct = savings_pct
        if savings_pct < INEFFECTIVE_SAVINGS_PCT_THRESHOLD:
            self._ineffective_compression_count += 1
        else:
            self._ineffective_compression_count = 0

    # ------------------------------------------------------------------
    # Cooldown management
    # ------------------------------------------------------------------

    def _is_in_cooldown(self) -> bool:
        """True if the LLM is in a failure cooldown period (in-memory check)."""
        return time.monotonic() < self._failure_cooldown_until

    def _record_failure(self, exc: Exception) -> None:
        """Record a failed LLM attempt, classify the error, and start cooldown.

        Failure classification:
          - Auth (401/403) → 60s cooldown, mark auth_failure.
          - Network (connection drop) → 30s cooldown, mark network_failure.
          - JSON decode (malformed proxy response) → 30s cooldown.
          - Timeout / 429 / 502/504 → 60s cooldown.
          - No provider configured → 600s cooldown.
          - Other → 60s cooldown.
        """
        self._consecutive_failures += 1
        self._last_summary_error = str(exc)

        error_str = str(exc).lower()
        status_code = getattr(exc, "status_code", None)

        # Classify the error.
        is_auth = (
            status_code in (401, 403)
            or "invalid api key" in error_str
            or "unauthorized" in error_str
            or "authentication" in error_str
        )
        is_network = (
            isinstance(exc, (ConnectionError, TimeoutError, OSError))
            or "connection" in error_str
            or "stream" in error_str and "close" in error_str
        )
        is_json_decode = isinstance(exc, json.JSONDecodeError) or "expecting value" in error_str
        is_no_provider = "no llm provider configured" in error_str
        is_transient = status_code in (408, 429, 502, 504) or "timeout" in error_str

        # Set failure flags for abort decisions.
        if is_auth:
            self._last_summary_auth_failure = True
        if is_network:
            self._last_summary_network_failure = True

        # Determine cooldown duration.
        if is_no_provider:
            cooldown_s = SUMMARY_FAILURE_COOLDOWN_SECONDS
        elif is_json_decode or is_network:
            cooldown_s = TRANSIENT_COOLDOWN_SECONDS
        elif is_transient or is_auth:
            cooldown_s = STANDARD_COOLDOWN_SECONDS
        elif self._consecutive_failures >= MAX_CONSECUTIVE_LLM_FAILURES:
            cooldown_s = SUMMARY_FAILURE_COOLDOWN_SECONDS
        else:
            cooldown_s = STANDARD_COOLDOWN_SECONDS

        self._failure_cooldown_until = time.monotonic() + cooldown_s
        logger.warning(
            "LLM compaction: %s failure (%s), entering %ds cooldown (consecutive=%d)",
            "auth" if is_auth else "network" if is_network else "transient" if is_transient else "other",
            error_str[:100],
            cooldown_s,
            self._consecutive_failures,
        )
        # Persist to DB.
        self._record_compression_failure_cooldown(cooldown_s, self._last_summary_error)

    def _record_compression_failure_cooldown(
        self,
        cooldown_seconds: float,
        error: Optional[str],
    ) -> None:
        """Persist the cooldown to session_db (wall-clock units)."""
        if self._session_db is None or not self._session_id:
            return
        record_fn = getattr(self._session_db, "record_compression_failure_cooldown", None)
        if record_fn is None:
            return
        try:
            cooldown_until = time.time() + cooldown_seconds
            record_fn(self._session_id, cooldown_until, error)
        except Exception as exc:
            logger.debug("Cooldown DB write failed: %s", exc)

    def _clear_compression_failure_cooldown(self) -> None:
        """Clear the cooldown (in-memory + DB)."""
        self._failure_cooldown_until = 0.0
        self._last_summary_error = None
        if self._session_db is None or not self._session_id:
            return
        clear_fn = getattr(self._session_db, "clear_compression_failure_cooldown", None)
        if clear_fn is None:
            return
        try:
            clear_fn(self._session_id)
        except Exception as exc:
            logger.debug("Cooldown DB clear failed: %s", exc)

    # ------------------------------------------------------------------
    # LLM compaction
    # ------------------------------------------------------------------

    async def _compact_with_llm(
        self, request: CompactionRequest, tokens_before: int, target: int
    ) -> CompactionResult:
        """Compact using LLM summarization with the 13-section template."""
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

        # Prune old tool results from middle (cheap, no LLM call).
        pruned_middle = self._prune_tool_results(middle)

        # Strip images from old messages.
        pruned_middle = self._strip_images(pruned_middle)

        # Build the structured LLM prompt.
        summary_budget = min(MAX_SUMMARY_TOKENS, target)
        prompt = _build_summary_prompt(
            pruned_middle,
            request.previous_summary,
            summary_budget,
            focus_topic=request.focus_topic,
        )

        # Call the aux model.
        assert self._aux_client is not None  # for type checker
        summary = await self._aux_client.complete(
            prompt, max_tokens=summary_budget, system=None, temperature=0.0,
        )

        if not summary or not summary.strip():
            raise RuntimeError("LLM returned empty summary")

        # Defense in depth: redact secrets from the LLM output too.
        summary = redact_sensitive_text(summary.strip())

        # Build the compacted message list: head + summary message + tail.
        summary_text = f"{SUMMARY_PREFIX}\n\n{summary}\n\n{_SUMMARY_END_MARKER}"
        summary_message = ConversationMessage(
            role="assistant",
            content=[TextBlock(text=summary_text)],
        )
        compacted = list(head) + [summary_message] + list(tail)
        tokens_after = sum(estimate_message_tokens(m) for m in compacted)

        return CompactionResult(
            success=True,
            compacted_messages=compacted,
            summary=summary,
            method="llm",
            tokens_before=tokens_before,
            tokens_after=tokens_after,
        )

    # ------------------------------------------------------------------
    # Text-flatten fallback
    # ------------------------------------------------------------------

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
        lines: List[str] = [f"{SUMMARY_PREFIX}", ""]
        for msg in middle:
            text = _extract_text(msg)
            if text:
                text = redact_sensitive_text(text)
                lines.append(f"[{msg.role}]: {text[:500]}")
        lines.append("")
        lines.append(_SUMMARY_END_MARKER)
        summary = "\n".join(lines)
        # Redact the full summary too (defense in depth).
        summary = redact_sensitive_text(summary)

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

    # ------------------------------------------------------------------
    # Tool-result pruning
    # ------------------------------------------------------------------

    def _prune_tool_results(self, messages: List[ConversationMessage]) -> List[ConversationMessage]:
        """Prune old tool results to breadcrumbs (tool name + truncated args).

        Tool results consume tokens but the model has already seen and
        reasoned about them. Replacing them with a breadcrumb preserves
        the fact that the tool was called without the full output.
        """
        pruned: List[ConversationMessage] = []
        for msg in messages:
            has_tool_result = any(
                hasattr(b, "type") and getattr(b, "type", "") == "tool_result"
                for b in msg.content
            )
            if has_tool_result:
                new_content = []
                for block in msg.content:
                    if hasattr(block, "type") and getattr(block, "type", "") == "tool_result":
                        tool_use_id = getattr(block, "tool_use_id", "unknown")
                        original_content = getattr(block, "content", "")
                        if isinstance(original_content, str):
                            breadcrumb = original_content[:TOOL_RESULT_BREADCRUMB_CHARS]
                        else:
                            breadcrumb = "[tool result pruned]"
                        new_content.append(TextBlock(
                            text=f"[tool_result for {tool_use_id}]: {breadcrumb}"
                        ))
                    else:
                        new_content.append(block)
                pruned.append(ConversationMessage(role=msg.role, content=new_content))
            else:
                pruned.append(msg)
        return pruned

    def _strip_images(self, messages: List[ConversationMessage]) -> List[ConversationMessage]:
        """Replace image content blocks with a placeholder text block.

        Images consume ~1600 tokens each but can't be meaningfully
        summarized. Replacing them with a placeholder saves tokens
        without losing the fact that an image was present.
        """
        stripped: List[ConversationMessage] = []
        for msg in messages:
            new_content = []
            for block in msg.content:
                block_type = getattr(block, "type", "")
                if block_type in {"image", "image_url", "input_image"}:
                    new_content.append(TextBlock(text="[image stripped to save context]"))
                else:
                    new_content.append(block)
            stripped.append(ConversationMessage(role=msg.role, content=new_content))
        return stripped


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _strip_summary_prefix(summary: str) -> str:
    """Strip the SUMMARY_PREFIX + end marker from a summary body.

    Used when carrying forward ``_previous_summary`` into the next
    iterative-update prompt — the prefix/marker are re-appended on
    insertion, so carrying them forward would duplicate the directives.
    """
    text = (summary or "").strip()
    for prefix in (SUMMARY_PREFIX, LEGACY_SUMMARY_PREFIX, *_HISTORICAL_SUMMARY_PREFIXES):
        if text.startswith(prefix):
            text = text[len(prefix):].lstrip()
            break
    if text.endswith(_SUMMARY_END_MARKER):
        text = text[: -len(_SUMMARY_END_MARKER)].rstrip()
    return text


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
    "DEFAULT_HEAD_PROTECT",
    "DEFAULT_TAIL_PROTECT",
    "HISTORICAL_IN_PROGRESS_HEADING",
    "HISTORICAL_PENDING_ASKS_HEADING",
    "HISTORICAL_REMAINING_WORK_HEADING",
    "HISTORICAL_TASK_HEADING",
    "INEFFECTIVE_SAVINGS_PCT_THRESHOLD",
    "LLMCompactor",
    "LEGACY_SUMMARY_PREFIX",
    "MAX_INEFFECTIVE_COMPRESSIONS",
    "MAX_SUMMARY_TOKENS",
    "STANDARD_COOLDOWN_SECONDS",
    "SUMMARY_FAILURE_COOLDOWN_SECONDS",
    "SUMMARY_PREFIX",
    "TRANSIENT_COOLDOWN_SECONDS",
    "_build_summary_prompt",
    "_build_summarizer_preamble",
    "_build_template_sections",
    "_build_temporal_anchoring_rule",
    "_serialize_for_summary",
    "_strip_summary_prefix",
    "get_default_compactor",
    "redact_sensitive_text",
]
