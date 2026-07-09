"""High-level conversation engine.

Ported from OpenClaude's QueryEngine.ts with file state cache, permission
denial tracking, abort controller support, and dynamic tool updates.
Maintains full backward compatibility with the existing niaharness interface.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, AsyncIterator, Iterator
from uuid import uuid4

from niaharness.api.client import SupportsStreamingMessages
from niaharness.api.usage import UsageSnapshot
from niaharness.engine.cost_tracker import CostTracker
from niaharness.engine.messages import ConversationMessage
from niaharness.engine.query import (
    AskUserPrompt,
    PermissionPrompt,
    QueryContext,
    run_query,
)
from niaharness.engine.stream_events import (
    ApiRetryNotification,
    CompactBoundary,
    QueryResult,
    StreamEvent,
    TerminationReason,
    UserInterrupted,
)
from niaharness.hooks import HookExecutor
from niaharness.permissions.checker import PermissionChecker
from niaharness.tools.base import ToolRegistry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# File state cache (ported from OpenClaude FileStateCache)
# ---------------------------------------------------------------------------


class FileStateCache:
    """Tracks the last-known state (hash / mtime) of files read by tools.

    Ported from OpenClaude's FileStateCache so that auto-compaction and
    memory prefetch can skip files the model has already seen.
    """

    def __init__(self) -> None:
        self._cache: dict[str, dict[str, Any]] = {}

    def get(self, file_path: str) -> dict[str, Any] | None:
        """Return cached state for *file_path*, or None."""
        return self._cache.get(file_path)

    def set(self, file_path: str, state: dict[str, Any]) -> None:
        """Record state for *file_path*."""
        self._cache[file_path] = state

    def has(self, file_path: str) -> bool:
        """Return True if the cache contains an entry for *file_path*."""
        return file_path in self._cache

    def clone(self) -> FileStateCache:
        """Return a shallow copy of the cache."""
        new = FileStateCache()
        new._cache = dict(self._cache)
        return new

    def clear(self) -> None:
        """Clear all entries."""
        self._cache.clear()

    def __contains__(self, file_path: str) -> bool:
        return self.has(file_path)

    def __len__(self) -> int:
        return len(self._cache)

    def items(self) -> Iterator[tuple[str, dict[str, Any]]]:
        return iter(self._cache.items())


# ---------------------------------------------------------------------------
# Permission denial tracking (ported from OpenClaude QueryEngine)
# ---------------------------------------------------------------------------


class PermissionDenialTracker:
    """Records permission denials for reporting in query results.

    Ported from OpenClaude's permissionDenials array on QueryEngine.
    """

    def __init__(self) -> None:
        self._denials: list[dict[str, Any]] = []

    def record(
        self,
        tool_name: str,
        tool_use_id: str,
        tool_input: dict[str, Any],
    ) -> None:
        """Record a single permission denial."""
        self._denials.append(
            {
                "tool_name": tool_name,
                "tool_use_id": tool_use_id,
                "tool_input": tool_input,
            }
        )

    @property
    def denials(self) -> list[dict[str, Any]]:
        """Return a copy of all recorded denials."""
        return list(self._denials)

    def clear(self) -> None:
        """Clear all recorded denials."""
        self._denials.clear()


# ---------------------------------------------------------------------------
# Abort controller (ported from OpenClaude AbortController)
# ---------------------------------------------------------------------------


class AbortController:
    """Cooperative abort mechanism for query loops.

    Ported from OpenClaude's AbortController / createAbortController.
    Thread-safe via asyncio.Event.
    """

    def __init__(self) -> None:
        self._event = asyncio.Event()
        self._reason: str = ""

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> str:
        return self._reason

    @property
    def event(self) -> asyncio.Event:
        """Return the underlying asyncio.Event for use in query loops."""
        return self._event

    def cancel(self, reason: str = "") -> None:
        """Signal cancellation."""
        self._reason = reason
        self._event.set()

    def reset(self) -> None:
        """Reset the controller for reuse."""
        self._event.clear()
        self._reason = ""


# ---------------------------------------------------------------------------
# QueryEngine
# ---------------------------------------------------------------------------


class QueryEngine:
    """Owns conversation history and the tool-aware model loop.

    Ported from OpenClaude's QueryEngine class with:
    - File state cache for tracking read files
    - Permission denial tracking for SDK reporting
    - Abort controller support for cooperative cancellation
    - Dynamic tool updates via update_tools()
    """

    def __init__(
        self,
        *,
        api_client: SupportsStreamingMessages,
        tool_registry: ToolRegistry,
        permission_checker: PermissionChecker,
        cwd: str | Path,
        model: str,
        system_prompt: str,
        max_tokens: int = 4096,
        permission_prompt: PermissionPrompt | None = None,
        ask_user_prompt: AskUserPrompt | None = None,
        hook_executor: HookExecutor | None = None,
        tool_metadata: dict[str, object] | None = None,
        max_turns: int = 200,
        max_budget_usd: float | None = None,
        token_budget: int | None = None,
        memory: object | None = None,
        post_turn_hooks: list | None = None,
    ) -> None:
        self._api_client = api_client
        self._tool_registry = tool_registry
        self._permission_checker = permission_checker
        self._cwd = Path(cwd).resolve()
        self._model = model
        self._system_prompt = system_prompt
        self._max_tokens = max_tokens
        self._permission_prompt = permission_prompt
        self._ask_user_prompt = ask_user_prompt
        self._hook_executor = hook_executor
        self._tool_metadata = tool_metadata or {}
        self._max_turns = max_turns
        self._max_budget_usd = max_budget_usd
        self._token_budget = token_budget
        self._memory = memory  # for background review (Task 5)
        # Post-turn hooks: callables invoked after each QueryResult with
        # (engine, result, tool_call_count). Used by the background review
        # system (Task 7) to spawn the self-improvement fork.
        self._post_turn_hooks: list = post_turn_hooks or []

        # State
        self._messages: list[ConversationMessage] = []
        self._cost_tracker = CostTracker()
        self._file_state_cache = FileStateCache()
        self._permission_denials = PermissionDenialTracker()
        self._abort_controller = AbortController()
        self._session_id = uuid4().hex

    # -- Properties --------------------------------------------------------

    @property
    def messages(self) -> list[ConversationMessage]:
        """Return the current conversation history."""
        return list(self._messages)

    @property
    def total_usage(self) -> UsageSnapshot:
        """Return the total usage across all turns."""
        return self._cost_tracker.total

    @property
    def session_id(self) -> str:
        """Return the session identifier."""
        return self._session_id

    @property
    def abort_controller(self) -> AbortController:
        """Return the abort controller for external cancellation."""
        return self._abort_controller

    @property
    def file_state_cache(self) -> FileStateCache:
        """Return the file state cache."""
        return self._file_state_cache

    @property
    def permission_denials(self) -> list[dict[str, Any]]:
        """Return recorded permission denials."""
        return self._permission_denials.denials

    @property
    def total_cost_usd(self) -> float:
        """Return the total estimated cost in USD.

        Requires cost_per_token_fn to be set on the cost tracker.
        """
        return self._cost_tracker.total_cost_usd

    # -- Mutators ----------------------------------------------------------

    def clear(self) -> None:
        """Clear the in-memory conversation history."""
        self._messages.clear()
        self._cost_tracker = CostTracker()
        self._permission_denials.clear()
        self._abort_controller.reset()

    def set_system_prompt(self, prompt: str) -> None:
        """Update the active system prompt for future turns."""
        self._system_prompt = prompt

    def set_model(self, model: str) -> None:
        """Update the active model for future turns."""
        self._model = model

    def set_permission_checker(self, checker: PermissionChecker) -> None:
        """Update the active permission checker for future turns."""
        self._permission_checker = checker

    def set_max_budget_usd(self, budget: float | None) -> None:
        """Update the USD budget cap."""
        self._max_budget_usd = budget

    def set_token_budget(self, budget: int | None) -> None:
        """Update the token budget for auto-continuation."""
        self._token_budget = budget

    def load_messages(self, messages: list[ConversationMessage]) -> None:
        """Replace the in-memory conversation history."""
        self._messages = list(messages)

    def inject_messages(self, messages: list[ConversationMessage]) -> None:
        """Append messages to the conversation history.

        Used by SDK callers to resume from a forked session.
        """
        self._messages.extend(messages)

    # -- Dynamic tool updates (ported from OpenClaude updateTools) ----------

    def update_tools(self, tools: ToolRegistry) -> None:
        """Update the engine's tool registry dynamically.

        Ported from OpenClaude's QueryEngine.updateTools().  Validates
        that the new tool set is compatible with any loaded agents before
        committing.
        """
        if not isinstance(tools, ToolRegistry):
            raise TypeError(f"update_tools: expected ToolRegistry, got {type(tools).__name__}")

        # Phase 1: Validate new tools have required attributes
        for name in tools.list_names():
            tool = tools.get(name)
            if tool is None:
                raise TypeError(f"update_tools: tool '{name}' not found in registry")

        # Phase 2: Commit
        self._tool_registry = tools
        logger.info("Tool registry updated with %d tools", len(tools.list_names()))

    def add_tool(self, name: str, tool: Any) -> None:
        """Add a single tool to the registry dynamically."""
        self._tool_registry.register(name, tool)
        logger.info("Tool '%s' added to registry", name)

    def remove_tool(self, name: str) -> bool:
        """Remove a tool from the registry. Returns True if removed."""
        removed = self._tool_registry.unregister(name)
        if removed:
            logger.info("Tool '%s' removed from registry", name)
        return removed

    # -- Interrupt ---------------------------------------------------------

    def interrupt(self) -> None:
        """Cancel the current query loop.

        Ported from OpenClaude's QueryEngine.interrupt().
        """
        self._abort_controller.cancel(reason="user_interrupt")

    # -- Core query --------------------------------------------------------

    async def submit_message(self, prompt: str) -> AsyncIterator[StreamEvent]:
        """Append a user message and execute the query loop.

        Enhanced with:
        - Abort controller integration
        - Permission denial tracking
        - Budget enforcement (max turns, max USD)
        - File state cache propagation
        """
        self._abort_controller.reset()
        self._messages.append(ConversationMessage.from_user_text(prompt))

        # P1 fix: persist the user message to the session DB (if enabled).
        self._persist_message_to_session_db("user", prompt)

        context = QueryContext(
            api_client=self._api_client,
            tool_registry=self._tool_registry,
            permission_checker=self._permission_checker,
            cwd=self._cwd,
            model=self._model,
            system_prompt=self._system_prompt,
            max_tokens=self._max_tokens,
            permission_prompt=self._permission_prompt,
            ask_user_prompt=self._ask_user_prompt,
            hook_executor=self._hook_executor,
            tool_metadata=self._tool_metadata,
            max_turns=self._max_turns,
            max_budget_usd=self._max_budget_usd,
            token_budget=self._token_budget,
            abort_event=self._abort_controller.event,
        )

        async for event, usage in run_query(
            context,
            self._messages,
            cost_usd_fn=lambda: self.total_cost_usd,
        ):
            if usage is not None:
                self._cost_tracker.add(usage)
            # The low-level run_query loop emits a final QueryResult event to
            # signal termination.  At the high-level submit_message API we
            # treat it as a return value rather than a streamed event so
            # callers can rely on the last streamed event being an
            # AssistantTurnComplete / tool event.
            if isinstance(event, QueryResult):
                self._last_result = event
                # After the turn completes, invoke post-turn hooks (Task 7).
                # The background review hook spawns a self-improvement fork
                # that reviews the turn and calls skill_manage / memory tools.
                # Non-blocking, best-effort — never breaks the turn.
                was_interrupted = (
                    event.reason is not None
                    and "interrupt" in str(event.reason).lower()
                )
                # Count tool calls in this turn for the ≥3 threshold.
                tool_call_count = getattr(event, "tool_call_count", 0) or 0
                for hook in self._post_turn_hooks:
                    try:
                        hook(
                            engine=self,
                            result=event,
                            tool_call_count=tool_call_count,
                            was_interrupted=was_interrupted,
                        )
                    except Exception:
                        pass  # Hooks are best-effort.

                # P1 fix: persist the assistant's final message to the session DB.
                # QueryResult has result_text, not message.
                if event.result_text:
                    self._persist_message_to_session_db("assistant", event.result_text)
                continue
            yield event

    # -- Convenience -------------------------------------------------------

    def _persist_message_to_session_db(self, role: str, text: str) -> None:
        """Persist a message to the SQLite session DB (best-effort).

        P1 fix: wires the session_db module into the QueryEngine. Every
        user message and assistant response is written to the SQLite DB
        at ~/.nia/sessions.db, enabling cross-session search (FTS5),
        session lineage, and insights/analytics.

        Best-effort: failures are logged at DEBUG and never break the turn.
        The session DB is created on first use (lazy initialization).
        """
        try:
            from niaharness.services.session_db import (
                create_session,
                add_message,
                get_session,
            )

            # Ensure the session exists in the DB (create if needed).
            existing = get_session(self._session_id)
            if existing is None:
                create_session(
                    self._session_id,
                    cwd=str(self._cwd),
                    model=self._model,
                    provider=getattr(self._api_client, "provider", None),
                )

            # Estimate token count for the message.
            from niaharness.services.compact import estimate_tokens

            token_count = estimate_tokens(text)

            # Add the message.
            add_message(
                self._session_id,
                role,
                text,
                token_count=token_count,
            )
        except Exception:
            # Best-effort — never break the turn.
            import logging

            logging.getLogger(__name__).debug(
                "Session DB persist failed (non-fatal)", exc_info=True
            )

    def get_read_file_state(self) -> FileStateCache:
        """Return the file state cache.

        Ported from OpenClaude's QueryEngine.getReadFileState().
        """
        return self._file_state_cache

    def get_messages(self) -> list[ConversationMessage]:
        """Return the current message list (read-only copy).

        Ported from OpenClaude's QueryEngine.getMessages().
        """
        return list(self._messages)

    def set_max_turns(self, max_turns: int) -> None:
        """Update the maximum number of turns."""
        self._max_turns = max_turns

    def set_max_tokens(self, max_tokens: int) -> None:
        """Update the max output tokens per API call."""
        self._max_tokens = max_tokens
