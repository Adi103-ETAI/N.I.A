"""Pluggable context engine — ABC + plugin discovery for context management.

Ported from the reference project's context engine pattern (4,835 lines),
providing an extensible system for managing conversation context.

The context engine is responsible for:
  - Deciding when to compact the conversation
  - Choosing what to keep vs. summarize
  - Producing the final message list sent to the model
  - Tracking token budgets and savings from compaction

NIA ships with two implementations:
  1. ``SimpleContextEngine`` — the default, uses the existing text-flatten
     compaction from ``services/compact.py``. No LLM calls.
  2. ``LLMContextEngine`` — uses the LLM-based compaction from
     ``engine/llm_compaction.py`` for higher-quality summaries.

Custom engines can be registered via the plugin system or set via
``context.engine`` in config.yaml.

Why pluggable?
--------------
Different users have different needs:
  - **CLI users** with short sessions → SimpleContextEngine (cheap, fast)
  - **Long-running sessions** → LLMContextEngine (better long-term context)
  - **Gateway users** → a custom engine that considers multiple sessions
  - **Enterprise users** → an engine that redacts PII before sending to the model

Usage::

    from niaharness.context_engine import get_context_engine

    engine = get_context_engine()
    messages = await engine.build_messages(history, system_prompt, new_user_msg)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from niaharness.engine.messages import ConversationMessage, TextBlock

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Context engine ABC
# ---------------------------------------------------------------------------


@dataclass
class ContextBuildResult:
    """Result of building context for a model call.

    Attributes:
        messages: The final message list to send to the model.
        token_count: Estimated tokens in the final message list.
        was_compacted: True if compaction was applied.
        compaction_method: "none" | "text_flatten" | "llm" | "truncate"
        tokens_saved: Estimated tokens saved by compaction (0 if not compacted).
    """

    messages: List[ConversationMessage]
    token_count: int = 0
    was_compacted: bool = False
    compaction_method: str = "none"
    tokens_saved: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class ContextEngine(ABC):
    """Abstract base for context engines.

    A context engine takes the conversation history + system prompt + new
    user message and produces the final message list to send to the model.
    It may compact the history if it exceeds the token budget.

    Lifecycle hooks:
      - ``on_session_start(session_id)`` — called when a session begins
      - ``on_session_end(session_id)`` — called when a session ends
      - ``on_session_reset()`` — called on /new or /reset
      - ``update_from_response(usage)`` — called after each model response
        with the actual token usage (for budget tracking)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Engine name (e.g. 'simple', 'llm')."""

    @abstractmethod
    async def build_messages(
        self,
        history: List[ConversationMessage],
        system_prompt: str,
        new_messages: List[ConversationMessage],
        *,
        model: str = "",
        context_window: int = 32_000,
        max_tokens: int = 4096,
    ) -> ContextBuildResult:
        """Build the final message list for a model call.

        Args:
            history: The conversation history (may be modified/compacted).
            system_prompt: The system prompt (always preserved).
            new_messages: New messages to append (user, tool results, etc.).
            model: The model name (for token estimation + context window).
            context_window: The model's context window in tokens.
            max_tokens: Max tokens for the response (reserved from budget).

        Returns:
            ContextBuildResult with the final message list.
        """

    def on_session_start(self, session_id: str, **kwargs: Any) -> None:
        """Called when a session begins. Override for custom behavior."""
        pass

    def on_session_end(self, session_id: str, messages: List[ConversationMessage]) -> None:
        """Called when a session ends. Override for custom behavior."""
        pass

    def on_session_reset(self) -> None:
        """Called on /new or /reset. Override to clear per-session state."""
        pass

    def update_from_response(self, usage: Any) -> None:
        """Called after each model response with the actual token usage.

        Override to track real token counts (vs. estimates) for budget
        tracking. ``usage`` is typically a ``UsageSnapshot``.
        """
        pass


# ---------------------------------------------------------------------------
# Simple context engine (text-flatten compaction)
# ---------------------------------------------------------------------------


class SimpleContextEngine(ContextEngine):
    """Default context engine — uses text-flatten compaction.

    No LLM calls. Uses the existing ``services/compact.py`` module for
    token estimation and compaction. Suitable for short-to-medium sessions
    where the compaction quality of text-flattening is acceptable.
    """

    @property
    def name(self) -> str:
        return "simple"

    async def build_messages(
        self,
        history: List[ConversationMessage],
        system_prompt: str,
        new_messages: List[ConversationMessage],
        *,
        model: str = "",
        context_window: int = 32_000,
        max_tokens: int = 4096,
    ) -> ContextBuildResult:
        from niaharness.services.compact import (
            AutoCompactState,
            auto_compact_if_needed,
            estimate_message_tokens,
        )

        # Combine history + new messages.
        all_messages = list(history) + list(new_messages)
        tokens_before = sum(estimate_message_tokens(m) for m in all_messages)

        # Check if compaction is needed.
        budget = context_window - max_tokens - 1000  # 1K safety margin
        if tokens_before <= budget:
            return ContextBuildResult(
                messages=all_messages,
                token_count=tokens_before,
                was_compacted=False,
                compaction_method="none",
                tokens_saved=0,
            )

        # Compact.
        state = AutoCompactState()
        compacted = auto_compact_if_needed(
            messages=all_messages,
            model=model,
            threshold=budget,
            state=state,
        )
        tokens_after = sum(estimate_message_tokens(m) for m in compacted)

        return ContextBuildResult(
            messages=compacted,
            token_count=tokens_after,
            was_compacted=True,
            compaction_method="text_flatten",
            tokens_saved=max(0, tokens_before - tokens_after),
            metadata={"compaction_count": getattr(state, "compaction_count", 0)},
        )


# ---------------------------------------------------------------------------
# LLM context engine (LLM-based compaction)
# ---------------------------------------------------------------------------


class LLMContextEngine(ContextEngine):
    """Context engine that uses LLM-based compaction for higher-quality summaries.

    Uses ``engine/llm_compaction.LLMCompactor`` to summarize old messages
    with an LLM. Falls back to text-flatten compaction if the LLM is
    unavailable or fails.
    """

    def __init__(self) -> None:
        from niaharness.engine.llm_compaction import LLMCompactor

        self._compactor = LLMCompactor()
        # P1 fix: auto-wire the auxiliary client so LLM summarization
        # actually works. Without this, _aux_client is always None and
        # compact() falls back to text_flatten every time.
        self._try_wire_aux_client()

    def _try_wire_aux_client(self) -> None:
        """Attempt to wire the auxiliary client into the compactor.

        Best-effort: if the aux client isn't configured (no env vars, no
        config), this is a no-op and the compactor falls back to text-flatten.
        """
        try:
            import asyncio
            from niaharness.auxiliary import get_aux_client

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop is not None:
                # We're inside an event loop — schedule the wiring.
                async def _wire():
                    client = await get_aux_client()
                    self._compactor.set_aux_client(client)
                loop.create_task(_wire())
            else:
                # No event loop — try sync (may return None if not configured).
                # We can't await get_aux_client() here, so just check config.
                from niaharness.auxiliary import get_aux_config
                config = get_aux_config()
                if config is not None:
                    from niaharness.auxiliary import AuxiliaryClient
                    self._compactor.set_aux_client(AuxiliaryClient(config))
        except Exception:
            pass  # Best-effort — text-flatten fallback is fine.

    @property
    def name(self) -> str:
        return "llm"

    async def build_messages(
        self,
        history: List[ConversationMessage],
        system_prompt: str,
        new_messages: List[ConversationMessage],
        *,
        model: str = "",
        context_window: int = 32_000,
        max_tokens: int = 4096,
        focus_topic: Optional[str] = None,
        force: bool = False,
    ) -> ContextBuildResult:
        """Build the final message list for a model call.

        Args:
            history: The conversation history (may be modified/compacted).
            system_prompt: The system prompt (always preserved).
            new_messages: New messages to append (user, tool results, etc.).
            model: The model name (for token estimation + context window).
            context_window: The model's context window in tokens.
            max_tokens: Max tokens for the response (reserved from budget).
            focus_topic: Optional focus topic for prioritized summarization
                (e.g. from /compress <topic>). When provided, the LLM
                summary preserves 60-70% of its budget for focus-topic
                content.
            force: If True, bypass the anti-thrash / cooldown gate (for
                manual /compress).

        Returns:
            ContextBuildResult with the final message list.
        """
        from niaharness.engine.llm_compaction import CompactionRequest
        from niaharness.services.compact import estimate_message_tokens

        # Combine history + new messages.
        all_messages = list(history) + list(new_messages)
        tokens_before = sum(estimate_message_tokens(m) for m in all_messages)

        # Check if compaction is needed.
        budget = context_window - max_tokens - 1000
        if tokens_before <= budget:
            return ContextBuildResult(
                messages=all_messages,
                token_count=tokens_before,
                was_compacted=False,
                compaction_method="none",
                tokens_saved=0,
            )

        # Anti-thrash + cooldown gate (unless force=True).
        if not force and not self._compactor.should_compress(prompt_tokens=tokens_before):
            # Gate says skip — return messages unchanged.
            return ContextBuildResult(
                messages=all_messages,
                token_count=tokens_before,
                was_compacted=False,
                compaction_method="skipped",
                tokens_saved=0,
                metadata={
                    "skip_reason": "cooldown_or_anti_thrash",
                    "ineffective_count": getattr(self._compactor, "_ineffective_compression_count", 0),
                },
            )

        # Compact with LLM.
        request = CompactionRequest(
            messages=all_messages,
            model=model,
            context_window=context_window,
            target_tokens=budget,
            previous_summary=getattr(self._compactor, "_previous_summary", None),
            focus_topic=focus_topic,
            force=force,
        )
        result = await self._compactor.compact(request)

        return ContextBuildResult(
            messages=result.compacted_messages,
            token_count=result.tokens_after,
            was_compacted=result.success and result.method != "none",
            compaction_method=result.method,
            tokens_saved=max(0, tokens_before - result.tokens_after),
            metadata={
                "savings_pct": result.savings_pct,
                "aborted": result.aborted,
                "error": result.error,
            },
        )

    def on_session_start(self, session_id: str, **kwargs: Any) -> None:
        """Called when a session begins.

        Binds the session DB + session ID to the compactor so durable
        cooldowns can round-trip. The session_db is passed via kwargs.
        """
        session_db = kwargs.get("session_db")
        self._compactor.bind_session_state(session_db=session_db, session_id=session_id)

    def on_session_reset(self) -> None:
        self._compactor.reset_session_state()

    def on_session_end(self, session_id: str, messages: List[ConversationMessage]) -> None:
        self._compactor.reset_session_state()


# ---------------------------------------------------------------------------
# Registry + factory
# ---------------------------------------------------------------------------


_ENGINES: Dict[str, type] = {
    "simple": SimpleContextEngine,
    "llm": LLMContextEngine,
}

_default_engine: Optional[ContextEngine] = None


def register_engine(name: str, engine_class: type) -> None:
    """Register a custom context engine class."""
    _ENGINES[name] = engine_class


def get_context_engine(name: Optional[str] = None) -> ContextEngine:
    """Return the named context engine (or the configured default).

    Resolution order for the default:
      1. ``context.engine`` in config.yaml
      2. ``"simple"`` (the default)
    """
    global _default_engine
    if name is None and _default_engine is not None:
        return _default_engine

    if name is None:
        # Try to load from config.
        try:
            from niaharness.config.settings import load_settings

            settings = load_settings()
            context_section = getattr(settings, "context", None) or {}
            if isinstance(context_section, dict):
                name = context_section.get("engine", "simple")
            else:
                name = "simple"
        except Exception:
            name = "simple"

    engine_class = _ENGINES.get(name or "simple", SimpleContextEngine)
    engine = engine_class()

    if name is None:
        _default_engine = engine

    return engine


def reset_default_engine() -> None:
    """Reset the default engine (useful for config changes / tests)."""
    global _default_engine
    _default_engine = None


__all__ = [
    "ContextBuildResult",
    "ContextEngine",
    "LLMContextEngine",
    "SimpleContextEngine",
    "get_context_engine",
    "register_engine",
    "reset_default_engine",
]
