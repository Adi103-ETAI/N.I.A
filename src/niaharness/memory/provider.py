"""Memory provider protocol — ABC for pluggable memory backends.

Ported from Hermes Agent's ``agent/memory_provider.py``. Defines the
:class:`MemoryProvider` ABC that all memory backends (built-in JSON,
Honcho, mem0, etc.) must implement.

The built-in provider wraps NIA's existing file-based Memory class.
External providers (vector DB, Honcho, etc.) implement the same protocol
and drop in without touching the agent loop.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class MemoryProvider(ABC):
    """Abstract base for memory providers.

    A memory provider supplies:
      - Tool schemas (function-calling tools the agent can invoke).
      - A system-prompt block (static text for the system prompt).
      - Prefetch (recall relevant memories before each turn).
      - Sync (persist a completed turn for future recall).
      - Lifecycle hooks (turn start, session end, session switch, shutdown).

    The built-in provider (``BuiltinJsonMemoryProvider``) is always
    registered first. At most one external provider may be registered
    alongside it.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier (e.g. 'builtin', 'honcho', 'mem0')."""

    @abstractmethod
    def is_available(self) -> bool:
        """Check config + deps without network. True if this provider can run."""

    @abstractmethod
    def initialize(self, session_id: str, **kwargs: Any) -> None:
        """Called once at startup. kwargs include 'hermes_home' / 'nia_home'."""

    @abstractmethod
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Return OpenAI function-calling tool schemas (bare {name, description, parameters})."""

    # -- Optional overrides (default no-op / empty) ---------------------

    def system_prompt_block(self) -> str:
        """Static text for the system prompt. Default: empty."""
        return ""

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Synchronous recall before each turn. Default: empty."""
        return ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Background prefetch for next turn. Default: no-op."""
        pass

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Persist a completed turn. Default: no-op."""
        pass

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs: Any) -> str:
        """Dispatch a tool call. Default: not implemented."""
        raise NotImplementedError(f"Provider '{self.name}' does not handle tool '{tool_name}'")

    def shutdown(self) -> None:
        """Clean exit. Default: no-op."""
        pass

    def on_turn_start(self, turn_number: int, message: str, **kwargs: Any) -> None:
        """Per-turn tick. Default: no-op."""
        pass

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """Session-end extraction. Default: no-op."""
        pass

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        **kwargs: Any,
    ) -> None:
        """Session ID rotation. Default: no-op."""
        pass

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        """Extract before compression. Default: empty."""
        return ""

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Mirror built-in memory writes. Default: no-op."""
        pass

    def on_delegation(
        self,
        task: str,
        result: str,
        *,
        child_session_id: str = "",
        **kwargs: Any,
    ) -> None:
        """Parent-side observation of subagent work. Default: no-op."""
        pass


__all__ = ["MemoryProvider"]
