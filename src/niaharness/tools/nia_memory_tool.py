"""Tool for N.I.A memory operations.

Exposes NIA's short-term and long-term memory system as a tool,
allowing the brain to search, store, and retrieve memories.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from niaharness.tools.base import BaseTool, ToolExecutionContext, ToolResult


class NiaMemoryInput(BaseModel):
    """Arguments for NIA memory operations."""

    action: str = Field(
        description="Operation: 'search' to find relevant memories, "
        "'add_fact' to store a fact, 'add_preference' to store a preference, "
        "'list_preferences' to see all preferences, 'recent' to get recent conversation, "
        "'stats' to get memory statistics"
    )
    query: str | None = Field(default=None, description="Search query (for search action)")
    key: str | None = Field(default=None, description="Preference key (for add_preference)")
    value: str | None = Field(default=None, description="Preference value (for add_preference)")
    fact: str | None = Field(default=None, description="Fact to store (for add_fact)")
    limit: int = Field(default=5, description="Max results for search/recent")


class NiaMemoryTool(BaseTool):
    """Access N.I.A's memory system.

    NIA maintains short-term memory (recent conversation) and long-term memory
    (preferences, patterns, facts). Use this tool to search for relevant context,
    store important information, or retrieve user preferences.
    """

    name = "nia_memory"
    description = (
        "Access N.I.A's memory system. Actions: search (find relevant memories by query), "
        "add_fact (store an important fact), add_preference (store a user preference with key=value), "
        "list_preferences (see all stored preferences), recent (get recent conversation turns), "
        "stats (get memory statistics)"
    )
    input_model = NiaMemoryInput

    def __init__(self, memory: object | None = None) -> None:
        self._memory = memory
        self._memory_manager: object | None = None

    def set_memory(self, memory: object) -> None:
        """Set the memory instance (called during NIA initialization)."""
        self._memory = memory

    def set_memory_manager(self, manager: object) -> None:
        """Set the MemoryManager instance (called during NIA initialization).

        P1 fix: wires the MemoryManager into the tool so built-in memory
        writes can mirror to external providers via
        ``manager.on_memory_write(action, target, content)``.
        """
        self._memory_manager = manager

    def _notify_memory_write(
        self,
        action: str,
        target: str,
        content: str,
    ) -> None:
        """Best-effort: notify the MemoryManager of a built-in write.

        P1 fix: implements notify_memory_tool_write so external memory
        providers (Honcho, mem0, etc.) can mirror writes from the
        nia_memory tool.
        """
        if self._memory_manager is None:
            return
        try:
            self._memory_manager.on_memory_write(action, target, content)
        except Exception:
            pass  # Best-effort — never break the tool call.

    async def execute(self, arguments: NiaMemoryInput, context: ToolExecutionContext) -> ToolResult:
        if self._memory is None:
            return ToolResult(output="Memory system not initialized", is_error=True)

        action = arguments.action

        if action == "search":
            if not arguments.query:
                return ToolResult(output="query is required for search", is_error=True)
            results = self._memory.search_relevant(arguments.query, limit=arguments.limit)
            if not results:
                return ToolResult(output="No relevant memories found")
            lines = []
            for entry in results:
                lines.append(f"[{entry.category}] {entry.content}")
            return ToolResult(output="\n".join(lines))

        elif action == "add_fact":
            if not arguments.fact:
                return ToolResult(output="fact is required for add_fact", is_error=True)
            self._memory.add_fact(arguments.fact)
            # P1 fix: mirror to external providers via the MemoryManager.
            self._notify_memory_write("add", "fact", arguments.fact)
            return ToolResult(output=f"Stored fact: {arguments.fact}")

        elif action == "add_preference":
            if not arguments.key or not arguments.value:
                return ToolResult(output="key and value are required for add_preference", is_error=True)
            self._memory.add_preference(arguments.key, arguments.value)
            # P1 fix: mirror to external providers via the MemoryManager.
            pref_content = f"{arguments.key}: {arguments.value}"
            self._notify_memory_write("add", "preference", pref_content)
            return ToolResult(output=f"Stored preference: {arguments.key} = {arguments.value}")

        elif action == "list_preferences":
            prefs = self._memory.get_preferences()
            if not prefs:
                return ToolResult(output="No preferences stored")
            lines = [f"{k}: {v}" for k, v in prefs.items()]
            return ToolResult(output="\n".join(lines))

        elif action == "recent":
            recent = self._memory.get_recent_conversation(limit=arguments.limit)
            if not recent:
                return ToolResult(output="No recent conversation")
            lines = []
            for entry in recent:
                role = entry.metadata.get("role", "unknown")
                lines.append(f"[{role}] {entry.content[:200]}")
            return ToolResult(output="\n".join(lines))

        elif action == "stats":
            stats = self._memory.get_stats()
            summary = self._memory.get_context_summary()
            return ToolResult(output=f"Stats: {stats}\nSummary: {summary}")

        else:
            return ToolResult(
                output=f"Unknown action: {action}. Use: search, add_fact, add_preference, list_preferences, recent, stats",
                is_error=True,
            )

    def is_read_only(self, arguments: NiaMemoryInput) -> bool:
        return arguments.action in ("search", "list_preferences", "recent", "stats")
