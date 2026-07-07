"""Memory batched-ops tool — manage NIA's memory with batched operations.

Adapted from Hermes Agent's tools/memory_tool.py.

Replaces the single-operation ``nia_memory`` tool with a batched interface
that accepts an ``operations`` array. Each operation is a dict with an
``action`` field (add, update, remove, search, list) and action-specific
parameters.

This matches Hermes's memory tool architecture: the agent can batch
multiple memory writes in a single tool call, reducing API round-trips.

The old ``nia_memory`` tool remains registered for backward compatibility.

Reference: Hermes Agent's tools/memory_tool.py (batched operations array).
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from pydantic import BaseModel, Field

from niaharness.tools.base import BaseTool, ToolExecutionContext, ToolResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class MemoryOperation(BaseModel):
    """A single memory operation in a batch."""

    action: Literal["add", "update", "remove", "search", "list", "get"] = Field(
        description="Memory operation type."
    )
    category: str | None = Field(
        default=None,
        description="Memory category (for 'add'): preference, fact, pattern.",
    )
    content: str | None = Field(
        default=None,
        description="Memory content (for 'add' and 'update').",
    )
    key: str | None = Field(
        default=None,
        description="Preference key (for 'add'/'update' with category='preference'). Required for 'remove'.",
    )
    query: str | None = Field(
        default=None,
        description="Search query (for 'search' action).",
    )
    limit: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Max results (for 'search' and 'list').",
    )


class MemoryToolInput(BaseModel):
    """Arguments for the memory tool.

    Supports two modes:
    - Batch: pass ``operations`` array for multiple writes in one call.
    - Single: pass ``action`` directly for a single operation (convenience).
    """

    operations: list[MemoryOperation] | None = Field(
        default=None,
        description="Array of memory operations to execute in batch.",
    )
    # Single-operation convenience fields (used when operations is None).
    action: Literal["add", "update", "remove", "search", "list", "get"] | None = Field(
        default=None,
        description="Single operation (convenience — use 'operations' for batch).",
    )
    category: str | None = Field(default=None)
    content: str | None = Field(default=None)
    key: str | None = Field(default=None)
    query: str | None = Field(default=None)
    limit: int = Field(default=5, ge=1, le=20)


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


class MemoryBatchedTool(BaseTool):
    """Manage NIA's memory with batched operations."""

    name = "memory"
    description = (
        "Manage NIA's persistent memory: add, update, remove, search, list, "
        "or get memory entries. Supports batched operations (pass an "
        "'operations' array for multiple writes in one call). Memory persists "
        "across sessions and is injected into the system prompt."
    )
    input_model = MemoryToolInput

    def is_read_only(self, arguments: MemoryToolInput) -> bool:
        # Check if all operations are read-only.
        ops = arguments.operations or []
        if not ops and arguments.action:
            return arguments.action in ("search", "list", "get")
        return all(op.action in ("search", "list", "get") for op in ops)

    async def execute(self, arguments: MemoryToolInput, context: ToolExecutionContext) -> ToolResult:
        # Get the memory instance from context metadata.
        memory = context.metadata.get("memory")
        if memory is None:
            return ToolResult(output="Memory system not available in this context.", is_error=True)

        # Build operations list.
        if arguments.operations:
            ops = arguments.operations
        elif arguments.action:
            ops = [MemoryOperation(
                action=arguments.action,
                category=arguments.category,
                content=arguments.content,
                key=arguments.key,
                query=arguments.query,
                limit=arguments.limit,
            )]
        else:
            return ToolResult(output="Provide 'operations' array or 'action' field.", is_error=True)

        # Execute operations.
        results: list[dict[str, Any]] = []
        for op in ops:
            result = self._execute_one(op, memory)
            results.append(result)

        # Persist if any writes occurred.
        has_writes = any(r.get("action") in ("add", "update", "remove") for r in results)
        if has_writes and hasattr(memory, "save"):
            try:
                memory.save()
            except Exception as exc:
                logger.warning("Failed to persist memory: %s", exc)

        # Format output.
        if len(results) == 1:
            return ToolResult(output=results[0].get("message", "Done"), metadata={"results": results})
        lines = [f"Batch memory: {len(results)} operation(s)"]
        for i, r in enumerate(results, 1):
            lines.append(f"  {i}. {r.get('action', '?')}: {r.get('message', 'ok')}")
        return ToolResult(output="\n".join(lines), metadata={"results": results})

    # ---- single operation ----------------------------------------------

    def _execute_one(self, op: MemoryOperation, memory: Any) -> dict[str, Any]:
        """Execute a single memory operation."""
        try:
            if op.action == "add":
                return self._add(op, memory)
            if op.action == "update":
                return self._update(op, memory)
            if op.action == "remove":
                return self._remove(op, memory)
            if op.action == "search":
                return self._search(op, memory)
            if op.action == "list":
                return self._list(op, memory)
            if op.action == "get":
                return self._get(op, memory)
            return {"action": op.action, "message": f"Unknown action: {op.action}"}
        except Exception as exc:
            return {"action": op.action, "message": f"Error: {exc}", "error": True}

    def _add(self, op: MemoryOperation, memory: Any) -> dict[str, Any]:
        if not op.content:
            return {"action": "add", "message": "add requires 'content'"}
        category = op.category or "fact"
        if category == "preference" and op.key:
            memory.add_preference(op.key, op.content)
            return {"action": "add", "category": category, "key": op.key, "message": f"Saved preference: {op.key}"}
        if category == "pattern":
            memory.add_pattern(op.content)
            return {"action": "add", "category": category, "message": "Saved pattern"}
        memory.add_fact(op.content)
        return {"action": "add", "category": category, "message": "Saved fact"}

    def _update(self, op: MemoryOperation, memory: Any) -> dict[str, Any]:
        if not op.content:
            return {"action": "update", "message": "update requires 'content'"}
        category = op.category or "fact"
        if category == "preference" and op.key:
            memory.add_preference(op.key, op.content)  # add_preference replaces existing key
            return {"action": "update", "category": category, "key": op.key, "message": f"Updated preference: {op.key}"}
        return {"action": "update", "message": "update only supported for preferences (with key)"}

    def _remove(self, op: MemoryOperation, memory: Any) -> dict[str, Any]:
        if not op.key:
            return {"action": "remove", "message": "remove requires 'key'"}
        # NIA's Memory class doesn't have a direct remove method —
        # filter out the preference with matching key.
        if hasattr(memory, "_long_term"):
            memory._long_term = [
                m for m in memory._long_term
                if not (m.category == "preference" and m.metadata.get("key") == op.key)
            ]
            return {"action": "remove", "key": op.key, "message": f"Removed preference: {op.key}"}
        return {"action": "remove", "message": "Memory backend doesn't support removal"}

    def _search(self, op: MemoryOperation, memory: Any) -> dict[str, Any]:
        if not op.query:
            return {"action": "search", "message": "search requires 'query'"}
        results = memory.search_relevant(op.query, limit=op.limit)
        if not results:
            return {"action": "search", "message": "No matching memories found."}
        lines = [f"Found {len(results)} matching memor{'y' if len(results) == 1 else 'ies'}:"]
        for m in results:
            lines.append(f"  [{m.category}] {m.content}")
        return {"action": "search", "message": "\n".join(lines), "count": len(results)}

    def _list(self, op: MemoryOperation, memory: Any) -> dict[str, Any]:
        stats = memory.get_stats() if hasattr(memory, "get_stats") else {}
        prefs = memory.get_preferences() if hasattr(memory, "get_preferences") else {}
        lines = [f"Memory stats: {stats}"]
        if prefs:
            lines.append("Preferences:")
            for k, v in prefs.items():
                lines.append(f"  {k}: {v}")
        return {"action": "list", "message": "\n".join(lines), "stats": stats}

    def _get(self, op: MemoryOperation, memory: Any) -> dict[str, Any]:
        if not op.key:
            return {"action": "get", "message": "get requires 'key'"}
        prefs = memory.get_preferences() if hasattr(memory, "get_preferences") else {}
        value = prefs.get(op.key)
        if value is None:
            return {"action": "get", "key": op.key, "message": f"Preference not found: {op.key}"}
        return {"action": "get", "key": op.key, "message": f"{op.key}: {value}", "value": value}
