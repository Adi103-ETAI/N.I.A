"""Session search tool — FTS5-backed search over past conversations.

The audit (P0 Task 3) flagged that NIA persists sessions but has no search.
This tool fills that gap, letting the agent recall past conversations
across all projects.

Three calling modes (inferred from args, no explicit mode parameter):

1. **DISCOVERY** — pass ``query``. Runs FTS5 over all indexed messages,
   returns top N matching sessions each with a snippet of the matching
   text and metadata. Zero LLM cost.

2. **SCROLL** — pass ``session_id`` + ``around_message_idx``. Returns a
   window of ±N messages centered on the anchor, no FTS5. To scroll
   forward/backward, re-anchor on the last/first message idx of the
   returned window.

3. **BROWSE** — no args (or ``action="browse"``). Returns recent sessions
   chronologically (id, summary, timestamp, message count).

The index lives at ``<data_dir>/sessions.sqlite`` and is maintained
incrementally by :mod:`niaharness.services.session_storage` on every
:func:`save_session_snapshot` call.  Use the ``rebuild`` action to
backfill the index from existing on-disk session JSON files (useful
the first time the search feature is enabled).

Reference: Hermes Agent's ``tools/session_search_tool.py`` (DISCOVERY /
SCROLL / BROWSE modes).  NIA's version is simpler — no session lineage
dedup, no source tagging — but mirrors the three-mode calling shape.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from niaharness.services.session_search import SessionSearchIndex, get_search_index
from niaharness.tools.base import BaseTool, ToolExecutionContext, ToolResult


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class SessionSearchToolInput(BaseModel):
    """Arguments for the session_search tool.

    The calling mode is inferred from which fields are set:

    - ``query`` set → DISCOVERY (search past sessions)
    - ``session_id`` + ``around_message_idx`` set → SCROLL (read a window)
    - neither → BROWSE (list recent sessions)
    - ``action="rebuild"`` → rebuild the FTS5 index from disk (admin op)
    """

    action: Literal["search", "scroll", "browse", "rebuild", "stats"] = Field(
        default="search",
        description=(
            "Explicit action override. If omitted, the action is inferred: "
            "query→search, session_id+around_message_idx→scroll, neither→browse."
        ),
    )
    query: str | None = Field(
        default=None,
        description="Search query (DISCOVERY mode). Searches all indexed message text.",
    )
    session_id: str | None = Field(
        default=None,
        description="Session ID (SCROLL mode). Use with around_message_idx to read a window.",
    )
    around_message_idx: int | None = Field(
        default=None,
        ge=0,
        description="Message index to center the window on (SCROLL mode).",
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Max results for search/browse modes.",
    )
    window: int = Field(
        default=5,
        ge=0,
        le=50,
        description="Half-window size for scroll mode (±N messages around the anchor).",
    )
    project_hash: str | None = Field(
        default=None,
        description="Optional: restrict search/browse to a specific project (16-char hash).",
    )


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


class SessionSearchTool(BaseTool):
    """Search past conversations via an FTS5 index."""

    name = "session_search"
    description = (
        "Search past conversations by content, scroll inside a session, or "
        "browse recent sessions. Uses an FTS5 index that updates automatically "
        "as sessions are saved. Use this to recall what was discussed in a "
        "previous session, find when a topic was mentioned, or retrieve a "
        "specific past conversation."
    )
    input_model = SessionSearchToolInput

    def is_read_only(self, arguments: SessionSearchToolInput) -> bool:
        # 'rebuild' modifies the index but not user data; treat as read-only
        # so the permission system doesn't gate it.
        return True

    async def execute(self, arguments: SessionSearchToolInput, context: ToolExecutionContext) -> ToolResult:
        # Infer action if not explicitly set.
        action = arguments.action
        if action == "search" and not arguments.query and not arguments.session_id:
            action = "browse"

        if action == "search" and arguments.query:
            return self._search(arguments)
        if action == "scroll" or (action == "search" and arguments.session_id):
            if not arguments.session_id:
                return ToolResult(output="scroll requires session_id", is_error=True)
            idx = arguments.around_message_idx or 0
            return self._scroll(arguments.session_id, idx, arguments.window)
        if action == "browse":
            return self._browse(arguments)
        if action == "rebuild":
            return self._rebuild()
        if action == "stats":
            return self._stats()

        # Fallback: infer from args.
        if arguments.query:
            return self._search(arguments)
        if arguments.session_id:
            idx = arguments.around_message_idx or 0
            return self._scroll(arguments.session_id, idx, arguments.window)
        return self._browse(arguments)

    # ---- actions -------------------------------------------------------

    def _search(self, arguments: SessionSearchToolInput) -> ToolResult:
        index = get_search_index()
        results = index.search(
            arguments.query or "",
            limit=arguments.limit,
            project_hash=arguments.project_hash,
        )
        if not results:
            return ToolResult(output=f"No sessions found matching: {arguments.query!r}")

        lines = [f"Found {len(results)} session(s) matching {arguments.query!r}:", ""]
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. [{r['session_id']}] {r['summary'][:80]}")
            lines.append(f"   cwd: {r['cwd']}")
            lines.append(f"   model: {r['model']} · {r['message_count']} msgs · {r['created_at'][:19]}")
            lines.append(f"   match at msg #{r['match_message_idx']} ({r['match_role']}):")
            lines.append(f"     {r['snippet']}")
            lines.append(
                f"   → scroll: session_search(session_id={r['session_id']!r}, around_message_idx={r['match_message_idx']})"
            )
            lines.append("")
        return ToolResult(output="\n".join(lines), metadata={"results": results})

    def _scroll(self, session_id: str, around_idx: int, window: int) -> ToolResult:
        index = get_search_index()
        result = index.get_messages_around(session_id, around_idx, window=window)
        if result is None:
            return ToolResult(output=f"Session not found in index: {session_id}", is_error=True)

        lines = [
            f"Session: {result['session_id']}",
            f"Summary: {result['summary'][:80]}",
            f"cwd: {result['cwd']}",
            f"model: {result['model']} · {result['message_count']} msgs · {result['created_at'][:19]}",
            f"Anchor: message #{around_idx} (window ±{window})",
            "",
        ]
        for m in result["messages"]:
            prefix = ">>>" if m["idx"] == around_idx else "   "
            role_tag = "U" if m["role"] == "user" else "A"
            text_preview = m["text"][:200].replace("\n", " ")
            if len(m["text"]) > 200:
                text_preview += "..."
            lines.append(f"{prefix} [{m['idx']:>3}] {role_tag}: {text_preview}")

        # Hint for further scrolling.
        if result["messages"]:
            first_idx = result["messages"][0]["idx"]
            last_idx = result["messages"][-1]["idx"]
            lines.append("")
            lines.append(
                f"Scroll earlier: session_search(session_id={session_id!r}, around_message_idx={max(0, first_idx - 1)})"
            )
            lines.append(
                f"Scroll later:   session_search(session_id={session_id!r}, around_message_idx={last_idx + 1})"
            )
        return ToolResult(output="\n".join(lines), metadata=result)

    def _browse(self, arguments: SessionSearchToolInput) -> ToolResult:
        index = get_search_index()
        sessions = index.list_recent(
            limit=arguments.limit, project_hash=arguments.project_hash
        )
        if not sessions:
            return ToolResult(
                output=(
                    "No sessions in the index. Sessions are indexed automatically "
                    "when saved. If you have existing session files, run "
                    "session_search(action='rebuild') to backfill the index."
                )
            )

        lines = [f"Recent sessions ({len(sessions)} shown):", ""]
        for i, s in enumerate(sessions, 1):
            lines.append(f"{i}. [{s['session_id']}] {s['summary'][:80]}")
            lines.append(f"   cwd: {s['cwd']}")
            lines.append(f"   model: {s['model']} · {s['message_count']} msgs · {s['created_at'][:19]}")
            lines.append("")
        return ToolResult(output="\n".join(lines), metadata={"sessions": sessions})

    def _rebuild(self) -> ToolResult:
        index = get_search_index()
        count = index.rebuild_from_sessions_dir()
        stats = index.stats()
        return ToolResult(
            output=(
                f"Rebuilt search index from {count} session file(s).\n"
                f"Index now contains: {stats['sessions']} sessions, "
                f"{stats['messages']} messages."
            ),
            metadata={"rebuilt_count": count, **stats},
        )

    def _stats(self) -> ToolResult:
        index = get_search_index()
        stats = index.stats()
        return ToolResult(
            output=(
                f"Session search index stats:\n"
                f"  Sessions indexed: {stats['sessions']}\n"
                f"  Messages indexed: {stats['messages']}\n"
                f"  Database: {index._db_path}"
            ),
            metadata=stats,
        )
