"""Service layer for niaharness.

Aggregates stateful helpers used across the harness:
- compact:        token estimation + auto-compaction of conversation history
- session_storage: session snapshot persistence
- cron:           cron job registry (CRUD + validation)
- cron_scheduler: background scheduler daemon
- lsp:            Python source code intelligence (no real LSP server required)

The package was originally advertised by the refactor in commit 6420439 but
the files were never committed. This module reconstructs the contracts that
the rest of the codebase (and the test suite) already depend on.
"""

from __future__ import annotations

from niaharness.services.compact import (
    AutoCompactState,
    auto_compact_if_needed,
    compact_messages,
    estimate_conversation_tokens,
    estimate_message_tokens,
    estimate_tokens,
    summarize_messages,
)
from niaharness.services.session_storage import (
    export_session_markdown,
    get_project_session_dir,
    list_session_snapshots,
    load_session_by_id,
    load_session_snapshot,
    save_session_snapshot,
)

__all__ = [
    # compact
    "AutoCompactState",
    "auto_compact_if_needed",
    "compact_messages",
    "estimate_conversation_tokens",
    "estimate_message_tokens",
    "estimate_tokens",
    "summarize_messages",
    # session_storage
    "export_session_markdown",
    "get_project_session_dir",
    "list_session_snapshots",
    "load_session_by_id",
    "load_session_snapshot",
    "save_session_snapshot",
]
