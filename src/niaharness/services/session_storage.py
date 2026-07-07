"""Session snapshot persistence.

Stores per-project conversation snapshots on disk so that the CLI's
``--continue`` and ``--resume <id>`` flows can restore a previous session.

Layout::

    <data_dir>/sessions/<project_hash>/<session_id>.json
    <data_dir>/sessions/<project_hash>/<session_id>.md   (export only)

The ``project_hash`` is a SHA-256 of the resolved absolute cwd, so the same
project always maps to the same session directory regardless of how the user
invokes the CLI (symlinks, relative paths, etc).
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from niaharness.api.usage import UsageSnapshot
from niaharness.engine.messages import ConversationMessage

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _get_data_dir() -> Path:
    """Return the data dir, honoring legacy ``OPENHARNESS_DATA_DIR`` env var.

    Imported lazily so test monkeypatching of ``niaharness.config.paths`` is
    always honoured.
    """
    # Legacy alias takes precedence for backward compat with tests/scripts
    # that still set OPENHARNESS_DATA_DIR.
    legacy = os.environ.get("OPENHARNESS_DATA_DIR")
    if legacy:
        return Path(legacy)
    from niaharness.config.paths import get_data_dir

    return get_data_dir()


def get_project_session_dir(cwd: str | Path) -> Path:
    """Return ``<data_dir>/sessions/<project_hash>/`` (creating it if needed)."""
    resolved = Path(cwd).resolve()
    project_hash = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:16]
    sessions_root = _get_data_dir() / "sessions" / project_hash
    sessions_root.mkdir(parents=True, exist_ok=True)
    return sessions_root


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def _serialise_messages(messages: list[ConversationMessage]) -> list[dict[str, Any]]:
    return [m.model_dump(mode="json") for m in messages]


def _serialise_usage(usage: UsageSnapshot | None) -> dict[str, Any]:
    if usage is None:
        return {"input_tokens": 0, "output_tokens": 0}
    return usage.model_dump(mode="json")


def _build_summary(messages: list[ConversationMessage]) -> str:
    """Return a short human-readable summary for list displays."""
    if not messages:
        return "(empty session)"
    first_user = next((m for m in messages if m.role == "user"), None)
    if first_user is not None and first_user.text:
        text = first_user.text.strip().replace("\n", " ")
        return text[:80] + ("..." if len(text) > 80 else "")
    return f"{len(messages)} messages"


def _new_session_id() -> str:
    """Return a timestamped session id."""
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + os.urandom(4).hex()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def save_session_snapshot(
    *,
    cwd: str | Path,
    model: str,
    system_prompt: str,
    messages: list[ConversationMessage],
    usage: UsageSnapshot | None = None,
    session_id: str | None = None,
    summary: str | None = None,
) -> Path:
    """Persist a session snapshot to disk.

    Returns the path to the written JSON file.  If ``session_id`` is omitted a
    timestamped id is generated.  Re-saving with the same ``session_id``
    overwrites the previous snapshot.
    """
    session_id = session_id or _new_session_id()
    session_dir = get_project_session_dir(cwd)
    path = session_dir / f"{session_id}.json"

    payload = {
        "session_id": session_id,
        "cwd": str(Path(cwd).resolve()),
        "model": model,
        "system_prompt": system_prompt,
        "messages": _serialise_messages(messages),
        "usage": _serialise_usage(usage),
        "summary": summary or _build_summary(messages),
        "message_count": len(messages),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    # Best-effort: update the FTS5 search index.  Never fails the save.
    try:
        from niaharness.services.session_search import index_session_on_save

        index_session_on_save(payload)
    except Exception:
        pass

    return path


def load_session_snapshot(cwd: str | Path) -> dict[str, Any] | None:
    """Return the most recently created snapshot for ``cwd``, or ``None``.

    Returns the full payload (including ``messages`` and ``usage``), not just
    the lightweight metadata summary used by :func:`list_session_snapshots`.
    """
    session_dir = get_project_session_dir(cwd)
    candidates = sorted(
        session_dir.glob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
    return None


def load_session_by_id(cwd: str | Path, session_id: str) -> dict[str, Any] | None:
    """Return the snapshot with ``session_id`` for ``cwd``, or ``None``."""
    session_dir = get_project_session_dir(cwd)
    path = session_dir / f"{session_id}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def list_session_snapshots(
    cwd: str | Path,
    *,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Return all snapshots for ``cwd`` sorted by creation time, newest first.

    Only lightweight metadata is returned (session_id, summary, message_count,
    model, created_at).  Use :func:`load_session_by_id` to fetch the full
    payload.
    """
    session_dir = get_project_session_dir(cwd)
    out: list[dict[str, Any]] = []
    for entry in session_dir.glob("*.json"):
        try:
            data = json.loads(entry.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        out.append(
            {
                "session_id": data.get("session_id", entry.stem),
                "summary": data.get("summary", "(no summary)"),
                "message_count": data.get("message_count", 0),
                "model": data.get("model", "unknown"),
                "created_at": data.get("created_at", ""),
                "path": str(entry),
            }
        )
    # Sort by created_at descending; entries without a timestamp sink to the bottom.
    out.sort(key=lambda s: s.get("created_at") or "", reverse=True)
    if limit > 0:
        out = out[:limit]
    return out


def export_session_markdown(
    *,
    cwd: str | Path,
    messages: list[ConversationMessage],
    session_id: str | None = None,
    usage: UsageSnapshot | None = None,
    model: str | None = None,
) -> Path:
    """Write a human-readable markdown transcript of ``messages``.

    The output file lives alongside the JSON snapshots in the project's
    session dir, named ``<session_id>.md``.  Returns the path written.
    """
    session_id = session_id or _new_session_id()
    session_dir = get_project_session_dir(cwd)
    path = session_dir / f"{session_id}.md"

    lines: list[str] = [
        "# OpenHarness Session Transcript",
        "",
        f"- Session ID: `{session_id}`",
    ]
    if model:
        lines.append(f"- Model: `{model}`")
    if usage is not None:
        lines.append(
            f"- Tokens: input={usage.input_tokens}, output={usage.output_tokens}"
        )
    lines.append(f"- Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append("")
    lines.append("---")
    lines.append("")

    for msg in messages:
        role_label = "User" if msg.role == "user" else "Assistant"
        lines.append(f"## {role_label}")
        lines.append("")
        for block in msg.content:
            cls = block.__class__.__name__
            if cls == "TextBlock":
                lines.append(block.text)
                lines.append("")
            elif cls == "ToolUseBlock":
                lines.append(f"```tool_use {block.name}")
                lines.append(json.dumps(block.input, indent=2, default=str))
                lines.append("```")
                lines.append("")
            elif cls == "ToolResultBlock":
                lines.append("```tool_result")
                lines.append(block.content)
                lines.append("```")
                lines.append("")
        lines.append("---")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path
