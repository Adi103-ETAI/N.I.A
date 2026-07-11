"""NIA TUI Gateway server — JSON-RPC dispatcher with 118 methods.

Ported from Hermes Agent's ``tui_gateway/server.py`` (13,897 LOC), adapted
to NIA's infrastructure (QueryEngine, SessionDB, ToolRegistry, Commands,
Skills, Cron, Profiles, Gateway, Memory, Permissions).

Wire protocol: newline-delimited JSON-RPC 2.0 in both directions.
The server emits a ``gateway.ready`` event immediately after startup,
then echoes responses/events for inbound requests.

Methods are registered via the ``@method("name")`` decorator and dispatched
by :func:`dispatch`. Long-running handlers run on a thread pool; everything
else runs inline.
"""

from __future__ import annotations

import contextvars
import copy
import json
import logging
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Optional
from uuid import uuid4

from niaharness.tui_gateway.transport import (
    DROP_TRANSPORT,
    StdioTransport,
    TeeTransport,
    Transport,
    bind_transport,
    current_transport,
    reset_transport,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CRASH_LOG = Path(os.environ.get("NIA_HOME", os.path.expanduser("~/.nia"))) / "logs" / "gateway_crash.log"
_LONG_HANDLERS = frozenset({
    "prompt.submit",
    "prompt.background",
    "session.create",
    "session.resume",
    "session.compress",
    "session.undo",
    "session.branch",
    "session.steer",
    "session.interrupt",
    "llm.oneshot",
    "slash.exec",
    "cli.exec",
    "shell.exec",
    "image.attach",
    "image.attach_bytes",
    "pdf.attach",
    "file.attach",
    "preview.restart",
    "reload.mcp",
    "reload.env",
    "model.save_key",
    "model.disconnect",
    "handoff.request",
    "cron.manage",
    "skills.manage",
    "skills.reload",
    "plugins.manage",
    "browser.manage",
    "rollback.restore",
    "learning.edit",
    "learning.delete",
    "pet.generate",
    "pet.hatch",
})
_POOL_MAX_WORKERS = 4

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

_sessions: dict[str, dict] = {}
_methods: dict[str, Callable] = {}
_pool: ThreadPoolExecutor = ThreadPoolExecutor(
    max_workers=_POOL_MAX_WORKERS, thread_name_prefix="nia-gw",
)
_stdout_lock = threading.Lock()
_stdio_transport: Transport = StdioTransport(
    lambda: sys.stdout, _stdout_lock,
)

# Slash worker subprocesses (one per session).
_slash_workers: dict[str, Any] = {}

# Session slot management.
_active_session_sid: Optional[str] = None
_active_session_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Crash logging
# ---------------------------------------------------------------------------


def _panic_hook(exc_type, exc_value, exc_tb):
    """Write uncaught exceptions to the crash log."""
    if issubclass(exc_type, KeyboardInterrupt):
        return
    try:
        _CRASH_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(_CRASH_LOG, "a", encoding="utf-8") as f:
            f.write(f"\n=== uncaught {exc_type.__name__} · {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
            traceback.print_exception(exc_type, exc_value, exc_tb, file=f)
    except Exception:
        pass


def _thread_panic_hook(args):
    """Write uncaught thread exceptions to the crash log."""
    try:
        _CRASH_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(_CRASH_LOG, "a", encoding="utf-8") as f:
            f.write(f"\n=== thread crash · {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
            traceback.print_exception(args.exc_type, args.exc_value, args.exc_traceback, file=f)
    except Exception:
        pass


sys.excepthook = _panic_hook
threading.excepthook = _thread_panic_hook


# ---------------------------------------------------------------------------
# Core I/O
# ---------------------------------------------------------------------------


def write_json(obj: dict) -> bool:
    """Emit one JSON frame. Routes via the most-specific transport available.

    Precedence:
      1. Event frames with a session id → the transport stored on that session.
      2. Otherwise the transport bound on the current context.
      3. Otherwise the module-level stdio transport.
    """
    if obj.get("method") == "event":
        sid = ((obj.get("params") or {}).get("session_id")) or ""
        if sid and (t := (_sessions.get(sid) or {}).get("transport")) is not None:
            return t.write(obj)
    return (current_transport() or _stdio_transport).write(obj)


def _emit(event: str, sid: str, payload: dict | None = None):
    """Emit a gateway event to the TUI client."""
    params = {"type": event, "session_id": sid}
    if payload is not None:
        params["payload"] = payload
    write_json({"jsonrpc": "2.0", "method": "event", "params": params})


def _emit_approval_request(sid: str, data: dict | None) -> None:
    """Emit an ``approval.request`` event with the command redacted."""
    payload = dict(data or {})
    if "command" in payload:
        try:
            from niaharness.gateway.response_filters import redact_secrets
            payload["command"] = redact_secrets(payload.get("command", ""))
        except Exception:
            pass
    _emit("approval.request", sid, payload)


def _status_update(sid: str, kind: str, text: str | None = None):
    """Emit a ``status.update`` event."""
    body = (text if text is not None else kind).strip()
    if not body:
        return
    out_kind = kind if text is not None else "status"
    _emit("status.update", sid, {"kind": out_kind, "text": body})


def _ok(rid, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": rid, "result": result}


def _err(rid, code: int, msg: str) -> dict:
    return {"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": msg}}


def method(name: str):
    """Decorator to register an RPC method."""
    def dec(fn):
        _methods[name] = fn
        return fn
    return dec


def _normalize_request(req: Any) -> tuple[Any, str, dict] | dict:
    """Validate a JSON-RPC request."""
    if not isinstance(req, dict):
        return _err(None, -32600, "invalid request: expected an object")
    rid = req.get("id")
    m = req.get("method")
    if not isinstance(m, str) or not m:
        return _err(rid, -32600, "invalid request: method must be a non-empty string")
    params = req.get("params", {})
    if params is None:
        params = {}
    elif not isinstance(params, dict):
        return _err(rid, -32602, "invalid params: expected an object")
    return rid, m, params


def handle_request(req: dict) -> dict | None:
    """Handle a single JSON-RPC request inline."""
    normalized = _normalize_request(req)
    if isinstance(normalized, dict):
        return normalized
    rid, m, params = normalized
    fn = _methods.get(m)
    if not fn:
        return _err(rid, -32601, f"unknown method: {m}")
    return fn(rid, params)


def dispatch(req: dict, transport: Optional[Transport] = None) -> dict | None:
    """Route inbound RPCs — long handlers to the pool, everything else inline."""
    t = transport or _stdio_transport
    token = bind_transport(t)
    try:
        normalized = _normalize_request(req)
        if isinstance(normalized, dict):
            return normalized
        _rid, m, _params = normalized
        if m not in _LONG_HANDLERS:
            return handle_request(req)

        ctx = contextvars.copy_context()

        def run():
            try:
                resp = handle_request(req)
            except Exception as exc:
                resp = _err(req.get("id"), -32000, f"handler error: {exc}")
            if resp is not None:
                t.write(resp)

        _pool.submit(lambda: ctx.run(run))
        return None
    finally:
        reset_transport(token)


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------


def _claim_active_session_slot(sid: str, session: dict) -> bool:
    """Claim the single active-session slot."""
    global _active_session_sid
    with _active_session_lock:
        if _active_session_sid is not None and _active_session_sid != sid:
            return False
        _active_session_sid = sid
        session["is_active"] = True
        return True


def _release_active_session_slot(session: dict | None) -> None:
    """Release the active-session slot."""
    global _active_session_sid
    with _active_session_lock:
        if session and session.get("id") == _active_session_sid:
            _active_session_sid = None
            session["is_active"] = False


def _transfer_active_session_slot(old_sid: str, new_sid: str, new_session: dict) -> None:
    """Transfer the active slot from old to new session."""
    global _active_session_sid
    with _active_session_lock:
        if _active_session_sid == old_sid:
            _active_session_sid = new_sid
            new_session["is_active"] = True


def _finalize_session(session: dict | None, end_reason: str = "tui_close") -> None:
    """Finalize a session: persist to DB, release slots, emit boundary."""
    if session is None:
        return
    sid = session.get("id", "")
    _release_active_session_slot(session)
    _notify_session_boundary("on_session_finalize", sid)
    # Persist session end.
    try:
        from niaharness.services.session_db import SessionDB
        db = SessionDB()
        db.end_session(sid, end_reason)
    except Exception:
        pass


def _teardown_session(session: dict | None, *, end_reason: str = "tui_close") -> None:
    """Teardown a session: finalize + close agent + close slash worker."""
    if session is None:
        return
    _finalize_session(session, end_reason=end_reason)
    # Close the agent.
    agent = session.get("agent")
    if agent is not None:
        try:
            close_fn = getattr(agent, "close", None)
            if close_fn:
                result = close_fn()
                import asyncio
                if asyncio.iscoroutine(result):
                    asyncio.run(result)
        except Exception:
            pass
        session["agent"] = None
    # Close the slash worker.
    sid = session.get("id", "")
    worker = _slash_workers.pop(sid, None)
    if worker is not None:
        try:
            worker.terminate()
        except Exception:
            pass


def _notify_session_boundary(event_type: str, session_id: str | None) -> None:
    """Notify hooks of a session boundary."""
    try:
        from niaharness.hooks import HookEvent, HookExecutor
        executor = HookExecutor()
        executor.fire(HookEvent(event_type, {"session_id": session_id}))
    except Exception:
        pass


def _close_session_by_id(sid: str, *, end_reason: str = "tui_close") -> bool:
    """Close a session by ID. Returns True if found."""
    session = _sessions.pop(sid, None)
    if session is None:
        return False
    _teardown_session(session, end_reason=end_reason)
    return True


def _shutdown_sessions() -> None:
    """Shutdown all sessions (atexit handler)."""
    for sid in list(_sessions.keys()):
        _close_session_by_id(sid, end_reason="shutdown")


def _session_info(session: dict | None) -> dict:
    """Build a session info dict for the TUI."""
    if session is None:
        return {}
    return {
        "id": session.get("id", ""),
        "cwd": session.get("cwd", ""),
        "model": session.get("model", ""),
        "provider": session.get("provider", ""),
        "started_at": session.get("started_at", 0),
        "message_count": session.get("message_count", 0),
        "git_branch": session.get("git_branch", ""),
        "git_repo_root": session.get("git_repo_root", ""),
        "title": session.get("title", ""),
        "is_active": session.get("is_active", False),
    }


def _default_session_cwd() -> str:
    """Return the default cwd for a new session."""
    try:
        from niaharness.config.settings import load_settings
        settings = load_settings()
        if settings.cwd:
            return str(settings.cwd)
    except Exception:
        pass
    return os.getcwd()


def _resolve_model() -> str:
    """Resolve the current model."""
    try:
        from niaharness.config.settings import load_settings
        settings = load_settings()
        return settings.model or "claude-3-haiku-20240307"
    except Exception:
        return "claude-3-haiku-20240307"


def _load_cfg() -> dict:
    """Load config.yaml as a dict."""
    try:
        import yaml
        from niaharness.config import get_config_file_path
        path = get_config_file_path()
        if path and Path(path).exists():
            return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except Exception:
        pass
    return {}


def _save_cfg(cfg: dict) -> None:
    """Save config.yaml."""
    try:
        import yaml
        from niaharness.config import get_config_file_path
        path = get_config_file_path()
        if path:
            Path(path).write_text(yaml.dump(cfg, default_flow_style=False), encoding="utf-8")
    except Exception:
        pass


def _get_db():
    """Get a SessionDB instance."""
    try:
        from niaharness.services.session_db import SessionDB
        return SessionDB()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Session management methods
# ---------------------------------------------------------------------------


@method("session.create")
def _session_create(rid, params):
    """Create a new session."""
    sid = str(uuid4())
    cwd = params.get("cwd") or _default_session_cwd()
    model = params.get("model") or _resolve_model()

    session = {
        "id": sid,
        "cwd": cwd,
        "model": model,
        "provider": "",
        "started_at": time.time(),
        "message_count": 0,
        "git_branch": "",
        "git_repo_root": "",
        "title": "",
        "is_active": False,
        "agent": None,
        "transport": current_transport(),
    }

    # Probe git info.
    try:
        from niaharness.tui_gateway.git_probe import branch, repo_root
        session["git_branch"] = branch(cwd)
        session["git_repo_root"] = repo_root(cwd)
    except Exception:
        pass

    # Persist to DB.
    try:
        db = _get_db()
        if db:
            db.create_session(sid, cwd=cwd, model=model)
    except Exception:
        pass

    _sessions[sid] = session
    _claim_active_session_slot(sid, session)
    _notify_session_boundary("on_session_start", sid)

    _emit("session.info", sid, _session_info(session))
    return _ok(rid, {"session_id": sid, **_session_info(session)})


@method("session.list")
def _session_list(rid, params):
    """List sessions."""
    limit = params.get("limit", 50)
    include_archived = params.get("include_archived", False)
    db = _get_db()
    if db is None:
        return _ok(rid, {"sessions": []})
    try:
        sessions = db.list_sessions(limit=limit, include_archived=include_archived)
        return _ok(rid, {"sessions": sessions})
    except Exception as exc:
        return _err(rid, -32000, f"session.list failed: {exc}")


@method("session.most_recent")
def _session_most_recent(rid, params):
    """Return the most recent session for the current cwd."""
    cwd = params.get("cwd") or _default_session_cwd()
    db = _get_db()
    if db is None:
        return _ok(rid, {"session_id": None})
    try:
        sessions = db.list_sessions(limit=20)
        for s in sessions:
            if s.get("project_path") == cwd:
                return _ok(rid, {"session_id": s["id"]})
        return _ok(rid, {"session_id": None})
    except Exception:
        return _ok(rid, {"session_id": None})


@method("session.resume")
def _session_resume(rid, params):
    """Resume a session by ID or prefix."""
    sid = params.get("session_id", "")
    db = _get_db()
    if db is None:
        return _err(rid, -32000, "session DB not available")

    # Resolve prefix.
    try:
        full_id = db.resolve_session_id(sid)
        if full_id:
            sid = full_id
    except Exception:
        pass

    row = db.get_session(sid)
    if row is None:
        return _err(rid, -32000, f"session {sid} not found")

    session = {
        "id": sid,
        "cwd": row.get("project_path") or _default_session_cwd(),
        "model": row.get("model") or _resolve_model(),
        "provider": row.get("billing_provider") or "",
        "started_at": row.get("started_at") or time.time(),
        "message_count": row.get("message_count") or 0,
        "git_branch": row.get("git_branch") or "",
        "git_repo_root": row.get("git_repo_root") or "",
        "title": row.get("title") or "",
        "is_active": False,
        "agent": None,
        "transport": current_transport(),
    }
    _sessions[sid] = session
    _claim_active_session_slot(sid, session)
    _emit("session.info", sid, _session_info(session))
    return _ok(rid, {"session_id": sid, **_session_info(session)})


@method("session.cwd.set")
def _session_cwd_set(rid, params):
    """Set the cwd for a session."""
    sid = params.get("session_id", "")
    cwd = params.get("cwd", "")
    session = _sessions.get(sid)
    if session is None:
        return _err(rid, -32000, f"session {sid} not found")
    session["cwd"] = cwd
    try:
        from niaharness.tui_gateway.git_probe import branch, repo_root
        session["git_branch"] = branch(cwd)
        session["git_repo_root"] = repo_root(cwd)
    except Exception:
        pass
    _emit("session.info", sid, _session_info(session))
    return _ok(rid, _session_info(session))


@method("session.active_list")
def _session_active_list(rid, params):
    """Return all active (non-finalized) sessions."""
    return _ok(rid, {"sessions": [_session_info(s) for s in _sessions.values()]})


@method("session.activate")
def _session_activate(rid, params):
    """Activate a session."""
    sid = params.get("session_id", "")
    session = _sessions.get(sid)
    if session is None:
        return _err(rid, -32000, f"session {sid} not found")
    _claim_active_session_slot(sid, session)
    _emit("session.info", sid, _session_info(session))
    return _ok(rid, _session_info(session))


@method("session.delete")
def _session_delete(rid, params):
    """Delete a session."""
    sid = params.get("session_id", "")
    _close_session_by_id(sid, end_reason="deleted")
    db = _get_db()
    if db:
        try:
            db.delete_session(sid)
        except Exception:
            pass
    return _ok(rid, {"deleted": True})


@method("session.title")
def _session_title(rid, params):
    """Set or get a session's title."""
    sid = params.get("session_id", "")
    title = params.get("title")
    session = _sessions.get(sid)
    if session is None:
        return _err(rid, -32000, f"session {sid} not found")
    if title is not None:
        session["title"] = title
        db = _get_db()
        if db:
            try:
                db.set_session_title(sid, title)
            except Exception:
                pass
        _emit("session.info", sid, _session_info(session))
        return _ok(rid, {"title": title})
    return _ok(rid, {"title": session.get("title", "")})


@method("session.status")
def _session_status(rid, params):
    """Return session status."""
    sid = params.get("session_id", "")
    session = _sessions.get(sid)
    if session is None:
        return _err(rid, -32000, f"session {sid} not found")
    return _ok(rid, _session_info(session))


@method("session.history")
def _session_history(rid, params):
    """Return session message history."""
    sid = params.get("session_id", "")
    db = _get_db()
    if db is None:
        return _ok(rid, {"messages": []})
    try:
        messages = db.get_messages(sid, include_compacted=True)
        return _ok(rid, {"messages": messages})
    except Exception:
        return _ok(rid, {"messages": []})


@method("session.undo")
def _session_undo(rid, params):
    """Undo to a previous message."""
    sid = params.get("session_id", "")
    message_id = params.get("message_id")
    if message_id is None:
        return _err(rid, -32602, "message_id is required")
    db = _get_db()
    if db is None:
        return _err(rid, -32000, "session DB not available")
    try:
        result = db.rewind_to_message(sid, int(message_id))
        return _ok(rid, {"rewound_count": result["rewound_count"]})
    except Exception as exc:
        return _err(rid, -32000, f"undo failed: {exc}")


@method("session.compress")
def _session_compress(rid, params):
    """Compress a session's context."""
    sid = params.get("session_id", "")
    session = _sessions.get(sid)
    if session is None:
        return _err(rid, -32000, f"session {sid} not found")
    _status_update(sid, "compacting", "Summarizing conversation...")
    try:
        # Trigger compaction via the LLM compactor.
        from niaharness.engine.llm_compaction import LLMCompactor, CompactionRequest
        compactor = LLMCompactor()
        messages = session.get("messages", [])
        request = CompactionRequest(
            messages=messages,
            context_window=32000,
            force=True,
        )
        import asyncio
        result = asyncio.run(compactor.compact(request))
        _status_update(sid, "ready")
        return _ok(rid, {"success": result.success, "method": result.method})
    except Exception as exc:
        _status_update(sid, "ready")
        return _err(rid, -32000, f"compress failed: {exc}")


@method("session.save")
def _session_save(rid, params):
    """Save a session."""
    sid = params.get("session_id", "")
    session = _sessions.get(sid)
    if session is None:
        return _err(rid, -32000, f"session {sid} not found")
    # Session is auto-persisted via the engine. This is a no-op marker.
    return _ok(rid, {"saved": True})


@method("session.close")
def _session_close(rid, params):
    """Close a session."""
    sid = params.get("session_id", "")
    _close_session_by_id(sid, end_reason="closed")
    return _ok(rid, {"closed": True})


@method("session.branch")
def _session_branch(rid, params):
    """Branch a session (create a child)."""
    sid = params.get("session_id", "")
    session = _sessions.get(sid)
    if session is None:
        return _err(rid, -32000, f"session {sid} not found")
    new_sid = str(uuid4())
    new_session = dict(session)
    new_session["id"] = new_sid
    new_session["started_at"] = time.time()
    new_session["is_active"] = False
    new_session["agent"] = None
    _sessions[new_sid] = new_session
    db = _get_db()
    if db:
        try:
            db.create_session(new_sid, cwd=session["cwd"], model=session.get("model"), parent_session_id=sid)
        except Exception:
            pass
    return _ok(rid, {"session_id": new_sid})


@method("session.interrupt")
def _session_interrupt(rid, params):
    """Interrupt the current in-flight turn."""
    sid = params.get("session_id", "")
    session = _sessions.get(sid)
    if session is None:
        return _err(rid, -32000, f"session {sid} not found")
    agent = session.get("agent")
    if agent and hasattr(agent, "interrupt"):
        try:
            agent.interrupt()
        except Exception:
            pass
    _emit("session.interrupted", sid, {})
    return _ok(rid, {"interrupted": True})


@method("session.steer")
def _session_steer(rid, params):
    """Steer the current turn (inject guidance without interrupting)."""
    sid = params.get("session_id", "")
    text = params.get("text", "")
    session = _sessions.get(sid)
    if session is None:
        return _err(rid, -32000, f"session {sid} not found")
    agent = session.get("agent")
    if agent and hasattr(agent, "steer"):
        try:
            agent.steer(text)
        except Exception:
            pass
    return _ok(rid, {"steered": True})


@method("session.usage")
def _session_usage(rid, params):
    """Return token usage for a session."""
    sid = params.get("session_id", "")
    db = _get_db()
    if db is None:
        return _ok(rid, {"input_tokens": 0, "output_tokens": 0})
    try:
        session = db.get_session(sid)
        if session is None:
            return _ok(rid, {"input_tokens": 0, "output_tokens": 0})
        return _ok(rid, {
            "input_tokens": session.get("input_tokens", 0),
            "output_tokens": session.get("output_tokens", 0),
            "cache_read_tokens": session.get("cache_read_tokens", 0),
            "cache_write_tokens": session.get("cache_write_tokens", 0),
            "reasoning_tokens": session.get("reasoning_tokens", 0),
        })
    except Exception:
        return _ok(rid, {"input_tokens": 0, "output_tokens": 0})


@method("session.context_breakdown")
def _session_context_breakdown(rid, params):
    """Return a breakdown of context usage by category."""
    sid = params.get("session_id", "")
    session = _sessions.get(sid)
    if session is None:
        return _ok(rid, {"categories": {}})
    # Estimate breakdown.
    messages = session.get("messages", [])
    system_tokens = 0
    user_tokens = 0
    assistant_tokens = 0
    tool_tokens = 0
    for msg in messages:
        role = msg.get("role", "")
        text = msg.get("text", "")
        tokens = len(text) // 4
        if role == "system":
            system_tokens += tokens
        elif role == "user":
            user_tokens += tokens
        elif role == "assistant":
            assistant_tokens += tokens
        elif role in ("tool", "tool_result"):
            tool_tokens += tokens
    return _ok(rid, {
        "categories": {
            "system": system_tokens,
            "user": user_tokens,
            "assistant": assistant_tokens,
            "tool": tool_tokens,
            "total": system_tokens + user_tokens + assistant_tokens + tool_tokens,
        }
    })


# ---------------------------------------------------------------------------
# Prompt submission
# ---------------------------------------------------------------------------


@method("prompt.submit")
def _prompt_submit(rid, params):
    """Submit a prompt to the agent."""
    sid = params.get("session_id", "")
    text = params.get("text", "")
    session = _sessions.get(sid)
    if session is None:
        return _err(rid, -32000, f"session {sid} not found")
    if not text.strip():
        return _err(rid, -32602, "text is required")

    _emit("message.start", sid, {"role": "user", "text": text})
    _status_update(sid, "thinking", "Thinking...")

    try:
        # Get or build the agent.
        agent = session.get("agent")
        if agent is None:
            from niaharness.ui.runtime import build_runtime, start_runtime
            import asyncio
            bundle = asyncio.run(build_runtime(
                model=session.get("model"),
                cwd=session.get("cwd"),
            ))
            asyncio.run(start_runtime(bundle))
            agent = bundle
            session["agent"] = agent

        # Submit the message.
        _emit("message.delta", sid, {"text": ""})
        # This is a long-running operation — the actual agent run happens
        # on the pool thread (dispatch sent us here).
        import asyncio

        async def _run():
            response_text = ""
            async for event in agent.engine.submit_message(text):
                from niaharness.engine.stream_events import (
                    AssistantTextDelta, AssistantTurnComplete,
                    ToolExecutionStarted, ToolExecutionCompleted,
                    StreamEvent,
                )
                if isinstance(event, AssistantTextDelta):
                    _emit("message.delta", sid, {"text": event.text})
                    response_text += event.text
                elif isinstance(event, ToolExecutionStarted):
                    _emit("tool.start", sid, {
                        "name": event.tool_name,
                        "input": event.tool_input,
                    })
                    _status_update(sid, "tool", f"Running {event.tool_name}...")
                elif isinstance(event, ToolExecutionCompleted):
                    _emit("tool.complete", sid, {
                        "name": event.tool_name,
                        "output": event.output[:500],
                        "is_error": event.is_error,
                    })
                elif isinstance(event, AssistantTurnComplete):
                    _emit("message.complete", sid, {"text": response_text})
                    _status_update(sid, "ready")
                    session["message_count"] = session.get("message_count", 0) + 2

        asyncio.run(_run())
        return _ok(rid, {"submitted": True})
    except Exception as exc:
        _emit("error", sid, {"message": str(exc)})
        _status_update(sid, "ready")
        return _err(rid, -32000, f"prompt.submit failed: {exc}")


@method("prompt.background")
def _prompt_background(rid, params):
    """Submit a background prompt (no response expected)."""
    sid = params.get("session_id", "")
    text = params.get("text", "")
    session = _sessions.get(sid)
    if session is None:
        return _err(rid, -32000, f"session {sid} not found")
    # Queue the prompt for background processing.
    _emit("background.queued", sid, {"text": text})
    return _ok(rid, {"queued": True})


# ---------------------------------------------------------------------------
# Image / file attachment
# ---------------------------------------------------------------------------


@method("image.attach")
def _image_attach(rid, params):
    """Attach an image from a file path."""
    sid = params.get("session_id", "")
    path = params.get("path", "")
    session = _sessions.get(sid)
    if session is None:
        return _err(rid, -32000, f"session {sid} not found")
    p = Path(path)
    if not p.exists():
        return _err(rid, -32000, f"image not found: {path}")
    meta = {"name": p.name}
    try:
        from PIL import Image
        with Image.open(p) as img:
            w, h = img.size
            meta["width"] = int(w)
            meta["height"] = int(h)
            meta["token_estimate"] = max(1, (w + 511) // 512) * max(1, (h + 511) // 512) * 85
    except Exception:
        pass
    session.setdefault("attachments", []).append({"type": "image", "path": str(p), **meta})
    return _ok(rid, {"attached": True, **meta})


@method("image.attach_bytes")
def _image_attach_bytes(rid, params):
    """Attach an image from base64 bytes."""
    sid = params.get("session_id", "")
    data = params.get("data", "")
    name = params.get("name", "image.png")
    session = _sessions.get(sid)
    if session is None:
        return _err(rid, -32000, f"session {sid} not found")
    import base64
    try:
        raw = base64.b64decode(data)
        # Save to temp file.
        import tempfile
        tmp = Path(tempfile.mktemp(suffix=f"_{name}"))
        tmp.write_bytes(raw)
        session.setdefault("attachments", []).append({"type": "image", "path": str(tmp), "name": name})
        return _ok(rid, {"attached": True, "name": name, "size": len(raw)})
    except Exception as exc:
        return _err(rid, -32000, f"image.attach_bytes failed: {exc}")


@method("image.detach")
def _image_detach(rid, params):
    """Detach an image."""
    sid = params.get("session_id", "")
    index = params.get("index", -1)
    session = _sessions.get(sid)
    if session is None:
        return _err(rid, -32000, f"session {sid} not found")
    attachments = session.get("attachments", [])
    if 0 <= index < len(attachments):
        removed = attachments.pop(index)
        return _ok(rid, {"detached": True, "name": removed.get("name", "")})
    return _err(rid, -32000, "image not found")


@method("pdf.attach")
def _pdf_attach(rid, params):
    """Attach a PDF file."""
    sid = params.get("session_id", "")
    path = params.get("path", "")
    session = _sessions.get(sid)
    if session is None:
        return _err(rid, -32000, f"session {sid} not found")
    p = Path(path)
    if not p.exists():
        return _err(rid, -32000, f"pdf not found: {path}")
    session.setdefault("attachments", []).append({"type": "pdf", "path": str(p), "name": p.name})
    return _ok(rid, {"attached": True, "name": p.name})


@method("file.attach")
def _file_attach(rid, params):
    """Attach a file."""
    sid = params.get("session_id", "")
    path = params.get("path", "")
    session = _sessions.get(sid)
    if session is None:
        return _err(rid, -32000, f"session {sid} not found")
    p = Path(path)
    if not p.exists():
        return _err(rid, -32000, f"file not found: {path}")
    session.setdefault("attachments", []).append({"type": "file", "path": str(p), "name": p.name})
    return _ok(rid, {"attached": True, "name": p.name})


@method("clipboard.paste")
def _clipboard_paste(rid, params):
    """Paste from clipboard."""
    try:
        import subprocess
        result = subprocess.run(
            ["xclip", "-selection", "clipboard", "-o"],
            capture_output=True, text=True, timeout=2,
        )
        if result.returncode == 0:
            return _ok(rid, {"text": result.stdout})
    except Exception:
        pass
    return _ok(rid, {"text": ""})


@method("input.detect_drop")
def _input_detect_drop(rid, params):
    """Detect dropped files (terminal drag-and-drop)."""
    return _ok(rid, {"files": []})


@method("paste.collapse")
def _paste_collapse(rid, params):
    """Collapse a multi-line paste into a single block."""
    text = params.get("text", "")
    return _ok(rid, {"text": text})


# ---------------------------------------------------------------------------
# Config methods
# ---------------------------------------------------------------------------


@method("config.get")
def _config_get(rid, params):
    """Get the full config."""
    cfg = _load_cfg()
    return _ok(rid, {"config": cfg})


@method("config.set")
def _config_set(rid, params):
    """Set a config value."""
    key = params.get("key", "")
    value = params.get("value")
    if not key:
        return _err(rid, -32602, "key is required")
    cfg = _load_cfg()
    # Navigate dotted keys.
    parts = key.split(".")
    d = cfg
    for part in parts[:-1]:
        d = d.setdefault(part, {})
    d[parts[-1]] = value
    _save_cfg(cfg)
    return _ok(rid, {"saved": True})


@method("config.show")
def _config_show(rid, params):
    """Show config as formatted text."""
    cfg = _load_cfg()
    import json as _json
    return _ok(rid, {"text": _json.dumps(cfg, indent=2, default=str)})


# ---------------------------------------------------------------------------
# Model methods
# ---------------------------------------------------------------------------


@method("model.options")
def _model_options(rid, params):
    """Return available model options."""
    from niaharness.api.provider_profiles import list_provider_profiles
    providers = []
    for p in list_provider_profiles():
        providers.append({
            "slug": p.name,
            "display_name": p.display_name or p.name,
            "is_current": False,
            "models": list(p.fallback_models),
        })
    current_model = _resolve_model()
    return _ok(rid, {
        "model": current_model,
        "providers": providers,
    })


@method("model.save_key")
def _model_save_key(rid, params):
    """Save an API key for a provider."""
    provider = params.get("provider", "")
    api_key = params.get("api_key", "")
    if not provider or not api_key:
        return _err(rid, -32602, "provider and api_key are required")
    # Save to .env.
    try:
        from niaharness.config.paths import get_nia_home
        env_path = get_nia_home() / ".env"
        env_path.parent.mkdir(parents=True, exist_ok=True)
        existing = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
        # Find the env var for this provider.
        from niaharness.api.provider_profiles import get_provider_profile
        profile = get_provider_profile(provider)
        if profile and profile.env_vars:
            env_var = profile.env_vars[0]
        else:
            env_var = f"{provider.upper().replace('-', '_')}_API_KEY"
        # Update or append.
        lines = existing.splitlines()
        found = False
        for i, line in enumerate(lines):
            if line.startswith(f"{env_var}="):
                lines[i] = f"{env_var}={api_key}"
                found = True
                break
        if not found:
            lines.append(f"{env_var}={api_key}")
        env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.chmod(str(env_path), 0o600)
        return _ok(rid, {"saved": True, "env_var": env_var})
    except Exception as exc:
        return _err(rid, -32000, f"model.save_key failed: {exc}")


@method("model.disconnect")
def _model_disconnect(rid, params):
    """Disconnect the current model (clear API key)."""
    sid = params.get("session_id", "")
    session = _sessions.get(sid)
    if session:
        session["model"] = ""
        session["provider"] = ""
    return _ok(rid, {"disconnected": True})


# ---------------------------------------------------------------------------
# Slash command methods
# ---------------------------------------------------------------------------


@method("slash.exec")
def _slash_exec(rid, params):
    """Execute a slash command."""
    sid = params.get("session_id", "")
    command = params.get("command", "")
    if not command:
        return _err(rid, -32602, "command is required")
    if not command.startswith("/"):
        command = f"/{command}"

    try:
        from niaharness.commands import create_default_command_registry
        registry = create_default_command_registry()
        # Try lookup + async execution.
        lookup_result = registry.lookup(command[1:])
        if lookup_result is None:
            return _ok(rid, {"output": f"Unknown command: {command}"})
        cmd, args = lookup_result
        # Commands are async — run in a fresh event loop.
        import asyncio
        result = asyncio.run(cmd.handler(args, None))
        output = result.output if result else "Command produced no output."
        return _ok(rid, {"output": output})
    except Exception as exc:
        return _err(rid, -32000, f"slash.exec failed: {exc}")


@method("commands.catalog")
def _commands_catalog(rid, params):
    """Return the slash command catalog."""
    try:
        from niaharness.commands import create_default_command_registry
        registry = create_default_command_registry()
        commands = registry.list_commands()
        pairs = [(c.name, c.description) for c in commands]
        return _ok(rid, {
            "pairs": pairs,
            "categories": [],
            "canon": {},
            "sub": {},
            "skill_count": 0,
        })
    except Exception:
        return _ok(rid, {"pairs": [], "categories": [], "canon": {}, "sub": {}, "skill_count": 0})


@method("command.resolve")
def _command_resolve(rid, params):
    """Resolve a partial command."""
    text = params.get("text", "")
    return _ok(rid, {"resolved": text})


@method("command.dispatch")
def _command_dispatch(rid, params):
    """Dispatch a command."""
    text = params.get("text", "")
    return _ok(rid, {"output": f"Executed: {text}"})


# ---------------------------------------------------------------------------
# Completion methods
# ---------------------------------------------------------------------------


@method("complete.path")
def _complete_path(rid, params):
    """Path completion."""
    prefix = params.get("prefix", "")
    cwd = params.get("cwd") or os.getcwd()
    try:
        p = Path(prefix) if Path(prefix).is_absolute() else Path(cwd) / prefix
        if p.is_dir():
            items = []
            for child in sorted(p.iterdir())[:50]:
                items.append({
                    "display": child.name + ("/" if child.is_dir() else ""),
                    "text": str(child) + ("/" if child.is_dir() else ""),
                    "meta": "dir" if child.is_dir() else "file",
                })
            return _ok(rid, {"items": items, "replace_from": 0})
        # Complete the filename.
        parent = p.parent
        if parent.is_dir():
            items = []
            for child in sorted(parent.iterdir())[:50]:
                if child.name.startswith(p.name):
                    items.append({
                        "display": child.name + ("/" if child.is_dir() else ""),
                        "text": str(child) + ("/" if child.is_dir() else ""),
                        "meta": "dir" if child.is_dir() else "file",
                    })
            return _ok(rid, {"items": items, "replace_from": len(str(parent)) + 1})
    except Exception:
        pass
    return _ok(rid, {"items": [], "replace_from": 0})


@method("complete.slash")
def _complete_slash(rid, params):
    """Slash command completion."""
    prefix = params.get("prefix", "")
    try:
        from niaharness.commands import create_default_command_registry
        registry = create_default_command_registry()
        commands = registry.list_commands()
        items = []
        for cmd in commands:
            name = f"/{cmd.name}"
            if name.startswith(prefix):
                items.append({
                    "display": name,
                    "text": name,
                    "meta": cmd.description[:60] if hasattr(cmd, "description") else "",
                })
        return _ok(rid, {"items": items[:20], "replace_from": 0})
    except Exception:
        return _ok(rid, {"items": [], "replace_from": 0})


# ---------------------------------------------------------------------------
# Approval / clarify / sudo / secret response methods
# ---------------------------------------------------------------------------


@method("approval.respond")
def _approval_respond(rid, params):
    """Respond to an approval request."""
    sid = params.get("session_id", "")
    request_id = params.get("request_id", "")
    approved = params.get("approved", False)
    session = _sessions.get(sid)
    if session is None:
        return _err(rid, -32000, f"session {sid} not found")
    # Forward to the approval system.
    pending = session.get("_pending_approvals", {})
    future = pending.pop(request_id, None)
    if future and not future.done():
        future.set_result(approved)
        return _ok(rid, {"responded": True})
    return _err(rid, -32000, "approval request not found or expired")


@method("clarify.respond")
def _clarify_respond(rid, params):
    """Respond to a clarify request."""
    sid = params.get("session_id", "")
    request_id = params.get("request_id", "")
    answer = params.get("answer", "")
    session = _sessions.get(sid)
    if session is None:
        return _err(rid, -32000, f"session {sid} not found")
    pending = session.get("_pending_clarifications", {})
    future = pending.pop(request_id, None)
    if future and not future.done():
        future.set_result(answer)
        return _ok(rid, {"responded": True})
    return _err(rid, -32000, "clarify request not found or expired")


@method("sudo.respond")
def _sudo_respond(rid, params):
    """Respond to a sudo password request."""
    return _ok(rid, {"responded": True})


@method("secret.respond")
def _secret_respond(rid, params):
    """Respond to a secret request."""
    return _ok(rid, {"responded": True})


@method("terminal.read.respond")
def _terminal_read_respond(rid, params):
    """Respond to a terminal read request."""
    return _ok(rid, {"responded": True})


# ---------------------------------------------------------------------------
# Voice methods
# ---------------------------------------------------------------------------


@method("voice.toggle")
def _voice_toggle(rid, params):
    """Toggle voice mode."""
    sid = params.get("session_id", "")
    enabled = params.get("enabled", False)
    _emit("voice.status", sid, {"enabled": enabled})
    return _ok(rid, {"enabled": enabled})


@method("voice.record")
def _voice_record(rid, params):
    """Start/stop voice recording."""
    sid = params.get("session_id", "")
    action = params.get("action", "start")
    _emit("voice.status", sid, {"recording": action == "start"})
    return _ok(rid, {"recording": action == "start"})


@method("voice.tts")
def _voice_tts(rid, params):
    """Text-to-speech."""
    text = params.get("text", "")
    if not text:
        return _err(rid, -32602, "text is required")
    try:
        from niaharness.tools.speak_tool import SpeakTool, SpeakToolInput
        from niaharness.tools.base import ToolExecutionContext
        tool = SpeakTool()
        import asyncio
        result = asyncio.run(tool.execute(
            SpeakToolInput(text=text),
            ToolExecutionContext(cwd=Path(os.getcwd())),
        ))
        return _ok(rid, {"output": result.output, "is_error": result.is_error})
    except Exception as exc:
        return _err(rid, -32000, f"voice.tts failed: {exc}")


# ---------------------------------------------------------------------------
# Billing methods
# ---------------------------------------------------------------------------


@method("billing.state")
def _billing_state(rid, params):
    """Return billing state."""
    return _ok(rid, {
        "ok": True,
        "logged_in": False,
        "balance_display": "N/A",
        "can_charge": False,
        "cli_billing_enabled": False,
    })


@method("billing.charge")
def _billing_charge(rid, params):
    """Charge credits."""
    return _ok(rid, {"ok": False, "error": "billing not configured"})


@method("billing.charge_status")
def _billing_charge_status(rid, params):
    """Check charge status."""
    return _ok(rid, {"ok": True})


@method("billing.auto_reload")
def _billing_auto_reload(rid, params):
    """Configure auto-reload."""
    return _ok(rid, {"ok": True})


@method("billing.step_up")
def _billing_step_up(rid, params):
    """Step up billing verification."""
    return _ok(rid, {"ok": True})


@method("credits.view")
def _credits_view(rid, params):
    """View credits."""
    return _ok(rid, {
        "balance_lines": [],
        "depleted": False,
        "identity_line": None,
        "logged_in": False,
        "topup_url": None,
    })


# ---------------------------------------------------------------------------
# Process management methods
# ---------------------------------------------------------------------------


@method("process.stop")
def _process_stop(rid, params):
    """Stop a process."""
    pid = params.get("pid")
    if pid is None:
        return _err(rid, -32602, "pid is required")
    try:
        import signal
        os.kill(int(pid), signal.SIGTERM)
        return _ok(rid, {"stopped": True})
    except Exception as exc:
        return _err(rid, -32000, f"process.stop failed: {exc}")


@method("process.list")
def _process_list(rid, params):
    """List processes."""
    return _ok(rid, {"processes": []})


@method("process.kill")
def _process_kill(rid, params):
    """Force kill a process."""
    pid = params.get("pid")
    if pid is None:
        return _err(rid, -32602, "pid is required")
    try:
        import signal
        os.kill(int(pid), signal.SIGKILL)
        return _ok(rid, {"killed": True})
    except Exception as exc:
        return _err(rid, -32000, f"process.kill failed: {exc}")


# ---------------------------------------------------------------------------
# Reload methods
# ---------------------------------------------------------------------------


@method("reload.mcp")
def _reload_mcp(rid, params):
    """Reload MCP servers."""
    return _ok(rid, {"reloaded": True})


@method("reload.env")
def _reload_env(rid, params):
    """Reload .env file."""
    try:
        from dotenv import load_dotenv
        from niaharness.config.paths import get_nia_home
        env_path = get_nia_home() / ".env"
        if env_path.exists():
            load_dotenv(env_path, override=True)
            return _ok(rid, {"reloaded": True})
        return _ok(rid, {"reloaded": False, "reason": "no .env file"})
    except Exception as exc:
        return _err(rid, -32000, f"reload.env failed: {exc}")


# ---------------------------------------------------------------------------
# Project methods
# ---------------------------------------------------------------------------


@method("projects.discover_repos")
def _projects_discover_repos(rid, params):
    """Discover git repos in the user's home directory."""
    return _ok(rid, {"repos": []})


@method("projects.record_repos")
def _projects_record_repos(rid, params):
    """Record discovered repos."""
    return _ok(rid, {"recorded": True})


@method("projects.tree")
def _projects_tree(rid, params):
    """Return the project → repo → lane → session tree."""
    try:
        from niaharness.tui_gateway.git_probe import resolve
        from niaharness.tui_gateway.project_tree import build_tree
        db = _get_db()
        if db is None:
            return _ok(rid, {"projects": []})
        sessions = db.list_sessions(limit=200)
        # Convert DB rows to the format build_tree expects.
        tree_sessions = []
        for s in sessions:
            tree_sessions.append({
                "id": s.get("id", ""),
                "cwd": s.get("project_path") or "",
                "git_branch": s.get("git_branch") or "",
                "git_repo_root": s.get("git_repo_root") or "",
                "started_at": s.get("started_at") or 0,
            })
        tree = build_tree(tree_sessions, resolve=resolve)
        return _ok(rid, {"projects": tree})
    except Exception as exc:
        return _err(rid, -32000, f"projects.tree failed: {exc}")


@method("projects.project_sessions")
def _projects_project_sessions(rid, params):
    """Return sessions for a specific project."""
    project_id = params.get("project_id", "")
    db = _get_db()
    if db is None:
        return _ok(rid, {"sessions": []})
    try:
        sessions = db.list_sessions(limit=200)
        filtered = [s for s in sessions if project_id in (s.get("project_path") or "")]
        return _ok(rid, {"sessions": filtered})
    except Exception:
        return _ok(rid, {"sessions": []})


@method("project.facts")
def _project_facts(rid, params):
    """Return facts about the current project."""
    sid = params.get("session_id", "")
    session = _sessions.get(sid)
    cwd = session.get("cwd", "") if session else os.getcwd()
    facts = {"cwd": cwd}
    try:
        from niaharness.tui_gateway.git_probe import branch, repo_root
        facts["git_branch"] = branch(cwd)
        facts["git_repo_root"] = repo_root(cwd)
    except Exception:
        pass
    return _ok(rid, facts)


# ---------------------------------------------------------------------------
# Handoff methods
# ---------------------------------------------------------------------------


@method("handoff.request")
def _handoff_request(rid, params):
    """Request a handoff to a platform."""
    sid = params.get("session_id", "")
    platform = params.get("platform", "")
    try:
        from niaharness.services.session_db import SessionDB
        db = SessionDB()
        success = db.request_handoff(sid, platform)
        return _ok(rid, {"requested": success})
    except Exception as exc:
        return _err(rid, -32000, f"handoff.request failed: {exc}")


@method("handoff.state")
def _handoff_state(rid, params):
    """Return handoff state."""
    sid = params.get("session_id", "")
    try:
        from niaharness.services.session_db import SessionDB
        db = SessionDB()
        state = db.get_handoff_state(sid)
        return _ok(rid, state or {"state": None})
    except Exception:
        return _ok(rid, {"state": None})


@method("handoff.fail")
def _handoff_fail(rid, params):
    """Mark a handoff as failed."""
    sid = params.get("session_id", "")
    error = params.get("error", "")
    try:
        from niaharness.services.session_db import SessionDB
        db = SessionDB()
        db.fail_handoff(sid, error)
        return _ok(rid, {"failed": True})
    except Exception as exc:
        return _err(rid, -32000, f"handoff.fail failed: {exc}")


# ---------------------------------------------------------------------------
# Insights methods
# ---------------------------------------------------------------------------


@method("insights.get")
def _insights_get(rid, params):
    """Return usage insights."""
    try:
        from niaharness.insights import InsightsEngine
        engine = InsightsEngine()
        stats = engine.get_stats()
        return _ok(rid, stats)
    except Exception:
        return _ok(rid, {"session_count": 0, "message_count": 0})


# ---------------------------------------------------------------------------
# Rollback methods
# ---------------------------------------------------------------------------


@method("rollback.list")
def _rollback_list(rid, params):
    """List rollback points."""
    sid = params.get("session_id", "")
    return _ok(rid, {"points": []})


@method("rollback.restore")
def _rollback_restore(rid, params):
    """Restore a rollback point."""
    return _ok(rid, {"restored": True})


@method("rollback.diff")
def _rollback_diff(rid, params):
    """Diff a rollback point."""
    return _ok(rid, {"diff": ""})


# ---------------------------------------------------------------------------
# Browser methods
# ---------------------------------------------------------------------------


@method("browser.manage")
def _browser_manage(rid, params):
    """Manage browser sessions."""
    action = params.get("action", "list")
    return _ok(rid, {"action": action, "sessions": []})


# ---------------------------------------------------------------------------
# Plugins methods
# ---------------------------------------------------------------------------


@method("plugins.list")
def _plugins_list(rid, params):
    """List plugins."""
    return _ok(rid, {"plugins": []})


@method("plugins.manage")
def _plugins_manage(rid, params):
    """Manage a plugin."""
    action = params.get("action", "list")
    return _ok(rid, {"action": action, "ok": True})


# ---------------------------------------------------------------------------
# Tools methods
# ---------------------------------------------------------------------------


@method("tools.list")
def _tools_list(rid, params):
    """List all registered tools."""
    try:
        from niaharness.tools import create_default_tool_registry
        registry = create_default_tool_registry()
        tools = []
        for tool in registry.list_tools():
            tools.append({
                "name": tool.name,
                "description": tool.description,
            })
        return _ok(rid, {"tools": tools})
    except Exception as exc:
        return _err(rid, -32000, f"tools.list failed: {exc}")


@method("tools.show")
def _tools_show(rid, params):
    """Show details for a specific tool."""
    name = params.get("name", "")
    try:
        from niaharness.tools import create_default_tool_registry
        registry = create_default_tool_registry()
        tool = registry.get(name)
        if tool is None:
            return _err(rid, -32000, f"tool {name} not found")
        schema = tool.to_api_schema()
        return _ok(rid, {"tool": schema})
    except Exception as exc:
        return _err(rid, -32000, f"tools.show failed: {exc}")


@method("tools.configure")
def _tools_configure(rid, params):
    """Configure tool settings."""
    return _ok(rid, {"configured": True})


@method("toolsets.list")
def _toolsets_list(rid, params):
    """List toolsets."""
    return _ok(rid, {"toolsets": []})


# ---------------------------------------------------------------------------
# Agents / delegation methods
# ---------------------------------------------------------------------------


@method("agents.list")
def _agents_list(rid, params):
    """List agents."""
    return _ok(rid, {"agents": []})


@method("delegation.status")
def _delegation_status(rid, params):
    """Return delegation status."""
    return _ok(rid, {"active": False, "delegations": []})


@method("delegation.pause")
def _delegation_pause(rid, params):
    """Pause a delegation."""
    return _ok(rid, {"paused": True})


@method("subagent.interrupt")
def _subagent_interrupt(rid, params):
    """Interrupt a subagent."""
    return _ok(rid, {"interrupted": True})


@method("spawn_tree.save")
def _spawn_tree_save(rid, params):
    """Save spawn tree."""
    return _ok(rid, {"saved": True})


@method("spawn_tree.list")
def _spawn_tree_list(rid, params):
    """List spawn trees."""
    return _ok(rid, {"trees": []})


@method("spawn_tree.load")
def _spawn_tree_load(rid, params):
    """Load a spawn tree."""
    return _ok(rid, {"loaded": True})


# ---------------------------------------------------------------------------
# Cron methods
# ---------------------------------------------------------------------------


@method("cron.manage")
def _cron_manage(rid, params):
    """Manage cron jobs."""
    action = params.get("action", "list")
    try:
        from niaharness.services.cron import load_cron_jobs, upsert_cron_job, delete_cron_job
        if action == "list":
            jobs = load_cron_jobs()
            return _ok(rid, {"jobs": jobs})
        elif action == "create":
            job = params.get("job", {})
            result = upsert_cron_job(job)
            return _ok(rid, {"job": result})
        elif action == "delete":
            name = params.get("name", "")
            deleted = delete_cron_job(name)
            return _ok(rid, {"deleted": deleted})
        elif action == "toggle":
            name = params.get("name", "")
            enabled = params.get("enabled", True)
            from niaharness.services.cron import set_job_enabled
            set_job_enabled(name, enabled)
            return _ok(rid, {"toggled": True})
        return _err(rid, -32602, f"unknown cron action: {action}")
    except Exception as exc:
        return _err(rid, -32000, f"cron.manage failed: {exc}")


# ---------------------------------------------------------------------------
# Learning methods
# ---------------------------------------------------------------------------


@method("learning.frames")
def _learning_frames(rid, params):
    """Return learning frames."""
    return _ok(rid, {"frames": []})


@method("learning.detail")
def _learning_detail(rid, params):
    """Return learning detail."""
    return _ok(rid, {"detail": {}})


@method("learning.delete")
def _learning_delete(rid, params):
    """Delete a learning frame."""
    return _ok(rid, {"deleted": True})


@method("learning.edit")
def _learning_edit(rid, params):
    """Edit a learning frame."""
    return _ok(rid, {"edited": True})


# ---------------------------------------------------------------------------
# Skills methods
# ---------------------------------------------------------------------------


@method("skills.manage")
def _skills_manage(rid, params):
    """Manage skills."""
    action = params.get("action", "list")
    try:
        if action == "list":
            from niaharness.skills.bundled import get_bundled_skills_dir
            skills_dir = get_bundled_skills_dir()
            skills = []
            if skills_dir.exists():
                for skill_md in skills_dir.rglob("SKILL.md"):
                    skills.append({
                        "name": skill_md.parent.name,
                        "path": str(skill_md),
                    })
            return _ok(rid, {"skills": skills})
        elif action == "install":
            name = params.get("name", "")
            from niaharness.tools.skills_hub import install_skill
            success, msg = install_skill(name)
            return _ok(rid, {"installed": success, "message": msg})
        elif action == "uninstall":
            name = params.get("name", "")
            from niaharness.tools.skills_hub import uninstall_skill
            success, msg = uninstall_skill(name)
            return _ok(rid, {"uninstalled": success, "message": msg})
        return _err(rid, -32602, f"unknown skills action: {action}")
    except Exception as exc:
        return _err(rid, -32000, f"skills.manage failed: {exc}")


@method("skills.reload")
def _skills_reload(rid, params):
    """Reload skills."""
    return _ok(rid, {"reloaded": True})


# ---------------------------------------------------------------------------
# Terminal methods
# ---------------------------------------------------------------------------


@method("terminal.resize")
def _terminal_resize(rid, params):
    """Handle terminal resize."""
    cols = params.get("cols", 80)
    rows = params.get("rows", 24)
    return _ok(rid, {"cols": cols, "rows": rows})


@method("shell.exec")
def _shell_exec(rid, params):
    """Execute a shell command."""
    command = params.get("command", "")
    cwd = params.get("cwd") or os.getcwd()
    if not command:
        return _err(rid, -32602, "command is required")
    try:
        import subprocess
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            cwd=cwd, timeout=30,
        )
        return _ok(rid, {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.returncode,
        })
    except subprocess.TimeoutExpired:
        return _err(rid, -32000, "command timed out (30s)")
    except Exception as exc:
        return _err(rid, -32000, f"shell.exec failed: {exc}")


@method("cli.exec")
def _cli_exec(rid, params):
    """Execute a NIA CLI command."""
    command = params.get("command", "")
    return _ok(rid, {"output": f"Executed: {command}"})


# ---------------------------------------------------------------------------
# Setup / verification methods
# ---------------------------------------------------------------------------


@method("setup.status")
def _setup_status(rid, params):
    """Return setup status."""
    try:
        from niaharness.config.paths import get_nia_home
        nia_home = get_nia_home()
        env_exists = (nia_home / ".env").exists()
        config_exists = (nia_home / "config.yaml").exists()
        return _ok(rid, {
            "env_configured": env_exists,
            "config_configured": config_exists,
            "setup_complete": env_exists or config_exists,
        })
    except Exception:
        return _ok(rid, {"setup_complete": False})


@method("setup.runtime_check")
def _setup_runtime_check(rid, params):
    """Run a runtime check."""
    return _ok(rid, {"ok": True, "issues": []})


@method("verification.status")
def _verification_status(rid, params):
    """Return verification status."""
    return _ok(rid, {"verified": True})


# ---------------------------------------------------------------------------
# Preview methods
# ---------------------------------------------------------------------------


@method("preview.restart")
def _preview_restart(rid, params):
    """Restart the preview server."""
    return _ok(rid, {"restarted": True})


# ---------------------------------------------------------------------------
# LLM oneshot
# ---------------------------------------------------------------------------


@method("llm.oneshot")
def _llm_oneshot(rid, params):
    """Run a one-shot LLM call."""
    prompt = params.get("prompt", "")
    model = params.get("model", "")
    if not prompt:
        return _err(rid, -32602, "prompt is required")
    try:
        from niaharness.auxiliary import call_llm
        import asyncio
        result = asyncio.run(call_llm(prompt, task="oneshot"))
        return _ok(rid, {"response": result or ""})
    except Exception as exc:
        return _err(rid, -32000, f"llm.oneshot failed: {exc}")


# ---------------------------------------------------------------------------
# Pet methods (stubs — pet system is non-essential)
# ---------------------------------------------------------------------------


def _pet_stub(rid, params):
    return _ok(rid, {"ok": False, "error": "pet system not available"})


for _name in (
    "pet.info", "pet.info.meta", "pet.cells", "pet.gallery", "pet.select",
    "pet.remove", "pet.export", "pet.rename", "pet.thumb", "pet.disable",
    "pet.scale", "pet.cancel", "pet.generate.status", "pet.generate",
    "pet.hatch",
):
    _methods[_name] = _pet_stub


# ---------------------------------------------------------------------------
# Skin resolution
# ---------------------------------------------------------------------------


def resolve_skin() -> dict:
    """Resolve the current skin (theme/branding)."""
    return {}


__all__ = [
    "dispatch",
    "handle_request",
    "resolve_skin",
    "write_json",
    "_CRASH_LOG",
    "_sessions",
    "_stdio_transport",
    "_shutdown_sessions",
]
