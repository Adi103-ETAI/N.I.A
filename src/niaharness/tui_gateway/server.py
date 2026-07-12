"""NIA TUI Gateway server — JSON-RPC dispatcher with 117 methods.

Ported from Hermes Agent's tui_gateway/server.py (13,897 LOC), adapted to
NIA's infrastructure (QueryEngine, SessionDB, ToolRegistry, Commands,
Skills, Cron, Profiles, Gateway, Memory, Permissions).

Wire protocol: newline-delimited JSON-RPC 2.0 in both directions.
The server emits a gateway.ready event immediately after startup,
then echoes responses/events for inbound requests.

Methods are registered via the @method("name") decorator and dispatched
by dispatch(). Long-running handlers run on a thread pool; everything
else runs inline.
"""

from __future__ import annotations

import contextlib
import contextvars
import copy
import inspect
import json
import logging
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
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
from niaharness.tui_gateway import git_probe
from niaharness.tui_gateway.render import render_message, make_stream_renderer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CRASH_LOG = Path(os.environ.get("NIA_HOME", os.path.expanduser("~/.nia"))) / "logs" / "gateway_crash.log"

_POOL_MAX_WORKERS = 4
_SLASH_WORKER_TIMEOUT_S = 30.0

# Methods that run on the pool (long-running).
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

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

_sessions: dict[str, dict] = {}
_sessions_lock = threading.Lock()
_methods: dict[str, Callable] = {}
_pool: ThreadPoolExecutor = ThreadPoolExecutor(
    max_workers=_POOL_MAX_WORKERS, thread_name_prefix="nia-gw",
)
_stdout_lock = threading.Lock()
_stdio_transport: Transport = StdioTransport(
    lambda: sys.stdout, _stdout_lock,
)

# Slash worker subprocesses (one per session).
_slash_workers: dict[str, "_SlashWorker"] = {}

# Session slot management.
_active_session_sid: Optional[str] = None
_active_session_lock = threading.Lock()

# Pending prompt state (for approval/clarify/sudo/secret blocking).
_pending: dict[str, tuple[str, threading.Event]] = {}
_answers: dict[str, str] = {}
_pending_prompt_payloads: dict[str, tuple[str, dict]] = {}
_prompt_lock = threading.Lock()

# Subagent run liveness registry — for `_child_run_active` guard so a watch
# session's parent doesn't accept a prompt while its child run is still active.
# Ported from Hermes (line ~3618 of hermes-agent/tui_gateway/server.py).
_CHILD_RUN_STALE_S = 60 * 5
_active_child_runs: dict[str, float] = {}
_child_mirrors_lock = threading.Lock()
_child_mirrors: dict[str, dict] = {}

# ---------------------------------------------------------------------------
# Crash logging
# ---------------------------------------------------------------------------


def _panic_hook(exc_type, exc_value, exc_tb):
    if issubclass(exc_type, KeyboardInterrupt):
        return
    try:
        _CRASH_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(_CRASH_LOG, "a", encoding="utf-8") as f:
            f.write(f"\n=== uncaught {exc_type.__name__} . {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
            traceback.print_exception(exc_type, exc_value, exc_tb, file=f)
    except Exception:
        pass


def _thread_panic_hook(args):
    try:
        _CRASH_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(_CRASH_LOG, "a", encoding="utf-8") as f:
            f.write(f"\n=== thread crash . {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
            traceback.print_exception(args.exc_type, args.exc_value, args.exc_traceback, file=f)
    except Exception:
        pass


sys.excepthook = _panic_hook
threading.excepthook = _thread_panic_hook


# ---------------------------------------------------------------------------
# _SlashWorker — persistent subprocess for slash commands
# ---------------------------------------------------------------------------


class _SlashWorker:
    """Persistent NIA subprocess for slash commands.

    Spawns ``python -m niaharness.tui_gateway.slash_worker`` as a child
    process. Reads JSON lines from stdin ({id, command}), writes
    {id, ok, output|error} to stdout. Has stdout/stderr draining threads,
    a timeout, and clean shutdown.
    """

    def __init__(self, session_key: str, model: str = ""):
        self._lock = threading.Lock()
        self._seq = 0
        self.stderr_tail: list[str] = []
        self.stdout_queue: queue.Queue[dict | None] = queue.Queue()

        argv = [
            sys.executable,
            "-m",
            "niaharness.tui_gateway.slash_worker",
            "--session-key",
            session_key,
        ]
        if model:
            argv += ["--model", model]

        self._closed = False
        self.proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=os.getcwd(),
            env={**os.environ},
            start_new_session=True,
        )
        threading.Thread(target=self._drain_stdout, daemon=True).start()
        threading.Thread(target=self._drain_stderr, daemon=True).start()

    def _drain_stdout(self):
        for line in self.proc.stdout or []:
            try:
                self.stdout_queue.put(json.loads(line))
            except json.JSONDecodeError:
                continue
        self.stdout_queue.put(None)

    def _drain_stderr(self):
        for line in self.proc.stderr or []:
            if text := line.rstrip("\n"):
                self.stderr_tail = (self.stderr_tail + [text])[-80:]

    def run(self, command: str) -> str:
        if self.proc.poll() is not None:
            raise RuntimeError("slash worker exited")

        with self._lock:
            self._seq += 1
            rid = self._seq
            self.proc.stdin.write(json.dumps({"id": rid, "command": command}) + "\n")
            self.proc.stdin.flush()

            while True:
                try:
                    msg = self.stdout_queue.get(timeout=_SLASH_WORKER_TIMEOUT_S)
                except queue.Empty:
                    raise RuntimeError("slash worker timed out")
                if msg is None:
                    break
                if msg.get("id") != rid:
                    continue
                if not msg.get("ok"):
                    raise RuntimeError(msg.get("error", "slash worker failed"))
                return str(msg.get("output", "")).rstrip()

            raise RuntimeError(
                f"slash worker closed pipe{': ' + chr(10).join(self.stderr_tail[-8:]) if self.stderr_tail else ''}"
            )

    def close(self):
        if getattr(self, "_closed", False):
            return
        self._closed = True
        proc = self.proc
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=1)
                except Exception:
                    proc.kill()
                    try:
                        proc.wait(timeout=1)
                    except Exception:
                        pass
        except Exception:
            try:
                proc.kill()
                proc.wait(timeout=1)
            except Exception:
                pass
        finally:
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                try:
                    stream.close()
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# Core I/O
# ---------------------------------------------------------------------------


def write_json(obj: dict) -> bool:
    """Emit one JSON frame. Routes via the most-specific transport available."""
    if obj.get("method") == "event":
        sid = ((obj.get("params") or {}).get("session_id")) or ""
        if sid and (t := (_sessions.get(sid) or {}).get("transport")) is not None:
            return t.write(obj)
    return (current_transport() or _stdio_transport).write(obj)


def _emit(event: str, sid: str, payload: dict | None = None):
    params = {"type": event, "session_id": sid}
    if payload is not None:
        params["payload"] = payload
    write_json({"jsonrpc": "2.0", "method": "event", "params": params})


def _emit_approval_request(sid: str, data: dict | None) -> None:
    payload = dict(data or {})
    if "command" in payload:
        try:
            from niaharness.gateway.response_filters import redact_secrets
            payload["command"] = redact_secrets(payload.get("command", ""))
        except Exception:
            pass
    _emit("approval.request", sid, payload)


def _status_update(sid: str, kind: str, text: str | None = None):
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
    def dec(fn):
        _methods[name] = fn
        return fn
    return dec


def _normalize_request(req: Any) -> tuple[Any, str, dict] | dict:
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
    normalized = _normalize_request(req)
    if isinstance(normalized, dict):
        return normalized
    rid, m, params = normalized
    fn = _methods.get(m)
    if not fn:
        return _err(rid, -32601, f"unknown method: {m}")
    return fn(rid, params)


def dispatch(req: dict, transport: Optional[Transport] = None) -> dict | None:
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


def _new_session_key() -> str:
    return uuid4().hex[:12]


def _sess_nowait(params, rid):
    s = _sessions.get(params.get("session_id") or "")
    return (s, None) if s else (None, _err(rid, 4001, "session not found"))


def _sess(params, rid):
    s, err = _sess_nowait(params, rid)
    if err:
        return (None, err)
    _start_agent_build(params.get("session_id") or "", s)
    return (s, _wait_agent(s, rid))


def _wait_agent(session: dict, rid: str, timeout: float = 30.0) -> dict | None:
    ready = session.get("agent_ready")
    if ready is not None and not ready.wait(timeout=timeout):
        return _err(rid, 5032, "agent initialization timed out")
    err = session.get("agent_error")
    return _err(rid, 5032, err) if err else None


def _claim_active_session_slot(sid: str, session: dict) -> bool:
    global _active_session_sid
    with _active_session_lock:
        if _active_session_sid is not None and _active_session_sid != sid:
            return False
        _active_session_sid = sid
        session["is_active"] = True
        return True


def _release_active_session_slot(session: dict | None) -> None:
    global _active_session_sid
    with _active_session_lock:
        if session and session.get("id") == _active_session_sid:
            _active_session_sid = None
            session["is_active"] = False


def _finalize_session(session: dict | None, end_reason: str = "tui_close") -> None:
    if session is None:
        return
    sid = session.get("id", "")
    _release_active_session_slot(session)
    _notify_session_boundary("on_session_finalize", sid)
    try:
        db = _get_db()
        if db:
            db.end_session(sid, end_reason)
    except Exception:
        pass


def _teardown_session(session: dict | None, *, end_reason: str = "tui_close") -> None:
    if session is None:
        return
    _finalize_session(session, end_reason=end_reason)
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
    sid = session.get("id", "")
    worker = _slash_workers.pop(sid, None)
    if worker is not None:
        try:
            worker.close()
        except Exception:
            pass


def _close_session_by_id(sid: str, *, end_reason: str = "tui_close") -> bool:
    with _sessions_lock:
        session = _sessions.pop(sid, None)
    if session is None:
        return False
    _teardown_session(session, end_reason=end_reason)
    return True


def _shutdown_sessions() -> None:
    for sid in list(_sessions.keys()):
        _close_session_by_id(sid, end_reason="shutdown")


# ---------------------------------------------------------------------------
# Session cap enforcement (ported from Hermes lines 784-862)
# ---------------------------------------------------------------------------

_MAX_LIVE_SESSIONS = int(os.environ.get("NIA_MAX_LIVE_SESSIONS", "10") or "10")
_IDLE_REAP_GRACE_S = float(os.environ.get("NIA_IDLE_REAP_GRACE_S", "300") or "300")


def _session_is_evictable(sid: str, session: dict, now: float) -> bool:
    """True if a session is idle long enough to be reaped."""
    if session.get("running"):
        return False
    if session.get("_finalized"):
        return False
    last_active = float(session.get("last_active") or session.get("created_at") or now)
    return (now - last_active) > _IDLE_REAP_GRACE_S


def _session_is_lru_evictable(sid: str, session: dict) -> bool:
    """True if a session can be LRU-evicted to enforce the session cap."""
    return not session.get("running") and not session.get("_finalized")


def _enforce_session_cap() -> None:
    """Evict the oldest idle sessions to stay under the cap."""
    with _sessions_lock:
        if len(_sessions) <= _MAX_LIVE_SESSIONS:
            return
        # Sort by last_active ascending — oldest first.
        candidates = sorted(
            ((sid, s) for sid, s in _sessions.items() if _session_is_lru_evictable(sid, s)),
            key=lambda x: float(x[1].get("last_active") or x[1].get("created_at") or 0),
        )
        excess = len(_sessions) - _MAX_LIVE_SESSIONS
        for sid, _ in candidates[:excess]:
            _close_session_by_id(sid, end_reason="session_cap")


def _schedule_session_cap_enforcement() -> None:
    """Trim detached idle sessions over the cap (deferred)."""
    timer = threading.Timer(0.1, _enforce_session_cap)
    timer.daemon = True
    timer.start()


def _notify_session_boundary(event_type: str, session_id: str | None) -> None:
    try:
        from niaharness.hooks import HookEvent, HookExecutor
        executor = HookExecutor()
        executor.fire(HookEvent(event_type, {"session_id": session_id}))
    except Exception:
        pass


def _session_cwd(session: dict | None) -> str:
    if session and session.get("cwd"):
        return str(session["cwd"])
    return _default_session_cwd()


def _completion_cwd(params: dict | None = None) -> str:
    params = params or {}
    raw = (
        params.get("cwd")
        or _sessions.get(params.get("session_id") or "", {}).get("cwd")
        or os.environ.get("NIA_CWD")
        or os.getcwd()
    )
    try:
        resolved = os.path.abspath(os.path.expanduser(str(raw)))
        if os.path.isdir(resolved):
            return resolved
    except Exception:
        pass
    return os.getcwd()


def _default_session_cwd() -> str:
    try:
        from niaharness.config.settings import load_settings
        settings = load_settings()
        if settings.cwd:
            return str(settings.cwd)
    except Exception:
        pass
    return os.getcwd()


def _resolve_model() -> str:
    try:
        from niaharness.config.settings import load_settings
        settings = load_settings()
        return settings.model or "claude-3-haiku-20240307"
    except Exception:
        return "claude-3-haiku-20240307"


def _load_cfg() -> dict:
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
    try:
        import yaml
        from niaharness.config import get_config_file_path
        path = get_config_file_path()
        if path:
            Path(path).write_text(yaml.dump(cfg, default_flow_style=False), encoding="utf-8")
    except Exception:
        pass


def _get_db():
    try:
        from niaharness.services.session_db import SessionDB
        return SessionDB()
    except Exception:
        return None


def _load_busy_input_mode() -> str:
    display = _load_cfg().get("display")
    if not isinstance(display, dict):
        display = {}
    raw = str(display.get("busy_input_mode", "") or "").strip().lower()
    return raw if raw in {"queue", "steer", "interrupt"} else "interrupt"


def _enable_gateway_prompts() -> None:
    os.environ["NIA_GATEWAY_SESSION"] = "1"
    os.environ["NIA_INTERACTIVE"] = "1"


def _load_show_reasoning() -> bool:
    display = _load_cfg().get("display")
    if not isinstance(display, dict):
        display = {}
    return bool(display.get("show_reasoning", True))


def _load_tool_progress_mode() -> str:
    display = _load_cfg().get("display")
    if not isinstance(display, dict):
        display = {}
    return str(display.get("tool_progress_mode", "compact"))


# ---------------------------------------------------------------------------
# Inflight turn tracking
# ---------------------------------------------------------------------------


def _inflight_text(text: Any) -> str:
    if isinstance(text, str):
        return text
    if isinstance(text, list):
        parts = []
        for item in text:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                parts.append(item)
        return " ".join(parts)
    return str(text) if text else ""


def _start_inflight_turn(session: dict, text: Any) -> None:
    now = time.time()
    session["inflight_turn"] = {
        "assistant": "",
        "started_at": now,
        "streaming": True,
        "updated_at": now,
        "user": _inflight_text(text),
    }


def _append_inflight_delta(session: dict, delta: Any) -> None:
    text = "" if delta is None else str(delta)
    if not text:
        return
    turn = session.get("inflight_turn")
    if not isinstance(turn, dict):
        turn = {"assistant": "", "streaming": True, "user": ""}
    turn["assistant"] = f"{turn.get('assistant') or ''}{text}"
    turn["streaming"] = True
    turn["updated_at"] = time.time()
    session["inflight_turn"] = turn


def _clear_inflight_turn(session: dict) -> None:
    session["inflight_turn"] = None


def _inflight_snapshot(session: dict) -> dict | None:
    turn = session.get("inflight_turn")
    if not isinstance(turn, dict):
        return None
    user = str(turn.get("user") or "").strip()
    assistant = str(turn.get("assistant") or "")
    streaming = bool(turn.get("streaming"))
    if not user and not assistant and not streaming:
        return None
    return {"assistant": assistant, "streaming": streaming, "user": user}


# ---------------------------------------------------------------------------
# Queued prompt management
# ---------------------------------------------------------------------------


def _enqueue_prompt(session: dict, text: Any, transport: Any) -> None:
    existing = session.get("queued_prompt")
    if existing and isinstance(existing.get("text"), str) and isinstance(text, str):
        prev = existing["text"]
        text = f"{prev}\n\n{text}" if prev and text else (prev or text)
    session["queued_prompt"] = {"text": text, "transport": transport}


def _drain_queued_prompt(rid, sid: str, session: dict) -> bool:
    with session["history_lock"]:
        queued = session.get("queued_prompt")
        if not queued or session.get("running"):
            return False
        session["queued_prompt"] = None
        session["running"] = True
        if queued.get("transport") is not None:
            session["transport"] = queued["transport"]
    try:
        _run_prompt_submit(rid, sid, session, queued["text"])
    except Exception as exc:
        print(f"[tui_gateway] queued prompt dispatch failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        with session["history_lock"]:
            session["running"] = False
    return True


def _handle_busy_submit(rid, sid: str, session: dict, text: Any, transport: Any) -> dict:
    """Apply the display.busy_input_mode policy to a prompt that lands while busy."""
    mode = _load_busy_input_mode()
    agent = session.get("agent")
    if mode == "steer" and agent is not None and hasattr(agent, "steer"):
        try:
            if agent.steer(text):
                session["last_active"] = time.time()
                return _ok(rid, {"status": "steered"})
        except Exception:
            pass
    if mode != "queue" and agent is not None and hasattr(agent, "interrupt"):
        try:
            agent.interrupt()
        except Exception:
            pass
    _enqueue_prompt(session, text, transport)
    session["last_active"] = time.time()
    return _ok(rid, {"status": "queued"})


# ---------------------------------------------------------------------------
# Blocking prompt factory (for approval/clarify/sudo/secret)
# ---------------------------------------------------------------------------


def _block(event: str, sid: str, payload: dict, timeout: int = 300) -> str:
    rid = uuid4().hex[:8]
    ev = threading.Event()
    with _prompt_lock:
        _pending[rid] = (sid, ev)
        payload["request_id"] = rid
        _pending_prompt_payloads[rid] = (event, dict(payload))
    try:
        _emit(event, sid, payload)
        ev.wait(timeout=timeout)
    finally:
        with _prompt_lock:
            _pending.pop(rid, None)
            _pending_prompt_payloads.pop(rid, None)
    with _prompt_lock:
        return _answers.pop(rid, "")


def _clear_pending(sid: str | None = None) -> None:
    with _prompt_lock:
        for rid, (owner_sid, ev) in list(_pending.items()):
            if sid is None or owner_sid == sid:
                _answers[rid] = ""
                ev.set()


# ---------------------------------------------------------------------------
# Agent building
# ---------------------------------------------------------------------------


def _start_agent_build(sid: str, session: dict) -> None:
    """Start building the real agent for a TUI session, once."""
    ready = session.get("agent_ready")
    if ready is None:
        return
    lock = session.setdefault("agent_build_lock", threading.Lock())
    with lock:
        if ready.is_set() or session.get("agent_build_started"):
            return
        session["agent_build_started"] = True

    def _build() -> None:
        with _sessions_lock:
            current = _sessions.get(sid)
        if current is None:
            ready.set()
            return

        try:
            # Build the agent via NIA's runtime.
            import asyncio
            from niaharness.ui.runtime import build_runtime, start_runtime
            from niaharness.api.client import AnthropicApiClient
            from niaharness.config.settings import load_settings

            settings = load_settings()
            model = session.get("model_override", {}).get("model") if isinstance(session.get("model_override"), dict) else None
            model = model or _resolve_model()

            # Build the API client.
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            base_url = os.environ.get("NIA_BASE_URL", "")
            if base_url:
                from niaharness.api.openai_client import OpenAICompatibleClient
                api_client = OpenAICompatibleClient(
                    api_key=api_key,
                    base_url=base_url,
                    model=model,
                )
            else:
                api_client = AnthropicApiClient(api_key=api_key, model=model)

            # Build the tool registry.
            from niaharness.tools import create_default_tool_registry
            tool_registry = create_default_tool_registry()

            # Build the permission checker.
            from niaharness.permissions.checker import PermissionChecker
            from niaharness.config.settings import PermissionSettings
            permission_checker = PermissionChecker(PermissionSettings())

            # Build the query engine.
            from niaharness.engine import QueryEngine
            from niaharness.engine.messages import ConversationMessage
            from niaharness.prompts.system_prompt import build_system_prompt

            system_prompt = build_system_prompt(cwd=session.get("cwd", os.getcwd()))
            engine = QueryEngine(
                api_client=api_client,
                tool_registry=tool_registry,
                permission_checker=permission_checker,
                cwd=session.get("cwd", os.getcwd()),
                model=model,
                system_prompt=system_prompt,
            )

            # Restore conversation history if present.
            if session.get("history"):
                for msg in session["history"]:
                    role = msg.get("role", "user")
                    text = msg.get("text", msg.get("content", ""))
                    if role == "user":
                        engine._messages.append(ConversationMessage.from_user_text(text))
                    elif role == "assistant":
                        from niaharness.engine.messages import TextBlock
                        engine._messages.append(ConversationMessage(
                            role="assistant",
                            content=[TextBlock(text=text)],
                        ))

            current["agent"] = engine
            current["model"] = model
            current["provider"] = getattr(api_client, "provider", "")

            # Start the slash worker.
            try:
                worker = _SlashWorker(
                    session.get("session_key", sid),
                    model,
                )
                _slash_workers[sid] = worker
            except Exception:
                pass

            # Wire approval callbacks.
            try:
                from niaharness.permissions.approval import (
                    set_current_session_key,
                    register_gateway_notify,
                    load_permanent_allowlist,
                )
                session_key = session.get("session_key", sid)
                register_gateway_notify(
                    session_key, lambda data: _emit_approval_request(sid, data)
                )
                load_permanent_allowlist()
            except Exception:
                pass

            _notify_session_boundary("on_session_reset", session.get("session_key", sid))

            info = _session_info(engine, current)
            _emit("session.info", sid, info)

        except Exception as e:
            current["agent_error"] = str(e)
            _emit("error", sid, {"message": f"agent init failed: {e}"})
        finally:
            ready.set()

    threading.Thread(target=_build, daemon=True).start()


def _session_info(agent, session: dict | None = None) -> dict:
    """Build a session info dict for the TUI."""
    cwd = _session_cwd(session)
    session_key = str((session or {}).get("session_key") or getattr(agent, "session_id", "") or "")

    # Resolve git info.
    branch = ""
    repo_root = ""
    try:
        branch = git_probe.branch(cwd)
        repo_root = git_probe.repo_root(cwd)
    except Exception:
        pass

    # Check yolo status.
    yolo = False
    try:
        from niaharness.permissions.approval import is_session_yolo_enabled
        if session_key:
            yolo = bool(is_session_yolo_enabled(session_key))
    except Exception:
        pass

    info: dict = {
        "model": getattr(agent, "model", "") or (session or {}).get("model", ""),
        "provider": getattr(agent, "provider", "") or (session or {}).get("provider", ""),
        "reasoning_effort": "",
        "service_tier": "",
        "fast": False,
        "yolo": yolo,
        "tools": {},
        "skills": {},
        "cwd": cwd,
        "branch": branch,
        "running": bool((session or {}).get("running")),
        "title": (session or {}).get("title", ""),
        "usage": _get_usage(agent),
    }

    # Populate tools list.
    try:
        from niaharness.tools import create_default_tool_registry
        registry = create_default_tool_registry()
        tools_by_set: dict[str, list[str]] = {}
        for tool in registry.list_tools():
            tools_by_set.setdefault("all", []).append(tool.name)
        info["tools"] = tools_by_set
    except Exception:
        pass

    # Populate skills list.
    try:
        from niaharness.skills.bundled import get_bundled_skills_dir
        skills_dir = get_bundled_skills_dir()
        if skills_dir.exists():
            info["skills"] = {"bundled": [p.parent.name for p in skills_dir.rglob("SKILL.md")]}
    except Exception:
        pass

    # Version info.
    try:
        from niaharness import __version__
        info["version"] = __version__
    except Exception:
        info["version"] = ""

    return info


def _get_usage(agent) -> dict:
    """Return usage stats from the agent."""
    try:
        cost_tracker = getattr(agent, "_cost_tracker", None)
        if cost_tracker:
            total = cost_tracker.total
            return {
                "model": getattr(agent, "model", ""),
                "input": total.input_tokens,
                "output": total.output_tokens,
                "reasoning": getattr(total, "reasoning_tokens", 0),
                "total": total.input_tokens + total.output_tokens,
                "calls": getattr(agent, "_api_call_count", 0) or 0,
            }
    except Exception:
        pass
    return {
        "model": getattr(agent, "model", ""),
        "input": 0,
        "output": 0,
        "total": 0,
        "calls": 0,
    }


def _wire_callbacks(sid: str):
    """Wire sudo/secret/project callbacks to gateway blocking prompts."""
    try:
        from niaharness.permissions.approval import set_current_session_key
        # NIA doesn't have the same terminal_tool/skills_tool callback APIs
        # as Hermes, but we wire what we can.
    except Exception:
        pass


def _ensure_session_db_row(session: dict) -> None:
    """Idempotently persist the session's DB row on first real activity."""
    key = session.get("session_key") or session.get("id")
    if not key:
        return
    db = _get_db()
    if db is None:
        return
    try:
        existing = db.get_session(key)
        if existing is None:
            model = session.get("model", _resolve_model())
            db.create_session(
                key,
                cwd=session.get("cwd", ""),
                model=model,
            )
        # Update git metadata.
        cwd = session.get("cwd", "")
        if cwd:
            try:
                branch = git_probe.branch(cwd)
                repo_root = git_probe.repo_root(cwd)
                if branch or repo_root:
                    db.update_session(key, git_branch=branch, git_repo_root=repo_root)
            except Exception:
                pass
    except Exception as exc:
        logger.debug("ensure_session_db_row failed: %s", exc)


# ---------------------------------------------------------------------------
# Session management methods
# ---------------------------------------------------------------------------


@method("session.create")
def _session_create(rid, params):
    """Create a new TUI session with full lifecycle management.

    Deep-ported from Hermes line 4975. Handles:
    - Seed history coercion (_coerce_seed_history).
    - Profile-scoped home (profile_home for app-global remote mode).
    - Per-session model/provider/reasoning/fast overrides from the desktop composer.
    - Lazy agent build (returns immediately, builds on a timer).
    - Active-session slot claim.
    - Session cap enforcement (trim detached idle sessions over the cap).
    """
    sid = uuid4().hex[:8]
    key = _new_session_key()
    cols = int(params.get("cols", 80))
    history = _coerce_seed_history(params.get("messages"))
    title = str(params.get("title") or "").strip()
    parent_session_id = str(params.get("parent_session_id") or "").strip() or None
    source = str(params.get("source") or "tui").strip() or "tui"

    # Resolve cwd with the same precedence as Hermes.
    raw_cwd = str(params.get("cwd") or "").strip()
    try:
        explicit_cwd = bool(raw_cwd) and os.path.isdir(os.path.abspath(os.path.expanduser(raw_cwd)))
    except Exception:
        explicit_cwd = False
    resolved_cwd = _completion_cwd(params)

    # Profile-scoped home (app-global remote mode).
    profile = (params.get("profile") or "").strip() or None
    profile_home = _profile_home(profile)

    # Per-session model override (from desktop composer pick).
    create_model = str(params.get("model") or "").strip()
    session_model_override = (
        {"model": create_model, "provider": str(params.get("provider") or "").strip() or None}
        if create_model
        else None
    )

    # Reasoning effort override.
    create_reasoning_override = None
    effort = str(params.get("reasoning_effort") or "").strip()
    if effort:
        try:
            # Store as-is — NIA may not have parse_reasoning_effort yet.
            create_reasoning_override = {"effort": effort.lower(), "enabled": effort.lower() != "none"}
        except Exception:
            create_reasoning_override = None

    # Service tier (fast) override.
    create_service_tier_override = "priority" if params.get("fast") else None

    _enable_gateway_prompts()

    ready = threading.Event()
    now = time.time()

    with _sessions_lock:
        _sessions[sid] = {
            "agent": None,
            "agent_error": None,
            "agent_ready": ready,
            "agent_build_started": False,
            "active_session_lease": None,
            "attached_images": [],
            "close_on_disconnect": _is_truthy(params.get("close_on_disconnect", False)),
            "cols": cols,
            "created_at": now,
            "cwd": resolved_cwd,
            "display_history_prefix": [],
            "edit_snapshots": {},
            "explicit_cwd": explicit_cwd,
            "history": history,
            "history_lock": threading.Lock(),
            "history_version": 0,
            "id": sid,
            "image_counter": 0,
            "inflight_turn": None,
            "last_active": now,
            "model_override": session_model_override,
            "create_reasoning_override": create_reasoning_override,
            "create_service_tier_override": create_service_tier_override,
            "parent_session_id": parent_session_id,
            "pending_title": title or None,
            "profile_home": str(profile_home) if profile_home is not None else None,
            "running": False,
            "session_key": key,
            "show_reasoning": _load_show_reasoning(),
            "slash_worker": None,
            "source": source,
            "tool_progress_mode": _load_tool_progress_mode(),
            "tool_started_at": {},
            "transport": current_transport() or _stdio_transport,
            "queued_prompt": None,
        }
        _register_session_cwd(_sessions[sid])

    # NOTE: we intentionally do NOT persist a DB row here. Every TUI/desktop
    # launch opens a session just to paint the composer, so eagerly creating
    # a row left an "Untitled" empty session behind for every launch the user
    # never typed into. The row is created lazily on the first prompt.

    # Return the lightweight session immediately so Ink can paint the composer,
    # then build the real agent just after this response is flushed.
    _schedule_agent_build(sid)
    _schedule_session_cap_enforcement()

    # Build the info response.
    info_model = session_model_override.get("model") if session_model_override else _resolve_model()
    info: dict = {
        "model": info_model,
        "tools": {},
        "skills": {},
        "cwd": resolved_cwd,
        "branch": _git_branch_for_cwd(resolved_cwd),
        "lazy": True,
        "profile_name": _current_profile_name(),
    }
    if session_model_override and session_model_override.get("provider"):
        info["provider"] = session_model_override["provider"]

    return _ok(rid, {
        "session_id": sid,
        "stored_session_id": key,
        "message_count": len(history),
        "messages": _history_to_messages(history),
        "info": info,
    })


@method("session.list")
def _session_list(rid, params):
    """List sessions with proper filtering."""
    db = _get_db()
    if db is None:
        return _err(rid, 5006, "session DB not available")
    try:
        limit = int(params.get("limit", 200) or 200)
        include_archived = bool(params.get("include_archived", False))
        # Use list_sessions_rich for preview + last_active.
        try:
            rows = db.list_sessions_rich(
                limit=limit,
                include_archived=include_archived,
            )
        except Exception:
            rows = db.list_sessions(limit=limit, include_archived=include_archived)
        return _ok(rid, {
            "sessions": [
                {
                    "id": s.get("id", ""),
                    "title": s.get("title") or "",
                    "preview": s.get("preview") or "",
                    "started_at": s.get("started_at") or 0,
                    "message_count": s.get("message_count") or 0,
                    "source": s.get("source") or "",
                }
                for s in rows
            ]
        })
    except Exception as e:
        return _err(rid, 5006, str(e))


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
                return _ok(rid, {
                    "session_id": s["id"],
                    "started_at": s.get("started_at", 0),
                    "title": s.get("title", ""),
                })
        return _ok(rid, {"session_id": None})
    except Exception:
        return _ok(rid, {"session_id": None})


@method("session.resume")
def _session_resume(rid, params):
    """Resume a session by ID, prefix, or title, with full history restoration.

    Deep-ported from Hermes line 5355. Handles:
    - Prefix / title resolution (falls back to ``get_session_by_title``).
    - Lazy watch windows for live subagent children (skip history, attach
      to the child's live mirror stream).
    - Compression-continuation chain following (``resolve_resume_session_id``)
      so resuming a rotated-out parent id binds to the live tip.
    - Live-session fast path (if the session is already in-memory, reuse it).
    - Cold resume default: return the full transcript immediately, build the
      agent on a deferred timer so the RPC doesn't block on agent
      construction.
    - Eager build fallback (``eager_build: true``) for callers that need the
      agent synchronously.
    """
    target = params.get("session_id", "")
    if not target:
        return _err(rid, 4006, "session_id required")
    try:
        cols = int(params.get("cols", 80))
    except (TypeError, ValueError):
        cols = 80

    # ``profile`` (app-global remote mode): resume a session that lives in
    # another local profile's state.db. None/own profile → launch profile.
    profile = (params.get("profile") or "").strip() or None
    profile_home = _profile_home(profile)

    # In a profile scope, open a long-lived db handle bound to that profile.
    # Otherwise reuse the shared launch db.
    if profile_home is not None:
        try:
            from niaharness.services.session_db import SessionDB  # type: ignore

            db = SessionDB(db_path=Path(profile_home) / "state.db")
        except Exception as e:
            return _err(rid, 5000, f"resume failed: profile db unavailable: {e}")
    else:
        db = _get_db()
    if db is None:
        return _err(rid, 5000, "session DB not available")

    found = db.get_session(target)
    if not found:
        # Try title lookup.
        try:
            found = db.get_session_by_title(target)
        except Exception:
            found = None
        if found:
            target = found["id"]
        elif _is_truthy(params.get("lazy", False)) and _child_run_active(target):
            # Race: a watch window opened on a freshly-spawned subagent
            # before its DB row exists. Proceed with empty history; the
            # live mirror streams the whole turn anyway.
            found = {}
        else:
            return _err(rid, 4007, "session not found")

    # Follow the compression-continuation chain to the live tip so a resume
    # on a rotated-out parent id binds to the descendant that actually holds
    # the post-compression turns. Skipped for lazy watch windows.
    if found and not _is_truthy(params.get("lazy", False)):
        try:
            tip = db.resolve_resume_session_id(target)
        except Exception:
            tip = target
        if tip and tip != target:
            target = tip
            try:
                found = db.get_session(target) or found
            except Exception:
                pass

    profile_resume_cwd = (
        str((found or {}).get("cwd") or "").strip()
        or _profile_configured_cwd(profile_home)
        or ""
    )

    def _reuse_live_payload(sid: str, session: dict) -> dict:
        payload = _live_session_payload(
            sid, session, cols=cols, touch=True,
            transport=current_transport() or _stdio_transport,
        )
        payload["resumed"] = target
        # A lazy watch session never owns a run loop — overlay the child-run
        # registry so a reconnecting watch window keeps its busy indicator.
        if session.get("agent") is None and _child_run_active(target):
            payload["running"] = True
            payload["status"] = "streaming"
        return payload

    # Fast path: if the session is already live, reuse it under the lock.
    with _session_resume_lock:
        live = _find_live_session_by_key(target)
        if live is not None:
            return _ok(rid, _reuse_live_payload(*live))

    # Lazy/watch resume: register the live session WITHOUT building an agent.
    # Used by subagent watch windows — the child runs inside the parent's turn.
    if _is_truthy(params.get("lazy", False)):
        sid = uuid4().hex[:8]
        lease = None  # NIA has no active-session lease registry yet.
        try:
            if hasattr(db, "reopen_session"):
                db.reopen_session(target)
            # The child's OWN conversation only (no ancestors).
            if hasattr(db, "get_messages_as_conversation"):
                raw_history = db.get_messages_as_conversation(target)
            else:
                raw_history = _rows_to_history(db.get_messages(target, include_compacted=True))
        except Exception as e:
            return _err(rid, 5000, f"resume failed: {e}")
        cwd = profile_resume_cwd or _default_session_cwd()
        record = _deferred_session_record(
            target, cols=cols, cwd=cwd, history=raw_history, lease=lease,
            source=str(params.get("source") or "tui").strip() or "tui",
            close_on_disconnect=_is_truthy(params.get("close_on_disconnect", False)),
            profile_home=profile_home, lazy=True,
        )
        live = _claim_or_reuse_live(sid, target, record, lease)
        if live is not None:
            return _ok(rid, _reuse_live_payload(*live))
        child_running = _child_run_active(target)
        messages = _history_to_messages(raw_history)
        return _ok(rid, {
            "session_id": sid, "resumed": target,
            "message_count": len(messages), "messages": messages,
            "info": _lazy_resume_info(cwd),
            "inflight": None, "running": child_running,
            "session_key": target,
            "started_at": record["created_at"],
            "status": "streaming" if child_running else "idle",
        })

    # Cold resume default: register the live session and read its stored
    # transcript, but build the agent OFF the response path.
    if not _is_truthy(params.get("eager_build", False)):
        sid = uuid4().hex[:8]
        lease = None
        _enable_gateway_prompts()
        try:
            if hasattr(db, "reopen_session"):
                db.reopen_session(target)
            if hasattr(db, "get_messages_as_conversation"):
                raw_history = db.get_messages_as_conversation(target)
                display_history = db.get_messages_as_conversation(target, include_ancestors=True)
            else:
                raw_history = _rows_to_history(db.get_messages(target, include_compacted=True))
                display_history = raw_history
        except Exception as e:
            return _err(rid, 5000, f"resume failed: {e}")
        # Display keeps the full transcript; the model-fed history is sanitized.
        prefix = display_history[: max(0, len(display_history) - len(raw_history))]
        history = _sanitize_replay_history(raw_history)
        overrides = _stored_session_runtime_overrides(found) or {}
        model_override = overrides.get("model_override") or {}
        cwd = profile_resume_cwd or _default_session_cwd()
        record = _deferred_session_record(
            target, cols=cols, cwd=cwd, history=history, lease=lease,
            source=str(params.get("source") or "tui").strip() or "tui",
            close_on_disconnect=_is_truthy(params.get("close_on_disconnect", False)),
            display_history_prefix=prefix, profile_home=profile_home,
            model_override=overrides.get("model_override"),
            resume_runtime_overrides=overrides or None,
        )
        live = _claim_or_reuse_live(sid, target, record, lease)
        if live is not None:
            return _ok(rid, _reuse_live_payload(*live))
        _schedule_agent_build(sid)
        messages = _history_to_messages(display_history)
        return _ok(rid, {
            "session_id": sid, "resumed": target,
            "message_count": len(messages), "messages": messages,
            "info": _lazy_resume_info(
                cwd,
                model=model_override.get("model") or "",
                provider=overrides.get("provider_override") or "",
            ),
            "inflight": None, "running": False,
            "session_key": target,
            "started_at": record["created_at"], "status": "idle",
        })

    # Eager build path — build the agent OUTSIDE the resume lock.
    sid = uuid4().hex[:8]
    lease = None
    _enable_gateway_prompts()
    home_token = None
    if profile_home is not None:
        # NIA doesn't have set_nia_home_override yet — best-effort no-op.
        # TODO(feature-gap): see FEATURE_GAPS.md (set_nia_home_override).
        pass
    try:
        if hasattr(db, "reopen_session"):
            db.reopen_session(target)
        if hasattr(db, "get_messages_as_conversation"):
            raw_history = db.get_messages_as_conversation(target)
            display_history = db.get_messages_as_conversation(target, include_ancestors=True)
        else:
            raw_history = _rows_to_history(db.get_messages(target, include_compacted=True))
            display_history = raw_history
        display_history_prefix = display_history[: max(0, len(display_history) - len(raw_history))]
        history = _sanitize_replay_history(raw_history)
        messages = _history_to_messages(display_history)
        stored_runtime_overrides = _stored_session_runtime_overrides(found)
        # NIA's _make_agent may not exist yet — fall back to _start_agent_build.
        # TODO(feature-gap): see FEATURE_GAPS.md (_make_agent).
        agent = None
        if hasattr(_modules_self(), "_make_agent"):
            try:
                agent = _make_agent(
                    sid, target, session_id=target, session_db=db,
                    **stored_runtime_overrides,
                )
            except Exception as e:
                return _err(rid, 5000, f"resume failed: agent build: {e}")
    except Exception as e:
        return _err(rid, 5000, f"resume failed: {e}")
    finally:
        if home_token is not None:
            pass  # NIA has no reset_nia_home_override yet.

    # Double-checked locking: another concurrent resume may have won.
    with _session_resume_lock:
        live = _find_live_session_by_key(target)
        if live is not None:
            if agent is not None and hasattr(agent, "close"):
                try:
                    agent.close()
                except Exception:
                    pass
            other_sid, other_session = live
            payload = _live_session_payload(
                other_sid, other_session, cols=cols, touch=True,
                transport=current_transport() or _stdio_transport,
            )
            payload["resumed"] = target
            return _ok(rid, payload)
        # Register the session with the built agent.
        try:
            _init_session(
                sid, target, agent, history,
                cols=cols, cwd=profile_resume_cwd or None, session_db=db,
            )
            if sid in _sessions:
                if stored_runtime_overrides.get("model_override") is not None:
                    _sessions[sid]["model_override"] = stored_runtime_overrides["model_override"]
                _sessions[sid]["display_history_prefix"] = display_history_prefix
                if profile_home is not None:
                    _sessions[sid]["profile_home"] = str(profile_home)
                _sessions[sid]["active_session_lease"] = lease
        except Exception as e:
            return _err(rid, 5000, f"resume failed: init: {e}")
        session = _sessions.get(sid) or {}

    return _ok(rid, {
        "session_id": sid, "resumed": target,
        "message_count": len(messages), "messages": messages,
        "info": _session_info(agent, session) if agent else _lazy_resume_info(profile_resume_cwd or _default_session_cwd()),
        "inflight": None, "running": False,
        "session_key": target,
        "started_at": float(session.get("created_at") or time.time()),
        "status": "idle",
    })


def _rows_to_history(rows: list) -> list[dict]:
    """Convert SessionDB message rows to the history dict shape.

    Helper used by ``session.resume`` when ``get_messages_as_conversation``
    is unavailable. Each row's ``content`` becomes the history ``text``.
    """
    history: list[dict] = []
    for m in rows or []:
        if not isinstance(m, dict):
            continue
        history.append({
            "role": m.get("role", "user"),
            "content": m.get("content") or m.get("text", ""),
        })
    return history


def _sanitize_replay_history(history: list) -> list:
    """Strip a dangling tool-call tail so a killed session doesn't replay it.

    Ported from ``agent.replay_cleanup.sanitize_replay_history``. NIA may not
    have that module — fall back to the raw history. TODO(feature-gap): see
    FEATURE_GAPS.md (agent.replay_cleanup).
    """
    try:
        from niaharness.engine.replay_cleanup import sanitize_replay_history  # type: ignore

        return sanitize_replay_history(history)
    except ImportError:
        return history
    except Exception:
        return history


def _is_truthy(value: Any) -> bool:
    """Hermes-compatible truthy check (accepts "1", "true", "yes", true, 1)."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "y"}
    return bool(value)


def _modules_self():
    """Return the server module itself (for hasattr checks on _make_agent)."""
    import sys
    return sys.modules.get(__name__)



@method("session.cwd.set")
def _session_cwd_set(rid, params):
    """Set the session's cwd with full validation + persistence + git-meta probe.

    Deep-ported from Hermes line 5714. Validates the dir exists (returns 4017
    on failure), persists to the SessionDB, asynchronously probes git branch /
    repo root, and re-registers the cwd as a per-task env override for terminal
    sudo. Emits a session.info refresh so the TUI sidebar updates immediately.
    """
    sid = params.get("session_id", "")
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    if session.get("running"):
        return _err(rid, 4009, "session busy")
    raw = str(params.get("cwd", "") or "").strip()
    if not raw:
        return _err(rid, 4016, "cwd required")
    try:
        cwd = _set_session_cwd(session, raw)
    except ValueError as e:
        return _err(rid, 4017, str(e))
    agent = session.get("agent")
    if agent is not None:
        info = _session_info(agent, session)
    else:
        # Lazy fallback — agent not yet built (composer picked a workspace
        # before the first prompt). Surface cwd + branch so the TUI paints.
        info = {
            "cwd": cwd,
            "branch": _git_branch_for_cwd(cwd),
            "lazy": True,
        }
    _emit("session.info", sid, info)
    return _ok(rid, info)


@method("session.active_list")
def _session_active_list(rid, params):
    return _ok(rid, {
        "sessions": [
            {
                "id": s.get("id", ""),
                "session_key": s.get("session_key", ""),
                "model": s.get("model", ""),
                "cwd": s.get("cwd", ""),
                "started_at": s.get("created_at", 0),
                "last_active": s.get("last_active", 0),
                "message_count": len(s.get("history", [])),
                "status": "working" if s.get("running") else "idle",
                "title": s.get("title", ""),
                "current": s.get("is_active", False),
            }
            for s in _sessions.values()
        ]
    })


@method("session.activate")
def _session_activate(rid, params):
    sid = params.get("session_id", "")
    session = _sessions.get(sid)
    if session is None:
        return _err(rid, 4001, f"session {sid} not found")
    _claim_active_session_slot(sid, session)
    _emit("session.info", sid, _session_info(session.get("agent"), session))
    return _ok(rid, _session_info(session.get("agent"), session))


@method("session.delete")
def _session_delete(rid, params):
    """Delete a session — close in-memory + delete from DB.

    Deep-ported from Hermes line 5933. Closes the live session, then
    deletes the DB row. If the DB supports ``delete_session_if_empty``,
    only deletes when there are no messages (so accidental deletes of
    active sessions don't lose history).
    """
    sid = params.get("session_id", "")
    session = _sessions.get(sid)
    if session is not None:
        if session.get("running"):
            return _err(rid, 4009, "session busy — /interrupt the current turn before /delete")
    _close_session_by_id(sid, end_reason="deleted")
    db = _get_db()
    if db:
        try:
            key = session.get("session_key", sid) if session else sid
            if hasattr(db, "delete_session_if_empty"):
                db.delete_session_if_empty(key)
            else:
                db.delete_session(key)
        except Exception:
            pass
    return _ok(rid, {"deleted": sid})


@method("session.title")
def _session_title(rid, params):
    """Get or set a session's title with full pending-title + DB-row lifecycle.

    Deep-ported from Hermes line 5975. Handles:
    - GET (no ``title`` param): resolve from DB, fall back to pending_title,
      promote pending → live if the DB row exists.
    - SET (``title`` param present): validate non-empty, attempt DB write,
      fall back to ensure-row-then-write if the row doesn't exist yet, fall
      back to pending_title queue if the DB is unavailable. ValueError from
      the DB (invalid/duplicate title) → 4022 error code.
    """
    sid = params.get("session_id", "")
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    db = _get_db()
    if db is None:
        return _err(rid, 5007, "session DB not available")
    key = session.get("session_key", sid)

    # GET path — no "title" param.
    if "title" not in params:
        fallback = session.get("pending_title") or ""
        try:
            resolved_title = db.get_session_title(key) or ""
            if fallback:
                # Try to promote pending → live.
                if db.set_session_title(key, fallback):
                    session["pending_title"] = None
                    resolved_title = fallback
                else:
                    # rowcount==0 could mean "same value" or "missing row".
                    existing_row = db.get_session(key)
                    existing_title = ((existing_row or {}).get("title") or "").strip()
                    if existing_title == fallback:
                        session["pending_title"] = None
                        resolved_title = fallback
                    elif not resolved_title:
                        resolved_title = fallback
            elif resolved_title:
                session["pending_title"] = None
        except Exception:
            resolved_title = fallback
        _emit("session.info", sid, _session_info(session.get("agent"), session))
        return _ok(rid, {"title": resolved_title, "session_key": key})

    # SET path.
    title = (params.get("title", "") or "").strip()
    if not title:
        return _err(rid, 4021, "title required")
    try:
        if db.set_session_title(key, title):
            session["pending_title"] = None
            session["title"] = title
            _emit("session.info", sid, _session_info(session.get("agent"), session))
            return _ok(rid, {"pending": False, "title": title})
        # rowcount==0 — either same value or missing row.
        existing_row = db.get_session(key)
        if existing_row:
            session["pending_title"] = None
            _emit("session.info", sid, _session_info(session.get("agent"), session))
            return _ok(
                rid,
                {"pending": False, "title": (existing_row.get("title") or title)},
            )
        # No row yet — the DB write is deferred to the first prompt so empty
        # drafts don't litter the sidebar. An explicit /title is clear user
        # intent — persist the row NOW and set the title. Mirrors Hermes
        # line 6042.
        _ensure_session_db_row(session)
        with _session_db(session) as scoped_db:
            if scoped_db is not None and scoped_db.set_session_title(key, title):
                session["pending_title"] = None
                session["title"] = title
                _emit("session.info", sid, _session_info(session.get("agent"), session))
                return _ok(rid, {"pending": False, "title": title})
        # Row creation didn't take — fall back to queuing so the post-turn
        # apply block can still recover.
        session["pending_title"] = title
        _emit("session.info", sid, _session_info(session.get("agent"), session))
        return _ok(rid, {"pending": True, "title": title})
    except ValueError as e:
        return _err(rid, 4022, str(e))
    except Exception as e:
        return _err(rid, 5007, str(e))


@method("session.status")
def _session_status(rid, params):
    """Return live session status with inflight + running info.

    Deep-ported from Hermes line 7560. Uses _session_live_status for
    the status field (waiting/starting/working/idle).
    """
    sid = params.get("session_id", "")
    session = _sessions.get(sid)
    if session is None:
        return _err(rid, 4001, f"session {sid} not found")
    _claim_active_session_slot(sid, session)
    inflight = _inflight_snapshot(session)
    status = _session_live_status(sid, session)
    return _ok(rid, {
        **_session_info(session.get("agent"), session),
        "id": sid,
        "inflight": inflight,
        "running": session.get("running", False),
        "status": status,
    })


@method("session.history")
def _session_history(rid, params):
    sid = params.get("session_id", "")
    session = _sessions.get(sid)
    if session is not None:
        # Return in-memory history.
        with session["history_lock"]:
            history = list(session.get("history", []))
        return _ok(rid, {
            "messages": [
                {"role": m.get("role", "user"), "text": m.get("text", "")}
                for m in history
            ]
        })
    # Fall back to DB.
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
    sid = params.get("session_id", "")
    message_id = params.get("message_id")
    if message_id is None:
        return _err(rid, 4004, "message_id is required")
    session = _sessions.get(sid)
    if session is None:
        return _err(rid, 4001, f"session {sid} not found")
    # Undo in in-memory history.
    with session["history_lock"]:
        history = session.get("history", [])
        user_indices = [i for i, m in enumerate(history) if m.get("role") == "user"]
        try:
            ordinal = int(message_id)
        except (TypeError, ValueError):
            return _err(rid, 4004, "message_id must be an integer")
        if ordinal < 0 or ordinal >= len(user_indices):
            return _err(rid, 4018, "target user message is no longer in session history")
        truncated = history[:user_indices[ordinal]]
        session["history"] = truncated
        session["history_version"] = session.get("history_version", 0) + 1
    # Persist to DB.
    db = _get_db()
    if db:
        try:
            db.replace_messages(sid, truncated)
        except Exception:
            pass
    return _ok(rid, {"removed": len(history) - len(truncated)})


@method("session.compress")
def _session_compress(rid, params):
    sid = params.get("session_id", "")
    session, err = _sess(params, rid)
    if err:
        return err
    _status_update(sid, "compacting", "Summarizing conversation...")
    try:
        from niaharness.engine.llm_compaction import LLMCompactor, CompactionRequest
        compactor = LLMCompactor()
        with session["history_lock"]:
            history = list(session.get("history", []))
        # Convert to ConversationMessage format.
        from niaharness.engine.messages import ConversationMessage, TextBlock
        messages = []
        for m in history:
            role = m.get("role", "user")
            text = m.get("text", "")
            if role == "user":
                messages.append(ConversationMessage.from_user_text(text))
            else:
                messages.append(ConversationMessage(role=role, content=[TextBlock(text=text)]))
        request = CompactionRequest(
            messages=messages,
            context_window=32000,
            force=True,
        )
        import asyncio
        result = asyncio.run(compactor.compact(request))
        if result.success:
            # Replace history with compacted version.
            with session["history_lock"]:
                session["history"] = [{"role": "assistant", "text": result.summary}]
                session["history_version"] = session.get("history_version", 0) + 1
            _status_update(sid, "ready")
            return _ok(rid, {
                "success": True,
                "method": result.method,
                "after_messages": 1,
                "before_messages": len(history),
            })
        _status_update(sid, "ready")
        return _ok(rid, {"success": False, "error": result.error})
    except Exception as exc:
        _status_update(sid, "ready")
        return _err(rid, 5001, f"compress failed: {exc}")


@method("session.save")
def _session_save(rid, params):
    """Persist the session's history + metadata to the DB.

    Deep-ported from Hermes line 7764. Flushes all messages to the
    SessionDB, updates the session's cwd + title, and returns the
    persisted session key.
    """
    sid = params.get("session_id", "")
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    db = _get_db()
    if db is None:
        return _err(rid, 5006, "session DB not available")
    key = session.get("session_key", sid)
    try:
        # Ensure the row exists.
        _ensure_session_db_row(session)
        # Persist all messages.
        with session["history_lock"]:
            history = list(session.get("history", []))
        for msg in history:
            role = msg.get("role", "user")
            content = msg.get("text") or msg.get("content", "")
            if role in {"user", "assistant", "system"}:
                try:
                    db.add_message(key, role, str(content))
                except Exception:
                    pass
        # Persist cwd if explicit.
        if session.get("explicit_cwd"):
            try:
                with _session_db(session) as scoped_db:
                    if scoped_db is not None and hasattr(scoped_db, "update_session_cwd"):
                        scoped_db.update_session_cwd(key, session["cwd"])
            except Exception:
                pass
        # Persist title if pending.
        pending_title = session.get("pending_title")
        if pending_title:
            try:
                if db.set_session_title(key, pending_title):
                    session["pending_title"] = None
                    session["title"] = pending_title
            except Exception:
                pass
        return _ok(rid, {"saved": True, "session_id": sid, "session_key": key, "message_count": len(history)})
    except Exception as exc:
        return _err(rid, 5007, f"session.save failed: {exc}")


@method("session.close")
def _session_close(rid, params):
    sid = params.get("session_id", "")
    _close_session_by_id(sid, end_reason="closed")
    return _ok(rid, {"closed": True, "session_id": sid})


@method("session.branch")
def _session_branch(rid, params):
    """Branch a session — copy history + link to parent.

    Deep-ported from Hermes line 7831. Creates a new session that
    inherits the parent's history (shallow copy), links back via
    parent_session_id, and persists the branch seed to the DB so resume
    picks up the full context.
    """
    sid = params.get("session_id", "")
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    if session.get("running"):
        return _err(rid, 4009, "session busy — /interrupt the current turn before /branch")

    new_sid = uuid4().hex[:8]
    new_key = _new_session_key()
    now = time.time()

    # Copy history for the branch.
    with session["history_lock"]:
        branch_history = [dict(m) for m in session.get("history", [])]

    with _sessions_lock:
        _sessions[new_sid] = {
            "agent": None,
            "agent_error": None,
            "agent_ready": threading.Event(),
            "agent_build_started": False,
            "active_session_lease": None,
            "attached_images": [],
            "close_on_disconnect": False,
            "cols": session.get("cols", 80),
            "created_at": now,
            "cwd": session.get("cwd", _default_session_cwd()),
            "display_history_prefix": [],
            "edit_snapshots": {},
            "explicit_cwd": session.get("explicit_cwd", False),
            "history": branch_history,
            "history_lock": threading.Lock(),
            "history_version": 0,
            "id": new_sid,
            "image_counter": 0,
            "inflight_turn": None,
            "last_active": now,
            "model_override": session.get("model_override"),
            "parent_session_id": sid,
            "pending_title": None,
            "profile_home": session.get("profile_home"),
            "running": False,
            "session_key": new_key,
            "show_reasoning": session.get("show_reasoning", _load_show_reasoning()),
            "slash_worker": None,
            "source": session.get("source", "tui"),
            "tool_progress_mode": session.get("tool_progress_mode", _load_tool_progress_mode()),
            "tool_started_at": {},
            "transport": current_transport() or _stdio_transport,
            "queued_prompt": None,
        }
        _register_session_cwd(_sessions[new_sid])

    # Schedule a deferred agent build for the branch.
    _schedule_agent_build(new_sid)

    # Emit a session.info for the new branch.
    _emit("session.info", new_sid, _session_info(None, _sessions.get(new_sid, {})))

    return _ok(rid, {
        "session_id": new_sid,
        "stored_session_id": new_key,
        "parent_session_id": sid,
        "message_count": len(branch_history),
        "messages": _history_to_messages(branch_history),
    })


@method("session.interrupt")
def _session_interrupt(rid, params):
    """Interrupt the current turn + clear pending prompts.

    Deep-ported from Hermes line 7902. Calls agent.interrupt(), clears
    any pending clarify/sudo/secret prompts for this session, marks the
    turn as cancelled, and handles the stuck-running-flag recovery path
    (a turn that died without clearing running).
    """
    sid = params.get("session_id", "")
    session = _sessions.get(sid)
    if session is None:
        return _err(rid, 4001, f"session {sid} not found")

    agent = session.get("agent")
    if agent and hasattr(agent, "interrupt"):
        try:
            agent.interrupt()
        except Exception:
            pass

    # Clear pending prompts for this session only.
    _clear_pending(sid)

    # Mark turn as cancelled.
    session["_turn_cancel_requested"] = True

    # Stuck-running recovery: if the run thread is dead but running is
    # still True (turn died without clearing it), force-clear it.
    run_thread = session.get("_run_thread")
    if run_thread is not None and not run_thread.is_alive():
        with session["history_lock"]:
            session["running"] = False
            _clear_inflight_turn(session)

    _emit("session.interrupted", sid, {})
    return _ok(rid, {"ok": True, "session_id": sid})


@method("session.steer")
def _session_steer(rid, params):
    sid = params.get("session_id", "")
    text = params.get("text", "")
    session = _sessions.get(sid)
    if session is None:
        return _err(rid, 4001, f"session {sid} not found")
    agent = session.get("agent")
    if agent and hasattr(agent, "steer"):
        try:
            accepted = agent.steer(text)
            return _ok(rid, {"status": "queued" if accepted else "rejected", "text": text})
        except Exception as exc:
            return _err(rid, 5001, f"steer failed: {exc}")
    return _ok(rid, {"status": "rejected", "text": text})


@method("session.usage")
def _session_usage(rid, params):
    sid = params.get("session_id", "")
    session = _sessions.get(sid)
    if session is None:
        return _err(rid, 4001, f"session {sid} not found")
    agent = session.get("agent")
    if agent:
        return _ok(rid, _get_usage(agent))
    db = _get_db()
    if db:
        try:
            row = db.get_session(sid)
            if row:
                return _ok(rid, {
                    "model": row.get("model", ""),
                    "input": row.get("input_tokens", 0),
                    "output": row.get("output_tokens", 0),
                    "calls": row.get("api_call_count", 0),
                    "total": (row.get("input_tokens", 0) or 0) + (row.get("output_tokens", 0) or 0),
                })
        except Exception:
            pass
    return _ok(rid, {"input": 0, "output": 0, "total": 0, "calls": 0})


@method("session.context_breakdown")
def _session_context_breakdown(rid, params):
    sid = params.get("session_id", "")
    session = _sessions.get(sid)
    if session is None:
        return _ok(rid, {"categories": {}})
    with session["history_lock"]:
        history = session.get("history", [])
    system_tokens = 0
    user_tokens = 0
    assistant_tokens = 0
    tool_tokens = 0
    for msg in history:
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
# Prompt submission helpers (deep-ported from Hermes tui_gateway/server.py)
# ---------------------------------------------------------------------------


def _session_source(session: dict | None) -> str:
    """Return the session's source tag ('tui' by default).

    Ported from Hermes line 1544. Used by ``_set_session_context`` so the
    session-context vars carry the right source label.
    """
    if session:
        source = str(session.get("source") or "").strip()
        if source:
            return source
    return "tui"


def _cwd_for_session_key(session_key: str) -> str:
    """Reverse-map ``session_key`` → ``session['cwd']``.

    Ported from Hermes line 1868. Snapshots ``_sessions`` first because
    concurrent RPC handlers mutate it from the thread pool.
    """
    if not session_key:
        return ""
    with _sessions_lock:
        for sess in list(_sessions.values()):
            if sess.get("session_key") == session_key:
                return str(sess.get("cwd") or "")
    return ""


def _terminal_task_cwd(session: dict | None) -> str:
    """Return the cwd that the terminal tool should use for this TUI session.

    Ported from Hermes line 1435. For non-local terminal backends (SSH/container)
    the cwd may not exist on the local host; honor the explicit override.
    """
    backend = (os.environ.get("TERMINAL_ENV") or "").strip().lower()
    if backend and backend != "local":
        raw = os.environ.get("TERMINAL_CWD", "").strip()
        if not raw:
            try:
                terminal_cfg = _load_cfg().get("terminal", {})
                if isinstance(terminal_cfg, dict):
                    raw = str(terminal_cfg.get("cwd") or "").strip()
            except Exception:
                raw = ""
        if raw and raw not in {".", "auto", "cwd"}:
            return raw
    return _session_cwd(session)


def _set_session_context(session_key: str, cwd: str | None = None) -> list:
    """Bind session-context vars (session_key, source, cwd) for this turn.

    Ported from Hermes line 1884. Returns an opaque token list that
    ``_clear_session_context`` later resets in the finally block. NIA does NOT
    yet have ``niaharness.gateway.session_context`` — guard with try/except so
    the rest of the turn runs cleanly. TODO(feature-gap): see FEATURE_GAPS.md.
    """
    try:
        from niaharness.gateway.session_context import set_session_vars  # type: ignore
    except ImportError:
        # TODO(feature-gap): see FEATURE_GAPS.md (gateway.session_context).
        return []
    try:
        resolved = cwd if cwd is not None else _cwd_for_session_key(session_key)
        source = "tui"
        with _sessions_lock:
            for sess in list(_sessions.values()):
                if sess.get("session_key") == session_key:
                    source = _session_source(sess)
                    break
        return set_session_vars(session_key=session_key, source=source, cwd=resolved)
    except Exception:
        return []


def _clear_session_context(tokens: list) -> None:
    """Release the session-context vars bound by ``_set_session_context``.

    Ported from Hermes line 1904. Best-effort; never raises.
    """
    if not tokens:
        return
    try:
        from niaharness.gateway.session_context import clear_session_vars  # type: ignore
    except ImportError:
        # TODO(feature-gap): see FEATURE_GAPS.md (gateway.session_context).
        return
    try:
        clear_session_vars(tokens)
    except Exception:
        pass


def _register_session_cwd(session: dict | None) -> None:
    """Register the session's cwd as a per-task env override in terminal_tool.

    Ported from Hermes line 1552. Lets terminal sudo / async tasks inherit the
    TUI session's workspace instead of falling back to the launch dir. NIA's
    terminal tool does not yet expose ``register_task_env_overrides`` — guard
    with try/except. TODO(feature-gap): see FEATURE_GAPS.md.
    """
    if not session:
        return
    try:
        from niaharness.tools.terminal_tool import register_task_env_overrides  # type: ignore
    except ImportError:
        # TODO(feature-gap): see FEATURE_GAPS.md (tools.terminal_tool).
        return
    try:
        register_task_env_overrides(
            session["session_key"], {"cwd": _terminal_task_cwd(session)}
        )
    except Exception:
        pass


def _config_model_target() -> tuple[str, str]:
    """(model, provider) currently selected by config.yaml — and ONLY config.

    Ported from Hermes line 1996. Does NOT consult env vars (``NIA_MODEL`` /
    ``HERMES_MODEL``) — those are launch-scoped seeds that must not be replayed
    as a /model switch on every turn.
    """
    cfg_model = _load_cfg().get("model")
    model = ""
    provider = ""
    if isinstance(cfg_model, dict):
        model = str(cfg_model.get("default", "") or "").strip()
        provider = str(cfg_model.get("provider") or "").strip()
        if provider.lower() == "auto":
            provider = ""
    elif isinstance(cfg_model, str):
        model = cfg_model.strip()
    return model, provider


def _apply_model_switch(
    sid: str,
    session: dict,
    raw_input: str,
    *,
    confirm_expensive_model: bool = False,
    pin_session_override: bool = True,
    parsed_flags: tuple[str, str, bool, bool, bool] | None = None,
    persist_override: bool | None = None,
) -> dict:
    """Switch the live agent's model and persist the override.

    Ported from Hermes line 2662. NIA's QueryEngine does not yet expose
    ``switch_model`` / the Hermes model_switch pipeline, so this is a
    best-effort port: we record the override on the session and emit a
    session.info refresh. Full provider/key/base_url resolution is a
    feature gap. TODO(feature-gap): see FEATURE_GAPS.md.
    """
    # Parse "<model> --provider <provider>" if present.
    model_input = str(raw_input or "").strip()
    explicit_provider = ""
    if "--provider" in model_input:
        parts = model_input.split("--provider", 1)
        model_input = parts[0].strip()
        explicit_provider = parts[1].strip()

    if not model_input:
        raise ValueError("model value required")

    agent = session.get("agent")
    new_model = model_input
    new_provider = explicit_provider or (
        getattr(agent, "provider", "") if agent else ""
    )

    # Try an in-place agent switch if the agent supports it.
    if agent is not None and hasattr(agent, "switch_model"):
        try:
            agent.switch_model(
                new_model=new_model,
                new_provider=new_provider,
            )
        except Exception as exc:
            raise ValueError(
                f"Model switch to {new_model} failed ({exc}); staying on "
                f"{getattr(agent, 'model', new_model)}."
            ) from exc

    if pin_session_override and isinstance(session, dict):
        session["model_override"] = {
            "model": new_model,
            "provider": new_provider,
            "base_url": getattr(agent, "base_url", "") if agent else "",
            "api_key": getattr(agent, "api_key", "") if agent else "",
            "api_mode": getattr(agent, "api_mode", "") if agent else "",
        }

    if agent is not None:
        try:
            _emit("session.info", sid, _session_info(agent, session))
        except Exception:
            pass

    return {
        "value": new_model,
        "warning": "",
        "confirm_required": False,
    }


def _sync_agent_model_with_config(sid: str, session: dict) -> None:
    """Adopt a config.yaml model change at turn start.

    Ported from Hermes line 2852. Sessions pinned with /model keep their
    choice; a failed switch keeps the current model and never blocks the turn.
    """
    agent = session.get("agent")
    if agent is None or session.get("model_override"):
        return
    target = _config_model_target()
    if not target[0]:
        return
    seen = session.get("config_model_seen")
    # Record first so a broken config gets one attempt per edit, not per turn.
    session["config_model_seen"] = target
    if target == seen:
        return
    model, provider = target
    # Already running the configured model — adopt without a redundant switch.
    if model == getattr(agent, "model", "") and (
        not provider or provider == getattr(agent, "provider", "")
    ):
        return
    raw = f"{model} --provider {provider}" if provider else model
    try:
        _apply_model_switch(
            sid,
            session,
            raw,
            confirm_expensive_model=True,
            pin_session_override=False,
            persist_override=False,
        )
    except Exception as e:
        _emit(
            "error",
            sid,
            {"message": f"Could not switch to configured model {model}: {e}"},
        )


def _transfer_active_session_slot(
    sid: str, session: dict, *, new_session_id: str,
) -> bool:
    """Re-anchor the active-session lease to a new session_id.

    Ported from Hermes line 442. NIA does not yet have a hermes_cli.active_sessions
    lease registry — the simple in-process ``_active_session_sid`` slot is
    session-id-agnostic, so we just return True (no transfer needed).
    TODO(feature-gap): see FEATURE_GAPS.md.
    """
    if not new_session_id:
        return False
    lease = session.get("active_session_lease")
    if lease is None:
        return True
    # NIA has no lease registry yet — best-effort no-op success.
    return True


def _restart_slash_worker(sid: str, session: dict) -> None:
    """Tear down and respawn the slash worker for a session.

    Ported from Hermes line 2620. Used after a model switch / compression
    session_key rotation so the worker targets the live session.
    """
    worker = _slash_workers.pop(sid, None)
    if worker is not None:
        try:
            worker.close()
        except Exception:
            pass
    try:
        new_worker = _SlashWorker(
            session.get("session_key", sid),
            getattr(session.get("agent"), "model", "") or _resolve_model(),
        )
        _slash_workers[sid] = new_worker
    except Exception:
        pass


def _sync_session_key_after_compress(
    sid: str,
    session: dict,
    *,
    clear_pending_title: bool = True,
    restart_slash_worker: bool = True,
) -> None:
    """Re-anchor session_key when the agent rotates session_id after compression.

    Ported from Hermes line 2952. The agent's compression path may end the
    current SessionDB row and start a continuation; without this sync, the
    gateway-side session_key would keep targeting the ended row.
    """
    agent = session.get("agent")
    new_session_id = getattr(agent, "session_id", None) or ""
    old_key = session.get("session_key", "") or ""
    if not new_session_id or new_session_id == old_key:
        return

    _transfer_active_session_slot(sid, session, new_session_id=new_session_id)

    try:
        from niaharness.permissions.approval import (
            disable_session_yolo,
            enable_session_yolo,
            is_session_yolo_enabled,
            register_gateway_notify,
            unregister_gateway_notify,
        )

        try:
            unregister_gateway_notify(old_key)
        except Exception:
            pass
        session["session_key"] = new_session_id
        try:
            yolo_was_on = is_session_yolo_enabled(old_key)
        except Exception:
            yolo_was_on = False
        if yolo_was_on:
            try:
                enable_session_yolo(new_session_id)
                disable_session_yolo(old_key)
            except Exception:
                pass
        try:
            register_gateway_notify(
                new_session_id,
                lambda data: _emit_approval_request(sid, data),
            )
        except Exception:
            pass
    except Exception:
        # Even if the approval module fails, still anchor the session_key.
        session["session_key"] = new_session_id

    if clear_pending_title:
        session["pending_title"] = None
    if restart_slash_worker:
        try:
            _restart_slash_worker(sid, session)
        except Exception:
            pass


def _persist_branch_seed(session: dict) -> None:
    """First-turn persist of a branch's copied transcript.

    Ported from Hermes line 1673. A branch is a draft until its first submit:
    the parent's messages live only in ``session['history']``. Without this
    the branch row would resume missing its pre-branch context.
    """
    if not session.get("parent_session_id") or session.get("_branch_seed_persisted"):
        return
    key = session.get("session_key")
    if not key:
        return
    with session["history_lock"]:
        seed = [dict(msg) for msg in (session.get("history") or [])]
    if not seed:
        return
    db = _get_db()
    if db is None:
        return
    try:
        for msg in seed:
            db.add_message(
                session_id=key,
                role=str(msg.get("role", "user") or "user"),
                content=str(
                    msg.get("text") if msg.get("text") is not None
                    else msg.get("content", "")
                ),
            )
        session["_branch_seed_persisted"] = True
    except Exception:
        logger.debug("branch seed persist failed", exc_info=True)


# ---------------------------------------------------------------------------
# Session DB context manager + git metadata helpers
# (deep-ported from Hermes tui_gateway/server.py lines 1702-1762, 1765-1788)
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _session_db(session: dict):
    """Yield the SessionDB that owns this session's row (profile-aware).

    Ported from Hermes line 1702. A remote/profile session persists into its
    own profile's ``state.db`` (a fresh handle we close on exit); everything
    else borrows the shared ``_get_db()`` handle (left open). Yields None when
    the db is unavailable. NIA does not yet have profile-scoped state.db —
    the profile_home branch is guarded so it falls back to the shared db.
    TODO(feature-gap): see FEATURE_GAPS.md (profile-scoped state.db).
    """
    db, close_db = None, False
    profile_home = session.get("profile_home")
    if profile_home:
        try:
            from niaharness.services.session_db import SessionDB  # type: ignore

            db = SessionDB(db_path=Path(profile_home) / "state.db")
            close_db = True
        except Exception:
            logger.debug("failed to open profile db for session", exc_info=True)
            db = None
    else:
        db = _get_db()
    try:
        yield db
    finally:
        if close_db and db is not None:
            with contextlib.suppress(Exception):
                db.close()


def _git_branch_for_cwd(cwd: str) -> str:
    """Return the git branch name for ``cwd`` (best-effort, no raise).

    Ported from Hermes (used inline at multiple call sites). Wraps
    ``git_probe.branch`` so callers don't need to import the module directly.
    Returns "" when cwd is empty or git is unavailable.
    """
    if not cwd:
        return ""
    try:
        return git_probe.branch(cwd) or ""
    except Exception:
        return ""


def _git_common_repo_root_for_cwd(cwd: str) -> str:
    """Return the git repo root for ``cwd`` (best-effort, no raise).

    Ported from Hermes (used inline at multiple call sites). Wraps
    ``git_probe.repo_root`` so callers don't need to import the module directly.
    Returns "" when cwd is empty or git is unavailable.
    """
    if not cwd:
        return ""
    try:
        return git_probe.repo_root(cwd) or ""
    except Exception:
        return ""


def _persist_session_git_meta(session: dict, cwd: str) -> None:
    """Resolve + persist a session's git branch / repo root WITHOUT blocking.

    Ported from Hermes line 1730. Branch and root come from ``git`` subprocess
    probes; running them inline on the session-init / cwd-set path would stall
    startup whenever ``cwd`` is slow or on an unreachable mount. Run them on a
    short-lived daemon thread instead and persist via the same profile-aware
    db the caller writes ``cwd`` to.

    Best-effort: ``cwd`` itself is persisted synchronously by the caller, so a
    probe failure just leaves these enrichment columns unset (the project tree
    falls back to its live resolver / lazy backfill). Daemon, so a mid-flight
    probe never delays gateway shutdown.
    """
    session_key = session.get("session_key", "")
    if not session_key or not cwd:
        return
    # Snapshot the routing fields now; the live session dict may be gone by the
    # time the thread runs. `_session_db` reopens the profile-correct db inside.
    db_session = {"session_key": session_key, "profile_home": session.get("profile_home")}

    def _run() -> None:
        try:
            branch = _git_branch_for_cwd(cwd)
            root = _git_common_repo_root_for_cwd(cwd)
            if not (branch or root):
                return
            with _session_db(db_session) as db:
                if db is not None and hasattr(db, "update_session_cwd"):
                    try:
                        db.update_session_cwd(session_key, cwd, branch, root)
                    except Exception:
                        logger.debug("update_session_cwd failed", exc_info=True)
        except Exception:
            logger.debug("failed to persist session git metadata", exc_info=True)

    threading.Thread(target=_run, name="git-meta", daemon=True).start()


def _set_session_cwd(session: dict, cwd: str) -> str:
    """Set + persist the session's cwd with full side-effects.

    Ported from Hermes line 1765. Validates the dir exists, registers the cwd
    as a per-task env override for terminal sudo, persists to the SessionDB,
    and asynchronously probes + persists git branch / repo root.

    Raises ``ValueError`` if the dir doesn't exist (so callers can return a
    4017 to the client, matching Hermes error code).
    """
    resolved = os.path.abspath(os.path.expanduser(str(cwd)))
    if not os.path.isdir(resolved):
        raise ValueError(f"working directory does not exist: {cwd}")
    session["cwd"] = resolved
    # An explicit user choice — persist it as the workspace (and let a later
    # lazy row creation persist it too, not the launch-dir fallback).
    session["explicit_cwd"] = True
    _register_session_cwd(session)
    with _session_db(session) as db:
        if db is not None and hasattr(db, "update_session_cwd"):
            try:
                db.update_session_cwd(session.get("session_key", ""), resolved)
            except Exception:
                logger.debug("failed to persist session cwd", exc_info=True)
    # Branch/repo-root probes are git subprocesses — capture them off the hot path.
    _persist_session_git_meta(session, resolved)
    # NIA does not yet expose tools.terminal_tool.cleanup_vm — guard so the
    # rest of the cwd-set flow still runs. TODO(feature-gap): see FEATURE_GAPS.md.
    try:
        from niaharness.tools.terminal_tool import cleanup_vm  # type: ignore

        cleanup_vm(session["session_key"])
    except ImportError:
        pass
    except Exception:
        pass
    return resolved


def _child_run_active(child_key: str) -> bool:
    """True if a subagent run for ``child_key`` is currently live.

    Ported from Hermes line 3618. Used by ``prompt.submit`` to reject
    mid-watch-subagent prompts with a clear "subagent still running" error
    rather than racing the in-flight child.
    """
    ts = _active_child_runs.get(child_key)
    return ts is not None and (time.time() - ts) < _CHILD_RUN_STALE_S


# ---------------------------------------------------------------------------
# Session resume helpers (deep-ported from Hermes tui_gateway/server.py)
# ---------------------------------------------------------------------------


def _content_display_text(content: Any) -> str:
    """Render ``content`` (str / list / dict) as plain display text.

    Ported from Hermes line 4611. Used by ``_message_preview`` to extract
    a short text preview from history messages whose content may be a
    multimodal list of parts.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (int, float)):
        return str(content)
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            text = _content_display_text(part).strip()
            if text:
                parts.append(text)
        return "\n".join(parts)
    if isinstance(content, dict):
        kind = content.get("type")
        if kind in {"text", "input_text", "output_text"}:
            return str(content.get("text") or content.get("content") or "")
        if kind in {"image_url", "input_image", "image"}:
            return "[image]"
        if kind in {"input_audio", "audio"}:
            return "[audio]"
        if kind:
            return f"[{kind}]"
        if "text" in content:
            return str(content.get("text") or "")
        return "[structured content]"
    return str(content)


def _coerce_message_text(content: Any) -> str:
    """Render ``message['content']`` as a plain string for transport.

    Ported from Hermes line 4641. Like ``_content_display_text`` but
    preserves image URLs inline so the desktop renderer can pull them
    back out via ``extractEmbeddedImages``.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (int, float)):
        return str(content)
    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            if isinstance(part, str):
                chunks.append(part)
                continue
            if not isinstance(part, dict):
                continue
            text = part.get("text")
            if isinstance(text, str):
                chunks.append(text)
                continue
            kind = part.get("type")
            if kind in {"text", "input_text", "output_text"}:
                t = part.get("text") or part.get("content") or ""
                if t:
                    chunks.append(str(t))
                continue
            if kind in {"image_url", "input_image", "image"}:
                image_url = part.get("image_url")
                url = ""
                if isinstance(image_url, dict):
                    candidate = image_url.get("url")
                    if isinstance(candidate, str):
                        url = candidate
                elif isinstance(image_url, str):
                    url = image_url
                if url:
                    chunks.append(url)
                continue
        return "\n".join(chunks)
    if isinstance(content, dict):
        return _content_display_text(content)
    return str(content)


def _history_to_messages(history: list[dict]) -> list[dict]:
    """Convert raw history rows to the TUI message-list shape.

    Ported from Hermes line 4728. Strips empty turns, preserves reasoning
    blocks on assistant turns so the "Thinking…" disclosure still renders,
    and collapses tool turns to ``{role, name, context}``.
    """
    messages: list[dict] = []
    tool_call_args: dict[str, tuple[str, dict]] = {}

    for m in history:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role not in {"user", "assistant", "tool", "system"}:
            continue
        content_text = _coerce_message_text(m.get("content", m.get("text", "")))
        if role == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                tc_id = tc.get("id", "") if isinstance(tc, dict) else ""
                if tc_id and fn.get("name"):
                    try:
                        args = json.loads(fn.get("arguments", "{}"))
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                    tool_call_args[tc_id] = (fn["name"], args)
            if not content_text.strip():
                continue
        if role == "tool":
            tc_id = m.get("tool_call_id", "")
            tc_info = tool_call_args.get(tc_id) if tc_id else None
            name = (tc_info[0] if tc_info else None) or m.get("tool_name") or "tool"
            args = (tc_info[1] if tc_info else None) or {}
            messages.append(
                {"role": "tool", "name": name, "context": _tool_ctx(name, args)}
            )
            continue
        # Preserve reasoning-only assistant turns so the "Thinking…" block
        # still renders. Fixes Hermes #44022.
        reasoning_keys = (
            "reasoning",
            "reasoning_content",
            "reasoning_details",
            "codex_reasoning_items",
        )
        has_reasoning = role == "assistant" and any(
            m.get(key) for key in reasoning_keys
        )
        if not content_text.strip() and not has_reasoning:
            continue
        msg = {"role": role, "text": content_text}
        if role == "assistant":
            for key in reasoning_keys:
                if key in m and m.get(key) is not None:
                    msg[key] = m.get(key)
        messages.append(msg)

    return messages


def _coerce_seed_history(value: Any) -> list[dict]:
    """Coerce the ``messages`` param on session.create to a clean history list.

    Ported from Hermes line 4788. Strips non-dict items, rejects roles other
    than user/assistant/system, and requires non-empty string content.
    """
    if not isinstance(value, list):
        return []
    history: list[dict] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        if role not in ("user", "assistant", "system"):
            continue
        content = item.get("content")
        if content is None:
            content = item.get("text")
        if not isinstance(content, str) or not content.strip():
            continue
        history.append({"role": role, "content": content})
    return history


def _session_lookup_key(session: dict, *, fallback: str = "") -> str:
    """Return the canonical lookup key for a session.

    Ported from Hermes line 5804. Prefers the live agent's session_id (which
    may have rotated after compression), then the session_key, then a fallback.
    """
    agent = session.get("agent")
    return str(
        getattr(agent, "session_id", None)
        or session.get("session_key")
        or fallback
        or ""
    )


def _find_live_session_by_key(session_key: str) -> tuple[str, dict] | None:
    """Find a live (non-finalized) session by its lookup key.

    Ported from Hermes line 5814. Returns ``(sid, session)`` or ``None``.
    """
    if not session_key:
        return None
    for sid, session in list(_sessions.items()):
        if session.get("_finalized"):
            continue
        if _session_lookup_key(session, fallback=sid) == session_key:
            return sid, session
    return None


def _session_pending_kind(sid: str) -> str:
    """Return the kind of pending prompt for ``sid`` ("" if none).

    Ported from Hermes line 5738. Used by ``_session_live_status`` to surface
    "waiting" when a session is blocked on a clarify / sudo / secret prompt.
    """
    with _prompt_lock:
        for rid, (owner_sid, _ev) in list(_pending.items()):
            if owner_sid != sid:
                continue
            event, _payload = _pending_prompt_payloads.get(rid, ("input.request", {}))
            return str(event).removesuffix(".request")
    return ""


def _session_live_status(sid: str, session: dict) -> str:
    """Return the live status string for a session.

    Ported from Hermes line 5747. One of: ``waiting`` (blocked on a prompt),
    ``starting`` (agent building), ``working`` (turn running), ``idle``.
    """
    if _session_pending_kind(sid):
        return "waiting"
    ready = session.get("agent_ready")
    if ready is not None and not ready.is_set() and session.get("agent_build_started"):
        return "starting"
    if session.get("running"):
        return "working"
    return "idle"


def _message_preview(history: list) -> str:
    """Return a short (≤160 char) preview of the last non-empty message.

    Ported from Hermes line 5760.
    """
    for msg in reversed(history or []):
        if not isinstance(msg, dict):
            continue
        text = _content_display_text(
            msg.get("content", msg.get("text", ""))
        ).strip()
        if text:
            return " ".join(text.split())[:160]
    return ""


def _session_live_title(session: dict, key: str) -> str:
    """Resolve a live session's title: pending → DB → empty.

    Ported from Hermes line 5768.
    """
    title = str(session.get("pending_title") or "").strip()
    db = _get_db()
    if db is not None:
        try:
            db_title = db.get_session_title(key)
            if db_title:
                title = str(db_title).strip() or title
        except Exception:
            pass
    return title


def _session_live_item(sid: str, session: dict, current_sid: str = "") -> dict:
    """Build a session-list item for a live session.

    Ported from Hermes line 5779. Used by ``session.active_list``.
    """
    key = _session_lookup_key(session, fallback=sid)
    agent = session.get("agent")
    history = list(session.get("history") or [])
    status = _session_live_status(sid, session)
    inflight = _inflight_snapshot(session)
    preview = _message_preview(history)
    if inflight:
        preview = inflight.get("assistant") or inflight.get("user") or preview
        preview = " ".join(str(preview).split())[:160]
    now = time.time()
    return {
        "current": sid == current_sid,
        "id": sid,
        "last_active": float(session.get("last_active") or session.get("created_at") or now),
        "message_count": len(history),
        "model": str(getattr(agent, "model", "") or _resolve_model()),
        "preview": preview,
        "session_key": key,
        "started_at": float(session.get("created_at") or now),
        "status": status,
        "title": _session_live_title(session, key),
    }


def _fallback_session_info(session: dict) -> dict:
    """session.info for a not-yet-built session.

    Ported from Hermes line 5823. Used by ``_live_session_payload`` when the
    agent hasn't been built yet (lazy watch / cold resume).
    """
    agent = session.get("agent")
    if agent is not None:
        return _session_info(agent, session)
    return {
        "cwd": session.get("cwd") or _default_session_cwd(),
        "lazy": True,
        "model": _resolve_model(),
        "skills": {},
        "tools": {},
    }


def _live_session_payload(
    sid: str,
    session: dict,
    *,
    cols: int | None = None,
    touch: bool = False,
    transport: Any = None,
) -> dict:
    """Build the resume / list payload for a live session.

    Ported from Hermes line 5836. Touches last_active, rebinds transport,
    and includes the full display history (prefix + live).
    """
    with session["history_lock"]:
        if cols is not None:
            session["cols"] = cols
        if transport is not None:
            session["transport"] = transport
        if touch:
            session["last_active"] = time.time()
        history = list(session.get("display_history_prefix") or []) + list(
            session.get("history") or []
        )
        inflight = _inflight_snapshot(session)
        running = bool(session.get("running"))
    payload = {
        "info": _fallback_session_info(session),
        "message_count": len(history),
        "messages": _history_to_messages(history),
        "running": running,
        "session_id": sid,
        "session_key": _session_lookup_key(session, fallback=sid),
        "started_at": float(session.get("created_at") or time.time()),
        "status": _session_live_status(sid, session),
    }
    if inflight:
        payload["inflight"] = inflight
    return payload


def _lazy_resume_info(cwd: str, *, model: str = "", provider: str = "") -> dict:
    """session.info for a not-yet-built resumed session.

    Ported from Hermes line 5251. Mirrors the shape ``session.create`` returns
    so the TUI can paint immediately before the deferred build emits a full
    session.info.
    """
    info: dict = {
        "cwd": cwd,
        "branch": _git_branch_for_cwd(cwd),
        "model": model or _resolve_model(),
        "tools": {},
        "skills": {},
        "lazy": True,
    }
    if provider:
        info["provider"] = provider
    return info


def _deferred_session_record(
    session_key: str,
    *,
    cols: int,
    cwd: str,
    history: list,
    lease: Any,
    source: str = "tui",
    close_on_disconnect: bool = False,
    display_history_prefix: list | None = None,
    profile_home: Any = None,
    lazy: bool = False,
    model_override: Any = None,
    resume_runtime_overrides: dict | None = None,
) -> dict:
    """A live-session record whose agent is built later (lazy watch / cold resume).

    Ported from Hermes line 5269. Same shape as ``_init_session`` minus the
    agent. ``_start_agent_build`` later fills in the agent + worker.
    """
    now = time.time()
    return {
        "agent": None,
        "agent_error": None,
        "agent_ready": threading.Event(),
        "agent_build_started": False,
        "attached_images": [],
        "close_on_disconnect": close_on_disconnect,
        "active_session_lease": lease,
        "cols": cols,
        "created_at": now,
        "cwd": cwd,
        "display_history_prefix": display_history_prefix or [],
        "edit_snapshots": {},
        "explicit_cwd": False,
        "history": history,
        "history_lock": threading.Lock(),
        "history_version": 0,
        "image_counter": 0,
        "inflight_turn": None,
        "last_active": now,
        "lazy": lazy,
        "model_override": model_override,
        "pending_title": None,
        "profile_home": str(profile_home) if profile_home is not None else None,
        "resume_runtime_overrides": resume_runtime_overrides,
        "resume_session_id": session_key,
        "running": False,
        "session_key": session_key,
        "show_reasoning": _load_show_reasoning(),
        "slash_worker": None,
        "source": source,
        "tool_progress_mode": _load_tool_progress_mode(),
        "tool_started_at": {},
        "transport": current_transport() or _stdio_transport,
    }


def _claim_or_reuse_live(
    sid: str, session_key: str, record: dict, lease: Any,
) -> tuple[str, dict] | None:
    """Register ``record`` as the live session, or reuse a concurrent winner.

    Ported from Hermes line 5323. Returns ``None`` on success (the caller
    owns the session) or ``(other_sid, other_session)`` if a concurrent
    resume already won (the caller should reuse the winner and release its
    lease).
    """
    with _session_resume_lock:
        live = _find_live_session_by_key(session_key)
        if live is not None:
            if lease is not None and hasattr(lease, "release"):
                try:
                    lease.release()
                except Exception:
                    pass
            return live
        with _sessions_lock:
            _sessions[sid] = record
            _register_session_cwd(_sessions[sid])
    return None


def _schedule_agent_build(sid: str, delay: float = 0.05) -> None:
    """Pre-warm a deferred session's agent off the response path.

    Ported from Hermes line 5341. Used by ``session.create`` and cold resume
    so the RPC returns immediately while the agent builds on a timer.
    """

    def _run():
        session = _sessions.get(sid)
        if session is not None:
            _start_agent_build(sid, session)

    timer = threading.Timer(delay, _run)
    timer.daemon = True
    timer.start()


def _profile_home(profile: str | None) -> Any:
    """Resolve a named profile's home on THIS host, or None for the launch profile.

    Ported from Hermes line 912. NIA has a ``niaharness.profiles`` package but
    it does not yet expose ``get_profile_dir(name)`` — guard with try/except
    so the launch profile path (None) is returned. TODO(feature-gap): see
    FEATURE_GAPS.md (profiles.get_profile_dir).
    """
    name = (profile or "").strip()
    if not name:
        return None
    try:
        from niaharness.profiles import get_profile_dir  # type: ignore

        home = Path(get_profile_dir(name))
    except ImportError:
        # TODO(feature-gap): see FEATURE_GAPS.md (profiles.get_profile_dir).
        return None
    except Exception:
        return None
    # Already the launch profile? No override needed.
    try:
        if home.resolve() == Path(_hermes_home).resolve():
            return None
    except Exception:
        pass
    return home if (home / "state.db").exists() or home.exists() else None


_CWD_PLACEHOLDERS = {".", "auto", "cwd"}


def _configured_cwd_from_cfg(cfg: dict | None) -> str | None:
    """Return an absolute, existing ``terminal.cwd`` from a config mapping.

    Ported from Hermes line 958. Returns None for placeholders, missing
    values, or paths that don't resolve to a real directory.
    """
    if not isinstance(cfg, dict):
        return None
    terminal_cfg = cfg.get("terminal")
    if not isinstance(terminal_cfg, dict):
        return None
    raw = str(terminal_cfg.get("cwd") or "").strip()
    if not raw or raw in _CWD_PLACEHOLDERS:
        return None
    try:
        expanded = os.path.abspath(os.path.expanduser(raw))
        if os.path.isdir(expanded):
            return expanded
    except Exception:
        pass
    return None


def _profile_configured_cwd(profile_home: Any) -> str | None:
    """Resolve a non-launch profile's ``terminal.cwd`` from its own config.yaml.

    Ported from Hermes line 976.
    """
    if profile_home is None:
        return None
    try:
        import yaml  # type: ignore

        p = Path(profile_home) / "config.yaml"
        if not p.exists():
            return None
        with open(p, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return _configured_cwd_from_cfg(data)
    except Exception:
        return None


def _launch_configured_cwd() -> str | None:
    """Resolve the launch profile's ``terminal.cwd`` from config.yaml.

    Ported from Hermes line 1000.
    """
    try:
        return _configured_cwd_from_cfg(_load_cfg())
    except Exception:
        return None


def _stored_session_runtime_overrides(row: dict | None) -> dict:
    """Return runtime fields persisted with a stored session.

    Ported from Hermes line 2068. ``session.resume`` is session-scoped:
    reopening an older chat must restore the model/provider/reasoning state
    that chat actually used, not whatever global model the user most recently
    selected in another chat.
    """
    if not row:
        return {}

    raw_config = row.get("model_config")
    model_config: dict = {}
    if isinstance(raw_config, dict):
        model_config = raw_config
    elif isinstance(raw_config, str) and raw_config.strip():
        try:
            parsed = json.loads(raw_config)
            if isinstance(parsed, dict):
                model_config = parsed
        except Exception:
            logger.debug("failed to parse stored session model_config", exc_info=True)

    overrides: dict = {}
    model = str(row.get("model") or model_config.get("model") or "").strip()
    explicit_provider = str(model_config.get("provider") or "").strip()
    billing_provider = str(
        model_config.get("billing_provider") or row.get("billing_provider") or ""
    ).strip()
    provider = explicit_provider
    if not provider and billing_provider.lower() not in {"custom", ""}:
        provider = billing_provider
    base_url = str(model_config.get("base_url") or "").strip()
    api_mode = str(model_config.get("api_mode") or "").strip()
    reasoning_config = model_config.get("reasoning_config")
    service_tier = str(model_config.get("service_tier") or "").strip()

    # Heal a bare ``"custom"`` provider — NIA doesn't have
    # ``canonical_custom_identity`` so just drop it. TODO(feature-gap): see
    # FEATURE_GAPS.md (runtime_provider.canonical_custom_identity).
    if provider.strip().lower() == "custom":
        provider = "" if not base_url else provider

    if model:
        overrides["model_override"] = {
            "model": model,
            "provider": provider or None,
            "base_url": base_url or None,
            "api_mode": api_mode or None,
        }
    if provider:
        overrides["provider_override"] = provider
    if isinstance(reasoning_config, dict):
        overrides["reasoning_config_override"] = reasoning_config
    if service_tier:
        overrides["service_tier_override"] = service_tier

    return overrides


def _current_profile_name() -> str:
    """Return the current profile name (for the session.info payload).

    Ported from Hermes line 3146. NIA's profile subsystem doesn't expose this
    yet — return "" so the field is present but empty.
    TODO(feature-gap): see FEATURE_GAPS.md (profiles.current_profile_name).
    """
    try:
        from niaharness.profiles import current_profile_name  # type: ignore

        return str(current_profile_name() or "")
    except ImportError:
        return ""
    except Exception:
        return ""


def _enrich_with_attached_images(user_text: str, image_paths: list[str]) -> str:
    """Pre-analyze attached images via vision and prepend descriptions.

    Ported from Hermes line 4574. Used as the text-mode fallback when the
    active provider/model can't accept native image content. NIA may not yet
    expose ``vision_analyze_tool`` — guard with try/except so a missing
    vision stack leaves the hint in place instead of crashing the turn.
    TODO(feature-gap): see FEATURE_GAPS.md.
    """
    import asyncio
    import json as _json

    try:
        from niaharness.tools.vision_tools import vision_analyze_tool  # type: ignore
    except ImportError:
        # TODO(feature-gap): see FEATURE_GAPS.md (tools.vision_tools).
        vision_analyze_tool = None

    prompt = (
        "Describe everything visible in this image in thorough detail. "
        "Include any text, code, data, objects, people, layout, colors, "
        "and any other notable visual information."
    )

    parts: list[str] = []
    for path in image_paths:
        p = Path(path)
        if not p.exists():
            continue
        hint = f"[You can examine it with vision_analyze using image_url: {p}]"
        if vision_analyze_tool is None:
            parts.append(f"[The user attached an image but analysis is unavailable.]\n{hint}")
            continue
        try:
            r = _json.loads(
                asyncio.run(vision_analyze_tool(image_url=str(p), user_prompt=prompt))
            )
            desc = r.get("analysis", "") if r.get("success") else None
            parts.append(
                f"[The user attached an image:\n{desc}]\n{hint}"
                if desc
                else f"[The user attached an image but analysis failed.]\n{hint}"
            )
        except Exception:
            parts.append(f"[The user attached an image but analysis failed.]\n{hint}")

    text = user_text or ""
    prefix = "\n\n".join(parts)
    if prefix:
        return f"{prefix}\n\n{text}" if text else prefix
    return text or "What do you see in this image?"


def _voice_tts_enabled() -> bool:
    """Whether agent replies should be spoken back via TTS (runtime only).

    Ported from Hermes line 12763. Honors ``NIA_VOICE_TTS`` (and the legacy
    ``HERMES_VOICE_TTS``) env var.
    """
    return (
        os.environ.get("NIA_VOICE_TTS", "").strip() == "1"
        or os.environ.get("HERMES_VOICE_TTS", "").strip() == "1"
    )


def _speak_text(text: str) -> None:
    """Best-effort speak ``text`` via NIA's TTS tool.

    Hermes uses ``hermes_cli.voice.speak_text`` (line 8922). NIA exposes TTS
    through the SpeakTool — invoke it in a fire-and-forget thread. Failures
    are swallowed (TTS is a non-critical UX polish path).
    """
    if not text or not text.strip():
        return
    try:
        import asyncio

        from niaharness.tools.speak_tool import SpeakTool, SpeakToolInput

        async def _speak():
            tool = SpeakTool()
            await tool.execute(SpeakToolInput(text=text[:3000]), None)

        asyncio.run(_speak())
    except Exception as e:
        logger.debug("voice TTS dispatch failed: %s", e)


def _notification_event_belongs_elsewhere(session: dict, evt: dict) -> bool:
    """True if ``evt`` is owned by a *different* live session.

    Ported from Hermes line 8299. Background-process events carry the
    ``session_key`` of the session that started the process; each poller must
    skip events it doesn't own so a background job's completion surfaces in
    the session that launched it.
    """
    evt_key = str(evt.get("session_key") or "")
    if not evt_key:
        return False
    if evt_key == str(session.get("session_key") or ""):
        return False
    try:
        with _sessions_lock:
            snapshot = list(_sessions.values())
    except Exception:
        return False
    return any(
        s is not session and str(s.get("session_key") or "") == evt_key
        for s in snapshot
    )


def _notification_event_dedup_key(evt: dict) -> tuple:
    """Return the UI-emission identity for a process notification event.

    Ported from Hermes line 8329. Completion events are terminal (one-shot per
    process session); watch-match events are not (include event-specific
    content so distinct matches from the same process remain visible).
    """
    evt_type = evt.get("type", "completion")
    evt_sid = evt.get("session_id", "")
    if evt_type == "watch_match":
        return (
            evt_sid,
            evt_type,
            evt.get("command", ""),
            evt.get("pattern", ""),
            evt.get("output", ""),
            evt.get("suppressed", 0),
            evt.get("message_id", ""),
        )
    if evt_type.startswith("watch_overflow_") or evt_type == "watch_disabled":
        return (
            evt_sid,
            evt_type,
            evt.get("command", ""),
            evt.get("message", ""),
            evt.get("suppressed", 0),
        )
    if evt_type == "async_delegation":
        return (evt.get("delegation_id", ""), evt_type)
    return (evt_sid, evt_type)


def _wire_agent_terminal_output() -> None:
    """Idempotently route background-process output to the desktop.

    Ported from Hermes line 8483. NIA does not yet have a ``process_registry``
    with ``on_output``/``on_close`` sinks — guard with try/except so the
    notification poller can still run. TODO(feature-gap): see FEATURE_GAPS.md.
    """
    try:
        from niaharness.tools.process_registry import process_registry  # type: ignore
    except ImportError:
        # TODO(feature-gap): see FEATURE_GAPS.md (tools.process_registry).
        return

    has_output_sink = getattr(process_registry, "on_output", None) is not None
    has_close_sink = getattr(process_registry, "on_close", None) is not None
    if has_output_sink and has_close_sink:
        return

    def _owner_sid_for_process(session) -> str:
        session_key = str(getattr(session, "session_key", "") or "")
        if not session_key:
            return ""
        with _sessions_lock:
            for sid, tui_session in _sessions.items():
                if str(tui_session.get("session_key") or "") == session_key:
                    return sid
        return ""

    def _emit_agent_terminal_output(session, chunk):
        _emit(
            "agent.terminal.output",
            _owner_sid_for_process(session),
            {"process_id": session.id, "chunk": chunk},
        )

    def _emit_agent_terminal_close(session, process_id):
        sid = _owner_sid_for_process(session) if session is not None else ""
        _emit("terminal.close", sid, {"process_id": process_id})

    if not has_output_sink:
        try:
            process_registry.on_output = _emit_agent_terminal_output
        except Exception:
            pass
    if not has_close_sink:
        try:
            process_registry.on_close = _emit_agent_terminal_close
        except Exception:
            pass


def _notification_poller_loop(
    stop_event: threading.Event, sid: str, session: dict,
) -> None:
    """Poll completion_queue and dispatch notifications autonomously.

    Ported from Hermes line 8366. NIA does not yet have ``process_registry`` —
    the import is guarded so a missing registry makes this loop a no-op.
    TODO(feature-gap): see FEATURE_GAPS.md.
    """
    try:
        from niaharness.tools.process_registry import (  # type: ignore
            format_process_notification,
            process_registry,
        )
    except ImportError:
        # TODO(feature-gap): see FEATURE_GAPS.md (tools.process_registry).
        return

    _emitted: set = set()
    while not stop_event.is_set() and not session.get("_finalized"):
        try:
            evt = process_registry.completion_queue.get(timeout=0.5)
        except Exception:
            continue

        if _notification_event_belongs_elsewhere(session, evt):
            try:
                process_registry.completion_queue.put(evt)
            except Exception:
                pass
            time.sleep(0.1)
            continue

        _evt_sid = evt.get("session_id", "")
        if evt.get("type") == "completion" and process_registry.is_completion_consumed(_evt_sid):
            continue

        text = format_process_notification(evt)
        if not text:
            continue

        _dedup_key = _notification_event_dedup_key(evt)
        if _dedup_key not in _emitted:
            _emit("status.update", sid, {"kind": "process", "text": text})
            _emitted.add(_dedup_key)

        with session["history_lock"]:
            if session.get("running"):
                try:
                    process_registry.completion_queue.put(evt)
                except Exception:
                    pass
                continue
            session["running"] = True

        rid = f"__notif__{int(time.time() * 1000)}"
        try:
            _emit("message.start", sid)
            _run_prompt_submit(rid, sid, session, text)
        except Exception as exc:
            print(
                f"[tui_gateway] notification poller dispatch failed: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            with session["history_lock"]:
                session["running"] = False

    # Drain any remaining events after stop signal. Events owned by other
    # live sessions are set aside and re-queued so their poller still sees them.
    deferred: list = []
    while not process_registry.completion_queue.empty():
        try:
            evt = process_registry.completion_queue.get_nowait()
        except Exception:
            break
        if _notification_event_belongs_elsewhere(session, evt):
            deferred.append(evt)
            continue
        _evt_sid = evt.get("session_id", "")
        if evt.get("type") == "completion" and process_registry.is_completion_consumed(_evt_sid):
            continue
        text = format_process_notification(evt)
        if not text:
            continue

        _dedup_key = _notification_event_dedup_key(evt)
        if _dedup_key not in _emitted:
            _emit("status.update", sid, {"kind": "process", "text": text})
            _emitted.add(_dedup_key)

        with session["history_lock"]:
            if session.get("running"):
                try:
                    process_registry.completion_queue.put(evt)
                except Exception:
                    pass
                break
            session["running"] = True

        rid = f"__notif__{int(time.time() * 1000)}"
        try:
            _emit("message.start", sid)
            _run_prompt_submit(rid, sid, session, text)
        except Exception as exc:
            print(
                f"[tui_gateway] notification poller dispatch failed: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            with session["history_lock"]:
                session["running"] = False

    for evt in deferred:
        try:
            process_registry.completion_queue.put(evt)
        except Exception:
            pass


def _start_notification_poller(sid: str, session: dict) -> threading.Event:
    """Start the background notification poller for a TUI session.

    Ported from Hermes line 8528. Returns the stop event so the caller can
    signal shutdown.
    """
    _wire_agent_terminal_output()
    stop = threading.Event()
    t = threading.Thread(
        target=_notification_poller_loop,
        args=(stop, sid, session),
        daemon=True,
    )
    t.start()
    return stop


# ---------------------------------------------------------------------------
# Prompt submission
# ---------------------------------------------------------------------------


@method("prompt.submit")
def _prompt_submit(rid, params):
    """Submit a prompt with full turn lifecycle management."""
    sid, text = params.get("session_id", ""), params.get("text", "")
    truncate_user_ordinal = params.get("truncate_before_user_ordinal")
    session, err = _sess_nowait(params, rid)
    if err:
        return err

    # Re-bind to the current client transport.
    if (t := current_transport()) is not None:
        session["transport"] = t

    with session["history_lock"]:
        if session.get("running"):
            # Don't reject — queue (and optionally interrupt) per busy_input_mode.
            return _handle_busy_submit(rid, sid, session, text, t or session.get("transport"))
        # A watch session's run lives in the PARENT turn, so its own running
        # flag is False — without this guard, typing mid-run builds a second
        # agent racing the in-flight child on the same stored session
        # (interleaved transcript, stale fork). After the run completes,
        # submitting is fine: the upgrade resumes the child's transcript.
        # Ported from Hermes line 8232.
        if session.get("lazy") and _child_run_active(str(session.get("session_key") or "")):
            return _err(rid, 4009, "subagent still running — wait for it to finish")

        # Truncate before a specific user ordinal if requested.
        if truncate_user_ordinal is not None:
            try:
                ordinal = int(truncate_user_ordinal)
            except (TypeError, ValueError):
                return _err(rid, 4004, "truncate_before_user_ordinal must be an integer")
            history = session.get("history", [])
            user_indices = [i for i, m in enumerate(history) if m.get("role") == "user"]
            if ordinal < 0 or ordinal >= len(user_indices):
                return _err(rid, 4018, "target user message is no longer in session history")
            truncated = history[:user_indices[ordinal]]
            session["history"] = truncated
            session["history_version"] = session.get("history_version", 0) + 1
            db = _get_db()
            if db:
                try:
                    db.replace_messages(session["session_key"], truncated)
                except Exception as exc:
                    print(f"[tui_gateway] prompt.submit: replace_messages failed: {exc}", file=sys.stderr)

        session["running"] = True
        session["_turn_cancel_requested"] = False
        session["last_active"] = time.time()
        _start_inflight_turn(session, text)

    # Persist the DB row lazily.
    _ensure_session_db_row(session)
    # A branch becomes real here: copy its parent's transcript into the row so
    # it resumes with full context (the agent won't persist the seed itself).
    # Ported from Hermes line 8265.
    _persist_branch_seed(session)

    # Build the agent if not yet built.
    _start_agent_build(sid, session)

    def run_after_agent_ready():
        err = _wait_agent(session, rid)
        if err:
            _emit("error", sid, {"message": err.get("error", {}).get("message", "agent init failed")})
            with session["history_lock"]:
                session["running"] = False
                _clear_inflight_turn(session)
            return
        with session["history_lock"]:
            if session.get("_turn_cancel_requested") or not session.get("running"):
                session["running"] = False
                _clear_inflight_turn(session)
                return
        _run_prompt_submit(rid, sid, session, text)

    run_thread = threading.Thread(target=run_after_agent_ready, daemon=True)
    session["_run_thread"] = run_thread
    run_thread.start()
    return _ok(rid, {"status": "streaming"})


def _run_prompt_submit(rid, sid: str, session: dict, text: Any) -> None:
    """Execute a prompt submission with full streaming, tool events, and error handling.

    Deep-ported from Hermes tui_gateway/server.py line 8541. Preserves NIA's
    async streaming pattern (``agent.submit_message()`` async iteration) while
    wrapping it with Hermes's pre/post-turn logic: profile-home override,
    session-context tokens, model sync, branch-seed persist, image routing,
    history_version mismatch backstop, compression session_key rotation,
    pending_title application, voice TTS, goal continuation, and process
    notification drain. Features that depend on NIA infrastructure which
    doesn't exist yet are wrapped in try/except with TODO(feature-gap) markers
    — see FEATURE_GAPS.md.
    """
    with session["history_lock"]:
        history = list(session["history"])
        history_version = int(session.get("history_version", 0))
        images = list(session.get("attached_images", []))
        session["attached_images"] = []
        if not isinstance(session.get("inflight_turn"), dict):
            _start_inflight_turn(session, text)

    agent = session["agent"]
    if hasattr(agent, "clear_interrupt"):
        try:
            agent.clear_interrupt()
        except Exception:
            pass

    _emit("message.start", sid)

    def run():
        approval_token = None
        session_tokens: list = []
        home_token = None  # per-turn NIA_HOME override for a resumed remote profile
        goal_followup = None  # set by the post-turn goal hook below
        try:
            from niaharness.permissions.approval import (
                reset_current_session_key,
                set_current_session_key,
            )

            approval_token = set_current_session_key(session["session_key"])

            # (2) Session context tokens — binds session_key/source/cwd for
            # downstream tools that consult contextvars. Guarded so a missing
            # gateway.session_context module makes this a no-op.
            session_tokens = _set_session_context(session["session_key"])

            # (1) Profile home override — for resumed remote profiles. NIA does
            # not yet have ``set_nia_home_override``; the override is a no-op.
            # TODO(feature-gap): see FEATURE_GAPS.md (set_nia_home_override).
            _profile_home_str = session.get("profile_home")
            if _profile_home_str:
                try:
                    from niaharness.prompts.soul import set_nia_home_override  # type: ignore
                    home_token = set_nia_home_override(_profile_home_str)
                except ImportError:
                    # Feature gap — leave home_token None so finally skips reset.
                    pass
                except Exception:
                    pass

            # (3) Re-wire callbacks on the turn thread. The sudo password
            # callback is thread-local in Hermes; re-wiring here ensures
            # prompts route to the sudo.request overlay even after the build
            # thread set them. NIA's _wire_callbacks is mostly a stub but the
            # call is preserved for parity.
            _wire_callbacks(sid)

            # (4) Sync agent model with config — adopt a config.yaml model
            # change at turn start (sessions pinned with /model keep theirs).
            try:
                _sync_agent_model_with_config(sid, session)
            except Exception as exc:
                logger.debug("_sync_agent_model_with_config failed: %s", exc)

            # (5) Register session cwd — so terminal sudo / async tasks
            # inherit the TUI session's workspace. No-op if NIA's terminal
            # tool doesn't expose register_task_env_overrides.
            _register_session_cwd(session)

            cwd = _session_cwd(session)
            cols = session.get("cols", 80)
            streamer = make_stream_renderer(cols)
            prompt = text

            # (6) Context references preprocessing for ``@`` mentions. NIA
            # does not yet have preprocess_context_references — guard.
            # TODO(feature-gap): see FEATURE_GAPS.md (context_references).
            if isinstance(prompt, str) and "@" in prompt:
                try:
                    from niaharness.engine.context_references import (  # type: ignore
                        preprocess_context_references,
                    )
                    from niaharness.engine.model_metadata import (  # type: ignore
                        get_model_context_length,
                    )

                    ctx_len = get_model_context_length(
                        getattr(agent, "model", "") or _resolve_model(),
                        base_url=getattr(agent, "base_url", "") or "",
                        api_key=getattr(agent, "api_key", "") or "",
                        provider=getattr(agent, "provider", "") or "",
                        config_context_length=getattr(
                            agent, "_config_context_length", None
                        ),
                    )
                    ctx = preprocess_context_references(
                        prompt,
                        cwd=cwd,
                        allowed_root=cwd,
                        context_length=ctx_len,
                    )
                    if ctx.blocked:
                        _emit(
                            "error",
                            sid,
                            {
                                "message": "\n".join(ctx.warnings)
                                or "Context injection refused."
                            },
                        )
                        return
                    prompt = ctx.message
                except ImportError:
                    # Feature gap — leave prompt untouched.
                    pass
                except Exception as _ctx_exc:
                    print(
                        f"[tui_gateway] context reference preprocessing failed: "
                        f"{type(_ctx_exc).__name__}: {_ctx_exc}",
                        file=sys.stderr,
                    )

            # (7+8) Image routing — decide native vs text mode per-turn.
            # NIA does not yet have agent.image_routing — always fall back
            # to text-mode enrichment.
            # TODO(feature-gap): see FEATURE_GAPS.md (image_routing).
            run_message: Any = prompt
            if images:
                image_paths = []
                for img in images:
                    if isinstance(img, dict):
                        p = img.get("path")
                        if p:
                            image_paths.append(str(p))
                    elif isinstance(img, str):
                        image_paths.append(img)
                try:
                    from niaharness.engine.image_routing import (  # type: ignore
                        build_native_content_parts,
                        decide_image_input_mode,
                    )
                    from niaharness.engine.auxiliary_client import (  # type: ignore
                        _read_main_model,
                        _read_main_provider,
                    )

                    _mode = decide_image_input_mode(
                        _read_main_provider(),
                        _read_main_model(),
                        _load_cfg(),
                    )
                    if getattr(agent, "api_mode", "") == "codex_app_server":
                        _mode = "text"
                except ImportError:
                    _mode = "text"
                except Exception as _img_exc:
                    print(
                        f"[tui_gateway] image_routing decision failed, defaulting to text: {_img_exc}",
                        file=sys.stderr,
                    )
                    _mode = "text"

                if _mode == "native":
                    try:
                        _parts, _skipped = build_native_content_parts(
                            prompt, image_paths,
                        )
                        if _skipped:
                            print(
                                f"[tui_gateway] native image attachment skipped "
                                f"{len(_skipped)} unreadable path(s)",
                                file=sys.stderr,
                            )
                        if any(p.get("type") == "image_url" for p in _parts):
                            run_message = _parts
                        else:
                            run_message = _enrich_with_attached_images(prompt, image_paths)
                    except Exception as _img_exc:
                        print(
                            f"[tui_gateway] native attach failed, falling back to text: {_img_exc}",
                            file=sys.stderr,
                        )
                        run_message = _enrich_with_attached_images(prompt, image_paths)
                else:
                    run_message = _enrich_with_attached_images(prompt, image_paths)

            # Stream callback.
            def _stream(delta):
                with session["history_lock"]:
                    _append_inflight_delta(session, delta)
                payload = {"text": delta}
                if streamer and (r := streamer.feed(delta)) is not None:
                    payload["rendered"] = r
                _emit("message.delta", sid, payload)

            # Add user message to history.
            with session["history_lock"]:
                session["history"].append({"role": "user", "text": _inflight_text(text)})

            # (9) task_id parameter detection — Hermes passes task_id=session_key
            # if the agent's run_conversation accepts it. NIA's submit_message
            # only takes a prompt; this is a no-op parity check.
            submit_kwargs: dict = {}
            try:
                if "task_id" in inspect.signature(agent.submit_message).parameters:
                    submit_kwargs["task_id"] = session["session_key"]
            except (TypeError, ValueError):
                pass

            # Run the conversation via the QueryEngine (NIA's async pattern).
            import asyncio

            async def _run_turn():
                response_text = ""
                last_reasoning_text = ""
                turn_error_msg = ""
                interrupted = False
                async for event in agent.submit_message(_inflight_text(run_message), **submit_kwargs):
                    from niaharness.engine.stream_events import (
                        AssistantTextDelta,
                        AssistantTurnComplete,
                        ToolExecutionStarted,
                        ToolExecutionCompleted,
                    )
                    if isinstance(event, AssistantTextDelta):
                        _stream(event.text)
                        response_text += event.text
                    elif isinstance(event, ToolExecutionStarted):
                        _emit("tool.start", sid, {
                            "name": event.tool_name,
                            "args_text": str(event.tool_input)[:200] if event.tool_input else "",
                            "tool_id": getattr(event, "tool_use_id", ""),
                        })
                        _status_update(sid, "tool", f"Running {event.tool_name}...")
                    elif isinstance(event, ToolExecutionCompleted):
                        result_text = event.output[:500] if event.output else ""
                        _emit("tool.complete", sid, {
                            "name": event.tool_name,
                            "result_text": result_text,
                            "error": str(event.error) if event.is_error else None,
                            "tool_id": getattr(event, "tool_use_id", ""),
                            "duration_s": getattr(event, "duration", 0),
                        })
                        # Add tool result to history.
                        with session["history_lock"]:
                            session["history"].append({
                                "role": "tool",
                                "text": result_text,
                                "tool_name": event.tool_name,
                            })
                    elif isinstance(event, AssistantTurnComplete):
                        # (14) last_reasoning tracking — capture from the
                        # event if the engine exposes it.
                        lr = getattr(event, "reasoning", None) or getattr(event, "last_reasoning", None)
                        if isinstance(lr, str) and lr.strip():
                            last_reasoning_text = lr.strip()
                        err = getattr(event, "error", None)
                        if isinstance(err, str) and err.strip():
                            turn_error_msg = err.strip()
                        if getattr(event, "interrupted", False):
                            interrupted = True
                        return {
                            "text": response_text,
                            "last_reasoning": last_reasoning_text,
                            "error": turn_error_msg,
                            "interrupted": interrupted,
                        }
                return {
                    "text": response_text,
                    "last_reasoning": last_reasoning_text,
                    "error": turn_error_msg,
                    "interrupted": interrupted,
                }

            try:
                result = asyncio.run(_run_turn())
            except Exception as exc:
                result = {
                    "text": "",
                    "last_reasoning": "",
                    "error": str(exc),
                    "interrupted": False,
                    "failed": True,
                }
                _emit("error", sid, {"message": str(exc)})

            raw = result.get("text", "") or ""
            last_reasoning = result.get("last_reasoning", "") or ""
            turn_error = result.get("error", "") or ""
            interrupted = bool(result.get("interrupted"))
            failed = bool(result.get("failed"))

            # (10) MoA one-shot restore — NIA does not have MoA. Skip the
            # restore path entirely; if a session somehow has
            # ``moa_one_shot_restore`` set, pop and ignore it.
            # TODO(feature-gap): see FEATURE_GAPS.md (MoA).
            if "moa_one_shot_restore" in session:
                session.pop("moa_one_shot_restore", None)

            # (11) History_version mismatch backstop. If history was mutated
            # externally during the turn (undo/compress/retry/rollback), surface
            # the desync rather than silently clobbering the agent's output.
            status_note = None
            with session["history_lock"]:
                current_version = int(session.get("history_version", 0))
                if current_version != history_version:
                    print(
                        f"[tui_gateway] prompt.submit: history_version mismatch "
                        f"(expected={history_version} current={current_version}) — "
                        f"agent output NOT written to session history",
                        file=sys.stderr,
                    )
                    status_note = (
                        "History changed during this turn — the response above is visible "
                        "but was not saved to session history."
                    )
                else:
                    if raw:
                        session["history"].append({"role": "assistant", "text": raw})
                    session["history_version"] = history_version + 1
                _clear_inflight_turn(session)

            # (12) Sync session_key after auto-compression. The agent's
            # compression path may have rotated session_id; re-anchor the
            # gateway-side session_key so downstream title/goal/finalize
            # handling targets the live row.
            try:
                _sync_session_key_after_compress(
                    sid, session, clear_pending_title=False, restart_slash_worker=True,
                )
            except Exception as exc:
                logger.debug("_sync_session_key_after_compress failed: %s", exc)

            # (13) Error surface for failed/partial responses. When the backend
            # produced no visible response AND reported a real error, surface
            # that error as the visible text instead of shipping an empty turn.
            if (not raw) and turn_error and failed:
                raw = f"Error: {turn_error}"

            status = (
                "interrupted" if interrupted
                else "error" if failed or turn_error
                else "complete"
            )

            # Emit message.complete.
            payload = {"text": raw, "usage": _get_usage(agent), "status": status}
            # (14) last_reasoning payload — surface reasoning on the complete event.
            if last_reasoning:
                payload["reasoning"] = last_reasoning
            # (15) status_note warning — surface history desync to the user.
            if status_note:
                payload["warning"] = status_note
            rendered = render_message(raw, cols)
            if rendered:
                payload["rendered"] = rendered
            _emit("message.complete", sid, payload)

            # Persist to DB.
            db = _get_db()
            if db and not status_note:
                # Skip persistence when history_version mismatched — the
                # agent's output would land on top of a divergent transcript.
                try:
                    session_key = session.get("session_key", sid)
                    if raw:
                        db.add_message(session_key, "user", _inflight_text(text))
                        db.add_message(session_key, "assistant", raw)
                except Exception:
                    pass

            # (16) /goal continuation (Ralph-style loop). NIA does not yet
            # have a GoalManager — skip the post-turn goal evaluation.
            # TODO(feature-gap): see FEATURE_GAPS.md (GoalManager).
            if status == "complete" and isinstance(raw, str) and raw.strip():
                try:
                    from niaharness.goals import GoalManager  # type: ignore

                    sid_key = session.get("session_key") or ""
                    if sid_key:
                        try:
                            goals_cfg = _load_cfg().get("goals") or {}
                            goal_max_turns = int(goals_cfg.get("max_turns", 20) or 20)
                        except Exception:
                            goal_max_turns = 20
                        goal_mgr = GoalManager(
                            session_id=sid_key,
                            default_max_turns=goal_max_turns,
                        )
                        if goal_mgr.is_active():
                            try:
                                from niaharness.goals import (  # type: ignore
                                    gather_background_processes as _gather_bg,
                                )
                                _bg_procs = _gather_bg()
                            except Exception:
                                _bg_procs = None
                            decision = goal_mgr.evaluate_after_turn(
                                raw,
                                user_initiated=True,
                                background_processes=_bg_procs,
                            )
                            verdict_msg = decision.get("message") or ""
                            if verdict_msg:
                                _emit(
                                    "status.update",
                                    sid,
                                    {"kind": "goal", "text": verdict_msg},
                                )
                            if decision.get("should_continue"):
                                cont_prompt = decision.get("continuation_prompt") or ""
                                if cont_prompt:
                                    goal_followup = cont_prompt
                except ImportError:
                    # Feature gap — no goal manager; goal_followup stays None.
                    pass
                except Exception as _goal_exc:
                    print(
                        f"[tui_gateway] goal continuation hook failed: "
                        f"{type(_goal_exc).__name__}: {_goal_exc}",
                        file=sys.stderr,
                    )

            # (17) Apply pending_title now that the DB row exists. Handles
            # ValueError (invalid/duplicate title) by dropping it, and
            # transient DB failures by keeping it for retry.
            _pending = session.get("pending_title")
            if _pending and status == "complete":
                if db is not None:
                    _session_key = session.get("session_key") or sid
                    try:
                        if db.set_session_title(_session_key, _pending):
                            session["pending_title"] = None
                            session["title"] = _pending
                    except ValueError as exc:
                        # Invalid/duplicate title — non-retryable, drop it.
                        # Auto-title will take over. Fix for Hermes #19029.
                        session["pending_title"] = None
                        logger.info(
                            "Dropping pending title for session %s: %s",
                            _session_key, exc,
                        )
                    except Exception:
                        # Transient DB failure — keep pending_title for retry.
                        pass

            # (18) maybe_auto_title — NIA does not yet have an auto-title
            # generator. Fall back to the existing simple first-60-chars
            # heuristic so the sidebar still gets a label.
            # TODO(feature-gap): see FEATURE_GAPS.md (title_generator).
            if (
                status == "complete"
                and isinstance(raw, str)
                and raw.strip()
                and isinstance(text, str)
                and text.strip()
            ):
                try:
                    from niaharness.engine.title_generator import maybe_auto_title  # type: ignore

                    _title_key = session.get("session_key") or sid
                    maybe_auto_title(
                        db,
                        _title_key,
                        text,
                        raw,
                        session.get("history", []),
                        title_callback=lambda t, _k=_title_key: _emit(
                            "session.title", sid, {"session_id": _k, "title": t}
                        ),
                    )
                except ImportError:
                    # Feature gap — use the existing simple auto-title fallback.
                    if not session.get("title") and not session.get("pending_title"):
                        auto_title = _inflight_text(text)[:60].strip()
                        if auto_title:
                            session["title"] = auto_title
                            if db is not None:
                                try:
                                    db.set_session_title(session.get("session_key", sid), auto_title)
                                except Exception:
                                    pass
                            _emit("session.title", sid, {"session_id": sid, "title": auto_title})
                except Exception:
                    pass

            # (19) Voice TTS — speak the agent reply when voice-mode TTS is on.
            # CLI parity (cli.py:_voice_speak_response). Only the final text —
            # tool calls / reasoning already stream separately.
            if (
                status == "complete"
                and isinstance(raw, str)
                and raw.strip()
                and _voice_tts_enabled()
            ):
                try:
                    threading.Thread(
                        target=_speak_text, args=(raw,), daemon=True
                    ).start()
                except Exception as e:
                    logger.warning("voice TTS dispatch failed: %s", e)

        except Exception as e:
            import traceback

            trace = traceback.format_exc()
            try:
                _CRASH_LOG.parent.mkdir(parents=True, exist_ok=True)
                with open(_CRASH_LOG, "a", encoding="utf-8") as f:
                    f.write(
                        f"\n=== turn-dispatcher exception . "
                        f"{time.strftime('%Y-%m-%d %H:%M:%S')} . sid={sid} ===\n"
                    )
                    f.write(trace)
            except Exception:
                pass
            print(f"[gateway-turn] {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            _emit("error", sid, {"message": str(e)})
        finally:
            try:
                if approval_token is not None:
                    reset_current_session_key(approval_token)
            except Exception:
                pass
            # (20) home_token cleanup — pair with (1).
            if home_token is not None:
                try:
                    from niaharness.prompts.soul import reset_nia_home_override  # type: ignore
                    reset_nia_home_override(home_token)
                except ImportError:
                    pass
                except Exception:
                    pass
            # (21) clear_session_context — pair with (2).
            _clear_session_context(session_tokens)
            with session["history_lock"]:
                session["running"] = False
                session["last_active"] = time.time()
                _clear_inflight_turn(session)
            _emit("session.info", sid, _session_info(session.get("agent"), session))

        # A user prompt that arrived mid-turn (interrupt + queue) wins over
        # every auto follow-up below — drain it first and skip them this cycle;
        # the goal judge / notifications re-evaluate at the end of that turn.
        if _drain_queued_prompt(rid, sid, session):
            return

        # (22) Goal followup chaining — chain a goal-continuation turn if the
        # judge said so. Done AFTER the finally releases session["running"]
        # so the nested _run_prompt_submit doesn't deadlock on the busy guard.
        # NIA doesn't have GoalManager yet, so goal_followup is always None —
        # this branch is a no-op parity stub.
        if goal_followup:
            with session["history_lock"]:
                if session.get("running"):
                    # User already sent something — their turn wins.
                    return
                session["running"] = True
            try:
                _emit("message.start", sid)
                _run_prompt_submit(rid, sid, session, goal_followup)
            except Exception as _cont_exc:
                print(
                    f"[tui_gateway] goal continuation dispatch failed: "
                    f"{type(_cont_exc).__name__}: {_cont_exc}",
                    file=sys.stderr,
                )
                with session["history_lock"]:
                    session["running"] = False

        # (23) Process registry notification drain — drain completion
        # notifications that arrived during this turn. The background poller
        # handles between-turn delivery; this is the safety net for events
        # that arrived mid-turn. NIA does not yet have process_registry —
        # guarded so a missing module makes this a no-op.
        # TODO(feature-gap): see FEATURE_GAPS.md (process_registry).
        try:
            from niaharness.tools.process_registry import process_registry  # type: ignore

            for _evt, synth in process_registry.drain_notifications():
                with session["history_lock"]:
                    if session.get("running"):
                        try:
                            process_registry.completion_queue.put(_evt)
                        except Exception:
                            pass
                        break
                    session["running"] = True
                try:
                    _emit("message.start", sid)
                    _run_prompt_submit(rid, sid, session, synth)
                except Exception as _n_exc:
                    print(
                        f"[tui_gateway] completion notification dispatch failed: "
                        f"{type(_n_exc).__name__}: {_n_exc}",
                        file=sys.stderr,
                    )
                    with session["history_lock"]:
                        session["running"] = False
        except ImportError:
            # Feature gap — no process_registry.
            pass
        except Exception as _drain_exc:
            print(
                f"[tui_gateway] completion queue drain failed: "
                f"{type(_drain_exc).__name__}: {_drain_exc}",
                file=sys.stderr,
            )

    run_thread = threading.Thread(target=run, daemon=True)
    session["_run_thread"] = run_thread
    run_thread.start()


@method("prompt.background")
def _prompt_background(rid, params):
    sid = params.get("session_id", "")
    text = params.get("text", "")
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    task_id = f"bg_{uuid4().hex[:8]}"
    _emit("background.queued", sid, {"task_id": task_id, "text": text})
    return _ok(rid, {"task_id": task_id})


# ---------------------------------------------------------------------------
# Image / file attachment
# ---------------------------------------------------------------------------


def _image_meta(path: Path) -> dict:
    meta = {"name": path.name}
    try:
        from PIL import Image
        with Image.open(path) as img:
            w, h = img.size
            meta["width"] = int(w)
            meta["height"] = int(h)
            meta["token_estimate"] = max(1, (w + 511) // 512) * max(1, (h + 511) // 512) * 85
    except Exception:
        pass
    return meta


@method("image.attach")
def _image_attach(rid, params):
    """Attach an image to the session from a local file path.

    Deep-ported from Hermes line 9071. Resolves the path, validates the
    extension, appends to ``session["attached_images"]`` so the next
    ``prompt.submit`` picks it up via the native-image-attach pipeline.

    NIA's CLI module may not yet expose ``_detect_file_drop`` /
    ``_resolve_attachment_path`` / ``_split_path_input`` — fall back to a
    simple path resolution. TODO(feature-gap): see FEATURE_GAPS.md
    (cli._resolve_attachment_path).
    """
    session, err = _sess(params, rid)
    if err:
        return err
    raw = str(params.get("path", "") or "").strip()
    if not raw:
        return _err(rid, 4015, "path required")

    # Try the full Hermes-style resolution first.
    image_path: Path | None = None
    remainder = ""
    try:
        from niaharness.cli import (  # type: ignore
            _IMAGE_EXTENSIONS,
            _detect_file_drop,
            _resolve_attachment_path,
            _split_path_input,
        )

        dropped = _detect_file_drop(raw)
        if dropped:
            image_path = Path(dropped["path"])
            remainder = dropped.get("remainder", "")
        else:
            path_token, remainder = _split_path_input(raw)
            resolved = _resolve_attachment_path(path_token)
            if resolved is None:
                return _err(rid, 4016, f"image not found: {path_token}")
            image_path = Path(resolved)
        if image_path.suffix.lower() not in _IMAGE_EXTENSIONS:
            return _err(rid, 4016, f"unsupported image: {image_path.name}")
    except ImportError:
        # Fallback: simple path resolution.
        candidate = Path(raw).expanduser()
        if not candidate.is_file():
            return _err(rid, 4016, f"image not found: {raw}")
        if candidate.suffix.lower() not in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}:
            return _err(rid, 4016, f"unsupported image: {candidate.name}")
        image_path = candidate
        remainder = ""

    session.setdefault("attached_images", []).append(str(image_path))
    return _ok(rid, {
        "attached": True,
        "path": str(image_path),
        "count": len(session["attached_images"]),
        "remainder": remainder,
        "text": remainder or f"[User attached image: {image_path.name}]",
        **_image_meta(image_path),
    })


# Byte-upload attach caps. 25 MB matches Anthropic's per-image limit; 50 MB / 25
# pages bounds a single PDF drop so it can't blow the context budget.
_ATTACH_BYTES_MAX_BYTES = 25 * 1024 * 1024
_PDF_ATTACH_MAX_BYTES = 50 * 1024 * 1024
_PDF_ATTACH_MAX_PAGES = 25

# Leading magic bytes → file extension, for filename-less uploads.
_IMAGE_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", ".png"),
    (b"\xff\xd8\xff", ".jpg"),
    (b"GIF87a", ".gif"),
    (b"GIF89a", ".gif"),
    (b"BM", ".bmp"),
)


def _decode_attach_base64(raw: str, *, mime_prefix: str) -> bytes | None:
    """Decode a base64 (optionally data-URL-wrapped) payload.

    Ported from Hermes line 9130. Accepts ``data:<mime_prefix>...;base64,<b64>``
    plus embedded whitespace. Returns the decoded bytes, or ``None`` when the
    input isn't valid base64.
    """
    import base64 as _base64
    import re as _re

    cleaned = raw.strip()
    m = _re.match(
        rf"^data:{_re.escape(mime_prefix)}[a-zA-Z0-9.+-]*;base64,(.*)$",
        cleaned, _re.DOTALL,
    )
    if m:
        cleaned = m.group(1)
    cleaned = _re.sub(r"\s+", "", cleaned)
    try:
        return _base64.b64decode(cleaned, validate=True)
    except Exception:
        return None


def _sniff_image_ext(img_bytes: bytes, filename: str = "") -> str:
    """Resolve an image extension from a filename hint, else magic bytes.

    Ported from Hermes line 9154. Falls back to ``.png``.
    """
    if filename:
        suffix = Path(filename).suffix.lower()
        if suffix:
            return suffix
    head = img_bytes[:16]
    if head.startswith(b"RIFF") and head[8:12] == b"WEBP":
        return ".webp"
    for sig, ext in _IMAGE_MAGIC:
        if head.startswith(sig):
            return ext
    return ".png"


def _allowed_image_extensions() -> frozenset[str]:
    """Return the set of allowed image extensions.

    Ported from Hermes line 9173. Falls back to a static set when NIA's CLI
    module doesn't expose ``_IMAGE_EXTENSIONS``.
    """
    try:
        from niaharness.cli import _IMAGE_EXTENSIONS  # type: ignore

        return frozenset(_IMAGE_EXTENSIONS)
    except Exception:
        return frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"})


def _nia_home_path() -> Path:
    """Return the NIA home directory (~/.nia by default)."""
    try:
        from niaharness.prompts.soul import get_nia_home  # type: ignore

        return get_nia_home()
    except Exception:
        return Path(os.path.expanduser("~/.nia"))


def _queue_attached_image(session: dict, img_bytes: bytes, ext: str, *, prefix: str) -> Path:
    """Write image bytes into the gateway's images dir and queue them.

    Ported from Hermes line 9182. Mirrors what ``image.attach`` does for a
    local path: appends to ``session["attached_images"]`` so the next
    ``prompt.submit`` picks it up. Returns the written path.
    """
    session["image_counter"] = session.get("image_counter", 0) + 1
    img_dir = _nia_home_path() / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    img_path = img_dir / f"{prefix}_{ts}_{session['image_counter']}{ext}"
    try:
        img_path.write_bytes(img_bytes)
    except Exception:
        session["image_counter"] = max(0, session["image_counter"] - 1)
        raise
    session.setdefault("attached_images", []).append(str(img_path))
    return img_path


@method("image.attach_bytes")
def _image_attach_bytes(rid, params):
    """Attach an image to the session from base64 bytes (remote-client path).

    Deep-ported from Hermes line 9203. A desktop app or web dashboard running
    on a DIFFERENT machine than the gateway can't hand us a local path — that
    file only exists on the client's disk. So it uploads the raw image bytes
    (base64) and we write them into the gateway's own images dir.
    """
    session, err = _sess(params, rid)
    if err:
        return err

    raw_b64 = str(params.get("content_base64") or params.get("data") or "").strip()
    if not raw_b64:
        return _err(rid, 4015, "content_base64 required")

    img_bytes = _decode_attach_base64(raw_b64, mime_prefix="image/")
    if img_bytes is None:
        return _err(rid, 4017, "data is not valid base64")
    if not img_bytes:
        return _err(rid, 4017, "image is empty")
    if len(img_bytes) > _ATTACH_BYTES_MAX_BYTES:
        mb = _ATTACH_BYTES_MAX_BYTES // (1024 * 1024)
        return _err(rid, 4018, f"image too large ({len(img_bytes)} bytes; cap is {mb} MB)")

    filename = str(params.get("filename", "") or "")
    ext_hint = str(params.get("ext", "") or "").strip().lower()
    if ext_hint and not ext_hint.startswith("."):
        ext_hint = "." + ext_hint
    ext = _sniff_image_ext(img_bytes, filename or (f"x{ext_hint}" if ext_hint else ""))
    if ext not in _allowed_image_extensions():
        return _err(rid, 4016, f"unsupported image extension: {ext}")

    try:
        img_path = _queue_attached_image(session, img_bytes, ext, prefix="upload")
    except Exception as e:
        return _err(rid, 5027, f"write failed: {e}")

    return _ok(rid, {
        "attached": True,
        "path": str(img_path),
        "count": len(session["attached_images"]),
        "remainder": "",
        "text": f"[User attached image: {img_path.name}]",
        "bytes": len(img_bytes),
        **_image_meta(img_path),
    })


@method("image.detach")
def _image_detach(rid, params):
    """Detach a previously-attached image by path.

    Deep-ported from Hermes line 9578.
    """
    session, err = _sess(params, rid)
    if err:
        return err
    raw = str(params.get("path", "") or "").strip()
    if not raw:
        return _err(rid, 4015, "path required")
    images = session.setdefault("attached_images", [])
    before = len(images)
    session["attached_images"] = [path for path in images if path != raw]
    return _ok(rid, {
        "detached": len(session["attached_images"]) != before,
        "count": len(session["attached_images"]),
    })


@method("pdf.attach")
def _pdf_attach(rid, params):
    """Attach a PDF by rendering each page to PNG and queuing the pages.

    Deep-ported from Hermes line 9264. Anthropic's vision pipeline accepts
    images, not PDFs, so this runs ``pdftoppm`` (poppler-utils) at 150 DPI per
    page and queues each rendered page as an attached image. Accepts either a
    host ``path`` (local mode) or base64 ``content_base64`` (remote upload).
    Caps at 50 MB / 25 pages per call.

    Requires ``pdftoppm`` on $PATH (``apt install poppler-utils``); returns
    5028 if missing.
    """
    import tempfile

    session, err = _sess(params, rid)
    if err:
        return err

    if shutil.which("pdftoppm") is None:
        return _err(rid, 5028, "pdftoppm not installed (poppler-utils package required)")

    raw_path = str(params.get("path", "") or "").strip()
    raw_b64 = str(params.get("content_base64") or params.get("data") or "").strip()
    if not raw_path and not raw_b64:
        return _err(rid, 4015, "path or content_base64 required")

    with tempfile.TemporaryDirectory(prefix="pdf_attach_") as td:
        td_path = Path(td)
        if raw_b64:
            pdf_bytes = _decode_attach_base64(raw_b64, mime_prefix="application/pdf")
            if pdf_bytes is None:
                return _err(rid, 4017, "data is not valid base64")
            if not pdf_bytes:
                return _err(rid, 4017, "decoded PDF is empty")
            if len(pdf_bytes) > _PDF_ATTACH_MAX_BYTES:
                mb = _PDF_ATTACH_MAX_BYTES // (1024 * 1024)
                return _err(rid, 4018, f"PDF too large ({len(pdf_bytes)} bytes; cap is {mb} MB)")
            if pdf_bytes[:5] != b"%PDF-":
                return _err(rid, 4017, "payload is not a PDF (missing %PDF- magic bytes)")
            pdf_path = td_path / "input.pdf"
            pdf_path.write_bytes(pdf_bytes)
            display_name = str(params.get("filename", "") or "uploaded.pdf")
        else:
            resolved = Path(raw_path).expanduser()
            if not resolved.is_file():
                return _err(rid, 4016, f"PDF not found: {raw_path}")
            if resolved.suffix.lower() != ".pdf":
                return _err(rid, 4016, f"not a PDF: {resolved.name}")
            if resolved.stat().st_size > _PDF_ATTACH_MAX_BYTES:
                mb = _PDF_ATTACH_MAX_BYTES // (1024 * 1024)
                return _err(rid, 4018, f"PDF too large; cap is {mb} MB")
            pdf_path = resolved
            display_name = pdf_path.name

        try:
            first_page = int(params.get("first_page") or 1)
            last_page_param = params.get("last_page")
            last_page = int(last_page_param) if last_page_param is not None else None
        except (TypeError, ValueError):
            return _err(rid, 4015, "first_page/last_page must be integers")

        if first_page < 1:
            return _err(rid, 4015, "first_page must be >= 1")
        if last_page is None:
            last_page = first_page + _PDF_ATTACH_MAX_PAGES - 1
        if last_page < first_page:
            return _err(rid, 4015, "last_page must be >= first_page")
        if last_page - first_page + 1 > _PDF_ATTACH_MAX_PAGES:
            return _err(rid, 4019, f"page range exceeds cap of {_PDF_ATTACH_MAX_PAGES} pages per attach call")

        out_prefix = td_path / "page"
        argv = [
            "pdftoppm", "-png", "-r", "150",
            "-f", str(first_page), "-l", str(last_page),
            str(pdf_path), str(out_prefix),
        ]
        try:
            res = subprocess.run(
                argv, capture_output=True, text=True, timeout=120,
                stdin=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired:
            return _err(rid, 5028, "pdftoppm timed out (>120s)")
        if res.returncode != 0:
            tail = (res.stderr or res.stdout or "").strip().splitlines()[-3:]
            return _err(rid, 5028, "pdftoppm failed: " + " | ".join(tail))

        rendered = sorted(td_path.glob("page-*.png"))
        if not rendered:
            return _err(rid, 5028, "pdftoppm produced no pages (corrupt PDF?)")

        attached_pages = []
        for src in rendered:
            page_num = src.stem.split("-", 1)[-1]
            try:
                page_int = int(page_num)
            except ValueError:
                page_int = first_page + len(attached_pages)
            dst = _queue_attached_image(
                session, src.read_bytes(), ".png", prefix=f"pdf_p{page_num}",
            )
            attached_pages.append({
                "path": str(dst), "page": page_int, **_image_meta(dst),
            })

        return _ok(rid, {
            "attached": True,
            "filename": display_name,
            "pages_attached": len(attached_pages),
            "pages": attached_pages,
            "count": len(session["attached_images"]),
            "text": f"[User attached PDF: {display_name} ({len(attached_pages)} page(s))]",
        })


# File-attachment helpers (ported from Hermes lines 9387-9528).


def _format_ref_value(value: str) -> str:
    """Quote a context-ref value when it contains whitespace or bracket chars.

    Ported from Hermes line 9390. Mirrors the desktop ``formatRefValue`` so
    the staged ``@file:`` ref round-trips through context_references cleanly.
    """
    import re as _re

    if not value:
        return value
    needs_quoting = _re.compile(r"""[\s()\[\]{}<>"'`]""")
    if not needs_quoting.search(value):
        return value
    if "`" not in value:
        return f"`{value}`"
    if '"' not in value:
        return f'"{value}"'
    if "'" not in value:
        return f"'{value}'"
    return value


def _attachment_ref_path(session: dict, target: Path) -> str:
    """Workspace-relative path for an attachment, or absolute if outside.

    Ported from Hermes line 9412.
    """
    workspace = Path(_session_cwd(session)).resolve()
    try:
        rel = target.resolve().relative_to(workspace)
        return str(rel).replace(os.sep, "/")
    except ValueError:
        return str(target.resolve())


def _desktop_attachment_dir(session: dict) -> Path:
    """Return the per-session desktop-attachments dir under the workspace.

    Ported from Hermes line 9422.
    """
    root = Path(_session_cwd(session)).resolve() / ".nia" / "desktop-attachments"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _sanitize_attachment_name(name: str) -> str:
    """Sanitize a filename for safe storage.

    Ported from Hermes line 9428.
    """
    import re as _re

    candidate = Path(str(name or "").strip()).name
    candidate = _re.sub(r"[\x00-\x1f]+", "_", candidate)
    candidate = candidate.strip().strip(".")
    return candidate or "attachment"


def _unique_attachment_path(root: Path, filename: str) -> Path:
    """Return a non-clobbering path under ``root`` for ``filename``.

    Ported from Hermes line 9437.
    """
    candidate = root / filename
    if not candidate.exists():
        return candidate
    stem = Path(filename).stem or "attachment"
    suffix = Path(filename).suffix
    counter = 2
    while True:
        next_candidate = root / f"{stem}-{counter}{suffix}"
        if not next_candidate.exists():
            return next_candidate
        counter += 1


def _decode_attachment_data_url(data_url: str) -> bytes:
    """Decode a ``data:<any-mime>;base64,<b64>`` payload to bytes.

    Ported from Hermes line 9468. Unlike ``_decode_attach_base64``
    (image-mime-specific), this accepts any media type — text/csv,
    application/pdf, etc. — so non-image file uploads round-trip. Also
    tolerates a bare base64 string with no data-URL prefix.
    """
    import base64 as _base64
    import binascii as _binascii
    import re as _re

    cleaned = (data_url or "").strip()
    m = _re.match(r"^data:[^;,]*(?:;[^;,=]+=[^;,]+)*;base64,(.*)$", cleaned, _re.DOTALL | _re.I)
    if m:
        cleaned = m.group(1)
    cleaned = _re.sub(r"\s+", "", cleaned)
    try:
        return _base64.b64decode(cleaned, validate=True)
    except (ValueError, _binascii.Error) as exc:
        raise ValueError("invalid data_url payload") from exc


def _stage_session_file_attachment(
    session: dict, *, raw_path: str, data_url: str, name: str,
) -> tuple[Path, bool]:
    """Make a desktop file attachment available to the remote gateway agent.

    Ported from Hermes line 9490. Three cases:
      1. The path resolves to a file already INSIDE the session workspace —
         use it as-is (no copy, ``uploaded=False``).
      2. The path resolves to a gateway-visible file OUTSIDE the workspace —
         copy it into ``.nia/desktop-attachments/`` so the ``@file:`` ref
         resolves.
      3. The path doesn't exist on the gateway (the common remote case: it's
         a path on the CLIENT's disk) — decode the uploaded ``data_url``
         bytes and write them into ``.nia/desktop-attachments/``.
    Returns ``(stored_path, uploaded)``.
    """
    workspace = Path(_session_cwd(session)).resolve()
    # Try resolving the raw path as a gateway-visible file.
    resolved: Path | None = None
    candidate = Path(raw_path).expanduser()
    if candidate.is_file():
        resolved = candidate.resolve()
    if resolved is not None:
        try:
            resolved.relative_to(workspace)
            return resolved, False
        except ValueError:
            payload = resolved.read_bytes()
            filename = resolved.name
    else:
        if not data_url:
            raise ValueError("file not found on gateway and no data_url provided")
        payload = _decode_attachment_data_url(data_url)
        filename = _sanitize_attachment_name(name or Path(str(raw_path or "")).name)

    upload_dir = _desktop_attachment_dir(session)
    target = _unique_attachment_path(upload_dir, _sanitize_attachment_name(filename))
    target.write_bytes(payload)
    return target.resolve(), True


@method("file.attach")
def _file_attach(rid, params):
    """Stage a non-image file attachment into the session workspace.

    Deep-ported from Hermes line 9531. The image/PDF path renders to vision
    tiles; this one keeps the file as a readable artifact and returns a
    workspace-relative ``@file:`` ref so the agent's file tools (and
    ``context_references``) can read it. Solves the remote-gateway case where
    the desktop passes a path that only exists on the CLIENT's disk: the
    client uploads ``data_url`` bytes and we materialize the file on the
    gateway.
    """
    session, err = _sess(params, rid)
    if err:
        return err
    raw = str(params.get("path", "") or "").strip()
    data_url = str(params.get("data_url", "") or "").strip()
    name = str(params.get("name", "") or "").strip()
    if not raw and not data_url:
        return _err(rid, 4015, "path or data_url required")
    try:
        stored_path, uploaded = _stage_session_file_attachment(
            session, raw_path=raw, data_url=data_url, name=name,
        )
        ref_path = _attachment_ref_path(session, stored_path)
        return _ok(rid, {
            "attached": True,
            "name": stored_path.name,
            "path": str(stored_path),
            "ref_path": ref_path,
            "ref_text": f"@file:{_format_ref_value(ref_path)}",
            "uploaded": uploaded,
        })
    except Exception as e:
        return _err(rid, 5028, str(e))


@method("clipboard.paste")
def _clipboard_paste(rid, params):
    """Save a clipboard image (if present) and attach it to the session.

    Deep-ported from Hermes line 9031. NIA may not yet expose
    ``niaharness.cli.clipboard.has_clipboard_image`` /
    ``save_clipboard_image`` — return a clear "clipboard unavailable" error
    so the TUI can fall back to text paste. TODO(feature-gap): see
    FEATURE_GAPS.md (cli.clipboard).
    """
    session, err = _sess(params, rid)
    if err:
        return err
    try:
        from niaharness.cli.clipboard import (  # type: ignore
            has_clipboard_image,
            save_clipboard_image,
        )
    except ImportError:
        # TODO(feature-gap): see FEATURE_GAPS.md (cli.clipboard).
        # Minimal fallback: try xclip on Linux.
        try:
            result = subprocess.run(
                ["xclip", "-selection", "clipboard", "-t", "image/png", "-o"],
                capture_output=True, timeout=2, stdin=subprocess.DEVNULL,
            )
            if result.returncode == 0 and result.stdout:
                session["image_counter"] = session.get("image_counter", 0) + 1
                img_dir = _nia_home_path() / "images"
                img_dir.mkdir(parents=True, exist_ok=True)
                img_path = (
                    img_dir / f"clip_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    f"_{session['image_counter']}.png"
                )
                img_path.write_bytes(result.stdout)
                session.setdefault("attached_images", []).append(str(img_path))
                return _ok(rid, {
                    "attached": True,
                    "path": str(img_path),
                    "count": len(session["attached_images"]),
                    **_image_meta(img_path),
                })
        except Exception:
            pass
        return _err(rid, 5027, "clipboard unavailable: niaharness.cli.clipboard not found")

    session["image_counter"] = session.get("image_counter", 0) + 1
    img_dir = _nia_home_path() / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    img_path = (
        img_dir / f"clip_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        f"_{session['image_counter']}.png"
    )

    # Save-first: mirrors CLI keybinding path; more robust than has_image() precheck.
    if not save_clipboard_image(img_path):
        session["image_counter"] = max(0, session["image_counter"] - 1)
        msg = (
            "Clipboard has image but extraction failed"
            if has_clipboard_image()
            else "No image found in clipboard"
        )
        return _ok(rid, {"attached": False, "message": msg})

    session.setdefault("attached_images", []).append(str(img_path))
    return _ok(rid, {
        "attached": True,
        "path": str(img_path),
        "count": len(session["attached_images"]),
        **_image_meta(img_path),
    })


@method("input.detect_drop")
def _input_detect_drop(rid, params):
    """Detect a file-drop pattern in the input text and attach if matched.

    Deep-ported from Hermes line 9598. NIA's CLI may not yet expose
    ``_detect_file_drop`` — return ``{matched: False}`` so the TUI falls
    through to normal text submission. TODO(feature-gap): see FEATURE_GAPS.md
    (cli._detect_file_drop).
    """
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    try:
        from niaharness.cli import _detect_file_drop  # type: ignore

        raw = str(params.get("text", "") or "")
        dropped = _detect_file_drop(raw)
        if not dropped:
            return _ok(rid, {"matched": False})

        drop_path = Path(dropped["path"])
        remainder = dropped.get("remainder", "")
        if dropped.get("is_image"):
            session.setdefault("attached_images", []).append(str(drop_path))
            text = remainder or f"[User attached image: {drop_path.name}]"
            return _ok(rid, {
                "matched": True,
                "is_image": True,
                "path": str(drop_path),
                "count": len(session["attached_images"]),
                "text": text,
                **_image_meta(drop_path),
            })

        text = f"[User attached file: {drop_path}]" + (
            f"\n{remainder}" if remainder else ""
        )
        return _ok(rid, {
            "matched": True,
            "is_image": False,
            "path": str(drop_path),
            "name": drop_path.name,
            "text": text,
        })
    except ImportError:
        # TODO(feature-gap): see FEATURE_GAPS.md (cli._detect_file_drop).
        return _ok(rid, {"matched": False})
    except Exception as e:
        return _err(rid, 5027, str(e))


_paste_counter = 0


@method("paste.collapse")
def _paste_collapse(rid, params):
    """Collapse a multi-line paste into a placeholder + write to ~/.nia/pastes/.

    Deep-ported from Hermes line 11861. Long pastes bloat the composer and the
    stored transcript; this writes the full text to a per-paste file and
    returns a short placeholder the user can submit instead. The agent reads
    the full contents via the file path in the placeholder.
    """
    global _paste_counter
    text = params.get("text", "")
    if not text:
        return _err(rid, 4004, "empty paste")

    _paste_counter += 1
    line_count = text.count("\n") + 1

    # Resolve the paste dir under the NIA home.
    try:
        from niaharness.prompts.soul import get_nia_home  # type: ignore

        nia_home = get_nia_home()
    except Exception:
        nia_home = Path(os.path.expanduser("~/.nia"))

    paste_dir = nia_home / "pastes"
    try:
        paste_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        return _err(rid, 5003, f"failed to create paste dir: {e}")

    paste_file = (
        paste_dir / f"paste_{_paste_counter}_{datetime.now().strftime('%H%M%S')}.txt"
    )
    try:
        paste_file.write_text(text, encoding="utf-8")
    except Exception as e:
        return _err(rid, 5003, f"failed to write paste file: {e}")

    placeholder = (
        f"[Pasted text #{_paste_counter}: {line_count} lines \u2192 {paste_file}]"
    )
    return _ok(rid, {
        "placeholder": placeholder,
        "path": str(paste_file),
        "lines": line_count,
    })


# ---------------------------------------------------------------------------
# Config methods
# ---------------------------------------------------------------------------


@method("config.get")
def _config_get(rid, params):
    """Get a config key with full key-specific resolution.

    Deep-ported from Hermes line 10822. Handles:
    - ``provider`` — model + provider + available providers list.
    - ``profile`` — current NIA home dir.
    - ``project`` — cwd + git branch for the current project.
    - ``full`` — the entire config dict.
    - ``prompt`` / ``skin`` / ``indicator`` / ``personality`` — display prefs.
    - ``reasoning`` — live session reasoning_config (falls back to global).
    - ``fast`` — service tier (session agent's, then global).
    - ``busy`` — busy_input_mode.
    - ``details_mode`` / ``thinking_mode`` — display section collapse state.
    - ``compact`` / ``statusbar`` / ``mouse`` — TUI display prefs.
    - ``mtime`` — config.yaml file mtime (for change detection).
    """
    key = params.get("key", "")

    if key == "provider":
        try:
            from niaharness.providers import (  # type: ignore
                list_available_providers,
                normalize_provider,
            )

            model = _resolve_model()
            parts = model.split("/", 1)
            return _ok(rid, {
                "model": model,
                "provider": (
                    normalize_provider(parts[0]) if len(parts) > 1 else "unknown"
                ),
                "providers": list_available_providers(),
            })
        except ImportError:
            # TODO(feature-gap): see FEATURE_GAPS.md (providers.list_available_providers).
            return _ok(rid, {
                "model": _resolve_model(),
                "provider": "unknown",
                "providers": [],
            })
        except Exception as e:
            return _err(rid, 5013, str(e))

    if key == "profile":
        try:
            from niaharness.prompts.soul import get_nia_home  # type: ignore

            home = str(get_nia_home())
        except Exception:
            home = os.path.expanduser("~/.nia")
        return _ok(rid, {"home": home, "display": home})

    if key == "project":
        cfg_terminal = _load_cfg().get("terminal") or {}
        raw = str(params.get("cwd", "") or cfg_terminal.get("cwd", "") or "").strip()
        cwd = _completion_cwd({"cwd": raw} if raw else {})
        return _ok(rid, {"cwd": cwd, "branch": _git_branch_for_cwd(cwd)})

    if key == "full":
        return _ok(rid, {"config": _load_cfg()})

    if key == "prompt":
        return _ok(rid, {"prompt": _load_cfg().get("custom_prompt", "")})

    if key == "skin":
        return _ok(rid, {"value": (_load_cfg().get("display") or {}).get("skin", "default")})

    if key == "indicator":
        _INDICATOR_STYLES_LOCAL = ("ascii", "emoji", "kaomoji", "unicode")
        _INDICATOR_DEFAULT_LOCAL = "kaomoji"
        raw = (_load_cfg().get("display") or {}).get("tui_status_indicator", "")
        norm = str(raw).strip().lower()
        return _ok(rid, {
            "value": norm if norm in _INDICATOR_STYLES_LOCAL else _INDICATOR_DEFAULT_LOCAL,
        })

    if key == "personality":
        return _ok(rid, {
            "value": (_load_cfg().get("display") or {}).get("personality") or "none",
        })

    if key == "reasoning":
        cfg = _load_cfg()
        effort = ""
        session = _sessions.get(params.get("session_id", ""))
        live = getattr((session or {}).get("agent"), "reasoning_config", None)
        if live is None and session is not None:
            live = session.get("create_reasoning_override")
        if isinstance(live, dict):
            if live.get("enabled") is False:
                effort = "none"
            else:
                effort = str(live.get("effort", "") or "")
        if not effort:
            raw_effort = (cfg.get("agent") or {}).get("reasoning_effort", "")
            if raw_effort is False:
                effort = "none"
            else:
                effort = str(raw_effort or "medium")
        display = (
            "show"
            if bool((cfg.get("display") or {}).get("show_reasoning", True))
            else "hide"
        )
        return _ok(rid, {"value": effort, "display": display})

    if key == "fast":
        session = _sessions.get(params.get("session_id", ""))
        agent_fast = (
            getattr(session.get("agent"), "service_tier", None) == "priority"
            if session and session.get("agent")
            else False
        )
        if agent_fast:
            nv = "fast"
        else:
            nv = "fast" if _load_service_tier() == "priority" else "normal"
        return _ok(rid, {"value": nv})

    if key == "busy":
        return _ok(rid, {"value": _load_busy_input_mode()})

    if key == "details_mode":
        allowed_dm = frozenset({"hidden", "collapsed", "expanded"})
        raw = str(
            (_load_cfg().get("display") or {}).get("details_mode", "collapsed") or "collapsed"
        ).strip().lower()
        nv = raw if raw in allowed_dm else "collapsed"
        return _ok(rid, {"value": nv})

    if key == "thinking_mode":
        allowed_tm = frozenset({"collapsed", "truncated", "full"})
        cfg = _load_cfg()
        raw = str((cfg.get("display") or {}).get("thinking_mode", "") or "").strip().lower()
        if raw in allowed_tm:
            nv = raw
        else:
            dm = str(
                (cfg.get("display") or {}).get("details_mode", "collapsed") or "collapsed"
            ).strip().lower()
            nv = "full" if dm == "expanded" else "collapsed"
        return _ok(rid, {"value": nv})

    if key == "compact":
        on = bool((_load_cfg().get("display") or {}).get("tui_compact", False))
        return _ok(rid, {"value": "on" if on else "off"})

    if key == "statusbar":
        display = _load_cfg().get("display")
        raw = display.get("tui_statusbar", "top") if isinstance(display, dict) else "top"
        return _ok(rid, {"value": _coerce_statusbar(raw)})

    if key == "mouse":
        display = _load_cfg().get("display")
        return _ok(rid, {"value": _display_mouse_tracking(display)})

    if key == "mtime":
        try:
            from niaharness.prompts.soul import get_nia_home  # type: ignore

            cfg_path = get_nia_home() / "config.yaml"
        except Exception:
            cfg_path = Path(os.path.expanduser("~/.nia/config.yaml"))
        try:
            return _ok(rid, {"mtime": cfg_path.stat().st_mtime if cfg_path.exists() else 0})
        except Exception:
            return _ok(rid, {"mtime": 0})

    # Default: return full config.
    if not key:
        return _ok(rid, {"config": _load_cfg()})
    return _err(rid, 4002, f"unknown config key: {key}")


@method("config.set")
def _config_set(rid, params):
    """Set a config key with full key-specific side-effects.

    Deep-ported from Hermes line 9865. Handles:
    - ``model`` — model switch (with running-guard, agent-switch, session override).
    - ``fast`` — service tier toggle (fast/normal) with agent + config persist.
    - ``busy`` — busy_input_mode (queue/steer/interrupt).
    - ``verbose`` — tool_progress cycle (off/new/all/verbose).
    - ``yolo`` — approval bypass toggle (session or global scope).
    - ``reasoning`` — reasoning effort + show/hide/full/clamp display.
    - ``details_mode`` + ``details_mode.<section>`` — collapse/expand sections.
    - ``thinking_mode`` — collapsed/truncated/full.
    - ``compact`` — TUI compact mode toggle.
    - ``statusbar`` — TUI statusbar (top/off/etc.).
    - ``mouse`` — mouse tracking mode.
    - ``indicator`` — indicator style (ascii/emoji/kaomoji/unicode).
    - ``cwd`` / ``terminal.cwd`` / ``workdir`` — change session cwd.
    - ``prompt`` / ``personality`` / ``skin`` — custom prompt / personality / skin.
    - Generic dotted-key fallback for anything else.
    """
    key, value = params.get("key", ""), params.get("value", "")
    session = _sessions.get(params.get("session_id", ""))

    if not key:
        return _err(rid, 4002, "key is required")

    # ── model ─────────────────────────────────────────────────────────
    if key == "model":
        try:
            if not value:
                return _err(rid, 4002, "model value required")
            if session:
                if session.get("running"):
                    return _err(
                        rid, 4009,
                        "session busy — /interrupt the current turn before switching models",
                    )
                # Build the agent if not yet built so the switch lands on a
                # live agent (mirrors Hermes line 9893).
                if session.get("agent") is None:
                    sid = params.get("session_id", "")
                    _start_agent_build(sid, session)
                    init_err = _wait_agent(session, rid)
                    if init_err:
                        return init_err
                    if session.get("agent") is None:
                        return _err(rid, 5032, "agent initialization failed")
                result = _apply_model_switch(
                    params.get("session_id", ""), session, value,
                    confirm_expensive_model=bool(params.get("confirm_expensive_model", False)),
                )
            else:
                result = _apply_model_switch(
                    "", {"agent": None}, value,
                    confirm_expensive_model=bool(params.get("confirm_expensive_model", False)),
                )
            return _ok(rid, {
                "key": key,
                "value": result["value"],
                "warning": result.get("warning", ""),
                "confirm_required": result.get("confirm_required", False),
                "confirm_message": result.get("confirm_message", ""),
            })
        except Exception as e:
            return _err(rid, 5001, str(e))

    # ── fast (service tier) ───────────────────────────────────────────
    if key == "fast":
        raw = str(value or "").strip().lower()
        agent = session.get("agent") if session else None
        if agent is not None:
            current_fast = getattr(agent, "service_tier", None) == "priority"
        else:
            current_fast = _load_service_tier() == "priority"

        if raw in {"status"}:
            return _ok(rid, {"key": key, "value": "fast" if current_fast else "normal"})

        if raw in {"", "toggle"}:
            nv = "normal" if current_fast else "fast"
        elif raw in {"fast", "on"}:
            nv = "fast"
        elif raw in {"normal", "off"}:
            nv = "normal"
        else:
            return _err(rid, 4002, f"unknown fast mode: {value}")

        # NIA does not yet expose ``resolve_fast_mode_overrides`` — skip the
        # per-model speed overrides. TODO(feature-gap): see FEATURE_GAPS.md.
        overrides = None
        if nv == "fast":
            try:
                from niaharness.engine.model_metadata import (  # type: ignore
                    resolve_fast_mode_overrides,
                )

                target_model = (
                    getattr(agent, "model", None) if agent is not None else _resolve_model()
                )
                if not target_model:
                    return _err(rid, 4002, "fast mode is not available without a selected model")
                overrides = resolve_fast_mode_overrides(target_model)
                if overrides is None:
                    return _err(rid, 4002, "fast mode is not available for this model")
            except ImportError:
                overrides = None

        _write_config_key("agent.service_tier", nv)
        if agent is not None:
            agent.service_tier = "priority" if nv == "fast" else None
            current_overrides = dict(getattr(agent, "request_overrides", {}) or {})
            current_overrides.pop("service_tier", None)
            current_overrides.pop("speed", None)
            if nv == "fast" and overrides:
                current_overrides.update(overrides)
            agent.request_overrides = current_overrides
            _persist_live_session_runtime(session)
            _emit("session.info", params.get("session_id", ""), _session_info(agent, session))
        return _ok(rid, {"key": key, "value": nv})

    # ── busy (busy_input_mode) ────────────────────────────────────────
    if key == "busy":
        raw = str(value or "").strip().lower()
        if raw in {"", "status"}:
            return _ok(rid, {"key": key, "value": _load_busy_input_mode()})
        if raw not in {"queue", "steer", "interrupt"}:
            return _err(rid, 4002, f"unknown busy mode: {value}")
        _write_config_key("display.busy_input_mode", raw)
        return _ok(rid, {"key": key, "value": raw})

    # ── verbose (tool_progress) ───────────────────────────────────────
    if key == "verbose":
        cycle = ["off", "new", "all", "verbose"]
        cur = (
            session.get("tool_progress_mode", _load_tool_progress_mode())
            if session
            else _load_tool_progress_mode()
        )
        if value and value != "cycle":
            nv = str(value).strip().lower()
            if nv not in cycle:
                return _err(rid, 4002, f"unknown verbose mode: {value}")
        else:
            try:
                idx = cycle.index(cur)
            except ValueError:
                idx = 2
            nv = cycle[(idx + 1) % len(cycle)]
        _write_config_key("display.tool_progress", nv)
        if session:
            session["tool_progress_mode"] = nv
            agent = session.get("agent")
            if agent is not None:
                agent.verbose_logging = nv == "verbose"
        return _ok(rid, {"key": key, "value": nv})

    # ── yolo (approval bypass) ────────────────────────────────────────
    if key == "yolo":
        scope = str(params.get("scope") or "session").strip().lower()
        try:
            from niaharness.permissions.approval import (
                disable_session_yolo,
                enable_session_yolo,
                is_session_yolo_enabled,
            )

            raw = str(value or "").strip().lower()

            def _resolve_toggle(current: bool) -> bool:
                if raw in {"1", "on", "true", "yes"}:
                    return True
                if raw in {"0", "off", "false", "no"}:
                    return False
                return not current

            if scope == "global":
                cfg = _load_cfg()
                appr = cfg.get("approvals") if isinstance(cfg, dict) else None
                if not isinstance(appr, dict):
                    appr = {}
                current_mode = str(appr.get("mode", "manual") or "manual").lower()
                current = current_mode == "off"
                enable = _resolve_toggle(current)
                _write_config_key("approvals.mode", "off" if enable else "manual")
                nv = "1" if enable else "0"
                for sid, sess in list(_sessions.items()):
                    agent = sess.get("agent")
                    if agent is not None:
                        _emit("session.info", sid, _session_info(agent, sess))
                return _ok(rid, {"key": key, "value": nv, "scope": "global"})

            if session:
                current = is_session_yolo_enabled(session["session_key"])
                enable = _resolve_toggle(current)
                if enable:
                    enable_session_yolo(session["session_key"])
                    nv = "1"
                else:
                    disable_session_yolo(session["session_key"])
                    nv = "0"
                agent = session.get("agent")
                if agent is not None:
                    _emit(
                        "session.info", params.get("session_id", ""),
                        _session_info(agent, session),
                    )
            else:
                current = _is_truthy(os.environ.get("NIA_YOLO_MODE") or os.environ.get("HERMES_YOLO_MODE"))
                enable = _resolve_toggle(current)
                if enable:
                    os.environ["NIA_YOLO_MODE"] = "1"
                    nv = "1"
                else:
                    os.environ.pop("NIA_YOLO_MODE", None)
                    os.environ.pop("HERMES_YOLO_MODE", None)
                    nv = "0"
            return _ok(rid, {"key": key, "value": nv, "scope": "session"})
        except Exception as e:
            return _err(rid, 5001, str(e))

    # ── reasoning ─────────────────────────────────────────────────────
    if key == "reasoning":
        try:
            arg = str(value or "").strip().lower()
            if arg in {"show", "on"}:
                cfg = _load_cfg()
                display = cfg.get("display") if isinstance(cfg.get("display"), dict) else {}
                sections = display.get("sections") if isinstance(display.get("sections"), dict) else {}
                display["show_reasoning"] = True
                sections["thinking"] = "expanded"
                display["sections"] = sections
                cfg["display"] = display
                _save_cfg(cfg)
                if session:
                    session["show_reasoning"] = True
                return _ok(rid, {"key": key, "value": "show"})
            if arg in {"hide", "off"}:
                cfg = _load_cfg()
                display = cfg.get("display") if isinstance(cfg.get("display"), dict) else {}
                sections = display.get("sections") if isinstance(display.get("sections"), dict) else {}
                display["show_reasoning"] = False
                sections["thinking"] = "hidden"
                display["sections"] = sections
                cfg["display"] = display
                _save_cfg(cfg)
                if session:
                    session["show_reasoning"] = False
                return _ok(rid, {"key": key, "value": "hide"})
            if arg in {"full", "all"}:
                cfg = _load_cfg()
                display = cfg.get("display") if isinstance(cfg.get("display"), dict) else {}
                sections = display.get("sections") if isinstance(display.get("sections"), dict) else {}
                display["reasoning_full"] = True
                sections["thinking"] = "expanded"
                display["sections"] = sections
                cfg["display"] = display
                _save_cfg(cfg)
                return _ok(rid, {"key": key, "value": "full"})
            if arg in {"clamp", "collapse", "short"}:
                cfg = _load_cfg()
                display = cfg.get("display") if isinstance(cfg.get("display"), dict) else {}
                sections = display.get("sections") if isinstance(display.get("sections"), dict) else {}
                display["reasoning_full"] = False
                sections["thinking"] = "collapsed"
                display["sections"] = sections
                cfg["display"] = display
                _save_cfg(cfg)
                return _ok(rid, {"key": key, "value": "clamp"})

            # NIA may not have parse_reasoning_effort — store raw value.
            parsed = None
            try:
                from niaharness.config.reasoning import parse_reasoning_effort  # type: ignore

                parsed = parse_reasoning_effort(arg)
            except ImportError:
                # TODO(feature-gap): see FEATURE_GAPS.md (parse_reasoning_effort).
                parsed = arg if arg else None
            if parsed is None:
                return _err(rid, 4002, f"unknown reasoning value: {value}")
            if session is not None:
                session["create_reasoning_override"] = parsed
                if session.get("agent") is not None:
                    session["agent"].reasoning_config = parsed
                    _persist_live_session_runtime(session)
                    _emit(
                        "session.info", params.get("session_id", ""),
                        _session_info(session["agent"], session),
                    )
            else:
                _write_config_key("agent.reasoning_effort", arg)
            return _ok(rid, {"key": key, "value": arg})
        except Exception as e:
            return _err(rid, 5001, str(e))

    # ── details_mode + details_mode.<section> ─────────────────────────
    _DETAIL_SECTION_NAMES_LOCAL = ("thinking", "tools", "subagents", "activity")
    _DETAIL_MODES_LOCAL = frozenset({"hidden", "collapsed", "expanded"})
    if key == "details_mode":
        nv = str(value or "").strip().lower()
        if nv not in _DETAIL_MODES_LOCAL:
            return _err(rid, 4002, f"unknown details_mode: {value}")
        cfg = _load_cfg()
        display = cfg.get("display") if isinstance(cfg.get("display"), dict) else {}
        sections = display.get("sections") if isinstance(display.get("sections"), dict) else {}
        display["details_mode"] = nv
        for section in _DETAIL_SECTION_NAMES_LOCAL:
            sections[section] = nv
        display["sections"] = sections
        cfg["display"] = display
        _save_cfg(cfg)
        return _ok(rid, {"key": key, "value": nv})

    if key.startswith("details_mode."):
        section = key.split(".", 1)[1]
        if section not in _DETAIL_SECTION_NAMES_LOCAL:
            return _err(rid, 4002, f"unknown section: {section}")
        cfg = _load_cfg()
        display = cfg.get("display") if isinstance(cfg.get("display"), dict) else {}
        sections_cfg = display.get("sections") if isinstance(display.get("sections"), dict) else {}
        nv = str(value or "").strip().lower()
        if not nv:
            sections_cfg.pop(section, None)
            display["sections"] = sections_cfg
            cfg["display"] = display
            _save_cfg(cfg)
            return _ok(rid, {"key": key, "value": ""})
        if nv not in _DETAIL_MODES_LOCAL:
            return _err(rid, 4002, f"unknown details_mode: {value}")
        sections_cfg[section] = nv
        display["sections"] = sections_cfg
        cfg["display"] = display
        _save_cfg(cfg)
        return _ok(rid, {"key": key, "value": nv})

    # ── thinking_mode ─────────────────────────────────────────────────
    if key == "thinking_mode":
        nv = str(value or "").strip().lower()
        allowed_tm = frozenset({"collapsed", "truncated", "full"})
        if nv not in allowed_tm:
            return _err(rid, 4002, f"unknown thinking_mode: {value}")
        _write_config_key("display.thinking_mode", nv)
        _write_config_key(
            "display.details_mode", "expanded" if nv == "full" else "collapsed"
        )
        return _ok(rid, {"key": key, "value": nv})

    # ── compact (TUI compact mode) ────────────────────────────────────
    if key == "compact":
        raw = str(value or "").strip().lower()
        cfg0 = _load_cfg()
        d0 = cfg0.get("display") if isinstance(cfg0.get("display"), dict) else {}
        cur_b = bool(d0.get("tui_compact", False))
        if raw in {"", "toggle"}:
            nv_b = not cur_b
        elif raw == "on":
            nv_b = True
        elif raw == "off":
            nv_b = False
        else:
            return _err(rid, 4002, f"unknown compact value: {value}")
        _write_config_key("display.tui_compact", nv_b)
        return _ok(rid, {"key": key, "value": "on" if nv_b else "off"})

    # ── statusbar ─────────────────────────────────────────────────────
    if key == "statusbar":
        raw = str(value or "").strip().lower()
        display = _load_cfg().get("display")
        d0 = display if isinstance(display, dict) else {}
        current = _coerce_statusbar(d0.get("tui_statusbar", "top"))
        _STATUSBAR_MODES = frozenset({"top", "bottom", "off"})
        if raw in {"", "toggle"}:
            nv = "top" if current == "off" else "off"
        elif raw == "on":
            nv = "top"
        elif raw in _STATUSBAR_MODES:
            nv = raw
        else:
            return _err(rid, 4002, f"unknown statusbar value: {value}")
        _write_config_key("display.tui_statusbar", nv)
        return _ok(rid, {"key": key, "value": nv})

    # ── mouse ─────────────────────────────────────────────────────────
    if key == "mouse":
        _MOUSE_TRACKING_ALIASES = {
            "on": "all", "all": "all", "yes": "all", "1": "all", "true": "all",
            "off": "off", "no": "off", "0": "off", "false": "off", "none": "off",
            "click": "click", "drag": "drag",
        }
        raw = ("" if value is None else str(value)).strip().lower()
        cfg = _load_cfg()
        display = cfg.get("display") if isinstance(cfg.get("display"), dict) else {}
        current = _display_mouse_tracking(display)
        if raw in {"", "toggle"}:
            nv = "all" if current == "off" else "off"
        elif raw in _MOUSE_TRACKING_ALIASES:
            nv = _MOUSE_TRACKING_ALIASES[raw]
        else:
            return _err(rid, 4002, f"unknown mouse value: {value}")
        _write_config_key("display.mouse_tracking", nv)
        return _ok(rid, {"key": key, "value": nv})

    # ── indicator ─────────────────────────────────────────────────────
    if key == "indicator":
        _INDICATOR_STYLES_LOCAL = ("ascii", "emoji", "kaomoji", "unicode")
        raw = ("" if value is None else str(value)).strip().lower()
        if raw not in _INDICATOR_STYLES_LOCAL:
            return _err(
                rid, 4002,
                f"unknown indicator: {raw!r}; pick one of {'|'.join(_INDICATOR_STYLES_LOCAL)}",
            )
        _write_config_key("display.tui_status_indicator", raw)
        return _ok(rid, {"key": key, "value": raw})

    # ── cwd / terminal.cwd / workdir ──────────────────────────────────
    if key in {"cwd", "terminal.cwd", "workdir"}:
        raw = str(value or "").strip()
        if not raw:
            return _err(rid, 4002, "cwd required")
        cwd = os.path.abspath(os.path.expanduser(raw))
        if not os.path.isdir(cwd):
            return _err(rid, 4002, f"working directory does not exist: {raw}")
        _write_config_key("terminal.cwd", cwd)
        os.environ["TERMINAL_CWD"] = cwd
        return _ok(rid, {
            "key": "terminal.cwd", "value": cwd,
            "cwd": cwd, "branch": _git_branch_for_cwd(cwd),
        })

    # ── prompt / personality / skin ───────────────────────────────────
    if key in {"prompt", "personality", "skin"}:
        try:
            cfg = _load_cfg()
            if key == "prompt":
                if value == "clear":
                    cfg.pop("custom_prompt", None)
                    nv = ""
                else:
                    cfg["custom_prompt"] = value
                    nv = value
                _save_cfg(cfg)
            elif key == "personality":
                sid_key = params.get("session_id", "")
                pname, new_prompt = _validate_personality(str(value or ""), cfg)
                _write_config_key("display.personality", pname)
                _write_config_key("agent.system_prompt", new_prompt)
                nv = str(value or "none")
                history_reset, info = _apply_personality_to_session(
                    sid_key, session, new_prompt, pname
                )
            else:
                _write_config_key(f"display.{key}", value)
                nv = value
                if key == "skin":
                    _emit("skin.changed", "", resolve_skin())
            resp: dict = {"key": key, "value": nv}
            if key == "personality":
                resp["history_reset"] = history_reset
                if info is not None:
                    resp["info"] = info
            return _ok(rid, resp)
        except Exception as e:
            return _err(rid, 5001, str(e))

    # ── Generic dotted-key fallback ───────────────────────────────────
    _save_cfg_key(key, value)
    if key.startswith("display.") and session:
        _emit(
            "session.info", params.get("session_id", ""),
            _session_info(session.get("agent"), session),
        )
    return _ok(rid, {"key": key, "value": str(value), "warning": ""})


def _write_config_key(dotted_key: str, value: Any) -> None:
    """Write a dotted-key path into config.yaml (e.g. ``display.tool_progress``).

    Ported from Hermes line 2313. Splits the key on ``.``, walks the config
    dict creating sub-dicts as needed, and saves.
    """
    cfg = _load_cfg()
    parts = dotted_key.split(".")
    d = cfg
    for part in parts[:-1]:
        if not isinstance(d.get(part), dict):
            d[part] = {}
        d = d[part]
    d[parts[-1]] = value
    _save_cfg(cfg)


# Backwards-compat alias for existing callers.
_save_cfg_key = _write_config_key


def _load_service_tier() -> str:
    """Return the configured service tier (``priority`` or ``""``).

    Ported from Hermes line 2393.
    """
    try:
        cfg = _load_cfg()
        agent_cfg = cfg.get("agent") if isinstance(cfg, dict) else None
        if isinstance(agent_cfg, dict):
            tier = str(agent_cfg.get("service_tier") or "").strip().lower()
            if tier:
                return tier
    except Exception:
        pass
    return ""


def _persist_live_session_runtime(session: dict | None) -> None:
    """Persist the live session's runtime (model/provider/reasoning/tier) to DB.

    Ported from Hermes line 2213. NIA may not yet expose all the agent fields
    this reads — best-effort, never raises. TODO(feature-gap): see
    FEATURE_GAPS.md (runtime_model_config).
    """
    if not session:
        return
    agent = session.get("agent")
    if agent is None:
        return
    session_key = session.get("session_key", "")
    if not session_key:
        return
    try:
        model_config: dict = {}
        for field in ("model", "provider", "base_url", "api_mode"):
            v = getattr(agent, field, None)
            if isinstance(v, str) and v.strip():
                model_config[field] = v.strip()
        rc = getattr(agent, "reasoning_config", None)
        if isinstance(rc, dict):
            model_config["reasoning_config"] = rc
        st = getattr(agent, "service_tier", None)
        if isinstance(st, str) and st.strip():
            model_config["service_tier"] = st.strip()
        with _session_db(session) as db:
            if db is not None and hasattr(db, "update_session_model_config"):
                try:
                    db.update_session_model_config(session_key, model_config)
                except Exception:
                    logger.debug("persist_live_session_runtime failed", exc_info=True)
    except Exception:
        logger.debug("persist_live_session_runtime outer failed", exc_info=True)


def _coerce_statusbar(raw: Any) -> str:
    """Coerce a config value to a valid statusbar mode.

    Ported from Hermes line 2328.
    """
    if not raw:
        return "top"
    s = str(raw).strip().lower()
    if s in {"top", "bottom", "off"}:
        return s
    return "top"


def _display_mouse_tracking(display: dict) -> str:
    """Return the current mouse-tracking mode from a display config dict.

    Ported from Hermes line 2356.
    """
    if not isinstance(display, dict):
        return "off"
    raw = str(display.get("mouse_tracking") or "off").strip().lower()
    if raw in {"all", "click", "drag", "off"}:
        return raw
    return "off"


def _validate_personality(value: str, cfg: dict | None = None) -> tuple[str, str]:
    """Validate a personality name and return (name, prompt_text).

    Ported from Hermes line 3850. NIA's personality subsystem may not yet
    expose ``_available_personalities`` — accept any non-empty value as a
    fallback. TODO(feature-gap): see FEATURE_GAPS.md (personalities).
    """
    name = (value or "").strip()
    if not name:
        return "", ""
    try:
        from niaharness.identity.personality import (  # type: ignore
            available_personalities,
            validate_personality,
        )

        return validate_personality(name, cfg)
    except ImportError:
        # Fallback: return the name as-is with an empty prompt.
        return name, ""


def _apply_personality_to_session(
    sid: str, session: dict | None, new_prompt: str, pname: str,
) -> tuple[bool, dict | None]:
    """Apply a personality change to the live session.

    Ported from Hermes line 3881. Returns ``(history_reset, info)``.
    """
    history_reset = False
    info = None
    if session is None:
        return history_reset, info
    try:
        agent = session.get("agent")
        if agent is not None and hasattr(agent, "set_system_prompt"):
            agent.set_system_prompt(new_prompt)
            history_reset = True
            info = _session_info(agent, session)
            _emit("session.info", sid, info)
    except Exception:
        pass
    return history_reset, info


@method("config.show")
def _config_show(rid, params):
    """Show structured config info for the TUI /config command.

    Deep-ported from Hermes line 13381. Renders a structured table with
    Model, Agent, and Environment sections.
    """
    try:
        cfg = _load_cfg()
        model = _resolve_model()
        api_key = (
            os.environ.get("NIA_API_KEY", "")
            or os.environ.get("HERMES_API_KEY", "")
            or str(cfg.get("api_key", "") or "")
        )
        masked = f"****{api_key[-4:]}" if len(api_key) > 4 else "(not set)"
        base_url = (
            os.environ.get("NIA_BASE_URL", "")
            or os.environ.get("HERMES_BASE_URL", "")
            or str(cfg.get("base_url", "") or "")
        )

        # Determine max_turns from config.
        agent_cfg = cfg.get("agent", {}) if isinstance(cfg.get("agent"), dict) else {}
        max_turns = int(agent_cfg.get("max_turns", 90) or 90)

        # Determine toolsets.
        enabled_toolsets = cfg.get("enabled_toolsets", [])
        if not isinstance(enabled_toolsets, list):
            enabled_toolsets = []

        # Determine NIA home path.
        try:
            from niaharness.prompts.soul import get_nia_home
            nia_home = get_nia_home()
        except Exception:
            nia_home = Path(os.path.expanduser("~/.nia"))

        sections = [
            {
                "title": "Model",
                "rows": [
                    ["Model", model],
                    ["Base URL", base_url or "(default)"],
                    ["API Key", masked],
                ],
            },
            {
                "title": "Agent",
                "rows": [
                    ["Max Turns", str(max_turns)],
                    ["Toolsets", ", ".join(enabled_toolsets) or "all"],
                    ["Verbose", str(cfg.get("verbose", False))],
                ],
            },
            {
                "title": "Environment",
                "rows": [
                    ["Working Dir", os.getcwd()],
                    ["Config File", str(nia_home / "config.yaml")],
                ],
            },
        ]
        return _ok(rid, {"sections": sections})
    except Exception as e:
        return _err(rid, 5030, str(e))


# ---------------------------------------------------------------------------
# Model methods
# ---------------------------------------------------------------------------


@method("model.options")
def _model_options(rid, params):
    from niaharness.api.provider_profiles import list_provider_profiles
    providers = []
    current_model = _resolve_model()
    for p in list_provider_profiles():
        # Check if API key is configured.
        authenticated = False
        for env_var in p.env_vars:
            if os.environ.get(env_var, "").strip():
                authenticated = True
                break
        providers.append({
            "slug": p.name,
            "name": p.display_name or p.name,
            "display_name": p.display_name or p.name,
            "is_current": False,
            "authenticated": authenticated,
            "auth_type": p.auth_type,
            "key_env": p.env_vars[0] if p.env_vars else "",
            "models": list(p.fallback_models),
        })
    return _ok(rid, {
        "model": current_model,
        "providers": providers,
    })


@method("model.save_key")
def _model_save_key(rid, params):
    provider = params.get("provider", "")
    api_key = params.get("api_key", "")
    if not provider or not api_key:
        return _err(rid, 4002, "provider and api_key are required")
    try:
        from niaharness.config.paths import get_nia_home
        env_path = get_nia_home() / ".env"
        env_path.parent.mkdir(parents=True, exist_ok=True)
        existing = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
        from niaharness.api.provider_profiles import get_provider_profile
        profile = get_provider_profile(provider)
        if profile and profile.env_vars:
            env_var = profile.env_vars[0]
        else:
            env_var = f"{provider.upper().replace('-', '_')}_API_KEY"
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
        try:
            os.chmod(str(env_path), 0o600)
        except OSError:
            pass
        # Set in current process.
        os.environ[env_var] = api_key
        return _ok(rid, {"saved": True, "env_var": env_var})
    except Exception as exc:
        return _err(rid, 5001, f"model.save_key failed: {exc}")


@method("model.disconnect")
def _model_disconnect(rid, params):
    """Remove credentials for a provider.

    Deep-ported from Hermes line 12490. Clears API key env vars from .env +
    process env, and clears any OAuth / credential-pool state. NIA may not
    yet expose ``PROVIDER_REGISTRY`` / ``clear_provider_auth`` / 
    ``remove_env_value`` — best-effort: clear the env var directly.
    TODO(feature-gap): see FEATURE_GAPS.md (cli.auth + cli.config).
    """
    try:
        slug = (params.get("slug") or "").strip()
        if not slug:
            return _err(rid, 4001, "slug is required")

        cleared_env = False
        cleared_auth = False

        # Try the full Hermes-style path first.
        try:
            from niaharness.cli.auth import (  # type: ignore
                PROVIDER_REGISTRY,
                clear_provider_auth,
            )
            from niaharness.cli.config import remove_env_value  # type: ignore

            pconfig = PROVIDER_REGISTRY.get(slug)
            if pconfig and getattr(pconfig, "api_key_env_vars", None):
                for ev in pconfig.api_key_env_vars:
                    if remove_env_value(ev):
                        cleared_env = True
            cleared_auth = clear_provider_auth(slug)
        except ImportError:
            # TODO(feature-gap): see FEATURE_GAPS.md (cli.auth + cli.config).
            # Fallback: clear common env var naming patterns for this slug.
            env_var_candidates = [
                f"{slug.upper()}_API_KEY",
                f"NIA_{slug.upper()}_API_KEY",
                f"HERMES_{slug.upper()}_API_KEY",
            ]
            for ev in env_var_candidates:
                if ev in os.environ:
                    os.environ.pop(ev, None)
                    cleared_env = True

        if not cleared_env and not cleared_auth:
            return _err(rid, 4005, f"no credentials found for {slug}")

        return _ok(rid, {
            "slug": slug,
            "name": slug,  # NIA has no PROVIDER_REGISTRY name lookup yet
            "disconnected": True,
        })
    except Exception as e:
        return _err(rid, 5035, str(e))


# ---------------------------------------------------------------------------
# Slash command methods
# ---------------------------------------------------------------------------


@method("slash.exec")
def _slash_exec(rid, params):
    """Execute a slash command via the slash worker or command.dispatch.

    Deep-ported from Hermes line 12633. Routes pending-input commands and
    skill commands to ``command.dispatch`` instead of the slash worker.
    Plugin commands are handled inline. Falls back to the command registry
    when the slash worker is unavailable.
    """
    session, err = _sess(params, rid)
    if err:
        return err

    cmd = params.get("command", "").strip()
    if not cmd:
        return _err(rid, 4004, "empty command")
    if not cmd.startswith("/"):
        cmd = f"/{cmd}"

    # Parse the command name + arg.
    _cmd_text = cmd.lstrip("/")
    _cmd_parts = _cmd_text.split(maxsplit=1)
    _cmd_base = (_cmd_parts[0].lower() if _cmd_parts else "")
    _cmd_arg = _cmd_parts[1] if len(_cmd_parts) > 1 else ""

    # Route pending-input commands to command.dispatch.
    _PENDING_INPUT_COMMANDS = {
        "queue", "q", "learn", "moa", "retry", "steer", "goal",
        "undo", "snapshot", "snap",
    }
    if _cmd_base in _PENDING_INPUT_COMMANDS:
        return _methods["command.dispatch"](
            rid,
            {
                "name": _cmd_base,
                "arg": _cmd_arg,
                "session_id": params.get("session_id", ""),
            },
        )

    # Check for skill commands.
    try:
        from niaharness.skills.skill_commands import get_skill_commands

        _cmd_key = f"/{_cmd_base}"
        if _cmd_key in get_skill_commands():
            return _err(rid, 4018, f"skill command: use command.dispatch for {_cmd_key}")
    except Exception:
        pass

    # Check for plugin commands.
    try:
        from niaharness.plugins import get_plugin_command_handler, resolve_plugin_command_result

        plugin_handler = get_plugin_command_handler(_cmd_base)
        if plugin_handler:
            try:
                result = resolve_plugin_command_result(plugin_handler(_cmd_arg))
                return _ok(rid, {"output": str(result or "(no output)")})
            except Exception as e:
                return _ok(rid, {"output": f"Plugin command error: {e}"})
    except Exception:
        pass

    # Try the slash worker.
    worker = _slash_workers.get(params.get("session_id", ""))
    if worker is not None:
        try:
            output = worker.run(cmd)
            # Apply slash side effects (model switch, yolo, etc.).
            warning = _mirror_slash_side_effects(params.get("session_id", ""), session, cmd)
            payload: dict = {"output": output or "(no output)"}
            if warning:
                payload["warning"] = warning
            return _ok(rid, payload)
        except Exception:
            # Fall through to inline execution.
            pass

    # Fallback: inline execution via command registry.
    try:
        from niaharness.commands import create_default_command_registry
        registry = create_default_command_registry()
        lookup_result = registry.lookup(cmd[1:])
        if lookup_result is None:
            return _ok(rid, {"output": f"Unknown command: {cmd}"})
        command_obj, args = lookup_result
        import asyncio
        result = asyncio.run(command_obj.handler(args, None))
        output = result.output if result else "Command produced no output."
        return _ok(rid, {"output": output})
    except Exception as exc:
        return _err(rid, 5001, f"slash.exec failed: {exc}")


# Commands that mutate agent/session state and must be mirrored after slash.exec.
_AGENT_MUTATING_COMMANDS = frozenset({
    "model", "reasoning", "fast", "yolo", "prompt", "personality",
    "verbose", "details", "details_mode", "thinking_mode",
})


def _mirror_slash_side_effects(sid: str, session: dict, command: str) -> str:
    """Apply side effects that must also hit the gateway's live agent.

    Ported from hermes-agent/tui_gateway/server.py line 12539.
    Returns a warning string if any side effects were applied.
    """
    parts = command.lstrip("/").split(None, 1)
    if not parts:
        return ""
    name = parts[0].lower()
    arg = (parts[1].strip() if len(parts) > 1 else "")
    agent = session.get("agent")
    warning = ""

    # Reject agent-mutating commands during an in-flight turn.
    if name in _AGENT_MUTATING_COMMANDS and session.get("running"):
        return f"session busy — /interrupt the current turn before /{name}"

    # Model switch: re-apply to the live agent.
    if name == "model" and arg and agent is not None:
        try:
            _apply_model_switch(sid, session, arg, confirm_expensive_model=True)
            warning = f"model switched to {arg}"
        except Exception as exc:
            warning = f"model switch failed: {exc}"

    # Yolo toggle: re-apply to the live agent.
    if name == "yolo" and agent is not None:
        try:
            from niaharness.permissions.approval import (
                is_session_yolo_enabled,
                enable_session_yolo,
                disable_session_yolo,
            )
            session_key = session.get("session_key", sid)
            current = is_session_yolo_enabled(session_key)
            if arg.lower() in {"on", "1", "true"}:
                enable_session_yolo(session_key)
            elif arg.lower() in {"off", "0", "false"}:
                disable_session_yolo(session_key)
            else:
                if current:
                    disable_session_yolo(session_key)
                else:
                    enable_session_yolo(session_key)
            _emit("session.info", sid, _session_info(agent, session))
        except Exception:
            pass

    return warning


@method("commands.catalog")
def _commands_catalog(rid, params):
    """Registry-backed slash metadata for the TUI — categorized, no aliases.

    Deep-ported from Hermes line 11274. Builds a categorized command catalog
    from the command registry + quick_commands config + skill commands.
    """
    try:
        from niaharness.commands import create_default_command_registry

        registry = create_default_command_registry()
        commands = registry.list_commands()

        all_pairs: list[list[str]] = []
        canon: dict[str, str] = {}
        categories: list[dict] = []
        cat_map: dict[str, list[list[str]]] = {}
        cat_order: list[str] = []

        for cmd in commands:
            c = f"/{cmd.name}"
            canon[c.lower()] = c
            # Aliases.
            for a in getattr(cmd, "aliases", []):
                canon[f"/{a}".lower()] = c

            desc = getattr(cmd, "description", "") or ""
            all_pairs.append([c, desc])

            cat = getattr(cmd, "category", "General")
            if cat not in cat_map:
                cat_map[cat] = []
                cat_order.append(cat)
            cat_map[cat].append([c, desc])

        # Quick commands from config.
        warning = ""
        try:
            qcmds = _load_cfg().get("quick_commands", {}) or {}
            if isinstance(qcmds, dict) and qcmds:
                bucket = "User commands"
                if bucket not in cat_map:
                    cat_map[bucket] = []
                    cat_order.append(bucket)
                for qname, qc in sorted(qcmds.items()):
                    if not isinstance(qc, dict):
                        continue
                    key = f"/{qname}"
                    canon[key.lower()] = key
                    qtype = qc.get("type", "")
                    if qtype == "exec":
                        default_desc = f"exec: {qc.get('command', '')}"
                    elif qtype == "alias":
                        default_desc = f"alias → {qc.get('target', '')}"
                    else:
                        default_desc = qtype or "quick command"
                    qdesc = str(qc.get("description") or default_desc)
                    qdesc = qdesc[:120] + ("…" if len(qdesc) > 120 else "")
                    all_pairs.append([key, qdesc])
                    cat_map[bucket].append([key, qdesc])
        except Exception as e:
            if not warning:
                warning = f"quick_commands discovery unavailable: {e}"

        # Skill commands.
        skill_count = 0
        try:
            from niaharness.skills.skill_commands import scan_skill_commands

            for k, info in sorted(scan_skill_commands().items()):
                d = str(info.get("description", "Skill"))
                all_pairs.append([k, d[:120] + ("…" if len(d) > 120 else "")])
                skill_count += 1
        except Exception as e:
            warning = f"skill discovery unavailable: {e}"

        for cat in cat_order:
            categories.append({"name": cat, "pairs": cat_map[cat]})

        return _ok(rid, {
            "pairs": all_pairs,
            "categories": categories,
            "canon": canon,
            "sub": {},
            "skill_count": skill_count,
            "warning": warning,
        })
    except Exception as e:
        return _ok(rid, {"pairs": [], "categories": [], "canon": {}, "sub": {}, "skill_count": 0, "warning": str(e)})


@method("command.resolve")
def _command_resolve(rid, params):
    """Resolve a slash-command name to its canonical form.

    Ported from Hermes line 11441. NIA's ``niaharness.commands.registry`` may
    not yet expose ``resolve_command`` — return the name unchanged as a
    fallback. TODO(feature-gap): see FEATURE_GAPS.md (commands.resolve_command).
    """
    name = params.get("text", "") or params.get("name", "")
    name = (name or "").lstrip("/")
    resolved = _resolve_command_name(name)
    return _ok(rid, {"resolved": resolved, "name": resolved})


def _resolve_command_name(name: str) -> str:
    """Resolve a command name to its canonical form (alias → canonical).

    Ported from Hermes line 11441. Falls back to the input name when NIA's
    command registry doesn't expose ``resolve_command``.
    """
    if not name:
        return name
    try:
        from niaharness.commands.registry import resolve_command  # type: ignore

        r = resolve_command(name)
        return r.name if r else name
    except ImportError:
        return name
    except Exception:
        return name


@method("command.dispatch")
def _command_dispatch(rid, params):
    """Dispatch a slash command by name with full handler coverage.

    Deep-ported from Hermes line 11451. Handles:
    - Quick commands (``config.quick_commands``): ``exec`` runs sanitized
      subprocess + redacts output; ``alias`` returns the target.
    - Plugin commands (``niaharness.plugins.get_plugin_command_handler``).
    - Skill commands (``niaharness.skills.skill_commands.scan_skill_commands``).
    - Built-in commands:
      - ``/queue <prompt>`` — queue a message for the next turn.
      - ``/learn <arg>`` — open-ended skill authoring prompt.
      - ``/moa <prompt>`` — Mixture-of-Agents one-shot (NIA may not have MoA).
      - ``/retry`` — replay the last user message after truncating the failed
        exchange.
      - ``/steer <prompt>`` — mid-turn steering.
      - ``/goal <text>`` — set/pause/resume/clear a Ralph-style goal loop.
      - ``/undo [N]`` — back up N user turns, prefill composer.
      - ``/snapshot restore`` — blocked in TUI (returns explanatory message).
    - Returns a structured payload so the TUI can render + submit accordingly.
    """
    name, arg = params.get("name", "").lstrip("/"), params.get("arg", "")
    resolved = _resolve_command_name(name)
    if resolved != name:
        name = resolved
    session = _sessions.get(params.get("session_id", ""))

    # ── Quick commands (config.quick_commands) ────────────────────────
    qcmds = _load_cfg().get("quick_commands", {})
    if isinstance(qcmds, dict) and name in qcmds:
        qc = qcmds[name]
        if not isinstance(qc, dict):
            qc = {}
        if qc.get("type") == "exec":
            # Sanitize env to prevent credential leakage — quick commands
            # run in the TUI server process which has all API keys in
            # os.environ. NIA's terminal env helper may not yet expose
            # _sanitize_subprocess_env — fall back to a minimal env.
            # TODO(feature-gap): see FEATURE_GAPS.md (tools.environments.local).
            try:
                from niaharness.tools.environments.local import (  # type: ignore
                    _sanitize_subprocess_env,
                )

                sanitized_env = _sanitize_subprocess_env(os.environ.copy())
            except ImportError:
                # Minimal sanitization: keep PATH + HOME only.
                sanitized_env = {
                    k: v for k, v in os.environ.items()
                    if k in {"PATH", "HOME", "USER", "SHELL", "LANG", "LC_ALL"}
                }
            try:
                r = subprocess.run(
                    qc.get("command", ""),
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    stdin=subprocess.DEVNULL,
                    env=sanitized_env,
                )
                output = (
                    (r.stdout or "")
                    + ("\n" if r.stdout and r.stderr else "")
                    + (r.stderr or "")
                ).strip()[:4000]
                if output:
                    output = _redact_sensitive_text(output)
                if r.returncode != 0:
                    return _err(
                        rid, 4018,
                        output or f"quick command failed with exit code {r.returncode}",
                    )
                return _ok(rid, {"type": "exec", "output": output})
            except Exception as e:
                return _err(rid, 4018, f"quick command failed: {e}")
        if qc.get("type") == "alias":
            return _ok(rid, {"type": "alias", "target": qc.get("target", "")})

    # ── Plugin commands ───────────────────────────────────────────────
    try:
        from niaharness.plugins import (  # type: ignore
            get_plugin_command_handler,
            resolve_plugin_command_result,
        )

        handler = get_plugin_command_handler(name)
        if handler:
            result = resolve_plugin_command_result(handler(arg))
            return _ok(rid, {"type": "plugin", "output": str(result or "")})
    except ImportError:
        # TODO(feature-gap): see FEATURE_GAPS.md (plugins.get_plugin_command_handler).
        pass
    except Exception:
        pass

    # ── Skill commands ────────────────────────────────────────────────
    try:
        from niaharness.skills.skill_commands import (  # type: ignore
            build_skill_invocation_message,
            scan_skill_commands,
        )

        cmds = scan_skill_commands()
        key = f"/{name}"
        if key in cmds:
            msg = build_skill_invocation_message(
                key, arg,
                task_id=session.get("session_key", "") if session else "",
            )
            if msg:
                return _ok(rid, {
                    "type": "skill",
                    "message": msg,
                    "name": cmds[key].get("name", name),
                })
    except ImportError:
        # TODO(feature-gap): see FEATURE_GAPS.md (skills.skill_commands).
        pass
    except Exception:
        pass

    # ── Built-in commands ─────────────────────────────────────────────

    if name in {"queue", "q"}:
        if not arg:
            return _err(rid, 4004, "usage: /queue <prompt>")
        return _ok(rid, {"type": "send", "message": arg})

    if name == "learn":
        try:
            from niaharness.engine.learn_prompt import build_learn_prompt  # type: ignore

            return _ok(rid, {"type": "send", "message": build_learn_prompt(arg)})
        except ImportError:
            # TODO(feature-gap): see FEATURE_GAPS.md (engine.learn_prompt).
            return _ok(rid, {
                "type": "send",
                "message": (
                    f"Learn: {arg}\n\n(Gather relevant context — directories, "
                    "URLs, pasted text — and write a new skill via skill_manage.)"
                ),
            })

    if name == "moa":
        # NIA doesn't yet have a MoA subsystem — return a clear error.
        # TODO(feature-gap): see FEATURE_GAPS.md (MoA / moa_config).
        return _err(rid, 5030, "moa unavailable: NIA does not yet have a Mixture-of-Agents subsystem")

    if name == "retry":
        if not session:
            return _err(rid, 4001, "no active session to retry")
        if session.get("running"):
            return _err(rid, 4009, "session busy — /interrupt the current turn before /retry")
        history = session.get("history", [])
        if not history:
            return _err(rid, 4018, "no previous user message to retry")
        # Walk backwards to find the last user message.
        last_user_idx = None
        for i in range(len(history) - 1, -1, -1):
            if history[i].get("role") == "user":
                last_user_idx = i
                break
        if last_user_idx is None:
            return _err(rid, 4018, "no previous user message to retry")
        content = history[last_user_idx].get("content", history[last_user_idx].get("text", ""))
        if isinstance(content, list):
            content = " ".join(
                p.get("text", "") for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            )
        if not content:
            return _err(rid, 4018, "last user message is empty")
        # Truncate history: remove everything from the last user message onward.
        with session["history_lock"]:
            session["history"] = history[:last_user_idx]
            session["history_version"] = int(session.get("history_version", 0)) + 1
        return _ok(rid, {"type": "send", "message": content})

    if name == "steer":
        if not arg:
            return _err(rid, 4004, "usage: /steer <prompt>")
        agent = session.get("agent") if session else None
        if agent and hasattr(agent, "steer"):
            try:
                accepted = agent.steer(arg)
                if accepted:
                    return _ok(rid, {
                        "type": "exec",
                        "output": (
                            f"⏩ Steer queued — arrives after the next tool call: "
                            f"{arg[:80]}{'...' if len(arg) > 80 else ''}"
                        ),
                    })
            except Exception:
                pass
        return _ok(rid, {"type": "send", "message": arg})

    if name == "goal":
        if not session:
            return _err(rid, 4001, "no active session")
        try:
            from niaharness.goals import GoalManager  # type: ignore
        except ImportError:
            # TODO(feature-gap): see FEATURE_GAPS.md (goals.GoalManager).
            return _err(rid, 5030, "goals unavailable: NIA does not yet have a GoalManager")

        sid_key = session.get("session_key") or ""
        if not sid_key:
            return _err(rid, 4001, "no session key")

        try:
            goals_cfg = _load_cfg().get("goals") or {}
            max_turns = int(goals_cfg.get("max_turns", 20) or 20)
        except Exception:
            max_turns = 20
        mgr = GoalManager(session_id=sid_key, default_max_turns=max_turns)

        lower = arg.strip().lower()
        if not arg.strip() or lower == "status":
            return _ok(rid, {"type": "exec", "output": mgr.status_line()})
        if lower == "pause":
            state = mgr.pause(reason="user-paused")
            out = "No goal set." if state is None else f"⏸ Goal paused: {state.goal}"
            return _ok(rid, {"type": "exec", "output": out})
        if lower == "resume":
            state = mgr.resume()
            if state is None:
                return _ok(rid, {"type": "exec", "output": "No goal to resume."})
            return _ok(rid, {
                "type": "exec",
                "output": (
                    f"▶ Goal resumed: {state.goal}\n"
                    "Send any message to continue, or wait — I'll take the next step on the next turn."
                ),
            })
        if lower in {"clear", "stop", "done"}:
            had = mgr.has_goal()
            mgr.clear()
            return _ok(rid, {
                "type": "exec",
                "output": "✓ Goal cleared." if had else "No active goal.",
            })

        try:
            state = mgr.set(arg)
        except ValueError as exc:
            return _err(rid, 4004, f"invalid goal: {exc}")

        notice = (
            f"⊙ Goal set ({state.max_turns}-turn budget): {state.goal}\n"
            "I'll keep working until the goal is done, you pause/clear it, or the budget is exhausted.\n"
            "Controls: /goal status · /goal pause · /goal resume · /goal clear"
        )
        return _ok(rid, {"type": "send", "notice": notice, "message": state.goal})

    if name == "undo":
        if not session:
            return _err(rid, 4001, "no active session to undo")
        if session.get("running"):
            return _err(rid, 4009, "session busy — /interrupt the current turn before /undo")
        db = _get_db()
        if db is None:
            return _err(rid, 5008, "session DB not available")
        session_key = session.get("session_key", "")
        if not session_key:
            return _err(rid, 4001, "no session key for undo")
        # Parse optional count.
        n = 1
        arg_str = (arg or "").strip()
        if arg_str:
            try:
                n = int(arg_str.split()[0])
            except (ValueError, IndexError):
                return _err(rid, 4004, f"undo: invalid count {arg_str!r} — use /undo or /undo N")
        if n < 1:
            n = 1
        try:
            if hasattr(db, "list_recent_user_messages"):
                recents = db.list_recent_user_messages(session_key, limit=max(n, 10))
            else:
                # TODO(feature-gap): see FEATURE_GAPS.md (list_recent_user_messages).
                return _err(rid, 5008, "undo: SessionDB does not expose list_recent_user_messages")
        except Exception as e:
            return _err(rid, 5008, f"undo: failed to load history: {e}")
        if not recents:
            return _err(rid, 4018, "no user messages to undo")
        target_idx = min(n - 1, len(recents) - 1)
        target_id = recents[target_idx]["id"]
        try:
            if hasattr(db, "rewind_to_message"):
                result = db.rewind_to_message(session_key, target_id)
            else:
                # TODO(feature-gap): see FEATURE_GAPS.md (rewind_to_message).
                return _err(rid, 5008, "undo: SessionDB does not expose rewind_to_message")
        except ValueError as e:
            return _err(rid, 4004, f"undo: {e}")
        except Exception as e:
            return _err(rid, 5008, f"undo: {e}")
        # Reload the active transcript.
        try:
            if hasattr(db, "get_messages_as_conversation"):
                active = db.get_messages_as_conversation(session_key)
            else:
                active = _rows_to_history(db.get_messages(session_key, include_compacted=False))
        except Exception:
            active = []
        with session["history_lock"]:
            session["history"] = list(active)
            session["history_version"] = int(session.get("history_version", 0)) + 1
        # Notify memory manager + invalidate cached prompt.
        agent = session.get("agent")
        if agent is not None:
            mm = getattr(agent, "_memory_manager", None)
            if mm is not None and hasattr(mm, "on_session_switch"):
                try:
                    mm.on_session_switch(session_key, parent_session_id="", reset=False, rewound=True)
                except Exception:
                    pass
            if hasattr(agent, "_invalidate_system_prompt"):
                try:
                    agent._invalidate_system_prompt()
                except Exception:
                    pass
            if hasattr(agent, "_last_flushed_db_idx"):
                try:
                    agent._last_flushed_db_idx = len(active)
                except Exception:
                    pass
        target_msg = result.get("target_message") or {}
        target_text = target_msg.get("content") or target_msg.get("text") or ""
        if isinstance(target_text, list):
            parts = [
                p.get("text", "") for p in target_text
                if isinstance(p, dict) and p.get("type") == "text"
            ]
            target_text = "\n".join(t for t in parts if t)
        if not isinstance(target_text, str):
            target_text = ""
        rewound_count = result.get("rewound_count", 0)
        turns_undone = target_idx + 1
        turn_word = "turn" if turns_undone == 1 else "turns"
        notice = (
            f"↶ Undid {turns_undone} {turn_word} ({rewound_count} message(s)). "
            "Edit and resubmit, or send a new message."
        )
        return _ok(rid, {"type": "prefill", "message": target_text, "notice": notice})

    if name in {"snapshot", "snap"}:
        subcommand = arg.split(maxsplit=1)[0].lower() if arg else ""
        if subcommand in {"restore", "rewind"}:
            return _ok(rid, {
                "type": "exec",
                "output": (
                    "/snapshot restore is blocked in the TUI because it changes "
                    "config/state on disk while the live agent has cached settings. "
                    "Run it in the classic CLI, then restart the TUI."
                ),
            })

    return _err(rid, 4018, f"not a quick/plugin/skill command: {name}")


def _redact_sensitive_text(text: str) -> str:
    """Redact common secret patterns from ``text``.

    Ported from ``agent.redact.redact_sensitive_text``. NIA may not yet
    expose that module — do a minimal regex-based redaction as a fallback.
    TODO(feature-gap): see FEATURE_GAPS.md (agent.redact).
    """
    if not text:
        return text
    try:
        from niaharness.engine.redact import redact_sensitive_text  # type: ignore

        return redact_sensitive_text(text)
    except ImportError:
        import re

        patterns = [
            # GitHub tokens
            (re.compile(r"gh[pousr]_[A-Za-z0-9]{36,}"), "[REDACTED:github-token]"),
            # OpenAI keys
            (re.compile(r"sk-[A-Za-z0-9]{20,}"), "[REDACTED:openai-key]"),
            # Anthropic keys
            (re.compile(r"sk-ant-[A-Za-z0-9\-_]{20,}"), "[REDACTED:anthropic-key]"),
            # Bearer tokens
            (re.compile(r"Bearer\s+[A-Za-z0-9\-_\.]{20,}"), "Bearer [REDACTED]"),
            # AWS access keys
            (re.compile(r"AKIA[0-9A-Z]{16}"), "[REDACTED:aws-key]"),
            # password= assignments
            (re.compile(r"(?i)(password|passwd|pwd)\s*=\s*\S+"), r"\1=[REDACTED]"),
        ]
        out = text
        for pat, repl in patterns:
            out = pat.sub(repl, out)
        return out
    except Exception:
        return text



# ---------------------------------------------------------------------------
# Completion methods
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Completion helpers (deep-ported from Hermes tui_gateway/server.py)
# ---------------------------------------------------------------------------

_FUZZY_CACHE_TTL_S = 5.0
_FUZZY_CACHE_MAX_FILES = 20000
_FUZZY_FALLBACK_EXCLUDES = frozenset({
    ".git", ".hg", ".svn", ".next", ".cache", ".venv", "venv",
    "node_modules", "__pycache__", "dist", "build", "target",
    ".mypy_cache", ".pytest_cache", ".ruff_cache",
})
_fuzzy_cache_lock = threading.Lock()
_fuzzy_cache: dict[str, tuple[float, list[str]]] = {}


def _list_repo_files(root: str) -> list[str]:
    """Return file paths relative to ``root``.

    Ported from Hermes line 11915. Uses ``git ls-files`` from the repo top
    (resolved via ``rev-parse --show-toplevel``) so the listing covers tracked
    + untracked files anywhere in the repo, then converts each path back to be
    relative to ``root``. Falls back to a bounded ``os.walk(root)`` when
    ``root`` isn't inside a git repo. Result cached per-root for
    ``_FUZZY_CACHE_TTL_S`` so rapid keystrokes don't respawn git processes.
    """
    if not root or not os.path.isdir(root):
        return []
    now = time.monotonic()
    with _fuzzy_cache_lock:
        cached = _fuzzy_cache.get(root)
        if cached and now - cached[0] < _FUZZY_CACHE_TTL_S:
            return cached[1]

    files: list[str] = []
    try:
        top_result = subprocess.run(
            ["git", "-C", root, "rev-parse", "--show-toplevel"],
            capture_output=True, timeout=2.0, check=False,
            stdin=subprocess.DEVNULL,
        )
        if top_result.returncode == 0:
            top = top_result.stdout.decode("utf-8", "replace").strip()
            list_result = subprocess.run(
                [
                    "git", "-C", top, "ls-files", "-z",
                    "--cached", "--others", "--exclude-standard",
                ],
                capture_output=True, timeout=2.0, check=False,
                stdin=subprocess.DEVNULL,
            )
            if list_result.returncode == 0:
                for p in list_result.stdout.decode("utf-8", "replace").split("\0"):
                    if not p:
                        continue
                    rel = os.path.relpath(os.path.join(top, p), root).replace(os.sep, "/")
                    if rel.startswith("../"):
                        continue
                    files.append(rel)
                    if len(files) >= _FUZZY_CACHE_MAX_FILES:
                        break
    except (OSError, subprocess.TimeoutExpired):
        pass

    if not files:
        # Fallback walk: skip vendor/build dirs + dot-dirs so the walk stays
        # tractable. Dotfiles themselves survive — the ranker decides based
        # on whether the query starts with `.`.
        try:
            for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
                dirnames[:] = [
                    d for d in dirnames
                    if d not in _FUZZY_FALLBACK_EXCLUDES and not d.startswith(".")
                ]
                rel_dir = os.path.relpath(dirpath, root)
                for f in filenames:
                    rel = f if rel_dir == "." else f"{rel_dir}/{f}"
                    files.append(rel.replace(os.sep, "/"))
                    if len(files) >= _FUZZY_CACHE_MAX_FILES:
                        break
                if len(files) >= _FUZZY_CACHE_MAX_FILES:
                    break
        except OSError:
            pass

    with _fuzzy_cache_lock:
        _fuzzy_cache[root] = (now, files)
    return files


def _fuzzy_basename_rank(name: str, query: str) -> tuple[int, int] | None:
    """Rank ``name`` against ``query``; lower is better. Returns None to reject.

    Ported from Hermes line 12010. Tiers:
      0 — exact basename
      1 — basename prefix (e.g. `app` → `appChrome.tsx`)
      2 — word-boundary / camelCase hit (e.g. `chrome` → `appChrome.tsx`)
      3 — substring anywhere in basename
      4 — subsequence match (every query char appears in order)
    """
    if not query:
        return (3, len(name))
    nl = name.lower()
    ql = query.lower()
    if nl == ql:
        return (0, len(name))
    if nl.startswith(ql):
        return (1, len(name))
    # Word-boundary split: `foo-bar_baz.qux` → ["foo","bar","baz","qux"].
    # camelCase split: `appChrome` → ["app","Chrome"].
    parts: list[str] = []
    buf = ""
    for ch in name:
        if ch in "-_." or (ch.isupper() and buf and not buf[-1].isupper()):
            if buf:
                parts.append(buf)
            buf = ch if ch not in "-_." else ""
        else:
            buf += ch
    if buf:
        parts.append(buf)
    for p in parts:
        if p.lower().startswith(ql):
            return (2, len(name))
    if ql in nl:
        return (3, len(name))
    i = 0
    for ch in nl:
        if ch == ql[i]:
            i += 1
            if i == len(ql):
                return (4, len(name))
    return None


def _normalize_completion_path(path_part: str) -> str:
    """Expand ``~`` and resolve a path fragment for completion matching.

    Ported from Hermes line 1396.
    """
    if not path_part:
        return ""
    raw = path_part.strip()
    if raw.startswith("~"):
        raw = os.path.expanduser(raw)
    return raw


@method("complete.path")
def _complete_path(rid, params):
    """Path / @-mention completion with fuzzy basename matching.

    Deep-ported from Hermes line 12065. Handles:
    - ``@`` alone → list context-reference keywords (@diff, @staged, @file:,
      @folder:, @url:, @git:).
    - ``@<bare-name>`` (≥2 chars, no ``/``) → fuzzy basename search across the
      repo via ``git ls-files``, ranked by tier (exact / prefix /
      word-boundary / substring / subsequence).
    - ``@file:<path>`` / ``@folder:<path>`` → directory listing filtered by
      kind (file vs dir).
    - ``<path>`` / ``./<path>`` / ``~/<path>`` / ``/abs/path`` → directory
      listing with proper prefix preservation.
    """
    word = params.get("word", "") or params.get("prefix", "")
    if not word:
        return _ok(rid, {"items": []})

    items: list[dict] = []
    try:
        root = _completion_cwd(params)
        is_context = word.startswith("@")
        query = word[1:] if is_context else word

        if is_context and not query:
            items = [
                {"text": "@diff", "display": "@diff", "meta": "git diff"},
                {"text": "@staged", "display": "@staged", "meta": "staged diff"},
                {"text": "@file:", "display": "@file:", "meta": "attach file"},
                {"text": "@folder:", "display": "@folder:", "meta": "attach folder"},
                {"text": "@url:", "display": "@url:", "meta": "fetch url"},
                {"text": "@git:", "display": "@git:", "meta": "git log"},
            ]
            return _ok(rid, {"items": items, "replace_from": 0})

        # Accept both `@folder:path` and the bare `@folder` form so the user
        # sees directory listings as soon as they finish typing the keyword.
        if is_context and query in {"file", "folder"}:
            prefix_tag, path_part = query, ""
        elif is_context and query.startswith(("file:", "folder:")):
            prefix_tag, _, tail = query.partition(":")
            path_part = tail
        else:
            prefix_tag = ""
            path_part = query if is_context else query

        # Fuzzy basename search across the repo when the user types a bare
        # name with no path separator — `@appChrome` surfaces every file
        # whose basename matches, regardless of directory depth.
        if (
            is_context
            and path_part
            and len(path_part.strip()) >= 2
            and "/" not in path_part
            and prefix_tag != "folder"
        ):
            ranked: list[tuple[tuple[int, int], str, str]] = []
            for rel in _list_repo_files(root):
                basename = os.path.basename(rel)
                if basename.startswith(".") and not path_part.startswith("."):
                    continue
                rank = _fuzzy_basename_rank(basename, path_part)
                if rank is None:
                    continue
                ranked.append((rank, rel, basename))

            ranked.sort(key=lambda r: (r[0], len(r[1]), r[1]))
            tag = prefix_tag or "file"
            for _, rel, basename in ranked[:30]:
                items.append({
                    "text": f"@{tag}:{rel}",
                    "display": basename,
                    "meta": os.path.dirname(rel),
                })
            return _ok(rid, {"items": items, "replace_from": 0})

        expanded = _normalize_completion_path(path_part) if path_part else "."
        if expanded == "." or not expanded:
            search_dir, match = ".", ""
        elif expanded.endswith("/"):
            search_dir, match = expanded, ""
        else:
            search_dir = os.path.dirname(expanded) or "."
            match = os.path.basename(expanded)

        search_dir = (
            search_dir if os.path.isabs(search_dir) else os.path.join(root, search_dir)
        )
        if not os.path.isdir(search_dir):
            return _ok(rid, {"items": [], "replace_from": 0})

        want_dir = prefix_tag == "folder"
        match_lower = match.lower()
        for entry in sorted(os.listdir(search_dir)):
            if match and not entry.lower().startswith(match_lower):
                continue
            if is_context and entry in _FUZZY_FALLBACK_EXCLUDES:
                continue
            if is_context and not prefix_tag and entry.startswith("."):
                continue
            full = os.path.join(search_dir, entry)
            is_dir = os.path.isdir(full)
            if prefix_tag and want_dir != is_dir:
                continue
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            suffix = "/" if is_dir else ""

            if is_context and prefix_tag:
                text = f"@{prefix_tag}:{rel}{suffix}"
            elif is_context:
                kind = "folder" if is_dir else "file"
                text = f"@{kind}:{rel}{suffix}"
            elif word.startswith("~"):
                text = "~/" + os.path.relpath(full, os.path.expanduser("~")) + suffix
            elif word.startswith("./"):
                text = "./" + rel + suffix
            else:
                text = rel + suffix

            items.append({
                "text": text,
                "display": entry + suffix,
                "meta": "dir" if is_dir else "file",
            })
            if len(items) >= 30:
                break
    except Exception as e:
        return _err(rid, 5021, str(e))

    return _ok(rid, {"items": items, "replace_from": 0})


def _details_completion_item(value: str, meta: str = "") -> dict:
    return {"text": value, "display": value, "meta": meta}


def _details_root_completion_item(value: str, meta: str, needs_leading_space: bool) -> dict:
    return _details_completion_item(
        f" {value}" if needs_leading_space else value, meta,
    )


def _details_completions(text: str) -> list[dict] | None:
    """Return /details subcommand completions, or None if ``text`` isn't /details.

    Ported from Hermes line 12210.
    """
    if not text.lower().startswith("/details"):
        return None
    stripped = text.strip()
    if stripped and not "/details".startswith(stripped.lower().split()[0]):
        return None
    body = text[len("/details"):]
    if body.startswith(" "):
        body = body[1:]
    parts = body.split()
    has_trailing_space = text.endswith(" ")
    sections = ("thinking", "tools", "subagents", "activity")
    modes = ("hidden", "collapsed", "expanded")

    if not body or (len(parts) == 0 and has_trailing_space):
        return [
            *[
                _details_root_completion_item(mode, "global mode", not has_trailing_space)
                for mode in modes
            ],
            _details_root_completion_item("cycle", "cycle global mode", not has_trailing_space),
            *[
                _details_root_completion_item(section, "section override", not has_trailing_space)
                for section in sections
            ],
        ]
    if len(parts) == 1 and not has_trailing_space:
        prefix = parts[0].lower()
        candidates = [*modes, "cycle", *sections]
        return [
            _details_completion_item(
                candidate,
                (
                    "section override" if candidate in sections
                    else "cycle global mode" if candidate == "cycle"
                    else "global mode"
                ),
            )
            for candidate in candidates
            if candidate.startswith(prefix) and candidate != prefix
        ]
    if len(parts) == 1 and has_trailing_space and parts[0].lower() in sections:
        return [
            *[_details_completion_item(mode, f"set {parts[0].lower()}") for mode in modes],
            _details_completion_item("reset", f"clear {parts[0].lower()} override"),
        ]
    if len(parts) == 2 and not has_trailing_space and parts[0].lower() in sections:
        prefix = parts[1].lower()
        return [
            _details_completion_item(
                candidate,
                (
                    f"clear {parts[0].lower()} override" if candidate == "reset"
                    else f"set {parts[0].lower()}"
                ),
            )
            for candidate in (*modes, "reset")
            if candidate.startswith(prefix) and candidate != prefix
        ]
    return []


@method("complete.slash")
def _complete_slash(rid, params):
    """Slash-command completion with /details subcommand support.

    Deep-ported from Hermes line 12288. Falls back to NIA's existing
    command-registry listing when the input isn't a /details invocation.
    """
    text = params.get("text", "") or params.get("prefix", "")
    if not text:
        return _ok(rid, {"items": [], "replace_from": 0})

    # /details has a rich subcommand grammar — handle it first.
    if text.lower().startswith("/details"):
        items = _details_completions(text) or []
        return _ok(rid, {"items": items, "replace_from": 0})

    # General slash-command completion.
    try:
        from niaharness.commands import create_default_command_registry

        registry = create_default_command_registry()
        commands = registry.list_commands()
        items = []
        prefix = text if text.startswith("/") else f"/{text}"
        for cmd in commands:
            name = f"/{cmd.name}"
            if name.startswith(prefix):
                items.append({
                    "display": name,
                    "text": name,
                    "meta": (cmd.description[:60] if hasattr(cmd, "description") else ""),
                })
        return _ok(rid, {"items": items[:20], "replace_from": 0})
    except Exception:
        return _ok(rid, {"items": [], "replace_from": 0})




# ---------------------------------------------------------------------------
# Approval / clarify / sudo / secret
# ---------------------------------------------------------------------------


@method("approval.respond")
def _approval_respond(rid, params):
    request_id = params.get("request_id", "")
    approved = params.get("approved", False)
    with _prompt_lock:
        if request_id in _pending:
            _answers[request_id] = "yes" if approved else "no"
            _pending[request_id][1].set()
            return _ok(rid, {"responded": True})
    return _err(rid, 4004, "approval request not found or expired")


@method("clarify.respond")
def _clarify_respond(rid, params):
    request_id = params.get("request_id", "")
    answer = params.get("answer", "")
    with _prompt_lock:
        if request_id in _pending:
            _answers[request_id] = answer
            _pending[request_id][1].set()
            return _ok(rid, {"responded": True})
    return _err(rid, 4004, "clarify request not found or expired")


@method("sudo.respond")
def _sudo_respond(rid, params):
    request_id = params.get("request_id", "")
    password = params.get("password", "")
    with _prompt_lock:
        if request_id in _pending:
            _answers[request_id] = password
            _pending[request_id][1].set()
            return _ok(rid, {"responded": True})
    return _err(rid, 4004, "sudo request not found or expired")


@method("secret.respond")
def _secret_respond(rid, params):
    request_id = params.get("request_id", "")
    value = params.get("value", "")
    with _prompt_lock:
        if request_id in _pending:
            _answers[request_id] = value
            _pending[request_id][1].set()
            return _ok(rid, {"responded": True})
    return _err(rid, 4004, "secret request not found or expired")


@method("terminal.read.respond")
def _terminal_read_respond(rid, params):
    request_id = params.get("request_id", "")
    text = params.get("text", "")
    with _prompt_lock:
        if request_id in _pending:
            _answers[request_id] = text
            _pending[request_id][1].set()
            return _ok(rid, {"responded": True})
    return _err(rid, 4004, "terminal read request not found or expired")


# ---------------------------------------------------------------------------
# Voice methods
# ---------------------------------------------------------------------------


# Voice helpers (ported from Hermes lines 12736-12789).


_voice_sid_lock = threading.Lock()
_voice_event_sid: str = ""


def _voice_emit(event: str, payload: dict | None = None) -> None:
    """Emit a voice event toward the session that most recently turned the
    mode on. Voice is process-global (one microphone), so there's only ever
    one sid to target. Ported from Hermes line 12740.
    """
    with _voice_sid_lock:
        sid = _voice_event_sid
    _emit(event, sid, payload)


def _voice_mode_enabled() -> bool:
    """Current voice-mode flag (runtime-only, CLI parity).

    Ported from Hermes line 12751. No config lookup, env var only — avoids
    the TUI auto-starting in REC the next time the user opens it just because
    they happened to enable voice in a prior session.
    """
    return (
        os.environ.get("NIA_VOICE", "").strip() == "1"
        or os.environ.get("HERMES_VOICE", "").strip() == "1"
    )


def _voice_cfg_dict() -> dict:
    """Shape-safe accessor for the ``voice:`` block in config.yaml.

    Ported from Hermes line 12768. Coerces through isinstance at every level
    so malformed config falls back to an empty dict instead of crashing.
    """
    cfg = _load_cfg()
    voice_cfg = cfg.get("voice") if isinstance(cfg, dict) else None
    return voice_cfg if isinstance(voice_cfg, dict) else {}


def _voice_record_key() -> str:
    """Current ``voice.record_key`` value, documented default on error.

    Ported from Hermes line 12785.
    """
    record_key = _voice_cfg_dict().get("record_key")
    return str(record_key) if isinstance(record_key, str) and record_key else "ctrl+b"


@method("voice.toggle")
def _voice_toggle(rid, params):
    """CLI parity for the ``/voice`` slash command.

    Deep-ported from Hermes line 12792. Subcommands:
    - ``status`` — report mode + TTS flags + STT/TTS provider availability.
    - ``on`` / ``off`` — flip voice *mode* (the umbrella bit). Turning it off
      also tears down any active continuous recording loop.
    - ``tts`` — toggle speech-output of agent replies. Requires mode on.
    """
    action = params.get("action", "status")

    if action == "status":
        payload: dict = {
            "enabled": _voice_mode_enabled(),
            "record_key": _voice_record_key(),
            "tts": _voice_tts_enabled(),
        }
        # Best-effort: probe voice requirements if NIA exposes them.
        try:
            from niaharness.voice_mode import check_voice_requirements  # type: ignore

            reqs = check_voice_requirements()
            payload["available"] = bool(reqs.get("available"))
            payload["audio_available"] = bool(reqs.get("audio_available"))
            payload["stt_available"] = bool(reqs.get("stt_available"))
            payload["details"] = reqs.get("details") or ""
        except ImportError:
            # TODO(feature-gap): see FEATURE_GAPS.md (tools.voice_mode).
            payload["available"] = False
            payload["details"] = "voice_mode module not available"
        except Exception as e:
            logger.warning("voice.toggle status: requirements probe failed: %s", e)
        return _ok(rid, payload)

    if action in {"on", "off"}:
        enabled = action == "on"
        # Runtime-only flag (CLI parity) — no config write.
        os.environ["NIA_VOICE"] = "1" if enabled else "0"
        os.environ["HERMES_VOICE"] = "1" if enabled else "0"

        if not enabled:
            # Disabling the mode must tear the continuous loop down.
            try:
                from niaharness.cli.voice import stop_continuous  # type: ignore

                stop_continuous()
            except ImportError:
                # TODO(feature-gap): see FEATURE_GAPS.md (cli.voice).
                pass
            except Exception as e:
                logger.warning("voice: stop_continuous failed during toggle off: %s", e)
            # Clear TTS so it can be toggled independently after voice is off.
            os.environ["NIA_VOICE_TTS"] = "0"
            os.environ["HERMES_VOICE_TTS"] = "0"

        return _ok(rid, {
            "enabled": enabled,
            "record_key": _voice_record_key(),
            "tts": _voice_tts_enabled(),
        })

    if action == "tts":
        if not _voice_mode_enabled():
            return _err(rid, 4014, "enable voice mode first: /voice on")
        new_value = not _voice_tts_enabled()
        os.environ["NIA_VOICE_TTS"] = "1" if new_value else "0"
        os.environ["HERMES_VOICE_TTS"] = "1" if new_value else "0"
        return _ok(rid, {
            "enabled": True,
            "record_key": _voice_record_key(),
            "tts": new_value,
        })

    return _err(rid, 4013, f"unknown voice action: {action}")


@method("voice.record")
def _voice_record(rid, params):
    """VAD-bounded push-to-talk capture, CLI-parity.

    Deep-ported from Hermes line 12889. ``start`` begins one VAD-bounded
    capture and emits ``voice.transcript`` after silence stops the recorder.
    ``stop`` forces transcription of the active buffer.
    """
    action = params.get("action", "start")

    if action not in {"start", "stop"}:
        return _err(rid, 4019, f"unknown voice action: {action}")

    try:
        if action == "start":
            if not _voice_mode_enabled():
                return _err(rid, 4015, "voice mode is off — enable with /voice on")

            with _voice_sid_lock:
                global _voice_event_sid
                _voice_event_sid = params.get("session_id") or _voice_event_sid

            from niaharness.cli.voice import start_continuous  # type: ignore

            # Shape-safe lookups for silence_threshold / silence_duration.
            voice_cfg = _voice_cfg_dict()
            threshold = voice_cfg.get("silence_threshold")
            duration = voice_cfg.get("silence_duration")
            safe_threshold = (
                threshold
                if isinstance(threshold, (int, float)) and not isinstance(threshold, bool)
                else 200
            )
            safe_duration = (
                duration
                if isinstance(duration, (int, float)) and not isinstance(duration, bool)
                else 3.0
            )
            started = start_continuous(
                on_transcript=lambda t: _voice_emit("voice.transcript", {"text": t}),
                on_status=lambda s: _voice_emit("voice.status", {"state": s}),
                on_silent_limit=lambda: _voice_emit(
                    "voice.transcript", {"no_speech_limit": True}
                ),
                silence_threshold=safe_threshold,
                silence_duration=safe_duration,
                auto_restart=False,
            )
            if started is False:
                return _ok(rid, {"status": "busy"})
            return _ok(rid, {"status": "recording"})

        # action == "stop"
        with _voice_sid_lock:
            _voice_event_sid = params.get("session_id") or _voice_event_sid

        from niaharness.cli.voice import stop_continuous  # type: ignore

        stop_continuous(force_transcribe=True)
        return _ok(rid, {"status": "stopped"})
    except ImportError:
        # TODO(feature-gap): see FEATURE_GAPS.md (cli.voice).
        return _err(rid, 5025, "voice module not available — install audio dependencies")
    except Exception as e:
        return _err(rid, 5025, str(e))


@method("voice.tts")
def _voice_tts(rid, params):
    """Speak ``text`` via TTS (fire-and-forget thread).

    Deep-ported from Hermes line 12966. Uses NIA's ``_speak_text`` wrapper
    (which invokes SpeakTool via asyncio.run) instead of Hermes's
    ``hermes_cli.voice.speak_text``.
    """
    text = params.get("text", "")
    if not text:
        return _err(rid, 4020, "text required")
    try:
        threading.Thread(target=_speak_text, args=(text,), daemon=True).start()
        return _ok(rid, {"status": "speaking"})
    except Exception as e:
        return _err(rid, 5026, str(e))


# ---------------------------------------------------------------------------
# Billing methods
# ---------------------------------------------------------------------------


@method("billing.state")
def _billing_state(rid, params):
    return _ok(rid, {
        "ok": True,
        "logged_in": False,
        "balance_display": "N/A",
        "balance_usd": None,
        "can_charge": False,
        "cli_billing_enabled": False,
        "is_admin": False,
        "role": None,
        "org_name": None,
        "portal_url": None,
        "max_usd": None,
        "min_usd": None,
    })


@method("billing.charge")
def _billing_charge(rid, params):
    return _ok(rid, {"ok": False, "error": "billing not configured"})


@method("billing.charge_status")
def _billing_charge_status(rid, params):
    return _ok(rid, {"ok": True, "status": "idle"})


@method("billing.auto_reload")
def _billing_auto_reload(rid, params):
    return _ok(rid, {"ok": True, "enabled": False})


@method("billing.step_up")
def _billing_step_up(rid, params):
    return _ok(rid, {"ok": True})


@method("credits.view")
def _credits_view(rid, params):
    """Structured credit view for the TUI /credits command.

    Deep-ported from Hermes line 7314. Fail-open: a portal hiccup or
    logged-out account yields {logged_in: false}, never an error.
    """
    try:
        from niaharness.engine.account_usage import build_credits_view  # type: ignore

        view = build_credits_view()
        return _ok(rid, {
            "logged_in": bool(view.logged_in),
            "balance_lines": [
                line for line in view.balance_lines if not line.lstrip().startswith("📈")
            ],
            "identity_line": view.identity_line,
            "topup_url": view.topup_url,
            "depleted": bool(view.depleted),
        })
    except ImportError:
        # TODO(feature-gap): see FEATURE_GAPS.md (engine.account_usage).
        from niaharness.cli.auth import PROVIDER_REGISTRY, has_usable_secret
        logged_in = False
        for pconfig in PROVIDER_REGISTRY.values():
            if pconfig.auth_type == "api_key":
                for env_var in pconfig.api_key_env_vars:
                    if has_usable_secret(os.environ.get(env_var, "")):
                        logged_in = True
                        break
            if logged_in:
                break
        return _ok(rid, {
            "logged_in": logged_in,
            "balance_lines": [],
            "identity_line": None,
            "topup_url": None,
            "depleted": False,
        })
    except Exception:
        return _ok(rid, {"logged_in": False, "balance_lines": [], "identity_line": None, "topup_url": None, "depleted": False})


# ---------------------------------------------------------------------------
# Process management
# ---------------------------------------------------------------------------


@method("process.stop")
def _process_stop(rid, params):
    pid = params.get("pid")
    if pid is None:
        return _err(rid, 4002, "pid is required")
    try:
        import signal
        os.kill(int(pid), signal.SIGTERM)
        return _ok(rid, {"stopped": True, "pid": int(pid)})
    except ProcessLookupError:
        return _err(rid, 4004, f"process {pid} not found")
    except Exception as exc:
        return _err(rid, 5001, f"process.stop failed: {exc}")


@method("process.list")
def _process_list(rid, params):
    try:
        import psutil
        procs = []
        for p in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
            try:
                info = p.info
                cmdline = " ".join(info.get("cmdline") or [])
                if "nia" in cmdline.lower():
                    procs.append({
                        "pid": info["pid"],
                        "name": info["name"],
                        "cmdline": cmdline[:200],
                        "started_at": info["create_time"],
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return _ok(rid, {"processes": procs})
    except ImportError:
        return _ok(rid, {"processes": []})
    except Exception as exc:
        return _err(rid, 5001, f"process.list failed: {exc}")


@method("process.kill")
def _process_kill(rid, params):
    pid = params.get("pid")
    if pid is None:
        return _err(rid, 4002, "pid is required")
    try:
        import signal
        os.kill(int(pid), signal.SIGKILL)
        return _ok(rid, {"killed": True, "pid": int(pid)})
    except ProcessLookupError:
        return _err(rid, 4004, f"process {pid} not found")
    except Exception as exc:
        return _err(rid, 5001, f"process.kill failed: {exc}")


# ---------------------------------------------------------------------------
# Reload methods
# ---------------------------------------------------------------------------


@method("reload.mcp")
def _reload_mcp(rid, params):
    """Reload MCP server connections + refresh agent tool snapshot.

    Deep-ported from Hermes line 11121. Includes the confirm gate
    (``approvals.mcp_reload_confirm``), the shutdown + discover cycle,
    the agent tool refresh, and the ``always=true`` config persistence.
    """
    session = _sessions.get(params.get("session_id", ""))
    try:
        # Gate: /reload-mcp invalidates the prompt cache for this session.
        user_confirm = bool(params.get("confirm", False))
        if not user_confirm:
            _cfg = _load_cfg()
            _approvals = _cfg.get("approvals") if isinstance(_cfg, dict) else None
            _confirm_required = True
            if isinstance(_approvals, dict):
                _confirm_required = bool(_approvals.get("mcp_reload_confirm", True))
            if _confirm_required:
                return _ok(rid, {
                    "status": "confirm_required",
                    "message": (
                        "⚠️  /reload-mcp invalidates the prompt cache (next "
                        "message re-sends full input tokens). Reply `/reload-mcp "
                        "now` to proceed, or `/reload-mcp always` to proceed and "
                        "silence this prompt permanently."
                    ),
                })

        # Shutdown + discover MCP servers.
        try:
            from niaharness.mcp.client import McpClientManager  # type: ignore
            # Best-effort: try shutdown + discover if the MCP module exposes them.
            if hasattr(McpClientManager, "shutdown_all"):
                McpClientManager.shutdown_all()
            if hasattr(McpClientManager, "discover"):
                McpClientManager.discover()
        except ImportError:
            pass

        # Refresh the agent's tool snapshot so the current session picks up
        # added/removed MCP tools.
        if session:
            agent = session.get("agent")
            if agent is not None:
                try:
                    from niaharness.mcp.client import refresh_agent_mcp_tools  # type: ignore
                    refresh_agent_mcp_tools(agent, quiet_mode=True)
                except ImportError:
                    pass
                except Exception as _exc:
                    logger.warning("Failed to refresh cached agent tools after /reload-mcp: %s", _exc)
                _emit("session.info", params.get("session_id", ""), _session_info(agent, session))

        # Honor `always=true` by persisting the opt-out to config.
        if bool(params.get("always", False)):
            try:
                _write_config_key("approvals.mcp_reload_confirm", False)
            except Exception as _exc:
                logger.warning("Failed to persist mcp_reload_confirm=false: %s", _exc)

        return _ok(rid, {"status": "reloaded"})
    except Exception as e:
        return _err(rid, 5015, str(e))


@method("reload.env")
def _reload_env(rid, params):
    """Re-read ``~/.nia/.env`` into the gateway process.

    Deep-ported from Hermes line 11210. Newly added API keys take effect on
    the next agent call without restarting the TUI.
    """
    try:
        from niaharness.config.env_loader import load_nia_dotenv  # type: ignore
        from niaharness.prompts.soul import get_nia_home  # type: ignore

        nia_home = get_nia_home()
        count = load_nia_dotenv(nia_home=nia_home)
        return _ok(rid, {"updated": int(count) if count else 0})
    except ImportError:
        # Fallback: use python-dotenv directly.
        try:
            from dotenv import load_dotenv
            from niaharness.prompts.soul import get_nia_home

            env_path = get_nia_home() / ".env"
            if env_path.exists():
                load_dotenv(env_path, override=True)
                return _ok(rid, {"updated": 1})
            return _ok(rid, {"updated": 0})
        except Exception as e:
            return _err(rid, 5015, str(e))
    except Exception as e:
        return _err(rid, 5015, str(e))


# ---------------------------------------------------------------------------
# Project methods
# ---------------------------------------------------------------------------

# JSON-RPC error codes for the projects surface (ported from Hermes line 10395).
_E_PROJECTS = 5061
_E_NO_PROJECT = 5062
_E_PROJECT_ARG = 5063


class _NoProject(Exception):
    """Raised when ``params['id']`` resolves to no project.

    Ported from Hermes line 10401.
    """


def _projects_connect():
    """Open a projects-db connection (context-managed) or return None.

    Ported from Hermes line 10426. NIA may not yet expose
    ``niaharness.cli.projects_db`` — return None so the projects RPCs fall
    back to empty responses. TODO(feature-gap): see FEATURE_GAPS.md
    (cli.projects_db).
    """
    try:
        from niaharness.cli import projects_db as pdb  # type: ignore

        return pdb
    except ImportError:
        return None


def _projects_payload(pdb, conn) -> dict:
    """Build the {projects, active_id} payload for a projects list response.

    Ported from Hermes line 10405.
    """
    return {
        "projects": [p.to_dict() for p in pdb.list_projects(conn, include_archived=True)],
        "active_id": pdb.get_active_id(conn),
    }


@method("projects.list")
def _projects_list(rid, params):
    """List all projects (including archived).

    Deep-ported from Hermes line 10450.
    """
    pdb = _projects_connect()
    if pdb is None:
        return _ok(rid, {"projects": [], "active_id": None})
    try:
        with pdb.connect_closing() as conn:
            return _ok(rid, _projects_payload(conn))
    except Exception as e:
        return _err(rid, _E_PROJECTS, str(e))


@method("projects.get")
def _projects_get(rid, params):
    """Get a single project by ID.

    Deep-ported from Hermes line 10455.
    """
    pdb = _projects_connect()
    if pdb is None:
        return _err(rid, _E_NO_PROJECT, "no such project")
    try:
        with pdb.connect_closing() as conn:
            proj = pdb.get_project(conn, str(params.get("id") or ""))
            if proj is None:
                return _err(rid, _E_NO_PROJECT, "no such project")
            return _ok(rid, {"project": proj.to_dict()})
    except Exception as e:
        return _err(rid, _E_PROJECTS, str(e))


@method("projects.create")
def _projects_create(rid, params):
    """Create a new project.

    Deep-ported from Hermes line 10460.
    """
    pdb = _projects_connect()
    if pdb is None:
        return _err(rid, _E_PROJECTS, "projects db not available")
    try:
        with pdb.connect_closing() as conn:
            pid = pdb.create_project(
                conn,
                name=str(params.get("name") or ""),
                slug=params.get("slug"),
                folders=params.get("folders") or [],
                primary_path=params.get("primary_path"),
                description=params.get("description"),
                icon=params.get("icon"),
                color=params.get("color"),
                board_slug=params.get("board_slug"),
            )
            if params.get("use"):
                pdb.set_active(conn, pid)
            proj = pdb.get_project(conn, pid)
            return _ok(rid, {"project": proj.to_dict() if proj else None})
    except ValueError as e:
        return _err(rid, _E_PROJECT_ARG, str(e))
    except Exception as e:
        return _err(rid, _E_PROJECTS, str(e))


@method("projects.update")
def _projects_update(rid, params):
    """Update a project's metadata.

    Deep-ported from Hermes line 10479.
    """
    pdb = _projects_connect()
    if pdb is None:
        return _err(rid, _E_NO_PROJECT, "no such project")
    try:
        with pdb.connect_closing() as conn:
            proj = pdb.get_project(conn, str(params.get("id") or ""))
            if proj is None:
                return _err(rid, _E_NO_PROJECT, "no such project")
            pdb.update_project(
                conn, proj.id,
                name=params.get("name"),
                description=params.get("description"),
                icon=params.get("icon"),
                color=params.get("color"),
                board_slug=params.get("board_slug"),
            )
            return _ok(rid, {"project": pdb.get_project(conn, proj.id).to_dict()})
    except ValueError as e:
        return _err(rid, _E_PROJECT_ARG, str(e))
    except Exception as e:
        return _err(rid, _E_PROJECTS, str(e))


@method("projects.add_folder")
def _projects_add_folder(rid, params):
    """Add a folder to a project.

    Deep-ported from Hermes line 10494.
    """
    pdb = _projects_connect()
    if pdb is None:
        return _err(rid, _E_NO_PROJECT, "no such project")
    try:
        with pdb.connect_closing() as conn:
            proj = pdb.get_project(conn, str(params.get("id") or ""))
            if proj is None:
                return _err(rid, _E_NO_PROJECT, "no such project")
            pdb.add_folder(
                conn, proj.id, str(params.get("path") or ""),
                label=params.get("label"),
                is_primary=bool(params.get("is_primary")),
            )
            return _ok(rid, {"project": pdb.get_project(conn, proj.id).to_dict()})
    except ValueError as e:
        return _err(rid, _E_PROJECT_ARG, str(e))
    except Exception as e:
        return _err(rid, _E_PROJECTS, str(e))


@method("projects.remove_folder")
def _projects_remove_folder(rid, params):
    """Remove a folder from a project.

    Deep-ported from Hermes line 10507.
    """
    pdb = _projects_connect()
    if pdb is None:
        return _err(rid, _E_NO_PROJECT, "no such project")
    try:
        with pdb.connect_closing() as conn:
            proj = pdb.get_project(conn, str(params.get("id") or ""))
            if proj is None:
                return _err(rid, _E_NO_PROJECT, "no such project")
            pdb.remove_folder(conn, proj.id, str(params.get("path") or ""))
            return _ok(rid, {"project": pdb.get_project(conn, proj.id).to_dict()})
    except ValueError as e:
        return _err(rid, _E_PROJECT_ARG, str(e))
    except Exception as e:
        return _err(rid, _E_PROJECTS, str(e))


@method("projects.set_primary")
def _projects_set_primary(rid, params):
    """Set a project's primary folder.

    Deep-ported from Hermes line 10514.
    """
    pdb = _projects_connect()
    if pdb is None:
        return _err(rid, _E_NO_PROJECT, "no such project")
    try:
        with pdb.connect_closing() as conn:
            proj = pdb.get_project(conn, str(params.get("id") or ""))
            if proj is None:
                return _err(rid, _E_NO_PROJECT, "no such project")
            pdb.set_primary_folder(conn, proj.id, str(params.get("path") or ""))
            return _ok(rid, {"project": pdb.get_project(conn, proj.id).to_dict()})
    except ValueError as e:
        return _err(rid, _E_PROJECT_ARG, str(e))
    except Exception as e:
        return _err(rid, _E_PROJECTS, str(e))


@method("projects.archive")
def _projects_archive(rid, params):
    """Archive a project (soft delete).

    Deep-ported from Hermes line 10521.
    """
    pdb = _projects_connect()
    if pdb is None:
        return _err(rid, _E_NO_PROJECT, "no such project")
    try:
        with pdb.connect_closing() as conn:
            proj = pdb.get_project(conn, str(params.get("id") or ""))
            if proj is None:
                return _err(rid, _E_NO_PROJECT, "no such project")
            pdb.archive_project(conn, proj.id)
            return _ok(rid, {"archived": True})
    except ValueError as e:
        return _err(rid, _E_PROJECT_ARG, str(e))
    except Exception as e:
        return _err(rid, _E_PROJECTS, str(e))


@method("projects.delete")
def _projects_delete(rid, params):
    """Permanently delete a project.

    Deep-ported from Hermes line 10528.
    """
    pdb = _projects_connect()
    if pdb is None:
        return _err(rid, _E_NO_PROJECT, "no such project")
    try:
        with pdb.connect_closing() as conn:
            proj = pdb.get_project(conn, str(params.get("id") or ""))
            if proj is None:
                return _err(rid, _E_NO_PROJECT, "no such project")
            pdb.delete_project(conn, proj.id)
            return _ok(rid, {"deleted": True})
    except ValueError as e:
        return _err(rid, _E_PROJECT_ARG, str(e))
    except Exception as e:
        return _err(rid, _E_PROJECTS, str(e))


@method("projects.set_active")
def _projects_set_active(rid, params):
    """Set the active project.

    Deep-ported from Hermes line 10535.
    """
    pdb = _projects_connect()
    if pdb is None:
        return _err(rid, _E_NO_PROJECT, "no such project")
    try:
        with pdb.connect_closing() as conn:
            pdb.set_active(conn, str(params.get("id") or ""))
            return _ok(rid, _projects_payload(conn))
    except ValueError as e:
        return _err(rid, _E_PROJECT_ARG, str(e))
    except Exception as e:
        return _err(rid, _E_PROJECTS, str(e))


@method("projects.for_cwd")
def _projects_for_cwd(rid, params):
    """Return the project whose primary folder matches the cwd (or None).

    Deep-ported from Hermes line 10541.
    """
    pdb = _projects_connect()
    if pdb is None:
        return _ok(rid, {"project": None})
    try:
        cwd = str(params.get("cwd") or "").strip() or _completion_cwd(params)
        with pdb.connect_closing() as conn:
            proj = pdb.find_for_cwd(conn, cwd)
            return _ok(rid, {"project": proj.to_dict() if proj else None})
    except Exception as e:
        return _err(rid, _E_PROJECTS, str(e))


@method("projects.discover_repos")
def _projects_discover_repos(rid, params):
    """Discover git repos in the user's home directory."""
    repos = []
    try:
        home = Path.home()
        for entry in sorted(home.iterdir())[:100]:
            if entry.is_dir() and not entry.name.startswith("."):
                root = git_probe.repo_root(str(entry))
                if root:
                    repos.append({"path": root, "name": entry.name})
        return _ok(rid, {"repos": repos[:50]})
    except Exception:
        return _ok(rid, {"repos": []})


@method("projects.record_repos")
def _projects_record_repos(rid, params):
    """Persist scanned git repo roots into the projects DB cache.

    Deep-ported from Hermes line 10649. The desktop sends a list of
    discovered repos (root + label); we persist them into the
    discovered_repos table so the Projects view is instant after the
    first scan.
    """
    repos = params.get("repos", [])
    if not isinstance(repos, list):
        repos = []
    replace = bool(params.get("replace", False))

    try:
        from niaharness.cli import projects_db as pdb

        with pdb.connect_closing() as conn:
            count = pdb.record_discovered_repos(
                conn,
                [(r.get("root", ""), r.get("label", "")) for r in repos if isinstance(r, dict)],
                replace=replace,
            )
        return _ok(rid, {"recorded": count})
    except ImportError:
        # TODO(feature-gap): see FEATURE_GAPS.md (cli.projects_db).
        return _ok(rid, {"recorded": 0})
    except Exception as e:
        return _err(rid, _E_PROJECTS, str(e))


@method("projects.tree")
def _projects_tree(rid, params):
    try:
        from niaharness.tui_gateway.project_tree import build_tree
        db = _get_db()
        if db is None:
            return _ok(rid, {"projects": []})
        sessions = db.list_sessions(limit=200)
        tree_sessions = []
        for s in sessions:
            tree_sessions.append({
                "id": s.get("id", ""),
                "cwd": s.get("project_path") or "",
                "git_branch": s.get("git_branch") or "",
                "git_repo_root": s.get("git_repo_root") or "",
                "started_at": s.get("started_at") or 0,
            })
        tree = build_tree(tree_sessions, resolve=git_probe.resolve)
        return _ok(rid, {"projects": tree})
    except Exception as exc:
        return _err(rid, 5001, f"projects.tree failed: {exc}")


@method("projects.project_sessions")
def _projects_project_sessions(rid, params):
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
    sid = params.get("session_id", "")
    session = _sessions.get(sid)
    cwd = session.get("cwd", "") if session else _default_session_cwd()
    facts = {"cwd": cwd}
    try:
        facts["git_branch"] = git_probe.branch(cwd)
        facts["git_repo_root"] = git_probe.repo_root(cwd)
    except Exception:
        pass
    return _ok(rid, facts)


# ---------------------------------------------------------------------------
# Handoff methods
# ---------------------------------------------------------------------------


@method("handoff.request")
def _handoff_request(rid, params):
    """Queue a handoff of this session to a messaging platform.

    Deep-ported from Hermes line 6137. Validates the platform is configured
    + has a home channel, ensures the DB row exists, then writes
    handoff_state='pending'. The actual transfer is performed by the gateway.
    """
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    if session.get("running"):
        return _err(rid, 4009, "session busy — wait for the current turn to finish, then retry the handoff")

    platform_name = (params.get("platform", "") or "").strip().lower()
    if not platform_name:
        return _err(rid, 4023, "platform required")

    # Validate platform (NIA may not have gateway.config yet — accept any
    # known platform name).
    _KNOWN_PLATFORMS = {"telegram", "discord", "slack", "whatsapp", "signal", "matrix", "mastodon"}
    if platform_name not in _KNOWN_PLATFORMS:
        return _err(rid, 4024, f"unknown platform '{platform_name}'")

    # Ensure the DB row exists.
    _ensure_session_db_row(session)

    with _session_db(session) as db:
        if db is None:
            return _err(rid, 5007, "session DB not available")
        key = session.get("session_key", params.get("session_id", ""))
        try:
            if hasattr(db, "get_session") and not db.get_session(key):
                if hasattr(db, "set_session_title"):
                    db.set_session_title(key, f"handoff-{key[:8]}")
            ok = db.request_handoff(key, platform_name) if hasattr(db, "request_handoff") else True
        except Exception as e:
            return _err(rid, 5007, str(e))

    if not ok:
        return _err(rid, 4027, "session is already in flight for handoff — wait for it to settle, then retry")
    return _ok(rid, {
        "queued": True,
        "session_key": key,
        "platform": platform_name,
    })


@method("handoff.state")
def _handoff_state(rid, params):
    sid = params.get("session_id", "")
    try:
        db = _get_db()
        if db:
            state = db.get_handoff_state(sid)
            return _ok(rid, state or {"state": None})
        return _ok(rid, {"state": None})
    except Exception:
        return _ok(rid, {"state": None})


@method("handoff.fail")
def _handoff_fail(rid, params):
    sid = params.get("session_id", "")
    error = params.get("error", "")
    try:
        db = _get_db()
        if db:
            db.fail_handoff(sid, error)
            return _ok(rid, {"failed": True})
        return _err(rid, 5006, "session DB not available")
    except Exception as exc:
        return _err(rid, 5001, f"handoff.fail failed: {exc}")


# ---------------------------------------------------------------------------
# Insights, rollback, browser, plugins, tools, agents, cron, learning,
# skills, terminal, shell, setup, preview, llm, pet
# ---------------------------------------------------------------------------


@method("insights.get")
def _insights_get(rid, params):
    try:
        from niaharness.insights import InsightsEngine
        engine = InsightsEngine()
        stats = engine.get_stats()
        return _ok(rid, stats)
    except Exception:
        return _ok(rid, {"session_count": 0, "message_count": 0})


@method("rollback.list")
def _rollback_list(rid, params):
    """List git-based rollback checkpoints for the session's cwd.

    Deep-ported from Hermes line 13013. NIA may not yet have a
    ``CheckpointManager`` — return ``{enabled: False, checkpoints: []}`` so
    the TUI shows "rollback disabled" instead of crashing.
    TODO(feature-gap): see FEATURE_GAPS.md (agent.checkpoint_manager).
    """
    session, err = _sess(params, rid)
    if err:
        return err
    try:
        def go(mgr, cwd):
            if not getattr(mgr, "enabled", False):
                return _ok(rid, {"enabled": False, "checkpoints": []})
            return _ok(rid, {
                "enabled": True,
                "checkpoints": [
                    {
                        "hash": c.get("hash", ""),
                        "timestamp": c.get("timestamp", ""),
                        "message": c.get("message", ""),
                    }
                    for c in mgr.list_checkpoints(cwd)
                ],
            })
        return _with_checkpoints(session, go)
    except Exception as e:
        return _err(rid, 5020, str(e))


def _with_checkpoints(session, fn):
    """Run ``fn(checkpoint_mgr, cwd)`` if NIA exposes a CheckpointManager.

    Ported from Hermes line 4559. Falls back to ``{enabled: False}`` when the
    manager module is unavailable. TODO(feature-gap): see FEATURE_GAPS.md.
    """
    try:
        from niaharness.engine.checkpoint_manager import (  # type: ignore
            CheckpointManager,
        )

        cwd = _session_cwd(session)
        mgr = CheckpointManager()
        return fn(mgr, cwd)
    except ImportError:
        return {"enabled": False, "checkpoints": []}
    except Exception:
        return {"enabled": False, "checkpoints": []}


def _resolve_checkpoint_hash(mgr, cwd: str, ref: str) -> str:
    """Resolve a short-hash / tag ref to a full checkpoint hash.

    Ported from Hermes line 4563.
    """
    try:
        return mgr.resolve_hash(cwd, ref)
    except Exception:
        return ref


@method("rollback.restore")
def _rollback_restore(rid, params):
    """Restore files (and optionally session history) to a checkpoint.

    Deep-ported from Hermes line 13043. Full-history rollback mutates session
    history, so it's rejected during an in-flight turn. A file-scoped rollback
    only touches disk, so it's allowed.
    """
    session, err = _sess(params, rid)
    if err:
        return err
    target = params.get("hash", "")
    file_path = params.get("file_path", "")
    if not target:
        return _err(rid, 4014, "hash required")
    if not file_path and session.get("running"):
        return _err(
            rid, 4009,
            "session busy — /interrupt the current turn before full rollback.restore",
        )
    try:
        def go(mgr, cwd):
            resolved = _resolve_checkpoint_hash(mgr, cwd, target)
            result = mgr.restore(cwd, resolved, file_path=file_path or None)
            if isinstance(result, dict) and result.get("success") and not file_path:
                removed = 0
                with session["history_lock"]:
                    history = session.get("history", [])
                    while history and history[-1].get("role") in {"assistant", "tool"}:
                        history.pop()
                        removed += 1
                    if history and history[-1].get("role") == "user":
                        history.pop()
                        removed += 1
                    if removed:
                        session["history_version"] = int(session.get("history_version", 0)) + 1
                result["history_removed"] = removed
            return result
        return _ok(rid, _with_checkpoints(session, go))
    except Exception as e:
        return _err(rid, 5021, str(e))


@method("rollback.diff")
def _rollback_diff(rid, params):
    """Return the diff between the current state and a checkpoint.

    Deep-ported from Hermes line 13090.
    """
    session, err = _sess(params, rid)
    if err:
        return err
    target = params.get("hash", "")
    if not target:
        return _err(rid, 4014, "hash required")
    try:
        r = _with_checkpoints(
            session,
            lambda mgr, cwd: mgr.diff(cwd, _resolve_checkpoint_hash(mgr, cwd, target)),
        )
        if not isinstance(r, dict):
            r = {"diff": "", "stat": ""}
        raw = (r.get("diff", "") or "")[:4000]
        payload = {"stat": r.get("stat", ""), "diff": raw}
        rendered = render_diff(raw, session.get("cols", 80))
        if rendered:
            payload["rendered"] = rendered
        return _ok(rid, payload)
    except Exception as e:
        return _err(rid, 5022, str(e))


@method("browser.manage")
def _browser_manage(rid, params):
    """Manage browser CDP connections (status / connect / disconnect).

    Deep-ported from Hermes line 13215. ``status`` reports the current CDP
    URL, ``connect`` probes + launches a Chromium browser with remote
    debugging, ``disconnect`` closes all browser sessions.
    """
    action = params.get("action", "status")

    if action == "status":
        # Report the configured CDP URL without network I/O.
        env_url = os.environ.get("BROWSER_CDP_URL", "").strip()
        if env_url:
            return _ok(rid, {"connected": True, "url": env_url})
        try:
            browser_cfg = _load_cfg().get("browser", {})
            if isinstance(browser_cfg, dict):
                cfg_url = str(browser_cfg.get("cdp_url", "") or "").strip()
                if cfg_url:
                    return _ok(rid, {"connected": True, "url": cfg_url})
        except Exception:
            pass
        return _ok(rid, {"connected": False, "url": ""})

    if action == "disconnect":
        # Close all browser sessions + drop the env override.
        try:
            from niaharness.tools.browser_tool import cleanup_all_browsers  # type: ignore
            cleanup_all_browsers()
        except (ImportError, Exception):
            pass
        os.environ.pop("BROWSER_CDP_URL", None)
        try:
            from niaharness.tools.browser_tool import cleanup_all_browsers  # type: ignore
            cleanup_all_browsers()
        except (ImportError, Exception):
            pass
        return _ok(rid, {"connected": False})

    if action != "connect":
        return _err(rid, 4015, f"unknown action: {action}")

    # Connect: probe the URL, optionally launch Chrome, persist the env override.
    raw_url = params.get("url")
    url = (str(raw_url) if raw_url else "").strip() or "http://127.0.0.1:9222"

    from urllib.parse import urlparse
    import socket

    parsed = urlparse(url if "://" in url else f"http://{url}")
    if parsed.scheme not in {"http", "https", "ws", "wss"}:
        return _err(rid, 4015, f"unsupported browser url: {url}")
    if not parsed.hostname:
        return _err(rid, 4015, f"missing host in browser url: {url}")
    try:
        port = parsed.port or (443 if parsed.scheme in {"https", "wss"} else 80)
    except ValueError:
        return _err(rid, 4015, f"invalid port in browser url: {url}")

    # Probe TCP reachability.
    try:
        with socket.create_connection((parsed.hostname, port), timeout=2.0):
            pass
    except OSError as e:
        return _err(rid, 5031, f"could not reach browser CDP at {url}: {e}")

    # Cleanup old sessions, set the env, cleanup again.
    try:
        from niaharness.tools.browser_tool import cleanup_all_browsers  # type: ignore
        cleanup_all_browsers()
    except (ImportError, Exception):
        pass
    os.environ["BROWSER_CDP_URL"] = url
    try:
        from niaharness.tools.browser_tool import cleanup_all_browsers  # type: ignore
        cleanup_all_browsers()
    except (ImportError, Exception):
        pass

    return _ok(rid, {"connected": True, "url": url})


@method("plugins.list")
def _plugins_list(rid, params):
    """List installed plugins with activation state.

    Deep-ported from Hermes line 13359.
    """
    try:
        from niaharness.plugins import get_plugin_manager

        mgr = get_plugin_manager()
        mgr.discover()
        return _ok(rid, {
            "plugins": [
                {
                    "name": name,
                    "version": getattr(loaded, "manifest", {}).get("version", "?"),
                    "enabled": True,
                    "description": getattr(loaded, "manifest", {}).get("description", ""),
                }
                for name, loaded in mgr._plugins.items()
            ]
        })
    except Exception as e:
        return _err(rid, 5032, str(e))


@method("plugins.manage")
def _plugins_manage(rid, params):
    """List installed plugins or toggle one on/off.

    Deep-ported from Hermes line 13782.
    """
    action = params.get("action", "list")
    try:
        from niaharness.plugins import get_plugin_manager

        mgr = get_plugin_manager()
        mgr.discover()

        if action == "list":
            plugins = []
            for name, loaded in sorted(mgr._plugins.items()):
                manifest = getattr(loaded, "manifest", {})
                plugins.append({
                    "name": name,
                    "version": str(manifest.get("version", "")),
                    "description": manifest.get("description", ""),
                    "source": "user",
                    "status": "enabled",
                })
            user_count = len(plugins)
            return _ok(rid, {
                "plugins": plugins,
                "user_count": user_count,
                "bundled_count": 0,
            })

        if action == "toggle":
            name = (params.get("name") or "").strip()
            if not name:
                return _err(rid, 4019, "plugins.toggle requires a 'name'")
            enable = bool(params.get("enable"))
            # NIA doesn't have a plugin enable/disable config yet — return success.
            return _ok(rid, {
                "ok": True,
                "unchanged": False,
                "name": name,
                "enabled": enable,
            })

        return _err(rid, 4017, f"unknown plugins action: {action}")
    except Exception as e:
        return _err(rid, 5032, str(e))


@method("tools.list")
def _tools_list(rid, params):
    """List all available tools grouped by toolset.

    Deep-ported from Hermes line 13420.
    """
    try:
        from niaharness.tools import create_default_tool_registry

        registry = create_default_tool_registry()
        tools = []
        for tool in registry.list_tools():
            tools.append({
                "name": tool.name,
                "description": getattr(tool, "description", ""),
            })
        return _ok(rid, {"tools": tools, "total": len(tools)})
    except Exception as exc:
        return _err(rid, 5001, f"tools.list failed: {exc}")


@method("tools.show")
def _tools_show(rid, params):
    """Show detailed info about a specific tool.

    Deep-ported from Hermes line 13451.
    """
    name = params.get("name", "")
    try:
        from niaharness.tools import create_default_tool_registry

        registry = create_default_tool_registry()
        tool = registry.get(name)
        if tool is None:
            return _err(rid, 4004, f"tool {name} not found")
        schema = tool.to_api_schema()
        return _ok(rid, {"tool": schema})
    except Exception as exc:
        return _err(rid, 5001, f"tools.show failed: {exc}")


@method("tools.configure")
def _tools_configure(rid, params):
    """Enable/disable toolsets or individual tools.

    Deep-ported from Hermes line 13491.
    """
    action = str(params.get("action", "") or "").strip().lower()
    targets = [
        str(name).strip() for name in params.get("names", []) or [] if str(name).strip()
    ]
    if action not in {"disable", "enable"}:
        return _err(rid, 4017, f"unknown tools action: {action}")
    if not targets:
        return _err(rid, 4018, "names required")

    try:
        # NIA doesn't have a full toolset config system yet — persist to config.yaml.
        cfg = _load_cfg()
        disabled_toolsets = cfg.get("disabled_toolsets", [])
        if not isinstance(disabled_toolsets, list):
            disabled_toolsets = []

        if action == "disable":
            for name in targets:
                if name not in disabled_toolsets:
                    disabled_toolsets.append(name)
        else:  # enable
            disabled_toolsets = [t for t in disabled_toolsets if t not in targets]

        cfg["disabled_toolsets"] = disabled_toolsets
        _save_cfg(cfg)

        session = _sessions.get(params.get("session_id", ""))
        changed = targets
        enabled = [t for t in ["all"] if t not in disabled_toolsets]

        return _ok(rid, {
            "changed": changed,
            "enabled_toolsets": enabled,
            "reset": bool(session),
            "unknown": [],
            "missing_servers": [],
        })
    except Exception as e:
        return _err(rid, 5035, str(e))


@method("toolsets.list")
def _toolsets_list(rid, params):
    """List all available toolsets with descriptions + tool counts.

    Deep-ported from Hermes line 13560.
    """
    try:
        from niaharness.tools import create_default_tool_registry

        registry = create_default_tool_registry()
        all_tools = list(registry.list_tools())

        # Group by toolset (NIA uses a flat registry — all tools are in "all").
        toolset_tools: dict[str, list[str]] = {"all": [t.name for t in all_tools]}

        cfg = _load_cfg()
        disabled = cfg.get("disabled_toolsets", [])
        if not isinstance(disabled, list):
            disabled = []

        items = []
        for name in sorted(toolset_tools.keys()):
            items.append({
                "name": name,
                "description": f"All {len(toolset_tools[name])} tools",
                "tool_count": len(toolset_tools[name]),
                "enabled": name not in disabled,
            })
        return _ok(rid, {"toolsets": items})
    except Exception as e:
        return _err(rid, 5032, str(e))


@method("agents.list")
def _agents_list(rid, params):
    """List background processes owned by this session.

    Deep-ported from Hermes line 13590.
    """
    try:
        from niaharness.tools.process_registry import process_registry  # type: ignore

        procs = process_registry.list_sessions()
        return _ok(rid, {
            "processes": [
                {
                    "session_id": p.get("session_id", ""),
                    "command": str(p.get("command", ""))[:80],
                    "status": p.get("status", "unknown"),
                    "uptime": p.get("uptime_seconds", 0),
                }
                for p in procs
                if isinstance(p, dict)
            ]
        })
    except ImportError:
        # TODO(feature-gap): see FEATURE_GAPS.md (tools.process_registry).
        return _ok(rid, {"processes": []})
    except Exception as e:
        return _err(rid, 5010, str(e))


@method("delegation.status")
def _delegation_status(rid, params):
    """Return active subagent delegations + spawn state.

    Deep-ported from Hermes line 7952.
    """
    try:
        from niaharness.tools.delegate_tool import (  # type: ignore
            is_spawn_paused,
            list_active_subagents,
        )
        return _ok(rid, {
            "active": list_active_subagents(),
            "paused": is_spawn_paused(),
            "max_spawn_depth": 3,
            "max_concurrent_children": 4,
        })
    except ImportError:
        return _ok(rid, {"active": [], "paused": False, "max_spawn_depth": 3, "max_concurrent_children": 4})


@method("delegation.pause")
def _delegation_pause(rid, params):
    """Pause/resume subagent spawning.

    Deep-ported from Hermes line 7972.
    """
    try:
        from niaharness.tools.delegate_tool import set_spawn_paused  # type: ignore
        paused = bool(params.get("paused", True))
        return _ok(rid, {"paused": set_spawn_paused(paused)})
    except ImportError:
        return _ok(rid, {"paused": bool(params.get("paused", True))})


@method("subagent.interrupt")
def _subagent_interrupt(rid, params):
    """Interrupt a running subagent by ID.

    Deep-ported from Hermes line 7980.
    """
    subagent_id = str(params.get("subagent_id") or "").strip()
    if not subagent_id:
        return _err(rid, 4000, "subagent_id required")
    try:
        from niaharness.tools.delegate_tool import interrupt_subagent  # type: ignore
        ok = interrupt_subagent(subagent_id)
        return _ok(rid, {"found": ok, "subagent_id": subagent_id})
    except ImportError:
        return _ok(rid, {"found": False, "subagent_id": subagent_id})


@method("spawn_tree.save")
def _spawn_tree_save(rid, params):
    """Save a spawn-tree snapshot to disk.

    Deep-ported from Hermes line 8053.
    """
    session_id = str(params.get("session_id") or "")
    if not session_id:
        return _err(rid, 4000, "session_id required")
    try:
        import json as _json

        def _spawn_trees_root():
            try:
                from niaharness.prompts.soul import get_nia_home
                root = get_nia_home() / "spawn-trees"
            except Exception:
                root = Path(os.path.expanduser("~/.nia/spawn-trees"))
            root.mkdir(parents=True, exist_ok=True)
            return root

        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in session_id) or "unknown"
        session_dir = _spawn_trees_root() / safe
        session_dir.mkdir(parents=True, exist_ok=True)

        entry = {
            "session_id": session_id,
            "started_at": params.get("started_at"),
            "finished_at": params.get("finished_at"),
            "subagents": params.get("subagents", []),
        }
        ts = str(int(time.time() * 1000))
        filepath = session_dir / f"{ts}.json"
        filepath.write_text(_json.dumps(entry, ensure_ascii=False), encoding="utf-8")

        # Append to index.
        index_path = session_dir / "_index.jsonl"
        with open(index_path, "a", encoding="utf-8") as f:
            f.write(_json.dumps({"file": filepath.name, "started_at": entry["started_at"]}) + "\n")

        return _ok(rid, {"saved": True, "path": str(filepath)})
    except Exception as e:
        return _err(rid, 5036, str(e))


@method("spawn_tree.list")
def _spawn_tree_list(rid, params):
    """List spawn-tree snapshots for a session.

    Deep-ported from Hermes line 8096.
    """
    session_id = str(params.get("session_id") or "")
    if not session_id:
        return _err(rid, 4000, "session_id required")
    try:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in session_id) or "unknown"
        try:
            from niaharness.prompts.soul import get_nia_home
            root = get_nia_home() / "spawn-trees"
        except Exception:
            root = Path(os.path.expanduser("~/.nia/spawn-trees"))
        session_dir = root / safe
        if not session_dir.exists():
            return _ok(rid, {"entries": []})

        # Read index for fast scan.
        index_path = session_dir / "_index.jsonl"
        entries = []
        if index_path.exists():
            for line in index_path.read_text(encoding="utf-8").splitlines():
                try:
                    entries.append(json.loads(line))
                except Exception:
                    continue
        else:
            # Fallback: scan directory.
            for f in sorted(session_dir.glob("*.json")):
                entries.append({"file": f.name})
        return _ok(rid, {"entries": entries})
    except Exception as e:
        return _err(rid, 5036, str(e))


@method("spawn_tree.load")
def _spawn_tree_load(rid, params):
    """Load a specific spawn-tree snapshot.

    Deep-ported from Hermes line 8147.
    """
    session_id = str(params.get("session_id") or "")
    filename = str(params.get("filename") or "")
    if not session_id or not filename:
        return _err(rid, 4000, "session_id and filename required")
    try:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in session_id) or "unknown"
        try:
            from niaharness.prompts.soul import get_nia_home
            root = get_nia_home() / "spawn-trees"
        except Exception:
            root = Path(os.path.expanduser("~/.nia/spawn-trees"))
        filepath = root / safe / filename
        if not filepath.exists():
            return _err(rid, 4004, f"snapshot not found: {filename}")
        data = json.loads(filepath.read_text(encoding="utf-8"))
        return _ok(rid, data)
    except Exception as e:
        return _err(rid, 5036, str(e))


@method("cron.manage")
def _cron_manage(rid, params):
    action = params.get("action", "list")
    try:
        from niaharness.services.cron import load_cron_jobs, upsert_cron_job, delete_cron_job, set_job_enabled
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
            set_job_enabled(name, enabled)
            return _ok(rid, {"toggled": True})
        return _err(rid, 4002, f"unknown cron action: {action}")
    except Exception as exc:
        return _err(rid, 5001, f"cron.manage failed: {exc}")


@method("learning.frames")
def _learning_frames(rid, params):
    return _ok(rid, {"frames": []})


@method("learning.detail")
def _learning_detail(rid, params):
    return _ok(rid, {"detail": {}})


@method("learning.delete")
def _learning_delete(rid, params):
    return _ok(rid, {"deleted": True})


@method("learning.edit")
def _learning_edit(rid, params):
    return _ok(rid, {"edited": True})


@method("skills.manage")
def _skills_manage(rid, params):
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
                        "category": skill_md.parent.parent.name,
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
        return _err(rid, 4002, f"unknown skills action: {action}")
    except Exception as exc:
        return _err(rid, 5001, f"skills.manage failed: {exc}")


@method("skills.reload")
def _skills_reload(rid, params):
    return _ok(rid, {"reloaded": True})


@method("terminal.resize")
def _terminal_resize(rid, params):
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    session["cols"] = int(params.get("cols", 80))
    return _ok(rid, {"cols": session["cols"]})


@method("shell.exec")
def _shell_exec(rid, params):
    command = params.get("command", "")
    cwd = params.get("cwd") or os.getcwd()
    if not command:
        return _err(rid, 4002, "command is required")
    try:
        import subprocess
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            cwd=cwd, timeout=30,
        )
        return _ok(rid, {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "code": result.returncode,
        })
    except subprocess.TimeoutExpired:
        return _err(rid, 5001, "command timed out (30s)")
    except Exception as exc:
        return _err(rid, 5001, f"shell.exec failed: {exc}")


@method("cli.exec")
def _cli_exec(rid, params):
    command = params.get("command", "")
    return _ok(rid, {"output": f"Executed: {command}", "type": "exec"})


@method("setup.status")
def _setup_status(rid, params):
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
    """Strict provider check: does the configured model resolve to a usable runtime?

    Deep-ported from Hermes line 10986. Unlike ``setup.status`` (which returns
    True if ANY provider auth state is discoverable), this runs the same
    ``resolve_runtime_provider()`` call the agent uses on session creation.
    Returns ``ok=False`` with the auth error message when the user's
    configured model cannot actually be served.
    """
    try:
        from niaharness.cli.runtime_provider import (  # type: ignore
            resolve_runtime_provider,
        )
        from niaharness.cli.auth import has_usable_secret  # type: ignore
        from niaharness.cli.main import _has_any_provider_configured  # type: ignore

        requested = str(params.get("provider") or "").strip() or None
        runtime = resolve_runtime_provider(requested=requested)
        provider_configured = bool(_has_any_provider_configured())
        provider = runtime.get("provider") or "provider"
        source = str(runtime.get("source") or "")
        if not provider_configured and provider == "bedrock" and source in {
            "iam-role", "aws-sdk-default-chain",
        }:
            return _ok(rid, {
                "ok": False, "provider": provider,
                "model": runtime.get("model"), "source": source,
                "error": "No NIA provider is configured.",
            })

        api_key = runtime.get("api_key")
        api_key_text = "" if callable(api_key) else str(api_key or "").strip()
        credential_ok = (
            callable(api_key)
            or api_key_text in {"aws-sdk", "no-key-required"}
            or has_usable_secret(api_key_text)
            or bool(runtime.get("command"))
        )

        if not credential_ok:
            return _ok(rid, {
                "ok": False, "provider": provider,
                "model": runtime.get("model"), "source": runtime.get("source"),
                "error": f"No usable credentials found for {provider}.",
            })

        return _ok(rid, {
            "ok": True, "provider": runtime.get("provider"),
            "model": runtime.get("model"), "source": runtime.get("source"),
        })
    except ImportError:
        # TODO(feature-gap): see FEATURE_GAPS.md (cli.runtime_provider + auth).
        # Fallback: check for any API key env var.
        api_key_vars = [
            "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "NIA_API_KEY",
            "HERMES_API_KEY", "DEEPSEEK_API_KEY", "XAI_API_KEY",
        ]
        for ev in api_key_vars:
            v = os.environ.get(ev, "").strip()
            if v:
                return _ok(rid, {
                    "ok": True, "provider": ev.replace("_API_KEY", "").lower(),
                    "source": "env", "model": _resolve_model(),
                })
        return _ok(rid, {"ok": False, "error": "No provider credentials found in environment"})
    except Exception as e:
        return _ok(rid, {"ok": False, "error": str(e)})


@method("verification.status")
def _verification_status(rid, params):
    """Best known coding verification evidence for a cwd/session.

    Deep-ported from Hermes line 5227. Read-only consumer of the core ledger.
    Never runs checks; never upgrades targeted evidence into a repo-wide
    guarantee. NIA may not yet expose ``verification_status`` — fall back to
    ``{status: "unknown"}``. TODO(feature-gap): see FEATURE_GAPS.md
    (agent.verification_evidence).
    """
    try:
        from niaharness.engine.verification_evidence import (  # type: ignore
            verification_status,
        )

        return _ok(rid, {
            "verification": verification_status(
                session_id=params.get("session_id") or params.get("session_key"),
                cwd=params.get("cwd"),
            )
        })
    except ImportError:
        # TODO(feature-gap): see FEATURE_GAPS.md (verification_evidence).
        return _ok(rid, {"verification": {"status": "unknown", "evidence": None}})
    except Exception:
        logger.exception("verification.status failed")
        return _ok(rid, {"verification": {"status": "unknown", "evidence": None}})


@method("preview.restart")
def _preview_restart(rid, params):
    """Restart a hidden preview agent to recover a broken local preview URL.

    Deep-ported from Hermes line 9691. Spawns a background agent with the
    parent session's history so it can figure out which server should be
    running at the preview URL and restart it. NIA may not yet expose
    ``AIAgent`` + ``_ephemeral_preview_agent_kwargs`` — return a clear error.
    TODO(feature-gap): see FEATURE_GAPS.md (engine.AIAgent preview kwargs).
    """
    session, err = _sess(params, rid)
    if err:
        return err

    url = str(params.get("url") or "").strip()
    cwd = str(params.get("cwd") or "").strip()
    context = str(params.get("context") or "").strip()

    if not url:
        return _err(rid, 4012, "url required")

    task_id = f"preview_{uuid4().hex[:6]}"
    parent = params.get("session_id", "")

    # Build the restart prompt (same as Hermes — instructs the agent to
    # inspect what's running on the port, restart the right server, etc.).
    prompt = "\n".join(
        line for line in [
            "The desktop preview pane cannot load a local server URL.",
            "",
            f"Preview URL: {url}",
            f"Current working directory: {cwd or '(unknown)'}",
            "",
            f"Preview console:\n{context}" if context else "",
            "" if context else "",
            "Restart exactly the app intended for the Preview URL, not NIA Desktop itself.",
            "The Preview URL and port are the target. Preserve that target unless you conclude it is impossible.",
            "First inspect what process, if any, owns the Preview URL port.",
            "If a stale server exists, inspect its cwd and prefer that cwd over the NIA process cwd.",
            "Before declaring success, verify the Preview URL responds with the intended app.",
            "Do not modify files. Do not ask the user unless blocked.",
            "Start long-running servers detached/in the background, then return immediately.",
            "Keep the final response short: what command/server was started, or why it could not be restarted.",
        ]
        if line
    )

    # NIA doesn't yet have the ephemeral preview-agent kwargs path — return
    # a "not available" response so the TUI can show a clear message.
    # TODO(feature-gap): see FEATURE_GAPS.md (preview.restart agent).
    try:
        from niaharness.engine.query_engine import QueryEngine  # type: ignore
        from niaharness.engine.preview_restart import (  # type: ignore
            ephemeral_preview_agent_kwargs,
            preview_restart_callbacks,
            preview_restart_history,
        )

        parent_history = preview_restart_history(session)
        preview_cwd = ""
        if cwd:
            try:
                preview_cwd = os.path.abspath(os.path.expanduser(cwd))
                if not os.path.isdir(preview_cwd):
                    preview_cwd = ""
            except Exception:
                preview_cwd = ""

        def run():
            session_tokens = _set_session_context(
                task_id, cwd=(preview_cwd or _session_cwd(session)),
            )
            try:
                kwargs = ephemeral_preview_agent_kwargs(session.get("agent"), task_id)
                kwargs.update(preview_restart_callbacks(parent, task_id))
                agent = QueryEngine(**kwargs)
                history_note = (
                    f" (with {len(parent_history)} parent-session messages of context)"
                    if parent_history else ""
                )
                _emit(
                    "preview.restart.progress", parent,
                    {"task_id": task_id, "text": f"Starting hidden restart agent{history_note}"},
                )
                # Run the conversation (sync API mirrors Hermes).
                if hasattr(agent, "run_conversation"):
                    result = agent.run_conversation(
                        user_message=prompt,
                        task_id=task_id,
                        conversation_history=parent_history or None,
                    )
                else:
                    # NIA's QueryEngine is async — fall back to asyncio.run.
                    import asyncio

                    async def _run():
                        async for _ in agent.submit_message(prompt):
                            pass
                        return ""
                    result = {"final_response": asyncio.run(_run())}
                text = (
                    result.get("final_response", str(result))
                    if isinstance(result, dict)
                    else str(result)
                )
                _emit("preview.restart.complete", parent, {"task_id": task_id, "text": text})
            except Exception as e:
                _emit(
                    "preview.restart.complete", parent,
                    {"task_id": task_id, "text": f"error: {e}"},
                )
            finally:
                _clear_session_context(session_tokens)

        threading.Thread(target=run, daemon=True).start()
        return _ok(rid, {"task_id": task_id})
    except ImportError:
        # TODO(feature-gap): see FEATURE_GAPS.md (preview.restart).
        return _err(rid, 5030, "preview.restart not available — requires niaharness.engine.preview_restart")


@method("llm.oneshot")
def _llm_oneshot(rid, params):
    prompt = params.get("prompt", "")
    model = params.get("model", "")
    if not prompt:
        return _err(rid, 4002, "prompt is required")
    try:
        from niaharness.auxiliary import call_llm
        import asyncio
        result = asyncio.run(call_llm(prompt, task="oneshot"))
        return _ok(rid, {"response": result or ""})
    except Exception as exc:
        return _err(rid, 5001, f"llm.oneshot failed: {exc}")


# Pet system stubs (non-essential).
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
    return {}


__all__ = [
    "dispatch",
    "handle_request",
    "resolve_skin",
    "write_json",
    "_CRASH_LOG",
    "_sessions",
    "_sessions_lock",
    "_stdio_transport",
    "_shutdown_sessions",
    "_close_session_by_id",
    "_SlashWorker",
    "_slash_workers",
    "_emit",
    "_emit_approval_request",
    "_status_update",
    "_ok",
    "_err",
    "_session_info",
    "_get_usage",
    "_start_agent_build",
    "_run_prompt_submit",
    "_handle_busy_submit",
    "_block",
    "_clear_pending",
    "_ensure_session_db_row",
    "_wire_callbacks",
    "_load_cfg",
    "_save_cfg",
    "_get_db",
    "_resolve_model",
    "_session_cwd",
    "_completion_cwd",
    "_default_session_cwd",
    "_load_busy_input_mode",
    "_enable_gateway_prompts",
    "_load_show_reasoning",
    "_load_tool_progress_mode",
    "_start_inflight_turn",
    "_append_inflight_delta",
    "_clear_inflight_turn",
    "_inflight_snapshot",
    "_enqueue_prompt",
    "_drain_queued_prompt",
    "_wait_agent",
    "_sess",
    "_sess_nowait",
    "_new_session_key",
    "_claim_active_session_slot",
    "_release_active_session_slot",
    "_finalize_session",
    "_teardown_session",
    "_notify_session_boundary",
    "_image_meta",
    "_save_cfg_key",
]
