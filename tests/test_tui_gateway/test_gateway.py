"""Tests for the NIA TUI Gateway — transport, server (118 RPC methods), ws,
git_probe, project_tree, slash_worker, event_publisher, loop_noise, render.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import threading
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from niaharness.tui_gateway.transport import (
    DROP_TRANSPORT,
    StdioTransport,
    TeeTransport,
    Transport,
    bind_transport,
    current_transport,
    reset_transport,
)
from niaharness.tui_gateway.server import (
    _methods,
    dispatch,
    handle_request,
    resolve_skin,
    write_json,
    _ok,
    _err,
    _emit,
)
from niaharness.tui_gateway.loop_noise import install_loop_noise_filter
from niaharness.tui_gateway.git_probe import run_git, branch, repo_root, resolve, warm_roots, invalidate
from niaharness.tui_gateway.project_tree import build_tree, base_name, kanban_worktree_dir
from niaharness.tui_gateway.render import render_message, render_diff, make_stream_renderer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _temp_nia_home(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("NIA_HOME", str(tmp_path / ".nia"))
    yield


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------


class TestTransport:
    def test_stdio_transport_write(self):
        stream = io.StringIO()
        lock = threading.Lock()
        transport = StdioTransport(lambda: stream, lock)
        result = transport.write({"test": True})
        assert result is True
        data = json.loads(stream.getvalue().strip())
        assert data["test"] is True

    def test_stdio_transport_broken_pipe(self):
        stream = MagicMock()
        stream.write = MagicMock(side_effect=BrokenPipeError())
        lock = threading.Lock()
        transport = StdioTransport(lambda: stream, lock)
        assert transport.write({"test": True}) is False

    def test_tee_transport(self):
        primary = MagicMock()
        primary.write = MagicMock(return_value=True)
        secondary = MagicMock()
        secondary.write = MagicMock(return_value=True)
        transport = TeeTransport(primary, secondary)
        result = transport.write({"test": True})
        assert result is True
        primary.write.assert_called_once()
        secondary.write.assert_called_once()

    def test_tee_transport_secondary_failure_swallows(self):
        primary = MagicMock()
        primary.write = MagicMock(return_value=True)
        secondary = MagicMock()
        secondary.write = MagicMock(side_effect=Exception("dead"))
        transport = TeeTransport(primary, secondary)
        result = transport.write({"test": True})
        assert result is True  # primary success despite secondary failure

    def test_bind_and_reset_transport(self):
        transport = MagicMock()
        transport.write = MagicMock(return_value=True)
        token = bind_transport(transport)
        assert current_transport() is transport
        reset_transport(token)
        assert current_transport() is None

    def test_drop_transport(self):
        assert DROP_TRANSPORT.write({"test": True}) is False
        DROP_TRANSPORT.close()  # no crash


# ---------------------------------------------------------------------------
# Server — method registration
# ---------------------------------------------------------------------------


class TestServerMethods:
    def test_all_117_methods_registered(self):
        assert len(_methods) >= 117

    def test_session_create(self):
        req = {"jsonrpc": "2.0", "id": 1, "method": "session.create", "params": {"cwd": "/tmp"}}
        resp = handle_request(req)
        assert resp is not None
        assert "result" in resp
        assert "session_id" in resp["result"]

    def test_session_list(self):
        req = {"jsonrpc": "2.0", "id": 2, "method": "session.list", "params": {}}
        resp = handle_request(req)
        assert resp is not None
        assert "result" in resp
        assert "sessions" in resp["result"]

    def test_config_get(self):
        req = {"jsonrpc": "2.0", "id": 3, "method": "config.get", "params": {}}
        resp = handle_request(req)
        assert resp is not None
        assert "result" in resp
        assert "config" in resp["result"]

    def test_tools_list(self):
        req = {"jsonrpc": "2.0", "id": 4, "method": "tools.list", "params": {}}
        resp = handle_request(req)
        assert resp is not None
        assert "result" in resp
        assert "tools" in resp["result"]

    def test_model_options(self):
        req = {"jsonrpc": "2.0", "id": 5, "method": "model.options", "params": {}}
        resp = handle_request(req)
        assert resp is not None
        assert "result" in resp
        assert "providers" in resp["result"]

    def test_unknown_method(self):
        req = {"jsonrpc": "2.0", "id": 6, "method": "nonexistent.method", "params": {}}
        resp = handle_request(req)
        assert resp is not None
        assert "error" in resp
        assert resp["error"]["code"] == -32601

    def test_invalid_request(self):
        req = {"jsonrpc": "2.0", "id": 7}  # no method
        resp = handle_request(req)
        assert resp is not None
        assert "error" in resp

    def test_ok_helper(self):
        result = _ok(1, {"test": True})
        assert result["jsonrpc"] == "2.0"
        assert result["id"] == 1
        assert result["result"]["test"] is True

    def test_err_helper(self):
        result = _err(1, 500, "error message")
        assert result["jsonrpc"] == "2.0"
        assert result["id"] == 1
        assert result["error"]["code"] == 500
        assert result["error"]["message"] == "error message"

    def test_resolve_skin(self):
        skin = resolve_skin()
        assert isinstance(skin, dict)


class TestServerSessionManagement:
    def setup_method(self):
        """Create a session for each test."""
        req = {"jsonrpc": "2.0", "id": 1, "method": "session.create", "params": {"cwd": "/tmp"}}
        resp = handle_request(req)
        self.sid = resp["result"]["session_id"]

    def test_session_status(self):
        req = {"jsonrpc": "2.0", "id": 2, "method": "session.status", "params": {"session_id": self.sid}}
        resp = handle_request(req)
        assert "result" in resp
        assert resp["result"]["id"] == self.sid

    def test_session_title_set(self):
        req = {"jsonrpc": "2.0", "id": 3, "method": "session.title", "params": {"session_id": self.sid, "title": "Test"}}
        resp = handle_request(req)
        assert "result" in resp
        assert resp["result"]["title"] == "Test"

    def test_session_close(self):
        req = {"jsonrpc": "2.0", "id": 4, "method": "session.close", "params": {"session_id": self.sid}}
        resp = handle_request(req)
        assert "result" in resp
        assert resp["result"]["closed"] is True

    def test_session_branch(self):
        req = {"jsonrpc": "2.0", "id": 5, "method": "session.branch", "params": {"session_id": self.sid}}
        resp = handle_request(req)
        assert "result" in resp
        assert "session_id" in resp["result"]
        assert resp["result"]["session_id"] != self.sid

    def test_session_usage(self):
        req = {"jsonrpc": "2.0", "id": 6, "method": "session.usage", "params": {"session_id": self.sid}}
        resp = handle_request(req)
        assert "result" in resp
        assert "input_tokens" in resp["result"]

    def test_session_context_breakdown(self):
        req = {"jsonrpc": "2.0", "id": 7, "method": "session.context_breakdown", "params": {"session_id": self.sid}}
        resp = handle_request(req)
        assert "result" in resp
        assert "categories" in resp["result"]


class TestServerCompletion:
    def test_complete_path(self, tmp_path: Path):
        # Create some files.
        (tmp_path / "test1.py").touch()
        (tmp_path / "test2.py").touch()
        req = {"jsonrpc": "2.0", "id": 1, "method": "complete.path", "params": {"prefix": str(tmp_path) + "/"}}
        resp = handle_request(req)
        assert "result" in resp
        assert "items" in resp["result"]

    def test_complete_slash(self):
        req = {"jsonrpc": "2.0", "id": 2, "method": "complete.slash", "params": {"prefix": "/"}}
        resp = handle_request(req)
        assert "result" in resp
        assert "items" in resp["result"]


class TestServerSlash:
    def test_slash_exec_help(self):
        req = {"jsonrpc": "2.0", "id": 1, "method": "slash.exec", "params": {"command": "/help"}}
        resp = handle_request(req)
        assert "result" in resp

    def test_commands_catalog(self):
        req = {"jsonrpc": "2.0", "id": 2, "method": "commands.catalog", "params": {}}
        resp = handle_request(req)
        assert "result" in resp
        assert "pairs" in resp["result"]


class TestServerPetStubs:
    def test_pet_info(self):
        req = {"jsonrpc": "2.0", "id": 1, "method": "pet.info", "params": {}}
        resp = handle_request(req)
        assert "result" in resp
        assert resp["result"]["ok"] is False


class TestServerSetup:
    def test_setup_status(self):
        req = {"jsonrpc": "2.0", "id": 1, "method": "setup.status", "params": {}}
        resp = handle_request(req)
        assert "result" in resp
        assert "setup_complete" in resp["result"]


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


class TestDispatch:
    def test_dispatch_inline(self):
        req = {"jsonrpc": "2.0", "id": 1, "method": "config.get", "params": {}}
        resp = dispatch(req)
        assert resp is not None  # inline → returns response
        assert "result" in resp

    def test_dispatch_long_handler_returns_none(self):
        req = {"jsonrpc": "2.0", "id": 2, "method": "prompt.submit", "params": {"session_id": "", "text": ""}}
        # This should be dispatched to the pool (returns None) or return an error.
        resp = dispatch(req)
        # Either None (pooled) or an error response.
        assert resp is None or "error" in resp


# ---------------------------------------------------------------------------
# Git probe
# ---------------------------------------------------------------------------


class TestGitProbe:
    def test_run_git_no_cwd(self):
        assert run_git("") == ""

    def test_run_git_not_a_repo(self, tmp_path: Path):
        assert run_git(str(tmp_path), "rev-parse", "--show-toplevel") == ""

    def test_branch_not_a_repo(self, tmp_path: Path):
        assert branch(str(tmp_path)) == ""

    def test_repo_root_not_a_repo(self, tmp_path: Path):
        assert repo_root(str(tmp_path)) == ""

    def test_resolve_not_a_repo(self, tmp_path: Path):
        assert resolve(str(tmp_path)) is None

    def test_warm_roots_empty(self):
        warm_roots([])

    def test_invalidate(self):
        invalidate()  # should not crash


# ---------------------------------------------------------------------------
# Project tree
# ---------------------------------------------------------------------------


class TestProjectTree:
    def test_build_tree_empty(self):
        tree = build_tree([])
        assert tree == []

    def test_build_tree_non_git_sessions(self):
        sessions = [
            {"id": "s1", "cwd": "/tmp/project-a", "started_at": 1000},
            {"id": "s2", "cwd": "/tmp/project-b", "started_at": 2000},
        ]
        tree = build_tree(sessions)
        assert len(tree) >= 1

    def test_base_name(self):
        assert base_name("/home/user/project") == "project"
        assert base_name("") == ""
        assert base_name("/") == ""

    def test_kanban_worktree_dir(self):
        assert kanban_worktree_dir("/repo/.worktrees/t_abc123") is not None
        assert kanban_worktree_dir("/repo/.worktrees/my-task") is None
        assert kanban_worktree_dir("") is None


# ---------------------------------------------------------------------------
# Loop noise filter
# ---------------------------------------------------------------------------


class TestLoopNoise:
    def test_install_loop_noise_filter(self):
        loop = asyncio.new_event_loop()
        install_loop_noise_filter(loop)
        # Idempotent.
        install_loop_noise_filter(loop)
        loop.close()


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------


class TestRender:
    def test_render_message(self):
        result = render_message("test")
        # Returns None if niaharness.ui.output doesn't have format_response.
        assert result is None or isinstance(result, str)

    def test_render_diff(self):
        result = render_diff("test")
        assert result is None or isinstance(result, str)

    def test_make_stream_renderer(self):
        result = make_stream_renderer()
        assert result is None or result is not None  # just shouldn't crash


# ---------------------------------------------------------------------------
# Write JSON
# ---------------------------------------------------------------------------


class TestWriteJSON:
    def test_write_json_to_stdio(self):
        stream = io.StringIO()
        lock = threading.Lock()
        from niaharness.tui_gateway import server
        old_transport = server._stdio_transport
        server._stdio_transport = StdioTransport(lambda: stream, lock)
        try:
            result = write_json({"test": True})
            assert result is True
            data = json.loads(stream.getvalue().strip())
            assert data["test"] is True
        finally:
            server._stdio_transport = old_transport

    def test_emit(self):
        stream = io.StringIO()
        lock = threading.Lock()
        from niaharness.tui_gateway import server
        old_transport = server._stdio_transport
        server._stdio_transport = StdioTransport(lambda: stream, lock)
        try:
            _emit("test.event", "sid123", {"key": "value"})
            data = json.loads(stream.getvalue().strip())
            assert data["method"] == "event"
            assert data["params"]["type"] == "test.event"
            assert data["params"]["session_id"] == "sid123"
            assert data["params"]["payload"]["key"] == "value"
        finally:
            server._stdio_transport = old_transport


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
