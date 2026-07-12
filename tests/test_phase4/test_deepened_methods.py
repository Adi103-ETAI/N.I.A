"""Tests for Phase 4 deepened tui_gateway methods."""

import json
import os
import time
from pathlib import Path

import pytest
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from niaharness.tui_gateway.server import (
    handle_request,
    _sessions,
    _sessions_lock,
    _session_live_status,
    _session_lookup_key,
    _find_live_session_by_key,
    _session_pending_kind,
    _session_live_item,
    _MAX_LIVE_SESSIONS,
    _session_is_evictable,
    _session_is_lru_evictable,
)


class TestSessionCreate:
    def test_create_returns_lazy_info(self):
        req = {"jsonrpc": "2.0", "id": 1, "method": "session.create", "params": {"cwd": "/tmp"}}
        resp = handle_request(req)
        assert "result" in resp
        result = resp["result"]
        assert "session_id" in result
        assert "stored_session_id" in result
        assert result["info"]["lazy"] is True
        sid = result["session_id"]
        with _sessions_lock:
            _sessions.pop(sid, None)

    def test_create_with_model_override(self):
        req = {
            "jsonrpc": "2.0", "id": 2, "method": "session.create",
            "params": {"cwd": "/tmp", "model": "claude-sonnet-4-20250514", "provider": "anthropic"},
        }
        resp = handle_request(req)
        result = resp["result"]
        assert result["info"]["model"] == "claude-sonnet-4-20250514"
        sid = result["session_id"]
        with _sessions_lock:
            _sessions.pop(sid, None)

    def test_create_with_reasoning_effort(self):
        req = {
            "jsonrpc": "2.0", "id": 3, "method": "session.create",
            "params": {"cwd": "/tmp", "reasoning_effort": "high"},
        }
        resp = handle_request(req)
        result = resp["result"]
        sid = result["session_id"]
        with _sessions_lock:
            session = _sessions.get(sid, {})
            assert session.get("create_reasoning_override") == {"effort": "high", "enabled": True}
            _sessions.pop(sid, None)

    def test_create_with_fast(self):
        req = {
            "jsonrpc": "2.0", "id": 4, "method": "session.create",
            "params": {"cwd": "/tmp", "fast": True},
        }
        resp = handle_request(req)
        result = resp["result"]
        sid = result["session_id"]
        with _sessions_lock:
            session = _sessions.get(sid, {})
            assert session.get("create_service_tier_override") == "priority"
            _sessions.pop(sid, None)


class TestSessionCapEnforcement:
    def test_max_live_sessions_default(self):
        assert _MAX_LIVE_SESSIONS >= 1

    def test_session_is_evictable_running(self):
        session = {"running": True, "last_active": time.time()}
        assert _session_is_evictable("sid", session, time.time()) is False

    def test_session_is_evictable_idle(self):
        old_time = time.time() - 999999
        session = {"running": False, "last_active": old_time}
        assert _session_is_evictable("sid", session, time.time()) is True

    def test_session_is_lru_evictable_running(self):
        session = {"running": True}
        assert _session_is_lru_evictable("sid", session) is False

    def test_session_is_lru_evictable_idle(self):
        session = {"running": False, "_finalized": False}
        assert _session_is_lru_evictable("sid", session) is True


class TestSessionLiveHelpers:
    def test_session_lookup_key_with_agent(self):
        class FakeAgent:
            session_id = "test-key"
        session = {"agent": FakeAgent(), "session_key": "fallback-key"}
        assert _session_lookup_key(session) == "test-key"

    def test_session_lookup_key_fallback(self):
        session = {"agent": None, "session_key": "fallback-key"}
        assert _session_lookup_key(session) == "fallback-key"

    def test_find_live_session_by_key_empty(self):
        assert _find_live_session_by_key("") is None

    def test_session_pending_kind_empty(self):
        assert _session_pending_kind("nonexistent") == ""

    def test_session_live_status_idle(self):
        session = {"running": False, "agent_ready": None}
        assert _session_live_status("sid", session) == "idle"

    def test_session_live_status_working(self):
        session = {"running": True, "agent_ready": None}
        assert _session_live_status("sid", session) == "working"

    def test_session_live_item(self):
        now = time.time()
        session = {
            "agent": None,
            "history": [{"role": "user", "content": "hello"}],
            "running": False,
            "session_key": "test-key",
            "created_at": now,
            "last_active": now,
            "pending_title": "Test",
        }
        item = _session_live_item("sid", session, current_sid="sid")
        assert item["id"] == "sid"
        assert item["current"] is True
        assert item["session_key"] == "test-key"
        assert item["title"] == "Test"
        assert item["status"] == "idle"
        assert item["message_count"] == 1


class TestSessionMethods:
    def test_delete_nonexistent(self):
        req = {"jsonrpc": "2.0", "id": 1, "method": "session.delete", "params": {"session_id": "nonexistent"}}
        resp = handle_request(req)
        assert "result" in resp

    def test_status_nonexistent(self):
        req = {"jsonrpc": "2.0", "id": 1, "method": "session.status", "params": {"session_id": "nonexistent"}}
        resp = handle_request(req)
        assert "error" in resp
        assert resp["error"]["code"] == 4001

    def test_interrupt_nonexistent(self):
        req = {"jsonrpc": "2.0", "id": 1, "method": "session.interrupt", "params": {"session_id": "nonexistent"}}
        resp = handle_request(req)
        assert "error" in resp

    def test_branch_nonexistent(self):
        req = {"jsonrpc": "2.0", "id": 1, "method": "session.branch", "params": {"session_id": "nonexistent"}}
        resp = handle_request(req)
        assert "error" in resp

    def test_active_list_empty(self):
        with _sessions_lock:
            old = dict(_sessions)
            _sessions.clear()
        try:
            req = {"jsonrpc": "2.0", "id": 1, "method": "session.active_list", "params": {}}
            resp = handle_request(req)
            assert resp["result"]["sessions"] == []
        finally:
            with _sessions_lock:
                _sessions.update(old)


class TestConfigMethods:
    def test_config_show_returns_sections(self):
        req = {"jsonrpc": "2.0", "id": 1, "method": "config.show", "params": {}}
        resp = handle_request(req)
        assert "sections" in resp["result"]
        titles = [s["title"] for s in resp["result"]["sections"]]
        assert "Model" in titles

    def test_config_get_mtime(self):
        req = {"jsonrpc": "2.0", "id": 1, "method": "config.get", "params": {"key": "mtime"}}
        resp = handle_request(req)
        assert "mtime" in resp["result"]

    def test_config_get_busy(self):
        req = {"jsonrpc": "2.0", "id": 1, "method": "config.get", "params": {"key": "busy"}}
        resp = handle_request(req)
        assert "value" in resp["result"]

    def test_config_get_unknown(self):
        req = {"jsonrpc": "2.0", "id": 1, "method": "config.get", "params": {"key": "nonexistent_key"}}
        resp = handle_request(req)
        assert "error" in resp


class TestOtherMethods:
    def test_credits_view(self):
        req = {"jsonrpc": "2.0", "id": 1, "method": "credits.view", "params": {}}
        resp = handle_request(req)
        assert "logged_in" in resp["result"]

    def test_browser_status(self):
        req = {"jsonrpc": "2.0", "id": 1, "method": "browser.manage", "params": {"action": "status"}}
        resp = handle_request(req)
        assert "connected" in resp["result"]

    def test_browser_unknown_action(self):
        req = {"jsonrpc": "2.0", "id": 1, "method": "browser.manage", "params": {"action": "unknown"}}
        resp = handle_request(req)
        assert "error" in resp

    def test_plugins_list(self):
        req = {"jsonrpc": "2.0", "id": 1, "method": "plugins.manage", "params": {"action": "list"}}
        resp = handle_request(req)
        assert "plugins" in resp["result"]

    def test_plugins_toggle_no_name(self):
        req = {"jsonrpc": "2.0", "id": 1, "method": "plugins.manage", "params": {"action": "toggle"}}
        resp = handle_request(req)
        assert "error" in resp

    def test_tools_configure_unknown_action(self):
        req = {"jsonrpc": "2.0", "id": 1, "method": "tools.configure", "params": {"action": "unknown"}}
        resp = handle_request(req)
        assert "error" in resp

    def test_subagent_interrupt_no_id(self):
        req = {"jsonrpc": "2.0", "id": 1, "method": "subagent.interrupt", "params": {}}
        resp = handle_request(req)
        assert "error" in resp

    def test_projects_record_repos_empty(self):
        req = {"jsonrpc": "2.0", "id": 1, "method": "projects.record_repos", "params": {"repos": []}}
        resp = handle_request(req)
        assert "recorded" in resp["result"]
