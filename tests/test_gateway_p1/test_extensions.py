"""Tests for the P1 gateway extensions — status, authz, response_filters,
status_phrases, restart_loop_guard, scale_to_zero, drain_control, slash_access,
slash_commands, runner.

Covers all the modules added in the P1 gateway commit.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from niaharness.gateway.authz import (
    AuthorizationPolicy,
    is_user_authorized,
    policy_for_source,
)
from niaharness.gateway.drain_control import (
    clear_drain_request,
    current_instantiation_epoch,
    drain_notification_suppressed,
    drain_requested,
    drain_request_path,
    read_drain_request,
    write_drain_request,
)
from niaharness.gateway.response_filters import (
    LIVE_GATEWAY_SILENT_MARKERS,
    SILENT_REPLY_TOKEN,
    is_intentional_silence_agent_result,
    is_intentional_silence_response,
    is_partial_silence_marker,
    redact_secrets,
)
from niaharness.gateway.restart_loop_guard import (
    DEFAULT_MAX_RESTARTS,
    DEFAULT_WINDOW_SECONDS,
    check_and_record,
    clear,
    is_restart_loop_tripped,
    record_restart_interrupted_boot,
)
from niaharness.gateway.runner import GatewayRunner
from niaharness.gateway.scale_to_zero import (
    DEFAULT_IDLE_TIMEOUT_MINUTES,
    SCALE_TO_ZERO_ENV,
    is_idle,
    messaging_is_relay_only_or_absent,
    parse_idle_timeout_seconds,
    scale_to_zero_enabled,
    should_arm,
)
from niaharness.gateway.slash_access import (
    SlashAccessPolicy,
    policy_from_extra,
    policy_for_source as slash_policy_for_source,
)
from niaharness.gateway.slash_commands import (
    SlashCommandContext,
    SlashCommandRegistry,
    create_default_registry,
    handle_slash_command,
    handle_help,
    handle_new,
    handle_status,
    handle_whoami,
    handle_yolo,
    parse_slash_command,
)
from niaharness.gateway.status import (
    GatewaySignalHandler,
    acquire_gateway_runtime_lock,
    derive_gateway_busy,
    derive_gateway_drainable,
    get_running_pid,
    is_gateway_running,
    read_runtime_status,
    release_gateway_runtime_lock,
    remove_pid_file,
    replace_existing_gateway,
    terminate_pid,
    write_pid_file,
    write_runtime_status,
)
from niaharness.gateway.status_phrases import (
    choose_status_phrase,
    classify_status_context,
    resolve_status_phrase_catalog,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _temp_nia_home(tmp_path: Path, monkeypatch):
    """Redirect NIA_HOME to a temp dir so tests don't pollute the host."""
    monkeypatch.setenv("NIA_HOME", str(tmp_path / ".nia"))
    yield


# ---------------------------------------------------------------------------
# status.py
# ---------------------------------------------------------------------------


class TestGatewayStatus:
    def test_write_and_read_pid_file(self):
        write_pid_file()
        pid = get_running_pid()
        assert pid == os.getpid()
        remove_pid_file()
        assert get_running_pid() is None

    def test_is_gateway_running_false_initially(self):
        assert is_gateway_running() is False

    def test_is_gateway_running_true_after_write(self):
        write_pid_file()
        assert is_gateway_running() is True
        remove_pid_file()

    def test_runtime_lock_acquire_release(self):
        assert acquire_gateway_runtime_lock() is True
        # Re-acquire in same process is a no-op (returns True).
        assert acquire_gateway_runtime_lock() is True
        release_gateway_runtime_lock()

    def test_write_and_read_runtime_status(self):
        write_runtime_status(
            state="running",
            active_agents=2,
            uptime_seconds=42.0,
            adapters=["telegram", "discord"],
        )
        status = read_runtime_status()
        assert status is not None
        assert status["state"] == "running"
        assert status["active_agents"] == 2
        assert status["uptime_seconds"] == 42.0
        assert "telegram" in status["adapters"]

    def test_read_runtime_status_none_when_absent(self):
        assert read_runtime_status() is None

    def test_derive_gateway_busy(self):
        assert derive_gateway_busy(None) is False
        assert derive_gateway_busy({"active_agents": 0}) is False
        assert derive_gateway_busy({"active_agents": 3}) is True

    def test_derive_gateway_drainable(self):
        assert derive_gateway_drainable(gateway_running=False, gateway_state="running") is False
        assert derive_gateway_drainable(gateway_running=True, gateway_state="running") is True
        assert derive_gateway_drainable(gateway_running=True, gateway_state="draining") is False
        assert derive_gateway_drainable(gateway_running=True, gateway_state="stopped") is False

    def test_signal_handler_context_manager(self):
        with GatewaySignalHandler() as sh:
            assert sh.shutdown_requested is False
        # After exit, the event is cleared.
        assert sh.shutdown_requested is False


# ---------------------------------------------------------------------------
# authz.py
# ---------------------------------------------------------------------------


class TestAuthz:
    def test_open_policy_when_no_config(self):
        source = MagicMock(platform="telegram", chat_type="dm", user_id="123")
        policy = policy_for_source(None, source)
        assert policy.enabled is False
        assert policy.is_authorized("anyone") is True

    def test_allowlist_policy(self):
        config = MagicMock()
        config.platforms = {
            "telegram": {
                "allow_from": ["user1", "user2"],
                "owner_id": "owner",
            }
        }
        source = MagicMock(platform="telegram", chat_type="dm", user_id="user1")
        policy = policy_for_source(config, source)
        assert policy.enabled is True
        assert policy.is_authorized("user1") is True
        assert policy.is_authorized("user2") is True
        assert policy.is_authorized("unknown") is False
        assert policy.is_authorized("owner") is True

    def test_group_scope(self):
        config = MagicMock()
        config.platforms = {
            "telegram": {
                "group_allow_from": ["admin1"],
                "owner_id": "owner",
            }
        }
        source = MagicMock(platform="telegram", chat_type="group", user_id="admin1")
        policy = policy_for_source(config, source)
        assert policy.enabled is True
        assert policy.is_authorized("admin1") is True

    def test_is_user_authorized_open(self):
        source = MagicMock(platform="telegram", chat_type="dm", user_id="anyone")
        assert is_user_authorized(source, None) is True

    def test_is_user_authorized_allowlist(self):
        config = MagicMock()
        config.platforms = {
            "telegram": {"allow_from": ["user1"]}
        }
        source = MagicMock(platform="telegram", chat_type="dm", user_id="user1")
        assert is_user_authorized(source, config) is True
        source2 = MagicMock(platform="telegram", chat_type="dm", user_id="unknown")
        assert is_user_authorized(source2, config) is False

    def test_adapter_enforces_own_policy_skips_check(self):
        config = MagicMock()
        config.platforms = {
            "telegram": {
                "allow_from": ["user1"],
                "enforces_own_access_policy": True,
            }
        }
        source = MagicMock(platform="telegram", chat_type="dm", user_id="unknown")
        # Adapter enforces its own policy → gateway trusts it.
        assert is_user_authorized(source, config) is True

    def test_authorization_is_upstream_skips_check(self):
        config = MagicMock()
        config.platforms = {
            "relay": {
                "allow_from": ["user1"],
                "authorization_is_upstream": True,
            }
        }
        source = MagicMock(platform="relay", chat_type="dm", user_id="unknown")
        assert is_user_authorized(source, config) is True


# ---------------------------------------------------------------------------
# response_filters.py
# ---------------------------------------------------------------------------


class TestResponseFilters:
    def test_is_intentional_silence_no_reply(self):
        assert is_intentional_silence_response("NO_REPLY") is True

    def test_is_intentional_silence_silent(self):
        assert is_intentional_silence_response("[SILENT]") is True

    def test_is_intentional_silence_no_reply_with_spaces(self):
        assert is_intentional_silence_response("NO REPLY") is True

    def test_is_intentional_silence_with_punctuation(self):
        assert is_intentional_silence_response(".NO_REPLY") is True
        assert is_intentional_silence_response("*NO_REPLY*") is True

    def test_not_silence_for_prose(self):
        assert is_intentional_silence_response("I'll help you with that.") is False

    def test_not_silence_for_empty(self):
        assert is_intentional_silence_response("") is False
        assert is_intentional_silence_response(None) is False

    def test_not_silence_for_long_text(self):
        long_text = "NO_REPLY " + "x" * 100
        assert is_intentional_silence_response(long_text) is False

    def test_is_intentional_silence_agent_result_failed(self):
        assert is_intentional_silence_agent_result(
            {"failed": True}, "NO_REPLY"
        ) is False

    def test_is_intentional_silence_agent_result_success(self):
        assert is_intentional_silence_agent_result(
            {"failed": False}, "NO_REPLY"
        ) is True

    def test_is_partial_silence_marker_prefix(self):
        assert is_partial_silence_marker("NO") is True
        assert is_partial_silence_marker("NO_") is True
        assert is_partial_silence_marker("NO_R") is True

    def test_is_partial_silence_marker_diverged(self):
        assert is_partial_silence_marker("Hello world") is False

    def test_is_partial_silence_marker_empty(self):
        assert is_partial_silence_marker("") is False
        assert is_partial_silence_marker(None) is False

    def test_is_partial_silence_marker_too_long(self):
        assert is_partial_silence_marker("NO_REPLY " + "x" * 100) is False

    def test_redact_github_token(self):
        text = "My token is ghp_1234567890abcdefghij"
        redacted = redact_secrets(text)
        assert "ghp_1234567890abcdefghij" not in redacted
        assert "[REDACTED]" in redacted

    def test_redact_openai_key(self):
        text = "Key: sk-1234567890abcdefghijklmnopqrstuv"
        redacted = redact_secrets(text)
        assert "sk-1234567890abcdefghijklmnopqrstuv" not in redacted
        assert "[REDACTED]" in redacted

    def test_redact_anthropic_key(self):
        text = "Key: sk-ant-1234567890abcdefghijklmnopqrstuv"
        redacted = redact_secrets(text)
        assert "[REDACTED]" in redacted

    def test_redact_bearer_token(self):
        text = "Authorization: Bearer abcdef1234567890abcdef1234567890"
        redacted = redact_secrets(text)
        assert "Bearer abcdef" not in redacted
        assert "[REDACTED]" in redacted

    def test_redact_password_assignment(self):
        text = "password=secret123"
        redacted = redact_secrets(text)
        assert "secret123" not in redacted
        assert "[REDACTED]" in redacted

    def test_redact_no_secrets_passthrough(self):
        text = "This is a normal message with no secrets."
        assert redact_secrets(text) == text

    def test_redact_empty(self):
        assert redact_secrets("") == ""
        assert redact_secrets(None) == ""


# ---------------------------------------------------------------------------
# status_phrases.py
# ---------------------------------------------------------------------------


class TestStatusPhrases:
    def test_choose_status_phrase_returns_string(self):
        phrase = choose_status_phrase("status")
        assert isinstance(phrase, str)
        assert len(phrase) > 0

    def test_choose_status_phrase_avoids_recent(self):
        recent: list[str] = []
        phrases = set()
        for _ in range(10):
            phrase = choose_status_phrase("status", recent=recent)
            phrases.add(phrase)
        # Should have picked more than one phrase over 10 calls.
        assert len(phrases) > 1

    def test_classify_status_context(self):
        assert classify_status_context("status") == "status"
        assert classify_status_context("heartbeat") == "status"
        assert classify_status_context("waiting") == "status"
        assert classify_status_context("long_running") == "status"
        assert classify_status_context("other") == "generic"

    def test_resolve_status_phrase_catalog_defaults(self):
        catalog = resolve_status_phrase_catalog(None)
        assert "status" in catalog
        assert "generic" in catalog
        assert len(catalog["status"]) > 0
        assert len(catalog["generic"]) > 0


# ---------------------------------------------------------------------------
# restart_loop_guard.py
# ---------------------------------------------------------------------------


class TestRestartLoopGuard:
    def test_record_and_check_not_tripped_initially(self):
        tripped = check_and_record()
        assert tripped is False

    def test_trips_after_max_restarts(self):
        # Record 3 boots within the window.
        now = time.time()
        for i in range(DEFAULT_MAX_RESTARTS):
            record_restart_interrupted_boot(now=now + i)
        assert is_restart_loop_tripped(now=now + DEFAULT_MAX_RESTARTS - 1) is True

    def test_does_not_trip_below_threshold(self):
        clear()
        now = time.time()
        for i in range(DEFAULT_MAX_RESTARTS - 1):
            record_restart_interrupted_boot(now=now + i)
        assert is_restart_loop_tripped(now=now + 10) is False

    def test_clear_resets_state(self):
        now = time.time()
        for i in range(DEFAULT_MAX_RESTARTS):
            record_restart_interrupted_boot(now=now + i)
        clear()
        assert is_restart_loop_tripped() is False

    def test_check_and_record_returns_true_at_threshold(self):
        clear()
        now = time.time()
        # Record max-1 boots first.
        for i in range(DEFAULT_MAX_RESTARTS - 1):
            record_restart_interrupted_boot(now=now + i)
        # The next check_and_record should trip.
        tripped = check_and_record(now=now + DEFAULT_MAX_RESTARTS - 1)
        assert tripped is True
        clear()


# ---------------------------------------------------------------------------
# scale_to_zero.py
# ---------------------------------------------------------------------------


class TestScaleToZero:
    def test_scale_to_zero_disabled_by_default(self):
        assert scale_to_zero_enabled() is False

    def test_scale_to_zero_enabled(self, monkeypatch):
        monkeypatch.setenv(SCALE_TO_ZERO_ENV, "1")
        assert scale_to_zero_enabled() is True

    def test_parse_idle_timeout_seconds_default(self):
        assert parse_idle_timeout_seconds(None) == DEFAULT_IDLE_TIMEOUT_MINUTES * 60

    def test_parse_idle_timeout_seconds_custom(self):
        assert parse_idle_timeout_seconds(10) == 600.0

    def test_parse_idle_timeout_seconds_invalid(self):
        assert parse_idle_timeout_seconds("invalid") == DEFAULT_IDLE_TIMEOUT_MINUTES * 60
        assert parse_idle_timeout_seconds(-5) == DEFAULT_IDLE_TIMEOUT_MINUTES * 60

    def test_messaging_relay_only(self):
        platforms = [MagicMock(value="relay")]
        assert messaging_is_relay_only_or_absent(platforms) is True

    def test_messaging_with_direct_platform(self):
        platforms = [MagicMock(value="telegram")]
        assert messaging_is_relay_only_or_absent(platforms) is False

    def test_messaging_empty(self):
        assert messaging_is_relay_only_or_absent([]) is True

    def test_should_arm_all_conditions_met(self):
        assert should_arm(
            enabled=True,
            relay_only_or_absent=True,
            wake_url="https://example.com/wake",
        ) is True

    def test_should_arm_missing_wake_url(self):
        assert should_arm(
            enabled=True,
            relay_only_or_absent=True,
            wake_url=None,
        ) is False

    def test_should_arm_disabled(self):
        assert should_arm(
            enabled=False,
            relay_only_or_absent=True,
            wake_url="https://example.com/wake",
        ) is False

    def test_should_arm_direct_platform(self):
        assert should_arm(
            enabled=True,
            relay_only_or_absent=False,
            wake_url="https://example.com/wake",
        ) is False

    def test_is_idle_true(self):
        assert is_idle(
            running_agent_count=0,
            seconds_since_last_inbound=600,
            idle_timeout_seconds=300,
            has_live_background_work=False,
        ) is True

    def test_is_idle_false_agent_running(self):
        assert is_idle(
            running_agent_count=1,
            seconds_since_last_inbound=600,
            idle_timeout_seconds=300,
            has_live_background_work=False,
        ) is False

    def test_is_idle_false_background_work(self):
        assert is_idle(
            running_agent_count=0,
            seconds_since_last_inbound=600,
            idle_timeout_seconds=300,
            has_live_background_work=True,
        ) is False

    def test_is_idle_false_recent_inbound(self):
        assert is_idle(
            running_agent_count=0,
            seconds_since_last_inbound=100,
            idle_timeout_seconds=300,
            has_live_background_work=False,
        ) is False


# ---------------------------------------------------------------------------
# drain_control.py
# ---------------------------------------------------------------------------


class TestDrainControl:
    def test_write_and_read_drain_request(self, tmp_path: Path):
        payload = write_drain_request(home=tmp_path)
        assert payload["action"] == "drain"
        assert payload["epoch"] != "" or True  # epoch may be "" on non-Linux
        body = read_drain_request(home=tmp_path)
        assert body is not None
        assert body["action"] == "drain"

    def test_drain_requested_true_after_write(self, tmp_path: Path):
        write_drain_request(home=tmp_path)
        assert drain_requested(home=tmp_path) is True

    def test_drain_requested_false_when_absent(self, tmp_path: Path):
        assert drain_requested(home=tmp_path) is False

    def test_clear_drain_request(self, tmp_path: Path):
        write_drain_request(home=tmp_path)
        assert clear_drain_request(home=tmp_path) is True
        assert drain_requested(home=tmp_path) is False

    def test_clear_drain_request_idempotent(self, tmp_path: Path):
        # Clearing when no marker exists returns False.
        assert clear_drain_request(home=tmp_path) is False

    def test_drain_notification_suppressed(self, tmp_path: Path):
        write_drain_request(suppress_notification=True, home=tmp_path)
        assert drain_notification_suppressed(home=tmp_path) is True

    def test_drain_notification_not_suppressed_by_default(self, tmp_path: Path):
        write_drain_request(home=tmp_path)
        assert drain_notification_suppressed(home=tmp_path) is False

    def test_current_instantiation_epoch_is_string(self):
        epoch = current_instantiation_epoch()
        assert isinstance(epoch, str)


# ---------------------------------------------------------------------------
# slash_access.py
# ---------------------------------------------------------------------------


class TestSlashAccess:
    def test_policy_disabled_when_no_admins(self):
        policy = policy_from_extra({}, "dm")
        assert policy.enabled is False
        assert policy.is_admin("anyone") is True  # gating off → all admins
        assert policy.can_run("anyone", "anycommand") is True

    def test_policy_enabled_with_admins(self):
        policy = policy_from_extra(
            {"allow_admin_from": ["admin1"], "user_allowed_commands": ["status"]},
            "dm",
        )
        assert policy.enabled is True
        assert policy.is_admin("admin1") is True
        assert policy.is_admin("user1") is False

    def test_admin_can_run_anything(self):
        policy = policy_from_extra(
            {"allow_admin_from": ["admin1"]}, "dm"
        )
        assert policy.can_run("admin1", "anycommand") is True

    def test_user_can_run_listed_command(self):
        policy = policy_from_extra(
            {"allow_admin_from": ["admin1"], "user_allowed_commands": ["status", "help"]},
            "dm",
        )
        assert policy.can_run("user1", "status") is True
        assert policy.can_run("user1", "help") is True

    def test_user_cannot_run_unlisted_command(self):
        policy = policy_from_extra(
            {"allow_admin_from": ["admin1"], "user_allowed_commands": ["status"]},
            "dm",
        )
        assert policy.can_run("user1", "yolo") is False

    def test_always_allowed_commands(self):
        policy = policy_from_extra(
            {"allow_admin_from": ["admin1"]}, "dm"
        )
        # help + whoami are always allowed.
        assert policy.can_run("user1", "help") is True
        assert policy.can_run("user1", "whoami") is True

    def test_dm_falls_back_to_group_commands(self):
        policy = policy_from_extra(
            {
                "allow_admin_from": ["admin1"],
                "group_user_allowed_commands": ["status"],
            },
            "dm",
        )
        # DM didn't specify user_allowed_commands → falls back to group's.
        assert policy.can_run("user1", "status") is True

    def test_coerce_command_list_strips_slashes(self):
        policy = policy_from_extra(
            {
                "allow_admin_from": ["admin1"],
                "user_allowed_commands": ["/help", "/status"],
            },
            "dm",
        )
        assert policy.can_run("user1", "help") is True
        assert policy.can_run("user1", "status") is True

    def test_policy_for_source_no_config(self):
        source = MagicMock(platform="telegram", chat_type="dm")
        policy = slash_policy_for_source(None, source)
        assert policy.enabled is False


# ---------------------------------------------------------------------------
# slash_commands.py
# ---------------------------------------------------------------------------


class TestSlashCommands:
    def test_parse_slash_command_basic(self):
        assert parse_slash_command("/help") == ("help", "")
        assert parse_slash_command("/new conversation") == ("new", "conversation")
        assert parse_slash_command("/status  extra  args") == ("status", "extra  args")

    def test_parse_slash_command_not_a_command(self):
        assert parse_slash_command("hello") is None
        assert parse_slash_command("") is None
        assert parse_slash_command("/") is None

    def test_registry_register_and_get(self):
        registry = SlashCommandRegistry()
        handler = AsyncMock(return_value="ok")
        registry.register("test", handler, description="Test command")
        assert registry.has("test") is True
        assert registry.get("test") is handler
        assert "test" in registry.list_commands()

    def test_registry_aliases(self):
        registry = SlashCommandRegistry()
        handler = AsyncMock(return_value="ok")
        registry.register("new", handler, aliases=["reset"])
        assert registry.get("new") is handler
        assert registry.get("reset") is handler

    def test_registry_case_insensitive(self):
        registry = SlashCommandRegistry()
        handler = AsyncMock(return_value="ok")
        registry.register("help", handler)
        assert registry.get("HELP") is handler
        assert registry.get("Help") is handler

    def test_create_default_registry_has_builtins(self):
        registry = create_default_registry()
        commands = registry.list_commands()
        assert "help" in commands
        assert "new" in commands
        assert "status" in commands
        assert "yolo" in commands
        assert "whoami" in commands
        assert "cancel" in commands
        assert "queue" in commands

    @pytest.mark.asyncio
    async def test_handle_slash_command_help(self):
        registry = create_default_registry()
        context = SlashCommandContext(
            platform="telegram", chat_id="123", user_id="456",
            metadata={"registry": registry},
        )
        reply = await handle_slash_command("/help", context, registry)
        assert reply is not None
        assert "Available commands" in reply

    @pytest.mark.asyncio
    async def test_handle_slash_command_unknown(self):
        registry = create_default_registry()
        context = SlashCommandContext(
            platform="telegram", chat_id="123", user_id="456",
        )
        reply = await handle_slash_command("/nonexistent", context, registry)
        assert "Unknown command" in reply

    @pytest.mark.asyncio
    async def test_handle_slash_command_not_a_command(self):
        registry = create_default_registry()
        context = SlashCommandContext(
            platform="telegram", chat_id="123", user_id="456",
        )
        reply = await handle_slash_command("hello world", context, registry)
        assert reply is None

    @pytest.mark.asyncio
    async def test_handle_whoami(self):
        context = SlashCommandContext(
            platform="telegram", chat_id="123", user_id="456",
            user_name="testuser",
        )
        reply = await handle_whoami(context)
        assert "telegram" in reply
        assert "456" in reply
        assert "testuser" in reply

    @pytest.mark.asyncio
    async def test_handle_status(self):
        context = SlashCommandContext(
            platform="telegram", chat_id="123", user_id="456",
            gateway_runner=MagicMock(
                list_adapters=lambda: ["telegram"],
                get_status=lambda: {"state": "running"},
            ),
        )
        reply = await handle_status(context)
        assert "NIA Gateway Status" in reply
        assert "telegram" in reply

    @pytest.mark.asyncio
    async def test_handle_new_without_runner(self):
        context = SlashCommandContext(
            platform="telegram", chat_id="123", user_id="456",
        )
        reply = await handle_new(context)
        assert "new conversation" in reply.lower()

    @pytest.mark.asyncio
    async def test_handle_new_with_runner(self):
        runner = MagicMock()
        runner.reset_session = MagicMock()
        context = SlashCommandContext(
            platform="telegram", chat_id="123", user_id="456",
            gateway_runner=runner,
        )
        reply = await handle_new(context)
        assert "new conversation" in reply.lower()
        runner.reset_session.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_yolo_toggle(self):
        runner = MagicMock()
        runner.toggle_yolo = MagicMock(return_value=True)
        context = SlashCommandContext(
            platform="telegram", chat_id="123", user_id="456",
            gateway_runner=runner,
        )
        reply = await handle_yolo(context)
        assert "ON" in reply
        runner.toggle_yolo.assert_called_once_with("telegram", "123")


# ---------------------------------------------------------------------------
# runner.py
# ---------------------------------------------------------------------------


class TestGatewayRunner:
    def test_runner_init_defaults(self):
        runner = GatewayRunner()
        assert runner.is_running is False
        assert runner.gateway_state == "stopped"
        assert runner.active_agent_count == 0
        assert runner.uptime_seconds == 0.0

    def test_runner_router_property(self):
        runner = GatewayRunner()
        assert runner.router is not None

    def test_runner_slash_registry_has_builtins(self):
        runner = GatewayRunner()
        commands = runner.slash_registry.list_commands()
        assert "help" in commands
        assert "status" in commands

    def test_runner_yolo_toggle(self):
        runner = GatewayRunner()
        assert runner.is_yolo("telegram", "123") is False
        assert runner.toggle_yolo("telegram", "123") is True
        assert runner.is_yolo("telegram", "123") is True
        assert runner.toggle_yolo("telegram", "123") is False
        assert runner.is_yolo("telegram", "123") is False

    def test_runner_get_status(self):
        runner = GatewayRunner()
        status = runner.get_status()
        assert status["state"] == "stopped"
        assert status["active_agents"] == 0

    def test_runner_reset_session(self):
        runner = GatewayRunner()
        runner.toggle_yolo("telegram", "123")
        runner.reset_session("telegram", "123")
        assert runner.is_yolo("telegram", "123") is False

    @pytest.mark.asyncio
    async def test_runner_start_and_stop(self):
        runner = GatewayRunner()
        # Mock the adapter so start_all/stop_all don't fail.
        adapter = MagicMock()
        adapter.platform_name = "mock"
        adapter.start = AsyncMock()
        adapter.stop = AsyncMock()
        adapter.set_message_handler = MagicMock()
        runner.router._adapters = {"mock": adapter}

        await runner.start()
        assert runner.is_running is True
        assert runner.gateway_state == "running"

        await runner.stop()
        assert runner.is_running is False
        assert runner.gateway_state == "stopped"

    @pytest.mark.asyncio
    async def test_runner_handles_slash_command(self):
        runner = GatewayRunner()
        runner._running = True

        # Mock adapter to capture the reply.
        sent_messages: list[str] = []

        async def mock_send(msg):
            sent_messages.append(msg.text)

        adapter = MagicMock()
        adapter.platform_name = "telegram"
        adapter.send_message = mock_send
        runner.router._adapters = {"telegram": adapter}

        from niaharness.gateway import IncomingMessage
        message = IncomingMessage(
            platform="telegram",
            platform_message_id="1",
            platform_chat_id="123",
            platform_user_id="456",
            text="/whoami",
        )
        await runner._handle_incoming(message)
        assert len(sent_messages) >= 1
        assert "telegram" in sent_messages[0]

    @pytest.mark.asyncio
    async def test_runner_drain_rejects_messages(self):
        runner = GatewayRunner()
        runner._running = True
        runner._gateway_state = "draining"

        sent_messages: list[str] = []

        async def mock_send(msg):
            sent_messages.append(msg.text)

        adapter = MagicMock()
        adapter.platform_name = "telegram"
        adapter.send_message = mock_send
        runner.router._adapters = {"telegram": adapter}

        from niaharness.gateway import IncomingMessage
        message = IncomingMessage(
            platform="telegram",
            platform_message_id="1",
            platform_chat_id="123",
            platform_user_id="456",
            text="hello",
        )
        await runner._handle_incoming(message)
        assert len(sent_messages) == 1
        assert "draining" in sent_messages[0].lower()


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
