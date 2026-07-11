"""Tests for the P1 cron extensions — lifecycle guard, ReadWriteLock, pools,
credential guard, wake gate, origin tracking, toolset resolver, continuable
threads, blueprint catalog, suggestions engine.

Covers all the modules added in the P1 cron commit.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from niaharness.cron.blueprint_catalog import (
    AutomationBlueprint,
    BlueprintFillError,
    BlueprintSlot,
    WEEKDAY_PRESETS,
    blueprint_catalog_entry,
    blueprint_deeplink,
    blueprint_form_schema,
    blueprint_slash_command,
    fill_blueprint,
    get_blueprint,
    list_blueprints,
)
from niaharness.cron.continuable_threads import (
    deliver_to_thread_or_mirror,
    open_continuable_cron_thread,
    seed_cron_thread_session,
)
from niaharness.cron.credential_guard import (
    guard_job_credential_exfil,
    validate_cron_base_url,
)
from niaharness.cron.lifecycle_guard import (
    GatewayLifecycleBlocked,
    check_gateway_lifecycle,
    contains_gateway_lifecycle_command,
)
from niaharness.cron.origin import (
    cron_job_origin_log_suffix,
    cron_mirror_delivery_enabled,
    maybe_mirror_cron_delivery,
    resolve_origin,
    target_matches_origin,
)
from niaharness.cron.pools import (
    DEFAULT_PARALLEL_WORKERS,
    get_parallel_pool,
    get_sequential_pool,
    interpreter_shutting_down,
    shutdown_pools,
    submit_parallel,
    submit_sequential,
)
from niaharness.cron.readwrite_lock import ReadWriteLock, terminal_cwd_lock
from niaharness.cron.suggestions import (
    accept_suggestion,
    add_suggestion,
    clear_all,
    clear_resolved,
    dismiss_suggestion,
    get_suggestion,
    list_accepted,
    list_dismissed,
    list_pending,
    load_suggestions,
)
from niaharness.cron.toolset_resolver import (
    merge_mcp_into_per_job_toolsets,
    resolve_cron_disabled_toolsets,
    resolve_cron_enabled_toolsets,
    resolve_cron_toolsets,
)
from niaharness.cron.wake_gate import build_wake_gate_output, parse_wake_gate


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _temp_nia_home(tmp_path: Path, monkeypatch):
    """Redirect NIA_HOME to a temp dir so tests don't pollute the host."""
    monkeypatch.setenv("NIA_HOME", str(tmp_path / ".nia"))
    yield


# ---------------------------------------------------------------------------
# lifecycle_guard.py
# ---------------------------------------------------------------------------


class TestLifecycleGuard:
    def test_blocks_nia_gateway_restart(self):
        assert contains_gateway_lifecycle_command("nia gateway restart") is True

    def test_blocks_nia_gateway_stop(self):
        assert contains_gateway_lifecycle_command("nia gateway stop") is True

    def test_does_not_block_nia_gateway_start(self):
        # `start` is intentionally excluded.
        assert contains_gateway_lifecycle_command("nia gateway start") is False

    def test_blocks_launchctl(self):
        assert contains_gateway_lifecycle_command(
            "launchctl kickstart ai.nia.gateway"
        ) is True

    def test_blocks_systemctl(self):
        assert contains_gateway_lifecycle_command(
            "systemctl restart nia-gateway"
        ) is True

    def test_blocks_pkill(self):
        assert contains_gateway_lifecycle_command(
            "pkill -f nia gateway"
        ) is True

    def test_does_not_block_prose(self):
        assert contains_gateway_lifecycle_command(
            "The Kong API gateway autoscaling and restart behavior"
        ) is False

    def test_does_not_block_empty(self):
        assert contains_gateway_lifecycle_command("") is False
        assert contains_gateway_lifecycle_command(None) is False

    def test_check_gateway_lifecycle_raises(self):
        with pytest.raises(GatewayLifecycleBlocked):
            check_gateway_lifecycle(prompt="nia gateway restart")

    def test_check_gateway_lifecycle_passes_safe(self):
        # Should not raise.
        check_gateway_lifecycle(prompt="ls -la")

    def test_check_gateway_lifecycle_with_script(self, tmp_path: Path):
        # Write a script with a gateway-lifecycle command.
        script = tmp_path / "evil.sh"
        script.write_text("#!/bin/sh\nnia gateway restart\n")
        with pytest.raises(GatewayLifecycleBlocked):
            check_gateway_lifecycle(
                prompt="innocent", script=str(script),
            )


# ---------------------------------------------------------------------------
# readwrite_lock.py
# ---------------------------------------------------------------------------


class TestReadWriteLock:
    def test_multiple_readers(self):
        lock = ReadWriteLock()
        lock.acquire_read()
        lock.acquire_read()
        assert lock.reader_count == 2
        lock.release_read()
        lock.release_read()
        assert lock.reader_count == 0

    def test_writer_excludes_readers(self):
        lock = ReadWriteLock()
        lock.acquire_write()
        assert lock.writer_active is True
        # Reader should block — test in a thread.
        result = []

        def try_read():
            lock.acquire_read()
            result.append("read")
            lock.release_read()

        t = threading.Thread(target=try_read)
        t.start()
        time.sleep(0.1)
        assert result == []  # Reader is blocked.
        lock.release_write()
        t.join(timeout=2)
        assert result == ["read"]

    def test_writer_preference(self):
        lock = ReadWriteLock()
        # Start a reader.
        lock.acquire_read()
        # Writer starts waiting.
        writer_done = []

        def try_write():
            lock.acquire_write()
            writer_done.append("written")
            lock.release_write()

        t = threading.Thread(target=try_write)
        t.start()
        time.sleep(0.1)
        assert lock.writers_waiting == 1
        # New reader should block (writer preference).
        reader2_done = []

        def try_read2():
            lock.acquire_read()
            reader2_done.append("read2")
            lock.release_read()

        t2 = threading.Thread(target=try_read2)
        t2.start()
        time.sleep(0.1)
        assert reader2_done == []  # Second reader blocked by writer preference.
        # Release first reader.
        lock.release_read()
        # Writer should proceed.
        t.join(timeout=2)
        assert writer_done == ["written"]
        # Then reader2.
        t2.join(timeout=2)
        assert reader2_done == ["read2"]

    def test_terminal_cwd_lock_is_global(self):
        assert terminal_cwd_lock is not None
        assert isinstance(terminal_cwd_lock, ReadWriteLock)


# ---------------------------------------------------------------------------
# pools.py
# ---------------------------------------------------------------------------


class TestCronPools:
    def test_get_parallel_pool(self):
        pool = get_parallel_pool()
        assert pool is not None

    def test_get_sequential_pool(self):
        pool = get_sequential_pool()
        assert pool is not None

    def test_submit_parallel(self):
        result = submit_parallel(lambda x: x * 2, 21)
        assert result.result(timeout=5) == 42

    def test_submit_sequential(self):
        result = submit_sequential(lambda x: x + 1, 41)
        assert result.result(timeout=5) == 42

    def test_parallel_pool_max_workers(self):
        pool = get_parallel_pool(max_workers=8)
        assert pool._max_workers == 8

    def test_interpreter_shutting_down(self):
        assert interpreter_shutting_down(None) is False
        assert interpreter_shutting_down(RuntimeError("cannot schedule new futures")) is True
        assert interpreter_shutting_down(RuntimeError("other error")) is False

    def test_shutdown_pools(self):
        shutdown_pools()
        # Pools should be None after shutdown.
        # Re-create for other tests.
        get_parallel_pool()
        get_sequential_pool()


# ---------------------------------------------------------------------------
# credential_guard.py
# ---------------------------------------------------------------------------


class TestCredentialGuard:
    def test_safe_no_base_url(self):
        assert validate_cron_base_url("anthropic", None) is None
        assert validate_cron_base_url("anthropic", "") is None

    def test_safe_official_endpoint(self):
        assert validate_cron_base_url(
            "anthropic", "https://api.anthropic.com"
        ) is None

    def test_safe_localhost(self):
        assert validate_cron_base_url(
            "anthropic", "http://localhost:8080"
        ) is None
        assert validate_cron_base_url(
            "anthropic", "http://127.0.0.1:8080"
        ) is None

    def test_unsafe_different_host(self):
        result = validate_cron_base_url("anthropic", "https://evil.com")
        assert result is not None
        assert "evil.com" in result
        assert "api.anthropic.com" in result

    def test_unknown_provider_no_validation(self):
        assert validate_cron_base_url(
            "unknown_provider", "https://evil.com"
        ) is None

    def test_openai_compatible_allows_custom(self):
        assert validate_cron_base_url(
            "openai-compatible", "https://custom.example.com"
        ) is None

    def test_guard_job_credential_exfil_safe(self):
        # Should not raise.
        guard_job_credential_exfil({
            "id": "test",
            "provider": "anthropic",
            "base_url": "https://api.anthropic.com",
        })

    def test_guard_job_credential_exfil_unsafe(self):
        with pytest.raises(RuntimeError, match="blocked for safety"):
            guard_job_credential_exfil({
                "id": "test",
                "provider": "anthropic",
                "base_url": "https://evil.com",
            })

    def test_guard_job_credential_exfil_no_base_url(self):
        # Should not raise — no base_url override means no exfil vector.
        guard_job_credential_exfil({
            "id": "test",
            "provider": "anthropic",
        })


# ---------------------------------------------------------------------------
# wake_gate.py
# ---------------------------------------------------------------------------


class TestWakeGate:
    def test_empty_output_wakes(self):
        assert parse_wake_gate("") is True

    def test_no_json_wakes(self):
        assert parse_wake_gate("some output") is True

    def test_wake_agent_false_skips(self):
        assert parse_wake_gate('{"wakeAgent": false}') is False

    def test_wake_agent_true_wakes(self):
        assert parse_wake_gate('{"wakeAgent": true}') is True

    def test_missing_flag_wakes(self):
        assert parse_wake_gate('{"other": "value"}') is True

    def test_last_line_only(self):
        assert parse_wake_gate('line1\nline2\n{"wakeAgent": false}') is False

    def test_non_dict_json_wakes(self):
        assert parse_wake_gate('["wakeAgent", false]') is True

    def test_build_wake_gate_output(self):
        output = build_wake_gate_output(wake_agent=False)
        assert json.loads(output) == {"wakeAgent": False}
        output2 = build_wake_gate_output(wake_agent=True, extra={"reason": "new items"})
        data = json.loads(output2)
        assert data["wakeAgent"] is True
        assert data["reason"] == "new items"


# ---------------------------------------------------------------------------
# origin.py
# ---------------------------------------------------------------------------


class TestOriginTracking:
    def test_resolve_origin_valid(self):
        job = {"origin": {"platform": "telegram", "chat_id": "123"}}
        origin = resolve_origin(job)
        assert origin is not None
        assert origin["platform"] == "telegram"

    def test_resolve_origin_missing(self):
        assert resolve_origin({}) is None

    def test_resolve_origin_non_dict(self):
        assert resolve_origin({"origin": "some string"}) is None

    def test_resolve_origin_missing_fields(self):
        assert resolve_origin({"origin": {"platform": "telegram"}}) is None

    def test_target_matches_origin(self):
        origin = {"platform": "telegram", "chat_id": "123"}
        assert target_matches_origin(origin, "telegram", "123") is True

    def test_target_does_not_match_different_platform(self):
        origin = {"platform": "telegram", "chat_id": "123"}
        assert target_matches_origin(origin, "discord", "123") is False

    def test_target_does_not_match_different_chat(self):
        origin = {"platform": "telegram", "chat_id": "123"}
        assert target_matches_origin(origin, "telegram", "456") is False

    def test_target_matches_with_thread(self):
        origin = {"platform": "telegram", "chat_id": "123", "thread_id": "t1"}
        assert target_matches_origin(origin, "telegram", "123", "t1") is True
        assert target_matches_origin(origin, "telegram", "123", "t2") is False

    def test_cron_mirror_delivery_disabled_by_default(self):
        assert cron_mirror_delivery_enabled({}) is False

    def test_cron_mirror_delivery_per_job(self):
        assert cron_mirror_delivery_enabled({"attach_to_session": True}) is True
        assert cron_mirror_delivery_enabled({"attach_to_session": False}) is False

    def test_maybe_mirror_disabled(self):
        result = maybe_mirror_cron_delivery(
            {"origin": {"platform": "tg", "chat_id": "123"}},
            "tg", "123", "text",
            enabled=False,
        )
        assert result is False

    def test_maybe_mirror_target_not_origin(self):
        result = maybe_mirror_cron_delivery(
            {"origin": {"platform": "telegram", "chat_id": "123"}},
            "discord", "456", "text",
            enabled=True,
        )
        assert result is False

    def test_cron_job_origin_log_suffix(self):
        job = {"origin": {"platform": "telegram", "chat_id": "123"}}
        suffix = cron_job_origin_log_suffix(job)
        assert "telegram" in suffix
        assert "123" in suffix

    def test_cron_job_origin_log_suffix_no_origin(self):
        assert cron_job_origin_log_suffix({}) == ""


# ---------------------------------------------------------------------------
# toolset_resolver.py
# ---------------------------------------------------------------------------


class TestToolsetResolver:
    def test_default_disabled_toolsets(self):
        disabled = resolve_cron_disabled_toolsets({})
        assert "cronjob" in disabled
        assert "messaging" in disabled
        assert "clarify" in disabled

    def test_user_disabled_toolsets(self):
        disabled = resolve_cron_disabled_toolsets({
            "agent": {"disabled_toolsets": ["browser", "bash"]}
        })
        assert "browser" in disabled
        assert "bash" in disabled
        assert "cronjob" in disabled

    def test_merge_mcp_no_mcp(self):
        result = merge_mcp_into_per_job_toolsets(["bash", "file_read"], {})
        assert "bash" in result
        assert "file_read" in result

    def test_merge_mcp_with_no_mcp_sentinel(self):
        result = merge_mcp_into_per_job_toolsets(["bash", "no_mcp"], {})
        assert "no_mcp" not in result
        assert "bash" in result

    def test_merge_mcp_adds_enabled_servers(self):
        cfg = {
            "mcp": {
                "servers": {
                    "github": {"enabled": True},
                    "slack": {"enabled": True},
                    "disabled_server": {"enabled": False},
                }
            }
        }
        result = merge_mcp_into_per_job_toolsets(["bash"], cfg)
        assert "bash" in result
        assert "github" in result
        assert "slack" in result
        assert "disabled_server" not in result

    def test_merge_mcp_preserves_existing(self):
        cfg = {"mcp": {"servers": {"github": {"enabled": True}}}}
        result = merge_mcp_into_per_job_toolsets(["bash", "github"], cfg)
        # github already listed → don't add more.
        assert "github" in result
        assert len([t for t in result if t == "github"]) == 1

    def test_resolve_cron_enabled_toolsets_none(self):
        assert resolve_cron_enabled_toolsets({}, {}) is None

    def test_resolve_cron_enabled_toolsets_per_job(self):
        result = resolve_cron_enabled_toolsets(
            {"enabled_toolsets": ["bash", "file_read"]}, {}
        )
        assert "bash" in result
        assert "file_read" in result

    def test_resolve_cron_toolsets(self):
        enabled, disabled = resolve_cron_toolsets({}, {})
        assert enabled is None
        assert "cronjob" in disabled


# ---------------------------------------------------------------------------
# continuable_threads.py
# ---------------------------------------------------------------------------


class TestContinuableThreads:
    @pytest.mark.asyncio
    async def test_open_thread_no_adapter_primitive(self):
        adapter = MagicMock()
        # No create_handoff_thread method.
        del adapter.create_handoff_thread
        result = await open_continuable_cron_thread(
            {"id": "test"}, adapter, "123", asyncio.get_event_loop()
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_open_thread_no_loop(self):
        adapter = MagicMock()
        adapter.create_handoff_thread = MagicMock()
        result = await open_continuable_cron_thread(
            {"id": "test"}, adapter, "123", None
        )
        assert result is None

    def test_seed_cron_thread_session_empty_text(self):
        adapter = MagicMock()
        result = seed_cron_thread_session(
            {"id": "test"}, adapter, "telegram", "123", "t1", ""
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_deliver_to_thread_or_mirror_fallback(self):
        adapter = MagicMock()
        adapter.platform_name = "whatsapp"
        # No thread primitive → fall back to mirror.
        result = await deliver_to_thread_or_mirror(
            {"id": "test", "origin": {"platform": "whatsapp", "chat_id": "123"}},
            adapter, "whatsapp", "123", "hello",
            None,  # no loop
            mirror_enabled=False,
        )
        assert result is False  # mirror not enabled → delivery failed.


# ---------------------------------------------------------------------------
# blueprint_catalog.py
# ---------------------------------------------------------------------------


class TestBlueprintCatalog:
    def test_list_blueprints(self):
        bps = list_blueprints()
        assert len(bps) >= 10  # we have 10 built-in blueprints
        keys = {bp.key for bp in bps}
        assert "morning-brief" in keys
        assert "weekly-review" in keys

    def test_list_blueprints_by_category(self):
        daily = list_blueprints(category="daily")
        assert all(bp.category == "daily" for bp in daily)
        assert len(daily) >= 3

    def test_list_blueprints_by_tag(self):
        reminder = list_blueprints(tag="reminder")
        assert all("reminder" in bp.tags for bp in reminder)

    def test_get_blueprint(self):
        bp = get_blueprint("morning-brief")
        assert bp is not None
        assert bp.title == "Morning briefing"

    def test_get_blueprint_not_found(self):
        assert get_blueprint("nonexistent") is None

    def test_fill_blueprint_morning_brief(self):
        bp = get_blueprint("morning-brief")
        spec = fill_blueprint(bp, {"time": "07:30", "deliver": "telegram"})
        assert spec["schedule"] == "30 7 * * *"
        assert spec["deliver"] == "telegram"
        assert "morning briefing" in spec["prompt"].lower()

    def test_fill_blueprint_missing_required(self):
        """A slot with no default and no value → BlueprintFillError."""
        bp = AutomationBlueprint(
            key="test-required",
            title="Test",
            description="Test",
            category="test",
            schedule_template="{minute} {hour} * * *",
            prompt_template="Do {what}",
            slots=[
                BlueprintSlot(name="time", type="time", label="Time?", default=""),
                BlueprintSlot(name="what", type="text", label="What?", default=""),
            ],
        )
        with pytest.raises(BlueprintFillError, match="missing required"):
            fill_blueprint(bp, {})  # no values + empty defaults → error

    def test_fill_blueprint_unknown_slot(self):
        bp = get_blueprint("morning-brief")
        with pytest.raises(BlueprintFillError, match="unknown slot"):
            fill_blueprint(bp, {"time": "08:00", "nonexistent": "value"})

    def test_fill_blueprint_invalid_time(self):
        bp = get_blueprint("morning-brief")
        with pytest.raises(BlueprintFillError, match="invalid time"):
            fill_blueprint(bp, {"time": "abc", "deliver": "origin"})

    def test_fill_blueprint_invalid_enum(self):
        bp = get_blueprint("weekly-review")
        with pytest.raises(BlueprintFillError, match="not allowed"):
            fill_blueprint(bp, {
                "time": "18:00", "day": "wednesday", "deliver": "origin"
            })  # wednesday not in options

    def test_fill_blueprint_with_origin(self):
        bp = get_blueprint("morning-brief")
        origin = {"platform": "telegram", "chat_id": "123"}
        spec = fill_blueprint(
            bp, {"time": "08:00", "deliver": "origin"}, origin=origin
        )
        assert spec["origin"] == origin

    def test_blueprint_form_schema(self):
        bp = get_blueprint("morning-brief")
        schema = blueprint_form_schema(bp)
        assert schema["key"] == "morning-brief"
        assert "fields" in schema
        assert len(schema["fields"]) == 2  # time + deliver

    def test_blueprint_slash_command(self):
        bp = get_blueprint("morning-brief")
        cmd = blueprint_slash_command(bp)
        assert cmd.startswith("/blueprint morning-brief")
        assert "time=" in cmd

    def test_blueprint_deeplink(self):
        bp = get_blueprint("morning-brief")
        link = blueprint_deeplink(bp)
        assert link.startswith("nia://blueprint/morning-brief")

    def test_blueprint_catalog_entry(self):
        bp = get_blueprint("morning-brief")
        entry = blueprint_catalog_entry(bp)
        assert entry["key"] == "morning-brief"
        assert "slash_command" in entry
        assert "deeplink" in entry
        assert "humanized_schedule" in entry

    def test_fill_blueprint_weekly_review_day(self):
        bp = get_blueprint("weekly-review")
        spec = fill_blueprint(bp, {
            "time": "18:00", "day": "monday", "deliver": "origin"
        })
        # Monday = cron dow 1.
        assert spec["schedule"] == "0 18 * * 1"

    def test_fill_blueprint_custom_reminder_recurrence(self):
        bp = get_blueprint("custom-reminder")
        spec = fill_blueprint(bp, {
            "what": "drink water", "time": "14:00",
            "recurrence": "weekdays", "deliver": "origin"
        })
        # weekdays = cron dow 1-5.
        assert spec["schedule"] == "0 14 * * 1-5"

    def test_fill_blueprint_important_mail_interval(self):
        bp = get_blueprint("important-mail")
        spec = fill_blueprint(bp, {
            "interval_min": "15",
            "criteria": "from my boss",
            "deliver": "origin"
        })
        assert spec["schedule"] == "*/15 * * * *"

    def test_slot_type_validation(self):
        with pytest.raises(ValueError, match="unknown slot type"):
            BlueprintSlot(name="x", type="invalid", label="X")


# ---------------------------------------------------------------------------
# suggestions.py
# ---------------------------------------------------------------------------


class TestSuggestions:
    def test_add_and_list_pending(self):
        add_suggestion(
            ref="test-1", title="Test", description="Test suggestion",
        )
        pending = list_pending()
        assert len(pending) == 1
        assert pending[0]["ref"] == "test-1"

    def test_add_duplicate_updates(self):
        add_suggestion(ref="test-1", title="Original", description="v1")
        add_suggestion(ref="test-1", title="Updated", description="v2")
        pending = list_pending()
        assert len(pending) == 1
        assert pending[0]["title"] == "Updated"

    def test_dismiss_suggestion(self):
        add_suggestion(ref="test-1", title="Test", description="Test")
        assert dismiss_suggestion("test-1") is True
        assert len(list_pending()) == 0
        assert len(list_dismissed()) == 1

    def test_dismiss_not_found(self):
        assert dismiss_suggestion("nonexistent") is False

    def test_accept_suggestion_with_blueprint(self):
        add_suggestion(
            ref="test-1",
            title="Morning brief",
            description="Test",
            blueprint_key="morning-brief",
            blueprint_values={"time": "07:30", "deliver": "telegram"},
        )
        spec = accept_suggestion("test-1")
        assert spec is not None
        assert spec["schedule"] == "30 7 * * *"
        assert spec["deliver"] == "telegram"
        assert len(list_accepted()) == 1

    def test_accept_suggestion_direct_prompt(self):
        add_suggestion(
            ref="test-1",
            title="Custom job",
            description="Test",
            prompt="Check the weather",
            schedule="0 8 * * *",
        )
        spec = accept_suggestion("test-1")
        assert spec is not None
        assert spec["prompt"] == "Check the weather"
        assert spec["schedule"] == "0 8 * * *"

    def test_accept_suggestion_not_found(self):
        assert accept_suggestion("nonexistent") is None

    def test_accept_suggestion_blueprint_not_found(self):
        add_suggestion(
            ref="test-1", title="Test", description="Test",
            blueprint_key="nonexistent",
        )
        spec = accept_suggestion("test-1")
        assert spec is None

    def test_clear_resolved(self):
        add_suggestion(ref="a", title="A", description="A")
        add_suggestion(ref="b", title="B", description="B")
        dismiss_suggestion("a")
        accept_suggestion("b")
        removed = clear_resolved()
        assert removed == 2
        assert len(list_pending()) == 0

    def test_clear_all(self):
        add_suggestion(ref="a", title="A", description="A")
        add_suggestion(ref="b", title="B", description="B")
        removed = clear_all()
        assert removed == 2
        assert len(load_suggestions()) == 0

    def test_get_suggestion(self):
        add_suggestion(ref="test-1", title="Test", description="Test")
        s = get_suggestion("test-1")
        assert s is not None
        assert s["title"] == "Test"

    def test_get_suggestion_not_found(self):
        assert get_suggestion("nonexistent") is None

    def test_dismissed_not_re_proposed(self):
        add_suggestion(ref="test-1", title="V1", description="v1")
        dismiss_suggestion("test-1")
        # Try to re-add → should return False (not re-proposed).
        result = add_suggestion(ref="test-1", title="V2", description="v2")
        assert result is False
        # Still dismissed.
        assert len(list_dismissed()) == 1
        assert len(list_pending()) == 0


# ---------------------------------------------------------------------------
# Integration: cron.py upsert_cron_job wiring
# ---------------------------------------------------------------------------


class TestCronWiring:
    def test_upsert_cron_job_rejects_gateway_lifecycle(self):
        """The upsert_cron_job function should reject jobs with gateway-lifecycle commands."""
        from niaharness.services.cron import upsert_cron_job

        with pytest.raises(GatewayLifecycleBlocked):
            upsert_cron_job({
                "name": "evil-restart",
                "schedule": "0 * * * *",
                "prompt": "Run nia gateway restart every hour",
            })

    def test_upsert_cron_job_rejects_credential_exfil(self):
        """The upsert_cron_job function should reject jobs with unsafe base_url."""
        from niaharness.services.cron import upsert_cron_job

        with pytest.raises(RuntimeError, match="blocked for safety"):
            upsert_cron_job({
                "name": "exfil-attempt",
                "schedule": "0 * * * *",
                "prompt": "Check something",
                "provider": "anthropic",
                "base_url": "https://evil.com",
            })

    def test_upsert_cron_job_accepts_safe_job(self):
        """The upsert_cron_job function should accept safe jobs."""
        from niaharness.services.cron import upsert_cron_job, delete_cron_job

        job = upsert_cron_job({
            "name": "safe-job",
            "schedule": "0 8 * * *",
            "prompt": "Good morning!",
        })
        assert job["name"] == "safe-job"
        # Cleanup.
        delete_cron_job("safe-job")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
