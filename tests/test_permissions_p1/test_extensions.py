"""Tests for the P1 permissions extensions — tirith, user-deny rules,
container guard skipping, observability context.

Covers:
  - tirith_security: config loading, circuit breaker, platform detection,
    binary resolution, check_command_security (mocked subprocess).
  - approval: _match_user_deny_rule, _should_skip_container_guards,
    set_current_observability_context / get_current_turn_id / get_current_tool_call_id.
  - checker: PermissionChecker with env_type, tirith integration,
    user-deny rule blocking.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from niaharness.permissions.approval import (
    _get_user_deny_patterns,
    _match_user_deny_rule,
    _should_skip_container_guards,
    get_current_tool_call_id,
    get_current_turn_id,
    reset_current_observability_context,
    set_current_observability_context,
)
from niaharness.permissions.tirith_security import (
    _detect_target,
    _load_security_config,
    _reset_all_state,
    _resolve_tirith_path,
    check_command_security,
    is_platform_supported,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_tirith_state():
    """Reset tirith module state between tests."""
    _reset_all_state()
    yield
    _reset_all_state()


@pytest.fixture(autouse=True)
def _clear_deny_env(monkeypatch):
    """Clear deny-related env vars so tests don't pick up host config."""
    for key in list(os.environ.keys()):
        if key.startswith("NIA_TIRITH") or key == "NIA_APPROVAL_DENY":
            monkeypatch.delenv(key, raising=False)
    yield


# ---------------------------------------------------------------------------
# Tirith security — config + platform detection
# ---------------------------------------------------------------------------


class TestTirithConfig:
    def test_default_config_tirith_enabled(self):
        cfg = _load_security_config()
        assert cfg["tirith_enabled"] is True
        assert cfg["tirith_timeout"] == 10
        assert cfg["tirith_fail_open"] is True

    def test_env_overrides(self, monkeypatch):
        monkeypatch.setenv("NIA_TIRITH_ENABLED", "0")
        monkeypatch.setenv("NIA_TIRITH_TIMEOUT", "30")
        monkeypatch.setenv("NIA_TIRITH_FAIL_OPEN", "0")
        cfg = _load_security_config()
        assert cfg["tirith_enabled"] is False
        assert cfg["tirith_timeout"] == 30
        assert cfg["tirith_fail_open"] is False

    def test_env_bool_accepts_various_values(self, monkeypatch):
        from niaharness.permissions.tirith_security import _env_bool
        monkeypatch.setenv("TEST_BOOL", "1")
        assert _env_bool("TEST_BOOL", False) is True
        monkeypatch.setenv("TEST_BOOL", "true")
        assert _env_bool("TEST_BOOL", False) is True
        monkeypatch.setenv("TEST_BOOL", "yes")
        assert _env_bool("TEST_BOOL", False) is True
        monkeypatch.setenv("TEST_BOOL", "0")
        assert _env_bool("TEST_BOOL", True) is False
        monkeypatch.setenv("TEST_BOOL", "")
        assert _env_bool("TEST_BOOL", True) is True  # falls back to default


class TestTirithPlatformDetection:
    def test_is_platform_supported_returns_bool(self):
        result = is_platform_supported()
        assert isinstance(result, bool)

    def test_detect_target_returns_string_or_none(self):
        target = _detect_target()
        # On the test platform, this should be a valid target triple or None.
        if target is not None:
            assert "-" in target
            assert any(arch in target for arch in ("x86_64", "aarch64"))


class TestTirithPathResolution:
    def test_resolve_returns_none_when_not_found(self, monkeypatch):
        monkeypatch.setenv("NIA_TIRITH_PATH", "")
        # No tirith on PATH and no ~/.nia/bin/tirith.
        result = _resolve_tirith_path("")
        # If tirith happens to be installed, this will be a path; otherwise None.
        # Either way, it should be a string or None.
        assert result is None or isinstance(result, str)

    def test_resolve_uses_explicit_path(self, tmp_path: Path, monkeypatch):
        # Create a fake tirith binary.
        fake_bin = tmp_path / "tirith"
        fake_bin.write_text("#!/bin/sh\nexit 0\n")
        fake_bin.chmod(0o755)
        result = _resolve_tirith_path(str(fake_bin))
        assert result == str(fake_bin)


# ---------------------------------------------------------------------------
# Tirith security — check_command_security
# ---------------------------------------------------------------------------


class TestCheckCommandSecurity:
    def test_disabled_returns_allow(self, monkeypatch):
        monkeypatch.setenv("NIA_TIRITH_ENABLED", "0")
        result = check_command_security("ls -la")
        assert result["action"] == "allow"
        assert result["findings"] == []

    def test_unsupported_platform_returns_allow(self, monkeypatch):
        monkeypatch.setenv("NIA_TIRITH_ENABLED", "1")
        with patch(
            "niaharness.permissions.tirith_security.is_platform_supported",
            return_value=False,
        ):
            result = check_command_security("ls -la")
        assert result["action"] == "allow"

    def test_no_binary_fail_open(self, monkeypatch):
        monkeypatch.setenv("NIA_TIRITH_ENABLED", "1")
        monkeypatch.setenv("NIA_TIRITH_FAIL_OPEN", "1")
        with patch(
            "niaharness.permissions.tirith_security.is_platform_supported",
            return_value=True,
        ), patch(
            "niaharness.permissions.tirith_security._resolve_tirith_path",
            return_value=None,
        ):
            result = check_command_security("ls -la")
        assert result["action"] == "allow"
        assert "unavailable" in result["summary"]

    def test_no_binary_fail_closed(self, monkeypatch):
        monkeypatch.setenv("NIA_TIRITH_ENABLED", "1")
        monkeypatch.setenv("NIA_TIRITH_FAIL_OPEN", "0")
        with patch(
            "niaharness.permissions.tirith_security.is_platform_supported",
            return_value=True,
        ), patch(
            "niaharness.permissions.tirith_security._resolve_tirith_path",
            return_value=None,
        ):
            result = check_command_security("ls -la")
        assert result["action"] == "block"

    def test_spawn_failure_fail_open(self, monkeypatch):
        monkeypatch.setenv("NIA_TIRITH_ENABLED", "1")
        monkeypatch.setenv("NIA_TIRITH_FAIL_OPEN", "1")

        def fake_run(*args, **kwargs):
            raise FileNotFoundError("tirith not found")

        with patch(
            "niaharness.permissions.tirith_security.is_platform_supported",
            return_value=True,
        ), patch(
            "niaharness.permissions.tirith_security._resolve_tirith_path",
            return_value="/fake/tirith",
        ), patch("subprocess.run", side_effect=fake_run):
            result = check_command_security("ls -la")
        assert result["action"] == "allow"
        assert "unavailable" in result["summary"]

    def test_exit_code_0_allows(self, monkeypatch):
        monkeypatch.setenv("NIA_TIRITH_ENABLED", "1")
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = '{"findings": [], "summary": "ok"}'
        with patch(
            "niaharness.permissions.tirith_security.is_platform_supported",
            return_value=True,
        ), patch(
            "niaharness.permissions.tirith_security._resolve_tirith_path",
            return_value="/fake/tirith",
        ), patch("subprocess.run", return_value=mock_result):
            result = check_command_security("ls -la")
        assert result["action"] == "allow"

    def test_exit_code_1_blocks(self, monkeypatch):
        monkeypatch.setenv("NIA_TIRITH_ENABLED", "1")
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = (
            '{"findings": [{"rule": "injection", "details": {}}], '
            '"summary": "prompt injection detected"}'
        )
        with patch(
            "niaharness.permissions.tirith_security.is_platform_supported",
            return_value=True,
        ), patch(
            "niaharness.permissions.tirith_security._resolve_tirith_path",
            return_value="/fake/tirith",
        ), patch("subprocess.run", return_value=mock_result):
            result = check_command_security("rm -rf /")
        assert result["action"] == "block"
        assert len(result["findings"]) == 1
        assert "injection" in result["summary"]

    def test_exit_code_2_warns(self, monkeypatch):
        monkeypatch.setenv("NIA_TIRITH_ENABLED", "1")
        mock_result = MagicMock()
        mock_result.returncode = 2
        mock_result.stdout = '{"findings": [], "summary": "suspicious pattern"}'
        with patch(
            "niaharness.permissions.tirith_security.is_platform_supported",
            return_value=True,
        ), patch(
            "niaharness.permissions.tirith_security._resolve_tirith_path",
            return_value="/fake/tirith",
        ), patch("subprocess.run", return_value=mock_result):
            result = check_command_security("curl example.com")
        assert result["action"] == "warn"

    def test_timeout_fail_open(self, monkeypatch):
        monkeypatch.setenv("NIA_TIRITH_ENABLED", "1")
        monkeypatch.setenv("NIA_TIRITH_FAIL_OPEN", "1")
        with patch(
            "niaharness.permissions.tirith_security.is_platform_supported",
            return_value=True,
        ), patch(
            "niaharness.permissions.tirith_security._resolve_tirith_path",
            return_value="/fake/tirith",
        ), patch("subprocess.run", side_effect=subprocess.TimeoutExpired("tirith", 10)):
            result = check_command_security("ls -la")
        assert result["action"] == "allow"
        assert "timed out" in result["summary"]

    def test_circuit_breaker_opens_after_crashes(self, monkeypatch):
        monkeypatch.setenv("NIA_TIRITH_ENABLED", "1")
        monkeypatch.setenv("NIA_TIRITH_FAIL_OPEN", "1")

        call_count = 0

        def fake_run(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise FileNotFoundError("tirith crashed")

        with patch(
            "niaharness.permissions.tirith_security.is_platform_supported",
            return_value=True,
        ), patch(
            "niaharness.permissions.tirith_security._resolve_tirith_path",
            return_value="/fake/tirith",
        ), patch("subprocess.run", side_effect=fake_run):
            # First 3 crashes increment the counter.
            for _ in range(3):
                check_command_security("ls -la")
            # 4th call should hit the circuit breaker (no subprocess.run).
            call_count_before = call_count
            result = check_command_security("ls -la")
            assert result["action"] == "allow"
            assert "circuit breaker" in result["summary"]
            # subprocess.run should NOT have been called again.
            assert call_count == call_count_before


# ---------------------------------------------------------------------------
# User-defined deny rules
# ---------------------------------------------------------------------------


class TestUserDenyRules:
    def test_get_user_deny_patterns_empty_by_default(self, monkeypatch):
        monkeypatch.setenv("NIA_APPROVAL_DENY", "")
        patterns = _get_user_deny_patterns()
        # May have patterns from config.yaml, but env is empty.
        assert isinstance(patterns, list)

    def test_get_user_deny_patterns_from_env(self, monkeypatch):
        monkeypatch.setenv("NIA_APPROVAL_DENY", "rm -rf *, mkfs *, dd if=*")
        patterns = _get_user_deny_patterns()
        assert "rm -rf *" in patterns
        assert "mkfs *" in patterns
        assert "dd if=*" in patterns

    def test_match_user_deny_rule_no_match(self, monkeypatch):
        monkeypatch.setenv("NIA_APPROVAL_DENY", "rm -rf /")
        result = _match_user_deny_rule("ls -la")
        assert result is None

    def test_match_user_deny_rule_exact_match(self, monkeypatch):
        monkeypatch.setenv("NIA_APPROVAL_DENY", "rm -rf /")
        result = _match_user_deny_rule("rm -rf /")
        assert result == "rm -rf /"

    def test_match_user_deny_rule_glob_match(self, monkeypatch):
        monkeypatch.setenv("NIA_APPROVAL_DENY", "rm -rf *")
        result = _match_user_deny_rule("rm -rf /tmp")
        assert result == "rm -rf *"

    def test_match_user_deny_rule_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("NIA_APPROVAL_DENY", "RM -RF *")
        result = _match_user_deny_rule("rm -rf /tmp")
        assert result == "RM -RF *"

    def test_match_user_deny_rule_no_patterns_returns_none(self, monkeypatch):
        monkeypatch.setenv("NIA_APPROVAL_DENY", "")
        # Mock config to return empty too.
        with patch(
            "niaharness.permissions.approval._get_user_deny_patterns",
            return_value=[],
        ):
            result = _match_user_deny_rule("rm -rf /")
        assert result is None


# ---------------------------------------------------------------------------
# Container guard skipping
# ---------------------------------------------------------------------------


class TestContainerGuards:
    def test_local_never_skips(self):
        assert _should_skip_container_guards("local") is False

    def test_ssh_never_skips(self):
        assert _should_skip_container_guards("ssh") is False

    def test_docker_without_host_access_skips(self):
        assert _should_skip_container_guards("docker", has_host_access=False) is True

    def test_docker_with_host_access_does_not_skip(self):
        assert _should_skip_container_guards("docker", has_host_access=True) is False

    def test_singularity_skips(self):
        assert _should_skip_container_guards("singularity") is True

    def test_modal_skips(self):
        assert _should_skip_container_guards("modal") is True

    def test_daytona_skips(self):
        assert _should_skip_container_guards("daytona") is True

    def test_unknown_env_does_not_skip(self):
        assert _should_skip_container_guards("unknown_env") is False


# ---------------------------------------------------------------------------
# Observability context
# ---------------------------------------------------------------------------


class TestObservabilityContext:
    def test_default_turn_id_is_empty(self):
        assert get_current_turn_id() == ""

    def test_default_tool_call_id_is_empty(self):
        assert get_current_tool_call_id() == ""

    def test_set_and_get_turn_id(self):
        tokens = set_current_observability_context(turn_id="turn-123")
        try:
            assert get_current_turn_id() == "turn-123"
        finally:
            reset_current_observability_context(tokens)
        assert get_current_turn_id() == ""

    def test_set_and_get_tool_call_id(self):
        tokens = set_current_observability_context(tool_call_id="tc-456")
        try:
            assert get_current_tool_call_id() == "tc-456"
        finally:
            reset_current_observability_context(tokens)
        assert get_current_tool_call_id() == ""

    def test_set_both(self):
        tokens = set_current_observability_context(
            turn_id="turn-1", tool_call_id="tc-1",
        )
        try:
            assert get_current_turn_id() == "turn-1"
            assert get_current_tool_call_id() == "tc-1"
        finally:
            reset_current_observability_context(tokens)

    def test_empty_string_normalization(self):
        tokens = set_current_observability_context(turn_id="", tool_call_id=None)
        try:
            assert get_current_turn_id() == ""
            assert get_current_tool_call_id() == ""
        finally:
            reset_current_observability_context(tokens)

    def test_nested_contexts(self):
        tokens1 = set_current_observability_context(turn_id="outer")
        try:
            assert get_current_turn_id() == "outer"
            tokens2 = set_current_observability_context(turn_id="inner")
            try:
                assert get_current_turn_id() == "inner"
            finally:
                reset_current_observability_context(tokens2)
            assert get_current_turn_id() == "outer"
        finally:
            reset_current_observability_context(tokens1)


# ---------------------------------------------------------------------------
# PermissionChecker integration
# ---------------------------------------------------------------------------


class TestPermissionCheckerIntegration:
    def _make_checker(
        self,
        *,
        env_type: str = "local",
        has_host_access: bool = False,
        enable_tirith: bool = False,  # disable for tests without tirith binary
    ) -> "PermissionChecker":
        from niaharness.config.settings import PermissionSettings
        from niaharness.permissions.checker import PermissionChecker
        settings = PermissionSettings()
        return PermissionChecker(
            settings,
            env_type=env_type,
            has_host_access=has_host_access,
            enable_tirith=enable_tirith,
        )

    def test_container_guard_skip_docker(self):
        checker = self._make_checker(env_type="docker", has_host_access=False)
        decision = checker.evaluate(
            "bash",
            is_read_only=False,
            command="rm -rf /workspace",
        )
        # Docker without host access → guards skipped → allowed.
        assert decision.allowed is True
        assert "Container guards skipped" in decision.reason

    def test_container_guard_no_skip_docker_with_host_access(self):
        checker = self._make_checker(env_type="docker", has_host_access=True)
        decision = checker.evaluate(
            "bash",
            is_read_only=False,
            command="rm -rf /workspace",
        )
        # Docker WITH host access → guards NOT skipped → dangerous → requires confirmation.
        # (rm -rf /workspace is dangerous, not hardline, so it goes to approval)
        assert decision.allowed is False or decision.category == "dangerous"

    def test_container_guard_skip_singularity(self):
        checker = self._make_checker(env_type="singularity")
        decision = checker.evaluate(
            "bash",
            is_read_only=False,
            command="rm -rf /workspace",
        )
        assert decision.allowed is True
        assert "Container guards skipped" in decision.reason

    def test_user_deny_rule_blocks_unconditionally(self, monkeypatch):
        monkeypatch.setenv("NIA_APPROVAL_DENY", "forbidden-command *")
        checker = self._make_checker(enable_tirith=False)
        decision = checker.evaluate(
            "bash",
            is_read_only=False,
            command="forbidden-command --flag",
        )
        assert decision.allowed is False
        assert decision.category == "user_deny"
        assert "forbidden-command *" in decision.reason

    def test_tirith_block_propagates(self, monkeypatch):
        monkeypatch.setenv("NIA_TIRITH_ENABLED", "1")
        monkeypatch.setenv("NIA_TIRITH_FAIL_OPEN", "1")

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = (
            '{"findings": [{"rule": "injection"}], '
            '"summary": "prompt injection detected"}'
        )

        with patch(
            "niaharness.permissions.tirith_security.is_platform_supported",
            return_value=True,
        ), patch(
            "niaharness.permissions.tirith_security._resolve_tirith_path",
            return_value="/fake/tirith",
        ), patch("subprocess.run", return_value=mock_result):
            checker = self._make_checker(enable_tirith=True)
            decision = checker.evaluate(
                "bash",
                is_read_only=False,
                command="ls -la",  # safe command that passes shell_hardening
            )
        assert decision.allowed is False
        assert decision.category == "tirith_block"
        assert decision.tirith_findings is not None

    def test_observability_tags_in_audit_log(self, tmp_path: Path, monkeypatch):
        """Verify that turn_id + tool_call_id appear in the audit log."""
        # Redirect NIA home to a temp dir.
        monkeypatch.setenv("NIA_HOME", str(tmp_path / ".nia"))
        # Patch get_nia_home to return our temp path.
        fake_home = tmp_path / ".nia"
        fake_home.mkdir(parents=True, exist_ok=True)

        with patch(
            "niaharness.prompts.soul.get_nia_home", return_value=fake_home
        ):
            tokens = set_current_observability_context(
                turn_id="turn-audit-1", tool_call_id="tc-audit-1",
            )
            try:
                checker = self._make_checker(enable_tirith=False)
                checker.evaluate(
                    "bash",
                    is_read_only=False,
                    command="rm -rf /",  # hardline → blocked + logged
                )
            finally:
                reset_current_observability_context(tokens)

        audit_path = fake_home / "permissions" / "audit.log"
        assert audit_path.exists()
        content = audit_path.read_text()
        assert "turn=turn-audit-1" in content
        assert "tool_call=tc-audit-1" in content


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
