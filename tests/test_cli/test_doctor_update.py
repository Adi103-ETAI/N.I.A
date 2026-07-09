"""Tests for NIA Doctor + Update modules (Task 10).

Covers:
  - Security advisories: Advisory dataclass, ADVISORIES catalog, detect_compromised, filter_unacked, _ack_advisory
  - Doctor: run_doctor (dry-run, --fix, --ack), DoctorResult fields, check sections present
  - Session DB repair: FTS check, WAL checkpoint (mocked)
  - Provider probes: _probe_provider (200/401/402/429/no-key), parallel execution
  - Update: detect_install_method, get_current_version, check_for_update, create_pre_update_backup, run_update (--check)
  - Restart: restart_process (mocked)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from niaharness.cli.doctor import (
    ADVISORIES,
    Advisory,
    AdvisoryHit,
    DoctorResult,
    _ack_advisory,
    _check_ok,
    _check_warn,
    _check_fail,
    _check_info,
    _probe_provider,
    _run_provider_probes,
    detect_compromised,
    filter_unacked,
    run_doctor,
)
from niaharness.cli.update import (
    UpdateResult,
    check_for_update,
    create_pre_update_backup,
    detect_install_method,
    get_current_version,
    prune_old_backups,
    run_update,
)


# ---------------------------------------------------------------------------
# Security advisories
# ---------------------------------------------------------------------------


class TestAdvisoryDataclass:
    def test_advisory_is_frozen(self):
        adv = Advisory(
            id="test-001", title="Test", summary="test", url="https://example.com",
            compromised=(("testpkg", frozenset({"1.0.0"})),),
            remediation=("uninstall",), published="2026-01-01", severity="high",
        )
        assert adv.id == "test-001"
        with pytest.raises(AttributeError):
            adv.id = "other"  # type: ignore[misc]


class TestAdvisoriesCatalog:
    def test_catalog_nonempty(self):
        assert len(ADVISORIES) >= 1

    def test_shai_hulud_entry(self):
        ids = {a.id for a in ADVISORIES}
        assert "shai-hulud-2026-05" in ids

    def test_all_have_required_fields(self):
        for adv in ADVISORIES:
            assert adv.id
            assert adv.title
            assert adv.url
            assert adv.compromised
            assert adv.remediation
            assert adv.severity in ("critical", "high", "medium", "low")


class TestDetectCompromised:
    def test_no_hits_when_package_not_installed(self):
        """If mistralai is not installed, no hits."""
        hits = detect_compromised()
        # Should be empty (or only contain packages that ARE installed).
        for hit in hits:
            assert hit.package != "nonexistent-package-xyz"

    def test_hit_when_compromised_version_installed(self):
        """Mock importlib.metadata.version to return the compromised version."""
        advisory = Advisory(
            id="test-001", title="Test", summary="test", url="https://example.com",
            compromised=(("mistralai", frozenset({"2.4.6"})),),
            remediation=("uninstall",), published="2026-01-01", severity="critical",
        )
        with patch("importlib.metadata.version", return_value="2.4.6"):
            hits = detect_compromised(advisories=(advisory,))
        assert len(hits) == 1
        assert hits[0].package == "mistralai"
        assert hits[0].installed_version == "2.4.6"
        assert hits[0].advisory.id == "test-001"

    def test_no_hit_when_version_not_compromised(self):
        advisory = Advisory(
            id="test-001", title="Test", summary="test", url="https://example.com",
            compromised=(("mistralai", frozenset({"2.4.6"})),),
            remediation=("uninstall",), published="2026-01-01", severity="critical",
        )
        with patch("importlib.metadata.version", return_value="2.4.7"):
            hits = detect_compromised(advisories=(advisory,))
        assert len(hits) == 0

    def test_hit_when_bad_versions_empty(self):
        """Empty bad_versions means any version is compromised."""
        advisory = Advisory(
            id="test-002", title="Test", summary="test", url="https://example.com",
            compromised=(("mistralai", frozenset()),),
            remediation=("uninstall",), published="2026-01-01", severity="critical",
        )
        with patch("importlib.metadata.version", return_value="99.99.99"):
            hits = detect_compromised(advisories=(advisory,))
        assert len(hits) == 1

    def test_package_not_installed_no_hit(self):
        advisory = Advisory(
            id="test-003", title="Test", summary="test", url="https://example.com",
            compromised=(("nonexistent-pkg-xyz", frozenset({"1.0.0"})),),
            remediation=("uninstall",), published="2026-01-01", severity="critical",
        )
        hits = detect_compromised(advisories=(advisory,))
        assert len(hits) == 0


class TestAckAdvisory:
    def test_ack_persists(self, tmp_path, monkeypatch):
        from niaharness.prompts.soul import get_nia_home
        monkeypatch.setattr("niaharness.prompts.soul.get_nia_home", lambda: tmp_path)
        assert _ack_advisory("test-ack-001") is True
        ack_path = tmp_path / "acked_advisories.json"
        assert ack_path.exists()
        acked = json.loads(ack_path.read_text())
        assert "test-ack-001" in acked

    def test_ack_idempotent(self, tmp_path, monkeypatch):
        monkeypatch.setattr("niaharness.prompts.soul.get_nia_home", lambda: tmp_path)
        _ack_advisory("test-ack-002")
        _ack_advisory("test-ack-002")  # Second time.
        ack_path = tmp_path / "acked_advisories.json"
        acked = json.loads(ack_path.read_text())
        assert acked.count("test-ack-002") == 1  # Not duplicated.


class TestFilterUnacked:
    def test_filters_acked(self, tmp_path, monkeypatch):
        monkeypatch.setattr("niaharness.prompts.soul.get_nia_home", lambda: tmp_path)
        _ack_advisory("test-filter-001")

        advisory = Advisory(
            id="test-filter-001", title="Test", summary="test", url="https://example.com",
            compromised=(("mistralai", frozenset({"2.4.6"})),),
            remediation=("uninstall",), published="2026-01-01", severity="critical",
        )
        hit = AdvisoryHit(advisory=advisory, package="mistralai", installed_version="2.4.6")
        filtered = filter_unacked([hit])
        assert len(filtered) == 0  # Acked → filtered out.

    def test_keeps_unacked(self):
        advisory = Advisory(
            id="test-filter-002", title="Test", summary="test", url="https://example.com",
            compromised=(("mistralai", frozenset({"2.4.6"})),),
            remediation=("uninstall",), published="2026-01-01", severity="critical",
        )
        hit = AdvisoryHit(advisory=advisory, package="mistralai", installed_version="2.4.6")
        filtered = filter_unacked([hit])
        assert len(filtered) == 1  # Not acked → kept.


# ---------------------------------------------------------------------------
# Check helpers
# ---------------------------------------------------------------------------


class TestCheckHelpers:
    def test_check_ok(self):
        assert "✓" in _check_ok("test")

    def test_check_warn(self):
        assert "⚠" in _check_warn("test")
        assert "hint" in _check_warn("test", "hint")

    def test_check_fail(self):
        assert "✗" in _check_fail("test")
        assert "hint" in _check_fail("test", "hint")

    def test_check_info(self):
        assert "ℹ" in _check_info("test")


# ---------------------------------------------------------------------------
# Provider probes
# ---------------------------------------------------------------------------


class TestProbeProvider:
    def test_no_api_key(self):
        with patch.dict(os.environ, {}, clear=False):
            for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN"):
                os.environ.pop(var, None)
            result = _probe_provider(
                "Anthropic", ("ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN"),
                "https://api.anthropic.com/v1/models", None, True,
            )
        assert result.label == "Anthropic"
        assert "no API key" in result.lines[0]
        assert result.issue is None

    def test_success_200(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test-123"}):
            mock_response = MagicMock()
            mock_response.status_code = 200
            with patch("httpx.get", return_value=mock_response):
                result = _probe_provider(
                    "Anthropic", ("ANTHROPIC_API_KEY",),
                    "https://api.anthropic.com/v1/models", None, True,
                )
        assert "connected" in result.lines[0]
        assert result.issue is None

    def test_invalid_key_401(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-bad-123"}):
            mock_response = MagicMock()
            mock_response.status_code = 401
            with patch("httpx.get", return_value=mock_response):
                result = _probe_provider(
                    "Anthropic", ("ANTHROPIC_API_KEY",),
                    "https://api.anthropic.com/v1/models", None, True,
                )
        assert "401" in result.lines[0]
        assert result.issue is not None

    def test_out_of_credits_402(self):
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-test"}):
            mock_response = MagicMock()
            mock_response.status_code = 402
            with patch("httpx.get", return_value=mock_response):
                result = _probe_provider(
                    "DeepSeek", ("DEEPSEEK_API_KEY",),
                    "https://api.deepseek.com/v1/models", None, True,
                )
        assert "402" in result.lines[0]
        assert result.issue is not None

    def test_rate_limited_429(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            mock_response = MagicMock()
            mock_response.status_code = 429
            with patch("httpx.get", return_value=mock_response):
                result = _probe_provider(
                    "OpenAI", ("OPENAI_API_KEY",),
                    "https://api.openai.com/v1/models", None, True,
                )
        assert "429" in result.lines[0]
        assert result.issue is None  # Rate limit is not a persistent issue.

    def test_supports_health_check_false(self):
        with patch.dict(os.environ, {"MINIMAX_CN_API_KEY": "sk-test"}):
            result = _probe_provider(
                "MiniMax CN", ("MINIMAX_CN_API_KEY",),
                "https://api.minimaxi.com/v1/models", None, False,
            )
        assert "health check skipped" in result.lines[0]

    def test_base_url_override(self):
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "sk-test",
            "OPENAI_BASE_URL": "https://custom.example.com/v1",
        }):
            mock_response = MagicMock()
            mock_response.status_code = 200
            with patch("httpx.get", return_value=mock_response) as mock_get:
                result = _probe_provider(
                    "OpenAI", ("OPENAI_API_KEY",),
                    "https://api.openai.com/v1/models", "OPENAI_BASE_URL", True,
                )
        # Should have used the override URL.
        called_url = str(mock_get.call_args[0][0])
        assert "custom.example.com" in called_url


class TestRunProviderProbes:
    def test_returns_all_providers(self):
        with patch.dict(os.environ, {}, clear=False):
            # Clear all provider keys.
            for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY",
                        "DEEPSEEK_API_KEY", "GLM_API_KEY", "KIMI_API_KEY",
                        "MINIMAX_API_KEY", "DASHSCOPE_API_KEY", "HF_TOKEN"):
                os.environ.pop(var, None)
            results = _run_provider_probes()
        assert len(results) == 9  # All 9 providers in the table.

    def test_results_in_table_order(self):
        with patch.dict(os.environ, {}, clear=False):
            for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
                os.environ.pop(var, None)
            results = _run_provider_probes()
        labels = [r.label for r in results]
        assert labels[0] == "Anthropic"
        assert labels[1] == "OpenAI"


# ---------------------------------------------------------------------------
# run_doctor
# ---------------------------------------------------------------------------


class TestRunDoctor:
    def test_dry_run_returns_report(self):
        result = run_doctor(fix=False)
        assert isinstance(result, DoctorResult)
        assert "NIA Doctor" in result.report
        assert "Security Advisories" in result.report
        assert "Python Environment" in result.report

    def test_fix_mode(self, tmp_path, monkeypatch):
        """--fix should create missing directories."""
        monkeypatch.setattr("niaharness.prompts.soul.get_nia_home", lambda: tmp_path)
        result = run_doctor(fix=True)
        assert result.fixed_count > 0
        # Directories should be created.
        assert (tmp_path / "cron").exists()
        assert (tmp_path / "sessions").exists()
        assert (tmp_path / "SOUL.md").exists()

    def test_ack_fast_path(self, tmp_path, monkeypatch):
        monkeypatch.setattr("niaharness.prompts.soul.get_nia_home", lambda: tmp_path)
        result = run_doctor(ack="shai-hulud-2026-05")
        assert "Acknowledged" in result.report

    def test_ack_unknown_id(self):
        result = run_doctor(ack="nonexistent-id")
        assert "Unknown advisory ID" in result.report

    def test_report_contains_all_sections(self):
        result = run_doctor(fix=False)
        for section in [
            "Security Advisories",
            "MCP Server Security",
            "Python Environment",
            "Configuration",
            "Session Database",
            "Directory Structure",
            "Provider Connectivity",
            "SSL / CA Certificates",
            "External Tools",
            "Summary",
        ]:
            assert section in result.report, f"Missing section: {section}"


# ---------------------------------------------------------------------------
# Update: install method detection
# ---------------------------------------------------------------------------


class TestDetectInstallMethod:
    def test_returns_string(self):
        method = detect_install_method()
        assert isinstance(method, str)
        assert method in ("uv-tool", "pipx", "venv-pip", "editable", "docker", "pip")

    def test_uv_tool_detection(self):
        with patch.object(sys, "prefix", "/home/user/.local/share/uv/tools/niaharness"):
            assert detect_install_method() == "uv-tool"

    def test_pipx_detection(self):
        with patch.object(sys, "prefix", "/home/user/.local/pipx/venvs/niaharness"):
            assert detect_install_method() == "pipx"

    def test_docker_detection(self):
        # Docker detection runs after uv-tool/pipx/editable checks.
        # We need to ensure none of those match first.
        # Only patch the /.dockerenv check, not all Path.exists calls.
        original_exists = Path.exists

        def _mocked_exists(self):
            if str(self) == "/.dockerenv":
                return True
            return original_exists(self)

        with patch.object(sys, "prefix", "/usr/local"), \
             patch("pathlib.Path.exists", _mocked_exists), \
             patch.dict(os.environ, {"NIA_DOCKER": "1"}):
            # Also need to ensure the editable check doesn't find .git.
            # The real repo has .git, so we need to prevent niaharness import
            # from finding it. Patch the niaharness module location.
            with patch.dict(sys.modules, {"niaharness": MagicMock(__file__="/nonexistent/pkg/__init__.py")}):
                assert detect_install_method() == "docker"


class TestGetCurrentVersion:
    def test_returns_string(self):
        version = get_current_version()
        assert isinstance(version, str)
        # Should be a valid semver or "0.0.0" / "unknown".
        assert len(version) > 0


# ---------------------------------------------------------------------------
# Update: check_for_update
# ---------------------------------------------------------------------------


class TestCheckForUpdate:
    def test_returns_tuple(self):
        current, latest, available = check_for_update()
        assert isinstance(current, str)
        # latest may be empty if network fails.
        assert isinstance(latest, str)
        assert isinstance(available, bool)


# ---------------------------------------------------------------------------
# Update: backup
# ---------------------------------------------------------------------------


class TestCreatePreUpdateBackup:
    def test_creates_backup(self, tmp_path, monkeypatch):
        monkeypatch.setattr("niaharness.prompts.soul.get_nia_home", lambda: tmp_path)
        # Create some files to back up.
        (tmp_path / "SOUL.md").write_text("test")
        (tmp_path / "config.json").write_text("{}")
        backup_path = create_pre_update_backup()
        assert backup_path is not None
        assert Path(backup_path).exists()
        assert backup_path.endswith(".zip")

    def test_returns_none_when_no_home(self, tmp_path, monkeypatch):
        nonexistent = tmp_path / "nonexistent"
        monkeypatch.setattr("niaharness.prompts.soul.get_nia_home", lambda: nonexistent)
        result = create_pre_update_backup()
        assert result is None


class TestPruneOldBackups:
    def test_keeps_only_n_most_recent(self, tmp_path, monkeypatch):
        monkeypatch.setattr("niaharness.prompts.soul.get_nia_home", lambda: tmp_path)
        backup_dir = tmp_path / "backups"
        backup_dir.mkdir(parents=True)
        # Create 10 backups.
        import time
        for i in range(10):
            backup_file = backup_dir / f"pre-update-2026-01-{i:02d}.zip"
            backup_file.write_text("test")
            time.sleep(0.01)
        prune_old_backups(keep=5)
        remaining = list(backup_dir.glob("pre-update-*.zip"))
        assert len(remaining) == 5


# ---------------------------------------------------------------------------
# run_update
# ---------------------------------------------------------------------------


class TestRunUpdate:
    def test_check_mode_returns_result(self):
        result = run_update(check=True)
        assert isinstance(result, UpdateResult)
        assert "NIA Update" in result.report
        assert result.current_version

    def test_check_mode_does_not_backup(self):
        result = run_update(check=True)
        assert result.backup_path is None  # No backup in check mode.

    def test_report_contains_install_method(self):
        result = run_update(check=True)
        assert "Install method:" in result.report


# ---------------------------------------------------------------------------
# UpdateResult dataclass
# ---------------------------------------------------------------------------


class TestUpdateResult:
    def test_defaults(self):
        result = UpdateResult()
        assert result.success is False
        assert result.current_version == ""
        assert result.update_available is False
        assert result.needs_restart is False
        assert result.errors == []


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
