"""Tests for Phase 4 ecosystem tasks (13-15).

Covers:
  Task 13: GitHubAuth (4-tier resolution), GitHubSource (search, fetch, trust, rate limit),
           SkillBundle/SkillMeta dataclasses, get_default_sources
  Task 14: validate_alias_name, check_alias_collision, create_wrapper_script,
           remove_wrapper_script, find_alias_for_profile, clone_profile_files, ProfileInfo
  Task 15: detect_install_method, get_current_version, run_update (--check),
           create_pre_update_backup, prune_old_backups
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ===========================================================================
# Task 13: GitHub Skill Source Adapter
# ===========================================================================


class TestGitHubAuth:
    def test_anonymous_when_no_credentials(self):
        from niaharness.tools.skills_hub_sources import GitHubAuth
        with patch.dict(os.environ, {}, clear=False):
            for var in ("GITHUB_TOKEN", "GH_TOKEN", "GITHUB_APP_ID"):
                os.environ.pop(var, None)
            auth = GitHubAuth()
            assert auth.is_authenticated() is False
            assert auth.auth_method() == "anonymous"
            headers = auth.get_headers()
            assert "Authorization" not in headers

    def test_pat_from_env(self):
        from niaharness.tools.skills_hub_sources import GitHubAuth
        with patch.dict(os.environ, {"GITHUB_TOKEN": "ghp_test123"}):
            auth = GitHubAuth()
            assert auth.is_authenticated() is True
            assert auth.auth_method() == "pat"
            headers = auth.get_headers()
            assert "token ghp_test123" in headers["Authorization"]

    def test_gh_token_env_alias(self):
        from niaharness.tools.skills_hub_sources import GitHubAuth
        with patch.dict(os.environ, {"GH_TOKEN": "ghp_alias456"}, clear=False):
            os.environ.pop("GITHUB_TOKEN", None)
            auth = GitHubAuth()
            assert auth.auth_method() == "pat"

    def test_gh_cli_fallback(self):
        from niaharness.tools.skills_hub_sources import GitHubAuth
        with patch.dict(os.environ, {}, clear=False):
            for var in ("GITHUB_TOKEN", "GH_TOKEN"):
                os.environ.pop(var, None)
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stdout="ghp_cli_token\n")
                auth = GitHubAuth()
                assert auth.is_authenticated() is True
                assert auth.auth_method() == "gh-cli"

    def test_token_cached(self):
        from niaharness.tools.skills_hub_sources import GitHubAuth
        with patch.dict(os.environ, {"GITHUB_TOKEN": "ghp_cached"}):
            auth = GitHubAuth()
            auth._resolve_token()
            # Remove env var — should still return cached token.
            del os.environ["GITHUB_TOKEN"]
            assert auth.is_authenticated() is True


class TestGitHubSource:
    def test_source_id(self):
        from niaharness.tools.skills_hub_sources import GitHubSource
        source = GitHubSource()
        assert source.source_id() == "github"

    def test_default_taps(self):
        from niaharness.tools.skills_hub_sources import GitHubSource
        source = GitHubSource()
        assert len(source.taps) == 6
        repos = [t["repo"] for t in source.taps]
        assert "openai/skills" in repos
        assert "anthropics/skills" in repos

    def test_extra_taps(self):
        from niaharness.tools.skills_hub_sources import GitHubSource
        source = GitHubSource(extra_taps=[{"repo": "my/repo", "path": "skills/"}])
        assert len(source.taps) == 7
        assert source.taps[-1]["repo"] == "my/repo"

    def test_trust_level_trusted(self):
        from niaharness.tools.skills_hub_sources import GitHubSource
        source = GitHubSource()
        assert source.trust_level_for("anthropics/skills/skill-name") == "trusted"
        assert source.trust_level_for("openai/skills/skill-name") == "trusted"
        assert source.trust_level_for("nvidia/skills/skill-name") == "trusted"

    def test_trust_level_community(self):
        from niaharness.tools.skills_hub_sources import GitHubSource
        source = GitHubSource()
        assert source.trust_level_for("random/repo/skill-name") == "community"

    def test_trust_level_short_identifier(self):
        from niaharness.tools.skills_hub_sources import GitHubSource
        source = GitHubSource()
        assert source.trust_level_for("just-a-name") == "community"

    def test_fetch_invalid_identifier(self):
        from niaharness.tools.skills_hub_sources import GitHubSource
        source = GitHubSource()
        # Too few parts.
        assert source.fetch("just-a-name") is None
        assert source.fetch("owner/repo") is None

    def test_rate_limited_default_false(self):
        from niaharness.tools.skills_hub_sources import GitHubSource
        source = GitHubSource()
        assert source.is_rate_limited is False


class TestSkillDataclasses:
    def test_skill_meta(self):
        from niaharness.tools.skills_hub_sources import SkillMeta
        meta = SkillMeta(
            name="test", description="Test skill", source="github",
            identifier="owner/repo/test", trust_level="community",
        )
        assert meta.name == "test"
        assert meta.tags == []
        assert meta.extra == {}

    def test_skill_bundle(self):
        from niaharness.tools.skills_hub_sources import SkillBundle
        bundle = SkillBundle(
            name="test", files={"SKILL.md": "content"},
            source="github", identifier="owner/repo/test", trust_level="trusted",
        )
        assert bundle.name == "test"
        assert "SKILL.md" in bundle.files
        assert bundle.metadata == {}


class TestGetDefaultSources:
    def test_returns_github_source(self):
        from niaharness.tools.skills_hub_sources import get_default_sources, GitHubSource
        sources = get_default_sources()
        assert len(sources) >= 1
        assert isinstance(sources[0], GitHubSource)


# ===========================================================================
# Task 14: Profile Aliases
# ===========================================================================


class TestValidateAliasName:
    def test_valid_name(self):
        from niaharness.profiles.aliases import validate_alias_name
        validate_alias_name("coder")  # Should not raise.
        validate_alias_name("work-123")
        validate_alias_name("a")

    def test_rejects_traversal(self):
        from niaharness.profiles.aliases import validate_alias_name
        with pytest.raises(ValueError):
            validate_alias_name("../../bad")
        with pytest.raises(ValueError):
            validate_alias_name("a/b")

    def test_rejects_uppercase(self):
        from niaharness.profiles.aliases import validate_alias_name
        with pytest.raises(ValueError):
            validate_alias_name("CamelCase")

    def test_rejects_empty(self):
        from niaharness.profiles.aliases import validate_alias_name
        with pytest.raises(ValueError):
            validate_alias_name("")

    def test_rejects_too_long(self):
        from niaharness.profiles.aliases import validate_alias_name
        with pytest.raises(ValueError):
            validate_alias_name("a" * 65)

    def test_allows_dashes_and_underscores(self):
        from niaharness.profiles.aliases import validate_alias_name
        validate_alias_name("my-profile_1")


class TestCheckAliasCollision:
    def test_reserved_name(self):
        from niaharness.profiles.aliases import check_alias_collision
        assert check_alias_collision("nia") is not None
        assert check_alias_collision("default") is not None

    def test_subcommand_conflict(self):
        from niaharness.profiles.aliases import check_alias_collision
        assert check_alias_collision("doctor") is not None
        assert check_alias_collision("gateway") is not None

    def test_safe_name(self):
        from niaharness.profiles.aliases import check_alias_collision
        assert check_alias_collision("coder123") is None
        assert check_alias_collision("my-profile") is None

    def test_invalid_name(self):
        from niaharness.profiles.aliases import check_alias_collision
        assert check_alias_collision("../../bad") is not None


class TestCreateWrapperScript:
    def test_creates_wrapper(self, tmp_path, monkeypatch):
        from niaharness.profiles.aliases import create_wrapper_script
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        path = create_wrapper_script("coder")
        assert path is not None
        assert path.exists()
        content = path.read_text()
        assert "nia -p coder" in content
        assert path.stat().st_mode & 0o100  # Executable.

    def test_wrapper_with_target(self, tmp_path, monkeypatch):
        from niaharness.profiles.aliases import create_wrapper_script
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        path = create_wrapper_script("alias1", target="real-profile")
        content = path.read_text()
        assert "nia -p real-profile" in content

    def test_invalid_name_returns_none(self, tmp_path, monkeypatch):
        from niaharness.profiles.aliases import create_wrapper_script
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        with pytest.raises(ValueError):
            create_wrapper_script("../../bad")


class TestRemoveWrapperScript:
    def test_removes_existing(self, tmp_path, monkeypatch):
        from niaharness.profiles.aliases import create_wrapper_script, remove_wrapper_script
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        create_wrapper_script("coder")
        assert remove_wrapper_script("coder") is True
        # File should be gone.
        wrapper_path = tmp_path / ".local" / "bin" / "coder"
        assert not wrapper_path.exists()

    def test_returns_false_for_nonexistent(self, tmp_path, monkeypatch):
        from niaharness.profiles.aliases import remove_wrapper_script
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        assert remove_wrapper_script("nonexistent") is False

    def test_returns_false_for_non_wrapper(self, tmp_path, monkeypatch):
        from niaharness.profiles.aliases import remove_wrapper_script
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        # Create a non-wrapper file.
        wrapper_dir = tmp_path / ".local" / "bin"
        wrapper_dir.mkdir(parents=True)
        (wrapper_dir / "other").write_text("#!/bin/sh\necho hello\n")
        assert remove_wrapper_script("other") is False  # Not our wrapper.


class TestFindAliasForProfile:
    def test_finds_alias(self, tmp_path, monkeypatch):
        from niaharness.profiles.aliases import create_wrapper_script, find_alias_for_profile
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        create_wrapper_script("coder", target="work-profile")
        alias = find_alias_for_profile("work-profile")
        assert alias == "coder"

    def test_no_alias_returns_none(self, tmp_path, monkeypatch):
        from niaharness.profiles.aliases import find_alias_for_profile
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        assert find_alias_for_profile("nonexistent-profile") is None


class TestCloneProfileFiles:
    def test_light_clone(self, tmp_path):
        from niaharness.profiles.aliases import clone_profile_files
        source = tmp_path / "source"
        target = tmp_path / "target"
        source.mkdir()
        (source / "config.yaml").write_text("model: test")
        (source / ".env").write_text("KEY=value")
        (source / "SOUL.md").write_text("# Soul")
        (source / "memories").mkdir()
        (source / "memories" / "MEMORY.md").write_text("# Memory")

        count = clone_profile_files(source, target)
        assert count == 4
        assert (target / "config.yaml").exists()
        assert (target / ".env").exists()
        assert (target / "SOUL.md").exists()
        assert (target / "memories" / "MEMORY.md").exists()

    def test_env_permissions(self, tmp_path):
        from niaharness.profiles.aliases import clone_profile_files
        import stat
        source = tmp_path / "source"
        target = tmp_path / "target"
        source.mkdir()
        (source / ".env").write_text("KEY=value")
        clone_profile_files(source, target)
        env_mode = (target / ".env").stat().st_mode
        assert env_mode & 0o077 == 0  # Owner-only (0o600).

    def test_missing_files_skipped(self, tmp_path):
        from niaharness.profiles.aliases import clone_profile_files
        source = tmp_path / "source"
        target = tmp_path / "target"
        source.mkdir()
        # Only create one file.
        (source / "SOUL.md").write_text("# Soul")
        count = clone_profile_files(source, target)
        assert count == 1


class TestProfileInfo:
    def test_dataclass_fields(self):
        from niaharness.profiles.aliases import ProfileInfo
        info = ProfileInfo(name="test", path=Path("/tmp/test"), is_default=False)
        assert info.name == "test"
        assert info.gateway_running is False
        assert info.model is None
        assert info.skill_count == 0

    def test_get_profile_info(self, tmp_path):
        from niaharness.profiles.aliases import get_profile_info
        # Create a minimal profile.
        (tmp_path / ".env").write_text("KEY=value")
        (tmp_path / "settings.json").write_text(json.dumps({"model": "claude-sonnet-4-6"}))
        info = get_profile_info("test-profile", tmp_path)
        assert info.name == "test-profile"
        assert info.is_default is False
        assert info.has_env is True
        assert info.model == "claude-sonnet-4-6"


# ===========================================================================
# Task 15: Update System (enhanced from Task 10)
# ===========================================================================


class TestDetectInstallMethod:
    def test_returns_valid_string(self):
        from niaharness.cli.update import detect_install_method
        method = detect_install_method()
        assert method in ("uv-tool", "pipx", "venv-pip", "editable", "docker", "pip")

    def test_uv_tool_detection(self):
        from niaharness.cli.update import detect_install_method
        with patch.object(sys, "prefix", "/home/user/.local/share/uv/tools/niaharness"):
            assert detect_install_method() == "uv-tool"

    def test_pipx_detection(self):
        from niaharness.cli.update import detect_install_method
        with patch.object(sys, "prefix", "/home/user/.local/pipx/venvs/niaharness"):
            assert detect_install_method() == "pipx"


class TestGetCurrentVersion:
    def test_returns_string(self):
        from niaharness.cli.update import get_current_version
        version = get_current_version()
        assert isinstance(version, str)
        assert len(version) > 0


class TestRunUpdate:
    def test_check_mode_returns_result(self):
        from niaharness.cli.update import run_update, UpdateResult
        result = run_update(check=True)
        assert isinstance(result, UpdateResult)
        assert "NIA Update" in result.report
        assert result.current_version

    def test_check_mode_no_backup(self):
        from niaharness.cli.update import run_update
        result = run_update(check=True)
        assert result.backup_path is None

    def test_report_contains_install_method(self):
        from niaharness.cli.update import run_update
        result = run_update(check=True)
        assert "Install method:" in result.report


class TestCreatePreUpdateBackup:
    def test_creates_backup(self, tmp_path, monkeypatch):
        from niaharness.cli.update import create_pre_update_backup
        monkeypatch.setattr("niaharness.prompts.soul.get_nia_home", lambda: tmp_path)
        (tmp_path / "SOUL.md").write_text("test")
        (tmp_path / "config.json").write_text("{}")
        backup_path = create_pre_update_backup()
        assert backup_path is not None
        assert Path(backup_path).exists()
        assert backup_path.endswith(".zip")

    def test_returns_none_when_no_home(self, tmp_path, monkeypatch):
        from niaharness.cli.update import create_pre_update_backup
        nonexistent = tmp_path / "nonexistent"
        monkeypatch.setattr("niaharness.prompts.soul.get_nia_home", lambda: nonexistent)
        result = create_pre_update_backup()
        assert result is None


class TestPruneOldBackups:
    def test_keeps_only_n_most_recent(self, tmp_path, monkeypatch):
        from niaharness.cli.update import prune_old_backups
        import time
        monkeypatch.setattr("niaharness.prompts.soul.get_nia_home", lambda: tmp_path)
        backup_dir = tmp_path / "backups"
        backup_dir.mkdir(parents=True)
        for i in range(10):
            f = backup_dir / f"pre-update-2026-01-{i:02d}.zip"
            f.write_text("test")
            time.sleep(0.01)
        prune_old_backups(keep=5)
        remaining = list(backup_dir.glob("pre-update-*.zip"))
        assert len(remaining) == 5


class TestUpdateResult:
    def test_defaults(self):
        from niaharness.cli.update import UpdateResult
        result = UpdateResult()
        assert result.success is False
        assert result.current_version == ""
        assert result.update_available is False
        assert result.needs_restart is False
        assert result.errors == []


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
