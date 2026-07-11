"""Tests for the P1 Doctor + Profiles extensions.

Covers:
  - Doctor extensions: version consistency, gateway linger, tool availability,
    skills hub, memory provider, profiles, required packages, command
    installation, config structure, xAI model retirement.
  - Profile extensions: export, import, rename, seed skills, backfill envs,
    profiles_to_serve, profile metadata YAML, distribution metadata.
"""

from __future__ import annotations

import json
import os
import tarfile
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from niaharness.cli.doctor_extensions import (
    EXTENSION_SECTIONS,
    FAIL,
    INFO,
    OK,
    WARN,
    check_command_installation,
    check_config_structure,
    check_gateway_service_linger,
    check_memory_provider,
    check_profiles,
    check_required_packages,
    check_skills_hub,
    check_tool_availability,
    check_version_consistency,
    check_xai_model_retirement,
    run_extension_checks,
)
from niaharness.profiles.extensions import (
    backfill_profile_envs,
    export_profile,
    get_distribution_meta,
    has_bundled_skills_opt_out,
    import_profile,
    profiles_to_serve,
    read_profile_meta,
    rename_profile,
    set_distribution_meta,
    write_profile_meta,
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
# Doctor extensions
# ---------------------------------------------------------------------------


class TestDoctorExtensions:
    def test_extension_sections_count(self):
        assert len(EXTENSION_SECTIONS) == 10

    def test_check_version_consistency_returns_list(self):
        results = check_version_consistency()
        assert isinstance(results, list)
        assert len(results) >= 1

    def test_check_gateway_service_linger_returns_list(self):
        results = check_gateway_service_linger()
        assert isinstance(results, list)
        assert len(results) >= 1

    def test_check_tool_availability(self):
        results = check_tool_availability()
        assert isinstance(results, list)
        assert any(r[0] == OK for r in results)  # at least one OK

    def test_check_skills_hub(self):
        results = check_skills_hub()
        assert isinstance(results, list)

    def test_check_memory_provider(self):
        results = check_memory_provider()
        assert isinstance(results, list)

    def test_check_profiles(self):
        results = check_profiles()
        assert isinstance(results, list)

    def test_check_required_packages(self):
        results = check_required_packages()
        assert isinstance(results, list)
        # Should find at least some packages.
        statuses = [r[0] for r in results]
        assert OK in statuses or WARN in statuses

    def test_check_command_installation(self):
        results = check_command_installation()
        assert isinstance(results, list)

    def test_check_config_structure_no_config(self):
        results = check_config_structure()
        assert isinstance(results, list)
        # No config.yaml → INFO.
        assert any(r[0] == INFO for r in results)

    def test_check_config_structure_with_config(self, tmp_path: Path):
        # Write a config.yaml.
        nia_home = Path(os.environ["NIA_HOME"])
        nia_home.mkdir(parents=True, exist_ok=True)
        config_path = nia_home / "config.yaml"
        config_path.write_text(
            "model:\n  provider: anthropic\n  default: claude-3-haiku\n"
            "permissions:\n  mode: default\n",
            encoding="utf-8",
        )
        results = check_config_structure()
        assert isinstance(results, list)
        assert any(r[0] == OK and "provider" in r[1] for r in results)

    def test_check_xai_model_retirement_no_config(self):
        results = check_xai_model_retirement()
        assert isinstance(results, list)
        assert len(results) == 0  # no config → no results

    def test_check_xai_model_retirement_retired_model(self, tmp_path: Path):
        nia_home = Path(os.environ["NIA_HOME"])
        nia_home.mkdir(parents=True, exist_ok=True)
        config_path = nia_home / "config.yaml"
        config_path.write_text(
            "model:\n  provider: xai\n  default: grok-2\n",
            encoding="utf-8",
        )
        results = check_xai_model_retirement()
        assert any(r[0] == WARN and "retired" in r[1] for r in results)

    def test_check_xai_model_retirement_current_model(self, tmp_path: Path):
        nia_home = Path(os.environ["NIA_HOME"])
        nia_home.mkdir(parents=True, exist_ok=True)
        config_path = nia_home / "config.yaml"
        config_path.write_text(
            "model:\n  provider: xai\n  default: grok-3\n",
            encoding="utf-8",
        )
        results = check_xai_model_retirement()
        assert any(r[0] == OK and "current" in r[1] for r in results)

    def test_run_extension_checks(self):
        results = run_extension_checks()
        assert isinstance(results, list)
        assert len(results) >= 10  # at least one result per check


# ---------------------------------------------------------------------------
# Profile extensions — export + import
# ---------------------------------------------------------------------------


class TestProfileExportImport:
    def test_export_default_profile(self, tmp_path: Path):
        # Create a minimal NIA home.
        nia_home = Path(os.environ["NIA_HOME"])
        nia_home.mkdir(parents=True, exist_ok=True)
        (nia_home / "config.yaml").write_text("model:\n  provider: anthropic\n", encoding="utf-8")
        (nia_home / ".env").write_text("ANTHROPIC_API_KEY=secret", encoding="utf-8")
        (nia_home / "SOUL.md").write_text("# My identity\n", encoding="utf-8")

        output = tmp_path / "default-export.tar.gz"
        result = export_profile("default", str(output))
        assert result.exists()
        assert result.suffix == ".gz"

        # Verify .env was excluded.
        with tarfile.open(result, "r:gz") as tf:
            names = tf.getnames()
            assert "default/config.yaml" in names
            assert "default/SOUL.md" in names
            assert "default/.env" not in names  # credential excluded

    def test_export_named_profile(self, tmp_path: Path):
        # Create a named profile.
        nia_home = Path(os.environ["NIA_HOME"])
        profiles_root = nia_home / "profiles"
        profile_dir = profiles_root / "test-profile"
        profile_dir.mkdir(parents=True)
        (profile_dir / "config.yaml").write_text("test: true\n", encoding="utf-8")
        (profile_dir / ".env").write_text("KEY=val\n", encoding="utf-8")

        output = tmp_path / "test-profile.tar.gz"
        result = export_profile("test-profile", str(output))
        assert result.exists()

        with tarfile.open(result, "r:gz") as tf:
            names = tf.getnames()
            assert "test-profile/config.yaml" in names
            assert "test-profile/.env" not in names

    def test_export_nonexistent_profile(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            export_profile("nonexistent", str(tmp_path / "out.tar.gz"))

    def test_import_profile(self, tmp_path: Path):
        # Create a profile to export.
        nia_home = Path(os.environ["NIA_HOME"])
        profiles_root = nia_home / "profiles"
        source_dir = profiles_root / "source-profile"
        source_dir.mkdir(parents=True)
        (source_dir / "config.yaml").write_text("test: true\n", encoding="utf-8")
        (source_dir / "SKILL.md").write_text("# Test skill\n", encoding="utf-8")

        # Export it.
        archive = tmp_path / "source-profile.tar.gz"
        export_profile("source-profile", str(archive))

        # Import it under a new name.
        result = import_profile(str(archive), name="imported-profile")
        assert result.exists()
        assert result.name == "imported-profile"
        assert (result / "config.yaml").exists()
        assert (result / "SKILL.md").exists()

    def test_import_profile_rejects_default_name(self, tmp_path: Path):
        # Create a minimal archive.
        source = tmp_path / "staged"
        (source / "default").mkdir(parents=True)
        (source / "default" / "config.yaml").write_text("test: true\n")
        archive = tmp_path / "default.tar.gz"
        with tarfile.open(archive, "w:gz") as tf:
            tf.add(str(source / "default"), "default")

        with pytest.raises(ValueError, match="Cannot import as 'default'"):
            import_profile(str(archive))


# ---------------------------------------------------------------------------
# Profile extensions — rename
# ---------------------------------------------------------------------------


class TestProfileRename:
    def test_rename_profile(self, tmp_path: Path):
        nia_home = Path(os.environ["NIA_HOME"])
        profiles_root = nia_home / "profiles"
        old_dir = profiles_root / "old-name"
        old_dir.mkdir(parents=True)
        (old_dir / "config.yaml").write_text("test: true\n")

        new_dir = rename_profile("old-name", "new-name")
        assert new_dir.exists()
        assert new_dir.name == "new-name"
        assert not old_dir.exists()

    def test_rename_profile_rejects_default(self):
        with pytest.raises(ValueError, match="Cannot rename the default"):
            rename_profile("default", "other")

    def test_rename_profile_rejects_to_default(self):
        with pytest.raises(ValueError, match="Cannot rename to 'default'"):
            rename_profile("some-profile", "default")

    def test_rename_nonexistent(self):
        with pytest.raises(FileNotFoundError):
            rename_profile("nonexistent", "other")

    def test_rename_to_existing(self, tmp_path: Path):
        nia_home = Path(os.environ["NIA_HOME"])
        profiles_root = nia_home / "profiles"
        for name in ("profile-a", "profile-b"):
            d = profiles_root / name
            d.mkdir(parents=True)
            (d / "config.yaml").write_text("test: true\n")

        with pytest.raises(FileExistsError):
            rename_profile("profile-a", "profile-b")


# ---------------------------------------------------------------------------
# Profile extensions — metadata YAML
# ---------------------------------------------------------------------------


class TestProfileMetadata:
    def test_read_profile_meta_no_file(self, tmp_path: Path):
        result = read_profile_meta(tmp_path)
        assert result["description"] == ""
        assert result["description_auto"] is False

    def test_write_and_read_meta(self, tmp_path: Path):
        write_profile_meta(tmp_path, description="Test profile", description_auto=True)
        meta = read_profile_meta(tmp_path)
        assert meta["description"] == "Test profile"
        assert meta["description_auto"] is True

    def test_write_meta_partial_update(self, tmp_path: Path):
        write_profile_meta(tmp_path, description="Original")
        write_profile_meta(tmp_path, description_auto=True)
        meta = read_profile_meta(tmp_path)
        assert meta["description"] == "Original"  # preserved
        assert meta["description_auto"] is True  # updated

    def test_write_meta_nonexistent_dir(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            write_profile_meta(tmp_path / "nonexistent")


# ---------------------------------------------------------------------------
# Profile extensions — distribution metadata
# ---------------------------------------------------------------------------


class TestDistributionMetadata:
    def test_get_distribution_meta_none(self, tmp_path: Path):
        assert get_distribution_meta(tmp_path) is None

    def test_set_and_get_distribution_meta(self, tmp_path: Path):
        set_distribution_meta(tmp_path, source="export", exported_at=1234567890)
        meta = get_distribution_meta(tmp_path)
        assert meta is not None
        assert meta["source"] == "export"
        assert meta["exported_at"] == 1234567890

    def test_distribution_meta_permissions(self, tmp_path: Path):
        set_distribution_meta(tmp_path, source="test")
        meta_path = tmp_path / ".distribution.json"
        assert meta_path.exists()
        # File should be 0600 (owner-only).
        if os.name == "posix":
            mode = oct(meta_path.stat().st_mode)[-3:]
            assert mode == "600"


# ---------------------------------------------------------------------------
# Profile extensions — backfill envs
# ---------------------------------------------------------------------------


class TestBackfillEnvs:
    def test_backfill_empty(self, tmp_path: Path):
        # No profiles → empty list.
        result = backfill_profile_envs()
        assert result == []

    def test_backfill_creates_env(self, tmp_path: Path):
        nia_home = Path(os.environ["NIA_HOME"])
        # Create default .env.
        nia_home.mkdir(parents=True, exist_ok=True)
        (nia_home / ".env").write_text("ANTHROPIC_API_KEY=test\n", encoding="utf-8")

        # Create a named profile without .env.
        profiles_root = nia_home / "profiles"
        profile_dir = profiles_root / "test-profile"
        profile_dir.mkdir(parents=True)

        backfilled = backfill_profile_envs()
        assert "test-profile" in backfilled
        env_path = profile_dir / ".env"
        assert env_path.exists()
        assert "ANTHROPIC_API_KEY=test" in env_path.read_text()

    def test_backfill_skips_existing(self, tmp_path: Path):
        nia_home = Path(os.environ["NIA_HOME"])
        nia_home.mkdir(parents=True, exist_ok=True)
        (nia_home / ".env").write_text("KEY=default\n", encoding="utf-8")

        profiles_root = nia_home / "profiles"
        profile_dir = profiles_root / "has-env"
        profile_dir.mkdir(parents=True)
        (profile_dir / ".env").write_text("KEY=custom\n", encoding="utf-8")

        backfilled = backfill_profile_envs()
        assert "has-env" not in backfilled  # already has .env
        assert (profile_dir / ".env").read_text() == "KEY=custom\n"  # not overwritten


# ---------------------------------------------------------------------------
# Profile extensions — profiles_to_serve
# ---------------------------------------------------------------------------


class TestProfilesToServe:
    def test_single_mode_returns_active(self):
        result = profiles_to_serve(multiplex=False)
        assert len(result) == 1
        name, path = result[0]
        assert name == "default"  # default is the initial active profile

    def test_multiplex_mode_includes_named_profiles(self, tmp_path: Path):
        nia_home = Path(os.environ["NIA_HOME"])
        profiles_root = nia_home / "profiles"
        for name in ("alpha", "beta"):
            d = profiles_root / name
            d.mkdir(parents=True)

        result = profiles_to_serve(multiplex=True)
        names = [r[0] for r in result]
        assert "default" in names
        assert "alpha" in names
        assert "beta" in names


# ---------------------------------------------------------------------------
# Profile extensions — has_bundled_skills_opt_out
# ---------------------------------------------------------------------------


class TestBundledSkillsOptOut:
    def test_no_opt_out(self, tmp_path: Path):
        assert has_bundled_skills_opt_out(tmp_path) is False

    def test_opt_out_marker(self, tmp_path: Path):
        (tmp_path / ".no-bundled-skills").touch()
        assert has_bundled_skills_opt_out(tmp_path) is True


# ---------------------------------------------------------------------------
# Integration: run_doctor with extensions
# ---------------------------------------------------------------------------


class TestDoctorIntegration:
    def test_run_doctor_includes_extensions(self):
        from niaharness.cli.doctor import run_doctor
        result = run_doctor(fix=False)
        # The report should include extension section titles.
        assert "Version Consistency" in result.report or "Version" in result.report
        assert "Tool Availability" in result.report or "tools" in result.report.lower()


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
