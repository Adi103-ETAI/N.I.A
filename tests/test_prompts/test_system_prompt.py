"""Tests for niaharness.prompts.system_prompt."""

from __future__ import annotations

from pathlib import Path

from niaharness.prompts.environment import EnvironmentInfo
from niaharness.prompts.system_prompt import build_system_prompt


def _make_env(**overrides) -> EnvironmentInfo:
    defaults = dict(
        os_name="Linux",
        os_version="5.15.0",
        platform_machine="x86_64",
        shell="bash",
        cwd="/home/user/project",
        home_dir="/home/user",
        date="2026-04-01",
        python_version="3.10.17",
        is_git_repo=True,
        git_branch="main",
        hostname="testhost",
    )
    defaults.update(overrides)
    return EnvironmentInfo(**defaults)


def test_build_system_prompt_contains_environment():
    env = _make_env()
    prompt = build_system_prompt(env=env, include_soul=False)
    assert "Linux 5.15.0" in prompt
    assert "x86_64" in prompt
    assert "bash" in prompt
    assert "/home/user/project" in prompt
    assert "2026-04-01" in prompt
    assert "3.10.17" in prompt
    assert "branch: main" in prompt


def test_build_system_prompt_no_git():
    env = _make_env(is_git_repo=False, git_branch=None)
    prompt = build_system_prompt(env=env, include_soul=False)
    assert "Git:" not in prompt


def test_build_system_prompt_git_no_branch():
    env = _make_env(is_git_repo=True, git_branch=None)
    prompt = build_system_prompt(env=env, include_soul=False)
    assert "Git: yes" in prompt
    assert "branch:" not in prompt


def test_build_system_prompt_custom_prompt():
    env = _make_env()
    prompt = build_system_prompt(
        custom_prompt="You are a helpful bot.", env=env, include_soul=False
    )
    assert prompt.startswith("You are a helpful bot.")
    assert "Linux 5.15.0" in prompt
    # Base prompt should not appear
    assert "NiaHarness" not in prompt


def test_build_system_prompt_default_includes_base():
    env = _make_env()
    prompt = build_system_prompt(env=env, include_soul=False)
    assert "NiaHarness" in prompt


# ---------------------------------------------------------------------------
# SOUL.md identity system
# ---------------------------------------------------------------------------


def test_soul_md_seeded_on_first_run(tmp_path: Path, monkeypatch):
    """First call to load_soul_md should seed ~/.nia/SOUL.md with the default."""
    monkeypatch.setenv("NIA_HOME", str(tmp_path / "nia"))
    from niaharness.prompts.soul import DEFAULT_SOUL_MD, get_soul_md_path, load_soul_md

    soul_path = get_soul_md_path()
    assert not soul_path.exists()

    content = load_soul_md()
    assert soul_path.exists()  # seeded
    assert content == DEFAULT_SOUL_MD.strip()
    assert "N.I.A" in content
    assert "JARVIS" in content


def test_soul_md_user_override_not_overwritten(tmp_path: Path, monkeypatch):
    """An existing user SOUL.md must never be overwritten."""
    monkeypatch.setenv("NIA_HOME", str(tmp_path / "nia"))
    from niaharness.prompts.soul import get_soul_md_path, load_soul_md

    soul_path = get_soul_md_path()
    soul_path.parent.mkdir(parents=True, exist_ok=True)
    soul_path.write_text("# My custom NIA\nYou are a pirate.", encoding="utf-8")

    content = load_soul_md()
    assert content == "# My custom NIA\nYou are a pirate."
    # File on disk unchanged
    assert soul_path.read_text(encoding="utf-8") == "# My custom NIA\nYou are a pirate."


def test_soul_md_empty_falls_back_to_default(tmp_path: Path, monkeypatch):
    """An empty SOUL.md falls back to the default (without overwriting the file)."""
    monkeypatch.setenv("NIA_HOME", str(tmp_path / "nia"))
    from niaharness.prompts.soul import DEFAULT_SOUL_MD, get_soul_md_path, load_soul_md

    soul_path = get_soul_md_path()
    soul_path.parent.mkdir(parents=True, exist_ok=True)
    soul_path.write_text("   \n\n  \n", encoding="utf-8")  # whitespace only

    content = load_soul_md()
    assert content == DEFAULT_SOUL_MD.strip()
    # File on disk still has the whitespace (we don't overwrite empty files
    # — user may be mid-edit).
    assert soul_path.read_text(encoding="utf-8").strip() == ""


def test_build_system_prompt_includes_soul(tmp_path: Path, monkeypatch):
    """build_system_prompt prepends SOUL.md as slot #1."""
    monkeypatch.setenv("NIA_HOME", str(tmp_path / "nia"))
    env = _make_env()
    prompt = build_system_prompt(env=env, include_soul=True)
    # SOUL.md content should appear before the base prompt
    soul_idx = prompt.find("N.I.A (Neural Intelligence Assistant)")
    base_idx = prompt.find("NiaHarness")
    assert soul_idx >= 0, "SOUL.md content not in prompt"
    assert base_idx >= 0, "base prompt not in prompt"
    assert soul_idx < base_idx, "SOUL.md should come before base prompt"


def test_build_system_prompt_include_soul_false_skips_soul(tmp_path: Path, monkeypatch):
    """include_soul=False should skip SOUL.md loading entirely."""
    monkeypatch.setenv("NIA_HOME", str(tmp_path / "nia"))
    env = _make_env()
    prompt = build_system_prompt(env=env, include_soul=False)
    # SOUL.md identity content should NOT appear (only the base prompt's
    # "NiaHarness" identity should).
    assert "JARVIS" not in prompt
    assert "NiaHarness" in prompt
