"""Tests for the unified NIA class."""

from __future__ import annotations

from pathlib import Path

import pytest

from agents.nia.core.personality import PersonalityConfig
from agents.nia.nia import NIA


def test_nia_instantiation(tmp_path: Path):
    """NIA can be instantiated without errors."""
    nia = NIA(working_directory=str(tmp_path))
    assert nia._working_directory == str(tmp_path)
    assert nia._engine is None  # Not initialized yet
    assert nia._initialized is False


def test_nia_builds_merged_system_prompt(tmp_path: Path):
    """NIA builds a merged system prompt with personality + niaharness base."""
    config = PersonalityConfig(
        name="TestNIA",
        base_tone="professional",
    )
    nia = NIA(working_directory=str(tmp_path), personality_config=config)
    nia._context.detect_environment(str(tmp_path))

    prompt = nia._build_merged_system_prompt()

    # Should contain NIA identity
    assert "N.I.A" in prompt or "Neural Intelligence" in prompt
    # Should contain niaharness base prompt elements
    assert "NiaHarness" in prompt or "tool" in prompt.lower()
    # Should contain personality guidance
    assert "professional" in prompt.lower() or "personality" in prompt.lower()


def test_nia_get_status_before_init(tmp_path: Path):
    """NIA status works before full initialization."""
    nia = NIA(working_directory=str(tmp_path))
    status = nia.get_status()
    assert status["state"] == "ready" or status["state"] == "initializing"
    assert "brain" in status
    assert "memory" in status


def test_nia_shutdown(tmp_path: Path):
    """NIA shutdown completes without errors."""
    nia = NIA(working_directory=str(tmp_path))
    nia.shutdown()
    assert nia._initialized is False or nia._state.system_state.value == "shutdown"
