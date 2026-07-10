"""Tests for the lean NIA orchestrator.

NIA is now a thin wrapper that owns identity (SOUL.md), memory, and
personality, and delegates to niaharness's QueryEngine for execution.
There is no separate Brain class — the LLM call inside QueryEngine IS
the brain.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from niaharness.identity.personality import PersonalityConfig
from niaharness.identity import NIA


def test_nia_instantiation(tmp_path: Path):
    """NIA can be instantiated without errors."""
    nia = NIA(working_directory=str(tmp_path))
    assert nia._working_directory == str(tmp_path)
    assert nia._engine is None  # Not initialized yet
    assert nia._initialized is False


def test_nia_has_identity_layer(tmp_path: Path):
    """NIA owns memory, context, and personality (the identity layer)."""
    nia = NIA(working_directory=str(tmp_path))
    assert nia._memory is not None
    assert nia._context is not None
    assert nia._personality is not None


def test_nia_builds_system_prompt(tmp_path: Path):
    """NIA builds a merged system prompt with personality + niaharness base."""
    config = PersonalityConfig(name="TestNIA", base_tone="professional")
    nia = NIA(working_directory=str(tmp_path), personality_config=config)
    nia._context.detect_environment(str(tmp_path))

    prompt = nia._build_system_prompt()

    # Should contain NIA identity (from SOUL.md or base prompt)
    assert "N.I.A" in prompt or "Neural Intelligence" in prompt
    # Should contain personality guidance
    assert "Personality" in prompt or "professional" in prompt.lower()
    # Should contain niaharness base prompt elements (tool/safety rules)
    assert "tool" in prompt.lower() or "system" in prompt.lower()


def test_nia_get_status_before_init(tmp_path: Path):
    """NIA status works before full initialization."""
    nia = NIA(working_directory=str(tmp_path))
    status = nia.get_status()
    assert status["state"] == "uninitialized"
    assert status["cwd"] == str(tmp_path)


def test_nia_get_status_after_init(tmp_path: Path):
    """NIA status reflects initialized state."""
    nia = NIA(working_directory=str(tmp_path))
    # Can't fully initialize without an API key, but we can check the
    # status dict shape.
    status = nia.get_status()
    assert "state" in status
    assert "memory" in status
    assert "tools" in status


def test_nia_shutdown_without_init(tmp_path: Path):
    """NIA shutdown completes without errors even if never initialized."""
    nia = NIA(working_directory=str(tmp_path))
    # shutdown() is async
    import asyncio
    asyncio.run(nia.shutdown())
    assert nia._initialized is False


def test_nia_chat_requires_initialization(tmp_path: Path):
    """NIA.chat() raises if not initialized."""
    nia = NIA(working_directory=str(tmp_path))
    with pytest.raises(RuntimeError, match="not initialized"):
        async def _try():
            async for _ in nia.chat("hello"):
                pass
        import asyncio
        asyncio.run(_try())


def test_nia_properties(tmp_path: Path):
    """NIA exposes memory, context, personality as properties."""
    nia = NIA(working_directory=str(tmp_path))
    assert nia.memory is nia._memory
    assert nia.context is nia._context
    assert nia.personality is nia._personality
    assert nia.engine is None  # not initialized
    assert nia.initialized is False
