"""Tests for NIA-specific tools (memory and context)."""

from __future__ import annotations

import pytest

from niaharness.tools.base import ToolExecutionContext
from niaharness.tools.nia_memory_tool import NiaMemoryInput, NiaMemoryTool
from niaharness.tools.nia_context_tool import NiaContextInput, NiaContextTool

from pathlib import Path


class FakeMemory:
    """Minimal memory stub for testing."""

    def __init__(self) -> None:
        self._facts: list[str] = []
        self._prefs: dict[str, str] = {}

    def search_relevant(self, query: str, limit: int = 5):
        from agents.nia.core.memory import MemoryEntry
        return [
            MemoryEntry(content=f"Fact about {query}", category="fact")
            for _ in range(min(2, limit))
        ]

    def add_fact(self, fact: str) -> None:
        self._facts.append(fact)

    def add_preference(self, key: str, value: str) -> None:
        self._prefs[key] = value

    def get_preferences(self) -> dict[str, str]:
        return dict(self._prefs)

    def get_recent_conversation(self, limit: int = 10):
        from agents.nia.core.memory import MemoryEntry
        return [
            MemoryEntry(content="Hello", category="conversation", metadata={"role": "user"}),
        ]

    def get_stats(self) -> dict:
        return {"short_term_count": 1, "long_term_count": 0, "total_memories": 1}

    def get_context_summary(self) -> str:
        return "1 recent exchange"


class FakeContext:
    """Minimal context stub for testing."""

    def __init__(self) -> None:
        self._user_name = None

    @property
    def time_of_day(self):
        from agents.nia.core.context import TimeOfDay
        return TimeOfDay.MORNING

    @property
    def user_state(self):
        from agents.nia.core.context import UserState
        return UserState.ACTIVE

    def get_full_context(self) -> dict:
        return {"time_of_day": "morning", "session": {"message_count": 5}}

    def set_user_name(self, name: str) -> None:
        self._user_name = name

    class _Environment:
        working_directory = "/tmp/test"
        platform = "posix"
        shell = "/bin/bash"
        python_version = "3.13"
        git_branch = "main"
        project_type = "python"

    _environment = _Environment()


@pytest.mark.asyncio
async def test_memory_tool_search(tmp_path: Path):
    tool = NiaMemoryTool(memory=FakeMemory())
    ctx = ToolExecutionContext(cwd=tmp_path)
    result = await tool.execute(NiaMemoryInput(action="search", query="test"), ctx)
    assert not result.is_error
    assert "Fact about test" in result.output


@pytest.mark.asyncio
async def test_memory_tool_add_fact(tmp_path: Path):
    mem = FakeMemory()
    tool = NiaMemoryTool(memory=mem)
    ctx = ToolExecutionContext(cwd=tmp_path)
    result = await tool.execute(NiaMemoryInput(action="add_fact", fact="The sky is blue"), ctx)
    assert not result.is_error
    assert "The sky is blue" in mem._facts


@pytest.mark.asyncio
async def test_memory_tool_add_preference(tmp_path: Path):
    mem = FakeMemory()
    tool = NiaMemoryTool(memory=mem)
    ctx = ToolExecutionContext(cwd=tmp_path)
    result = await tool.execute(
        NiaMemoryInput(action="add_preference", key="theme", value="dark"), ctx
    )
    assert not result.is_error
    assert mem._prefs["theme"] == "dark"


@pytest.mark.asyncio
async def test_memory_tool_list_preferences(tmp_path: Path):
    mem = FakeMemory()
    mem._prefs = {"theme": "dark", "lang": "en"}
    tool = NiaMemoryTool(memory=mem)
    ctx = ToolExecutionContext(cwd=tmp_path)
    result = await tool.execute(NiaMemoryInput(action="list_preferences"), ctx)
    assert not result.is_error
    assert "theme: dark" in result.output
    assert "lang: en" in result.output


@pytest.mark.asyncio
async def test_memory_tool_no_memory(tmp_path: Path):
    tool = NiaMemoryTool(memory=None)
    ctx = ToolExecutionContext(cwd=tmp_path)
    result = await tool.execute(NiaMemoryInput(action="stats"), ctx)
    assert result.is_error
    assert "not initialized" in result.output


@pytest.mark.asyncio
async def test_context_tool_full(tmp_path: Path):
    tool = NiaContextTool(context=FakeContext())
    ctx = ToolExecutionContext(cwd=tmp_path)
    result = await tool.execute(NiaContextInput(action="full"), ctx)
    assert not result.is_error
    assert "morning" in result.output


@pytest.mark.asyncio
async def test_context_tool_time(tmp_path: Path):
    tool = NiaContextTool(context=FakeContext())
    ctx = ToolExecutionContext(cwd=tmp_path)
    result = await tool.execute(NiaContextInput(action="time"), ctx)
    assert not result.is_error
    assert "morning" in result.output


@pytest.mark.asyncio
async def test_context_tool_set_user_name(tmp_path: Path):
    fake_ctx = FakeContext()
    tool = NiaContextTool(context=fake_ctx)
    ctx = ToolExecutionContext(cwd=tmp_path)
    result = await tool.execute(NiaContextInput(action="set_user_name", user_name="Alice"), ctx)
    assert not result.is_error
    assert fake_ctx._user_name == "Alice"


@pytest.mark.asyncio
async def test_context_tool_no_context(tmp_path: Path):
    tool = NiaContextTool(context=None)
    ctx = ToolExecutionContext(cwd=tmp_path)
    result = await tool.execute(NiaContextInput(action="full"), ctx)
    assert result.is_error
    assert "not initialized" in result.output
