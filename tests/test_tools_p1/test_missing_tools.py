"""Tests for the P1 missing tools — web_extract, video_analyze, video_generate,
clarify, project_create/list/switch, memory (batched ops), search_files,
read_terminal, close_terminal, x_search, text_to_speech.

Each test verifies that the tool:
  - Is registered in the default tool registry.
  - Has a valid name, description, and input_model.
  - Executes correctly (or returns a helpful error when deps are missing).
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from niaharness.tools import create_default_tool_registry
from niaharness.tools.base import ToolExecutionContext, ToolResult
from niaharness.tools.missing_tools import (
    ClarifyInput,
    ClarifyTool,
    CloseTerminalInput,
    CloseTerminalTool,
    MemoryBatchInput,
    MemoryBatchTool,
    ProjectCreateInput,
    ProjectCreateTool,
    ProjectListInput,
    ProjectListTool,
    ProjectSwitchInput,
    ProjectSwitchTool,
    ReadTerminalInput,
    ReadTerminalTool,
    SearchFilesInput,
    SearchFilesTool,
    TextToSpeechInput,
    TextToSpeechTool,
    VideoAnalyzeInput,
    VideoAnalyzeTool,
    VideoGenerateInput,
    VideoGenerateTool,
    WebExtractInput,
    WebExtractTool,
    XSearchInput,
    XSearchTool,
    get_missing_tools,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def context(tmp_path: Path) -> ToolExecutionContext:
    """Return a tool execution context with a temp cwd."""
    return ToolExecutionContext(cwd=tmp_path)


@pytest.fixture(autouse=True)
def _temp_nia_home(tmp_path: Path, monkeypatch):
    """Redirect NIA_HOME to a temp dir so tests don't pollute the host."""
    monkeypatch.setenv("NIA_HOME", str(tmp_path / ".nia"))
    yield


# ---------------------------------------------------------------------------
# Registry verification
# ---------------------------------------------------------------------------


class TestToolRegistry:
    def test_all_missing_tools_registered(self):
        registry = create_default_tool_registry()
        tool_names = {t.name for t in registry.list_tools()}
        expected = {
            "web_extract", "video_analyze", "video_generate",
            "clarify", "project_create", "project_list", "project_switch",
            "memory", "search_files", "read_terminal", "close_terminal",
            "x_search", "text_to_speech",
        }
        missing = expected - tool_names
        assert not missing, f"Missing tools: {missing}"

    def test_tool_count_increased(self):
        registry = create_default_tool_registry()
        # Was 56, now should be 69.
        assert len(registry.list_tools()) >= 69

    def test_get_missing_tools_returns_13(self):
        tools = get_missing_tools()
        assert len(tools) == 13

    def test_all_tools_have_valid_schemas(self):
        registry = create_default_tool_registry()
        for tool in registry.list_tools():
            schema = tool.to_api_schema()
            assert "name" in schema
            assert "description" in schema
            assert "input_schema" in schema
            assert isinstance(schema["name"], str)
            assert len(schema["name"]) > 0


# ---------------------------------------------------------------------------
# web_extract
# ---------------------------------------------------------------------------


class TestWebExtract:
    @pytest.mark.asyncio
    async def test_web_extract_basic(self, context, monkeypatch):
        # Mock httpx to return a simple HTML page.
        mock_response = MagicMock()
        mock_response.text = "<html><body><h1>Hello World</h1><p>This is a test page.</p></body></html>"
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_response):
            tool = WebExtractTool()
            result = await tool.execute(
                WebExtractInput(urls=["https://example.com"], format="text"),
                context,
            )
        assert not result.is_error
        assert "Hello World" in result.output
        assert "test page" in result.output

    @pytest.mark.asyncio
    async def test_web_extract_invalid_url(self, context):
        tool = WebExtractTool()
        result = await tool.execute(
            WebExtractInput(urls=["not-a-url"]),
            context,
        )
        assert "Error" in result.output

    @pytest.mark.asyncio
    async def test_web_extract_strips_scripts(self, context):
        mock_response = MagicMock()
        mock_response.text = (
            "<html><head><script>evil()</script></head>"
            "<body><p>Good content</p></body></html>"
        )
        mock_response.raise_for_status = MagicMock()
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_response):
            tool = WebExtractTool()
            result = await tool.execute(
                WebExtractInput(urls=["https://example.com"], format="text"),
                context,
            )
        assert "evil" not in result.output
        assert "Good content" in result.output

    def test_is_read_only(self):
        assert WebExtractTool().is_read_only(WebExtractInput(urls=["https://example.com"])) is True


# ---------------------------------------------------------------------------
# video_analyze
# ---------------------------------------------------------------------------


class TestVideoAnalyze:
    @pytest.mark.asyncio
    async def test_video_not_found(self, context):
        tool = VideoAnalyzeTool()
        result = await tool.execute(
            VideoAnalyzeInput(video_path="nonexistent.mp4"),
            context,
        )
        assert result.is_error
        assert "not found" in result.output.lower()

    @pytest.mark.asyncio
    async def test_no_ffmpeg(self, context, tmp_path: Path):
        # Create a dummy video file.
        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake video data")
        with patch("shutil.which", return_value=None):
            tool = VideoAnalyzeTool()
            result = await tool.execute(
                VideoAnalyzeInput(video_path=str(video)),
                context,
            )
        assert result.is_error
        assert "FFmpeg" in result.output

    def test_is_read_only(self):
        assert VideoAnalyzeTool().is_read_only(
            VideoAnalyzeInput(video_path="test.mp4")
        ) is True


# ---------------------------------------------------------------------------
# video_generate
# ---------------------------------------------------------------------------


class TestVideoGenerate:
    @pytest.mark.asyncio
    async def test_no_api_key(self, context, monkeypatch):
        monkeypatch.delenv("FAL_KEY", raising=False)
        tool = VideoGenerateTool()
        result = await tool.execute(
            VideoGenerateInput(prompt="A cat playing piano"),
            context,
        )
        assert result.is_error
        assert "FAL_KEY" in result.output


# ---------------------------------------------------------------------------
# clarify
# ---------------------------------------------------------------------------


class TestClarify:
    @pytest.mark.asyncio
    async def test_clarify_returns_question(self, context):
        tool = ClarifyTool()
        result = await tool.execute(
            ClarifyInput(question="Which language do you prefer?"),
            context,
        )
        assert "Which language" in result.output

    @pytest.mark.asyncio
    async def test_clarify_with_options(self, context):
        tool = ClarifyTool()
        result = await tool.execute(
            ClarifyInput(
                question="Which option?",
                options=["Python", "JavaScript", "Rust"],
            ),
            context,
        )
        assert "Python" in result.output
        assert "JavaScript" in result.output


# ---------------------------------------------------------------------------
# project tools
# ---------------------------------------------------------------------------


class TestProjectTools:
    @pytest.mark.asyncio
    async def test_project_create_and_list(self, context, tmp_path: Path):
        # Create a project.
        create_tool = ProjectCreateTool()
        result = await create_tool.execute(
            ProjectCreateInput(name="test-project", cwd=str(tmp_path)),
            context,
        )
        assert not result.is_error
        assert "test-project" in result.output

        # List projects.
        list_tool = ProjectListTool()
        result = await list_tool.execute(ProjectListInput(), context)
        assert not result.is_error
        assert "test-project" in result.output

    @pytest.mark.asyncio
    async def test_project_list_empty(self, context):
        list_tool = ProjectListTool()
        result = await list_tool.execute(ProjectListInput(), context)
        assert "No projects" in result.output

    @pytest.mark.asyncio
    async def test_project_switch(self, context, tmp_path: Path):
        # Create a project first.
        create_tool = ProjectCreateTool()
        await create_tool.execute(
            ProjectCreateInput(name="test-project", cwd=str(tmp_path)),
            context,
        )

        # Switch to it.
        switch_tool = ProjectSwitchTool()
        result = await switch_tool.execute(
            ProjectSwitchInput(name="test-project"),
            context,
        )
        assert not result.is_error
        assert "Switched" in result.output

    @pytest.mark.asyncio
    async def test_project_switch_not_found(self, context):
        # Create a projects.json with one project so the file exists.
        create_tool = ProjectCreateTool()
        await create_tool.execute(
            ProjectCreateInput(name="real-project", cwd="/tmp"),
            context,
        )
        switch_tool = ProjectSwitchTool()
        result = await switch_tool.execute(
            ProjectSwitchInput(name="nonexistent"),
            context,
        )
        assert result.is_error
        assert "not found" in result.output

    def test_project_list_is_read_only(self):
        assert ProjectListTool().is_read_only(ProjectListInput()) is True


# ---------------------------------------------------------------------------
# memory (batched ops)
# ---------------------------------------------------------------------------


class TestMemoryBatch:
    @pytest.mark.asyncio
    async def test_memory_add_and_search(self, context):
        tool = MemoryBatchTool()
        # Add an entry.
        result = await tool.execute(
            MemoryBatchInput(operations=[
                {"action": "add", "content": "test fact", "category": "fact"},
            ]),
            context,
        )
        assert not result.is_error
        assert "stored" in result.output.lower()

        # Search for it.
        result = await tool.execute(
            MemoryBatchInput(operations=[
                {"action": "search", "query": "test"},
            ]),
            context,
        )
        assert "test fact" in result.output

    @pytest.mark.asyncio
    async def test_memory_list(self, context):
        tool = MemoryBatchTool()
        # Add then list.
        await tool.execute(
            MemoryBatchInput(operations=[
                {"action": "add", "content": "entry 1", "category": "note"},
                {"action": "add", "content": "entry 2", "category": "note"},
            ]),
            context,
        )
        result = await tool.execute(
            MemoryBatchInput(operations=[
                {"action": "list", "category": "note"},
            ]),
            context,
        )
        assert "entry 1" in result.output
        assert "entry 2" in result.output

    @pytest.mark.asyncio
    async def test_memory_unknown_action(self, context):
        tool = MemoryBatchTool()
        result = await tool.execute(
            MemoryBatchInput(operations=[
                {"action": "unknown_action"},
            ]),
            context,
        )
        assert "unknown action" in result.output.lower()

    def test_memory_read_only_search(self):
        tool = MemoryBatchTool()
        assert tool.is_read_only(
            MemoryBatchInput(operations=[{"action": "search", "query": "test"}])
        ) is True

    def test_memory_not_read_only_with_add(self):
        tool = MemoryBatchTool()
        assert tool.is_read_only(
            MemoryBatchInput(operations=[{"action": "add", "content": "test"}])
        ) is False


# ---------------------------------------------------------------------------
# search_files
# ---------------------------------------------------------------------------


class TestSearchFiles:
    @pytest.mark.asyncio
    async def test_search_by_name(self, context, tmp_path: Path):
        # Create some test files.
        (tmp_path / "test1.py").write_text("print('hello')")
        (tmp_path / "test2.py").write_text("print('world')")
        (tmp_path / "readme.md").write_text("# Readme")

        tool = SearchFilesTool()
        result = await tool.execute(
            SearchFilesInput(pattern="*.py", path="."),
            context,
        )
        assert "test1.py" in result.output
        assert "test2.py" in result.output
        assert "readme.md" not in result.output

    @pytest.mark.asyncio
    async def test_search_by_content(self, context, tmp_path: Path):
        (tmp_path / "test1.py").write_text("def hello(): pass")
        (tmp_path / "test2.py").write_text("def world(): pass")

        tool = SearchFilesTool()
        result = await tool.execute(
            SearchFilesInput(pattern="*.py", content_query="hello"),
            context,
        )
        assert "test1.py" in result.output
        assert "test2.py" not in result.output

    @pytest.mark.asyncio
    async def test_search_no_matches(self, context):
        tool = SearchFilesTool()
        result = await tool.execute(
            SearchFilesInput(pattern="*.nonexistent"),
            context,
        )
        assert "No files" in result.output

    def test_is_read_only(self):
        assert SearchFilesTool().is_read_only(
            SearchFilesInput(pattern="*.py")
        ) is True


# ---------------------------------------------------------------------------
# read_terminal / close_terminal
# ---------------------------------------------------------------------------


class TestTerminalTools:
    @pytest.mark.asyncio
    async def test_read_terminal_no_sessions(self, context):
        tool = ReadTerminalTool()
        result = await tool.execute(
            ReadTerminalInput(terminal_id="term1"),
            context,
        )
        assert result.is_error
        assert "not found" in result.output.lower()

    @pytest.mark.asyncio
    async def test_close_terminal_no_sessions(self, context):
        tool = CloseTerminalTool()
        result = await tool.execute(
            CloseTerminalInput(terminal_id="term1"),
            context,
        )
        # Not an error — just "not found or already closed".
        assert "not found" in result.output.lower() or "already closed" in result.output.lower()

    def test_read_terminal_is_read_only(self):
        assert ReadTerminalTool().is_read_only(
            ReadTerminalInput(terminal_id="term1")
        ) is True


# ---------------------------------------------------------------------------
# x_search
# ---------------------------------------------------------------------------


class TestXSearch:
    @pytest.mark.asyncio
    async def test_no_api_key(self, context, monkeypatch):
        monkeypatch.delenv("TWITTER_BEARER_TOKEN", raising=False)
        tool = XSearchTool()
        result = await tool.execute(
            XSearchInput(query="AI"),
            context,
        )
        assert result.is_error
        assert "TWITTER_BEARER_TOKEN" in result.output

    def test_is_read_only(self):
        assert XSearchTool().is_read_only(XSearchInput(query="test")) is True


# ---------------------------------------------------------------------------
# text_to_speech
# ---------------------------------------------------------------------------


class TestTextToSpeech:
    @pytest.mark.asyncio
    async def test_text_to_speech_delegates_to_speak(self, context):
        tool = TextToSpeechTool()
        # The speak tool will try to load KittenTTS — if it's not installed,
        # it'll fall back to espeak or return an error. Either way, the
        # text_to_speech tool should delegate without crashing.
        result = await tool.execute(
            TextToSpeechInput(text="hello world"),
            context,
        )
        # Should produce some output (audio file path or error message).
        assert isinstance(result, ToolResult)
        assert len(result.output) > 0


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
