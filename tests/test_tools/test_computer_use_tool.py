"""Tests for the computer_use tool."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from niaharness.tools.base import ToolExecutionContext
from niaharness.tools.computer_use_tool import ComputerUseTool, ComputerUseInput


@pytest.fixture
def context(tmp_path: Path) -> ToolExecutionContext:
    return ToolExecutionContext(cwd=tmp_path)


# ---------------------------------------------------------------------------
# Schema + read-only flag
# ---------------------------------------------------------------------------


class TestReadOnly:
    def test_capture_is_read_only(self):
        tool = ComputerUseTool()
        assert tool.is_read_only(ComputerUseInput(action="capture")) is True

    def test_list_windows_is_read_only(self):
        tool = ComputerUseTool()
        assert tool.is_read_only(ComputerUseInput(action="list_windows")) is True

    def test_click_is_not_read_only(self):
        tool = ComputerUseTool()
        assert tool.is_read_only(ComputerUseInput(action="click", x=10, y=20)) is False

    def test_type_is_not_read_only(self):
        tool = ComputerUseTool()
        assert tool.is_read_only(ComputerUseInput(action="type", text="hello")) is False


# ---------------------------------------------------------------------------
# Missing pyautogui
# ---------------------------------------------------------------------------


class TestMissingPyAutoGUI:
    @pytest.mark.asyncio
    async def test_no_pyautogui_returns_error(self, context: ToolExecutionContext, monkeypatch: pytest.MonkeyPatch):
        """When pyautogui is not available, return a helpful error."""
        import niaharness.tools.computer_use_tool as mod

        monkeypatch.setattr(mod, "_pyautogui", None)
        monkeypatch.setattr(mod, "_pyautogui_error", "pyautogui is not installed.")

        result = await ComputerUseTool().execute(
            ComputerUseInput(action="capture"),
            context,
        )
        assert result.is_error is True
        assert "pyautogui" in result.output.lower()


# ---------------------------------------------------------------------------
# Mocked actions
# ---------------------------------------------------------------------------


class TestMockedActions:
    @pytest.mark.asyncio
    async def test_capture(self, context: ToolExecutionContext, monkeypatch: pytest.MonkeyPatch):
        """Test that capture takes a screenshot and saves it."""
        import niaharness.tools.computer_use_tool as mod

        mock_pg = MagicMock()
        mock_img = MagicMock()
        mock_img.width = 1920
        mock_img.height = 1080
        mock_img.save = MagicMock(side_effect=lambda path=None, format=None: None)
        mock_pg.screenshot = MagicMock(return_value=mock_img)
        monkeypatch.setattr(mod, "_pyautogui", mock_pg)
        monkeypatch.setattr(mod, "_pyautogui_error", None)

        result = await ComputerUseTool().execute(
            ComputerUseInput(action="capture"),
            context,
        )
        assert result.is_error is False
        assert "1920x1080" in result.output
        assert "screenshot" in result.output.lower()

    @pytest.mark.asyncio
    async def test_click(self, context: ToolExecutionContext, monkeypatch: pytest.MonkeyPatch):
        import niaharness.tools.computer_use_tool as mod

        mock_pg = MagicMock()
        monkeypatch.setattr(mod, "_pyautogui", mock_pg)
        monkeypatch.setattr(mod, "_pyautogui_error", None)

        result = await ComputerUseTool().execute(
            ComputerUseInput(action="click", x=100, y=200),
            context,
        )
        assert result.is_error is False
        assert "(100, 200)" in result.output
        mock_pg.click.assert_called_once_with(100, 200)

    @pytest.mark.asyncio
    async def test_click_missing_coords(self, context: ToolExecutionContext, monkeypatch: pytest.MonkeyPatch):
        import niaharness.tools.computer_use_tool as mod

        mock_pg = MagicMock()
        monkeypatch.setattr(mod, "_pyautogui", mock_pg)
        monkeypatch.setattr(mod, "_pyautogui_error", None)

        result = await ComputerUseTool().execute(
            ComputerUseInput(action="click"),
            context,
        )
        assert result.is_error is True
        assert "x and y" in result.output

    @pytest.mark.asyncio
    async def test_type(self, context: ToolExecutionContext, monkeypatch: pytest.MonkeyPatch):
        import niaharness.tools.computer_use_tool as mod

        mock_pg = MagicMock()
        monkeypatch.setattr(mod, "_pyautogui", mock_pg)
        monkeypatch.setattr(mod, "_pyautogui_error", None)

        result = await ComputerUseTool().execute(
            ComputerUseInput(action="type", text="hello world"),
            context,
        )
        assert result.is_error is False
        assert "11 characters" in result.output
        mock_pg.typewrite.assert_called_once()

    @pytest.mark.asyncio
    async def test_key(self, context: ToolExecutionContext, monkeypatch: pytest.MonkeyPatch):
        import niaharness.tools.computer_use_tool as mod

        mock_pg = MagicMock()
        monkeypatch.setattr(mod, "_pyautogui", mock_pg)
        monkeypatch.setattr(mod, "_pyautogui_error", None)

        result = await ComputerUseTool().execute(
            ComputerUseInput(action="key", key="enter"),
            context,
        )
        assert result.is_error is False
        assert "enter" in result.output
        mock_pg.press.assert_called_once_with("enter")

    @pytest.mark.asyncio
    async def test_key_combo(self, context: ToolExecutionContext, monkeypatch: pytest.MonkeyPatch):
        import niaharness.tools.computer_use_tool as mod

        mock_pg = MagicMock()
        monkeypatch.setattr(mod, "_pyautogui", mock_pg)
        monkeypatch.setattr(mod, "_pyautogui_error", None)

        result = await ComputerUseTool().execute(
            ComputerUseInput(action="key_combo", key_combo="ctrl+c"),
            context,
        )
        assert result.is_error is False
        assert "ctrl+c" in result.output
        mock_pg.hotkey.assert_called_once_with("ctrl", "c")

    @pytest.mark.asyncio
    async def test_scroll(self, context: ToolExecutionContext, monkeypatch: pytest.MonkeyPatch):
        import niaharness.tools.computer_use_tool as mod

        mock_pg = MagicMock()
        monkeypatch.setattr(mod, "_pyautogui", mock_pg)
        monkeypatch.setattr(mod, "_pyautogui_error", None)

        result = await ComputerUseTool().execute(
            ComputerUseInput(action="scroll", scroll_clicks=-5),
            context,
        )
        assert result.is_error is False
        assert "down" in result.output
        assert "5" in result.output
        mock_pg.scroll.assert_called_once_with(-5)

    @pytest.mark.asyncio
    async def test_drag(self, context: ToolExecutionContext, monkeypatch: pytest.MonkeyPatch):
        import niaharness.tools.computer_use_tool as mod

        mock_pg = MagicMock()
        monkeypatch.setattr(mod, "_pyautogui", mock_pg)
        monkeypatch.setattr(mod, "_pyautogui_error", None)

        result = await ComputerUseTool().execute(
            ComputerUseInput(action="drag", x=0, y=0, x2=100, y2=200),
            context,
        )
        assert result.is_error is False
        assert "(0, 0)" in result.output
        assert "(100, 200)" in result.output
        mock_pg.dragTo.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_windows(self, context: ToolExecutionContext, monkeypatch: pytest.MonkeyPatch):
        import niaharness.tools.computer_use_tool as mod

        mock_pg = MagicMock()
        mock_window1 = MagicMock()
        mock_window1.title = "Terminal"
        mock_window1.left = 0
        mock_window1.top = 0
        mock_window1.width = 800
        mock_window1.height = 600
        mock_window2 = MagicMock()
        mock_window2.title = "Firefox"
        mock_window2.left = 100
        mock_window2.top = 100
        mock_window2.width = 1200
        mock_window2.height = 800
        mock_pg.getAllWindows = MagicMock(return_value=[mock_window1, mock_window2])
        monkeypatch.setattr(mod, "_pyautogui", mock_pg)
        monkeypatch.setattr(mod, "_pyautogui_error", None)

        result = await ComputerUseTool().execute(
            ComputerUseInput(action="list_windows"),
            context,
        )
        assert result.is_error is False
        assert "Terminal" in result.output
        assert "Firefox" in result.output
        assert "2" in result.output
