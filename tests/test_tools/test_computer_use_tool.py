"""Tests for the computer_use tool (backend abstraction)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from niaharness.tools.base import ToolExecutionContext
from niaharness.tools.computer_use.backend import (
    ActionResult,
    CaptureResult,
    CUADriverBackend,
    PyAutoGUIBackend,
    get_backend,
    get_backend_name,
    reset_backend,
)
from niaharness.tools.computer_use.tool import ComputerUseTool, ComputerUseInput


@pytest.fixture
def context(tmp_path: Path) -> ToolExecutionContext:
    return ToolExecutionContext(cwd=tmp_path)


@pytest.fixture(autouse=True)
def reset_backend_after():
    """Reset the cached backend after each test."""
    yield
    reset_backend()


# ---------------------------------------------------------------------------
# Read-only flag
# ---------------------------------------------------------------------------


class TestReadOnly:
    def test_capture_is_read_only(self):
        tool = ComputerUseTool()
        assert tool.is_read_only(ComputerUseInput(action="capture")) is True

    def test_list_apps_is_read_only(self):
        tool = ComputerUseTool()
        assert tool.is_read_only(ComputerUseInput(action="list_apps")) is True

    def test_wait_is_read_only(self):
        tool = ComputerUseTool()
        assert tool.is_read_only(ComputerUseInput(action="wait", wait_seconds=0.1)) is True

    def test_click_is_not_read_only(self):
        tool = ComputerUseTool()
        assert tool.is_read_only(ComputerUseInput(action="click", x=10, y=20)) is False

    def test_type_is_not_read_only(self):
        tool = ComputerUseTool()
        assert tool.is_read_only(ComputerUseInput(action="type", text="hello")) is False


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------


class TestBackendSelection:
    def test_get_backend_name_none_when_unavailable(self, monkeypatch: pytest.MonkeyPatch):
        """When no backend is available, get_backend_name returns 'none'."""
        reset_backend()
        monkeypatch.setattr(
            "niaharness.tools.computer_use.backend.cua_driver_binary_available",
            lambda: False,
        )
        # Make pyautogui unavailable too.
        import niaharness.tools.computer_use.backend as mod

        original_pg = mod.PyAutoGUIBackend
        monkeypatch.setattr(
            mod.PyAutoGUIBackend, "available", lambda self: False
        )
        assert get_backend_name() == "none"

    def test_cua_driver_takes_priority(self, monkeypatch: pytest.MonkeyPatch):
        """When cua-driver is available, it's used over pyautogui."""
        reset_backend()
        monkeypatch.setattr(
            "niaharness.tools.computer_use.backend.cua_driver_binary_available",
            lambda: True,
        )
        backend = get_backend()
        assert isinstance(backend, CUADriverBackend)


# ---------------------------------------------------------------------------
# Mocked actions via PyAutoGUI backend
# ---------------------------------------------------------------------------


class TestMockedPyAutoGUI:
    @pytest.mark.asyncio
    async def test_capture(self, context: ToolExecutionContext, monkeypatch: pytest.MonkeyPatch):
        """Test capture via mocked pyautogui backend."""
        reset_backend()
        # Force pyautogui backend.
        monkeypatch.setattr(
            "niaharness.tools.computer_use.backend.cua_driver_binary_available",
            lambda: False,
        )

        mock_pg = MagicMock()
        mock_img = MagicMock()
        mock_img.width = 1920
        mock_img.height = 1080
        mock_pg.screenshot = MagicMock(return_value=mock_img)

        with patch("builtins.__import__") as mock_import:
            def _import(name, *args, **kwargs):
                if name == "pyautogui":
                    return mock_pg
                return __builtins__.__import__(name, *args, **kwargs) if hasattr(__builtins__, "__import__") else __import__(name)

            # Simpler: just patch the backend's _get_pg directly
            pass

        # Patch PyAutoGUIBackend._get_pg to return our mock.
        with patch.object(PyAutoGUIBackend, "_get_pg", return_value=mock_pg):
            result = await ComputerUseTool().execute(
                ComputerUseInput(action="capture"),
                context,
            )

        assert result.is_error is False
        assert "1920x1080" in result.output
        assert "pyautogui" in result.output

    @pytest.mark.asyncio
    async def test_click(self, context: ToolExecutionContext, monkeypatch: pytest.MonkeyPatch):
        reset_backend()
        monkeypatch.setattr(
            "niaharness.tools.computer_use.backend.cua_driver_binary_available",
            lambda: False,
        )
        mock_pg = MagicMock()
        with patch.object(PyAutoGUIBackend, "_get_pg", return_value=mock_pg):
            result = await ComputerUseTool().execute(
                ComputerUseInput(action="click", x=100, y=200),
                context,
            )
        assert result.is_error is False
        assert "(100, 200)" in result.output or "100" in result.output
        mock_pg.click.assert_called_once()

    @pytest.mark.asyncio
    async def test_type(self, context: ToolExecutionContext, monkeypatch: pytest.MonkeyPatch):
        reset_backend()
        monkeypatch.setattr(
            "niaharness.tools.computer_use.backend.cua_driver_binary_available",
            lambda: False,
        )
        mock_pg = MagicMock()
        with patch.object(PyAutoGUIBackend, "_get_pg", return_value=mock_pg):
            result = await ComputerUseTool().execute(
                ComputerUseInput(action="type", text="hello world"),
                context,
            )
        assert result.is_error is False
        assert "11" in result.output  # 11 chars
        mock_pg.typewrite.assert_called_once()

    @pytest.mark.asyncio
    async def test_key(self, context: ToolExecutionContext, monkeypatch: pytest.MonkeyPatch):
        reset_backend()
        monkeypatch.setattr(
            "niaharness.tools.computer_use.backend.cua_driver_binary_available",
            lambda: False,
        )
        mock_pg = MagicMock()
        with patch.object(PyAutoGUIBackend, "_get_pg", return_value=mock_pg):
            result = await ComputerUseTool().execute(
                ComputerUseInput(action="key", key="enter"),
                context,
            )
        assert result.is_error is False
        assert "enter" in result.output.lower()
        mock_pg.press.assert_called_once_with("enter")

    @pytest.mark.asyncio
    async def test_key_combo(self, context: ToolExecutionContext, monkeypatch: pytest.MonkeyPatch):
        reset_backend()
        monkeypatch.setattr(
            "niaharness.tools.computer_use.backend.cua_driver_binary_available",
            lambda: False,
        )
        mock_pg = MagicMock()
        with patch.object(PyAutoGUIBackend, "_get_pg", return_value=mock_pg):
            result = await ComputerUseTool().execute(
                ComputerUseInput(action="key_combo", key_combo="ctrl+c"),
                context,
            )
        assert result.is_error is False
        mock_pg.hotkey.assert_called_once_with("ctrl", "c")

    @pytest.mark.asyncio
    async def test_scroll(self, context: ToolExecutionContext, monkeypatch: pytest.MonkeyPatch):
        reset_backend()
        monkeypatch.setattr(
            "niaharness.tools.computer_use.backend.cua_driver_binary_available",
            lambda: False,
        )
        mock_pg = MagicMock()
        with patch.object(PyAutoGUIBackend, "_get_pg", return_value=mock_pg):
            result = await ComputerUseTool().execute(
                ComputerUseInput(action="scroll", scroll_clicks=-5),
                context,
            )
        assert result.is_error is False
        assert "down" in result.output.lower()
        mock_pg.scroll.assert_called_once_with(-5)

    @pytest.mark.asyncio
    async def test_drag(self, context: ToolExecutionContext, monkeypatch: pytest.MonkeyPatch):
        reset_backend()
        monkeypatch.setattr(
            "niaharness.tools.computer_use.backend.cua_driver_binary_available",
            lambda: False,
        )
        mock_pg = MagicMock()
        with patch.object(PyAutoGUIBackend, "_get_pg", return_value=mock_pg):
            result = await ComputerUseTool().execute(
                ComputerUseInput(action="drag", x=0, y=0, x2=100, y2=200),
                context,
            )
        assert result.is_error is False
        mock_pg.moveTo.assert_called_once_with(0, 0)
        mock_pg.dragTo.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_apps(self, context: ToolExecutionContext, monkeypatch: pytest.MonkeyPatch):
        reset_backend()
        monkeypatch.setattr(
            "niaharness.tools.computer_use.backend.cua_driver_binary_available",
            lambda: False,
        )
        mock_pg = MagicMock()
        mock_w1 = MagicMock()
        mock_w1.title = "Terminal"
        mock_w1.left = 0
        mock_w1.top = 0
        mock_w1.width = 800
        mock_w1.height = 600
        mock_w2 = MagicMock()
        mock_w2.title = "Firefox"
        mock_w2.left = 100
        mock_w2.top = 100
        mock_w2.width = 1200
        mock_w2.height = 800
        mock_pg.getAllWindows = MagicMock(return_value=[mock_w1, mock_w2])
        with patch.object(PyAutoGUIBackend, "_get_pg", return_value=mock_pg):
            result = await ComputerUseTool().execute(
                ComputerUseInput(action="list_apps"),
                context,
            )
        assert result.is_error is False
        assert "Terminal" in result.output
        assert "Firefox" in result.output

    @pytest.mark.asyncio
    async def test_wait(self, context: ToolExecutionContext):
        reset_backend()
        result = await ComputerUseTool().execute(
            ComputerUseInput(action="wait", wait_seconds=0.1),
            context,
        )
        assert result.is_error is False
        assert "0.1" in result.output

    @pytest.mark.asyncio
    async def test_click_missing_coords(self, context: ToolExecutionContext, monkeypatch: pytest.MonkeyPatch):
        reset_backend()
        monkeypatch.setattr(
            "niaharness.tools.computer_use.backend.cua_driver_binary_available",
            lambda: False,
        )
        mock_pg = MagicMock()
        with patch.object(PyAutoGUIBackend, "_get_pg", return_value=mock_pg):
            result = await ComputerUseTool().execute(
                ComputerUseInput(action="click"),
                context,
            )
        assert result.is_error is True
        assert "x and y" in result.output

    @pytest.mark.asyncio
    async def test_no_backend_available(self, context: ToolExecutionContext, monkeypatch: pytest.MonkeyPatch):
        """When no backend is available, return a helpful error."""
        reset_backend()
        monkeypatch.setattr(
            "niaharness.tools.computer_use.backend.cua_driver_binary_available",
            lambda: False,
        )
        monkeypatch.setattr(
            "niaharness.tools.computer_use.backend.PyAutoGUIBackend.available",
            lambda self: False,
        )
        result = await ComputerUseTool().execute(
            ComputerUseInput(action="capture"),
            context,
        )
        assert result.is_error is True
        assert "backend" in result.output.lower() or "install" in result.output.lower()
