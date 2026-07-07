"""Tests for the computer_use tool (cua-driver backend only).

Mirrors Hermes Agent's approach: cua-driver is the sole backend, no
PyAutoGUI fallback. Tests mock the cua-driver MCP communication.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from niaharness.tools.base import ToolExecutionContext
from niaharness.tools.computer_use.backend import (
    ActionResult,
    CaptureResult,
    UIElement,
)
from niaharness.tools.computer_use.cua_backend import (
    CUADriverBackend,
    cua_driver_binary_available,
    get_backend,
    get_backend_name,
    reset_backend,
)
from niaharness.tools.computer_use.schema import ComputerUseInput
from niaharness.tools.computer_use.tool import ComputerUseTool


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
        assert ComputerUseTool().is_read_only(ComputerUseInput(action="capture")) is True

    def test_list_apps_is_read_only(self):
        assert ComputerUseTool().is_read_only(ComputerUseInput(action="list_apps")) is True

    def test_wait_is_read_only(self):
        assert ComputerUseTool().is_read_only(ComputerUseInput(action="wait", seconds=0.1)) is True

    def test_click_is_not_read_only(self):
        assert ComputerUseTool().is_read_only(ComputerUseInput(action="click")) is False

    def test_type_is_not_read_only(self):
        assert ComputerUseTool().is_read_only(ComputerUseInput(action="type", text="hi")) is False


# ---------------------------------------------------------------------------
# Backend availability
# ---------------------------------------------------------------------------


class TestBackendAvailability:
    def test_get_backend_name_none_when_unavailable(self, monkeypatch: pytest.MonkeyPatch):
        reset_backend()
        monkeypatch.setattr(
            "niaharness.tools.computer_use.cua_backend.cua_driver_binary_available",
            lambda: False,
        )
        assert get_backend_name() == "none"

    def test_get_backend_name_cua_when_available(self, monkeypatch: pytest.MonkeyPatch):
        reset_backend()
        monkeypatch.setattr(
            "niaharness.tools.computer_use.cua_backend.cua_driver_binary_available",
            lambda: True,
        )
        assert get_backend_name() == "cua-driver"

    def test_get_backend_raises_when_unavailable(self, monkeypatch: pytest.MonkeyPatch):
        reset_backend()
        monkeypatch.setattr(
            "niaharness.tools.computer_use.cua_backend.cua_driver_binary_available",
            lambda: False,
        )
        with pytest.raises(RuntimeError, match="cua-driver is not installed"):
            get_backend()


# ---------------------------------------------------------------------------
# Tool error handling
# ---------------------------------------------------------------------------


class TestToolErrors:
    @pytest.mark.asyncio
    async def test_no_cua_driver_returns_install_error(self, context: ToolExecutionContext, monkeypatch: pytest.MonkeyPatch):
        """When cua-driver is not installed, return a helpful install error."""
        reset_backend()
        monkeypatch.setattr(
            "niaharness.tools.computer_use.cua_backend.cua_driver_binary_available",
            lambda: False,
        )
        result = await ComputerUseTool().execute(
            ComputerUseInput(action="capture"),
            context,
        )
        assert result.is_error is True
        assert "cua-driver" in result.output
        assert "install" in result.output.lower()

    @pytest.mark.asyncio
    async def test_wait_works_without_backend(self, context: ToolExecutionContext):
        """wait doesn't need cua-driver — it should work even without it."""
        reset_backend()
        result = await ComputerUseTool().execute(
            ComputerUseInput(action="wait", seconds=0.1),
            context,
        )
        assert result.is_error is False
        assert "0.1" in result.output

    @pytest.mark.asyncio
    async def test_type_missing_text(self, context: ToolExecutionContext, monkeypatch: pytest.MonkeyPatch):
        reset_backend()
        monkeypatch.setattr(
            "niaharness.tools.computer_use.cua_backend.cua_driver_binary_available",
            lambda: True,
        )
        # Mock the backend to avoid real subprocess.
        mock_backend = MagicMock()
        mock_backend.type_text = MagicMock(return_value=ActionResult(ok=True, action="type", message="ok"))
        with patch("niaharness.tools.computer_use.tool.get_backend", return_value=mock_backend):
            result = await ComputerUseTool().execute(
                ComputerUseInput(action="type"),
                context,
            )
        assert result.is_error is True
        assert "text" in result.output.lower()

    @pytest.mark.asyncio
    async def test_key_missing_keys(self, context: ToolExecutionContext, monkeypatch: pytest.MonkeyPatch):
        reset_backend()
        monkeypatch.setattr(
            "niaharness.tools.computer_use.cua_backend.cua_driver_binary_available",
            lambda: True,
        )
        mock_backend = MagicMock()
        with patch("niaharness.tools.computer_use.tool.get_backend", return_value=mock_backend):
            result = await ComputerUseTool().execute(
                ComputerUseInput(action="key"),
                context,
            )
        assert result.is_error is True
        assert "key" in result.output.lower()


# ---------------------------------------------------------------------------
# Mocked cua-driver actions
# ---------------------------------------------------------------------------


class TestMockedCUADriver:
    @pytest.mark.asyncio
    async def test_capture(self, context: ToolExecutionContext, monkeypatch: pytest.MonkeyPatch):
        """Test capture via mocked cua-driver backend."""
        reset_backend()
        monkeypatch.setattr(
            "niaharness.tools.computer_use.cua_backend.cua_driver_binary_available",
            lambda: True,
        )
        mock_backend = MagicMock()
        mock_backend.capture = MagicMock(return_value=CaptureResult(
            mode="som",
            width=1920,
            height=1080,
            png_b64=None,
            elements=[
                UIElement(index=1, role="AXButton", label="Submit"),
                UIElement(index=2, role="AXTextField", label="Search"),
            ],
        ))
        with patch("niaharness.tools.computer_use.tool.get_backend", return_value=mock_backend):
            result = await ComputerUseTool().execute(
                ComputerUseInput(action="capture", mode="som"),
                context,
            )
        assert result.is_error is False
        assert "1920x1080" in result.output
        assert "cua-driver" in result.output
        assert "Submit" in result.output
        assert "Search" in result.output

    @pytest.mark.asyncio
    async def test_click_by_element(self, context: ToolExecutionContext, monkeypatch: pytest.MonkeyPatch):
        reset_backend()
        monkeypatch.setattr(
            "niaharness.tools.computer_use.cua_backend.cua_driver_binary_available",
            lambda: True,
        )
        mock_backend = MagicMock()
        mock_backend.click = MagicMock(return_value=ActionResult(
            ok=True, action="click", message="Clicked element 3"
        ))
        with patch("niaharness.tools.computer_use.tool.get_backend", return_value=mock_backend):
            result = await ComputerUseTool().execute(
                ComputerUseInput(action="click", element=3),
                context,
            )
        assert result.is_error is False
        assert "element 3" in result.output
        mock_backend.click.assert_called_once()

    @pytest.mark.asyncio
    async def test_click_by_coordinates(self, context: ToolExecutionContext, monkeypatch: pytest.MonkeyPatch):
        reset_backend()
        monkeypatch.setattr(
            "niaharness.tools.computer_use.cua_backend.cua_driver_binary_available",
            lambda: True,
        )
        mock_backend = MagicMock()
        mock_backend.click = MagicMock(return_value=ActionResult(
            ok=True, action="click", message="Clicked (100, 200)"
        ))
        with patch("niaharness.tools.computer_use.tool.get_backend", return_value=mock_backend):
            result = await ComputerUseTool().execute(
                ComputerUseInput(action="click", coordinate=[100, 200]),
                context,
            )
        assert result.is_error is False
        mock_backend.click.assert_called_once()
        call_kwargs = mock_backend.click.call_args.kwargs
        assert call_kwargs["x"] == 100
        assert call_kwargs["y"] == 200

    @pytest.mark.asyncio
    async def test_double_click(self, context: ToolExecutionContext, monkeypatch: pytest.MonkeyPatch):
        reset_backend()
        monkeypatch.setattr(
            "niaharness.tools.computer_use.cua_backend.cua_driver_binary_available",
            lambda: True,
        )
        mock_backend = MagicMock()
        mock_backend.click = MagicMock(return_value=ActionResult(
            ok=True, action="click", message="Double-clicked"
        ))
        with patch("niaharness.tools.computer_use.tool.get_backend", return_value=mock_backend):
            result = await ComputerUseTool().execute(
                ComputerUseInput(action="double_click", element=1),
                context,
            )
        assert result.is_error is False
        call_kwargs = mock_backend.click.call_args.kwargs
        assert call_kwargs["click_count"] == 2

    @pytest.mark.asyncio
    async def test_right_click(self, context: ToolExecutionContext, monkeypatch: pytest.MonkeyPatch):
        reset_backend()
        monkeypatch.setattr(
            "niaharness.tools.computer_use.cua_backend.cua_driver_binary_available",
            lambda: True,
        )
        mock_backend = MagicMock()
        mock_backend.click = MagicMock(return_value=ActionResult(
            ok=True, action="click", message="Right-clicked"
        ))
        with patch("niaharness.tools.computer_use.tool.get_backend", return_value=mock_backend):
            result = await ComputerUseTool().execute(
                ComputerUseInput(action="right_click", element=1),
                context,
            )
        assert result.is_error is False
        call_kwargs = mock_backend.click.call_args.kwargs
        assert call_kwargs["button"] == "right"

    @pytest.mark.asyncio
    async def test_type(self, context: ToolExecutionContext, monkeypatch: pytest.MonkeyPatch):
        reset_backend()
        monkeypatch.setattr(
            "niaharness.tools.computer_use.cua_backend.cua_driver_binary_available",
            lambda: True,
        )
        mock_backend = MagicMock()
        mock_backend.type_text = MagicMock(return_value=ActionResult(
            ok=True, action="type", message="Typed 11 chars"
        ))
        with patch("niaharness.tools.computer_use.tool.get_backend", return_value=mock_backend):
            result = await ComputerUseTool().execute(
                ComputerUseInput(action="type", text="hello world"),
                context,
            )
        assert result.is_error is False
        assert "11" in result.output
        mock_backend.type_text.assert_called_once_with("hello world")

    @pytest.mark.asyncio
    async def test_key(self, context: ToolExecutionContext, monkeypatch: pytest.MonkeyPatch):
        reset_backend()
        monkeypatch.setattr(
            "niaharness.tools.computer_use.cua_backend.cua_driver_binary_available",
            lambda: True,
        )
        mock_backend = MagicMock()
        mock_backend.key = MagicMock(return_value=ActionResult(
            ok=True, action="key", message="Pressed cmd+s"
        ))
        with patch("niaharness.tools.computer_use.tool.get_backend", return_value=mock_backend):
            result = await ComputerUseTool().execute(
                ComputerUseInput(action="key", keys="cmd+s"),
                context,
            )
        assert result.is_error is False
        assert "cmd+s" in result.output
        mock_backend.key.assert_called_once_with("cmd+s")

    @pytest.mark.asyncio
    async def test_scroll(self, context: ToolExecutionContext, monkeypatch: pytest.MonkeyPatch):
        reset_backend()
        monkeypatch.setattr(
            "niaharness.tools.computer_use.cua_backend.cua_driver_binary_available",
            lambda: True,
        )
        mock_backend = MagicMock()
        mock_backend.scroll = MagicMock(return_value=ActionResult(
            ok=True, action="scroll", message="Scrolled down 5"
        ))
        with patch("niaharness.tools.computer_use.tool.get_backend", return_value=mock_backend):
            result = await ComputerUseTool().execute(
                ComputerUseInput(action="scroll", direction="down", amount=5),
                context,
            )
        assert result.is_error is False
        assert "down" in result.output.lower()
        mock_backend.scroll.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_apps(self, context: ToolExecutionContext, monkeypatch: pytest.MonkeyPatch):
        reset_backend()
        monkeypatch.setattr(
            "niaharness.tools.computer_use.cua_backend.cua_driver_binary_available",
            lambda: True,
        )
        mock_backend = MagicMock()
        mock_backend.list_apps = MagicMock(return_value=[
            {"name": "Safari", "pid": 123, "window_count": 2},
            {"name": "Terminal", "pid": 456, "window_count": 1},
        ])
        with patch("niaharness.tools.computer_use.tool.get_backend", return_value=mock_backend):
            result = await ComputerUseTool().execute(
                ComputerUseInput(action="list_apps"),
                context,
            )
        assert result.is_error is False
        assert "Safari" in result.output
        assert "Terminal" in result.output

    @pytest.mark.asyncio
    async def test_focus_app(self, context: ToolExecutionContext, monkeypatch: pytest.MonkeyPatch):
        reset_backend()
        monkeypatch.setattr(
            "niaharness.tools.computer_use.cua_backend.cua_driver_binary_available",
            lambda: True,
        )
        mock_backend = MagicMock()
        mock_backend.focus_app = MagicMock(return_value=ActionResult(
            ok=True, action="focus_app", message="Focused Safari"
        ))
        with patch("niaharness.tools.computer_use.tool.get_backend", return_value=mock_backend):
            result = await ComputerUseTool().execute(
                ComputerUseInput(action="focus_app", app="Safari"),
                context,
            )
        assert result.is_error is False
        assert "Safari" in result.output

    @pytest.mark.asyncio
    async def test_set_value(self, context: ToolExecutionContext, monkeypatch: pytest.MonkeyPatch):
        reset_backend()
        monkeypatch.setattr(
            "niaharness.tools.computer_use.cua_backend.cua_driver_binary_available",
            lambda: True,
        )
        mock_backend = MagicMock()
        mock_backend.set_value = MagicMock(return_value=ActionResult(
            ok=True, action="set_value", message="Set value: Blue"
        ))
        with patch("niaharness.tools.computer_use.tool.get_backend", return_value=mock_backend):
            result = await ComputerUseTool().execute(
                ComputerUseInput(action="set_value", value="Blue", element=5),
                context,
            )
        assert result.is_error is False
        assert "Blue" in result.output

    @pytest.mark.asyncio
    async def test_drag(self, context: ToolExecutionContext, monkeypatch: pytest.MonkeyPatch):
        reset_backend()
        monkeypatch.setattr(
            "niaharness.tools.computer_use.cua_backend.cua_driver_binary_available",
            lambda: True,
        )
        mock_backend = MagicMock()
        mock_backend.drag = MagicMock(return_value=ActionResult(
            ok=True, action="drag", message="Drag complete"
        ))
        with patch("niaharness.tools.computer_use.tool.get_backend", return_value=mock_backend):
            result = await ComputerUseTool().execute(
                ComputerUseInput(
                    action="drag",
                    from_element=1,
                    to_element=2,
                ),
                context,
            )
        assert result.is_error is False
        mock_backend.drag.assert_called_once()
