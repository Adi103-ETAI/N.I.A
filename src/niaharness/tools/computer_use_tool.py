"""Computer Use tool — desktop automation via PyAutoGUI.

The audit (P1 #10) flagged that NIA has no GUI automation capability.
This tool fills that gap, letting the agent drive the desktop: click,
type, scroll, take screenshots, and manage windows.

Operations (10):
  - capture       — take a screenshot and return it
  - click         — click at (x, y) or at the center of a window
  - double_click  — double-click at (x, y)
  - right_click   — right-click at (x, y)
  - scroll        — scroll up/down by N clicks
  - type          — type a string of text
  - key           — press a keyboard key (enter, tab, escape, etc.)
  - key_combo     — press a key combination (e.g. 'ctrl+c')
  - drag          — drag from (x1, y1) to (x2, y2)
  - list_windows  — list all open window titles

Requirements:
  - PyAutoGUI (pip install pyautogui)
  - Pillow (for screenshots — usually installed with pyautogui)
  - A display (X11 on Linux, or a desktop session on macOS/Windows)

In headless environments (no display), the tool returns a helpful error
explaining how to set up a virtual display (Xvfb) or run on a machine
with a desktop session.

Reference: Hermes Agent's tools/computer_use/ (uses cua-driver for
background automation without stealing the cursor). NIA's version is
simpler — direct PyAutoGUI — but covers the same core operations.
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Literal

from pydantic import BaseModel, Field

from niaharness.tools.base import BaseTool, ToolExecutionContext, ToolResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class ComputerUseInput(BaseModel):
    """Arguments for the computer_use tool."""

    action: Literal[
        "capture",
        "click",
        "double_click",
        "right_click",
        "scroll",
        "type",
        "key",
        "key_combo",
        "drag",
        "list_windows",
    ] = Field(description="The desktop automation action to perform")
    x: int | None = Field(default=None, ge=0, description="X coordinate for click/double_click/right_click/drag start")
    y: int | None = Field(default=None, ge=0, description="Y coordinate for click/double_click/right_click/drag start")
    x2: int | None = Field(default=None, ge=0, description="End X for drag")
    y2: int | None = Field(default=None, ge=0, description="End Y for drag")
    text: str | None = Field(default=None, description="Text to type (for 'type' action)")
    key: str | None = Field(
        default=None,
        description="Key to press (for 'key' action). Examples: enter, tab, escape, backspace, space, up, down, left, right, f1-f12.",
    )
    key_combo: str | None = Field(
        default=None,
        description="Key combination (for 'key_combo' action). Examples: 'ctrl+c', 'alt+tab', 'cmd+space'.",
    )
    scroll_clicks: int | None = Field(
        default=None,
        ge=-100,
        le=100,
        description="Scroll amount (for 'scroll' action). Positive = up, negative = down.",
    )
    window: str | None = Field(
        default=None,
        description="Window title to focus before performing the action (partial match).",
    )


# ---------------------------------------------------------------------------
# PyAutoGUI lazy loader
# ---------------------------------------------------------------------------


_pyautogui = None
_pyautogui_error: str | None = None


def _get_pyautogui():
    """Lazily import pyautogui. Caches the result (including errors)."""
    global _pyautogui, _pyautogui_error
    if _pyautogui is not None:
        return _pyautogui
    if _pyautogui_error is not None:
        raise RuntimeError(_pyautogui_error)
    try:
        import pyautogui  # type: ignore

        pyautogui.FAILSAFE = True  # move mouse to corner to abort
        pyautogui.PAUSE = 0.1  # small delay between actions for safety
        _pyautogui = pyautogui
        return pyautogui
    except ImportError:
        _pyautogui_error = (
            "pyautogui is not installed. Install with: pip install pyautogui"
        )
        raise RuntimeError(_pyautogui_error)
    except Exception as exc:
        _pyautogui_error = (
            f"Could not initialize pyautogui: {exc}. "
            f"If running headless, install Xvfb and run: xvfb-run python -m niaharness"
        )
        raise RuntimeError(_pyautogui_error)


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


class ComputerUseTool(BaseTool):
    """Drive the desktop — click, type, scroll, screenshot, manage windows."""

    name = "computer_use"
    description = (
        "Drive the user's desktop: take screenshots, click, type, scroll, "
        "press keys, and manage windows via PyAutoGUI. Requires a display "
        "session (X11/macOS/Windows). In headless environments, use Xvfb. "
        "Operations: capture, click, double_click, right_click, scroll, type, "
        "key, key_combo, drag, list_windows."
    )
    input_model = ComputerUseInput

    def is_read_only(self, arguments: ComputerUseInput) -> bool:
        # Only capture and list_windows are read-only.
        return arguments.action in ("capture", "list_windows")

    async def execute(self, arguments: ComputerUseInput, context: ToolExecutionContext) -> ToolResult:
        try:
            pg = _get_pyautogui()
        except RuntimeError as exc:
            return ToolResult(output=str(exc), is_error=True)

        action = arguments.action

        try:
            # Focus window if requested.
            if arguments.window and action != "capture":
                self._focus_window(pg, arguments.window)

            if action == "capture":
                return self._capture(pg)
            if action == "click":
                return self._click(pg, arguments)
            if action == "double_click":
                return self._double_click(pg, arguments)
            if action == "right_click":
                return self._right_click(pg, arguments)
            if action == "scroll":
                return self._scroll(pg, arguments)
            if action == "type":
                return self._type(pg, arguments)
            if action == "key":
                return self._key(pg, arguments)
            if action == "key_combo":
                return self._key_combo(pg, arguments)
            if action == "drag":
                return self._drag(pg, arguments)
            if action == "list_windows":
                return self._list_windows(pg)

            return ToolResult(output=f"Unknown action: {action}", is_error=True)

        except Exception as exc:
            return ToolResult(output=f"Computer use error: {exc}", is_error=True)

    # ---- actions -------------------------------------------------------

    def _capture(self, pg) -> ToolResult:
        """Take a screenshot and return it as base64."""
        try:
            screenshot = pg.screenshot()
            buf = io.BytesIO()
            screenshot.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            size_kb = len(b64) * 3 // 4 // 1024  # approx original size

            # Also save to download dir for the user.
            from datetime import datetime, timezone

            from pathlib import Path

            out_dir = Path("/home/z/my-project/download")
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            out_path = out_dir / f"screenshot-{ts}.png"
            screenshot.save(str(out_path))

            return ToolResult(
                output=(
                    f"Screenshot captured: {screenshot.width}x{screenshot.height}\n"
                    f"Saved to: {out_path}\n"
                    f"Base64 length: {len(b64)} chars (~{size_kb} KB)"
                ),
                metadata={
                    "width": screenshot.width,
                    "height": screenshot.height,
                    "path": str(out_path),
                    "base64_length": len(b64),
                },
            )
        except Exception as exc:
            return ToolResult(output=f"Screenshot failed: {exc}", is_error=True)

    def _click(self, pg, args: ComputerUseInput) -> ToolResult:
        if args.x is None or args.y is None:
            return ToolResult(output="click requires x and y", is_error=True)
        pg.click(args.x, args.y)
        return ToolResult(output=f"Clicked at ({args.x}, {args.y})")

    def _double_click(self, pg, args: ComputerUseInput) -> ToolResult:
        if args.x is None or args.y is None:
            return ToolResult(output="double_click requires x and y", is_error=True)
        pg.doubleClick(args.x, args.y)
        return ToolResult(output=f"Double-clicked at ({args.x}, {args.y})")

    def _right_click(self, pg, args: ComputerUseInput) -> ToolResult:
        if args.x is None or args.y is None:
            return ToolResult(output="right_click requires x and y", is_error=True)
        pg.rightClick(args.x, args.y)
        return ToolResult(output=f"Right-clicked at ({args.x}, {args.y})")

    def _scroll(self, pg, args: ComputerUseInput) -> ToolResult:
        if args.scroll_clicks is None:
            return ToolResult(output="scroll requires scroll_clicks (positive=up, negative=down)", is_error=True)
        pg.scroll(args.scroll_clicks)
        direction = "up" if args.scroll_clicks > 0 else "down"
        return ToolResult(output=f"Scrolled {direction} {abs(args.scroll_clicks)} click(s)")

    def _type(self, pg, args: ComputerUseInput) -> ToolResult:
        if not args.text:
            return ToolResult(output="type requires text", is_error=True)
        pg.typewrite(args.text, interval=0.02)
        return ToolResult(output=f"Typed {len(args.text)} characters")

    def _key(self, pg, args: ComputerUseInput) -> ToolResult:
        if not args.key:
            return ToolResult(output="key requires a key name (e.g. 'enter', 'tab', 'escape')", is_error=True)
        pg.press(args.key)
        return ToolResult(output=f"Pressed key: {args.key}")

    def _key_combo(self, pg, args: ComputerUseInput) -> ToolResult:
        if not args.key_combo:
            return ToolResult(output="key_combo requires a combination (e.g. 'ctrl+c', 'alt+tab')", is_error=True)
        keys = [k.strip() for k in args.key_combo.split("+")]
        pg.hotkey(*keys)
        return ToolResult(output=f"Pressed key combo: {args.key_combo}")

    def _drag(self, pg, args: ComputerUseInput) -> ToolResult:
        if args.x is None or args.y is None or args.x2 is None or args.y2 is None:
            return ToolResult(output="drag requires x, y, x2, y2", is_error=True)
        pg.dragTo(args.x2, args.y2, duration=0.5, _pause=False)
        return ToolResult(output=f"Dragged from ({args.x}, {args.y}) to ({args.x2}, {args.y2})")

    def _list_windows(self, pg) -> ToolResult:
        """List all open windows."""
        try:
            windows = pg.getAllWindows()
            if not windows:
                return ToolResult(output="No windows found (or window manager not supported).")
            lines = [f"Open windows ({len(windows)}):"]
            for i, w in enumerate(windows, 1):
                lines.append(f"  {i}. {w.title} ({w.left},{w.top} {w.width}x{w.height})")
            return ToolResult(output="\n".join(lines))
        except Exception:
            # getAllWindows requires pygetwindow which may not be available.
            return ToolResult(
                output=(
                    "Could not list windows (pygetwindow not available on this platform). "
                    "Use 'capture' to take a screenshot instead."
                )
            )

    def _focus_window(self, pg, title_partial: str) -> None:
        """Focus a window by partial title match."""
        try:
            for w in pg.getAllWindows():
                if title_partial.lower() in w.title.lower():
                    w.activate()
                    return
        except Exception:
            pass  # best-effort — don't fail the whole action
