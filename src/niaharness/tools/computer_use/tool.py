"""Computer Use tool — desktop automation via pluggable backends.

The audit (P1 #10) flagged that NIA has no GUI automation capability.
This tool fills that gap, letting the agent drive the desktop: click,
type, scroll, take screenshots, and manage windows.

Backend architecture (mirrors Hermes Agent's tools/computer_use/):
- **CUADriverBackend** (primary) — cross-platform (macOS, Windows, Linux
  including X11 + XWayland). Runs in the background without stealing the
  user's cursor or keyboard focus. Requires the `cua-driver` binary.
- **PyAutoGUIBackend** (fallback) — works on all three platforms but
  steals focus. Requires `pip install pyautogui`.

The backend is auto-selected at runtime — cua-driver takes priority if
available, else pyautogui, else a helpful install error.

Operations (13 — matches Hermes's schema):
  capture, click, double_click, right_click, scroll, type, key,
  key_combo, drag, list_apps, focus_app, wait, set_value

Reference: Hermes Agent's tools/computer_use/ (backend.py, cua_backend.py,
schema.py). The backend abstraction and CUADriverBackend are adapted from
Hermes's implementation.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from niaharness.tools.base import BaseTool, ToolExecutionContext, ToolResult
from niaharness.tools.computer_use.backend import (
    ActionResult,
    CaptureResult,
    get_backend,
    get_backend_name,
    reset_backend,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema (adapted from Hermes's COMPUTER_USE_SCHEMA)
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
        "list_apps",
        "focus_app",
        "wait",
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
    app: str | None = Field(
        default=None,
        description="App name (for 'focus_app') or window title filter (for 'capture').",
    )
    mode: Literal["vision", "som", "ax"] = Field(
        default="vision",
        description="Capture mode (for 'capture' action). 'vision' = screenshot only, 'som' = screenshot + numbered elements, 'ax' = elements only.",
    )
    wait_seconds: float | None = Field(
        default=None,
        ge=0.1,
        le=30.0,
        description="Seconds to wait (for 'wait' action).",
    )


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


class ComputerUseTool(BaseTool):
    """Drive the desktop — click, type, scroll, screenshot, manage windows.

    Uses cua-driver (cross-platform, background-safe) when available, falls
    back to PyAutoGUI. Works on macOS, Windows, and Linux.
    """

    name = "computer_use"
    description = (
        "Drive the desktop: take screenshots, click, type, scroll, press "
        "keys, drag, and manage windows. Uses cua-driver (cross-platform, "
        "background-safe) when available, falls back to PyAutoGUI. Works on "
        "macOS, Windows, and Linux. Operations: capture, click, double_click, "
        "right_click, scroll, type, key, key_combo, drag, list_apps, "
        "focus_app, wait."
    )
    input_model = ComputerUseInput

    def is_read_only(self, arguments: ComputerUseInput) -> bool:
        return arguments.action in ("capture", "list_apps", "wait")

    async def execute(self, arguments: ComputerUseInput, context: ToolExecutionContext) -> ToolResult:
        # 'wait' doesn't need a backend — handle it first.
        if arguments.action == "wait":
            return self._wait(arguments)

        # Get the backend.
        try:
            backend = get_backend()
        except RuntimeError as exc:
            return ToolResult(output=str(exc), is_error=True)

        action = arguments.action

        try:
            if action == "capture":
                return self._capture(backend, arguments)
            if action == "click":
                return self._click(backend, arguments)
            if action == "double_click":
                return self._double_click(backend, arguments)
            if action == "right_click":
                return self._right_click(backend, arguments)
            if action == "scroll":
                return self._scroll(backend, arguments)
            if action == "type":
                return self._type(backend, arguments)
            if action == "key":
                return self._key(backend, arguments)
            if action == "key_combo":
                return self._key_combo(backend, arguments)
            if action == "drag":
                return self._drag(backend, arguments)
            if action == "list_apps":
                return self._list_apps(backend)
            if action == "focus_app":
                return self._focus_app(backend, arguments)

            return ToolResult(output=f"Unknown action: {action}", is_error=True)

        except Exception as exc:
            return ToolResult(output=f"Computer use error: {exc}", is_error=True)

    # ---- actions -------------------------------------------------------

    def _capture(self, backend, args: ComputerUseInput) -> ToolResult:
        """Take a screenshot."""
        result = backend.capture(mode=args.mode, app=args.app)

        # Save to download dir if we have base64 image data.
        from datetime import datetime, timezone
        import base64 as _b64

        out_path = None
        if result.png_b64:
            out_dir = Path("/home/z/my-project/download")
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            out_path = out_dir / f"screenshot-{ts}.png"
            out_path.write_bytes(_b64.b64decode(result.png_b64))

        lines = [
            f"Screenshot captured: {result.width}x{result.height}",
            f"  Mode: {result.mode}",
            f"  Backend: {get_backend_name()}",
        ]
        if out_path:
            lines.append(f"  Saved to: {out_path}")
        if result.elements:
            lines.append(f"  Elements: {len(result.elements)}")
            for el in result.elements[:20]:  # show first 20
                lines.append(f"    [{el.index}] {el.role}: {el.label[:60]}")

        return ToolResult(
            output="\n".join(lines),
            metadata={
                "width": result.width,
                "height": result.height,
                "path": str(out_path) if out_path else None,
                "mode": result.mode,
                "backend": get_backend_name(),
                "element_count": len(result.elements),
            },
        )

    def _click(self, backend, args: ComputerUseInput) -> ToolResult:
        if args.x is None or args.y is None:
            return ToolResult(output="click requires x and y", is_error=True)
        result = backend.click(args.x, args.y)
        return ToolResult(output=result.message, is_error=not result.ok)

    def _double_click(self, backend, args: ComputerUseInput) -> ToolResult:
        if args.x is None or args.y is None:
            return ToolResult(output="double_click requires x and y", is_error=True)
        result = backend.double_click(args.x, args.y)
        return ToolResult(output=result.message, is_error=not result.ok)

    def _right_click(self, backend, args: ComputerUseInput) -> ToolResult:
        if args.x is None or args.y is None:
            return ToolResult(output="right_click requires x and y", is_error=True)
        result = backend.right_click(args.x, args.y)
        return ToolResult(output=result.message, is_error=not result.ok)

    def _scroll(self, backend, args: ComputerUseInput) -> ToolResult:
        if args.scroll_clicks is None:
            return ToolResult(output="scroll requires scroll_clicks (positive=up, negative=down)", is_error=True)
        result = backend.scroll(args.scroll_clicks)
        return ToolResult(output=result.message, is_error=not result.ok)

    def _type(self, backend, args: ComputerUseInput) -> ToolResult:
        if not args.text:
            return ToolResult(output="type requires text", is_error=True)
        result = backend.type_text(args.text)
        return ToolResult(output=result.message, is_error=not result.ok)

    def _key(self, backend, args: ComputerUseInput) -> ToolResult:
        if not args.key:
            return ToolResult(output="key requires a key name (e.g. 'enter', 'tab', 'escape')", is_error=True)
        result = backend.key(args.key)
        return ToolResult(output=result.message, is_error=not result.ok)

    def _key_combo(self, backend, args: ComputerUseInput) -> ToolResult:
        if not args.key_combo:
            return ToolResult(output="key_combo requires a combination (e.g. 'ctrl+c', 'alt+tab')", is_error=True)
        result = backend.key_combo(args.key_combo)
        return ToolResult(output=result.message, is_error=not result.ok)

    def _drag(self, backend, args: ComputerUseInput) -> ToolResult:
        if args.x is None or args.y is None or args.x2 is None or args.y2 is None:
            return ToolResult(output="drag requires x, y, x2, y2", is_error=True)
        result = backend.drag(args.x, args.y, args.x2, args.y2)
        return ToolResult(output=result.message, is_error=not result.ok)

    def _list_apps(self, backend) -> ToolResult:
        """List running applications."""
        apps = backend.list_apps()
        if not apps:
            return ToolResult(output="No apps found (or not supported on this backend).")
        lines = [f"Running apps ({len(apps)}):"]
        for i, app in enumerate(apps, 1):
            name = app.get("name", "?")
            lines.append(f"  {i}. {name}")
        return ToolResult(output="\n".join(lines), metadata={"apps": apps})

    def _focus_app(self, backend, args: ComputerUseInput) -> ToolResult:
        if not args.app:
            return ToolResult(output="focus_app requires an app name", is_error=True)
        result = backend.focus_app(args.app)
        return ToolResult(output=result.message, is_error=not result.ok)

    def _wait(self, args: ComputerUseInput) -> ToolResult:
        """Wait for a specified duration."""
        seconds = args.wait_seconds or 1.0
        time.sleep(seconds)
        return ToolResult(output=f"Waited {seconds}s")
