"""Computer Use tool — desktop automation via cua-driver.

Adapted from Hermes Agent's tools/computer_use/tool.py.

Drives the desktop in the background via cua-driver — screenshots, mouse,
keyboard, scroll, drag — without stealing the user's cursor or keyboard
focus. Supported on macOS, Windows, and Linux (X11 + XWayland).

Preferred workflow: call with action='capture' (mode='som' gives numbered
element overlays), then click by element index for reliability. Pixel
coordinates are supported for models trained on them.

Requires cua-driver to be installed. No fallback — mirrors Hermes Agent's
approach of cua-driver as the sole backend.
"""

from __future__ import annotations

import base64
import logging
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

from niaharness.tools.base import BaseTool, ToolExecutionContext, ToolResult
from niaharness.tools.computer_use.backend import ActionResult, CaptureResult
from niaharness.tools.computer_use.cua_backend import (
    CUADriverBackend,
    cua_driver_binary_available,
    get_backend,
    reset_backend,
)
from niaharness.tools.computer_use.schema import ComputerUseInput

logger = logging.getLogger(__name__)


class ComputerUseTool(BaseTool):
    """Drive the desktop via cua-driver — screenshots, mouse, keyboard, scroll, drag.

    Works on macOS, Windows, and Linux without stealing the user's cursor
    or keyboard focus. Requires cua-driver to be installed.
    """

    name = "computer_use"
    description = (
        "Drive the desktop in the background via cua-driver — screenshots, "
        "mouse, keyboard, scroll, drag — without stealing the user's cursor "
        "or keyboard focus. Supported on macOS, Windows, and Linux. "
        "Preferred workflow: call with action='capture' (mode='som' gives "
        "numbered element overlays), then click by element index for "
        "reliability. Pixel coordinates are supported for models trained "
        "on them. Works on any window — hidden, minimized, or behind "
        "another app. Requires cua-driver to be installed."
    )
    input_model = ComputerUseInput

    def is_read_only(self, arguments: ComputerUseInput) -> bool:
        # capture, list_apps, and wait are read-only (no side effects).
        return arguments.action in ("capture", "list_apps", "wait")

    async def execute(self, arguments: ComputerUseInput, context: ToolExecutionContext) -> ToolResult:
        # 'wait' doesn't need the backend — handle it first.
        if arguments.action == "wait":
            seconds = arguments.seconds or 1.0
            import time

            time.sleep(max(0.0, min(seconds, 30.0)))
            return ToolResult(output=f"Waited {seconds}s")

        # Get the cua-driver backend.
        try:
            backend = get_backend()
        except RuntimeError as exc:
            return ToolResult(output=str(exc), is_error=True)

        action = arguments.action

        try:
            if action == "capture":
                return self._capture(backend, arguments)
            if action == "click":
                return self._do_click(backend, arguments, click_count=1)
            if action == "double_click":
                return self._do_click(backend, arguments, click_count=2)
            if action == "right_click":
                return self._do_click(backend, arguments, button="right")
            if action == "middle_click":
                return self._do_click(backend, arguments, button="middle")
            if action == "drag":
                return self._do_drag(backend, arguments)
            if action == "scroll":
                return self._do_scroll(backend, arguments)
            if action == "type":
                return self._do_type(backend, arguments)
            if action == "key":
                return self._do_key(backend, arguments)
            if action == "set_value":
                return self._do_set_value(backend, arguments)
            if action == "list_apps":
                return self._do_list_apps(backend)
            if action == "focus_app":
                return self._do_focus_app(backend, arguments)

            return ToolResult(output=f"Unknown action: {action}", is_error=True)

        except Exception as exc:
            return ToolResult(output=f"Computer use error: {exc}", is_error=True)

    # ---- actions -------------------------------------------------------

    def _capture(self, backend: CUADriverBackend, args: ComputerUseInput) -> ToolResult:
        """Take a screenshot and optionally list elements."""
        result = backend.capture(mode=args.mode, app=args.app)

        # Save screenshot to download dir if we have base64 data.
        out_path = None
        if result.png_b64:
            out_dir = Path("/home/z/my-project/download")
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            out_path = out_dir / f"screenshot-{ts}.png"
            out_path.write_bytes(base64.b64decode(result.png_b64))

        lines = [
            f"Screenshot captured: {result.width}x{result.height}",
            f"  Mode: {result.mode}",
            f"  Backend: cua-driver",
        ]
        if out_path:
            lines.append(f"  Saved to: {out_path}")
        if result.elements:
            lines.append(f"  Elements: {len(result.elements)}")
            for el in result.elements[:30]:  # show first 30
                label = el.label[:60] if el.label else ""
                lines.append(f"    [{el.index}] {el.role}: {label}")
            if len(result.elements) > 30:
                lines.append(f"    ... ({len(result.elements) - 30} more)")

        return ToolResult(
            output="\n".join(lines),
            metadata={
                "width": result.width,
                "height": result.height,
                "path": str(out_path) if out_path else None,
                "mode": result.mode,
                "backend": "cua-driver",
                "element_count": len(result.elements),
            },
        )

    def _do_click(
        self, backend: CUADriverBackend, args: ComputerUseInput, *, click_count: int = 1, button: str = "left"
    ) -> ToolResult:
        x, y = self._extract_xy(args)
        result = backend.click(
            element=args.element,
            x=x,
            y=y,
            button=button if button != "left" else args.button,
            click_count=click_count,
            modifiers=args.modifiers,
        )
        return ToolResult(output=result.message, is_error=not result.ok)

    def _do_drag(self, backend: CUADriverBackend, args: ComputerUseInput) -> ToolResult:
        from_xy = tuple(args.from_coordinate) if args.from_coordinate else None
        to_xy = tuple(args.to_coordinate) if args.to_coordinate else None
        result = backend.drag(
            from_element=args.from_element,
            to_element=args.to_element,
            from_xy=from_xy,  # type: ignore[arg-type]
            to_xy=to_xy,  # type: ignore[arg-type]
            button=args.button,
            modifiers=args.modifiers,
        )
        return ToolResult(output=result.message, is_error=not result.ok)

    def _do_scroll(self, backend: CUADriverBackend, args: ComputerUseInput) -> ToolResult:
        x, y = self._extract_xy(args)
        result = backend.scroll(
            direction=args.direction,
            amount=args.amount,
            element=args.element,
            x=x,
            y=y,
            modifiers=args.modifiers,
        )
        return ToolResult(output=result.message, is_error=not result.ok)

    def _do_type(self, backend: CUADriverBackend, args: ComputerUseInput) -> ToolResult:
        if not args.text:
            return ToolResult(output="type requires text", is_error=True)
        result = backend.type_text(args.text)
        return ToolResult(output=result.message, is_error=not result.ok)

    def _do_key(self, backend: CUADriverBackend, args: ComputerUseInput) -> ToolResult:
        if not args.keys:
            return ToolResult(output="key requires a key combo (e.g. 'cmd+s', 'enter')", is_error=True)
        result = backend.key(args.keys)
        return ToolResult(output=result.message, is_error=not result.ok)

    def _do_set_value(self, backend: CUADriverBackend, args: ComputerUseInput) -> ToolResult:
        if args.value is None:
            return ToolResult(output="set_value requires a value", is_error=True)
        result = backend.set_value(args.value, element=args.element)
        return ToolResult(output=result.message, is_error=not result.ok)

    def _do_list_apps(self, backend: CUADriverBackend) -> ToolResult:
        apps = backend.list_apps()
        if not apps:
            return ToolResult(output="No apps found.")
        lines = [f"Running apps ({len(apps)}):"]
        for i, app in enumerate(apps, 1):
            name = app.get("name", "?")
            pid = app.get("pid", "")
            windows = app.get("window_count", "")
            lines.append(f"  {i}. {name} (pid={pid}, windows={windows})")
        return ToolResult(output="\n".join(lines), metadata={"apps": apps})

    def _do_focus_app(self, backend: CUADriverBackend, args: ComputerUseInput) -> ToolResult:
        if not args.app:
            return ToolResult(output="focus_app requires an app name", is_error=True)
        result = backend.focus_app(args.app, raise_window=args.raise_window)
        return ToolResult(output=result.message, is_error=not result.ok)

    # ---- helpers -------------------------------------------------------

    @staticmethod
    def _extract_xy(args: ComputerUseInput) -> tuple[int | None, int | None]:
        """Extract (x, y) from the coordinate field."""
        if args.coordinate and len(args.coordinate) >= 2:
            return args.coordinate[0], args.coordinate[1]
        return None, None
