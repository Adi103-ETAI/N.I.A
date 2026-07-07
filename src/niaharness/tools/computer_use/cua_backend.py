"""cua-driver backend — cross-platform (macOS, Windows, Linux).

Adapted from Hermes Agent's tools/computer_use/cua_backend.py.

Speaks MCP over stdio to ``cua-driver``, a cross-platform Rust binary that
works on macOS, Windows, and Linux (X11 + XWayland). Runs in the background
without stealing the user's cursor or keyboard focus.

Install cua-driver:
  macOS:   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/trycua/cua/main/libs/cua-driver/scripts/install.sh)"
  Windows: irm https://raw.githubusercontent.com/trycua/cua/main/libs/cua-driver/scripts/install.ps1 | iex
  Linux:   same as macOS (needs X11 or XWayland)

The macOS path uses private SkyLight SPIs. The Windows path uses stable
Win32 APIs (SendInput + UI Automation). Linux uses X11 today (Wayland via
XWayland; pure-Wayland progress tracked upstream).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import threading
from typing import Any, Optional

from niaharness.tools.computer_use.backend import (
    ActionResult,
    CaptureResult,
    ComputerUseBackend,
    UIElement,
)

logger = logging.getLogger(__name__)

# Allow override via env var (mirrors Hermes's HERMES_CUA_DRIVER_CMD).
_CUA_DRIVER_CMD = os.environ.get("NIA_CUA_DRIVER_CMD", "cua-driver")
_CUA_DRIVER_ARGS = ["mcp"]  # stdio MCP transport

# Whole-screen / desktop capture sentinels (mirrors Hermes).
_SCREEN_CAPTURE_SENTINELS = {"screen", "desktop", "fullscreen", "full screen", "all"}


def cua_driver_binary_available() -> bool:
    """True if cua-driver is on $PATH or NIA_CUA_DRIVER_CMD resolves."""
    return bool(shutil.which(_CUA_DRIVER_CMD))


class CUADriverBackend(ComputerUseBackend):
    """cua-driver backend — cross-platform, background-safe.

    Speaks MCP over stdio to the ``cua-driver`` binary. Works in the
    background without stealing the user's cursor or keyboard focus.
    """

    def __init__(self) -> None:
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._started = False

    # ── lifecycle ───────────────────────────────────────────────────

    def start(self) -> None:
        """Start the cua-driver MCP process."""
        if self._started:
            return
        with self._lock:
            if self._started:
                return
            try:
                self._proc = subprocess.Popen(
                    [_CUA_DRIVER_CMD, *_CUA_DRIVER_ARGS],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                )
                self._started = True
                logger.info("cua-driver MCP process started (pid=%d)", self._proc.pid)
            except FileNotFoundError as exc:
                raise RuntimeError(
                    f"cua-driver not found at {_CUA_DRIVER_CMD!r}. "
                    f"Install from https://github.com/trycua/cua"
                ) from exc

    def stop(self) -> None:
        """Stop the cua-driver MCP process."""
        with self._lock:
            if self._proc is not None:
                try:
                    self._proc.terminate()
                    self._proc.wait(timeout=5)
                except Exception:
                    try:
                        self._proc.kill()
                    except Exception:
                        pass
                self._proc = None
                self._started = False

    def is_available(self) -> bool:
        return cua_driver_binary_available()

    # ── MCP communication ───────────────────────────────────────────

    def _call_mcp(self, tool_name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """Call a cua-driver MCP tool and return the parsed result.

        Speaks JSON-RPC 2.0 over stdio. cua-driver's MCP server reads
        requests from stdin and writes responses to stdout.
        """
        if not self._started:
            self.start()

        assert self._proc is not None
        assert self._proc.stdin is not None
        assert self._proc.stdout is not None

        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments or {},
            },
        }

        with self._lock:
            self._proc.stdin.write(json.dumps(request) + "\n")
            self._proc.stdin.flush()
            line = self._proc.stdout.readline()
            if not line:
                # Check stderr for error info.
                stderr = ""
                if self._proc.stderr:
                    try:
                        stderr = self._proc.stderr.read()
                    except Exception:
                        pass
                raise RuntimeError(
                    f"cua-driver returned empty response. stderr: {stderr[:500]}"
                )
            response = json.loads(line)
            if "error" in response:
                err = response["error"]
                raise RuntimeError(f"cua-driver error: {err}")
            result = response.get("result", {})
            # MCP returns content as a list of content blocks.
            content = result.get("content", [])
            if content and isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            try:
                                return json.loads(block["text"])
                            except (json.JSONDecodeError, KeyError):
                                return {"text": block.get("text", "")}
                        elif block.get("type") == "image":
                            return {"image": block.get("data"), "mimeType": block.get("mimeType")}
            return result

    # ── capture ─────────────────────────────────────────────────────

    def capture(self, mode: str = "som", app: Optional[str] = None) -> CaptureResult:
        """Capture the screen or a specific app window.

        Modes:
        - ``som`` (default): screenshot with numbered overlays + AX tree
        - ``vision``: plain screenshot only
        - ``ax``: accessibility tree only (no image)
        """
        args: dict[str, Any] = {"mode": mode}
        if app:
            args["app"] = app

        result = self._call_mcp("screenshot", args)

        # Parse elements if present.
        elements: list[UIElement] = []
        raw_elements = result.get("elements", [])
        if isinstance(raw_elements, list):
            for i, el in enumerate(raw_elements, 1):
                if not isinstance(el, dict):
                    continue
                elements.append(
                    UIElement(
                        index=el.get("index", i),
                        role=el.get("role", ""),
                        label=el.get("label", ""),
                        bounds=tuple(el.get("bounds", (0, 0, 0, 0))),
                        app=el.get("app", ""),
                        pid=el.get("pid", 0),
                        window_id=el.get("window_id", 0),
                        element_token=el.get("element_token"),
                    )
                )

        png_b64 = result.get("image") or result.get("png_b64")
        png_bytes_len = len(png_b64) * 3 // 4 if png_b64 else 0

        return CaptureResult(
            mode=mode,
            width=result.get("width", 0),
            height=result.get("height", 0),
            png_b64=png_b64,
            elements=elements,
            app=app or result.get("app", ""),
            window_title=result.get("window_title", ""),
            png_bytes_len=png_bytes_len,
            image_mime_type=result.get("mimeType"),
        )

    # ── pointer actions ─────────────────────────────────────────────

    def click(
        self,
        *,
        element: Optional[int] = None,
        x: Optional[int] = None,
        y: Optional[int] = None,
        button: str = "left",
        click_count: int = 1,
        modifiers: Optional[list[str]] = None,
    ) -> ActionResult:
        args: dict[str, Any] = {"button": button, "click_count": click_count}
        if element is not None:
            args["element"] = element
        elif x is not None and y is not None:
            args["x"] = x
            args["y"] = y
        else:
            return ActionResult(ok=False, action="click", message="click requires element or x+y")
        if modifiers:
            args["modifiers"] = modifiers

        result = self._call_mcp("click", args)
        return ActionResult(
            ok=True,
            action="click",
            message=result.get("message", f"Clicked {'element ' + str(element) if element else f'({x}, {y})'}"),
        )

    def drag(
        self,
        *,
        from_element: Optional[int] = None,
        to_element: Optional[int] = None,
        from_xy: Optional[tuple[int, int]] = None,
        to_xy: Optional[tuple[int, int]] = None,
        button: str = "left",
        modifiers: Optional[list[str]] = None,
    ) -> ActionResult:
        args: dict[str, Any] = {"button": button}
        if from_element is not None:
            args["from_element"] = from_element
        elif from_xy is not None:
            args["from_x"], args["from_y"] = from_xy
        else:
            return ActionResult(ok=False, action="drag", message="drag requires from_element or from_xy")
        if to_element is not None:
            args["to_element"] = to_element
        elif to_xy is not None:
            args["to_x"], args["to_y"] = to_xy
        else:
            return ActionResult(ok=False, action="drag", message="drag requires to_element or to_xy")
        if modifiers:
            args["modifiers"] = modifiers

        result = self._call_mcp("drag", args)
        return ActionResult(ok=True, action="drag", message=result.get("message", "Drag complete"))

    def scroll(
        self,
        *,
        direction: str,
        amount: int = 3,
        element: Optional[int] = None,
        x: Optional[int] = None,
        y: Optional[int] = None,
        modifiers: Optional[list[str]] = None,
    ) -> ActionResult:
        args: dict[str, Any] = {"direction": direction, "amount": amount}
        if element is not None:
            args["element"] = element
        elif x is not None and y is not None:
            args["x"] = x
            args["y"] = y
        if modifiers:
            args["modifiers"] = modifiers

        result = self._call_mcp("scroll", args)
        return ActionResult(
            ok=True, action="scroll", message=result.get("message", f"Scrolled {direction} {amount}")
        )

    # ── keyboard ────────────────────────────────────────────────────

    def type_text(self, text: str) -> ActionResult:
        result = self._call_mcp("type_text", {"text": text})
        return ActionResult(ok=True, action="type", message=result.get("message", f"Typed {len(text)} chars"))

    def key(self, keys: str) -> ActionResult:
        """Send a key combo, e.g. 'cmd+s', 'ctrl+alt+t', 'return'."""
        key_list = [k.strip() for k in keys.split("+")]
        result = self._call_mcp("hotkey", {"keys": key_list})
        return ActionResult(ok=True, action="key", message=result.get("message", f"Pressed {keys}"))

    # ── introspection ───────────────────────────────────────────────

    def list_apps(self) -> list[dict[str, Any]]:
        result = self._call_mcp("list_apps", {})
        apps = result.get("apps", [])
        return apps if isinstance(apps, list) else []

    def focus_app(self, app: str, raise_window: bool = False) -> ActionResult:
        args: dict[str, Any] = {"app": app, "raise_window": raise_window}
        result = self._call_mcp("focus_app", args)
        return ActionResult(ok=True, action="focus_app", message=result.get("message", f"Focused {app}"))

    # ── native-value mutation ───────────────────────────────────────

    def set_value(self, value: str, element: Optional[int] = None) -> ActionResult:
        args: dict[str, Any] = {"value": value}
        if element is not None:
            args["element"] = element
        result = self._call_mcp("set_value", args)
        return ActionResult(ok=True, action="set_value", message=result.get("message", f"Set value: {value}"))


# ---------------------------------------------------------------------------
# Backend singleton
# ---------------------------------------------------------------------------

_backend: Optional[CUADriverBackend] = None
_backend_lock = threading.Lock()


def get_backend() -> CUADriverBackend:
    """Return the cua-driver backend singleton.

    Raises RuntimeError if cua-driver is not installed.
    """
    global _backend
    if _backend is not None:
        return _backend
    with _backend_lock:
        if _backend is not None:
            return _backend
        if not cua_driver_binary_available():
            raise RuntimeError(
                "cua-driver is not installed. NIA's computer_use tool requires "
                "cua-driver — a cross-platform Rust binary that works on macOS, "
                "Windows, and Linux (X11 + XWayland) without stealing the user's "
                "cursor or keyboard focus.\n\n"
                "Install cua-driver:\n"
                "  macOS:   /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/trycua/cua/main/libs/cua-driver/scripts/install.sh)\"\n"
                "  Windows: irm https://raw.githubusercontent.com/trycua/cua/main/libs/cua-driver/scripts/install.ps1 | iex\n"
                "  Linux:   same as macOS (needs X11 or XWayland)\n\n"
                "Or set NIA_CUA_DRIVER_CMD to point to an existing binary.\n"
                "Source: https://github.com/trycua/cua"
            )
        _backend = CUADriverBackend()
        logger.info("computer_use: using cua-driver backend")
        return _backend


def get_backend_name() -> str:
    """Return 'cua-driver' if available, 'none' otherwise."""
    return "cua-driver" if cua_driver_binary_available() else "none"


def reset_backend() -> None:
    """Reset the cached backend (for tests)."""
    global _backend
    if _backend is not None:
        _backend.stop()
    _backend = None
