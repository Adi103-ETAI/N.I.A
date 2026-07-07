"""Computer Use backend abstraction.

Mirrors Hermes Agent's tools/computer_use/backend.py design: a pluggable
backend interface so computer_use can work across macOS, Windows, and Linux
without being tied to one automation library.

Two backends:
1. **CUADriverBackend** (primary) — speaks MCP over stdio to `cua-driver`,
   a cross-platform Rust binary that works on macOS, Windows, and Linux
   (X11 + XWayland). Runs in the background without stealing the user's
   cursor or keyboard focus. Install: https://github.com/trycua/cua
2. **PyAutoGUIBackend** (fallback) — direct PyAutoGUI. Works on all three
   platforms but steals focus and requires an active display session.
   Useful when cua-driver isn't installed.

The backend is selected automatically:
- If ``cua-driver`` is on $PATH (or ``NIA_CUA_DRIVER_CMD`` env var points
  to it), use CUADriverBackend.
- Else if pyautogui is importable, use PyAutoGUIBackend.
- Else, the tool returns a helpful install error.

Reference: Hermes Agent's tools/computer_use/ (backend.py, cua_backend.py).
"""

from __future__ import annotations

import abc
import base64
import io
import logging
import os
import shutil
import subprocess
import threading
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes (mirror Hermes's backend.py)
# ---------------------------------------------------------------------------


@dataclass
class UIElement:
    """One interactable element on the current screen."""

    index: int  # 1-based SOM index
    role: str = ""
    label: str = ""
    bounds: tuple[int, int, int, int] = (0, 0, 0, 0)  # x, y, w, h
    app: str = ""

    def center(self) -> tuple[int, int]:
        x, y, w, h = self.bounds
        return x + w // 2, y + h // 2


@dataclass
class CaptureResult:
    """Result of a screen capture."""

    mode: str  # "vision", "som", "ax"
    width: int = 0
    height: int = 0
    png_b64: str | None = None
    elements: list[UIElement] = field(default_factory=list)
    app: str = ""
    window_title: str = ""


@dataclass
class ActionResult:
    """Result of any action (click / type / scroll / drag / key)."""

    ok: bool
    action: str
    message: str = ""
    capture: CaptureResult | None = None


# ---------------------------------------------------------------------------
# Abstract backend
# ---------------------------------------------------------------------------


class ComputerUseBackend(abc.ABC):
    """Abstract backend interface for computer use."""

    @abc.abstractmethod
    def capture(
        self,
        *,
        mode: str = "vision",
        app: str | None = None,
    ) -> CaptureResult:
        """Capture the screen or a specific app window."""

    @abc.abstractmethod
    def click(self, x: int, y: int, *, button: str = "left") -> ActionResult:
        """Click at (x, y)."""

    @abc.abstractmethod
    def double_click(self, x: int, y: int) -> ActionResult:
        """Double-click at (x, y)."""

    @abc.abstractmethod
    def right_click(self, x: int, y: int) -> ActionResult:
        """Right-click at (x, y)."""

    @abc.abstractmethod
    def scroll(self, clicks: int, *, x: int | None = None, y: int | None = None) -> ActionResult:
        """Scroll by N clicks (positive=up, negative=down)."""

    @abc.abstractmethod
    def type_text(self, text: str) -> ActionResult:
        """Type a string of text."""

    @abc.abstractmethod
    def key(self, key: str) -> ActionResult:
        """Press a single key (enter, tab, escape, etc.)."""

    @abc.abstractmethod
    def key_combo(self, keys: str) -> ActionResult:
        """Press a key combination (e.g. 'ctrl+c')."""

    @abc.abstractmethod
    def drag(self, x1: int, y1: int, x2: int, y2: int) -> ActionResult:
        """Drag from (x1, y1) to (x2, y2)."""

    @abc.abstractmethod
    def list_apps(self) -> list[dict[str, Any]]:
        """List running applications."""

    @abc.abstractmethod
    def focus_app(self, app: str) -> ActionResult:
        """Focus an application by name."""

    @abc.abstractmethod
    def available(self) -> bool:
        """Return True if this backend is ready to use."""


# ---------------------------------------------------------------------------
# cua-driver backend (primary — cross-platform, background-safe)
# ---------------------------------------------------------------------------


_CUA_DRIVER_CMD = os.environ.get("NIA_CUA_DRIVER_CMD", "cua-driver")


def cua_driver_binary_available() -> bool:
    """True if cua-driver is on $PATH or NIA_CUA_DRIVER_CMD resolves."""
    return bool(shutil.which(_CUA_DRIVER_CMD))


class CUADriverBackend(ComputerUseBackend):
    """cua-driver backend — cross-platform (macOS, Windows, Linux).

    Speaks MCP over stdio to the ``cua-driver`` binary.  Works in the
    background without stealing the user's cursor or keyboard focus.

    Install cua-driver:
      macOS:   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/trycua/cua/main/libs/cua-driver/scripts/install.sh)"
      Windows: irm https://raw.githubusercontent.com/trycua/cua/main/libs/cua-driver/scripts/install.ps1 | iex
      Linux:   same as macOS (needs X11 or XWayland)
    """

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()

    def available(self) -> bool:
        return cua_driver_binary_available()

    def _ensure_process(self) -> subprocess.Popen:
        """Start the cua-driver MCP process if not running."""
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                return self._proc
            try:
                self._proc = subprocess.Popen(
                    [_CUA_DRIVER_CMD, "mcp"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                )
                return self._proc
            except FileNotFoundError as exc:
                raise RuntimeError(
                    f"cua-driver not found at {_CUA_DRIVER_CMD!r}. "
                    f"Install from https://github.com/trycua/cua"
                ) from exc

    def _call_mcp(self, tool_name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """Call a cua-driver MCP tool and return the result.

        This is a simplified JSON-RPC client. cua-driver's MCP server reads
        JSON-RPC requests from stdin and writes responses to stdout.
        """
        import json

        proc = self._ensure_process()
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
            proc.stdin.write(json.dumps(request) + "\n")
            proc.stdin.flush()
            line = proc.stdout.readline()
            if not line:
                raise RuntimeError("cua-driver returned empty response")
            response = json.loads(line)
            if "error" in response:
                raise RuntimeError(f"cua-driver error: {response['error']}")
            result = response.get("result", {})
            content = result.get("content", [])
            if content and isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        try:
                            return json.loads(block["text"])
                        except (json.JSONDecodeError, KeyError):
                            return {"text": block.get("text", "")}
            return result

    def capture(self, *, mode: str = "vision", app: str | None = None) -> CaptureResult:
        args: dict[str, Any] = {"mode": mode}
        if app:
            args["app"] = app
        result = self._call_mcp("screenshot", args)
        return CaptureResult(
            mode=mode,
            width=result.get("width", 0),
            height=result.get("height", 0),
            png_b64=result.get("image"),
            app=app or "",
        )

    def click(self, x: int, y: int, *, button: str = "left") -> ActionResult:
        result = self._call_mcp("click", {"x": x, "y": y, "button": button})
        return ActionResult(ok=True, action="click", message=result.get("message", f"Clicked ({x}, {y})"))

    def double_click(self, x: int, y: int) -> ActionResult:
        result = self._call_mcp("double_click", {"x": x, "y": y})
        return ActionResult(ok=True, action="double_click", message=result.get("message", f"Double-clicked ({x}, {y})"))

    def right_click(self, x: int, y: int) -> ActionResult:
        result = self._call_mcp("right_click", {"x": x, "y": y})
        return ActionResult(ok=True, action="right_click", message=result.get("message", f"Right-clicked ({x}, {y})"))

    def scroll(self, clicks: int, *, x: int | None = None, y: int | None = None) -> ActionResult:
        args: dict[str, Any] = {"clicks": clicks}
        if x is not None:
            args["x"] = x
        if y is not None:
            args["y"] = y
        result = self._call_mcp("scroll", args)
        direction = "up" if clicks > 0 else "down"
        return ActionResult(ok=True, action="scroll", message=result.get("message", f"Scrolled {direction} {abs(clicks)}"))

    def type_text(self, text: str) -> ActionResult:
        result = self._call_mcp("type_text", {"text": text})
        return ActionResult(ok=True, action="type", message=result.get("message", f"Typed {len(text)} chars"))

    def key(self, key: str) -> ActionResult:
        result = self._call_mcp("hotkey", {"keys": [key]})
        return ActionResult(ok=True, action="key", message=result.get("message", f"Pressed {key}"))

    def key_combo(self, keys: str) -> ActionResult:
        key_list = [k.strip() for k in keys.split("+")]
        result = self._call_mcp("hotkey", {"keys": key_list})
        return ActionResult(ok=True, action="key_combo", message=result.get("message", f"Pressed {keys}"))

    def drag(self, x1: int, y1: int, x2: int, y2: int) -> ActionResult:
        result = self._call_mcp("drag", {"from_x": x1, "from_y": y1, "to_x": x2, "to_y": y2})
        return ActionResult(ok=True, action="drag", message=result.get("message", f"Dragged ({x1},{y1}) → ({x2},{y2})"))

    def list_apps(self) -> list[dict[str, Any]]:
        result = self._call_mcp("list_apps", {})
        return result.get("apps", []) if isinstance(result, dict) else []

    def focus_app(self, app: str) -> ActionResult:
        result = self._call_mcp("launch_app", {"app": app})
        return ActionResult(ok=True, action="focus_app", message=result.get("message", f"Focused {app}"))


# ---------------------------------------------------------------------------
# PyAutoGUI backend (fallback — works everywhere but steals focus)
# ---------------------------------------------------------------------------


class PyAutoGUIBackend(ComputerUseBackend):
    """PyAutoGUI fallback backend.

    Works on macOS, Windows, and Linux (X11) but steals the user's cursor
    and keyboard focus.  Useful when cua-driver isn't installed.

    Requires:
    - pip install pyautogui
    - An active display session (X11 on Linux, or desktop on macOS/Windows)
    - On Linux: DISPLAY env var set (or use Xvfb for headless)
    """

    def __init__(self) -> None:
        self._pg = None

    def available(self) -> bool:
        try:
            self._get_pg()
            return True
        except RuntimeError:
            return False

    def _get_pg(self):
        if self._pg is not None:
            return self._pg
        try:
            import pyautogui  # type: ignore

            pyautogui.FAILSAFE = True
            pyautogui.PAUSE = 0.1
            self._pg = pyautogui
            return pyautogui
        except ImportError:
            raise RuntimeError(
                "pyautogui is not installed. Install with: pip install pyautogui"
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not initialize pyautogui: {exc}. "
                f"If running headless on Linux, install Xvfb and run: "
                f"xvfb-run python -m niaharness"
            ) from exc

    def capture(self, *, mode: str = "vision", app: str | None = None) -> CaptureResult:
        pg = self._get_pg()
        screenshot = pg.screenshot()
        buf = io.BytesIO()
        screenshot.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return CaptureResult(
            mode=mode,
            width=screenshot.width,
            height=screenshot.height,
            png_b64=b64,
        )

    def click(self, x: int, y: int, *, button: str = "left") -> ActionResult:
        pg = self._get_pg()
        pg.click(x, y, button=button)
        return ActionResult(ok=True, action="click", message=f"Clicked ({x}, {y})")

    def double_click(self, x: int, y: int) -> ActionResult:
        pg = self._get_pg()
        pg.doubleClick(x, y)
        return ActionResult(ok=True, action="double_click", message=f"Double-clicked ({x}, {y})")

    def right_click(self, x: int, y: int) -> ActionResult:
        pg = self._get_pg()
        pg.rightClick(x, y)
        return ActionResult(ok=True, action="right_click", message=f"Right-clicked ({x}, {y})")

    def scroll(self, clicks: int, *, x: int | None = None, y: int | None = None) -> ActionResult:
        pg = self._get_pg()
        pg.scroll(clicks)
        direction = "up" if clicks > 0 else "down"
        return ActionResult(ok=True, action="scroll", message=f"Scrolled {direction} {abs(clicks)}")

    def type_text(self, text: str) -> ActionResult:
        pg = self._get_pg()
        pg.typewrite(text, interval=0.02)
        return ActionResult(ok=True, action="type", message=f"Typed {len(text)} chars")

    def key(self, key: str) -> ActionResult:
        pg = self._get_pg()
        pg.press(key)
        return ActionResult(ok=True, action="key", message=f"Pressed {key}")

    def key_combo(self, keys: str) -> ActionResult:
        pg = self._get_pg()
        key_list = [k.strip() for k in keys.split("+")]
        pg.hotkey(*key_list)
        return ActionResult(ok=True, action="key_combo", message=f"Pressed {keys}")

    def drag(self, x1: int, y1: int, x2: int, y2: int) -> ActionResult:
        pg = self._get_pg()
        pg.moveTo(x1, y1)
        pg.dragTo(x2, y2, duration=0.5)
        return ActionResult(ok=True, action="drag", message=f"Dragged ({x1},{y1}) → ({x2},{y2})")

    def list_apps(self) -> list[dict[str, Any]]:
        pg = self._get_pg()
        try:
            windows = pg.getAllWindows()
            return [
                {"name": w.title, "left": w.left, "top": w.top, "width": w.width, "height": w.height}
                for w in windows
                if w.title
            ]
        except Exception:
            return []

    def focus_app(self, app: str) -> ActionResult:
        pg = self._get_pg()
        try:
            for w in pg.getAllWindows():
                if app.lower() in w.title.lower():
                    w.activate()
                    return ActionResult(ok=True, action="focus_app", message=f"Focused {w.title}")
            return ActionResult(ok=False, action="focus_app", message=f"Window not found: {app}")
        except Exception:
            return ActionResult(ok=False, action="focus_app", message=f"Could not focus {app}")


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------


_backend: ComputerUseBackend | None = None
_backend_lock = threading.Lock()


def get_backend() -> ComputerUseBackend:
    """Return the best available computer use backend.

    Resolution order:
    1. cua-driver (if on $PATH or NIA_CUA_DRIVER_CMD) — cross-platform, background-safe.
    2. pyautogui (if importable) — fallback, steals focus.
    3. Raise RuntimeError with install instructions.
    """
    global _backend
    if _backend is not None:
        return _backend
    with _backend_lock:
        if _backend is not None:
            return _backend
        # Try cua-driver first.
        cua = CUADriverBackend()
        if cua.available():
            logger.info("computer_use: using cua-driver backend")
            _backend = cua
            return _backend
        # Fall back to pyautogui.
        pg_backend = PyAutoGUIBackend()
        if pg_backend.available():
            logger.info("computer_use: using pyautogui backend (fallback)")
            _backend = pg_backend
            return _backend
        raise RuntimeError(
            "No computer use backend available.\n"
            "Install one of:\n"
            "  1. cua-driver (recommended, cross-platform, background-safe):\n"
            "     https://github.com/trycua/cua\n"
            "  2. pyautogui (fallback, steals focus):\n"
            "     pip install pyautogui"
        )


def get_backend_name() -> str:
    """Return the name of the active backend, or 'none' if unavailable."""
    try:
        backend = get_backend()
        if isinstance(backend, CUADriverBackend):
            return "cua-driver"
        if isinstance(backend, PyAutoGUIBackend):
            return "pyautogui"
        return "unknown"
    except RuntimeError:
        return "none"


def reset_backend() -> None:
    """Reset the cached backend (for tests)."""
    global _backend
    _backend = None
