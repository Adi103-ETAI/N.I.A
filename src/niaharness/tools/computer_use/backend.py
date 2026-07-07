"""Abstract backend interface for computer use.

Adapted from Hermes Agent's tools/computer_use/backend.py.
Any implementation (cua-driver over MCP, future backends) must return
the shape described below. All methods synchronous; async is handled
inside the backend implementation if needed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class UIElement:
    """One interactable element on the current screen."""

    index: int  # 1-based SOM index
    role: str  # AX role (AXButton, AXTextField, ...)
    label: str = ""  # AXTitle / AXDescription / AXValue snippet
    bounds: tuple[int, int, int, int] = (0, 0, 0, 0)  # x, y, w, h (logical px)
    app: str = ""  # owning bundle ID or app name
    pid: int = 0  # owning process PID
    window_id: int = 0  # SkyLight / CG window ID
    attributes: dict[str, Any] = field(default_factory=dict)
    element_token: Optional[str] = None  # opaque per-snapshot handle

    def center(self) -> tuple[int, int]:
        x, y, w, h = self.bounds
        return x + w // 2, y + h // 2


@dataclass
class CaptureResult:
    """Result of a screen capture call.

    At least one of png_b64 / elements is populated depending on capture mode:
      * mode="vision" → png_b64 only
      * mode="ax"     → elements only
      * mode="som"    → both (default): PNG has numbered overlays + elements
    """

    mode: str
    width: int = 0  # screenshot width (logical px)
    height: int = 0
    png_b64: Optional[str] = None
    elements: list[UIElement] = field(default_factory=list)
    app: str = ""
    window_title: str = ""
    png_bytes_len: int = 0
    image_mime_type: Optional[str] = None


@dataclass
class ActionResult:
    """Result of any action (click / type / scroll / drag / key / wait)."""

    ok: bool
    action: str
    message: str = ""  # human-readable summary
    capture: Optional[CaptureResult] = None  # optional trailing screenshot
    meta: dict[str, Any] = field(default_factory=dict)


class ComputerUseBackend(ABC):
    """Lifecycle: ``start()`` before first use, ``stop()`` at shutdown."""

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if the backend can be used on this host right now."""

    # ── Capture ─────────────────────────────────────────────────────
    @abstractmethod
    def capture(self, mode: str = "som", app: Optional[str] = None) -> CaptureResult: ...

    # ── Pointer actions ─────────────────────────────────────────────
    @abstractmethod
    def click(
        self,
        *,
        element: Optional[int] = None,
        x: Optional[int] = None,
        y: Optional[int] = None,
        button: str = "left",
        click_count: int = 1,
        modifiers: Optional[list[str]] = None,
    ) -> ActionResult: ...

    @abstractmethod
    def drag(
        self,
        *,
        from_element: Optional[int] = None,
        to_element: Optional[int] = None,
        from_xy: Optional[tuple[int, int]] = None,
        to_xy: Optional[tuple[int, int]] = None,
        button: str = "left",
        modifiers: Optional[list[str]] = None,
    ) -> ActionResult: ...

    @abstractmethod
    def scroll(
        self,
        *,
        direction: str,  # up | down | left | right
        amount: int = 3,  # wheel ticks
        element: Optional[int] = None,
        x: Optional[int] = None,
        y: Optional[int] = None,
        modifiers: Optional[list[str]] = None,
    ) -> ActionResult: ...

    # ── Keyboard ────────────────────────────────────────────────────
    @abstractmethod
    def type_text(self, text: str) -> ActionResult: ...

    @abstractmethod
    def key(self, keys: str) -> ActionResult:
        """Send a key combo, e.g. 'cmd+s', 'ctrl+alt+t', 'return'."""

    # ── Introspection ───────────────────────────────────────────────
    @abstractmethod
    def list_apps(self) -> list[dict[str, Any]]:
        """Return running apps with bundle IDs, PIDs, window counts."""

    @abstractmethod
    def focus_app(self, app: str, raise_window: bool = False) -> ActionResult:
        """Route input to ``app`` (by name or bundle ID)."""

    # ── Native-value mutation ────────────────────────────────────────
    @abstractmethod
    def set_value(self, value: str, element: Optional[int] = None) -> ActionResult:
        """Set a native value on an element (e.g. AXPopUpButton selection)."""

    # ── Timing ──────────────────────────────────────────────────────
    def wait(self, seconds: float) -> ActionResult:
        """Default implementation: time.sleep."""
        import time

        time.sleep(max(0.0, min(seconds, 30.0)))
        return ActionResult(ok=True, action="wait", message=f"waited {seconds:.2f}s")
