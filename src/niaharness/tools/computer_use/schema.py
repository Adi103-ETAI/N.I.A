"""Schema for the computer_use tool.

Adapted from Hermes Agent's tools/computer_use/schema.py.
Model-agnostic — any tool-calling model can drive this. Vision-capable
models should prefer ``capture(mode='som')`` then ``click(element=N)``
— much more reliable than pixel coordinates.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ComputerUseInput(BaseModel):
    """Arguments for the computer_use tool.

    One consolidated tool with an ``action`` discriminator. Keeps the
    schema compact and the per-turn token cost low.
    """

    action: Literal[
        "capture",
        "click",
        "double_click",
        "right_click",
        "middle_click",
        "drag",
        "scroll",
        "type",
        "key",
        "set_value",
        "wait",
        "list_apps",
        "focus_app",
    ] = Field(
        description=(
            "Which action to perform. `capture` is free (no side effects). "
            "All other actions require approval unless auto-approved. Use "
            "`set_value` for select/popup elements and sliders — it selects "
            "the matching option directly without opening the native menu."
        )
    )

    # ── capture ────────────────────────────────────────────────────
    mode: Literal["som", "vision", "ax"] = Field(
        default="som",
        description=(
            "Capture mode. `som` (default) is a screenshot with numbered "
            "overlays on every interactable element plus the AX tree — best "
            "for vision models, lets you click by element index. `vision` is "
            "a plain screenshot. `ax` is the accessibility tree only."
        ),
    )
    app: str | None = Field(
        default=None,
        description=(
            "Optional. Limit capture/action to a specific app (by name, e.g. "
            "'Safari', or bundle ID 'com.apple.Safari'). If omitted, operates "
            "on the frontmost app's window. Pass app='screen' (or 'desktop') "
            "to capture the OS desktop/shell surface."
        ),
    )
    max_elements: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Optional cap on the AX elements array returned by capture.",
    )

    # ── click / drag / scroll targeting ────────────────────────────
    element: int | None = Field(
        default=None,
        description=(
            "The 1-based SOM index returned by the last capture(mode='som') "
            "call. Strongly preferred over raw coordinates."
        ),
    )
    coordinate: list[int] | None = Field(
        default=None,
        description=(
            "Pixel coordinates [x, y] in logical screen space (as returned "
            "by capture width/height). Only use if no element index is available."
        ),
    )
    button: Literal["left", "right", "middle"] = Field(
        default="left",
        description="Mouse button. Defaults to left.",
    )
    modifiers: list[
        Literal["cmd", "shift", "option", "alt", "ctrl", "fn", "win", "windows", "super", "meta"]
    ] | None = Field(
        default=None,
        description="Modifier keys held during the action.",
    )

    # ── drag ───────────────────────────────────────────────────────
    from_element: int | None = Field(default=None, description="Source element index (drag).")
    to_element: int | None = Field(default=None, description="Target element index (drag).")
    from_coordinate: list[int] | None = Field(
        default=None, description="Source [x,y] (drag; use when no element available)."
    )
    to_coordinate: list[int] | None = Field(
        default=None, description="Target [x,y] (drag; use when no element available)."
    )

    # ── scroll ─────────────────────────────────────────────────────
    direction: Literal["up", "down", "left", "right"] = Field(
        default="down",
        description="Scroll direction.",
    )
    amount: int = Field(
        default=3,
        ge=1,
        le=100,
        description="Scroll wheel ticks. Default 3.",
    )

    # ── set_value ──────────────────────────────────────────────────
    value: str | None = Field(
        default=None,
        description=(
            "For action='set_value': the value to set on the element. For "
            "AXPopUpButton / select dropdowns, pass the option's display "
            "label. For sliders, pass the numeric or string value."
        ),
    )

    # ── type / key / wait ──────────────────────────────────────────
    text: str | None = Field(
        default=None,
        description="Text to type (respects the current layout).",
    )
    keys: str | None = Field(
        default=None,
        description=(
            "Key combo, e.g. 'cmd+s', 'ctrl+alt+t', 'return', 'escape', 'tab'. "
            "Use '+' to combine."
        ),
    )
    seconds: float | None = Field(
        default=None,
        ge=0.1,
        le=30.0,
        description="Seconds to wait. Max 30.",
    )

    # ── focus_app ──────────────────────────────────────────────────
    raise_window: bool = Field(
        default=False,
        description=(
            "Only for action='focus_app'. If true, brings the window to front "
            "(DISRUPTS the user). Default false — input is routed to the app "
            "without raising, matching the background co-work model."
        ),
    )

    # ── return shape ───────────────────────────────────────────────
    capture_after: bool = Field(
        default=False,
        description=(
            "If true, take a follow-up capture after the action and include "
            "it in the response. Saves a round-trip when you need to verify "
            "an action's effect."
        ),
    )
