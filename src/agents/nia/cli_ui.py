"""N.I.A CLI UI — medical-themed terminal interface with NIA caduceus.

Layout (ported from NIA's banner.py):
  - Left column: golden caduceus (braille art)
  - Right column: NIA block letters + model + built by + tools/skills

Features:
  - NIA-style caduceus (braille art, gold/orange/bronze gradient)
  - NIA name in block letters
  - Two-column layout: caduceus on left, info on right
  - Flicker-free streaming output (buffered rendering)
  - Color scheme: gold + orange + bronze on black
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Optional

from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.rule import Rule
from rich.align import Align
from rich.theme import Theme

# ---------------------------------------------------------------------------
# Color theme
# ---------------------------------------------------------------------------

NIA_THEME = Theme({
    "nia.gold": "#FFD700",
    "nia.gold.dim": "#B8860B",
    "nia.orange": "#FF8C00",
    "nia.bronze": "#CD7F32",
    "nia.cyan": "#00CED1",
    "nia.green": "#50FA7B",
    "nia.red": "#FF5555",
    "nia.gray": "#6272A4",
    "nia.white": "#F8F8F2",
    "nia.purple": "#BD93F9",
})

console = Console(theme=NIA_THEME, force_terminal=True)

# ---------------------------------------------------------------------------
# ASCII Art — NIA caduceus (braille art, ported from banner.py)
# ---------------------------------------------------------------------------

# Each line has a color. Gradient: bronze → orange → gold → orange → bronze → dark gold
CADUCEUS_LINES = [
    ("⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⡀⠀⣀⣀⠀⢀⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀", "#CD7F32"),
    ("⠀⠀⠀⠀⠀⠀⢀⣠⣴⣾⣿⣿⣇⠸⣿⣿⠇⣸⣿⣿⣷⣦⣄⡀⠀⠀⠀⠀⠀⠀", "#CD7F32"),
    ("⠀⢀⣠⣴⣶⠿⠋⣩⡿⣿⡿⠻⣿⡇⢠⡄⢸⣿⠟⢿⣿⢿⣍⠙⠿⣶⣦⣄⡀⠀", "#FFBF00"),
    ("⠀⠀⠉⠉⠁⠶⠟⠋⠀⠉⠀⢀⣈⣁⡈⢁⣈⣁⡀⠀⠉⠀⠙⠻⠶⠈⠉⠉⠀⠀", "#FFBF00"),
    ("⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣴⣿⡿⠛⢁⡈⠛⢿⣿⣦⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀", "#FFD700"),
    ("⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠿⣿⣦⣤⣈⠁⢠⣴⣿⠿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀", "#FFD700"),
    ("⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠉⠻⢿⣿⣦⡉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀", "#FFBF00"),
    ("⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⢷⣦⣈⠛⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀", "#FFBF00"),
    ("⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⣴⠦⠈⠙⠿⣦⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀", "#CD7F32"),
    ("⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠸⣿⣤⡈⠁⢤⣿⠇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀", "#CD7F32"),
    ("⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠛⠷⠄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀", "#B8860B"),
    ("⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⠑⢶⣄⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀", "#B8860B"),
    ("⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⠁⢰⡆⠈⡿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀", "#B8860B"),
    ("⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠳⠈⣡⠞⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀", "#B8860B"),
    ("⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀", "#B8860B"),
]

# NIA in block letters (same style as NIA logo)
NIA_LOGO_LINES = [
    ("██╗  ██╗███████╗██████╗ ███╗   ███╗███████╗███████╗", "#FFD700"),
    ("██║  ██║██╔════╝██╔══██╗████╗ ████║██╔════╝██╔════╝", "#FFD700"),
    ("███████║█████╗  ██████╔╝██╔████╔██║█████╗  ███████╗", "#FFBF00"),
    ("██╔══██║██╔══╝  ██╔══██╗██║╚██╔╝██║██╔══╝  ╚════██║", "#FFBF00"),
    ("██║  ██║███████╗██║  ██║██║ ╚═╝ ██║███████╗███████║", "#CD7F32"),
    ("╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝╚══════╝", "#CD7F32"),
]


def render_banner(
    model: str = "",
    provider: str = "",
    session_id: str = "",
    cwd: str = "",
    tool_count: int = 0,
    skill_count: int = 0,
    built_by: str = "Aditya",
    version: str = "0.1.0",
) -> Table:
    """Render the two-column banner: caduceus on left, NIA + info on right.

    Ported from NIA's build_welcome_banner() layout.
    """
    # Build the two-column grid layout (like NIA Table.grid)
    layout = Table.grid(padding=(0, 3))
    layout.add_column(justify="center", vertical="middle")  # Left: caduceus
    layout.add_column(justify="left", vertical="top")       # Right: NIA + info

    # Left column: caduceus
    caduceus_text = Text()
    for line, color in CADUCEUS_LINES:
        caduceus_text.append(line + "\n", style=color)

    # Right column: NIA name + model + built by + tools/skills
    right_text = Text()

    # NIA block letters
    for line, color in NIA_LOGO_LINES:
        right_text.append(line + "\n", style=f"bold {color}")

    right_text.append("\n")

    # Model + provider
    if model:
        model_short = model.split("/")[-1] if "/" in model else model
        if len(model_short) > 28:
            model_short = model_short[:25] + "..."
        right_text.append(model_short, style="#FFD700 bold")
        if provider:
            right_text.append(f" · {provider}", style="#6272A4")
        right_text.append("\n")

    # Built by
    right_text.append("Built by ", style="#6272A4")
    right_text.append(built_by, style="#FF8C00 bold")
    right_text.append(f" · v{version}", style="#6272A4")
    right_text.append("\n\n")

    # Tools and skills
    right_text.append(str(tool_count), style="#FF8C00 bold")
    right_text.append(" tools", style="#6272A4")
    right_text.append(" · ", style="#6272A4")
    right_text.append(str(skill_count), style="#BD93F9 bold")
    right_text.append(" skills", style="#6272A4")
    right_text.append("\n\n")

    # Session + path
    if session_id:
        right_text.append(f"Session: {session_id}\n", style="#6272A4")
    if cwd:
        right_text.append(f"{cwd}\n", style="#6272A4")

    layout.add_row(caduceus_text, right_text)
    return layout


def render_startup_screen(
    model: str,
    session_id: str,
    cwd: str,
    provider: str = "",
    tool_count: int = 0,
    skill_count: int = 0,
) -> None:
    """Render the full startup screen with the two-column banner."""
    console.clear()

    # Two-column banner
    console.print(render_banner(
        model=model,
        provider=provider,
        session_id=session_id,
        cwd=cwd,
        tool_count=tool_count,
        skill_count=skill_count,
    ))

    console.print()
    console.print(Rule(style="#B8860B"))
    console.print()

    # Command hints
    hints = Text(
        "  /help for commands  ·  /tools to list  ·  /skills to browse  ·  /model to switch  ·  Ctrl+C exit",
        style="#6272A4",
    )
    console.print(hints)
    console.print()
    console.print(Text("  Type your message below. Press Enter to send.", style="#6272A4"))
    console.print()


def render_input_prompt() -> None:
    """Render the input prompt area with orange accent line."""
    console.print(Rule(style="#FF8C00"))
    console.print(Text("> ", style="#FF8C00 bold"), end="")


def render_tool_start(tool_name: str) -> None:
    """Render a tool execution start indicator."""
    console.print(Text(f"  ⚡ {tool_name}", style="#FF8C00"), end="")


def render_tool_complete(tool_name: str, is_error: bool = False) -> None:
    """Render a tool execution completion indicator."""
    if is_error:
        console.print(Text(" ✗", style="#FF5555"))
    else:
        console.print(Text(" ✓", style="#50FA7B"))


class StreamingRenderer:
    """Flicker-free streaming text renderer.

    Buffers text and writes in chunks to minimize terminal redraws.
    Uses sys.stdout.write + flush instead of rich.Live to avoid the
    flickering that comes from rich.Live's full-screen refresh.
    """

    def __init__(self) -> None:
        self._buffer: list[str] = []
        self._last_flush = time.monotonic()
        self._flush_interval = 0.05  # 50ms

    def add_text(self, text: str) -> None:
        """Add text to the buffer and flush if needed."""
        self._buffer.append(text)
        now = time.monotonic()
        if now - self._last_flush >= self._flush_interval:
            self._flush()

    def _flush(self) -> None:
        """Flush buffered text to stdout."""
        if not self._buffer:
            return
        text = "".join(self._buffer)
        self._buffer.clear()
        sys.stdout.write(text)
        sys.stdout.flush()
        self._last_flush = time.monotonic()

    def finish(self) -> None:
        """Flush any remaining buffered text and add a newline."""
        self._flush()
        sys.stdout.write("\n")
        sys.stdout.flush()


# ---------------------------------------------------------------------------
# Interactive REPL
# ---------------------------------------------------------------------------


async def run_interactive(nia) -> None:
    """Run the interactive REPL with the NIA UI.

    Args:
        nia: An initialized NIA instance.
    """
    from niaharness.engine.stream_events import (
        AssistantTextDelta,
        AssistantTurnComplete,
        ToolExecutionStarted,
        ToolExecutionCompleted,
    )

    # Gather session info
    model = nia.engine._model if nia.engine else "unknown"
    session_id = nia.engine._session_id if nia.engine else "unknown"
    cwd = nia._working_directory
    provider = ""

    try:
        from niaharness.api.provider import detect_provider
        from niaharness.config.settings import load_settings
        provider_info = detect_provider(load_settings())
        provider = provider_info.name
    except Exception:
        pass

    # Count tools and skills
    tool_count = 0
    skill_count = 0
    try:
        tool_count = len(nia.engine._tool_registry._tools)
    except Exception:
        pass
    try:
        from niaharness.tools.skills_loader import load_skill_registry
        skill_count = len(load_skill_registry().list_skills())
    except Exception:
        pass

    # Render startup screen
    render_startup_screen(
        model=model,
        session_id=session_id,
        cwd=cwd,
        provider=provider,
        tool_count=tool_count,
        skill_count=skill_count,
    )

    # Interactive loop
    while True:
        try:
            render_input_prompt()
            user_input = input().strip()
        except (KeyboardInterrupt, EOFError):
            console.print(Text("\n  Goodbye.", style="#FFD700"))
            break

        if not user_input:
            continue

        # Handle slash commands
        if user_input.startswith("/"):
            if user_input.lower() in ("/exit", "/quit", "exit", "quit"):
                console.print(Text("  Goodbye.", style="#FFD700"))
                break
            if user_input.lower() in ("/help", "/h"):
                console.print(Text("\n  Commands:", style="#FFD700 bold"))
                console.print(Text("    /help     — Show this help", style="#6272A4"))
                console.print(Text("    /tools    — List available tools", style="#6272A4"))
                console.print(Text("    /skills   — List available skills", style="#6272A4"))
                console.print(Text("    /model    — Show or switch model", style="#6272A4"))
                console.print(Text("    /insights — Show usage analytics", style="#6272A4"))
                console.print(Text("    /profile  — Manage profiles", style="#6272A4"))
                console.print(Text("    /oauth    — Anthropic OAuth login", style="#6272A4"))
                console.print(Text("    /soul     — Edit NIA's identity", style="#6272A4"))
                console.print(Text("    /exit     — Quit NIA\n", style="#6272A4"))
                continue
            if user_input.lower() in ("/tools", "/t"):
                try:
                    tools = sorted(nia.engine._tool_registry._tools.keys())
                    console.print(Text(f"\n  Available tools ({len(tools)}):", style="#FF8C00 bold"))
                    for i in range(0, len(tools), 4):
                        row = tools[i:i+4]
                        console.print(Text("    " + "  ".join(f"{t:<20}" for t in row), style="#F8F8F2"))
                    console.print()
                except Exception:
                    pass
                continue
            if user_input.lower() in ("/skills", "/sk"):
                try:
                    from niaharness.tools.skills_loader import load_skill_registry
                    skills = sorted(load_skill_registry().list_skills(), key=lambda s: s.name)
                    console.print(Text(f"\n  Available skills ({len(skills)}):", style="#BD93F9 bold"))
                    for s in skills:
                        console.print(Text(f"    {s.name:<30} {s.description[:50]}", style="#F8F8F2"))
                    console.print()
                except Exception:
                    pass
                continue

        # Send message to NIA and stream the response
        try:
            renderer = StreamingRenderer()
            async for event in nia.chat(user_input):
                if isinstance(event, AssistantTextDelta):
                    renderer.add_text(event.text)
                elif isinstance(event, ToolExecutionStarted):
                    renderer._flush()
                    sys.stdout.write("\n")
                    render_tool_start(event.tool_name)
                elif isinstance(event, ToolExecutionCompleted):
                    render_tool_complete(event.tool_name, event.is_error)
                elif isinstance(event, AssistantTurnComplete):
                    renderer.finish()
                    sys.stdout.write("\n")
                    sys.stdout.flush()
            sys.stdout.write("\n")
            sys.stdout.flush()
        except KeyboardInterrupt:
            console.print(Text("\n  [interrupted]", style="#FF8C00"))
        except Exception as exc:
            console.print(Text(f"\n  [error: {exc}]", style="#FF5555"))


__all__ = [
    "CADUCEUS_LINES",
    "NIA_LOGO_LINES",
    "NIA_THEME",
    "StreamingRenderer",
    "render_banner",
    "render_input_prompt",
    "render_startup_screen",
    "run_interactive",
]
