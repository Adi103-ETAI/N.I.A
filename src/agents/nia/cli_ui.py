"""N.I.A CLI UI — medical-themed terminal interface with caduceus logo.

Features:
  - Golden caduceus ASCII art (medical symbol — NIA wanted to be a doctor)
  - "NIA" name in large figlet-style text
  - Startup screen showing model, session ID, provider, tools, skills
  - Clean input prompt with orange accent line
  - Flicker-free streaming output (buffered rendering)
  - Color scheme: gold + cyan + orange on black

Usage:
    from agents.nia.cli_ui import run_interactive
    await run_interactive(nia)
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.rule import Rule
from rich.align import Align
from rich.live import Live
from rich.theme import Theme

# ---------------------------------------------------------------------------
# Color theme — gold + cyan + orange on black (medical + tech aesthetic)
# ---------------------------------------------------------------------------

NIA_THEME = Theme({
    # Primary brand colors
    "nia.gold": "#FFD700",        # Gold — caduceus, NIA name, headers
    "nia.gold.dim": "#B8860B",     # Dark gold — secondary accents
    "nia.cyan": "#00CED1",         # Cyan — tech accent, paths, URLs
    "nia.orange": "#FF8C00",       # Orange — input line, active items
    "nia.green": "#50FA7B",        # Green — success, tool completion
    "nia.red": "#FF5555",          # Red — errors, tool failure
    "nia.gray": "#6272A4",         # Gray — secondary info, hints
    "nia.white": "#F8F8F2",        # Off-white — body text
    "nia.purple": "#BD93F9",       # Purple — skills, special items
    # Semantic
    "nia.model": "#00CED1",        # Model name in cyan
    "nia.session": "#6272A4",      # Session ID in gray
    "nia.path": "#6272A4",         # Working directory in gray
    "nia.tools": "#FF8C00",        # Tools count in orange
    "nia.skills": "#BD93F9",       # Skills count in purple
})

console = Console(theme=NIA_THEME, force_terminal=True)

# ---------------------------------------------------------------------------
# ASCII Art — Caduceus (medical symbol) + NIA name
# ---------------------------------------------------------------------------

# The caduceus: two snakes winding around a winged staff.
# Designed to look good in a terminal at 80+ columns.
CADUCEUS = r"""
                    .:.
                   .' '.
              .-.-.`     `.-.-.
            .'     \       /     `.
                   _\     /_
              .-"".__\   /__.""-.
             /    .-"  |  "-.    \
            |   .'  .-"|`"-.  `.   |
            |  |   |   |   |   |  |
            |  |   |.-.|.-.|   |  |
            |  |   |   |   |   |  |
            |  |   |.-.|.-.|   |  |
            |  |   |   |   |   |  |
            |  |   |.-.|.-.|   |  |
            |  |   |   |   |   |  |
             \  `. |.-.|.-.| .'  /
              `-._|____|____|_.-'
                  |    |    |
                  |    |    |
                 /|    |    |\
                / |    |    | \
                  |____|____|
                  \    |    /
                   \   |   /
                    \  |  /
                     \ | /
                      \|/
                      ` '
"""

# NIA name in large figlet-style block letters
NIA_BANNER = r"""
  ███╗   ██╗ █████╗ ███████╗
  ████╗  ██║██╔══██╗██╔════╝
  ██╔██╗ ██║███████║███████╗
  ██║╚██╗██║██╔══██║╚════██║
  ██║ ╚████║██║  ██║███████║
  ╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝
"""

# Compact NIA for smaller terminals
NIA_COMPACT = "N.I.A"


def render_banner() -> Text:
    """Render the startup banner: caduceus + NIA name + tagline."""
    parts = []

    # Caduceus in gold
    caduceus = Text(CADUCEUS, style="nia.gold")
    parts.append(Align.center(caduceus))

    # NIA name in gold
    nia_name = Text(NIA_BANNER, style="nia.gold bold")
    parts.append(Align.center(nia_name))

    # Tagline
    tagline = Text("Neural Intelligence Assistant", style="nia.cyan")
    parts.append(Align.center(tagline))

    subtitle = Text("Your AI partner, inspired by J.A.R.V.I.S", style="nia.gray")
    parts.append(Align.center(subtitle))

    return Group(*parts)


def render_session_info(
    model: str,
    session_id: str,
    cwd: str,
    provider: str = "",
    built_by: str = "Adi103-ETAI",
) -> Panel:
    """Render the session info panel (model, session ID, path, built by)."""
    info = Table(show_header=False, show_edge=False, padding=(0, 1), expand=True)
    info.add_column(style="nia.gray", no_wrap=True)
    info.add_column(style="nia.white")

    info.add_row("Model:", Text(model or "unknown", style="nia.model"))
    if provider:
        info.add_row("Provider:", Text(provider, style="nia.cyan"))
    info.add_row("Session:", Text(session_id, style="nia.session"))
    info.add_row("Path:", Text(cwd, style="nia.path"))
    info.add_row("Built by:", Text(built_by, style="nia.orange bold"))

    return Panel(
        info,
        border_style="#B8860B",
        title="[bold #FFD700] Session [/]",
        title_align="left",
        padding=(1, 2),
    )


def render_tools_and_skills(tool_count: int, skill_count: int) -> Panel:
    """Render the available tools and skills summary."""
    # Tools table
    tools_table = Table(show_header=False, show_edge=False, padding=(0, 1), expand=True)
    tools_table.add_column(style="nia.orange", no_wrap=True)
    tools_table.add_column(style="nia.gray")

    tools_table.add_row(
        Text(f"  {tool_count}", style="nia.orange bold"),
        Text("tools available", style="nia.gray"),
    )
    tools_table.add_row(
        Text(f"  {skill_count}", style="nia.purple bold"),
        Text("skills loaded", style="nia.gray"),
    )

    hints = Text(
        "  /help for commands  ·  /model to switch  ·  /tools to list  ·  /skills to browse",
        style="nia.gray",
    )

    content = Group(tools_table, Text(""), hints)

    return Panel(
        content,
        border_style="#B8860B",
        title="[bold #FFD700] Ready [/]",
        title_align="left",
        padding=(1, 2),
    )


def render_startup_screen(
    model: str,
    session_id: str,
    cwd: str,
    provider: str = "",
    tool_count: int = 0,
    skill_count: int = 0,
) -> None:
    """Render the full startup screen."""
    console.clear()

    # Banner
    console.print(render_banner())
    console.print()

    # Divider
    console.print(Rule(style="nia.gold.dim"))
    console.print()

    # Session info
    console.print(render_session_info(model, session_id, cwd, provider))
    console.print()

    # Tools and skills
    console.print(render_tools_and_skills(tool_count, skill_count))
    console.print()

    # Input hint
    console.print(Text("  Type your message below. Press Enter to send.", style="nia.gray"))
    console.print()


def render_input_prompt() -> None:
    """Render the input prompt area with orange accent line."""
    console.print(Rule(style="nia.orange"))
    console.print(Text("> ", style="nia.orange bold"), end="")


def render_tool_start(tool_name: str) -> None:
    """Render a tool execution start indicator."""
    console.print(Text(f"  ⚡ {tool_name}", style="nia.orange"), end="")


def render_tool_complete(tool_name: str, is_error: bool = False) -> None:
    """Render a tool execution completion indicator."""
    if is_error:
        console.print(Text(f" ✗", style="nia.red"))
    else:
        console.print(Text(f" ✓", style="nia.green"))


def render_assistant_text(text: str) -> None:
    """Render assistant text (non-streaming, for complete responses)."""
    console.print(Text(text, style="nia.white"))


class StreamingRenderer:
    """Flicker-free streaming text renderer.

    Buffers text and writes in chunks to minimize terminal redraws.
    Uses sys.stdout.write + flush instead of rich.Live to avoid the
    flickering that comes from rich.Live's full-screen refresh.
    """

    def __init__(self) -> None:
        self._buffer: list[str] = []
        self._last_flush = time.monotonic()
        self._flush_interval = 0.05  # 50ms — smooth but not too frequent
        self._total_text = ""

    def add_text(self, text: str) -> None:
        """Add text to the buffer and flush if needed."""
        self._buffer.append(text)
        self._total_text += text
        now = time.monotonic()
        if now - self._last_flush >= self._flush_interval:
            self._flush()

    def _flush(self) -> None:
        """Flush buffered text to stdout."""
        if not self._buffer:
            return
        text = "".join(self._buffer)
        self._buffer.clear()
        # Use plain stdout write — no rich formatting to avoid flicker.
        sys.stdout.write(text)
        sys.stdout.flush()
        self._last_flush = time.monotonic()

    def finish(self) -> None:
        """Flush any remaining buffered text and add a newline."""
        self._flush()
        sys.stdout.write("\n")
        sys.stdout.flush()


def render_thinking_indicator() -> None:
    """Render a subtle thinking indicator (non-flickering).

    Uses a simple dot animation that doesn't clear the screen.
    """
    sys.stdout.write(Text("  thinking", style="nia.gray").__str__())
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Interactive REPL
# ---------------------------------------------------------------------------


async def run_interactive(nia) -> None:
    """Run the interactive REPL with the new NIA UI.

    Args:
        nia: An initialized NIA instance.
    """
    from niaharness.engine.stream_events import (
        AssistantTextDelta,
        AssistantTurnComplete,
        ToolExecutionStarted,
        ToolExecutionCompleted,
    )

    # Gather session info for the startup screen.
    model = nia.engine._model if nia.engine else "unknown"
    session_id = nia.engine._session_id if nia.engine else "unknown"
    cwd = nia._working_directory
    provider = ""

    # Try to get provider name.
    try:
        from niaharness.api.provider import detect_provider
        from niaharness.config.settings import load_settings
        provider_info = detect_provider(load_settings())
        provider = provider_info.name
    except Exception:
        pass

    # Count tools and skills.
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

    # Render the startup screen.
    render_startup_screen(
        model=model,
        session_id=session_id,
        cwd=cwd,
        provider=provider,
        tool_count=tool_count,
        skill_count=skill_count,
    )

    # Interactive loop.
    while True:
        try:
            # Input prompt with orange accent line.
            render_input_prompt()
            user_input = input().strip()
        except (KeyboardInterrupt, EOFError):
            console.print(Text("\n  Goodbye.", style="nia.gold"))
            break

        if not user_input:
            continue

        # Handle slash commands.
        if user_input.startswith("/"):
            if user_input.lower() in ("/exit", "/quit", "exit", "quit"):
                console.print(Text("  Goodbye.", style="nia.gold"))
                break
            # Other slash commands — pass to the command registry if available.
            # For now, just show a hint.
            if user_input.lower() in ("/help", "/h"):
                console.print(Text("\n  Commands:", style="nia.gold bold"))
                console.print(Text("    /help     — Show this help", style="nia.gray"))
                console.print(Text("    /tools    — List available tools", style="nia.gray"))
                console.print(Text("    /skills   — List available skills", style="nia.gray"))
                console.print(Text("    /model    — Show or switch model", style="nia.gray"))
                console.print(Text("    /insights — Show usage analytics", style="nia.gray"))
                console.print(Text("    /profile  — Manage profiles", style="nia.gray"))
                console.print(Text("    /oauth    — Anthropic OAuth login", style="nia.gray"))
                console.print(Text("    /soul     — Edit NIA's identity", style="nia.gray"))
                console.print(Text("    /exit     — Quit NIA\n", style="nia.gray"))
                continue
            if user_input.lower() in ("/tools", "/t"):
                try:
                    tools = sorted(nia.engine._tool_registry._tools.keys())
                    console.print(Text(f"\n  Available tools ({len(tools)}):", style="nia.orange bold"))
                    for i in range(0, len(tools), 4):
                        row = tools[i:i+4]
                        console.print(Text("    " + "  ".join(f"{t:<20}" for t in row), style="nia.white"))
                    console.print()
                except Exception:
                    pass
                continue
            if user_input.lower() in ("/skills", "/sk"):
                try:
                    from niaharness.tools.skills_loader import load_skill_registry
                    skills = sorted(load_skill_registry().list_skills(), key=lambda s: s.name)
                    console.print(Text(f"\n  Available skills ({len(skills)}):", style="nia.purple bold"))
                    for s in skills:
                        console.print(Text(f"    {s.name:<30} {s.description[:50]}", style="nia.white"))
                    console.print()
                except Exception:
                    pass
                continue

        # Send message to NIA and stream the response.
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
            console.print(Text("\n  [interrupted]", style="nia.orange"))
        except Exception as exc:
            console.print(Text(f"\n  [error: {exc}]", style="nia.red"))


__all__ = [
    "CADUCEUS",
    "NIA_BANNER",
    "NIA_THEME",
    "StreamingRenderer",
    "render_banner",
    "render_input_prompt",
    "render_startup_screen",
    "render_tools_and_skills",
    "render_session_info",
    "run_interactive",
]
