"""N.I.A. ASCII Art Banners and CLI presentation helpers."""
from __future__ import annotations

import os
import sys

# Force UTF-8 for Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass  # Python < 3.7

BANNER = r"""
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║    ███╗   ██╗   ██╗    █████╗                                             ║
║    ████╗  ██║   ██║   ██╔══██╗     Neural Intelligence Assistant          ║
║    ██╔██╗ ██║   ██║   ███████║     ─────────────────────────────          ║
║    ██║╚██╗██║   ██║   ██╔══██║     CLASSIFICATION: DIRECTOR_LEVEL_ACCESS  ║
║    ██║ ╚████║██╗██║██╗██║  ██║     DEVELOPER: SentArc Labs                ║
║    ╚═╝  ╚═══╝╚═╝╚═╝╚═╝╚═╝  ╚═╝     VERSION: Velocity                      ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""

MINI_BANNER = """
╭──────────────────────────────────────────╮
│  N.I.A. - Neural Intelligence Assistant  │
╰──────────────────────────────────────────╯
"""


def supports_color() -> bool:
    """Return True when ANSI color styling should be enabled."""
    if os.environ.get("NO_COLOR"):
        return False
    if not sys.stdout.isatty():
        return False
    term = os.environ.get("TERM", "")
    return term not in {"", "dumb"}


def style(text: str, code: str) -> str:
    """Apply ANSI style code when supported."""
    if not supports_color():
        return text
    return f"\033[{code}m{text}\033[0m"


def render_banner() -> str:
    """Render the main banner with optional gradient-like accent color."""
    return style(BANNER, "1;96")


def render_hint() -> str:
    """Render startup command hint."""
    return (
        f"{style('Commands', '1;94')}: "
        f"{style('help', '1;92')}, "
        f"{style('status', '1;93')}, "
        f"{style('exit', '1;91')}"
    )


# Version info
VERSION = "4.0.0"
