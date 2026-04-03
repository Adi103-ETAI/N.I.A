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
███╗   ██╗██╗ █████╗ 
████╗  ██║██║██╔══██╗
██╔██╗ ██║██║███████║    Neural Intelligence Assistant
██║╚██╗██║██║██╔══██║    ────────────────────────────────
██║ ╚████║██║██║  ██║    Version: 4.0.0 (Velocity)
╚═╝  ╚═══╝╚═╝╚═╝  ╚═╝    SentArc Labs
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
    if term in {"", "dumb"}:
        return False
    return True


def detect_theme() -> str:
    """Detect terminal theme (dark/light) from environment hints."""
    theme_hint = os.environ.get("COLORFGBG", "")
    if theme_hint:
        try:
            bg = theme_hint.split(";")[-1]
            if bg in {"0", "8", "black"}:
                return "dark"
            elif bg in {"7", "15", "white"}:
                return "light"
        except Exception:
            pass
    return "dark"


def style(text: str, code: str) -> str:
    """Apply ANSI style code when supported."""
    if not supports_color():
        return text
    return f"\033[{code}m{text}\033[0m"


def render_banner() -> str:
    """Render the main banner with theme-aware styling."""
    theme = detect_theme()
    color = "1;96" if theme == "dark" else "1;94"
    return style(BANNER, color)


def render_hint() -> str:
    """Render startup command hint."""
    return (
        f"\n{style('Ready to assist.', '2;37')} "
        f"Type {style('help', '1;92')} for commands or just ask a question.\n"
    )


def render_separator(char: str = "─", width: int = 80) -> str:
    """Render a horizontal separator line."""
    return style(char * width, "2;90")


# Version info
VERSION = "4.0.0"
