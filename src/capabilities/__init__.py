# src/capabilities/__init__.py
"""Unified Capability Layer — Tools available to TARA and other N.I.A. agents.

All tools are organized by domain.  TARA loads them via the executor node;
other agents may import individual subpackages directly.

Subpackages:

    desktop/
        Window management, application launching, UI interaction (clicks,
        typing, screenshots), and UIA element inspection.

    system/
        File I/O, process management, and system statistics.

    web/
        Browser automation (Playwright-backed ``BrowserManager``), web
        scraping, and HTTP utilities.

    vision/
        Screenshot capture helpers shared between IRIS and TARA.

    execution/
        Tool lifecycle management (tool loading, hot-reload) and
        decorator-based tool registration.

See Also:
    ``src.capabilities.decorators``  — ``@security_level`` decorator.
    ``src.agents.tara.protocols``    — ``@tara_tool`` decorator.
"""

__all__ = [
    "desktop",
    "system",
    "web",
    "vision",
    "memory",
    "ai",
]
