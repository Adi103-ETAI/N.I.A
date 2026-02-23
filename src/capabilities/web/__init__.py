# src/capabilities/web/__init__.py
"""Web Automation Capabilities.

Playwright-backed browser control and web scraping tools.

Modules:
    browser.py — ``BrowserManager`` (session control, navigation, scraping, form filling)

The ``BrowserManager`` wraps a Playwright Chromium instance.  It supports:
    - Persistent browser sessions across tool calls
    - Page navigation, element interaction, and screenshot capture
    - JavaScript evaluation and network request inspection
"""

from .browser import *
