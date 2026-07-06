"""Interactive browser tool — Playwright-based with persistent sessions.

Provides the "always-ready" web interaction layer that the audit (P1) flagged
as NIA's single biggest capability gap vs modern AI coding assistants.

Operations
----------
- ``navigate``   — open a URL in a headless browser
- ``click``      — click an element by CSS selector or text
- ``type``       — fill an input/textarea by selector
- ``snapshot``   — capture the current DOM as visible text + element list
- ``screenshot`` — save a PNG screenshot to ``download/``
- ``back``/``forward``/``reload`` — history controls
- ``eval_js``    — evaluate a JS expression in the page (read-only by default)
- ``close``      — close the current page (session is preserved)

Sessions
--------
Each tool invocation reuses a singleton browser+context, so cookies, localStorage,
and the current page persist across calls within the same process.  Pass
``new_session=True`` to ``navigate`` to start fresh.

Safety
------
- ``eval_js`` is gated by the permission system (writes flag) since arbitrary
  JS can be destructive.
- All URLs are validated; ``file://`` and ``data://`` schemes are rejected by
  default to limit SSRF surface (use the dedicated ``web_fetch`` tool for
  static HTTP).
"""

from __future__ import annotations

import base64
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from niaharness.tools.base import BaseTool, ToolExecutionContext, ToolResult


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class BrowserToolInput(BaseModel):
    """Arguments for the browser tool."""

    operation: Literal[
        "navigate",
        "click",
        "type",
        "snapshot",
        "screenshot",
        "back",
        "forward",
        "reload",
        "eval_js",
        "close",
    ] = Field(description="The browser operation to perform")
    url: str | None = Field(default=None, description="URL for the navigate operation")
    selector: str | None = Field(
        default=None,
        description="CSS selector for click / type / eval_js targets (when text is not given)",
    )
    text: str | None = Field(
        default=None,
        description=(
            "Visible text to match for click operations (alternative to selector). "
            "Useful when the agent doesn't know the exact selector."
        ),
    )
    value: str | None = Field(default=None, description="Text to type into the selected element")
    new_session: bool = Field(
        default=False,
        description="For navigate: discard the current session and start a fresh browser context",
    )
    full_page: bool = Field(
        default=False,
        description="For screenshot: capture the entire scrollable page, not just the viewport",
    )
    js_expression: str | None = Field(
        default=None,
        description="For eval_js: a JavaScript expression to evaluate in the page context",
    )
    timeout_ms: int = Field(
        default=30_000,
        ge=1000,
        le=120_000,
        description="Per-operation timeout in milliseconds",
    )


# ---------------------------------------------------------------------------
# Session manager — singleton browser per process
# ---------------------------------------------------------------------------


class _BrowserSession:
    """Holds a single Playwright browser + context + page across calls."""

    def __init__(self) -> None:
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None
        self._import_error: str | None = None

    async def ensure(self):
        """Lazily start Playwright and open a context + page."""
        if self._import_error is not None:
            raise RuntimeError(self._import_error)
        if self._page is not None:
            return self._page
        try:
            # Import lazily so the tool loads even when Playwright isn't installed.
            from playwright.async_api import async_playwright
        except ImportError as exc:
            self._import_error = (
                f"playwright is not installed: {exc}. "
                "Install with: pip install playwright && playwright install chromium"
            )
            raise RuntimeError(self._import_error)

        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=True)
        self._context = await self._browser.new_context(
            viewport={"width": 1280, "height": 800},
            user_agent=(
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
        )
        self._page = await self._context.new_page()
        return self._page

    async def reset(self) -> None:
        """Discard the current context (cookies, localStorage, page)."""
        if self._context is not None:
            try:
                await self._context.close()
            except Exception:
                pass
        self._context = None
        self._page = None
        if self._browser is not None:
            self._context = await self._browser.new_context(
                viewport={"width": 1280, "height": 800},
            )
            self._page = await self._context.new_page()

    async def close(self) -> None:
        if self._context is not None:
            try:
                await self._context.close()
            except Exception:
                pass
        if self._browser is not None:
            try:
                await self._browser.close()
            except Exception:
                pass
        if self._playwright is not None:
            try:
                await self._playwright.stop()
            except Exception:
                pass
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None
        # Don't keep the import error — let the user retry after install.


_SESSION = _BrowserSession()


# ---------------------------------------------------------------------------
# URL safety
# ---------------------------------------------------------------------------

_BLOCKED_SCHEMES = ("file://", "data:", "javascript:", "vbscript:")
_PRIVATE_HOSTS_RE = re.compile(
    r"^(localhost|127\.|10\.|192\.168\.|172\.(1[6-9]|2[0-9]|3[01])\.|169\.254\.|::1|fe80:|fc00:)",
    re.IGNORECASE,
)


def _validate_url(url: str) -> tuple[bool, str]:
    """Return ``(ok, reason)``.  ``ok=False`` if URL is unsafe."""
    if not url:
        return False, "empty url"
    lowered = url.strip().lower()
    if any(lowered.startswith(s) for s in _BLOCKED_SCHEMES):
        return False, f"blocked scheme: {lowered.split(':')[0]}"
    if not lowered.startswith(("http://", "https://")):
        return False, "only http/https URLs are allowed"
    # Lightweight private-host check (defense in depth; Playwright itself
    # would happily hit these).
    for private in ("localhost", "127.0.0.1", "0.0.0.0", "::1"):
        if private in lowered:
            return False, f"private host blocked: {private}"
    return True, ""


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


class BrowserTool(BaseTool):
    """Interactive browser tool backed by Playwright."""

    name = "browser"
    description = (
        "Drive a headless browser: navigate pages, click elements, type into "
        "forms, take DOM snapshots, and capture screenshots.  Sessions persist "
        "across calls so cookies and login state are preserved."
    )
    input_model = BrowserToolInput

    def is_read_only(self, arguments: BrowserToolInput) -> bool:
        # eval_js can be destructive; gate it as a write.  All other operations
        # are read-only or scoped to the ephemeral browser session.
        return arguments.operation != "eval_js"

    async def execute(self, arguments: BrowserToolInput, context: ToolExecutionContext) -> ToolResult:
        try:
            page = await _SESSION.ensure()
        except RuntimeError as exc:
            return ToolResult(output=str(exc), is_error=True)

        # Handle new_session before dispatching.
        if arguments.operation == "navigate" and arguments.new_session:
            await _SESSION.reset()
            page = await _SESSION.ensure()

        op = arguments.operation
        try:
            if op == "navigate":
                return await self._navigate(page, arguments)
            if op == "snapshot":
                return await self._snapshot(page, arguments)
            if op == "screenshot":
                return await self._screenshot(page, arguments, context)
            if op == "click":
                return await self._click(page, arguments)
            if op == "type":
                return await self._type(page, arguments)
            if op == "back":
                await page.go_back(timeout=arguments.timeout_ms)
                return await self._snapshot(page, arguments)
            if op == "forward":
                await page.go_forward(timeout=arguments.timeout_ms)
                return await self._snapshot(page, arguments)
            if op == "reload":
                await page.reload(timeout=arguments.timeout_ms)
                return await self._snapshot(page, arguments)
            if op == "eval_js":
                return await self._eval_js(page, arguments)
            if op == "close":
                await _SESSION.close()
                return ToolResult(output="Browser session closed.")
            return ToolResult(output=f"Unknown operation: {op}", is_error=True)
        except Exception as exc:
            return ToolResult(output=f"Browser error: {exc}", is_error=True)

    # ---- operation handlers --------------------------------------------

    async def _navigate(self, page, args: BrowserToolInput) -> ToolResult:
        if not args.url:
            return ToolResult(output="navigate requires url", is_error=True)
        ok, reason = _validate_url(args.url)
        if not ok:
            return ToolResult(output=f"URL rejected: {reason}", is_error=True)
        response = await page.goto(args.url, timeout=args.timeout_ms, wait_until="domcontentloaded")
        status = response.status if response is not None else None
        title = await page.title()
        url_now = page.url
        return ToolResult(
            output=(
                f"Navigated to {url_now}\n"
                f"HTTP status: {status}\n"
                f"Title: {title}"
            ),
            metadata={"url": url_now, "status": status, "title": title},
        )

    async def _snapshot(self, page, args: BrowserToolInput) -> ToolResult:
        """Capture the page as a structured text snapshot.

        Returns:
        - URL + title
        - Visible text content (truncated to ~4000 chars)
        - List of clickable elements (a, button, input) with selectors + text
        """
        title = await page.title()
        url = page.url
        # Extract visible text via document.body.innerText.
        try:
            body_text: str = await page.evaluate("() => document.body ? document.body.innerText : ''")
        except Exception as exc:
            body_text = f"(could not extract body text: {exc})"
        if len(body_text) > 4000:
            body_text = body_text[:4000] + "\n... [truncated]"

        # Collect interactive elements.
        try:
            elements: list[dict[str, Any]] = await page.evaluate(
                """() => {
                    const out = [];
                    const nodes = document.querySelectorAll('a, button, input, textarea, select');
                    for (const n of nodes) {
                        if (out.length >= 50) break;
                        const text = (n.innerText || n.value || n.placeholder || '').trim().slice(0, 60);
                        const href = n.href || '';
                        const type = n.type || n.tagName.toLowerCase();
                        out.push({tag: n.tagName.toLowerCase(), id: n.id, name: n.name, text, href, type});
                    }
                    return out;
                }"""
            )
        except Exception:
            elements = []

        lines = [f"URL: {url}", f"Title: {title}", "", "Visible text:", body_text, ""]
        if elements:
            lines.append("Interactive elements:")
            for el in elements:
                selector_parts = []
                if el.get("id"):
                    selector_parts.append(f"#{el['id']}")
                elif el.get("name"):
                    selector_parts.append(f"[name={el['name']}]")
                tag = el.get("tag", "?")
                text = el.get("text", "")
                href = el.get("href", "")
                type_ = el.get("type", "")
                desc = text or href or type_ or "(no label)"
                lines.append(f"  <{tag}> {desc}  selector={selector_parts[0] if selector_parts else tag}")
        return ToolResult(output="\n".join(lines), metadata={"url": url, "title": title})

    async def _screenshot(self, page, args: BrowserToolInput, context: ToolExecutionContext) -> ToolResult:
        # Save to the project's download dir so the user can retrieve it.
        out_dir = Path("/home/z/my-project/download")
        out_dir.mkdir(parents=True, exist_ok=True)
        # Build a filename from the current URL host.
        try:
            host = page.url.split("/")[2] if "://" in page.url else "page"
            safe_host = re.sub(r"[^a-zA-Z0-9.-]", "_", host) or "page"
        except Exception:
            safe_host = "page"
        from datetime import datetime, timezone

        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        out_path = out_dir / f"browser-{safe_host}-{ts}.png"
        await page.screenshot(path=str(out_path), full_page=args.full_page)
        size_kb = out_path.stat().st_size // 1024
        return ToolResult(
            output=f"Screenshot saved: {out_path} ({size_kb} KB, full_page={args.full_page})",
            metadata={"path": str(out_path), "full_page": args.full_page, "size_bytes": out_path.stat().st_size},
        )

    async def _click(self, page, args: BrowserToolInput) -> ToolResult:
        if args.selector:
            await page.click(args.selector, timeout=args.timeout_ms)
            what = args.selector
        elif args.text:
            # Playwright's text selector: click on element containing exact text.
            await page.click(f"text={args.text}", timeout=args.timeout_ms)
            what = f"text={args.text!r}"
        else:
            return ToolResult(output="click requires selector or text", is_error=True)
        # Snapshot after click so the agent sees the result.
        result = await self._snapshot(page, args)
        return ToolResult(
            output=f"Clicked {what}\n\n{result.output}",
            metadata=result.metadata,
        )

    async def _type(self, page, args: BrowserToolInput) -> ToolResult:
        if not args.value:
            return ToolResult(output="type requires value", is_error=True)
        if not args.selector:
            return ToolResult(output="type requires selector", is_error=True)
        # Clear the field first, then type.
        await page.fill(args.selector, args.value, timeout=args.timeout_ms)
        return ToolResult(
            output=f"Typed {len(args.value)} chars into {args.selector}",
            metadata={"selector": args.selector, "value_length": len(args.value)},
        )

    async def _eval_js(self, page, args: BrowserToolInput) -> ToolResult:
        if not args.js_expression:
            return ToolResult(output="eval_js requires js_expression", is_error=True)
        try:
            result = await page.evaluate(args.js_expression)
        except Exception as exc:
            return ToolResult(output=f"JS evaluation failed: {exc}", is_error=True)
        # Stringify result for display.
        if result is None:
            text = "null"
        elif isinstance(result, (dict, list)):
            import json
            text = json.dumps(result, indent=2, default=str)[:4000]
        else:
            text = str(result)[:4000]
        return ToolResult(output=f"JS result:\n{text}", metadata={"result": result})
