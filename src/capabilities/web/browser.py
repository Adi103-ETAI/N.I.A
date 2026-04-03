"""
MODULE: Web Browser Automation (Async Playwright-based)
VERSION: 3.0.0
STRICT SCOPE: Navigation, Tabs, URL handling, Page interaction.
CONSTRAINTS: Uses Async Playwright for non-blocking browser control.

TARA 2.0 Atomic Tool Module - ASYNC VERSION.

Verification Logic (Trust But Verify):
    - browser_open_url(): Uses `wait_until="domcontentloaded"` with 30s timeout.
      Returns page title + interactive elements list as proof of success.
    - browser_click(): Uses Playwright's implicit wait with timeout.
      Returns specific error on PlaywrightTimeout ("Element not found").
    - browser_type(): Same wait pattern with clear error messages.

Error Handling:
    - PlaywrightTimeout errors caught and converted to LLM-readable strings.
    - All returns use emoji prefixes (✅/❌/⏱️) for easy parsing.

Architecture:
    AsyncBrowserManager (Singleton) maintains Playwright session across tool calls.
    Individual tool functions are async interfaces to the manager.

Exports:
    - browser_open_url(url: str) -> str
    - browser_get_content() -> str
    - browser_click(selector: str) -> str
    - browser_type(selector: str, text: str) -> str
    - browser_scroll(direction: str) -> str
    - browser_screenshot() -> str
    - browser_close() -> str
"""
from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

from src.core.logger import setup_logger
from src.core.config import get_settings

logger = setup_logger("TARA.Tools.BrowserOps")

# =============================================================================
# Optional Dependencies - ASYNC API
# =============================================================================

try:
    from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout
    _HAS_PLAYWRIGHT = True
except ImportError:
    _HAS_PLAYWRIGHT = False
    async_playwright = None  # type: ignore
    PlaywrightTimeout = TimeoutError  # type: ignore
    logger.warning("playwright not available - browser operations disabled")

# Load settings
settings = get_settings()

# Screenshot output directory
SCREENSHOT_DIR = Path("data/screenshots")
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)


async def _ensure_playwright_chromium() -> bool:
    """Ensure Playwright Chromium browser binary exists.

    If missing, auto-install via `uv run playwright install chromium` when possible.
    """
    if not _HAS_PLAYWRIGHT:
        return False

    cache_root = Path.home() / ".cache" / "ms-playwright"
    existing = list(cache_root.glob("chromium-*/chrome-linux64/chrome"))
    if existing:
        return True

    # Fall back to playwright CLI check/install.
    install_cmd = (
        ["uv", "run", "playwright", "install", "chromium"]
        if shutil.which("uv")
        else ["playwright", "install", "chromium"]
    )

    try:
        logger.warning("Playwright Chromium missing. Attempting auto-install...")
        proc = await asyncio.to_thread(
            subprocess.run,
            install_cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            logger.error(
                "Playwright auto-install failed (code=%s): %s",
                proc.returncode,
                (proc.stderr or proc.stdout or "").strip()[:800],
            )
            return False
        logger.info("Playwright Chromium installed successfully.")
    except Exception as exc:
        logger.error("Playwright auto-install error: %s", exc)
        return False

    existing_after = list(cache_root.glob("chromium-*/chrome-linux64/chrome"))
    return bool(existing_after)


def _linux_playwright_deps_hint(error_text: str) -> str:
    """Return actionable Linux dependency hint for common Playwright failures."""
    if "libatk-1.0.so.0" in error_text:
        return (
            "❌ Missing Linux shared libraries for Chromium.\n"
            "Run one of:\n"
            "  - sudo apt-get update && sudo apt-get install -y libatk1.0-0 libatk-bridge2.0-0 libcups2 libnss3 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 libasound2t64 libpangocairo-1.0-0 libgtk-3-0\n"
            "  - uv run playwright install-deps chromium\n"
        )
    return ""


# =============================================================================
# AsyncBrowserManager Singleton
# =============================================================================

class AsyncBrowserManager:
    """
    Manager for async Playwright browser session.
    
    Use get_browser_manager() or container.browser_manager to access.
    
    🌊 ASYNC VERSION: All operations are non-blocking.
    
    Maintains state across individual tool calls:
    - playwright: The Playwright instance
    - browser: Chromium browser instance
    - context: Browser context (cookies, storage)
    - current_page: Active page for operations
    """
    
    def __init__(self) -> None:
        """Initialize a new AsyncBrowserManager instance."""
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None
        self._initialized = False
        self._lock = asyncio.Lock()
        logger.debug("AsyncBrowserManager initialized")
    
    async def start(self, headless: bool = False) -> bool:
        """
        Initialize Playwright and launch browser (async).
        
        Args:
            headless: Run in headless mode (default: False for visibility).
            
        Returns:
            True if started successfully.
        """
        if not _HAS_PLAYWRIGHT:
            logger.error("Playwright not installed")
            return False
        
        async with self._lock:
            if self._initialized and self._browser:
                logger.debug("Browser already running")
                return True
            
            try:
                ready = await _ensure_playwright_chromium()
                if not ready:
                    logger.error("Chromium binary unavailable for Playwright launch")
                    return False

                logger.info("🌐 Starting async Playwright browser...")
                
                # Start Playwright (async)
                self._playwright = await async_playwright().start()
                
                # Build launch options
                launch_options = {
                    "headless": headless,
                    "args": [
                        "--disable-blink-features=AutomationControlled",
                        "--no-sandbox",
                    ],
                }
                
                # Use custom browser executable if configured
                if settings.BROWSER_EXECUTABLE_PATH:
                    if os.path.exists(settings.BROWSER_EXECUTABLE_PATH):
                        launch_options["executable_path"] = settings.BROWSER_EXECUTABLE_PATH
                        logger.info(f"Using custom browser: {settings.BROWSER_EXECUTABLE_PATH}")
                    else:
                        logger.warning(f"Custom browser not found: {settings.BROWSER_EXECUTABLE_PATH}")
                
                # Launch browser (async)
                self._browser = await self._playwright.chromium.launch(**launch_options)
                
                # Create context with reasonable viewport
                self._context = await self._browser.new_context(
                    viewport={
                        "width": settings.BROWSER_VIEWPORT_WIDTH,
                        "height": settings.BROWSER_VIEWPORT_HEIGHT,
                    },
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                )
                
                # Create initial page
                self._page = await self._context.new_page()
                
                self._initialized = True
                logger.info("✅ Async browser started successfully")
                return True
                
            except Exception as e:
                hint = _linux_playwright_deps_hint(str(e))
                if hint:
                    logger.error(hint.strip())
                logger.error(f"Failed to start browser: {e}")
                await self._cleanup()
                return False
    
    async def get_page(self):
        """
        Get current page, auto-starting browser if needed (async).
        
        Returns:
            Playwright Page object or None.
        """
        if not self._initialized or not self._page:
            if not await self.start():
                return None
        return self._page
    
    async def stop(self) -> None:
        """Close browser and cleanup resources (async)."""
        logger.info("🛑 Stopping async browser...")
        await self._cleanup()
        logger.info("Browser stopped")
    
    async def _cleanup(self) -> None:
        """Internal async cleanup helper."""
        try:
            if self._page:
                await self._page.close()
        except Exception:
            pass
        
        try:
            if self._context:
                await self._context.close()
        except Exception:
            pass
        
        try:
            if self._browser:
                await self._browser.close()
        except Exception:
            pass
        
        try:
            if self._playwright:
                await self._playwright.stop()
        except Exception:
            pass
        
        self._page = None
        self._context = None
        self._browser = None
        self._playwright = None
        self._initialized = False
    
    @property
    def is_running(self) -> bool:
        """Check if browser is running."""
        return self._initialized and self._browser is not None


# =============================================================================
# ServiceRegistry Integration
# =============================================================================

def get_browser_manager() -> AsyncBrowserManager:
    """Get or create the AsyncBrowserManager via ServiceRegistry.
    
    The AsyncBrowserManager is registered as "browser" in the ServiceRegistry.
    If not yet registered, it will be created and registered automatically.
    This ensures only one instance exists per session while enabling 
    centralized lifecycle management.
    """
    from src.core.di import ServiceRegistry
    
    manager = ServiceRegistry.get("browser")
    if manager is None:
        manager = AsyncBrowserManager()
        ServiceRegistry.register("browser", manager)
        logger.info("AsyncBrowserManager registered in ServiceRegistry")
    return manager


# =============================================================================
# Atomic Tool: browser_open_url (ASYNC)
# =============================================================================

async def browser_open_url(url: str) -> str:
    """Navigate to a URL and return page structure with interactive elements.
    
    Opens the specified URL in a headless Chromium browser and scans the page
    for interactive elements. Auto-starts browser if not running.
    
    Args:
        url: Target URL. Protocol (https://) auto-prepended if missing.
        
    Returns:
        str: Success message with page title and interactive elements list,
             or error message if navigation fails.
             
    Raises:
        None - All errors returned as string messages for LLM consumption.
        
    Example:
        >>> await browser_open_url("google.com")
        "✅ Navigated to https://google.com
         📄 Title: Google
         
         Interactive Elements:
         [INPUT] name='q' placeholder='Search'
         [BUTTON] text='Google Search'"
    """
    if not _HAS_PLAYWRIGHT:
        return "❌ Playwright not installed. Run: uv add playwright && uv run playwright install chromium"
    
    if not url:
        return "❌ URL is required"
    
    # Auto-prepend https if missing
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    
    manager = get_browser_manager()
    page = await manager.get_page()
    
    if not page:
        return (
            "❌ Failed to start browser. "
            "If running on Linux, install runtime deps: "
            "`uv run playwright install-deps chromium` or apt packages."
        )
    
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=30000)
        title = await page.title() or "No title"
        logger.info(f"Navigated to: {url}")
        
        # Scan for interactive elements
        elements_info = await _scan_page_elements(page)
        
        result = f"✅ Navigated to {url}\n"
        result += f"📄 Title: {title}\n\n"
        result += elements_info
        
        return result
        
    except PlaywrightTimeout:
        return f"⏱️ Timeout loading {url} (page may still be loading)"
    except Exception as e:
        return f"❌ Navigation failed: {e}"


async def _scan_page_elements(page, max_elements: int = 15) -> str:
    """
    Scan page for interactive elements and return their selectors (async).
    
    This helps the agent know WHERE to type or click.
    """
    elements = []
    
    try:
        # Scan inputs
        inputs = await page.locator("input:visible").all()
        for inp in inputs[:5]:
            try:
                name = await inp.get_attribute("name") or ""
                placeholder = await inp.get_attribute("placeholder") or ""
                input_type = await inp.get_attribute("type") or "text"
                aria = await inp.get_attribute("aria-label") or ""
                
                selector = f"input[name='{name}']" if name else f"input[type='{input_type}']"
                desc = placeholder or aria or name or input_type
                elements.append(f"[INPUT] selector='{selector}' ({desc})")
            except Exception as e:
                logger.debug(f"⚠️ Input scan skipped: {e}")
        
        # Scan textareas
        textareas = await page.locator("textarea:visible").all()
        for ta in textareas[:3]:
            try:
                name = await ta.get_attribute("name") or "textarea"
                placeholder = await ta.get_attribute("placeholder") or ""
                selector = f"textarea[name='{name}']" if name != "textarea" else "textarea"
                elements.append(f"[TEXTAREA] selector='{selector}' ({placeholder or name})")
            except Exception as e:
                logger.debug(f"⚠️ Textarea scan skipped: {e}")
        
        # Scan buttons
        buttons = await page.locator("button:visible, input[type='submit']:visible").all()
        for btn in buttons[:5]:
            try:
                text = (await btn.inner_text())[:30].strip() or await btn.get_attribute("value") or ""
                aria = await btn.get_attribute("aria-label") or ""
                if text:
                    selector = f"text='{text}'"
                    elements.append(f"[BUTTON] selector=\"{selector}\" ({text})")
                elif aria:
                    elements.append(f"[BUTTON] aria-label='{aria}'")
            except Exception as e:
                logger.debug(f"⚠️ Button scan skipped: {e}")
        
        # Scan links
        links = await page.locator("a:visible").all()
        for link in links[:5]:
            try:
                text = (await link.inner_text())[:25].strip()
                if text and len(text) > 2:
                    elements.append(f"[LINK] text='{text}'")
            except Exception as e:
                logger.debug(f"⚠️ Link scan skipped: {e}")
        
    except Exception as e:
        logger.error(f"Element scan partial failure: {e}")
    
    if elements:
        return "🔍 Interactive Elements:\n" + "\n".join(elements[:max_elements])
    else:
        logger.warning(f"No interactive elements found on {page.url}")
        return "⚠️ No interactive elements found. Try browser_get_content() for page text."


# =============================================================================
# Atomic Tool: browser_get_content (ASYNC)
# =============================================================================

async def browser_get_content(max_length: int = 5000) -> str:
    """
    Get text content of the current page (async).
    
    ONE ACTION: Read page text for LLM understanding.
    
    Args:
        max_length: Maximum characters to return (default: 5000).
        
    Returns:
        Page text content or error message.
    """
    if not _HAS_PLAYWRIGHT:
        return "❌ Playwright not installed"
    
    manager = get_browser_manager()
    page = await manager.get_page()
    
    if not page:
        return "❌ Browser not running. Use browser_open_url first."
    
    try:
        # Get visible text content
        text = await page.inner_text("body", timeout=5000)
        
        # Clean up whitespace
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        clean_text = "\n".join(lines)
        
        # Truncate if needed
        if len(clean_text) > max_length:
            clean_text = clean_text[:max_length] + "\n... (truncated)"
        
        url = page.url
        return f"📄 Content from {url}:\n\n{clean_text}"
        
    except Exception as e:
        return f"❌ Failed to get content: {e}"


# =============================================================================
# Atomic Tool: browser_click (ASYNC)
# =============================================================================

async def browser_click(selector: str, timeout: int = 5000) -> str:
    """
    Click an element on the page (async).
    
    ONE ACTION: Click element matching selector.
    
    Args:
        selector: CSS selector or text selector (e.g., "button.submit", "text=Login").
        timeout: Milliseconds to wait for element.
        
    Returns:
        Success or failure message.
    """
    if not _HAS_PLAYWRIGHT:
        return "❌ Playwright not installed"
    
    if not selector:
        return "❌ Selector is required"
    
    manager = get_browser_manager()
    page = await manager.get_page()
    
    if not page:
        return "❌ Browser not running"
    
    try:
        await page.click(selector, timeout=timeout)
        logger.debug(f"Clicked: {selector}")
        return f"✅ Clicked '{selector}'"
    except PlaywrightTimeout:
        return f"❌ Element not found: '{selector}' (timeout)"
    except Exception as e:
        return f"❌ Click failed: {e}"


# =============================================================================
# Atomic Tool: browser_type (ASYNC)
# =============================================================================

async def browser_type(selector: str, text: str, clear_first: bool = True) -> str:
    """
    Type text into an input element (async).
    
    ONE ACTION: Fill text into element matching selector.
    
    SELECTOR TIPS:
    - Use selectors from browser_open_url output (e.g., "input[name='q']")
    - For text-based: "text='Button Text'" or "placeholder='Search...'"
    - For Google search: "input[name='q']" or "textarea[name='q']"
    - If unsure, try "input:visible" or use keyboard_type() for blind typing
    
    Args:
        selector: CSS selector for input element.
        text: Text to type.
        clear_first: Clear existing text before typing (default: True).
        
    Returns:
        Success or failure message.
        
    Example:
        >>> await browser_type("input[name='q']", "Agentic AI")
        "✅ Typed 'Agentic AI' into 'input[name='q']'"
    """
    if not _HAS_PLAYWRIGHT:
        return "❌ Playwright not installed"
    
    if not selector:
        return "❌ Selector is required"
    
    if not text:
        return "❌ Text is required"
    
    manager = get_browser_manager()
    page = await manager.get_page()
    
    if not page:
        return "❌ Browser not running"
    
    try:
        if clear_first:
            await page.fill(selector, text, timeout=5000)
        else:
            await page.type(selector, text, timeout=5000)
        
        display_text = text[:30] + "..." if len(text) > 30 else text
        logger.debug(f"Typed into {selector}: {display_text}")
        return f"✅ Typed '{display_text}' into '{selector}'"
    except PlaywrightTimeout:
        return f"❌ Input not found: '{selector}' (timeout)"
    except Exception as e:
        return f"❌ Type failed: {e}"


# =============================================================================
# Atomic Tool: browser_scroll (ASYNC)
# =============================================================================

async def browser_scroll(direction: str = "down", amount: int = 500) -> str:
    """
    Scroll the page (async).
    
    ONE ACTION: Scroll up or down.
    
    Args:
        direction: "up" or "down" (default: "down").
        amount: Pixels to scroll (default: 500).
        
    Returns:
        Success message.
    """
    if not _HAS_PLAYWRIGHT:
        return "❌ Playwright not installed"
    
    manager = get_browser_manager()
    page = await manager.get_page()
    
    if not page:
        return "❌ Browser not running"
    
    direction = direction.lower().strip()
    
    try:
        if direction == "down":
            await page.mouse.wheel(0, amount)
        elif direction == "up":
            await page.mouse.wheel(0, -amount)
        else:
            return f"❌ Invalid direction '{direction}'. Use 'up' or 'down'"
        
        logger.debug(f"Scrolled {direction} by {amount}px")
        return f"📜 Scrolled {direction}"
    except Exception as e:
        return f"❌ Scroll failed: {e}"


# =============================================================================
# Atomic Tool: browser_screenshot (ASYNC)
# =============================================================================

async def browser_screenshot(filename: str = None) -> str:
    """
    Take a screenshot of the current page (async).
    
    ONE ACTION: Capture page screenshot.
    
    Args:
        filename: Optional filename (auto-generated if not provided).
        
    Returns:
        Path to saved screenshot.
    """
    if not _HAS_PLAYWRIGHT:
        return "❌ Playwright not installed"
    
    manager = get_browser_manager()
    page = await manager.get_page()
    
    if not page:
        return "❌ Browser not running"
    
    try:
        # Generate filename
        if not filename:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.png"
        
        # Ensure .png extension
        if not filename.endswith(".png"):
            filename += ".png"
        
        # Full path
        filepath = SCREENSHOT_DIR / filename
        
        # Take screenshot (async)
        await page.screenshot(path=str(filepath), full_page=False)
        
        logger.info(f"Screenshot saved: {filepath}")
        return f"📸 Screenshot saved: {filepath}"
    except Exception as e:
        return f"❌ Screenshot failed: {e}"


# =============================================================================
# Atomic Tool: browser_close (ASYNC)
# =============================================================================

async def browser_close() -> str:
    """
    Close the browser (async).
    
    ONE ACTION: Stop the Playwright session.
    
    Returns:
        Confirmation message.
    """
    manager = get_browser_manager()
    
    if not manager.is_running:
        return "⚠️ Browser is not running"
    
    await manager.stop()
    return "🛑 Browser closed"


# =============================================================================
# Atomic Tool: browser_new_tab (ASYNC)
# =============================================================================

async def browser_new_tab(url: str = None) -> str:
    """
    Open a new browser tab (async).
    
    ONE ACTION: Create new tab, optionally navigate to URL.
    
    Args:
        url: Optional URL to open in new tab.
        
    Returns:
        Success message.
    """
    if not _HAS_PLAYWRIGHT:
        return "❌ Playwright not installed"
    
    manager = get_browser_manager()
    
    if not manager.is_running:
        return "❌ Browser not running"
    
    try:
        # Create new page in context (async)
        new_page = await manager._context.new_page()
        
        # Switch to new page
        manager._page = new_page
        
        if url:
            if not url.startswith(("http://", "https://")):
                url = "https://" + url
            await new_page.goto(url, wait_until="domcontentloaded", timeout=30000)
            return f"➕ New tab opened: {url}"
        
        return "➕ New tab opened"
    except Exception as e:
        return f"❌ Failed to open new tab: {e}"


__all__ = [
    "AsyncBrowserManager",
    "get_browser_manager",
    "browser_open_url",
    "browser_get_content",
    "browser_click",
    "browser_type",
    "browser_scroll",
    "browser_screenshot",
    "browser_close",
    "browser_new_tab",
]
