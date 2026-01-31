---
name: Web Browsing
description: Automated web browsing with Playwright: navigate, click, type, and extract content
tools: ["browser_open_url", "browser_click", "browser_type", "browser_get_content", "browser_screenshot", "browser_close"]
platforms: ["windows", "linux", "darwin"]
---
To automate web browsing:

**Starting a Session:**
1. Use `browser_open_url("https://example.com")` to navigate
2. This returns page title and interactive elements
3. The browser auto-starts if not already running

**Interacting with Pages:**
- `browser_click("selector")` - Click element by CSS selector or text
  - CSS: `browser_click("button.submit")`
  - Text: `browser_click("text=Login")`
- `browser_type("selector", "text")` - Type into input field
  - `browser_type("input[name='q']", "search query")`

**Reading Content:**
- `browser_get_content()` - Extract page text for analysis
- `browser_screenshot()` - Capture page as image

**Ending Session:**
- `browser_close()` - Close browser and cleanup

**Workflow Example:**
1. `browser_open_url("https://google.com")` → Opens Google
2. `browser_type("input[name='q']", "weather today")` → Type search
3. `browser_click("text=Google Search")` → Submit search
4. `browser_get_content()` → Read results

> [!TIP]
> Use the interactive elements list from `browser_open_url` to find selectors.
