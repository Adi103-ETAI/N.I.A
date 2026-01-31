---
name: Input Control
description: Raw mouse and keyboard simulation using screen coordinates
tools: ["mouse_click", "mouse_drag", "mouse_scroll", "keyboard_type", "keyboard_hotkey", "keyboard_press"]
platforms: ["windows", "linux", "darwin"]
---
To simulate raw mouse and keyboard input:

> [!WARNING]
> This is the **fallback layer**. Use UI Automation first when available.
> Requires screen coordinates. Use Vision (capture_screen) to find positions.

**Mouse Operations:**
- `mouse_click(x, y)` - Left click at coordinates
- `mouse_click(x, y, button="right")` - Right click
- `mouse_click(x, y, double=True)` - Double click
- `mouse_drag(x1, y1, x2, y2)` - Drag from point A to B
- `mouse_scroll(clicks)` - Scroll wheel (positive=up, negative=down)

**Keyboard Operations:**
- `keyboard_type("Hello")` - Type text character by character
- `keyboard_press("enter")` - Press single key
- `keyboard_hotkey("ctrl", "s")` - Press key combination (e.g., Ctrl+S)
- `keyboard_hotkey("alt", "f4")` - Close window
- `keyboard_hotkey("win", "d")` - Show desktop

**When to Use:**
- Games and canvas applications (no accessibility tree)
- When UI Automation fails to find elements
- For keyboard shortcuts and hotkeys
- Cross-platform fallback on Linux/macOS

**Workflow Example:**
1. `capture_screen()` → Get screenshot
2. Analyze image to find button coordinates
3. `mouse_click(500, 300)` → Click at those coordinates
