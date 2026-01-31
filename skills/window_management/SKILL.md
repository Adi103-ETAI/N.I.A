---
name: Window Management
description: Control window states: focus, minimize, maximize, snap, and close
tools: ["focus_window", "minimize_window", "maximize_window", "snap_window", "close_window", "list_open_windows", "show_desktop"]
platforms: ["windows"]
---
To manage application windows:

> [!IMPORTANT]
> Always call `list_open_windows()` first to get exact window aliases.

**Finding Windows:**
1. Use `list_open_windows()` to see all tracked windows
2. Windows are identified by aliases like "notepad_1", "chrome_2"
3. The alias is assigned when the app is launched

**Window Operations:**
- `focus_window(alias)` - Bring window to front (restores if minimized)
- `minimize_window(alias)` - Minimize to taskbar
- `maximize_window(alias)` - Fill the screen
- `snap_window(alias, "left"|"right")` - Snap to screen half
- `close_window(alias)` - Close gracefully
- `show_desktop()` - Minimize all windows (Win+D)

**Workflow Example:**
1. `list_open_windows()` → Find "notepad_1"
2. `focus_window("notepad_1")` → Bring to front
3. `snap_window("notepad_1", "left")` → Snap to left

> [!NOTE]
> This skill is Windows-only. On Linux/macOS, use input_control as fallback.
