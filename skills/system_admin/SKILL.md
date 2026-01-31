---
name: System Administration
description: Clipboard operations, time queries, volume control, and shell commands
tools: ["get_clipboard_text", "set_clipboard_text", "get_current_time", "mute_volume"]
platforms: ["windows", "linux", "darwin"]
---
To perform system administration tasks:

**Clipboard Operations:**
- `get_clipboard_text()` - Read current clipboard content
- `set_clipboard_text("text")` - Copy text to clipboard

**Time Queries:**
- `get_current_time()` - Get formatted current date/time

**Volume Control (Windows):**
- `mute_volume(True)` - Mute system audio
- `mute_volume(False)` - Unmute system audio

**Advanced Tasks:**
For tasks not covered by specific tools, consider:
1. Using file_operations to read/write config files
2. Using process_control to launch system utilities
3. Combining multiple tools for complex workflows

**Example Workflows:**
- "Copy this to clipboard" → `set_clipboard_text("content")`
- "What time is it?" → `get_current_time()`
- "Mute the computer" → `mute_volume(True)`
