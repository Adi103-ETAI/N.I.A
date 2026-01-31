---
name: Process Control
description: Launch, monitor, and terminate system processes and applications
tools: ["launch_app", "kill_app", "list_processes"]
platforms: ["windows", "linux", "darwin"]
---
To manage processes and applications:

**Launching Applications:**
1. Use `launch_app(app_name)` with common names like "notepad", "chrome", "spotify"
2. The function returns an alias (e.g., "notepad_1") for window management
3. Wait for the app to fully load before interacting

**Listing Processes:**
1. Use `list_processes()` to see all running processes
2. Use `list_processes(filter_name="chrome")` to filter by name
3. Note the PID for process management

**Terminating Processes:**
1. Use `kill_app(app_name)` with the app name or alias
2. This gracefully closes the application
3. For stubborn apps, the system will force-terminate

**Examples:**
- "Open Notepad" → `launch_app("notepad")`
- "Close Chrome" → `kill_app("chrome")`
- "What's running?" → `list_processes()`
