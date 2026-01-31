---
name: UI Automation
description: Semantic interaction with UI elements via the Accessibility Tree (Windows UIAutomation)
tools: ["dump_ui_tree", "click_element", "type_in_element", "read_element_text", "get_element_by_type"]
platforms: ["windows"]
---
To interact with application UI elements semantically:

> [!IMPORTANT]
> This is the **preferred** method for Windows apps. Use element names, not coordinates.

**Discovery Phase (ALWAYS START HERE):**
1. Use `dump_ui_tree(window_alias)` to scan the UI
2. This returns a map of elements with their Names and Types
3. Look for buttons, text fields, labels by their displayed text

**Interaction Phase:**
- `click_element(alias, "Button Name")` - Click a button/link
- `type_in_element(alias, "Field Name", "text")` - Type into input field
- `read_element_text(alias, "Element Name")` - Read text content
- `get_element_by_type(alias, "Button")` - List all buttons

**Workflow Example:**
1. `dump_ui_tree("notepad_1")` → See "File", "Edit", "Format" menus
2. `click_element("notepad_1", "File")` → Open File menu
3. `click_element("notepad_1", "Save")` → Click Save option

> [!CAUTION]
> Windows-only. On Linux/macOS, fall back to input_control (coordinate-based).
