# 🛠️ TARA Tool & Plugin Standards (v3.0 Live)

## 🎯 Scope
- Applies to: `tara/tools/` AND `tara/plugin_system/`
- **CRITICAL PATH:** The plugin loader is at `tara/plugin_system/loader.py`.

## 1. The Async Bridge
- TARA runs in a `ThreadPoolExecutor`.
- **Rule:** If adding a new tool, ensure it is thread-safe.
- **Plugins:** New plugins go in `plugins/` (root). They are auto-loaded by `tara.plugin_system.loader`.

## 2. Architecture & Safety
- **No Circular Imports:** Do not import `nia` logic into `tara`.
- **Large Files:** `loader.py` is >27KB. DO NOT read the whole file unless necessary. Use `grep` or read specific line ranges.

## 3. Error Handling
- **No Crashes:** Tools must catch exceptions and return a string: `return f"Error: {e}"`.
- **Hot-Reload:** If modifying `watcher.py`, ensure you do not create zombie file observers.
