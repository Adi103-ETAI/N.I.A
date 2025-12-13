# NIA Tooling and Architecture Guidance

Preferred ToolManager API
- Use `core.tool_manager.ToolManager` for all new code.
- It returns raw results and raises on errors, which simplifies control flow.

Legacy Adapter
- Use `core.tool_manager.ToolManager` only. The legacy adapter has been removed.

Dev Tools Registration
- Do not register tools at import time.
- Explicitly call `core.tools.register_dev_tools(mgr)` during startup (e.g., `interface/chat.py`).
- Optional: set `NIA_AUTO_REGISTER_DEV_TOOLS=1` to enable gated, best-effort auto-registration for local demos.

Plugin & reload helper
- Prefer the `ToolManager.reload_plugins(directory)` helper for hot-reloading/unloading plugins rather than re-importing modules yourself.

Lazy / optional dependencies
- Heavy optional libraries (e.g., `numpy`, `faiss`) are lazily imported in `core.memory.VectorStore` to avoid import-time failures. Ensure generated code handles ImportError gracefully and provides a clear fallback.

Compatibility note
- A small legacy adapter exists in `core.tools.__init__` for compatibility with older tests and consumer expectations, but prefer `core.tool_manager.ToolManager` for new code.

Brain Registration Policy
- `core.brain.CognitiveLoop` checks `has_tool()` and registers `EchoTool` and `HelloTool` only if missing.
- Fallback minimal implementations are used only when dev tools are unavailable.

Testing Side Effects
- Add tests that import modules without environment flags and verify no tools are registered implicitly.
- Ensure registration idempotency and consistent API behavior across tests.
- Add tests that simulate missing optional dependencies (e.g., temporarily block `faiss` and `numpy`) and assert that `MemoryManager` falls back without raising at import time.
