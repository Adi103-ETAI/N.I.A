# NIA Tooling and Architecture Guidance

Preferred ToolManager API
- Use `core.tool_manager.ToolManager` for all new code.
- It returns raw results and raises on errors, which simplifies control flow.

Legacy Adapter
- `core.tools.ToolManager` wraps results in dicts. Use only when interacting with older code that expects `{"success", "output", "error"}`.

Dev Tools Registration
- Do not register tools at import time.
- Explicitly call `core.tools.register_dev_tools(mgr)` during startup (e.g., `interface/chat.py`).
- Optional: set `NIA_AUTO_REGISTER_DEV_TOOLS=1` to enable gated, best-effort auto-registration for local demos.

Brain Registration Policy
- `core.brain.CognitiveLoop` checks `has_tool()` and registers `EchoTool` and `HelloTool` only if missing.
- Fallback minimal implementations are used only when dev tools are unavailable.

Testing Side Effects
- Add tests that import modules without environment flags and verify no tools are registered implicitly.
- Ensure registration idempotency and consistent API behavior across tests.
