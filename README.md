# N.I.A — Neural Intelligent Assistant

This repository contains the N.I.A modular assistant core components (brain, tools, memory, model manager).

This README provides quick setup instructions, notes for FAISS on Windows, and how to run tests and static checks.

## Quick setup

1. Create a virtual environment (recommended):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

3. Configure API keys:

Create a `.env` file in the project root (copy from `env.example`):

```powershell
copy env.example .env
```

Then edit `.env` with your actual API keys:

```env
OPENAI_API_KEY=sk-your-actual-key-here
HUGGINGFACE_API_KEY=your-actual-key-here
OLLAMA_HOST=http://localhost:11434
```

Notes:
- `faiss`/`faiss-cpu` is intentionally not a hard requirement in `requirements.txt` because installation is platform-specific.
- API keys are loaded automatically from `.env` file using `python-dotenv`.

## FAISS (vector store) on Windows

FAISS can be tricky to install on Windows via pip. Options:

- Use conda (recommended on Windows):

```powershell
conda create -n nia python=3.11 -y
conda activate nia
conda install -c pytorch faiss-cpu -y
pip install -r requirements.txt
```

- Or use WSL (Ubuntu) where `pip install faiss-cpu` is more likely to work.

If FAISS is not installed the memory system will fall back to disabled vector search; the code is designed to run without FAISS but some features will be limited.

## Running tests and checks

Run unit tests:

```powershell
pytest -q
```

Run static checks (mypy, flake8):

```powershell
mypy .
flake8
```

# NIA — Cognitive Loop skeleton

This repository contains a modular skeleton for NIA's Cognitive Loop
architecture (Perceive → Reason → Act → Reflect).

Files created in this step:
- `core/brain.py` — main `CognitiveLoop` class (parse_input, reason, perform_action, reflect, run)
- `core/memory.py` — MemoryManager interface and `InMemoryMemory` stub
- `core/tools/tool_manager.py` — ToolManager stub (includes an `echo` tool)
- `models/model_manager.py` — ModelManager stub for model interactions
- `interface/chat.py` — Minimal CLI to demo the loop

How to try (PowerShell on Windows):

```powershell
# ensure you are in the repo root, e.g.:
cd 'C:\Users\rajbh\N.I.A'
# Run the demo CLI
python -m interface.chat
```

Notes:
- The code is intentionally a skeleton. TODO comments mark where to
  connect real LLMs, tools, and persistent memory.
- Follow the module interfaces when replacing stubs to keep the core
  `CognitiveLoop` unchanged.

## Testing & Extending

To run all tests (unit, integration):

```powershell
pytest
```

## Memory Vector Backend
- `MemoryManager` supports FAISS by default when installed.
- Configure alternate backends via `vector_backend` parameter (e.g., `"faiss"`, `"external"`).
- When no vector backend is available, similarity search raises: `Vector search not available (configure vector_backend or integrate an external provider)`.

To add tests, create files as `tests/test_*.py` (see scaffold below) covering the following modules:
- `core/memory.py` — Test MemoryManager (CRUD, TTL, vector fallback)
- `core/brain.py` — Test CognitiveLoop for multi-step, fallback, and error cases
- `core/tools` — Test registration/discovery, permission hooks

Sample test skeleton:
```python
import pytest
from core.memory import MemoryManager, InMemoryMemory


def test_inmemory_basic():
    mem = InMemoryMemory()
    mem.store('demo', 'foo', {'x': 1})
    assert mem.retrieve('demo', 'foo')['x'] == 1
```

## Voice Features

You can enable 'talk' and 'listen' by plugging in 3rd-party libraries to ToolManager:
- `ToolManager.speak(text)`: TTS, e.g., via `pyttsx3`
- `ToolManager.listen()`: ASR, e.g., via `speech_recognition`

Default CLI accepts `--voice` to enable voice mode (falls back silently if not available).

### Async vs Sync Usage
- Implement `arun(self, params)` for asynchronous tools, or `run(self, params)` for synchronous ones.
- Use `ToolManager.execute_async(...)` to run tools concurrently with other async tasks.
- `ToolManager.execute(...)` will raise if a coroutine is detected while an event loop is already running; prefer the async API in concurrent contexts.

## How to Add a Tool
1. Create a Python file in `core/tools/`.
2. Inherit from `object` and provide `.name` and `.run(params)`.
3. Register it from your code or rely on `ToolManager.discover_and_register()`.

Sample:
```python
class HelloTool:
    name = 'hello'
    def run(self, params):
        return {'message': 'hi'}
```

## Architectural Highlights (2025)
- All ToolManager logic is registry/discovery-based, async-capable, and supports permission hooks.
- Voice capability is modular and togglable.
- ModelManager and Memory are injectable; core has no hard deps on any single backend.
- Reflection and trace logging are robust and extensible for future LLM-driven improvements.

## ToolManager APIs and Usage

- Preferred API: `core.tool_manager.ToolManager` (raw result API). Methods:
  - `register(name, func)` — register callable tools
  - `register_tool(cls_or_instance)` — register class/instance with `name` and `run`
  - `execute(tool_name, params, timeout=None)` — returns raw tool output or raises
  - `execute_async(tool_name, params, timeout=None)` — async variant
  - `list_tools()`, `has_tool()`, plugin helpers

- Legacy adapter removed: use only `core.tool_manager.ToolManager` for all new and existing code.
- Dev tool registration is explicit via `core.tools.register_dev_tools(mgr)` and called in `interface/chat.py` on startup. Import-time auto-registration is disabled by default. To enable gated auto-registration, set `NIA_AUTO_REGISTER_DEV_TOOLS=1` in the environment.
- Internal code (brain, chat) uses the preferred API. For all code, use `core.tool_manager.ToolManager` and call `register_dev_tools` when you need built-in tools.

Notes and examples (2025 updates)

- Explicit dev tool registration: The project now uses an explicit helper `core.tools.register_dev_tools(mgr)` to register the small set of local "dev" tools (echo, hello, etc.). This avoids import-time side effects. Call this from a startup path (for example, `interface.chat.main()` registers them at runtime).

Example (register dev tools):

```python
from core.tool_manager import ToolManager
from core.tools import register_dev_tools

mgr = ToolManager()
register_dev_tools(mgr)
```

- Plugin reload helper: `ToolManager` now provides a `reload_plugins(directory: str) -> int` helper to centralize unload-and-load semantics. Use it to hot-reload plugins safely from the running process.

Example (reload plugins):

```python
from core.tool_manager import ToolManager

mgr = ToolManager()
loaded = mgr.reload_plugins('plugins')
print(f"Reloaded {loaded} plugin(s)")
```

- Import-time side effects and dotenv: Loading environment variables (via `load_dotenv()`) was moved to runtime in `interface.chat.main()` to avoid mutating global state on import. Also, `interface.main()` is now a lazy wrapper that imports `interface.chat` only when executed — keep imports lightweight for tests and library consumers.

- Lazy imports for optional dependencies: Heavy optional dependencies such as `numpy` and `faiss` are now lazy-imported inside the `VectorStore` constructor. `MemoryManager` will attempt to instantiate a FAISS-backed `VectorStore` only when requested and will fall back gracefully if the packages are not installed. Add tests that simulate missing modules to ensure fallback behavior remains stable.

- Backwards compatibility: A small legacy adapter remains available for compatibility in `core.tools.__init__` to reduce test friction; prefer `core.tool_manager.ToolManager` for all new code.

## Plugin System: Extending NIA with Hot-Plug Tools

- Place any `.py` tool file in the `plugins/` directory. Each file must contain at least one class with `name` and `run(params)` attributes (see below).
- Use the CLI commands:
    - `list plugins` — view all loaded plugin tools
    - `reload plugins` — hot-reload all plugins without restarting
    - `unload plugin <tool_name>` — unload a specific plugin tool
- Plugins are loaded at startup and can be hot-reloaded/unloaded at any time.

### Secure Plugin Loading (Allowlist)
- Set `NIA_PLUGIN_SAFE_MODE=1` to enable allowlist-based plugin loading.
- Create `plugins/ALLOWLIST.txt` with one plugin base name per line (e.g., `hello_plugin`). Only allowlisted `.py` files will be loaded.
- This reduces risk by preventing unreviewed plugin execution.

### Authoring a Plugin Example
Create a file, e.g. `plugins/hello_plugin.py`:
```python
class HelloPlugin:
    name = "hello_plugin"
    def run(self, params):
        return {"hello": "from your plugin!", "params": params}
```
- These are loaded and become callable tool names in NIA automatically.

### Add & reload a sample plugin (runtime)
Create `plugins/my_plugin.py`:

```python
class MyPlugin:
  name = "my_plugin"
  def run(self, params):
    return {"ok": True, "params": params}
```

Reload from Python at runtime (hot-reload):

```python
from core.tool_manager import ToolManager

mgr = ToolManager()
# load plugins (or reload existing ones)
loaded_count = mgr.reload_plugins('plugins')
print(f"Reloaded {loaded_count} plugin(s)")

# run the plugin after reload
result = mgr.execute('my_plugin', params={"x": 1})
print(result)
```

Reload via the CLI sample flow (when running `python -m interface.chat`):

```powershell
python -m interface.chat
> reload plugins
Reloaded 1 plugin(s)
> run my_plugin {"x": 1}
{"ok": true, "params": {"x": 1}}
```

### Best Practices and Safety
- Only keep trusted plugins in the plugins directory (plugins are Python code and execute in-process).
- Use exception handling inside your plugin's `run()` for stability.
- Keep plugin dependencies to a minimum or handle imports inside plugin classes.

### Is the plugin system production-ready?
- Short answer: the plugin system is ready for development and local testing, including hot-reload; it is not hardened for running untrusted plugins in production.

- Why: plugins execute in-process with the main application. This makes them efficient for development but places no sandboxing boundary around code — a malicious or buggy plugin can access the app's memory, file system, and environment.

- Recommendations for production use:
  - Use `NIA_PLUGIN_SAFE_MODE=1` and maintain a vetted allowlist (`plugins/ALLOWLIST.txt`).
  - Review plugin code or adopt a signed-plugin model for trust.
  - Run untrusted plugins in a separate process or container and communicate via IPC or RPC if you need strong isolation (future improvement).

See built-in `plugins/demo_plugin.py` as a reference.
# N.I.A

## GitHub Copilot Configuration & Usage

This project includes guidance for GitHub Copilot under `.github/copilot-instructions.md`.

Purpose
- Provide Copilot with architectural guardrails to generate code consistent with NIA’s patterns.
- Reinforce the preferred `ToolManager` API, explicit tool registration, and no import-time side effects.

Usage Guidelines
- Read `.github/copilot-instructions.md` before using Copilot on this repo.
- Use `core.tool_manager.ToolManager` (raw API) universally; the legacy adapter has been removed.
- Do not register tools at import time. Call `core.tools.register_dev_tools(mgr)` explicitly from startup code (see `interface/chat.py`).
- Optional environment gating for demos: set `NIA_AUTO_REGISTER_DEV_TOOLS` to enable best-effort dev tool registration at import time.

Relevant Configuration
- Environment variable: `NIA_AUTO_REGISTER_DEV_TOOLS` controls import-time dev tool auto-registration.

```powershell
# Disabled by default (recommended for production/testing)
$env:NIA_AUTO_REGISTER_DEV_TOOLS = "0"

# Enable for local demos (best-effort, safe to leave off)
$env:NIA_AUTO_REGISTER_DEV_TOOLS = "1"
```

Maintenance & Version Control
- Keep `.github/copilot-instructions.md` up to date when refactoring architecture (tool registration policies, API changes).
- Changes should be reviewed via pull requests together with code updates that alter `ToolManager` behavior or registration flow.
- Ensure tests remain aligned with the documented behavior (no import-time side effects, idempotent registration).
