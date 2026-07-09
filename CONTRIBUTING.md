# Contributing to N.I.A

Thank you for your interest in contributing to N.I.A! This guide will help you get started.

## Development Setup

```bash
# Clone the repo
git clone https://github.com/Adi103-ETAI/N.I.A.git
cd N.I.A
git checkout insight

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .

# Install frontend dependencies (for React TUI)
cd frontend/terminal && npm install && cd ../..

# Run tests
uv run python -m pytest tests/ -q

# Run NIA
uv run python -m niaharness
```

## Project Structure

```
src/niaharness/          # The harness (engine, tools, providers, permissions)
src/agents/nia/          # The NIA agent (identity, memory, personality)
frontend/terminal/       # React + Ink terminal UI
tests/                   # Python test suite
skills/bundled/          # Bundled skill definitions
```

## Key Principles

1. **One LLM call per turn.** NIA does NOT make a separate "thinking" call. The QueryEngine's LLM call IS the brain.
2. **NIA is the agent, niaharness is the runtime.** NIA owns identity (SOUL.md, memory, personality). niaharness owns execution (tools, permissions, hooks, MCP).
3. **Port from Hermes, don't transliterate.** Read the Hermes code, understand the behavior, implement a simpler version that fits NIA's architecture. Don't copy 7,000 LOC when 500 will do.
4. **Test before committing.** Run `uv run python -m pytest tests/ -q` before every commit.

## Hard File Locks

The following files require explicit approval before modification:
- `AGENTS.md`
- `pyproject.toml`
- `prompts/system.md`
- `src/niaharness/cli.py`
- `src/niaharness/tools/__init__.py`
- `src/niaharness/api/client.py`
- `src/niaharness/engine/query_engine.py`

## Commit Convention

Use conventional commits:
```
feat(scope): description
fix(scope): description
refactor(scope): description
docs(scope): description
cleanup: description
```

## Adding a New Tool

1. Create `src/niaharness/tools/my_tool_tool.py`
2. Implement `MyTool(BaseTool)` with `name`, `description`, `input_model`, `execute()`
3. Register in `src/niaharness/tools/__init__.py` (import + add to tuple)
4. Add tests in `tests/test_tools/`
5. Run `uv run python -m pytest tests/test_tools/ -q`

## Adding a New Provider

1. Create `src/niaharness/providers/my_provider.py`
2. Implement `MyProvider(LLMProvider)` with `config`, `get_client()`, `resolve_api_key()`
3. Register in `src/niaharness/providers/registry.py`
4. Add env var to `.env.example`
5. Test with `--provider my_provider --list-models`

## Code Style

- Python: `ruff check` + `mypy` (per `pyproject.toml`)
- TypeScript: `eslint` + `prettier` (per frontend config)
- Line length: 100 chars
- Type hints: required on all public functions

## Questions?

Open a GitHub issue or check `AGENTS.md` for the full architecture guide.
