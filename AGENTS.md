# AGENTS.md — N.I.A Multi-Agent Coordination Instructions

## Scope
This file applies to work in the repository root for coordinating multiple AI coding agents. If per-folder AGENTS.md files exist, they take precedence over this file.

## Required Git Rules
- Commit every turn of work
- Do not amend commits
- Do not change branches without explicit user permission
- Do not push, pull, or rebase unless explicitly requested

## Commit Expectations
- Focused commits scoped to the requested task
- Conventional commit messages (feat:, fix:, refactor:, chore:, docs:, test:)
- No generated-with lines, attribution blocks, or command transcripts in commit messages

## Validation
uv run python -m pytest -q

## Safety
- Never delete or truncate persisted session/state artifacts under `src/niaharness/state/` and session tooling used by `nia_session`.
- Never change tool names or default tool registry wiring in `src/niaharness/tools/__init__.py` without running the full test suite.
- Never bypass or relax permission/auth behavior without updating/confirming tests under `tests/test_permissions/` and `tests/test_api/`.
- Never alter the CLI contract without running `tests/test_commands/` (Typer-based commands).
- Never manually edit dependency lockfiles; use `uv` to update.

## Project Overview
N.I.A is an AI-powered CLI coding assistant (niaharness + agents/nia). It runs an interactive or print-mode conversation loop that calls tools (files, shell, web, MCP, and task execution) under a permission and hook system. It supports session continuation and includes a pytest suite covering coordinator, engine, tools, MCP flows, UI, and permissions.

## Architecture
```
main.py / python -m niaharness
        |
        v
niaharness CLI (Typer)
        |
        v
niaharness UI (REPL / print-mode) ---- frontend/terminal (React/Ink)
        |
        v
QueryEngine (conversation loop)
  -> PermissionChecker -> HookExecutor -> CostTracker -> FileStateCache
        |
        v
ToolRegistry (built-ins + MCP adapters + skills/plugins)
        |
        v
API Client (Anthropic/OpenAI-compatible streaming)
        |
        v
Model responses -> tool calls -> tool results -> next turn
```

## Multi-Agent Role Table
All agents have read access to the full codebase. This table defines who
leads decisions in each area — not who is allowed to open which files.
When making changes outside your primary area, update TASKS.md first.

| Agent         | Role                   | Leads On                                                                 | Hard Restrictions |
|---------------|------------------------|---------------------------------------------------------------------------|-------------------|
| OpenCode      | Orchestrator / Arch   | `src/niaharness/engine/*`, `src/niaharness/coordinator/*`, `src/agents/nia/*` (architecture + agent wiring) | AGENTS.md, TASKS.md, Hard File Locks |
| Cursor Agent  | Feature implementation| `src/`, `tests/`, `frontend/terminal/` (implementation + tests)          | AGENTS.md, TASKS.md, Hard File Locks |
| BlackboxAI    | Config / Boilerplate  | `pyproject.toml`, repo-wide tooling scripts/docs (non-runtime wiring)    | AGENTS.md, TASKS.md, Hard File Locks |

Hard File Locks — no agent touches these without explicit user instruction:
Always include AGENTS.md and TASKS.md. Do not modify any of the following without direct user permission:
- AGENTS.md
- TASKS.md
- pyproject.toml
- prompts/system.md
- src/niaharness/cli.py
- src/niaharness/tools/__init__.py
- src/niaharness/api/client.py
- src/niaharness/engine/query_engine.py

## Project Structure
| Path | Purpose |
|------|---------|
| `src/` | Python implementation (niaharness + agents/nia) |
| `src/agents/nia/` | NIA-specific wiring and entrypoints |
| `src/niaharness/` | Harness: CLI, engine, tools, permissions, hooks, MCP, UI |
| `tests/` | Pytest test suite |
| `docs/` | Developer and coordination documentation |
| `prompts/` | Prompt assets and system prompt |
| `frontend/` | React terminal UI workspace |
| `frontend/terminal/` | Ink/React TUI (TypeScript) |
| `scripts/` | Automation and smoke-test scripts |
| `main.py` | Entrypoint wrapper calling `niaharness.cli:app` |
| `architecture_suggestions.md` | Architecture improvement ideas |
| `CHANGELOG.md` | Repo change log |

## Key Files
| Path | Purpose |
|------|---------|
| `pyproject.toml` | Dependencies, pytest, ruff, mypy config |
| `main.py` | CLI entrypoint wrapper |
| `src/niaharness/cli.py` | Typer CLI definition and session/print-mode flags |
| `src/niaharness/__main__.py` | `python -m niaharness` entrypoint |
| `src/niaharness/api/openai_client.py` | OpenAI-compatible streaming + retry logic |
| `src/niaharness/api/client.py` | Shared API event types and retry utilities |
| `src/niaharness/engine/query_engine.py` | Core conversation engine (async loop, abort handling) |
| `src/niaharness/tools/__init__.py` | Default tool registry and NIA tool wiring |
| `src/niaharness/permissions/checker.py` | Permission enforcement for tool execution |
| `src/niaharness/hooks/executor.py` | Hook pipeline around the tool/model loop |
| `src/niaharness/mcp/` | MCP client/server integration |
| `src/niaharness/plugins/` | Plugin loading and installer |
| `src/niaharness/services/session_storage.py` | Session snapshot persistence used by CLI `--continue/--resume` |
| `prompts/system.md` | System prompt defining tool expectations and contract |
| `frontend/terminal/package.json` | Frontend dev/start command |
| `frontend/terminal/src/App.tsx` | Frontend root component |

## Development
uv run python -m niaharness
uv run python -m niaharness -p "Summarize the purpose of this repository"
cd frontend/terminal && npm install
cd frontend/terminal && npm run start

## Testing
uv run python -m pytest -q

## Build Requirements
- Python >= 3.10
- Node.js for `frontend/terminal` dev server
- `uv` for dependency management
- Linux-compatible runtime

## Conventions
- Ruff line length: 100 (from `pyproject.toml`)
- mypy strict mode enabled (from `pyproject.toml`)
- Async tool execution and streaming APIs are preferred in the engine/tool loop
- Keep tool names stable; prompt/tool contracts and tests rely on them

## Session persistence invariants
CLI session flows depend on snapshot persistence. `src/niaharness/cli.py` uses session storage when running with `--continue` and `--resume`, and tests validate storage behavior under `tests/test_services/test_session_storage.py`. Do not change snapshot schema or storage locations without updating tests.

## Permission gating contracts
Tool execution depends on `src/niaharness/permissions/checker.py` behavior and tool declared read-only semantics. Changes affecting permission outcomes must include test updates under `tests/test_permissions/`.

## Tool registry / prompt contract
Tool identifiers and wiring are part of a contract used by the system prompt (`prompts/system.md`) and by tool/permission tests. Do not rename or remove tools in `src/niaharness/tools/__init__.py` without updating prompt content and running the full suite.

## Multi-backend API behavior
Streaming and retry behavior differ by provider, but the agent loop expects compatible message/tool call semantics. OpenAI-compatible behavior is implemented in `src/niaharness/api/openai_client.py`; changes that affect streaming parsing must preserve the `Api*Event` protocol expected by the engine and tests.

## Pull Requests / Handoffs
Update `TASKS.md` before stopping and before switching agents.
