# NIA Audit Checklist

> Generated from full audit of `/home/kali/Desktop/N.I.A` (branch: `insight`, commit: `67b251e`)
> 263 Python source files, ~39K lines, 41 registered tools, 11+ LLM providers
> Test suite: 288 PASS, 24 FAILED, 32 ERRORS (excluding UI tests)

---

## P0 — CRITICAL (runtime crashes / broken core)

- [ ] **LSP tool calls async function without `await`**
  - File: `src/niaharness/tools/lsp_tool.py:79`
  - `list_document_symbols()` is a coroutine but is called without `await`
  - Results in `TypeError: 'coroutine' object is not iterable` at runtime
  - Fix: add `await` before the call

- [ ] **Stubs in `src/niaharness/services/__init__.py` shadow real implementations**
  - All functions (`compact_messages`, `save_session_snapshot`, `estimate_tokens`, etc.) are `*args, **kwargs -> return None` stubs
  - These silently override real implementations in `session_storage.py` and `compact.py`
  - Any code importing from `niaharness.services` gets a no-op instead of real behavior
  - Fix: remove stubs or re-export from the real modules

- [ ] **Cron service module crashes on collection — 32 tests error out**
  - All tests in `tests/test_services/test_cron.py` and `tests/test_services/test_cron_scheduler.py` fail at collection phase
  - Module has an import-level crash that prevents pytest from even loading it
  - Fix: identify the import chain failure and resolve it

---

## P1 — HIGH (test failures, broken features, missing core capabilities)

- [ ] **FileWriteTool field name mismatch**
  - Tool input model uses `file_path` but integration tests pass `path`
  - Causes `ValidationError` at runtime for any test/caller using the wrong key
  - Fix: align field name in the Pydantic model to `file_path` or rename to `path` and fix the tool

- [ ] **`save_session_snapshot()` API drift**
  - `tests/test_untested_features.py:403` calls with `cwd=` keyword arg
  - Real function in `session_storage.py` has different signature
  - Fix: update the test or fix the function signature

- [ ] **Interactive browser tool — MISSING**
  - NIA has `web_search` and `web_fetch` (static HTTP requests only)
  - No ability to navigate pages, click elements, type into forms, take snapshots
  - This is the single biggest capability gap vs modern AI coding assistants
  - Fix: add a browser tool (Playwright/Selenium-based with session management)

- [ ] **Dual CLI entry points cause confusion**
  - `niaharness` CLI (at `src/niaharness/cli.py`) has full feature set
  - `agents.nia` CLI (at `src/agents/nia/__main__.py`) has different flags and flow
  - Unclear which is the primary/recommended entry point for users
  - Fix: converge to one CLI or make `agents.nia` delegate to `niaharness`

- [ ] **Web search tests fail — no mock network layer**
  - `test_web_search_tool_reads_results` fails because it hits real network
  - Fix: add pytest mocking/VCR integration for HTTP tests

---

## P2 — MEDIUM (code quality, tool gaps, developer experience)

- [ ] **Dev dependencies broken — `uv sync --extra dev` fails**
  - `pyproject.toml` defines dev deps under `[project.optional-dependencies] dev`
  - `uv` expects different format — can't install pytest/ruff/mypy via project command
  - Fix: migrate to `[dependency-groups]` or use `uv pip install` workaround

- [ ] **`ruff` and `mypy` not runnable**
  - Both listed as dev tools but never executed in CI or pre-commit
  - No linting/type-checking baseline established
  - Fix: add `ruff check` and `mypy src/` to validation workflow

- [ ] **No syntax or type consistency enforcement**
  - Some modules have full type annotations, many don't
  - No pre-commit hooks or CI gate for code quality
  - Fix: add pre-commit config, run ruff/mypy, fix existing violations

- [ ] **`prompts/system.md` tool names may not match actual registry**
  - The system prompt lists tool contracts that were manually maintained
  - Could drift from actual tool names/args in `tools/__init__.py`
  - Fix: add a CI check that validates tool names in system prompt match registered tools

- [ ] **Skills tool is read-only — no CRUD**
  - `skill_tool.py` only reads existing skills
  - Cannot create, edit, or delete skills through the tool interface
  - Fix: extend skill tool with create/update/delete actions

- [ ] **Code execution tool — MISSING**
  - No sandboxed Python/JS execution tool (like `execute_code` in Hermes)
  - Only `bash` tool for arbitrary shell commands
  - Fix: add a `run_code` tool with timeout, output capture, and resource limits

---

## P3 — LOW (nice-to-have features, integrations)

- [ ] **Computer Use / GUI automation — MISSING**
  - No ability to drive desktop applications (click, type, scroll, screenshot)
  - Fix: add desktop automation via PyAutoGUI/cua-driver or similar

- [ ] **GitHub integration tools — MISSING**
  - No PR review, issue management, repo cloning via tools
  - Fix: add `github_pr`, `github_issue`, `github_search` tools using `gh` CLI or REST API

- [ ] **Image/Vision analysis — MISSING**
  - No tool to analyze images/screenshots
  - Fix: add `vision_analyze` tool that sends images to multimodal LLM

- [ ] **Text-to-Speech — MISSING**
  - `nia_voice` does transcription (speech-to-text) but no text-to-speech
  - Fix: add a `speak` or `tts` tool

- [ ] **Integration tools — MISSING**
  - No tools for: Google Workspace, Notion, Airtable, Obsidian, email
  - Fix: add integrations as needed per use case

- [ ] **YouTube / media extraction — MISSING**
  - No YouTube transcript or media content extraction
  - Fix: add `youtube_transcript` or similar tool

- [ ] **Diagramming / creative tools — MISSING**
  - No Excalidraw, SVG, p5.js, or ASCII art generation
  - Fix: add as separate tools or MCP servers

---

## P4 — INFRASTRUCTURE (testing, CI/CD, docs)

- [ ] **Frontend `node_modules` was committed to git**
  - ~170K lines of deleted node_modules in latest pull shows it was checked in
  - Fix: ensure `frontend/terminal/node_modules` is in `.gitignore`, verify it stays out

- [ ] **No CI pipeline**
  - No GitHub Actions or other CI running tests on push/PR
  - Fix: add a CI workflow that runs `pytest`, `ruff`, `mypy`

- [ ] **Test coverage gaps**
  - No tests for: permissions module, hooks, MCP auth flow, team lifecycle, frontend API
  - Cron tests completely broken (32 errors)
  - Fix: add coverage reporting, target 60%+ coverage

- [ ] **`AGENTS.md` hard file locks may block development**
  - Files like `cli.py`, `tools/__init__.py`, `query_engine.py` are locked from agent editing
  - Good for safety but may slow iteration
  - Fix: review lock list periodically, keep only essential locks

---

## QUICK WINS (estimated effort)

| Fix | Effort | Impact |
|-----|--------|--------|
| Add `await` to LSP tool call | 5 min | Fixes runtime crash |
| Remove stubs from `services/__init__.py` | 15 min | Fixes silent no-op behavior |
| Fix `FileWriteTool` field name | 15 min | Fixes validation errors |
| Fix dev dependency config | 30 min | Enables ruff/mypy |
| Fix cron module import crash | 1-2 hr | Unblocks 32 tests |
| Add browser tool | 4-8 hr | Biggest capability gap |
| Fix `save_session_snapshot` test | 15 min | Fixes 1 failing test |

---

*Generated by Hermes Agent during audit of NIA project on 2026-07-06*
