# Changelog

All notable changes to N.I.A will be documented in this file.

## [Unreleased]

## 2026-07-07 — Hermes/Jarvis Capability Layer + P0 Audit Tasks

This session delivered 7 commits across 2 phases: (1) a repair pass that
fixed the missing `services/` package and multiple audit P0/P1 bugs, and
(2) the 5 P0 tasks from the Hermes vs NIA gap analysis.

### Phase 2 — P0 Audit Tasks (5 commits)

#### P0 Task 5 — Self-improving background memory review loop
- **New:** `src/niaharness/engine/background_review.py` — after every turn,
  a daemon thread reviews the conversation and saves durable
  facts/preferences/patterns to memory. Non-blocking, best-effort.
  Configurable via `NIA_BACKGROUND_REVIEW`, `NIA_BACKGROUND_REVIEW_MODEL`,
  `NIA_BACKGROUND_REVIEW_INTERVAL` env vars.
- **Changed:** `QueryEngine.__init__` accepts optional `memory=` param;
  `submit_message` spawns the review after each turn completes.
- **Tests:** 36 new tests in `tests/test_engine/test_background_review.py`.

#### P0 Task 4 — `vision_analyze` tool
- **New:** `src/niaharness/tools/vision_analyze_tool.py` — multimodal image
  analysis. Accepts URLs or local file paths, makes its own LLM call to a
  vision-capable model (separate from the main conversation). 3-tier
  config: `NIA_VISION_*` env vars → main agent settings → `OPENAI_API_KEY`.
- **Tests:** 24 new tests in `tests/test_tools/test_vision_analyze_tool.py`.

#### P0 Task 3 — FTS5-backed session search
- **New:** `src/niaharness/services/session_search.py` — SQLite FTS5 index
  over all session snapshots. Three modes: DISCOVERY (search), SCROLL
  (read a message window), BROWSE (list recent).
- **New:** `src/niaharness/tools/session_search_tool.py` — agent-callable
  tool with `search`, `scroll`, `browse`, `rebuild`, `stats` actions.
- **Changed:** `save_session_snapshot()` now auto-indexes via
  `index_session_on_save()` hook. Every saved session is immediately
  searchable.
- **Tests:** 19 new tests in `tests/test_tools/test_session_search_tool.py`.

#### P0 Task 2 — `skill_manage` tool
- **New:** `src/niaharness/tools/skill_manage_tool.py` — 6 operations
  (create, update, edit, delete, list, info) for agent-driven skill CRUD.
  Only writes to user skills dir; bundled skills are read-only. YAML
  frontmatter validation, path-traversal protection, 100K char limit.
- **Tests:** 22 new tests in `tests/test_tools/test_skill_manage_tool.py`.

#### P0 Task 1 — SOUL.md identity file system
- **New:** `src/niaharness/prompts/soul.py` — `load_soul_md()`,
  `get_nia_home()`, `DEFAULT_SOUL_MD`, `is_default_soul()`. Lives at
  `~/.nia/SOUL.md` (or `$NIA_HOME`). Seeded automatically on first run
  with a Jarvis-flavored default. Loaded as slot #1 in the system prompt.
- **Changed:** `build_system_prompt()` now prepends SOUL.md (new
  `include_soul=True` kwarg). `NIA._build_merged_system_prompt()` loads
  SOUL.md as the first section.
- **New:** `/soul` slash command (`show`, `path`, `edit`, `reset`).
- **Tests:** 5 new tests in `tests/test_prompts/test_system_prompt.py`.

### Phase 1 — Repair Pass (7 commits)

#### Hermes/Jarvis capability layer
- **New:** `browser_tool.py` — Playwright-based interactive browser
  (navigate, click, type, snapshot, screenshot, eval_js, back/forward/
  reload, close). Singleton session per process. URL safety: blocks
  `file://`, `data://`, private hosts.
- **New:** `run_code_tool.py` — sandboxed Python subprocess with timeout
  (1-120s), stdout/stderr capture, exit code, duration.
- **New:** `speak_tool.py` — text-to-speech via KittenTTS (Jasper voice,
  ~15M params, runs on CPU). Falls back to espeak.

#### Missing `services/` package — built from scratch
The entire `src/niaharness/services/` package was advertised by commit
`6420439` but never actually committed. 18 source files + 32 tests
imported from it and crashed at collection time.

- **New:** `services/compact.py` — token estimation, `summarize_messages`,
  `compact_messages`, `microcompact_messages`, `AutoCompactState`,
  `auto_compact_if_needed`.
- **New:** `services/session_storage.py` — `save_session_snapshot`,
  `load_session_snapshot`, `load_session_by_id`, `list_session_snapshots`,
  `export_session_markdown`, `get_project_session_dir`.
- **New:** `services/cron.py` — full CRUD (`upsert_cron_job`,
  `delete_cron_job`, `get_cron_job`, `set_job_enabled`, `mark_job_run`),
  5-field cron validator, `next_run_time` calculator.
- **New:** `services/cron_scheduler.py` — `run_scheduler_loop`,
  `execute_job`, `append_history`, `load_history`, daemon control.
- **New:** `services/lsp.py` — pure-Python AST-based code intelligence
  (`list_document_symbols`, `workspace_symbol_search`, `go_to_definition`,
  `find_references`, `hover`).

#### Other repair-pass fixes
- **Fixed:** `paths.py` now accepts `OPENHARNESS_*` env vars as legacy
  aliases (30+ tests relied on them).
- **Fixed:** FileWriteTool/FileReadTool/FileEditTool schemas — added
  `path`/`old_str`/`new_str` aliases via Pydantic v2 `populate_by_name`.
- **Fixed:** Tool name constants renamed for consistency (`file_read` →
  `read_file`, `file_edit` → `edit_file`). Updated `prompts/system.md`.
- **Fixed:** `cron_list_tool` NoneType crash when `last_run` is None.
- **Fixed:** `grep_tool` broken `from .utils import to_relative_path`
  import. Changed default `output_mode` to `content`.
- **Fixed:** `to_relative_path` now accepts optional `base` arg.
- **Fixed:** Wired `read_state_tracker` into FileReadTool + FileWriteTool.
- **Fixed:** Critical `run_query` mutation bug — `auto_compact_if_needed`
  returned a list copy, so `messages.append(final_message)` modified the
  copy, not the caller's `self._messages`. Assistant replies were silently
  dropped. Fixed with `messages[:] = compacted_messages`.
- **Fixed:** `submit_message` filters out the final `QueryResult` event
  so callers see `AssistantTurnComplete` as the last streamed event.
- **Fixed:** `.gitignore` `services/` rule scoped to `/services/` so the
  new `src/niaharness/services/` package is tracked.

#### Test impact
- **Before this session:** ~288 passing, 24 failed, 32 errors
- **After:** 458 passing, 2 pre-existing failures (network + React UI)
- **+170 tests fixed/added**

---

## NIA Inventory (as of 2026-07-07)

### Commits
- **90 total commits** in the repo
- **7 commits this session** (all pushed to `insight` branch):
  - `63cc327` feat(learning): self-improving background memory review loop (P0 Task 5)
  - `1cb2c9b` feat(tools): vision_analyze tool (P0 Task 4)
  - `1708b52` feat(search): FTS5 session search (P0 Task 3)
  - `9a4d7b0` feat(tools): skill_manage tool (P0 Task 2)
  - `f047d08` feat(identity): SOUL.md identity file system (P0 Task 1)
  - `8541609` docs: Hermes vs NIA working audit + gap analysis
  - `55c03f7` feat(tools): Hermes/Jarvis capability layer — browser, run_code, speak

### Codebase
- **272 Python files** in `src/`
- **~39,400 lines of Python code**
- **13 React/TypeScript components** in `frontend/terminal/`

### Tools — 47 registered

| Category | Tools |
|---|---|
| File ops (5) | `read_file`, `write_file`, `edit_file`, `notebook_edit`, `glob` |
| Search (3) | `grep`, `lsp`, `tool_search` |
| Shell & code (2) | `bash`, `run_code` ✨ |
| Web (3) | `web_search`, `web_fetch`, `browser` ✨ |
| Vision & voice (2) | `vision_analyze` ✨, `speak` ✨ |
| Skills (2) | `skill`, `skill_manage` ✨ |
| Memory & search (3) | `nia_memory`, `nia_context`, `session_search` ✨ |
| Tasks & agents (8) | `task_create`, `task_get`, `task_list`, `task_output`, `task_stop`, `task_update`, `agent`, `send_message` |
| Teams (2) | `team_create`, `team_delete` |
| Cron (5) | `cron_create`, `cron_list`, `cron_delete`, `cron_toggle`, `remote_trigger` |
| MCP (3) | `mcp_auth`, `list_mcp_resources`, `read_mcp_resource` |
| Session (1) | `nia_session` |
| Voice (1) | `nia_voice` (STT) |
| Planning (4) | `todo_write`, `enter_plan_mode`, `exit_plan_mode`, `brief` |
| Git worktrees (2) | `enter_worktree`, `exit_worktree` |
| Meta (3) | `config`, `sleep`, `ask_user_question` |

✨ = added this session (7 new tools)

### Skills — 7 bundled

All in `software-development` category:
- `plan`, `debug`, `diagnose`, `review`, `simplify`, `commit`, `test`

Plus the `skill_manage` tool now lets the agent create its own skills at
runtime (stored in `~/.niaharness/skills/`).

### Providers — 15 LLM providers

Two provider layers, unified through `ProviderRegistry`:

**niaharness/providers/ (8):**
- `anthropic`, `openai`, `bedrock`, `vertex`, `azure`, `mistral`

**agents/nia/providers/ (12, with overlap):**
- `anthropic`, `openai`, `openrouter`, `groq`, `together`, `deepseek`,
  `google`, `nvidia`, `cerebras`, `fireworks`, `ollama`

**Unique total: 15 providers** with dynamic switching + model fetching.

### Services — 7 modules

| Module | Purpose |
|---|---|
| `compact.py` | Token estimation + auto-compaction |
| `session_storage.py` | Session snapshot persistence (JSON) |
| `session_search.py` ✨ | FTS5-backed session search (SQLite) |
| `cron.py` | Cron job registry (CRUD + validation) |
| `cron_scheduler.py` | Background cron daemon |
| `lsp.py` | Python code intelligence (AST-based) |
| `__init__.py` | Re-exports |

✨ = added this session

### NIA Core (the soul) — 6 modules

| Module | Purpose |
|---|---|
| `brain.py` | LLM-powered decision making (structured JSON output) |
| `personality.py` | Jarvis personality (moods, greetings, tone) |
| `memory.py` | Short-term + long-term memory (JSON file) |
| `context.py` | Context awareness (time, cwd, git, project type) |
| `react.py` | ReAct loop (Plan → Act → Reflect) |
| `soul.py` ✨ (in `niaharness/prompts/`) | SOUL.md identity file system |

### Engine — 7 modules

| Module | Purpose |
|---|---|
| `query_engine.py` | High-level conversation engine |
| `query.py` | Low-level tool-aware query loop |
| `messages.py` | Conversation message models |
| `stream_events.py` | 18 event types |
| `cost_tracker.py` | Token/cost tracking |
| `background_review.py` ✨ | Self-improving learning loop |
| `__init__.py` | Re-exports |

### Other subsystems

| Subsystem | Modules | Purpose |
|---|---|---|
| Permissions | 3 | Tool permission enforcement (4 modes) |
| Hooks | 7 | Pre/post hook pipeline around tool/model loop |
| MCP | 4 | Model Context Protocol client |
| Plugins | 6 | Plugin loading + installer |
| Swarm | 10 | Multi-agent team delegation |
| Tasks | 6 | Background task management |
| Voice | 4 | Speech-to-text (streaming) |
| Coordinator | 3 | Coordinator mode + agent definitions |
| Memory | 7 | File-based memory (memdir) |
| Config | 3 | Settings + paths |
| Keybindings | 4 | Customizable keybindings |
| Vim | 2 | Vim mode |
| Output styles | 2 | Customizable output |

### Slash commands — 57 commands

`agents`, `branch`, `bridge`, `clear`, `commit`, `compact`, `config`,
`context`, `copy`, `cost`, `diff`, `doctor`, `effort`, `exit`, `export`,
`fast`, `feedback`, `files`, `help`, `hooks`, `init`, `issue`,
`keybindings`, `login`, `logout`, `mcp`, `memory`, `model`, `onboarding`,
`output-style`, `passes`, `permissions`, `plan`, `plugin`, `pr_comments`,
`privacy-settings`, `rate-limit-options`, `release-notes`, `reload-plugins`,
`resume`, `rewind`, `session`, `share`, `skills`, **`soul`** ✨, `stats`,
`status`, `summary`, `tag`, `tasks`, `theme`, `upgrade`, `usage`,
`version`, `vim`, `voice`.

### Tests — 466 tests

- **458 passing** (was ~288 at session start — **+170 tests**)
- 5 skipped, 1 xfailed
- 2 pre-existing failures (network-dependent + React UI)
- 0 regressions from this session's work

### Frontend — React/Ink TUI

13 components in `frontend/terminal/src/components/`: CommandPicker,
Composer, ConversationView, Footer, ModalHost, PromptInput, SelectModal,
SidePanel, Spinner, StatusBar, ToolCallDisplay, TranscriptPane,
WelcomeBanner.

### NIA's unique strengths (vs Hermes)

1. **Jarvis personality is first-class** — `Personality` class with moods + SOUL.md identity
2. **Voice layer (STT + TTS)** — `nia_voice` + `speak` (KittenTTS Jasper voice)
3. **Explicit ReAct loop** — Plan → Act → Reflect with structured `ReasoningStep`
4. **LSP tool** — Python code intelligence via AST
5. **Simpler architecture** — 272 files vs Hermes's 6,126

---

### Phase 5 - Always-Ready Infrastructure (2026-06-03)

Session persistence allows NIA to save and restore conversations across restarts, making it a true always-ready partner.

#### Added
- `src/niaharness/tools/nia_session_tool.py` - `nia_session` tool: save, restore, list, new sessions
- Sessions stored in `~/.nia/sessions/` as JSON files
- `register_nia_tools()` now also wires the engine into the session tool

#### Changed
- `src/niaharness/tools/__init__.py` - NiaSessionTool registered, `register_nia_tools()` accepts engine parameter
- `src/agents/nia/nia.py` - Passes engine to `register_nia_tools()` for session persistence

### Phase 4 - Voice and MCP Connected (2026-06-03)

NIA now connects to niaharness's MCP (Model Context Protocol) system for remote tool access and voice integration for speech-to-text.

#### Added
- `src/niaharness/tools/nia_voice_tool.py` - `nia_voice` tool: transcribe audio, check capabilities, extract keyterms
- MCP auto-connection in `_build_engine()` — loads MCP server configs, creates McpClientManager, connects all servers
- MCP tools automatically available to brain via registry

#### Changed
- `src/agents/nia/nia.py` - `_build_engine()` now creates MCP manager and passes to tool registry; shutdown closes MCP connections
- `src/niaharness/tools/__init__.py` - NiaVoiceTool registered in default tool registry

### Phase 3 - Swarm Delegation (2026-06-03)

NIA's custom Coordinator and Dispatcher are now dead code — the QueryEngine handles all orchestration. Multi-agent execution is delegated to niaharness's swarm system via the `agent`, `team_create`, `team_delete`, and `send_message` tools already in the registry.

#### Changed
- `src/agents/nia/orchestration/__init__.py` - Updated docstring to document swarm delegation
- Removed dead imports of Coordinator and Dispatcher from NIA (already removed in Phase 1)

### Phase 2 - NIA Tools Exposed (2026-06-03)

NIA's unique features (memory and context awareness) are now available as tools that the brain can use during conversations.

#### Added
- `src/niaharness/tools/nia_memory_tool.py` - `nia_memory` tool: search, add_fact, add_preference, list_preferences, recent, stats
- `src/niaharness/tools/nia_context_tool.py` - `nia_context` tool: full, environment, time, session, set_user_name
- `register_nia_tools()` function in `tools/__init__.py` to wire memory/context instances into tools
- `tests/test_nia/test_tools.py` - 9 tests for both NIA tools

#### Changed
- `src/niaharness/tools/__init__.py` - NiaMemoryTool and NiaContextTool registered in default tool registry
- `src/agents/nia/nia.py` - Calls `register_nia_tools()` after engine build to wire instances
- `prompts/system.md` - Added nia_memory and nia_context to tool list

### Phase 1 - Unified Architecture (2026-06-03)

**Core Change**: NIA now delegates to niaharness's QueryEngine for the conversation loop, tool execution, permissions, hooks, cost tracking, and file state management. Previously, NIA reimplemented all of this from scratch, missing ~80% of niaharness's production features.

#### Added
- `src/agents/nia/providers/adapter.py` - Adapter bridging NIA's LLMProvider to niaharness's `SupportsStreamingMessages` protocol. Converts request/response formats between the two interfaces.
- `tests/test_nia/test_adapter.py` - 4 tests for the provider adapter (text response, tool calls, error handling, empty response)
- `tests/test_nia/test_nia.py` - 4 tests for the unified NIA class (instantiation, merged prompt, status, shutdown)

#### Changed
- **`src/agents/nia/nia.py`** - Rewritten to use `QueryEngine` instead of `Dispatcher` + `HarnessExecutorBridge`. The `process()` method now delegates to `QueryEngine.submit_message()` which handles the full conversation loop with permissions, hooks, cost tracking, and abort control. Removed `_execute_simple()` and `_execute_with_react()` methods (QueryEngine handles this now).
- **`prompts/system.md`** - Expanded tool list from 8 to 38+ tools organized by category (File Ops, Shell, Web, Code Intelligence, Agents, Teams, Scheduling, Workspace, MCP, Utility) with correct arg signatures.
- **`src/agents/nia/orchestration/dispatcher.py`** - Fixed `"file_write"` → `"write_file"` (tool name was wrong, causing "create" intent to fail)
- **`src/agents/nia/orchestration/bridge.py`** - Fixed docstring references from `"file_write"` to `"write_file"`
- **`src/niaharness/coordinator/agent_definitions.py`** - Fixed `disallowed_tools` lists: `"file_write"` → `"write_file"` (3 occurrences)
- **`src/niaharness/coordinator/coordinator_mode.py`** - Fixed worker tools list: `"file_write"` → `"write_file"`
- **`src/niaharness/ui/output.py`** - Added `"write_file"` to UI output formatting aliases

#### Fixed
- **`src/niaharness/engine/query.py:507`** - Fixed `auto_compact_if_needed()` call: removed invalid `api_client` and `system_prompt` kwargs, added missing `compact_state` return value unpacking (3-tuple instead of 2)

#### Architecture (before → after)
```
BEFORE: NIA Brain → Dispatcher → HarnessExecutorBridge → ToolRegistry (no permissions, no hooks, no cost)
AFTER:  NIA Brain → QueryEngine → ToolRegistry + Permissions + Hooks + Cost + FileState + Abort
```

NIA now gets for free:
- Permission checking (PermissionChecker)
- Pre/post tool hooks (HookExecutor)
- Cost tracking (CostTracker)
- File state cache (FileStateCache)
- Abort controller (AbortController)
- Auto-compaction
- Budget enforcement
- Tool failure loop detection
- Continuation nudge detection
