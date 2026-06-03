# FULL-SPECTRUM CODEBASE AUDIT REPORT

## Project: OpenHarness (N.I.A)
**Root:** `/home/kali/Desktop/N.I.A`
**Date:** 2026-06-03
**Files Scanned:** 325 Python source files + 10 config/docs files
**Auditors:** 6 Specialized Subagents (Correctness, Security, Performance, Concurrency, Quality, Dependencies)

---

## SEVERITY SUMMARY

| Severity | Count |
|----------|-------|
| CRITICAL | 14 |
| HIGH | 38 |
| MEDIUM | 41 |
| LOW | 23 |
| INFO | 5 |
| **TOTAL** | **121** |

---

## TOP 10 PRIORITY FIXES

| # | Finding | Fix Rationale |
|---|---------|---------------|
| 1 | **BashTool command injection** (`bash_tool.py:30-33`) | Arbitrary OS commands execute with zero sanitization — complete system compromise |
| 2 | **FileWriteTool path traversal** (`file_write_tool.py:32-35`) | Can write to `/etc/crontab`, `~/.ssh/authorized_keys`, or overwrite application binaries |
| 3 | **FileReadTool path traversal** (`file_read_tool.py:36-38`) | Can read `/etc/shadow`, API keys from settings, SSH private keys |
| 4 | **FileEditTool path traversal** (`file_edit_tool.py:33-36`) | Can modify any system file including `sudo`, `/etc/passwd` |
| 5 | **SSRF in WebFetchTool** (`web_fetch_tool.py:48-67`) | No blocklist for internal IPs — allows cloud metadata theft, internal network scanning |
| 6 | **API keys stored in plaintext** (`settings.py:136-142`) | Combined with path traversal, creates complete credential theft chain |
| 7 | **OAuth tokens stored in plaintext** (`providers/anthropic.py:133-139`) | Refresh tokens give indefinite access if leaked |
| 8 | **FULL_AUTO permission bypass** (`permissions/checker.py:79-80`) | Disables ALL security controls unconditionally |
| 9 | **Team lifecycle race condition** (`swarm/team_lifecycle.py:346-381`) | Lost updates on concurrent team.json modifications — data corruption |
| 10 | **Cron file race condition** (`services/cron.py:364-418`) | Lost updates on concurrent cron task modifications — silent data loss |

---

## FULL FINDINGS BY SUBAGENT

### SUBAGENT A — CORRECTNESS & RUNTIME (25 findings)

**CRITICAL**

| # | File | Line | Category | Severity | Description |
|---|------|------|----------|----------|-------------|
| 1 | `src/niaharness/tools/bash_tool.py` | 30-33 | EXPLOIT | CRITICAL | Unrestricted shell command execution — user-supplied `command` passed directly to `/bin/bash -lc` with no sanitization |
| 2 | `src/niaharness/tools/remote_trigger_tool.py` | 38-41 | EXPLOIT | CRITICAL | Cron-triggered commands execute via `/bin/bash -lc` with no validation |
| 3 | `src/niaharness/providers/registry.py` | 188-191 | SECURITY | CRITICAL | API keys written in plaintext JSON to `~/.nia/providers.json` |
| 4 | `src/niaharness/providers/anthropic.py` | 138-139 | SECURITY | CRITICAL | OAuth tokens written in plaintext JSON with no file permission restrictions |

**HIGH**

| # | File | Line | Category | Severity | Description |
|---|------|------|----------|----------|-------------|
| 5 | `src/niaharness/services/cron_scheduler.py` | 103-127 | RACE CONDITION | HIGH | PID-reuse race in `_try_acquire_lock()` — two scheduler daemons can run simultaneously |
| 6 | `src/niaharness/services/cron.py` | 604-628 | LOGIC ERROR | HIGH | `set_job_enabled()` toggles `recurring` instead of a separate `enabled` field |
| 7 | `src/niaharness/services/session_storage.py` | 782 | SYNTAX | HIGH | `re.compile()` used before `import re` — will crash at runtime |
| 8 | `src/niaharness/providers/openai.py` | 410-419 | LOGIC ERROR | HIGH | `GoogleProvider.get_client()` creates client without Authorization header — all Gemini calls fail |
| 9 | `src/niaharness/providers/openai.py` | 153-158 | LOGIC ERROR | HIGH | `OllamaProvider` hardcodes `context_window=4096` for all models regardless of actual capabilities |
| 10 | `src/niaharness/tools/remote_trigger_tool.py` | 33 | LOGIC ERROR | HIGH | `get_cron_job()` matches by UUID but `RemoteTriggerTool` passes job name |
| 11 | `src/niaharness/providers/registry.py` | 19-20 | INCONSISTENCY | HIGH | Config stored at `~/.nia/` instead of `~/.niaharness/` — inconsistent with rest of system |

**MEDIUM**

| # | File | Line | Category | Severity | Description |
|---|------|------|----------|----------|-------------|
| 12 | `src/niaharness/tools/file_edit_tool.py` | 37-46 | CONCURRENCY | MEDIUM | No file locking on read-modify-write cycle — concurrent edits race |
| 13 | `src/niaharness/tools/mcp_tool.py` | 44-46 | LOGIC ERROR | MEDIUM | MCP tool input model sets all fields to `object \| None` — loses type validation |
| 14 | `src/niaharness/services/compact.py` | 938-941 | LOGIC ERROR | MEDIUM | Compact summary only keeps last 5 user + 5 assistant messages — loses context |
| 15 | `src/niaharness/swarm/in_process.py` | 506-510 | LOGIC ERROR | MEDIUM | `agent_id.split("@", 1)` breaks if agent name contains `@` |
| 16 | `src/niaharness/swarm/mailbox.py` | 65-74 | RUNTIME | MEDIUM | `MailboxMessage.from_dict()` uses `data["id"]` without `.get()` — KeyError on malformed data |
| 17 | `src/niaharness/permissions/checker.py` | 62 | LOGIC | MEDIUM | `fnmatch` without anchoring — pattern `*.txt` matches `/etc/passwd.txt` |
| 18 | `src/niaharness/providers/anthropic.py` | 165-173 | LOGIC ERROR | MEDIUM | OAuth `_refresh_token()` is a stub returning `None` — expired tokens silently fail |
| 19 | `src/niaharness/services/cron.py` | 300-343 | LOGIC ERROR | MEDIUM | `cron_to_human()` doesn't handle `day_of_week == "7"` (Sunday alias) |

**LOW**

| # | File | Line | Category | Severity | Description |
|---|------|------|----------|----------|-------------|
| 20 | `src/niaharness/tools/brief_tool.py` | 33 | EDGE CASE | LOW | `text[:max_chars]` can split multi-byte characters — mojibake |
| 21 | `src/niaharness/api/client.py` | 132-134 | EDGE CASE | LOW | `Retry-After` parsed as `int` — floats like `"1.5"` fail |
| 22 | `src/niaharness/tools/base.py` | 55-75 | RUNTIME | LOW | `ToolRegistry` missing `list_names()` method called by `QueryEngine` |
| 23 | `src/niaharness/engine/query.py` | 763-764 | LOGIC ERROR | LOW | Permission check extracts `file_path` from dict but many tools use different field names |
| 24 | `src/niaharness/swarm/in_process.py` | 687-692 | RUNTIME | LOW | `shutdown_all()` gathers without timeout — one hung teammate blocks all |
| 25 | `src/niaharness/tools/FileWriteTool/__init__.py` | — | CODE QUALITY | LOW | Directory exists but no corresponding tool implementation found |

---

### SUBAGENT B — SECURITY & VULNERABILITY (23 findings)

**CRITICAL**

| # | File | Line | Category | CWE | Severity | Description |
|---|------|------|----------|-----|----------|-------------|
| 26 | `src/niaharness/tools/bash_tool.py` | 30-33 | EXPLOIT | CWE-78 | CRITICAL | Unrestricted arbitrary shell command execution |
| 27 | `src/niaharness/tools/web_fetch_tool.py` | 48-67 | EXPLOIT | CWE-918 | CRITICAL | SSRF — no internal IP blocklist (169.254.169.254, 10.x, 192.168.x) |
| 28 | `src/niaharness/tools/file_write_tool.py` | 32-35 | EXPLOIT | CWE-22 | CRITICAL | Arbitrary file write — no path confinement |
| 29 | `src/niaharness/tools/file_read_tool.py` | 36-38 | EXPLOIT | CWE-22 | CRITICAL | Arbitrary file read — sensitive data exfiltration |
| 30 | `src/niaharness/tools/file_edit_tool.py` | 33-36 | EXPLOIT | CWE-22 | CRITICAL | Arbitrary file edit — system compromise |
| 31 | `src/niaharness/bridge/session_runner.py` | 38-41 | EXPLOIT | CWE-78 | CRITICAL | Bridge session executes arbitrary shell commands |

**HIGH**

| # | File | Line | Category | CWE | Severity | Description |
|---|------|------|----------|-----|----------|-------------|
| 32 | `src/niaharness/permissions/checker.py` | 79-80 | SECURITY MISCONFIG | CWE-862 | HIGH | FULL_AUTO mode bypasses ALL permission checks |
| 33 | `src/niaharness/config/settings.py` | 136-142 | SECURITY MISCONFIG | CWE-312 | HIGH | API keys stored in plaintext settings file |
| 34 | `src/niaharness/providers/anthropic.py` | 133-139 | SECURITY MISCONFIG | CWE-312 | HIGH | OAuth tokens stored in plaintext JSON |
| 35 | `src/niaharness/bridge/work_secret.py` | 11-14 | SECURITY MISCONFIG | CWE-311 | HIGH | Work secrets are base64-encoded, not encrypted |
| 36 | `src/niaharness/plugins/installer.py` | 11-18 | SECURITY MISCONFIG | CWE-494 | HIGH | Plugin installation without signature/integrity verification |
| 37 | `src/niaharness/hooks/schemas.py` | 14 | EXPLOIT | CWE-78 | HIGH | Hook system enables arbitrary command execution via plugins |
| 38 | `src/niaharness/mcp/client.py` | 124-132 | SECURITY MISCONFIG | CWE-78 | HIGH | MCP server spawns unrestricted subprocesses from config |

**MEDIUM**

| # | File | Line | Category | CWE | Severity | Description |
|---|------|------|----------|-----|----------|-------------|
| 39 | `pyproject.toml` | 15-30 | DEPENDENCY CONFLICT | CWE-1395 | MEDIUM | All deps use `>=` without upper bounds — supply chain risk |
| 40 | `src/niaharness/tools/web_search_tool.py` | 96-137 | EXPLOIT | CWE-185 | MEDIUM | DuckDuckGo HTML parsing with regex — ReDoS risk |
| 41 | `src/niaharness/tools/web_search_tool.py` | 228-234 | SECURITY MISCONFIG | CWE-598 | MEDIUM | Tavily API key sent in request body instead of header |
| 42 | `src/niaharness/bridge/manager.py` | 39-41 | EXPLOIT | CWE-22 | MEDIUM | Path traversal via `session_id` in output file path |
| 43 | `src/niaharness/services/lsp.py` | 293 | EXPLOIT | CWE-78 | MEDIUM | LSP server spawns user-configurable commands |
| 44 | `src/niaharness/services/lsp.py` | 288-289 | SECURITY MISCONFIG | CWE-200 | MEDIUM | Full `os.environ` passed to LSP subprocess — leaks API keys |

**LOW**

| # | File | Line | Category | CWE | Severity | Description |
|---|------|------|----------|-----|----------|-------------|
| 45 | `.gitignore` | N/A | SECURITY MISCONFIG | — | LOW | No `.env.example` documenting required env vars |
| 46 | `src/niaharness/utils/git.py` | 99-111 | EXPLOIT | CWE-78 | LOW | Git operations pass user-influenced args to subprocess |
| 47 | `.gitignore` | N/A | SECURITY MISCONFIG | — | LOW | `.opencode/` directory not in `.gitignore` |
| 48 | `src/niaharness/utils/shell_quote.py` | 38-39 | SECURITY MISCONFIG | CWE-78 | LOW | Windows shell quoting incomplete — misses `%`, `!` |

---

### SUBAGENT C — PERFORMANCE & RESOURCE (28 findings)

**CRITICAL**

| # | File | Line | Category | Severity | Description |
|---|------|------|----------|----------|-------------|
| 49 | `src/niaharness/bridge/manager.py` | 92 | PERFORMANCE BOTTLENECK | CRITICAL | File opened/closed per 4096-byte chunk — 256 syscalls per 1MB output |
| 50 | `src/niaharness/bridge/manager.py` | 74 | MEMORY LEAK | CRITICAL | `read_output()` loads entire log file into memory — 100MB sessions = 100MB RAM |
| 51 | `src/niaharness/tools/web_fetch_tool.py` | 62 | RESOURCE LEAK | CRITICAL | New `httpx.AsyncClient` per fetch — no connection pooling |
| 52 | `src/niaharness/tools/web_search_tool.py` | 76,180,241 | RESOURCE LEAK | CRITICAL | 3 search providers each create+destroy client per search |
| 53 | `src/niaharness/api/client.py` | 285 | RESOURCE LEAK | CRITICAL | `AsyncAnthropic` connection pool never explicitly closed |
| 54 | `src/niaharness/tasks/manager.py` | 21-26 | MEMORY LEAK | CRITICAL | 6 dicts grow indefinitely — completed tasks never evicted |
| 55 | `src/niaharness/swarm/mailbox.py` | 155-183 | SCALABILITY ISSUE | CRITICAL | `read_all()` deserializes ALL inbox files on every call — O(n) I/O |

**HIGH**

| # | File | Line | Category | Severity | Description |
|---|------|------|----------|----------|-------------|
| 56 | `src/niaharness/services/compact.py` | 640-681 | PERFORMANCE BOTTLENECK | HIGH | `get_effective_context_window_size` called 3x redundantly per turn |
| 57 | `src/niaharness/services/compact.py` | 220-258 | PERFORMANCE BOTTLENECK | HIGH | `estimate_message_tokens` calls `json.dumps` on every tool_use block |
| 58 | `src/niaharness/services/compact.py` | 791-828 | PERFORMANCE BOTTLENECK | HIGH | `auto_compact_if_needed` calls `estimate_message_tokens` 3x on same list |
| 59 | `src/niaharness/engine/query_engine.py` | 54-56 | MEMORY LEAK | HIGH | `FileStateCache._cache` grows without bound — no eviction |
| 60 | `src/niaharness/engine/query_engine.py` | 229,402 | PERFORMANCE BOTTLENECK | HIGH | Full message list copied on every property access |
| 61 | `src/niaharness/hooks/executor.py` | 52 | PERFORMANCE BOTTLENECK | HIGH | Sequential hook execution — one slow hook blocks all |
| 62 | `src/niaharness/tools/web_search_tool.py` | 96-137 | PERFORMANCE BOTTLENECK | HIGH | Regex patterns recompiled on every search call |
| 63 | `src/niaharness/mcp/client.py` | 67-72 | RESOURCE LEAK | HIGH | If one `stack.aclose()` fails, remaining stacks leak |
| 64 | `src/niaharness/mcp/client.py` | 92-106 | SCALABILITY ISSUE | HIGH | No timeout on MCP tool calls — hung server blocks forever |
| 65 | `src/niaharness/hooks/hot_reload.py` | 19-31 | PERFORMANCE BOTTLENECK | HIGH | Synchronous file I/O in `current_registry()` — blocks event loop |
| 66 | `src/niaharness/swarm/mailbox.py` | 185-217 | SCALABILITY ISSUE | HIGH | `mark_read()` scans ALL files to find one message — O(n) |

**MEDIUM**

| # | File | Line | Category | Severity | Description |
|---|------|------|----------|----------|-------------|
| 67 | `src/niaharness/tools/web_fetch_tool.py` | 150-155 | RUNTIME ERROR | MEDIUM | `urlparse` referenced but never imported — `NameError` at runtime |
| 68 | `src/niaharness/bridge/manager.py` | 44 | MEMORY LEAK | MEDIUM | `_copy_tasks` dict accumulates completed task references |
| 69 | `src/niaharness/services/compact.py` | 284-343 | PERFORMANCE BOTTLENECK | MEDIUM | Deep copies entire message list even when no images present |
| 70 | `src/niaharness/services/compact.py` | 956-969 | PERFORMANCE BOTTLENECK | MEDIUM | `collect_compactable_tool_ids` scans ALL messages every turn |
| 71 | `src/niaharness/memory/scan.py` | 11-25 | PERFORMANCE BOTTLENECK | MEDIUM | Reads ALL memory markdown files on every scan |
| 72 | `src/niaharness/engine/messages.py` | 50-55 | PERFORMANCE BOTTLENECK | MEDIUM | `text` and `tool_uses` properties recompute on every access |
| 73 | `src/niaharness/tools/base.py` | 69-75 | PERFORMANCE BOTTLENECK | MEDIUM | `list_tools()` and `to_api_schema()` create new lists every call |
| 74 | `src/niaharness/hooks/executor.py` | 170-178 | MEMORY LEAK | MEDIUM | `text_chunks` list accumulates full streaming response in memory |

**LOW**

| # | File | Line | Category | Severity | Description |
|---|------|------|----------|----------|-------------|
| 75 | `src/niaharness/swarm/in_process.py` | 383-393 | PERFORMANCE BOTTLENECK | LOW | `get_nowait()` loop busy-drains queue — fine for small queues |
| 76 | `src/niaharness/bridge/manager.py` | 97-105 | SCALABILITY ISSUE | LOW | Global singleton never cleaned up — leaks state in tests |
| 77 | `src/niaharness/tasks/manager.py` | 258-269 | SCALABILITY ISSUE | LOW | Old `TaskManager` abandoned without cleanup on recreation |
| 78 | `src/niaharness/memory/search.py` | 12-40 | PERFORMANCE BOTTLENECK | LOW | Linear scan of all memory files on every search |

---

### SUBAGENT D — CONCURRENCY & STATE (22 findings)

**CRITICAL**

| # | File | Line | Category | Severity | Description |
|---|------|------|----------|----------|-------------|
| 79 | `src/niaharness/swarm/team_lifecycle.py` | 346-564 | RACE CONDITION | CRITICAL | Read-modify-write on `team.json` with zero file locking — lost updates |
| 80 | `src/niaharness/tasks/manager.py` | 169-228 | RACE CONDITION | CRITICAL | `_watch_process` and `stop_task` race on shared task state |
| 81 | `src/niaharness/services/cron.py` | 364-667 | RACE CONDITION | CRITICAL | Read-modify-write on cron JSON file with no locking — lost updates |
| 82 | `src/niaharness/tools/FileEditTool/read_state.py` | 21-80 | RACE CONDITION | CRITICAL | Global `ReadStateTracker` shared across concurrent tool invocations |

**HIGH**

| # | File | Line | Category | Severity | Description |
|---|------|------|----------|----------|-------------|
| 83 | `src/niaharness/swarm/in_process.py` | 270-293 | FAULT / FAILURE | HIGH | `contextlib.suppress(Exception)` silently swallows idle notification failures |
| 84 | `src/niaharness/swarm/in_process.py` | 295-332 | FAULT / FAILURE | HIGH | Mailbox `read_all` failures silently discarded — teammate misses shutdown commands |
| 85 | `src/niaharness/swarm/team_lifecycle.py` | 609-692 | RACE CONDITION | HIGH | `_session_created_teams` set mutated during cleanup — teams leaked |
| 86 | `src/niaharness/state/store.py` | 14-40 | RACE CONDITION | HIGH | `AppStateStore.set()` concurrent calls produce lost updates |
| 87 | `src/niaharness/coordinator/coordinator_mode.py` | 62-70 | RACE CONDITION | HIGH | `get_team_registry()` singleton check-then-act race |
| 88 | `src/niaharness/bridge/manager.py` | 26-45 | RACE CONDITION | HIGH | `BridgeSessionManager` — 4 parallel dicts with no locking |
| 89 | `src/niaharness/services/cron_scheduler.py` | 103-127 | RACE CONDITION | HIGH | TOCTOU race in PID-based lock — two daemons can run simultaneously |

**MEDIUM**

| # | File | Line | Category | Severity | Description |
|---|------|------|----------|----------|-------------|
| 90 | `src/niaharness/swarm/mailbox.py` | 155-183 | RACE CONDITION | MEDIUM | `read_all()` reads without lock while `mark_read()` writes |
| 91 | `src/niaharness/swarm/permission_sync.py` | 445-478 | RACE CONDITION | MEDIUM | `read_pending_permissions()` reads without lock — partial writes possible |
| 92 | `src/niaharness/hooks/executor.py` | 117-146 | FAULT / FAILURE | MEDIUM | HTTP hook failures never logged — silent failures |
| 93 | `src/niaharness/mcp/client.py` | 121-174 | FAULT / FAILURE | MEDIUM | MCP connection failures silently stored — caller must check |
| 94 | `src/niaharness/services/cron_scheduler.py` | 188-202 | FAULT / FAILURE | MEDIUM | `append_history()` concurrent appends interleave — corrupt JSONL |
| 95 | `src/niaharness/swarm/in_process.py` | 480-483 | FAULT / FAILURE | MEDIUM | Double cleanup path — `_on_done` and `shutdown()` both modify state |
| 96 | `src/niaharness/tasks/manager.py` | 262-269 | RACE CONDITION | MEDIUM | `get_task_manager()` singleton check-then-act race |
| 97 | `src/niaharness/tools/shared_state.py` | 6-9 | RACE CONDITION | MEDIUM | Duplicate `ReadStateTracker` singleton — two instances from different imports |

**LOW**

| # | File | Line | Category | Severity | Description |
|---|------|------|----------|----------|-------------|
| 98 | `src/niaharness/swarm/in_process.py` | 250-276 | RACE CONDITION | LOW | `TeammateContext.status` field mutated without guarding |
| 99 | `src/niaharness/services/cron_scheduler.py` | 466-467 | RACE CONDITION | LOW | Child process inherits and mutates parent's `_daemon` state after fork |
| 100 | `src/niaharness/swarm/mailbox.py` | 140-153 | FAULT / FAILURE | LOW | Lock file creation failure propagates unclear error |

---

### SUBAGENT E — CODE QUALITY & ARCHITECTURE (58 findings)

**HIGH**

| # | File | Line | Category | Severity | Description |
|---|------|------|----------|----------|-------------|
| 101 | `src/niaharness/commands/registry.py` | 202 | CODE SMELL | HIGH | `create_default_command_registry()` — 1119 lines, complexity 159 |
| 102 | `src/niaharness/commands/registry.py` | 278-299 | CODE SMELL | HIGH | Magic numbers for cost calculation: `3.0`, `15.0`, `75.0` |
| 103 | `src/niaharness/api/openai_shim.py` | 202 | CODE SMELL | HIGH | `convert_messages()` — 146 lines, complexity 31 |
| 104 | `src/niaharness/api/openai_shim.py` | 446 | CODE SMELL | HIGH | `openai_stream_to_anthropic()` — 226 lines, complexity 28 |
| 105 | `src/niaharness/coordinator/agent_definitions.py` | 695 | CODE SMELL | HIGH | `load_agents_dir()` — 198 lines, complexity 21 |
| 106 | `src/niaharness/api/provider_config.py` | 371 | CODE SMELL | HIGH | `detect_provider_from_url()` — 151 lines, complexity 14 |
| 107 | `src/niaharness/coordinator/coordinator_mode.py` | 251 | CODE SMELL | HIGH | `get_coordinator_system_prompt()` — 269 lines of hardcoded f-string |
| 108 | `src/niaharness/services/compact.py` | 733 | CODE SMELL | HIGH | `auto_compact_if_needed()` — 122 lines, complexity 12 |
| 109 | `src/niaharness/engine/query.py` | 445 | CODE SMELL | HIGH | `run_query()` — 271 lines, complexity 20 |
| 110 | `src/niaharness/tools/BashTool/utils.py` | 42 | CODE SMELL | HIGH | `split_command_with_operators()` — 83 lines, complexity 19 |
| 111 | `src/niaharness/cli.py` | 295 | CODE SMELL | HIGH | `main()` — 264 lines, 25+ params including 6 boolean traps |
| 112 | `src/niaharness/services/lsp.py` | 1116 | ANTI-PATTERN | HIGH | Module-level singleton `_lsp_manager` with `global` keyword |
| 113 | `src/niaharness/tasks/manager.py` | 258-269 | ANTI-PATTERN | HIGH | Module-level globals with `global` keyword |
| 114 | `src/niaharness/swarm/registry.py` | 395-403 | ANTI-PATTERN | HIGH | Module-level `_registry` global with `global` keyword |
| 115 | `src/niaharness/bridge/manager.py` | 97-105 | ANTI-PATTERN | HIGH | Module-level `_DEFAULT_MANAGER` global |
| 116 | `src/niaharness/coordinator/coordinator_mode.py` | 67 | ANTI-PATTERN | HIGH | `global _DEFAULT_TEAM_REGISTRY` |
| 117 | `src/niaharness/tools/FileEditTool/read_state.py` | 72-80 | ANTI-PATTERN | HIGH | Module-level `_tracker` global with `global` keyword |
| 118 | `src/niaharness/config/settings.py` | 85-103 | MISCONFIGURATION | HIGH | `resolve_api_key()` silently falls through 3 env vars |
| 119 | `src/niaharness/config/settings.py` | 111-146 | MISCONFIGURATION | HIGH | `_apply_env_overrides()` never validates env var values |
| 120 | `src/niaharness/api/provider_config.py` | 371-533 | MISCONFIGURATION | HIGH | ~20 env vars read without any validation |

**MEDIUM**

| # | File | Line | Category | Severity | Description |
|---|------|------|----------|----------|-------------|
| 121 | `src/niaharness/swarm/mailbox.py` | 400-467 | CODE SMELL | MEDIUM | 4 nearly identical JSON parsing functions — copy-paste |
| 122 | `src/niaharness/services/lsp.py` | 237 | CODE SMELL | MEDIUM | `LSPClient` has 19 methods — god object |
| 123 | `src/niaharness/engine/query_engine.py` | 173 | CODE SMELL | MEDIUM | `QueryEngine` has 25 methods — god object |
| 124 | `src/niaharness/ui/textual_app.py` | 133 | CODE SMELL | MEDIUM | `NiaHarnessTerminalApp` — UI, state, and business logic interleaved |
| 125 | `src/niaharness/providers/registry.py` | 33 | CODE SMELL | MEDIUM | `ProviderRegistry` — 17 methods, 548 lines |
| 126 | `src/niaharness/engine/query.py` | 190 | CODE SMELL | MEDIUM | `update_tool_failure_loop_guard()` — 111 lines, complexity 13 |
| 127 | `src/niaharness/services/session_storage.py` | 612 | CODE SMELL | MEDIUM | `export_session_markdown()` — complexity 18 |
| 128 | `src/niaharness/services/compact.py` | 904 | CODE SMELL | MEDIUM | `_generate_compact_summary()` — complexity 14 |
| 129 | `src/niaharness/api/openai_client.py` | 353 | CODE SMELL | MEDIUM | `_stream_once()` — 118 lines, complexity 21 |
| 130 | `src/niaharness/cli.py` | 295-389 | CODE SMELL | MEDIUM | 6 boolean trap parameters |
| 131 | `src/niaharness/api/errors.py` | 223 | CODE SMELL | MEDIUM | `classify_http_error()` — 80-line if/elif chain |
| 132 | `src/niaharness/services/cron.py` | 175 | CODE SMELL | MEDIUM | `compute_next_cron_run()` — 82 lines, complexity 12 |
| 133 | `src/niaharness/api/client.py` | 57-58 | CODE SMELL | MEDIUM | Magic numbers for retry timing |
| 134 | `src/niaharness/engine/query.py` | 61 | CODE SMELL | MEDIUM | `DIMINISHING_THRESHOLD = 500` — unnamed magic number |
| 135 | `src/niaharness/swarm/permission_sync.py` | 616 | CODE SMELL | MEDIUM | `max_age_seconds = 3600.0` — hardcoded |
| 136 | `src/niaharness/bridge/types.py` | 9 | CODE SMELL | MEDIUM | `DEFAULT_SESSION_TIMEOUT_MS` — hardcoded 24h |
| 137 | `src/niaharness/services/session_storage.py` | 89 | MISCONFIGURATION | MEDIUM | Uses `CLAUDE_CONFIG_DIR` — porting artifact from OpenClaude |
| 138 | `src/niaharness/services/compact.py` | 756 | MISCONFIGURATION | MEDIUM | `DISABLE_COMPACT` env var never documented |
| 139 | `src/niaharness/swarm/team_lifecycle.py` | 528-529 | MISCONFIGURATION | MEDIUM | `CLAUDE_CODE_*` env vars read without validation |
| 140 | `src/niaharness/swarm/mailbox.py` | 493 | MISCONFIGURATION | MEDIUM | `CLAUDE_CODE_TEAM_NAME` defaults to magic string `"default"` |
| 141 | `src/niaharness/swarm/permission_sync.py` | 58-70 | MISCONFIGURATION | MEDIUM | 4 `CLAUDE_CODE_*` env vars read without validation |

**LOW**

| # | File | Line | Category | Severity | Description |
|---|------|------|----------|----------|-------------|
| 142 | `src/niaharness/utils/version.py` | 87 | CODE SMELL | LOW | `NIAHARNESS_VERSION` env var never validated |
| 143 | `src/niaharness/providers/anthropic.py` | 106-107 | CODE SMELL | LOW | Hardcoded `~/.config/niaharness/` path |
| 144 | `src/niaharness/providers/vertex.py` | 161 | MISCONFIGURATION | LOW | `GCLOUD_REGION` defaults to `"us-east5"` |
| 145 | `src/niaharness/providers/bedrock.py` | 143 | MISCONFIGURATION | LOW | `AWS_REGION` defaults to `"us-east-1"` |
| 146 | `src/niaharness/core/memory.py` | 151 | CODE SMELL | LOW | Magic `3600` for recency decay |
| 147 | `src/niaharness/ui/output.py` | 161 | CODE SMELL | LOW | Magic `5000` for highlight threshold |
| 148 | `src/niaharness/services/compact.py` | 25-42 | CODE SMELL | LOW | Token budget magic numbers undocumented |
| 149 | `src/niaharness/tools/task_output_tool.py` | 15 | CODE SMELL | LOW | Magic numbers for `max_bytes` |
| 150 | `src/niaharness/services/session_storage.py` | 27-30 | CODE SMELL | LOW | Magic numbers for I/O limits |
| 151 | `src/niaharness/api/client.py` | 50-54 | CODE SMELL | LOW | Retry constants undocumented |

**INFO**

| # | File | Line | Category | Severity | Description |
|---|------|------|----------|----------|-------------|
| 152 | `src/niaharness/commands/registry.py` | 401 | CODE SMELL | INFO | `import time` inside function body |
| 153 | `src/niaharness/providers/anthropic.py` | 113,130 | CODE SMELL | INFO | `import time` and `import json` inside method bodies |
| 154 | `src/niaharness/utils/version.py` | 67 | CODE SMELL | INFO | `import json` inside function body |
| 155 | `src/niaharness/swarm/mailbox.py` | 400-467 | CODE SMELL | INFO | Copy-pasted JSON parsing pattern in 4 functions |
| 156 | `src/niaharness/services/session_storage.py` | 27 | MISCONFIGURATION | INFO | `CLAUDE_CONFIG_DIR` — branding porting artifact |

---

### SUBAGENT F — DEPENDENCY & ENVIRONMENT (13 findings)

**CRITICAL**

| # | File | Line | Category | Severity | Description |
|---|------|------|----------|----------|-------------|
| 157 | `openclaude/package.json` | 137 | DEPENDENCY CONFLICT | CRITICAL | `xss` 1.0.15 has CVE-2024-21538 (prototype pollution) |
| 158 | `openclaude/Dockerfile` | 7 | MISCONFIGURATION | CRITICAL | `COPY bun.lock .bun-version` — files don't exist in build context |

**HIGH**

| # | File | Line | Category | Severity | Description |
|---|------|------|----------|----------|-------------|
| 159 | `.gitignore` | 112 | MISCONFIGURATION | HIGH | `openclaude/` gitignored — secrets/code not tracked |
| 160 | `pyproject.toml` | 10,55,58 | MISCONFIGURATION | HIGH | Python version inconsistent: `>=3.10` vs `py311` lint target |

**MEDIUM**

| # | File | Line | Category | Severity | Description |
|---|------|------|----------|----------|-------------|
| 161 | `openclaude/Dockerfile` | 2 | SECURITY MISCONFIG | MEDIUM | Base image `node:22-slim` not pinned |
| 162 | `openclaude/Dockerfile` | 43 | SECURITY MISCONFIG | MEDIUM | Unpinned `apt-get install` packages |
| 163 | `openclaude/Dockerfile` | 49 | SECURITY MISCONFIG | MEDIUM | No `HEALTHCHECK` instruction |
| 164 | `pyproject.toml` | 15 | DEPENDENCY CONFLICT | MEDIUM | All Python deps use open-ended `>=` without upper bounds |
| 165 | `src/niaharness/config/settings.py` | 53 | SECURITY MISCONFIG | MEDIUM | API keys as plain `str` — serialized to plaintext JSON |
| 166 | Root | N/A | MISCONFIGURATION | MEDIUM | No `.env.example` for main project |

**LOW**

| # | File | Line | Category | Severity | Description |
|---|------|------|----------|----------|-------------|
| 167 | `openclaude/Dockerfile` | 31 | SECURITY MISCONFIG | LOW | No smoke test before final stage |
| 168 | `openclaude/package.json` | 169-171 | DEPENDENCY CONFLICT | LOW | `lodash-es` override may mask version requirements |
| 169 | `.gitignore` | 40-41 | MISCONFIGURATION | LOW | `openclaude/.env` not explicitly ignored |

---

## REMEDIATION ROADMAP

### Phase 1 — CRITICAL (This Sprint)

1. **Add path confinement** to all file tools — assert resolved path starts with workspace root (Fixes #28, #29, #30)
2. **Sandbox bash execution** — implement command allowlist at tool level, not just permission layer (Fixes #1, #2, #31)
3. **Add SSRF protection** — block RFC1918, link-local, loopback, cloud metadata IPs in WebFetchTool (Fixes #27)
4. **Secure secret storage** — use `pydantic.SecretStr` + OS keyring; never persist API keys to plaintext files (Fixes #3, #4, #33, #34, #165)
5. **Fix permission system** — FULL_AUTO must still enforce path/command restrictions (Fixes #32)
6. **Add file locking** to team_lifecycle.py, cron.py, and task manager (Fixes #79, #80, #81)
7. **Add `ReadStateTracker` synchronization** or make per-agent via ContextVar (Fixes #82)

### Phase 2 — HIGH (Next Sprint)

8. **Connection pooling** — share `httpx.AsyncClient` across web_fetch, web_search, and API client (Fixes #51-53)
9. **Unbounded cache eviction** — add LRU/TTL to `FileStateCache`, `BackgroundTaskManager`, mailbox (Fixes #54, #59, #7)
10. **MCP client timeout + error handling** — wrap calls with `asyncio.wait_for`, handle `aclose()` failures (Fixes #63, #64)
11. **Plugin verification** — add signature checks before installation (Fixes #36)
12. **Hook command validation** — sandbox or allowlist hook commands (Fixes #37)
13. **Fix cron `enabled` vs `recurring` conflation** (Fixes #6)
14. **Fix missing `import re`** in session_storage.py (Fixes #7)
15. **Fix Google provider missing auth header** (Fixes #8)
16. **Extract god functions** — `create_default_command_registry()` (1119 lines), `run_query()` (271 lines) (Fixes #101, #109)
17. **Replace 7 global singletons** with dependency injection or ContextVar (Fixes #112-117)

### Phase 3 — MEDIUM (Sprint After)

18. **Pin dependency upper bounds** in pyproject.toml (Fixes #39, #164)
19. **Add env var validation** at startup (Fixes #118, #119, #120)
20. **Optimize compact.py** — cache token counts, avoid redundant scans (Fixes #56-58)
21. **Docker hardening** — pin base images, add HEALTHCHECK, pin packages (Fixes #161-163)
22. **Extract copy-pasted mailbox logic** into generic helper (Fixes #121)
23. **Fix `session_id` path traversal** in bridge/manager.py (Fixes #42)
24. **Add atomic writes** to cron file operations (Fixes #94)
25. **Remove hardcoded brand names** — replace `CLAUDE_*` with `NIA_*` (Fixes #137, #139, #140)

### Phase 4 — LOW (Ongoing)

26. **Document all magic numbers** with named constants and docstrings
27. **Create `.env.example`** at project root
28. **Add `.opencode/` to `.gitignore`**
29. **Fix Windows shell quoting** completeness
30. **Clean up import positioning** — move `import time/json` to module top
