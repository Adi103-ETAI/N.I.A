# NIA Deep Audit Report — Full Codebase Comparison

> **Date:** 2026-07-08
> **Auditors:** 3 parallel agents, each reading actual code (not just filenames)
> **NIA repo:** `/home/z/my-project/nia-insight/src/niaharness/`
> **Reference repo:** `/home/z/my-project/hermes-agent/`
> **Methodology:** Every file cited below was read at the code level. No inferences from filenames.

---

## Executive Summary

NIA has a functional agent with 54 tools, 20 providers, 34 skills, and a working conversation loop. However, compared to the reference codebase, NIA is at **~15-20% of the reference's scale and sophistication** across every major subsystem. The gaps are not incremental — they are architectural.

| Subsystem | NIA LOC | Reference LOC | Gap Severity |
|---|---|---|---|
| Skills infrastructure | 4,758 | 12,133 | HIGH (2 modules broken on import) |
| Tools | ~3,400 | ~20,900 | CRITICAL (no PTC, no multi-backend) |
| Engine | 2,168 | 15,559 | CRITICAL (no LLM compaction, no error recovery) |
| Providers | ~3,700 | ~8,760 | CRITICAL (no OAuth, no failover) |
| Permissions | 112 | 3,645+ | CRITICAL (no shell deobfuscation) |
| MCP | 325 | 6,976 | CRITICAL (stdio only, no OAuth) |
| Messaging gateway | 402 | 41,192 | CRITICAL (none) |
| Profiles | 0 | 3,249 | CRITICAL (none) |
| Context engine | 264 | 4,835 | HIGH (not pluggable) |
| Auxiliary model | 640 | 8,121 | HIGH (no separate client) |
| Insights | 44 | 946 | CRITICAL (none) |
| Session management | 770 | 6,322 | HIGH (JSON files, no SQLite) |
| Memory | 629 | 2,232 | HIGH (two parallel systems) |
| Cron | 1,200 | 7,401 | HIGH (shell-only, no agent execution) |
| Doctor + Update | 35 | 2,488 | HIGH (stubs only) |

---

## 1. SKILLS INFRASTRUCTURE

### CRITICAL: 2 modules broken on import

| Module | Imports work? | Used at runtime? |
|---|---|---|
| `tools/skills_hub.py` (568 lines) | ✅ | ✅ |
| `tools/skills_loader.py` (138 lines) | ✅ | ✅ |
| `tools/skill_tool.py` (33 lines) | ✅ | ✅ |
| `tools/skill_manage_tool.py` (479 lines) | ✅ | ✅ |
| `tools/skill_provenance.py` (78 lines) | ✅ | ❌ Dead code |
| `tools/skill_usage.py` (947 lines) | ❌ `ImportError: cannot import name 'get_nia_home' from 'niaharness.config'` | ❌ |
| `tools/skills_guard.py` (1086 lines) | ✅ | ❌ Never called by install_skill |
| `tools/skills_sync.py` (1182 lines) | ❌ `ImportError: cannot import name 'get_bundled_skills_dir' from 'niaharness.config'` | ❌ |

**Root causes:**
- `skill_usage.py` imports from `niaharness.config` (wrong module), `agent.skill_utils` (doesn't exist), `niaharness_cli.config` (doesn't exist)
- `skills_sync.py` imports from `niaharness.config` (wrong module), `agent.skill_utils` (doesn't exist), `utils.atomic_replace` (doesn't exist)
- Both modules are sed-replaced copies that reference non-existent NIA modules

### HIGH: skill_tool.py is 33 lines vs 1,662 lines

NIA's `skill` tool returns ONLY `skill.content`. The agent CANNOT read `references/`, `templates/`, `scripts/`, or `assets/` from any skill directory — including NIA's own bundled skills that ship these files (himalaya/references/, popular-web-designs/templates/, excalidraw/scripts/).

### HIGH: skill_manage_tool.py missing write_file/remove_file

NIA's bundled skills use directory-based layout with `references/`, `templates/`, `scripts/` subdirs, but `skill_manage` cannot create or manage these support files. The reference's `skill_manage` has `write_file` and `remove_file` actions.

### HIGH: No YAML frontmatter parsing

NIA's `_parse_skill_markdown` is a line-by-line regex parser. It cannot parse nested YAML (metadata, tags, platforms). The `metadata: niaharness: tags: [...]` block in NIA's own bundled skills is silently ignored.

### HIGH: No security scan between quarantine and install

NIA's `install_skill` does fetch → quarantine → install → lock. The reference does fetch → quarantine → **scan_skill** → should_allow_install → install → lock + audit. NIA's `skills_guard.scan_skill` exists but is never called.

### Missing features (from reference)
- 10 source adapters (GitHubSource, UrlSource, etc.) — only OptionalSkillSource exists
- TapsManager, audit log, index cache
- `_normalize_lock_install_path` / `_resolve_lock_install_path` (rmtree-escape fix)
- Stacked slash-skill invocations (`/skill-a /skill-b do X`)
- Platform/environment filtering
- Supporting-files hint in skill slash command invocation
- `_build_skill_message` (template vars, inline shell, config injection, setup notes)

---

## 2. TOOLS

### CRITICAL: No Programmatic Tool Calling (`execute_code`)

The reference's `execute_code` (1,910 lines) lets the model write Python scripts that call tools via RPC. NIA has no PTC — the model must do one tool call per inference turn. This is a 10x cost penalty for multi-step workflows.

### CRITICAL: bash tool is 70-378 lines vs 3,030 lines

NIA's bash tool is local-only `/bin/bash -c`. The reference has 7 backends (Local, Docker, Singularity, Modal, SSH, Daytona, managed Modal), PTY mode, sudo support, per-thread approval callbacks, `notify_on_complete`, `watch_patterns`.

### CRITICAL: browser tool is 411 lines vs 4,803 lines

NIA has 1 tool with 9 operations. The reference has 10 separate registered tools plus CDP override and dialog handler. Cloud browser (Browserbase, Modal, Camofox stealth), SSRF guards, secret exfiltration blocking, orphan reaper.

### HIGH: Missing tools entirely
- `web_extract` (extract page content as markdown)
- `video_analyze`, `video_generate`, `xai_video_edit`, `xai_video_extend`
- `kanban_*` (9 tools — task orchestration)
- `clarify` (ask clarifying questions)
- `project_list`, `project_create`, `project_switch`
- `skills_list`, `skill_view` (split from skill_manage)

---

## 3. ENGINE

### CRITICAL: No LLM-based context compaction

NIA uses pure text flattening (`summarize_messages` = `role: text` lines). The reference uses structured LLM summarization with iterative updates, session-scoped state, aux-model failure cooldowns, prompt-too-long vs max-tokens distinction, image-part stripping.

### CRITICAL: No error recovery beyond prompt-too-long

NIA has 2 recovery paths. The reference has 16 one-shot recovery guards:
- Per-provider OAuth refresh (Anthropic, Codex, Nous, Copilot, Vertex)
- Format/payload recovery (thinking sig, image shrink, multimodal tool content, llama.cpp grammar)
- Transport/rate-limit (429 retry, auth failover)
- Restart signals (compress, length-continue, rebuild messages)

### CRITICAL: No provider failover

NIA retries with backoff but never switches providers. The reference has a 2,384-line credential pool with priority entries, exhaustion tracking, retry-delay extraction, write-through to global state.

### MEDIUM: Background review gaps
- No digest mode for cold-cache cost savings
- No auto-deny for dangerous commands in review thread
- No skill read-before-write guard

---

## 4. PROVIDERS

### CRITICAL: Zero functional OAuth providers

NIA's Anthropic OAuth stub returns `None` on refresh. The reference has full OAuth for: Anthropic (PKCE), OpenAI Codex (Responses API), Nous Portal, Qwen, MiniMax, xAI, Copilot (GitHub), Kimi.

### CRITICAL: No credential pool / failover

NIA has no `agent._try_activate_fallback()`, no credential pool, no primary/failover chain. A single 401/429 from the only configured provider terminates the turn.

---

## 5. PERMISSIONS

### CRITICAL: 99-line stub vs 3,645+ lines

NIA's permission system has no:
- Shell deobfuscation (so `$(rm -rf /)` slips through)
- Hardline blocklist (never-approvable patterns)
- Smart approval via auxiliary LLM
- Per-session approval state
- Gateway async approval
- Plugin hooks
- Sudo stdin guard
- Home-prefix folding
- Container guards
- Cron approval mode

Even FULL_AUTO bypass has no audit log and doesn't block hardline patterns.

---

## 6. MCP

### CRITICAL: Only stdio transport, no OAuth, no reconnect

NIA has 1/3 transports (stdio only). No HTTP/SSE, no WebSocket. No OAuth manager. No reconnect/circuit breaker. No sampling/elicitation handlers. No loop isolation. No orphan reaping. No URL validation. No capability gating. No stderr capture. No env interpolation.

---

## 7. P3 ITEMS — Architecture

### #19 Messaging Gateway — CRITICAL (0 vs 41,192 lines)
NIA has zero gateway code. No platform adapters, no session management, no delivery routing.

### #20 Profiles — CRITICAL (0 vs 3,249 lines)
No profile concept at all. Everything in one shared `data_dir`.

### #21 Pluggable Context Engine — HIGH (264 vs 4,835 lines)
No ABC, no plugin discovery, no `context.engine` config. No `update_from_response(usage)`. No LLM-backed summarization.

### #22 Auxiliary Model — HIGH (640 vs 8,121 lines)
No separate auxiliary client. No `call_llm()` function. No provider fallback chain. No per-task overrides.

### #23 Insights — CRITICAL (44 vs 946 lines)
Zero insights code. No aggregation, no breakdowns, no cost estimation, no terminal formatter.

### #24 Multi-Backend Terminal — HIGH (849 vs 10,914 lines)
Local-only. No Docker, SSH, Modal, Daytona, Singularity. No BaseEnvironment ABC. No file sync.

### #25 Doctor + Update — HIGH (35 vs 2,488 lines)
`/doctor` is a config printer (20 lines). `/upgrade` is a help-text printer (15 lines). No security checks, no auto-fix, no actual update execution.

---

## 8. SESSION MANAGEMENT — HIGH

NIA uses per-project JSON files + separate FTS5 SQLite index (770 lines total). The reference uses a single SQLite DB with WAL mode, 45-column sessions table, 18-column messages table, FTS5 + trigram, session lineage, write contention tuning, malformed DB auto-repair, cross-process advisory locking (6,322 lines).

---

## 9. MEMORY SYSTEM — HIGH

NIA has TWO parallel memory systems that don't talk to each other:
1. `agents/nia/core/memory.py` — in-memory with JSON persistence
2. `memory/` directory — file-based MEMORY.md + topic files

Neither has: frozen snapshot for prefix-cache invariant, threat-pattern scanning, drift detection, character limits with consolidation cap, USER.md, external memory provider plugins, MemoryManager orchestration.

---

## 10. CRON SYSTEM — HIGH

NIA's cron is shell-command-only with email/webhook delivery (1,200 lines). The reference has agent execution (LLM with tools), 19+ delivery platforms, suggestion/blueprint UX, per-profile isolation, prompt-injection defense, lifecycle guards, pluggable trigger provider (7,401 lines).

---

## Dependency Graph

```
#20 Profiles ──────┐
                   ├─→ #19 Gateway (per-profile cron, per-profile gateway)
                   ├─→ #10 Cron (per-profile isolation)
                   └─→ #9 Memory (per-profile MEMORY.md/USER.md)

#22 Auxiliary Model ─→ #21 Pluggable Context Engine (LLM-backed summarization)
                   ─→ #23 Insights (cost estimation via aux pricing)

#8 Session DB ────┬─→ #23 Insights (SQL aggregation)
                   └─→ #19 Gateway (source tagging, session lineage)

#19 Gateway ──────┬─→ #10 Cron (delivery to chat platforms)
                  └─→ #25 Doctor (provider connectivity checks)
```

---

## Recommended Execution Order

| Priority | Item | Effort | Why first |
|---|---|---|---|
| 1 | Fix broken skill modules (skill_usage, skills_sync imports) | 1 day | Silent broken code |
| 2 | Wire skills_guard.scan_skill into install_skill | 1 day | Security gap |
| 3 | Add file_path param to skill tool (read references/templates/scripts) | 2 days | Agent can't access support files |
| 4 | Replace _parse_skill_markdown with yaml.safe_load | 1 day | Can't parse NIA's own frontmatter |
| 5 | Add write_file/remove_file to skill_manage | 2 days | Can't manage support files |
| 6 | Port execute_code (PTC) from reference | 2-3 weeks | Biggest capability win |
| 7 | Implement provider failover + credential pool | 2-3 weeks | Single-provider fragility |
| 8 | Port error recovery (16 one-shot guards) | 2 weeks | Turn-killing errors |
| 9 | LLM-based context compaction | 2 weeks | Token waste |
| 10 | Port permission system (shell deobfuscation, hardline blocklist) | 2 weeks | Security critical |
| 11 | MCP HTTP/SSE + OAuth + reconnect | 2-3 weeks | MCP usability |
| 12 | #20 Profiles | 3-4 weeks | Foundation for gateway, cron, memory |
| 13 | #8 Session DB (SQLite + WAL) | 3-4 weeks | Foundation for insights, gateway |
| 14 | #22 Auxiliary Model | 3-4 weeks | Foundation for context engine, insights |
| 15 | #19 Messaging Gateway (Telegram MVP) | 3-5 weeks | Platform delivery |
| 16 | #23 Insights | 2 weeks | Analytics |
| 17 | #21 Pluggable Context Engine | 2-3 weeks | Architecture cleanup |
| 18 | #9 Memory System consolidation | 2-3 weeks | Identity layer |
| 19 | #10 Cron (agent execution + delivery) | 3-4 weeks | Automation |
| 20 | #24 Multi-Backend Terminal (Docker MVP) | 2-3 weeks | Sandboxed execution |
| 21 | #25 Doctor + Update | 2-4 weeks | Operational hardening |

**Total estimated effort:** ~30-40 engineer-weeks (7-10 engineer-months) for full parity.

---

*Generated by 3 parallel audit agents reading actual source code across both repositories.*
