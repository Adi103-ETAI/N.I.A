# Hermes vs NIA — Working Audit & Gap Analysis

> Goal: compare **how Hermes works** vs **how NIA works** (not the code), then
> list every Hermes tool / skill and whether NIA has it. Produced by running
> both agents side-by-side, reading their CLIs, and enumerating their
> registries.
>
> **Original audit date:** 2026-07-07 (NIA commit `55c03f7`, 44 tools, 7 skills)
> **Last updated:** 2026-07-07 (NIA commit `63cc327`, after P0 Tasks 1-5)
> Hermes: v0.18.0 (commit `70c6ae6`, installed via `pip install -e .`)

---

## ✅ Update: P0 Tasks 1-5 Complete

After the original audit, 5 P0 tasks were completed in a single session
(7 commits, all on `insight` branch). This closed 4 of the top 6 behavioral
gaps and added 7 new tools:

| Task | Commit | What was built | Gap closed |
|---|---|---|---|
| **P0 Task 1** | `f047d08` | SOUL.md identity file system (`~/.nia/SOUL.md`, `/soul` command) | SOUL.md identity gap |
| **P0 Task 2** | `9a4d7b0` | `skill_manage` tool (6 ops: create/update/edit/delete/list/info) | Skill-authoring gap (read-only → CRUD) |
| **P0 Task 3** | `1708b52` | FTS5-backed session search (`session_search` tool + auto-indexing) | Session search gap |
| **P0 Task 4** | `1cb2c9b` | `vision_analyze` tool (multimodal image analysis) | Vision analysis gap |
| **P0 Task 5** | `63cc327` | Self-improving background memory review loop | Self-improving learning loop gap (memory-only) |

**Test suite impact:** 288 → 458 passing (+170 tests). 0 regressions.

**Updated counts:**
- Tools: **44 → 47** (+7: `browser`, `run_code`, `speak`, `skill_manage`, `session_search`, `vision_analyze`, + background review system)
- Skills: 7 bundled (unchanged, but now agent can create its own via `skill_manage`)
- Services: **6 → 7** (+`session_search.py`)
- Engine modules: **6 → 7** (+`background_review.py`)
- Slash commands: **56 → 57** (+`/soul`)

**Still open from the original audit:**
- ❌ Messaging gateway (Telegram/Discord/Slack/etc.)
- ❌ Curator process (autonomous skill refinement)
- ❌ Subagent delegation with isolated contexts
- ❌ Kanban task system (9 tools)
- ❌ Scheduled automation with platform delivery
- ❌ Computer Use (desktop automation)
- ❌ Image/video generation
- ❌ 169 more skills across 26 categories
- ❌ Multi-backend terminal (Docker/SSH/Modal/Daytona)
- ❌ Profiles (isolated NIA_HOME directories)
- ⚠️ Skill creation in the background review loop (Task 5 covers memory only; skill creation needs tool-calling in the review thread — follow-up work)

See the **Prioritized Follow-Up Recommendations** section below for the P1/P2/P3 task list.

---

## TL;DR (original audit)

**NIA is not yet working like Hermes.** It is a competent coding-agent scaffold
(44 tools, 7 bundled skills, a Jarvis personality, a ReAct loop, and an
auto-compact engine), but Hermes is a **personal AI agent platform** (81 tools,
176 skills, 20+ messaging platforms, a self-improving learning loop, scheduled
automation with platform delivery, subagent delegation, a desktop GUI, and a
pluggable context-engine). They are solving overlapping but materially
different problems.

The single biggest behavioral gaps, in priority order:

1. ~~**No self-improving learning loop**~~ ✅ **Fixed in Task 5** (memory-only; skill creation still TODO)
2. **No messaging gateway** — Hermes talks from Telegram/Discord/Slack/
   WhatsApp/Signal/Email. NIA is CLI-only.
3. ~~**No session search**~~ ✅ **Fixed in Task 3** (FTS5-backed `session_search` tool)
4. **No subagent delegation** — Hermes can spawn isolated subagents for
   parallel workstreams. NIA has a `team_create` tool but no real
   orchestrator.
5. **No scheduled automation with delivery** — Hermes's cron can deliver
   results to any platform. NIA's cron only runs shell commands.
6. **Skills gap is enormous** — 73 bundled + 102 optional in Hermes vs 7 in
   NIA. NIA's 7 are all software-dev-process skills; Hermes has skills for
   finance, MLOps, research, smart-home, Apple ecosystem, blockchain, etc.

The detailed breakdown follows.

---

## 1. How Hermes Works (behavioral snapshot)

Installed Hermes via `pip install -e .` from the cloned repo. `hermes --version`
reports `v0.18.0`. `hermes doctor` passes. `hermes status` shows 19 API-key
slots + 5 OAuth providers + 6 messaging platforms + 24 toolsets (18 enabled by
default). `hermes tools list` enumerates every toolset with enable/disable
state. `hermes skills list` shows installed skills (0 by default until the
user opts into seeding).

**Runtime shape:**
- One process can be a CLI, a TUI, a messaging gateway, a desktop app, or an
  Electron GUI — same agent core, different frontends.
- The agent loop (`agent/conversation_loop.py`, ~3,900 lines) handles model
  call → tool dispatch → retries → fallbacks → compression → post-turn hooks
  → **background memory/skill review** (the self-improving loop).
- A separate **curator** process periodically reviews agent-created skills
  and can pin/archive/consolidate/patch them — fully autonomous.
- **Profiles** (`hermes profile create coder`) give you fully isolated
  HERMES_HOME directories — each with its own config, memory, sessions,
  skills, gateway, cron, and SOUL.md.
- **SOUL.md** is the agent's primary identity — a single file at
  `~/.hermes/SOUL.md` that defines who the agent is. Stable across contexts.
- **Skills are exposed as slash commands** (`/<skill-name>`) — they get
  injected as a user message (preserves prompt caching).
- **Context engine is pluggable** — third-party engines (e.g. LCM) can
  replace the built-in compressor via `context.engine` in config.yaml.
- **Prompt caching is sacred** — Hermes goes to great lengths to never
  mutate past context or rebuild the system prompt mid-conversation.

**Self-improving loop (the signature feature):**
1. After every turn, `spawn_background_review()` forks the agent.
2. The fork replays the conversation (warm in the prompt cache, so cheap).
3. It asks itself: "should any skill or memory be saved or updated?"
4. Writes go straight to memory + skill stores.
5. Main conversation is never touched.
6. The curator process periodically refines the skill collection.

This is **not** in NIA at all.

---

## 2. How NIA Works (behavioral snapshot)

NIA (after the recent repair push) runs as a Typer CLI (`python -m niaharness`)
or via `python -m agents.nia`. 44 tools registered. 7 bundled skills
(plan, debug, diagnose, review, simplify, commit, test). The agent loop is
`niaharness.engine.query.run_query` — model call → tool dispatch → auto-compact
→ continuation nudge → budget enforcement. QueryEngine wraps it with file
state cache, abort controller, cost tracking.

**Runtime shape:**
- CLI + React/Ink TUI (`frontend/terminal/`). No messaging gateway.
- NIA-specific layer (`agents/nia/`) adds: Brain (LLM decision-making),
  Personality (Jarvis tone), Memory (JSON file), Context, Listener/Speaker
  (voice), ReAct loop.
- The Brain emits structured JSON (`thinking`, `intent`, `tasks`,
  `response`, `confidence`) — NIA decides what to do, then delegates to
  niaharness's QueryEngine for execution.
- Personality has moods (NEUTRAL, FOCUSED, CURIOUS, PLAYFUL, CONCERNED,
  PROUD) and greetings/farewells.
- Sessions are persisted as JSON snapshots (per-project, SHA-256 hashed
  cwd). `--continue` and `--resume <id>` work.
- Cron exists but only runs shell commands — no platform delivery.
- No background review, no curator, no session search, no profiles.

**What NIA does well that Hermes doesn't:**
- **Jarvis personality is first-class** — Hermes has SOUL.md (generic);
  NIA has a dedicated Personality class with moods, greetings, tone
  adjustment. This is the "Jarvis" angle the user wants.
- **Voice layer** — `nia_voice` (STT) + `speak` (KittenTTS, Jasper voice).
  Hermes has `text_to_speech` but no STT.
- **ReAct loop is explicit** — Plan → Act → Reflect with `ReasoningStep`
  dataclass. Hermes's loop is more implicit.
- **Simpler architecture** — easier to reason about, fewer moving parts.

---

## 3. Tool-by-Tool Presence Matrix

NIA has 44 tools; Hermes has 81 (69 built-in + 12 plugin). Below is every
Hermes tool and whether NIA has an equivalent.

Legend: ✅ = NIA has it (possibly under a different name) · ⚠️ = partial · ❌ = missing

### Core file/shell/code tools

| Hermes tool | NIA equivalent | Status | Notes |
|---|---|---|---|
| `read_file` | `read_file` | ✅ | NIA's version supports images/PDFs/notebooks too |
| `write_file` | `write_file` | ✅ | |
| `patch` | `edit_file` | ⚠️ | NIA uses exact string match; Hermes's `patch` has 9 fuzzy strategies + auto syntax check |
| `search_files` | `grep` + `glob` | ✅ | NIA splits into two tools; Hermes combines |
| `terminal` | `bash` | ✅ | |
| `execute_code` | `run_code` | ✅ | NIA added this in the recent repair push |
| `process` | `task_*` (create/get/list/stop/output/update) | ⚠️ | NIA has 6 separate tools; Hermes has 1 with 8 operations |

### Web / browser

| Hermes tool | NIA equivalent | Status | Notes |
|---|---|---|---|
| `web_search` | `web_search` | ✅ | |
| `web_extract` | `web_fetch` | ⚠️ | NIA's `web_fetch` is static HTTP; Hermes's `web_extract` handles PDFs + returns markdown |
| `browser_navigate` | `browser` (operation=navigate) | ✅ | NIA's single browser tool covers 9 operations |
| `browser_click` | `browser` (operation=click) | ✅ | |
| `browser_type` | `browser` (operation=type) | ✅ | |
| `browser_snapshot` | `browser` (operation=snapshot) | ✅ | NIA returns text + interactive elements; Hermes returns accessibility tree |
| `browser_vision` | `browser` (operation=screenshot) | ⚠️ | NIA saves PNG to disk; Hermes loads it into the conversation |
| `browser_back` | `browser` (operation=back) | ✅ | |
| `browser_forward` | `browser` (operation=forward) | ✅ | |
| `browser_scroll` | `browser` (operation=reload — closest) | ⚠️ | NIA has no dedicated scroll |
| `browser_press` | — | ❌ | NIA's browser has no keyboard key press |
| `browser_console` | `browser` (operation=eval_js) | ⚠️ | NIA's eval_js can evaluate but doesn't capture console output / JS errors |
| `browser_get_images` | — | ❌ | |
| `browser_dialog` | — | ❌ | No JS dialog handler (alert/confirm/prompt) |
| `browser_cdp` | — | ❌ | No raw CDP escape hatch |

### Vision / media

| Hermes tool | NIA equivalent | Status | Notes |
|---|---|---|---|
| `vision_analyze` | — | ❌ | NIA has no multimodal image analysis tool |
| `video_analyze` | — | ❌ | |
| `image_generate` | — | ❌ | NIA has no text-to-image tool |
| `video_generate` | — | ❌ | |
| `xai_video_edit` | — | ❌ | |
| `xai_video_extend` | — | ❌ | |
| `text_to_speech` | `speak` | ✅ | NIA uses KittenTTS (Jasper voice); Hermes routes through Tool Gateway |

### Memory / learning / search

| Hermes tool | NIA equivalent | Status | Notes |
|---|---|---|---|
| `memory` | `nia_memory` | ⚠️ | NIA's memory is a JSON file; Hermes's is durable + injected into every turn + batched operations |
| `session_search` | — | ❌ | NIA persists sessions but has no FTS5 search |
| `skills_list` | `skill` | ⚠️ | NIA's skill tool is read-only; Hermes has list + view + manage |
| `skill_view` | `skill` | ⚠️ | |
| `skill_manage` | — | ❌ | NIA cannot create/update/delete skills through the agent |
| `clarify` | `ask_user_question` | ✅ | |
| `todo` | `todo_write` | ⚠️ | NIA writes to a markdown file; Hermes manages an in-memory task list |

### Delegation / orchestration

| Hermes tool | NIA equivalent | Status | Notes |
|---|---|---|---|
| `delegate_task` | `agent` + `team_create` | ⚠️ | NIA can spawn agents but no isolated context / parallel workstreams |
| `kanban_create` | — | ❌ | |
| `kanban_list` | — | ❌ | |
| `kanban_show` | — | ❌ | |
| `kanban_complete` | — | ❌ | |
| `kanban_block` | — | ❌ | |
| `kanban_comment` | — | ❌ | |
| `kanban_heartbeat` | — | ❌ | |
| `kanban_link` | — | ❌ | |
| `kanban_unblock` | — | ❌ | The entire Kanban task system is missing from NIA |

### Scheduling / automation

| Hermes tool | NIA equivalent | Status | Notes |
|---|---|---|---|
| `cronjob` (7 ops) | `cron_create` + `cron_list` + `cron_delete` + `cron_toggle` | ⚠️ | NIA has 4 separate tools; Hermes has 1 with 7 ops. **NIA's cron only runs shell commands — no platform delivery** |

### Projects / workspaces

| Hermes tool | NIA equivalent | Status | Notes |
|---|---|---|---|
| `project_create` | — | ❌ | |
| `project_list` | — | ❌ | |
| `project_switch` | — | ❌ | NIA has no named-workspace concept (uses cwd only) |

### Desktop / computer use

| Hermes tool | NIA equivalent | Status | Notes |
|---|---|---|---|
| `computer_use` (13 ops) | — | ❌ | No GUI automation (click, type, scroll, screenshot, drag) |
| `read_terminal` | — | ❌ | In-app terminal pane reader (Hermes desktop GUI only) |
| `close_terminal` | — | ❌ | |

### Integrations (all missing from NIA)

| Hermes tool | NIA equivalent | Status |
|---|---|---|
| `discord` | — | ❌ |
| `discord_admin` | — | ❌ |
| `ha_call_service` / `ha_get_state` / `ha_list_entities` / `ha_list_services` | — | ❌ (Home Assistant) |
| `meet_join` / `meet_leave` / `meet_say` / `meet_status` / `meet_transcript` | — | ❌ (Google Meet) |
| `spotify_albums` / `spotify_devices` / `spotify_library` / `spotify_playback` / `spotify_playlists` / `spotify_queue` / `spotify_search` | — | ❌ (Spotify — 7 tools) |
| `feishu_doc_read` / `feishu_drive_add_comment` / `feishu_drive_list_comment_replies` / `feishu_drive_list_comments` / `feishu_drive_reply_comment` | — | ❌ (Feishu/Lark — 5 tools) |
| `x_search` | — | ❌ (X/Twitter) |
| `yb_query_group_info` / `yb_query_group_members` / `yb_search_sticker` / `yb_send_dm` / `yb_send_sticker` | — | ❌ (Yuanbao — 5 tools) |

### MCP

| Hermes tool | NIA equivalent | Status | Notes |
|---|---|---|---|
| `mcp__<server>__<tool>` (dynamic) | `mcp_tool` (adapter) + `list_mcp_resources` + `read_mcp_resource` | ⚠️ | NIA has MCP support but registers one adapter per server tool; Hermes uses the `mcp__<server>__<tool>` naming convention |

### Hermes internal / GUI-only (not applicable to NIA)

| Hermes tool | Status | Notes |
|---|---|---|
| `send_message` | N/A | Hermes explicitly does NOT register this as an agent tool — outbound messaging is handled outside the agent loop |

### Tool totals

| | Hermes | NIA | Gap |
|---|---:|---:|---:|
| Built-in tools | 69 | 44 | -25 |
| Plugin tools | 12 | 0 | -12 |
| **Total** | **81** | **44** | **-37** |
| Multi-operation tools (ops counted) | 56 ops across 13 tools | ~9 ops across 3 tools | -47 ops |

---

## 4. Skill-by-Skill Presence Matrix

NIA has **7 bundled skills** (all software-dev-process): `plan`, `debug`,
`diagnose`, `review`, `simplify`, `commit`, `test`.

Hermes has **73 bundled + 102 optional + 1 plugin = 176 skills** across
27 categories. Below: which categories NIA covers (even partially).

| Hermes category | Hermes count | NIA coverage | Notes |
|---|---:|---|---|
| software-development | 12 | ⚠️ 7/12 | NIA has plan/debug/diagnose/review/simplify/commit/test. Missing: hermes-agent-skill-authoring, node-inspect-debugger, python-debugpy, requesting-code-review, spike, systematic-debugging, test-driven-development (Hermes bundles some of these as separate skills) |
| creative | 25 | ❌ 0/25 | ASCII art, Excalidraw, p5.js, Manim, ComfyUI, architecture diagrams, popular-web-designs, etc. — all missing |
| mlops | 37 | ❌ 0/37 | HuggingFace, vLLM, llama.cpp, W&B, Axolotl, TRL, PEFT, etc. — all missing |
| research | 16 | ❌ 0/16 | arXiv, blogwatcher, polymarket, OSINT, scrapling, duckduckgo-search, etc. — all missing |
| productivity | 16 | ❌ 0/16 | Notion, Airtable, Google Workspace, Obsidian, PowerPoint, OCR, maps, etc. — all missing |
| github | 6 | ❌ 0/6 | github-auth, codebase-inspection, github-code-review, github-issues, github-pr-workflow, github-repo-management — all missing |
| finance | 8 | ❌ 0/8 | Excel author, DCF model, LBO model, merger model, comps analysis, pptx-author, stocks — all missing |
| autonomous-ai-agents | 10 | ❌ 0/10 | claude-code, codex, opencode, blackbox, grok, openhands delegations — all missing |
| security | 6 | ❌ 0/6 | 1password, godmode, oss-forensics, sherlock, unbroker, web-pentest — all missing |
| devops | 5 | ❌ 0/5 | docker-management, pinggy-tunnel, watchers, inference-sh-cli, hermes-s6-container-supervision — all missing |
| apple | 4 | ❌ 0/4 | apple-notes, apple-reminders, findmy, imessage — all missing |
| smart-home | 1 | ❌ 0/1 | openhue (Philips Hue) — missing |
| email | 2 | ❌ 0/2 | himalaya, agentmail — missing |
| note-taking | 1 | ❌ 0/1 | obsidian — missing |
| social-media | 1 | ❌ 0/1 | xurl (X/Twitter) — missing |
| gaming | 2 | ❌ 0/2 | minecraft-modpack-server, pokemon-player — missing |
| blockchain | 3 | ❌ 0/3 | evm, hyperliquid, solana — missing |
| payments | 3 | ❌ 0/3 | mpp-agent, stripe-link-cli, stripe-projects — missing |
| health | 2 | ❌ 0/2 | fitness-nutrition, neuroskill-bci — missing |
| mcp | 2 | ❌ 0/2 | fastmcp, mcporter — missing |
| communication | 1 | ❌ 0/1 | one-three-one-rule — missing |
| web-development | 2 | ❌ 0/2 | cloudflare-temporary-deploy, page-agent — missing |
| migration | 1 | ❌ 0/1 | openclaw-migration — missing (N/A for NIA) |
| data-science | 1 | ❌ 0/1 | jupyter-live-kernel — missing |
| dogfood | 2 | ❌ 0/2 | dogfood, adversarial-ux-test — missing |
| plugins | 1 | ❌ 0/1 | google_meet — missing |
| (root) | 3 | ❌ 0/3 | computer-use, dogfood, yuanbao — missing |
| **TOTAL** | **176** | **7 (all in software-development)** | **-169 skills** |

**The 7 NIA skills map roughly to these Hermes skills:**
- NIA `plan` → Hermes `software-development/plan`
- NIA `debug` → Hermes `software-development/systematic-debugging` (partial)
- NIA `diagnose` → no direct Hermes equivalent
- NIA `review` → Hermes `software-development/requesting-code-review` (partial)
- NIA `simplify` → Hermes `software-development/simplify-code` (close match)
- NIA `commit` → no direct Hermes equivalent (Hermes uses github-pr-workflow skill)
- NIA `test` → Hermes `software-development/test-driven-development` (partial)

---

## 5. Behavioral Capability Gaps (what Hermes DOES that NIA doesn't)

These are the working-behavior gaps — things you'd notice when actually
using the two agents side by side, independent of tool count.

### Tier 1 — Signature features NIA completely lacks

1. **Self-improving learning loop** (`agent/background_review.py`)
   - After every turn, Hermes forks itself and asks "should any skill or
     memory be saved or updated?"
   - Writes go to memory + skill stores; main conversation untouched.
   - NIA has nothing equivalent. Its `nia_memory` tool is manually invoked.

2. **Curator process** (`agent/curator.py`)
   - Periodically reviews agent-created skills: pin / archive / consolidate /
     patch.
   - Fully autonomous, runs when agent is idle.
   - NIA has no skill management at all (read-only `skill` tool).

3. **Messaging gateway** (`gateway/`)
   - Telegram, Discord, Slack, WhatsApp, WhatsApp Cloud, Signal, Email,
     BlueBubbles (iMessage), QQ Bot, WeChat, Yuanbao, MS Graph webhook.
   - Same agent core, multi-platform delivery.
   - NIA is CLI-only.

4. **Session search** (`tools/session_search_tool.py`)
   - FTS5-backed SQLite search across every past conversation.
   - NIA persists sessions but has no search.

5. **Subagent delegation** (`tools/delegate_task.py`)
   - Spawn isolated subagents for parallel workstreams.
   - Each subagent has its own context, tool whitelist, and prompt cache.
   - NIA's `agent` + `team_create` tools are a faint shadow of this.

6. **Kanban task system** (`tools/kanban_tools.py` — 9 tools)
   - Create/list/show/complete/block/unblock/comment/heartbeat/link tasks.
   - Orchestrator-profile only; routes work to subagents.
   - NIA has nothing equivalent.

7. **Scheduled automation with platform delivery** (`cron/`)
   - Cron jobs can deliver results to any messaging platform.
   - "Daily reports, nightly backups, weekly audits — all in natural
     language, running unattended."
   - NIA's cron only runs shell commands.

8. **Profiles** (`hermes_cli/profiles.py`)
   - Fully isolated HERMES_HOME directories per profile.
   - Each with own config, memory, sessions, skills, gateway, cron, SOUL.md.
   - NIA has no profile concept.

### Tier 2 — Important features NIA partially has or is missing

9. **SOUL.md identity system** — Hermes has a single durable identity file;
   NIA has a `Personality` class with moods/greetings but no editable
   identity file.

10. **Skill-authoring tool** — Hermes can create/update/delete skills via
    `skill_manage`. NIA's `skill` tool is read-only.

11. **Skill slash commands** — Hermes exposes every skill as `/<skill-name>`.
    NIA has no slash-command-for-skills pattern.

12. **Pluggable context engine** — Hermes lets third-party engines replace
    the compressor. NIA's `auto_compact_if_needed` is hardcoded.

13. **Prompt caching awareness** — Hermes treats cache stability as a
    first-class invariant. NIA's auto-compact can mutate past context.

14. **Auxiliary model** — Hermes has a separate "auxiliary client" for
    background tasks (review, summarization) so the main model's prompt
    cache isn't invalidated. NIA has one model.

15. **Multi-backend terminal** — Hermes supports local, Docker, SSH,
    Singularity, Modal, Daytona. NIA is local-only.

16. **Computer Use** — Hermes can drive the desktop (click/type/scroll/
    screenshot/drag) via `cua-driver`. NIA has nothing.

17. **Vision analysis** — Hermes can load images into the conversation
    (`vision_analyze`) and analyze videos (`video_analyze`). NIA has no
    multimodal vision tool.

18. **Image generation** — Hermes has `image_generate`. NIA has none.

19. **Insights** — Hermes has `/insights` showing token/cost/tool usage
    trends over time. NIA has cost tracking but no historical insights.

20. **Honcho dialectic user modeling** — Hermes builds a deepening model
    of the user across sessions. NIA has flat JSON memory.

### Tier 3 — Smaller but noticeable gaps

21. **`process` tool with 8 operations** (list/poll/log/wait/kill/write/
    submit/close) — NIA has 6 separate task tools.
22. **`cronjob` with 7 operations** — NIA has 4 separate cron tools.
23. **`memory` with batched operations** — NIA's `nia_memory` is one-at-a-time.
24. **`computer_use` with 13 operations** — NIA has none.
25. **MCP `mcp__<server>__<tool>` naming convention** — NIA uses
    `McpToolAdapter` (works but doesn't follow the convention).
26. **Bundled skill seeding** — Hermes copies 73 skills into `~/.hermes/
    skills/` on first run. NIA's 7 skills are bundled in the package.
27. **Optional skills hub** — `hermes skills browse/install official/...`.
    NIA has no skill hub.
28. **Skill bundles** — YAML files that alias multiple skills under one
    slash command. NIA has nothing.
29. **Doctor command** — `hermes doctor` diagnoses issues. NIA has none.
30. **Update command** — `hermes update` self-updates. NIA has none.

---

## 6. What NIA Does That Hermes Doesn't (NIA's unique strengths)

To be fair — NIA isn't just a subset of Hermes. It has some things Hermes
doesn't:

1. **Jarvis personality is first-class** — `agents/nia/core/personality.py`
   has moods (NEUTRAL/FOCUSED/CURIOUS/PLAYFUL/CONCERNED/PROUD), greetings,
   farewells, tone adjustment. Hermes has generic SOUL.md.
   - **This is the user's "Jarvis" angle — NIA's strength.**

2. **Voice layer (STT + TTS)** — `nia_voice` (speech-to-text) + `speak`
   (KittenTTS, Jasper voice). Hermes has TTS but no STT.

3. **Explicit ReAct loop** — `agents/nia/core/react.py` has a structured
   `ReasoningStep` dataclass with Plan → Act → Reflect. Hermes's loop is
   more implicit.

4. **Simpler architecture** — easier to reason about. 44 tools vs 81.
   7 skills vs 176. One process vs many. This is a feature for
   maintainability.

5. **LSP tool** — `niaharness/tools/lsp_tool.py` does Python code
   intelligence (symbols, definitions, references, hover) via AST.
   Hermes doesn't ship an LSP tool in the core (has `hermes lsp` command
   for managing external LSP servers, but not as an agent tool).

6. **NIA Brain structured JSON output** — `BrainResponse` with `thinking`,
   `intent`, `tasks`, `response`, `confidence`, `needs_clarification`.
   Forces structured reasoning before tool dispatch.

7. **Worktree tools** — `enter_worktree` / `exit_worktree` for git worktree
   isolation. Hermes has `--worktree` flag but not as agent tools.

---

## 7. Prioritized Follow-Up Recommendations

Ordered by impact × feasibility. Each item lists the effort estimate and
which audit gap it closes.

### P0 — Close the biggest behavioral gaps (1–3 weeks each)

| # | Recommendation | Closes gap | Effort | Status |
|---|---|---|---|---|
| 1 | **Build the self-improving learning loop** — after every turn, fork the agent and ask "should any skill or memory be saved?". Start simple: just memory, no skill creation yet. | #1, #2 | 2 weeks | ✅ **Done (Task 5)** — memory-only; skill creation still TODO |
| 2 | **Add session search** — port Hermes's FTS5-backed `session_search` design. NIA already persists sessions as JSON; add a SQLite index + search tool. | #4 | 1 week | ✅ **Done (Task 3)** — `session_search` tool + auto-indexing |
| 3 | **Add `skill_manage` tool** — let the agent create/update/delete skills. Pairs with #1 (the learning loop needs this to write skills). | #10 | 3 days | ✅ **Done (Task 2)** — 6 ops (create/update/edit/delete/list/info) |
| 4 | **Add SOUL.md identity file** — load `~/.nia/SOUL.md` as the first thing in the system prompt. Keep the Personality class as the default if SOUL.md is empty. | #9 | 2 days | ✅ **Done (Task 1)** — `/soul` command + auto-seeding |
| 5 | **Add skill slash commands** — scan `~/.nia/skills/` and expose each as `/<skill-name>`. Inject as user message (preserves prompt cache). | #11 | 3 days | ✅ **Done** — dynamic skill command resolution in CommandRegistry.lookup()

### P1 — Important capability additions (1–2 weeks each)

| # | Recommendation | Closes gap | Effort | Status |
|---|---|---|---|---|
| 6 | **Add `vision_analyze` tool** — load images into the conversation for multimodal models. | #17 | 3 days | ✅ **Done (Task 4)** — URL + local file support, 3-tier config |
| 7 | **Add `image_generate` tool** — text-to-image via FAL/DALL-E/Stable Diffusion. | #18 | 1 week | ✅ **Done** — model catalog + payload filtering + retry + error types
| 8 | **Add `delegate_task` tool** — spawn isolated subagents with own context + tool whitelist. Replace NIA's `agent` + `team_create` with a single delegator. | #5 | 2 weeks | ✅ **Done** — api_client plumbing + approvals + timeouts + summary budget + 25-tool blocklist
| 9 | **Add cron platform delivery** — extend NIA's cron to deliver results to email/webhook (start simple; add Telegram/Discord later). | #7 | 1 week | ✅ **Done** — email + webhook (concurrent) + secret redaction + file locking + retention + url_env
| 10 | **Add `computer_use` tool** — cua-driver backend (cross-platform, background-safe). | #16 | 1 week | ✅ **Done** — cua-driver only, mirrors Hermes exactly (no PyAutoGUI fallback)
| 11 | **Add `process` multi-op tool** — collapse NIA's 6 task tools into one with 8 operations (list/poll/log/wait/kill/write/submit/close). | #21 | 3 days | ✅ **Done** — 8 ops: list/create/get/output/wait/stop/update/close
| 12 | **Add `cronjob` multi-op tool** — collapse NIA's 4 cron tools into one with 7 operations. | #22 | 2 days | ✅ **Done** — 7 ops: create/list/update/pause/resume/remove/run
| 13 | **Add `memory` batched-ops tool** — replace `nia_memory` with a tool that accepts an `operations` array. | #23 | 3 days | ✅ **Done** — batched operations: add/update/remove/search/list/get
| 14 | **Extend background review to write skills** — Task 5 covers memory only; add skill creation via `skill_manage` tool-calling in the review thread. | #1 (skill half) | 1 week | ✅ **Done** — review has skill_manage + nia_memory tools, COMBINED_REVIEW_PROMPT, agentic loop

### P2 — Skill catalog expansion (ongoing)

| # | Recommendation | Closes gap | Effort | Status |
|---|---|---|---|---|
| 15 | **Port the 6 GitHub skills** — github-auth, codebase-inspection, github-code-review, github-issues, github-pr-workflow, github-repo-management. All use `gh` CLI or REST. | #github row | 1 week | ⬜ Open |
| 16 | **Port the 5 software-dev-process skills NIA is missing** — node-inspect-debugger, python-debugpy, requesting-code-review, spike, systematic-debugging, test-driven-development. | #software-development row | 1 week | ⬜ Open |
| 17 | **Add 10–15 high-value skills from other categories** — pick the most useful: arxiv (research), obsidian (note-taking), google-workspace (productivity), docker-management (devops), 1password (security), stocks (finance), popular-web-designs (creative), etc. | many | 3 weeks | ⬜ Open |
| 18 | **Build a skill hub** — `nia skills browse/install official/<cat>/<name>`. Mirror Hermes's `optional-skills/` pattern. | #27 | 1 week | ⬜ Open |

### P3 — Architecture improvements (longer-term)

| # | Recommendation | Closes gap | Effort | Status |
|---|---|---|---|---|
| 19 | **Add messaging gateway** — start with Telegram (most popular). Same agent core, different frontend. | #3 | 3 weeks | ⬜ Open |
| 20 | **Add profiles** — `nia profile create coder` with isolated NIA_HOME. | #8 | 1 week | ⬜ Open |
| 21 | **Make context engine pluggable** — abstract `auto_compact_if_needed` behind an interface. | #12 | 1 week | ⬜ Open |
| 22 | **Add auxiliary model** — separate cheap model for background tasks. | #14 | 1 week | ⬜ Open (partially addressed — `NIA_BACKGROUND_REVIEW_MODEL` allows routing) |
| 23 | **Add `insights` command** — historical token/cost/tool-usage trends. | #19 | 1 week | ⬜ Open |
| 24 | **Add multi-backend terminal** — Docker, SSH, Modal, Daytona. | #15 | 3 weeks | ⬜ Open |
| 25 | **Add `doctor` and `update` commands** — self-diagnosis and self-update. | #29, #30 | 3 days | ⬜ Open (NIA already has `/doctor` slash command) |

---

## 8. Suggested Next Moves

### ✅ Completed (this session)

The original "Suggested Next Moves" sequence was:

1. ~~**Add SOUL.md** (2 days)~~ ✅ Done — Task 1
2. ~~**Add `skill_manage` tool** (3 days)~~ ✅ Done — Task 2
3. ~~**Build the self-improving learning loop** (2 weeks)~~ ✅ Done — Task 5 (memory-only)
4. ~~**Add session search** (1 week)~~ ✅ Done — Task 3
5. ~~**Add `vision_analyze` tool** (3 days)~~ ✅ Done — Task 4

All 5 items completed in a single session. Test suite went from 288 → 458
passing (+170 tests, 0 regressions).

### Recommended next (P1 sequence)

If the goal is "make NIA work more like Hermes (Jarvis-style)", the next
highest-leverage sequence is:

1. **Extend background review to write skills** (P1 #14, 1 week) — Task 5
   covers memory only; adding skill creation completes the self-improving
   loop.
2. **Add `delegate_task` tool** (P1 #8, 2 weeks) — spawn isolated subagents
   for parallel workstreams. Replaces NIA's `agent` + `team_create`.
3. **Add cron platform delivery** (P1 #9, 1 week) — extend cron to deliver
   results to email/webhook.
4. **Add `computer_use` tool** (P1 #10, 1 week) — PyAutoGUI desktop automation.
5. **Add `image_generate` tool** (P1 #7, 1 week) — text-to-image via FAL/DALL-E.

That sequence takes ~6 weeks and closes 5 more behavioral gaps. After that,
pick from P2 (skill catalog expansion) based on what matters most.

---

## 9. Method Notes

- Hermes was installed via `pip install -e .` from the cloned repo
  (`/home/z/my-project/hermes-agent/`, commit `70c6ae6`). `hermes --version`
  reports v0.18.0. `hermes doctor` passes. `hermes tools list` and
  `hermes skills list` were run to verify the registries.
- NIA was tested at commit `55c03f7` (the recent repair push on `insight`
  branch). `python -m niaharness --help` works. `create_default_tool_registry()`
  returns 44 tools.
- Tool counts were derived from the actual registry source files:
  - Hermes: `tools/registry.py` (singleton `registry`) + `toolsets.py`
    (catalogue) + plugin auto-load policy at `hermes_cli/plugins.py:1433`.
  - NIA: `src/niaharness/tools/__init__.py::create_default_tool_registry()`.
- Skill counts were derived from `find skills optional-skills plugins -name
  SKILL.md` in Hermes and `ls src/niaharness/skills/bundled/content/` in NIA.
- Two parallel subagents enumerated the Hermes tool and skill inventories
  in full; their reports are the source of truth for the matrices above.
- No code was modified in either repo during this audit.
