# N.I.A — Neural Intelligence Assistant

<p align="center">
  <strong>An AI partner inspired by J.A.R.V.I.S. — thinks, plans, and executes with calm authority.</strong>
</p>

<p align="center">
  NIA is the soul (personality, reasoning, memory). niaharness is the body (tools, execution, permissions, hooks). Together they form a single agent that can read your codebase, run shell commands, browse the web, analyze images, speak aloud, and learn from every conversation.
</p>

---

## Table of Contents

- [What NIA Is](#what-nia-is)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Interactive Mode](#interactive-mode)
  - [Print Mode (Non-Interactive)](#print-mode-non-interactive)
  - [Slash Commands](#slash-commands)
  - [SOUL.md — Your Agent's Identity](#soulmd--your-agents-identity)
- [Architecture](#architecture)
- [Tools (47+)](#tools-47)
- [Skills](#skills)
  - [GitHub Skill Hub](#github-skill-hub)
- [Providers (20+)](#providers-20)
- [Self-Improving Learning Loop](#self-improving-learning-loop)
- [Session Search & Insights](#session-search--insights)
- [Voice (STT + TTS)](#voice-stt--tts)
- [MCP Integration](#mcp-integration)
  - [MCP OAuth 2.1 + PKCE](#mcp-oauth-21--pkce)
- [Gateway — Chat Platform Integration](#gateway--chat-platform-integration)
- [Cron Jobs — Scheduled Agent Tasks](#cron-jobs--scheduled-agent-tasks)
- [Per-Session Approval Layer](#per-session-approval-layer)
- [Context Engine — Structured Summaries](#context-engine--structured-summaries)
- [Memory Manager — Provider Architecture](#memory-manager--provider-architecture)
- [Doctor & Update System](#doctor--update-system)
- [Profiles & Aliases](#profiles--aliases)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [License](#license)

---

## What NIA Is

NIA is two layers fused into one agent:

| Layer | Role | Lives in |
|---|---|---|
| **NIA (the soul)** | Personality, reasoning (ReAct loop), memory, context awareness | `src/agents/nia/` |
| **niaharness (the body)** | Tools, execution, permissions, hooks, MCP, swarm, cost tracking | `src/niaharness/` |

The **Brain** (NIA's LLM-powered decision maker) decides *what* to do. The **QueryEngine** (niaharness) handles *how* — tool execution, permission checks, retries, auto-compaction, cost tracking, and the self-improving background review loop.

### What makes NIA different

- **Jarvis personality is first-class** — a dedicated `Personality` class with moods (NEUTRAL, FOCUSED, CURIOUS, PLAYFUL, CONCERNED, PROUD), greetings, and tone adjustment. Plus a user-editable `SOUL.md` identity file.
- **Voice layer** — speech-to-text (`nia_voice`) + text-to-speech (`speak`, powered by KittenTTS with the Jarvis-like "Jasper" voice).
- **Self-improving** — after every turn with ≥3 tool calls, a background thread reviews the conversation and saves durable facts/preferences/patterns to memory. It can also create and patch skills autonomously.
- **Explicit ReAct loop** — Plan → Act → Reflect with structured `ReasoningStep` objects.
- **47+ tools** — files, shell, code execution, browser, vision, web search, skills, session search, cron, tasks, MCP, and more.
- **20+ LLM providers** — Anthropic, OpenAI, OpenRouter, Groq, Together, DeepSeek, Google, NVIDIA, Cerebras, Fireworks, Ollama, Bedrock, Vertex, Azure, Mistral, and more.
- **First-class Claude support** — prompt caching (~10× cost reduction), extended thinking with signature management, adaptive effort levels, OAuth token resolution (Claude Code / NIA-managed OAuth).
- **Gateway integration** — connect NIA to Telegram (and future Discord/Slack/Matrix) for always-available chat access. Persistent sessions survive restarts. PII-redacted context prompts.
- **Scheduled agent tasks** — cron jobs with full LLM agent access. "Summarize my GitHub issues every morning at 9am" actually works.
- **Per-session approval layer** — concurrent sessions (Telegram + CLI) each have isolated approval state. Smart-approve uses the auxiliary LLM to auto-approve low-risk commands.
- **Memory provider architecture** — pluggable memory backends (built-in JSON, future vector DB / Honcho) behind a unified `MemoryProvider` ABC. Streaming scrubber prevents memory context from leaking to the user.

---

## Quick Start

### Prerequisites

- Python 3.10+
- An API key for at least one provider (Anthropic, OpenAI, OpenRouter, etc.)

### Install

```bash
git clone https://github.com/Adi103-ETAI/N.I.A.git
cd N.I.A
git checkout insight
pip install -e .
```

### Set your API key

```bash
# Pick one (or more):
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
export OPENROUTER_API_KEY="sk-or-..."
```

### Run

```bash
# Interactive REPL (default)
python -m niaharness

# Or: one-shot print mode
python -m niaharness -p "Summarize this repository"

# Or: the NIA-specific entry point (with Jarvis personality)
python -m agents.nia
```

The first run seeds `~/.nia/SOUL.md` with a Jarvis-flavored default identity. Edit it to customize how NIA speaks.

---

## Configuration

NIA reads configuration from multiple sources, in priority order:

1. **CLI flags** (highest priority) — see `python -m niaharness --help`
2. **Environment variables** — `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `NIAHARNESS_MODEL`, `ANTHROPIC_BASE_URL`, etc.
3. **Settings file** — `~/.niaharness/settings.json`
4. **SOUL.md** — `~/.nia/SOUL.md` (identity, loaded as slot #1 in the system prompt)

### Key environment variables

| Variable | Purpose | Default |
|---|---|---|
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | API credentials | — |
| `ANTHROPIC_TOKEN` / `CLAUDE_CODE_OAUTH_TOKEN` | OAuth tokens (Claude Pro/Max) | — |
| `NIAHARNESS_MODEL` / `ANTHROPIC_MODEL` | Default model | `claude-sonnet-4-20250514` |
| `NIAHARNESS_CONFIG_DIR` | Config directory | `~/.niaharness/` |
| `NIAHARNESS_DATA_DIR` | Data directory (sessions, search index) | `~/.niaharness/data/` |
| `NIA_HOME` | NIA identity directory (SOUL.md lives here) | `~/.nia/` |
| `NIA_BACKGROUND_REVIEW` | Enable self-improving loop (`0` to disable) | disabled |
| `NIA_BACKGROUND_REVIEW_MODEL` | Model for background review | inherits main model |
| `NIA_VISION_API_KEY` / `NIA_VISION_MODEL` | Dedicated vision provider for `vision_analyze` | falls back to main |
| `NIA_YOLO_MODE` | Bypass all approval prompts (process-scoped) | off |
| `NIA_GATEWAY_SESSION` | Mark session as gateway (chat platform) | off |
| `NIA_CRON_SESSION` | Mark session as cron (auto-approve all tools) | off |
| `NIA_AUTO_CONTINUE_FRESHNESS` | Zombie session gate (seconds) | 3600 |

> **Note:** Legacy `OPENHARNESS_*` env vars are still accepted as aliases for `NIAHARNESS_*` for backward compatibility.

---

## Usage

### Interactive Mode

```bash
python -m niaharness
```

Starts a REPL. Type a message and press Enter. NIA will think, call tools as needed, and respond. Use slash commands (see below) for control.

### Print Mode (Non-Interactive)

```bash
# Single prompt, print the response, exit
python -m niaharness -p "What does this codebase do?"

# JSON output (for piping into other tools)
python -m niaharness -p "List all Python files" --output-format json

# Limit agentic turns
python -m niaharness -p "Fix the bug" --max-turns 5
```

### Common flags

```bash
# Continue the most recent session in this directory
python -m niaharness --continue

# Resume a specific session by ID
python -m niaharness --resume <session-id>

# Use a specific model
python -m niaharness --model sonnet
python -m niaharness --model claude-opus-4-20250514

# Use OpenRouter or other OpenAI-compatible providers
python -m niaharness --base-url https://openrouter.ai/api/v1 --api-format openai

# Run in plan mode (read-only, no writes)
python -m niaharness --permission-mode plan

# Restrict which tools are available
python -m niaharness --allowed-tools "read_file,grep,glob,bash"
```

### Slash Commands

NIA has **57+ slash commands**. The most useful:

| Command | Purpose |
|---|---|
| `/help` | Show all available commands |
| `/soul` | View/edit NIA's SOUL.md identity file |
| `/skills` | Browse available skills |
| `/model` | Show or switch the active model |
| `/status` | Show session usage (tokens, messages, effort) |
| `/context` | Show the full system prompt |
| `/compact` | Compress the conversation context |
| `/insights` | Show usage analytics + cost estimation (optional: days, --source, --gateway) |
| `/doctor` | Run diagnostics and auto-repair (optional: --fix, --ack \<id\>) |
| `/upgrade` | Check for and install updates (optional: --check, --no-backup) |
| `/clear` | Clear the conversation |
| `/exit` | Exit |

### SOUL.md — Your Agent's Identity

NIA's identity lives at `~/.nia/SOUL.md`. It's loaded as the **first slot** in the system prompt — defining who NIA is, ahead of the base tool instructions.

```bash
# View current identity
/soul

# Open in $EDITOR
/soul edit

# Reset to default
/soul reset

# Just print the path
/soul path
```

The default SOUL.md is Jarvis-flavored: professional, confident, slightly witty, dry humor when appropriate. Edit it freely — changes are picked up on the next message, no restart needed.

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    N.I.A (THE SOUL)                       │
│  ┌──────────┐  ┌──────────┐  ┌────────────────────┐      │
│  │  Brain   │  │  Memory  │  │    Personality     │      │
│  │ (reason) │  │ Manager  │  │ (JARVIS tone)      │      │
│  └────┬─────┘  └────┬─────┘  └────────────────────┘      │
│       │              │                                     │
│  ┌────▼──────────────▼──────────────────────────────┐     │
│  │         QueryEngine (THE BODY)                   │     │
│  │  • Conversation loop (ReAct)                     │     │
│  │  • 47+ tools                                     │     │
│  │  • Permission checks + approval layer            │     │
│  │  • Pre/post hooks + post_turn_hooks              │     │
│  │  • Cost tracking (real per-token)                │     │
│  │  • Auto-compaction (13-section structured)       │     │
│  │  • Anthropic transport (caching + thinking)      │     │
│  │  • Background review (self-improving)            │     │
│  │  • MCP integration (stdio + HTTP + OAuth)        │     │
│  │  • Gateway (Telegram + delivery routing)         │     │
│  │  • Cron agent (LLM execution + injection scan)   │     │
│  └──────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────┘
```

**Flow:** User message → Brain decides intent → QueryEngine executes tools → results fed back → loop until done → background review saves learnings to memory + creates skills.

---

## Tools (47+)

NIA ships with 47+ registered tools. Here's the full list by category:

| Category | Tools |
|---|---|
| **File ops** | `read_file`, `write_file`, `edit_file`, `notebook_edit`, `glob` |
| **Search** | `grep`, `lsp`, `tool_search` |
| **Shell & code** | `bash`, `run_code`, `execute_code` |
| **Web** | `web_search`, `web_fetch`, `browser` |
| **Vision & voice** | `vision_analyze`, `speak` |
| **Skills** | `skill`, `skill_manage`, `skills_list`, `skill_view` |
| **Memory & search** | `nia_memory`, `nia_context`, `session_search` |
| **Tasks & agents** | `task_create`, `task_get`, `task_list`, `task_output`, `task_stop`, `task_update`, `agent`, `send_message` |
| **Teams** | `team_create`, `team_delete` |
| **Cron** | `cron_create`, `cron_list`, `cron_delete`, `cron_toggle`, `remote_trigger` |
| **MCP** | `mcp_auth`, `list_mcp_resources`, `read_mcp_resource` |
| **Session** | `nia_session` |
| **Voice** | `nia_voice` |
| **Planning** | `todo_write`, `enter_plan_mode`, `exit_plan_mode`, `brief` |
| **Git worktrees** | `enter_worktree`, `exit_worktree` |
| **Meta** | `config`, `sleep`, `ask_user_question` |

### Highlight tools

- **`browser`** — Playwright-based interactive browser (navigate, click, type, snapshot, screenshot, eval_js)
- **`run_code`** — Sandboxed Python subprocess with timeout + output capture
- **`execute_code`** — Programmatic Tool Calling (PTC) — model writes Python that calls tools via RPC
- **`speak`** — Neural TTS via KittenTTS (Jasper voice, runs on CPU)
- **`vision_analyze`** — Multimodal image analysis (URLs or local files)
- **`skill_manage`** — Create/update/edit/delete skills at runtime
- **`session_search`** — FTS5-backed search over all past conversations (trigram tokenizer for CJK)

---

## Skills

NIA ships with **7 bundled skills** (all software-dev-process):

| Skill | Purpose |
|---|---|
| `plan` | Design an implementation plan before coding |
| `debug` | Diagnose and fix bugs systematically |
| `diagnose` | Diagnose why an agent run failed |
| `review` | Review code before committing |
| `simplify` | Clean up recent code changes |
| `commit` | Create clean, well-structured git commits |
| `test` | Write and run tests |

### Creating your own skills

Use the `skill_manage` tool (or the agent can create them autonomously via the background review loop):

```python
skill_manage(
    action="create",
    name="my-deploy-workflow",
    description="How I deploy this project to production",
    content="## Steps\n1. Run tests\n2. Build Docker image\n3. Push to registry\n4. Deploy",
)
```

Skills live at `~/.niaharness/skills/*.md` and are immediately available via the `skill` tool.

### GitHub Skill Hub

Install skills from GitHub repositories using the `skills_hub` tool:

```python
# Search for skills
skills_hub(action="search", query="git")

# Install a skill from GitHub
skills_hub(action="install", identifier="anthropics/skills/skill-name")
```

**Default taps** (curated skill repositories):
- `openai/skills` (curated + system)
- `anthropics/skills`
- `huggingface/skills`
- `NVIDIA/skills`
- `garrytan/gstack`

GitHub authentication supports 4 methods (in priority order):
1. `GITHUB_TOKEN` / `GH_TOKEN` env var (PAT — 5,000 req/hr)
2. `gh auth token` (gh CLI — picks up your existing login)
3. GitHub App JWT (for automated environments)
4. Anonymous (60 req/hr, public repos only)

---

## Providers (20+)

NIA supports 20+ LLM providers through a unified registry:

| Provider | Env var | Notes |
|---|---|---|
| Anthropic | `ANTHROPIC_API_KEY` / `ANTHROPIC_TOKEN` | Claude family — prompt caching, extended thinking, OAuth |
| OpenAI | `OPENAI_API_KEY` | GPT family |
| OpenRouter | `OPENROUTER_API_KEY` | 300+ models via one API |
| Groq | `GROQ_API_KEY` | Fast inference |
| Together | `TOGETHER_API_KEY` | Open-source models |
| DeepSeek | `DEEPSEEK_API_KEY` | DeepSeek models |
| Google | `GOOGLE_API_KEY` / `GEMINI_API_KEY` | Gemini |
| NVIDIA | `NVIDIA_API_KEY` | NVIDIA NIM |
| Cerebras | `CEREBRAS_API_KEY` | Fast inference |
| Fireworks | `FIREWORKS_API_KEY` | Open-source models |
| Ollama | (none — local) | Local models |
| Bedrock | AWS credentials | AWS Bedrock |
| Vertex | GCP credentials | Google Vertex AI |
| Azure OpenAI | `AZURE_OPENAI_API_KEY` | Azure OpenAI |
| Mistral | `MISTRAL_API_KEY` | Mistral AI |

Switch providers at runtime with `/model` or `--model`.

### Anthropic Transport Layer

NIA includes a full Anthropic transport layer ported from Hermes Agent:

- **Prompt caching** — `cache_control` markers on system prompt + last tool + last assistant block (~10× cost reduction on long sessions)
- **Extended thinking** — adaptive effort levels (max/xhigh/high/medium/low) with signature management across compaction boundaries
- **OAuth token resolution** — 5-step priority chain: `ANTHROPIC_TOKEN` → `CLAUDE_CODE_OAUTH_TOKEN` → NIA OAuth manager → credential pool → `ANTHROPIC_API_KEY`
- **Model capability detection** — knows which Claude models support adaptive thinking, xhigh effort, and sampling parameters
- **Message repair** — orphan-strip, role-merge, thinking-signature management (prevents HTTP 400 after compaction)

---

## Self-Improving Learning Loop

After every turn with ≥3 tool calls, NIA spawns a background daemon thread that:

1. Snapshots the conversation messages
2. Forks a restricted `QueryEngine` with only `memory` + `skill_manage` + `skill_view` tools
3. Sends one of 3 review prompts (memory-only, skill-only, or combined)
4. The forked engine executes any resulting `skill_manage` / `memory` tool calls
5. Surfaces a compact action summary to the user via callback

**Safety features:**
- **Skill provenance gate** — the fork can only patch skill files it has actually read via `skill_view` in the current review turn
- **Tool whitelist** — the fork can't call `bash`, `file_write`, `delegate_task`, etc.
- **Persistence isolation** — the fork does NOT write to the session DB
- **Thread-scoped silence** — only the review thread's stdout/stderr is silenced

### Configuration

```bash
# Enable the loop
export NIA_BACKGROUND_REVIEW=1

# Use a cheaper model for reviews
export NIA_BACKGROUND_REVIEW_MODEL=gpt-4o-mini
```

---

## Session Search & Insights

### Session Search

Every saved session is automatically indexed in a SQLite FTS5 database (trigram tokenizer for CJK support). Search past conversations with the `session_search` tool:

```python
# Find sessions that mentioned "flask"
session_search(query="flask")

# Browse recent sessions
session_search(action="browse")

# Rebuild the index from disk (admin)
session_search(action="rebuild")
```

### Insights (`/insights`)

Real per-token cost tracking with a 39-model pricing table:

```
/insights                    # Last 30 days, terminal format
/insights 7                  # Last 7 days
/insights --source telegram  # Filter by platform
/insights --gateway          # Markdown format for chat delivery
```

Shows: total sessions, messages, tool calls, input/output/cache tokens, estimated cost, per-model breakdown, per-platform breakdown, top tools, top skills, activity patterns (day-of-week, hour, busiest, streaks), and notable sessions.

---

## Voice (STT + TTS)

NIA has both speech-to-text and text-to-speech:

### Speech-to-text (`nia_voice`)

Transcribes audio files to text. Useful for voice-driven interactions.

### Text-to-speech (`speak`)

Generates speech from text using **KittenTTS** — an open-source neural TTS that runs on CPU (~56MB model, ~1.6s per synthesis after warmup).

```python
speak(text="Hello. NIA online and ready.", voice="Jasper")
```

**Voices:** Bella, **Jasper** (default — Jarvis-like male), Luna, Bruno, Rosie, Hugo, Kiki, Leo.

Install requirements:
```bash
pip install kittentts soundfile
# (also needs: misaki, onnxruntime, num2words, phonemizer)
```

---

## MCP Integration

NIA supports the Model Context Protocol for extending tools via external servers:

```bash
# Add an MCP server
python -m niaharness mcp add my-server '{"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]}'

# List configured servers
python -m niaharness mcp list

# Remove a server
python -m niaharness mcp remove my-server
```

MCP tools are automatically registered and appear as `mcp__<server>__<tool>` in the registry.

### MCP OAuth 2.1 + PKCE

NIA supports OAuth 2.1 + PKCE for hosted MCP servers (Notion, Slack, Linear, GitHub):

```yaml
# In your MCP config:
mcpServers:
  notion:
    type: http
    url: https://api.notion.com/mcp
    auth: oauth
    oauth:
      client_id: "your-client-id"
      scope: "read write"
```

**Features:**
- Browser-based PKCE flow with stdin paste fallback for SSH
- Token persistence at `~/.nia/mcp-tokens/` (profile-isolated, 0o600)
- Cross-process token reload (mtime watch — picks up `nia mcp login` refreshes)
- 401 dedup via in-flight futures (N concurrent calls → 1 refresh)
- `invalid_client` auto-heal (backup + re-register)
- `--ack <id>` for security advisory suppression

---

## Gateway — Chat Platform Integration

Connect NIA to chat platforms for always-available access:

```bash
# Start the Telegram gateway
export NIA_TELEGRAM_BOT_TOKEN="your-bot-token"
python -m niaharness gateway run
```

**Features:**
- **Persistent sessions** — conversations survive restarts, backed by SQLite session DB
- **PII redaction** — user/chat IDs SHA-256-hashed on safe platforms (Telegram, Signal, WhatsApp); Discord/Slack keep raw IDs for mentions
- **Session context prompt** — every gateway message tells the agent: platform, user, connected platforms, delivery options
- **Structured session keys** — `agent:<namespace>:<platform>:<chat_type>:<chat_id>[:<thread>][:<participant>]`, profile-isolated
- **Reset policies** — idle, daily, suspended (/stop), resume-pending zombie gate
- **Delivery routing** — cron job outputs route to "origin" (back to the chat that created the job), "local" (saved to disk), or specific platforms
- **Dead target registry** — deleted groups / blocked bots are short-circuited (saves flood-control quota), self-healing on successful send
- **Oversized output** — >4000 chars audit-saved to disk + truncated with footer
- **Silence-narration filter** — `(silent)`, `🔇`, bare `.` dropped before adapter (anti-loop guard)

---

## Cron Jobs — Scheduled Agent Tasks

Create cron jobs that run NIA's agent with full tool access on a schedule:

```python
# Shell command cron job (classic)
cron_create(
    name="nightly-backup",
    schedule="0 2 * * *",
    command="rsync -av /data /backup",
)

# LLM agent cron job (new!)
cron_create(
    name="morning-summary",
    schedule="0 9 * * *",
    prompt="Summarize my GitHub issues from the last 24 hours",
    delivery_targets=["telegram:123456"],
)
```

**Agent cron features:**
- Full LLM agent execution with restricted toolset (cronjob/messaging/clarify always disabled)
- Two-tier prompt injection scanner (strict on bare prompts, loose on skill/data-injected with defense-in-depth)
- `cron_hint` tells the agent: you're a scheduled job, your response is auto-delivered, use `[SILENT]` to suppress delivery
- Script + `context_from` support (pre-job data collection + upstream-job output chaining)
- Per-profile isolation (each profile gets its own cron jobs)
- Delivery via `DeliveryRouter` (origin/local/platform targets)

---

## Per-Session Approval Layer

Concurrent sessions (Telegram + CLI + cron) each have isolated approval state:

- **Per-session approval** — `approve_session(key, pattern)` scopes approval to one session via `ContextVars`
- **Permanent allowlist** — persisted to `~/.nia/approvals.json` (exact match or `fnmatch` glob)
- **Smart-approve** — auxiliary LLM auto-approves low-risk commands that fired the dangerous-pattern detector (strips shell comments first, fails open to "escalate")
- **Gateway async approval** — agent thread blocks on `threading.Event` while gateway sends approval request to the user via chat
- **MCP elicitation consent** — routes MCP server elicitation requests through the same gateway/CLI surfaces
- **Failure classification** — auth (401/403) → 60s + abort; network (connection drop) → 30s + abort; JSON decode → 30s; timeout/429 → 60s; no-provider → 300s

---

## Context Engine — Structured Summaries

When the conversation gets too long, NIA compacts it with a 13-section structured summary:

1. **Historical Task Snapshot** — the user's most recent unfulfilled input verbatim
2. **Goal** — what the user is trying to accomplish
3. **Constraints & Preferences** — coding style, constraints, decisions
4. **Completed Actions** — numbered list with tool + target + outcome
5. **Active State** — working directory, modified files, test status
6. **Historical In-Progress State** — what was being done when compaction fired
7. **Blocked** — errors and blockers with exact messages
8. **Key Decisions** — and WHY they were made
9. **Resolved Questions** — already answered
10. **Historical Pending User Asks** — STALE, reference only
11. **Relevant Files** — read, modified, or created
12. **Historical Remaining Work** — STALE, reference only
13. **Critical Context** — specific values, secrets redacted

**Features:**
- **Iterative updates** — previous summary carried forward so re-compaction preserves info
- **Temporal anchoring** — "email John" → "Sent email to John on 2026-07-10"
- **Secret redaction** — at 3 sites (input, LLM output, fallback)
- **Anti-thrash** — 2 consecutive <10% savings → skip until /new
- **Cooldown persistence** — survives restarts via session_db
- **SUMMARY_PREFIX** — handoff banner: "treat as reference only, respond to latest message"

---

## Memory Manager — Provider Architecture

Pluggable memory backends behind a unified `MemoryProvider` ABC:

- **Built-in JSON provider** — file-based memory (MEMORY.md + USER.md)
- **External providers** — future vector DB, Honcho, mem0 implement the same protocol
- **Threat-pattern scanner** — 36 patterns across 3 scopes (all/context/strict) + 18 invisible-unicode codepoints + NFKC normalization
- **StreamingContextScrubber** — stateful state machine strips `<memory-context>` blocks from streaming model output across chunk boundaries (prevents the agent from echoing its memory back to the user)
- **build_memory_context_block** — wraps prefetched memory with `[System note: ... authoritative reference data ...]` preamble
- **Background sync** — daemon `ThreadPoolExecutor(max_workers=1)` serializes provider writes

---

## Doctor & Update System

### Doctor (`/doctor`)

Self-diagnose + auto-repair:

```
/doctor           # Dry-run (report only)
/doctor --fix     # Auto-repair fixable issues
/doctor --ack <id>  # Acknowledge a security advisory
```

**12 check sections:** security advisories, MCP security, Python env, config files, session DB health (FTS rebuild + WAL checkpoint), directory structure, provider connectivity (9 providers in parallel), SSL CA bundle, external tools, summary.

**Auto-repaired by `--fix`:** missing dirs/files, FTS rebuild, WAL checkpoint (>50MB), config migration, `.env` creation (0o600).

### Update (`/upgrade`)

One-command updates:

```
/upgrade           # Check + install if available
/upgrade --check   # Check only
/upgrade --no-backup  # Skip pre-update backup
```

Detects install method (uv-tool / pipx / editable / docker / venv-pip / pip), creates ZIP backup of `~/.nia`, executes the appropriate upgrade command, runs config migration, verifies version change.

---

## Profiles & Aliases

Create multiple NIA personas with isolated data:

```bash
# Create a new profile (light clone of default)
nia profile create coder --clone

# Launch it by typing the alias
coder  # ↔ nia -p coder
```

**Features:**
- **Wrapper scripts** — `~/.local/bin/<name>` → `nia -p <profile>`
- **Alias validation** — regex guard against path traversal, reserved names, subcommand conflicts
- **`--clone`** — copies config.yaml + .env (0o600) + SOUL.md + MEMORY.md + USER.md
- **Profile isolation** — each profile gets its own cron jobs, sessions, skills, memories, MCP tokens, credentials

---

## Testing

```bash
# Run the full test suite (1338+ tests)
python -m pytest

# Run a specific test module
python -m pytest tests/test_providers/test_anthropic_transport.py

# Skip UI tests (requires npm install)
python -m pytest --ignore=tests/test_ui/test_textual_app.py
```

**Current status:** 1338 passing, 1 xfailed, 1 pre-existing failure (network-dependent).

---

## Project Structure

```
N.I.A/
├── src/
│   ├── agents/nia/              # NIA (the soul)
│   │   ├── core/                # brain, personality, memory, context, react
│   │   ├── providers/           # NIA-layer LLM providers
│   │   ├── communication/       # listener + speaker (voice)
│   │   ├── orchestration/       # coordinator, dispatcher, state
│   │   ├── ui/                  # backend_host, launcher, protocol
│   │   └── nia.py               # main NIA class
│   └── niaharness/              # niaharness (the body)
│       ├── api/                 # API client, credential pool, failover, usage
│       ├── cli/                 # doctor (auto-repair), update (install/backup/restart)
│       ├── commands/            # 57+ slash commands
│       ├── context_engine/      # pluggable context engines (simple + LLM with 13-section summary)
│       ├── engine/              # query_engine, query, messages, background_review, llm_compaction
│       ├── gateway/             # session persistence, delivery routing, Telegram adapter
│       ├── insights/            # real-token cost tracking, 39-model pricing, analytics
│       ├── mcp/                 # MCP client (stdio/HTTP/WS), OAuth 2.1 + PKCE, security
│       ├── memory/              # provider architecture, threat scanner, streaming scrubber
│       ├── permissions/         # checker, modes, shell hardening, approval layer
│       ├── profiles/            # per-profile isolation + aliases + wrapper scripts
│       ├── providers/           # 20+ LLM providers + Anthropic transport layer
│       ├── services/            # session_db (45-col), cron (agent + shell), compact, lsp
│       ├── tools/               # 47+ tools + skill hub sources (GitHub)
│       ├── prompts/             # system_prompt, soul, environment, context
│       ├── hooks/               # executor, loader, schemas
│       ├── plugins/             # plugin loader + installer
│       ├── swarm/               # multi-agent team delegation
│       ├── tasks/               # background task management
│       ├── voice/               # speech-to-text
│       ├── config/              # settings, paths
│       └── ui/                  # textual app, react launcher, backend
├── tests/                       # 1338+ tests
├── frontend/terminal/           # React/Ink TUI
├── prompts/                     # system.md
├── docs/                        # developer docs
└── pyproject.toml
```

---

## License

See the repository for license details.
