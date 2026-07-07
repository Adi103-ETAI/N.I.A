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
- [Tools (47)](#tools-47)
- [Skills](#skills)
- [Providers (15)](#providers-15)
- [Self-Improving Learning Loop](#self-improving-learning-loop)
- [Session Search](#session-search)
- [Voice (STT + TTS)](#voice-stt--tts)
- [MCP Integration](#mcp-integration)
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
- **Self-improving** — after every turn, a background thread reviews the conversation and saves durable facts/preferences/patterns to memory automatically.
- **Explicit ReAct loop** — Plan → Act → Reflect with structured `ReasoningStep` objects.
- **47 tools** — files, shell, code execution, browser, vision, web search, skills, session search, cron, tasks, MCP, and more.
- **15 LLM providers** — Anthropic, OpenAI, OpenRouter, Groq, Together, DeepSeek, Google, NVIDIA, Cerebras, Fireworks, Ollama, Bedrock, Vertex, Azure, Mistral.

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
| `NIAHARNESS_MODEL` / `ANTHROPIC_MODEL` | Default model | `claude-sonnet-4-20250514` |
| `NIAHARNESS_CONFIG_DIR` | Config directory | `~/.niaharness/` |
| `NIAHARNESS_DATA_DIR` | Data directory (sessions, search index) | `~/.niaharness/data/` |
| `NIA_HOME` | NIA identity directory (SOUL.md lives here) | `~/.nia/` |
| `NIA_BACKGROUND_REVIEW` | Enable self-improving loop (`0` to disable) | enabled |
| `NIA_BACKGROUND_REVIEW_MODEL` | Model for background review | inherits main model |
| `NIA_VISION_API_KEY` / `NIA_VISION_MODEL` | Dedicated vision provider for `vision_analyze` | falls back to main |

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

NIA has **57 slash commands**. The most useful:

| Command | Purpose |
|---|---|
| `/help` | Show all available commands |
| `/soul` | View/edit NIA's SOUL.md identity file |
| `/skills` | Browse available skills |
| `/model` | Show or switch the active model |
| `/status` | Show session usage (tokens, messages, effort) |
| `/context` | Show the full system prompt |
| `/compact` | Compress the conversation context |
| `/usage` | Show token/cost usage |
| `/doctor` | Show environment diagnostics |
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
┌──────────────────────────────────────────────────────┐
│                    N.I.A (THE SOUL)                   │
│  ┌──────────┐  ┌──────────┐  ┌────────────────────┐  │
│  │  Brain   │  │  Memory  │  │    Personality     │  │
│  │ (reason) │  │ (file)   │  │ (JARVIS tone)      │  │
│  └────┬─────┘  └──────────┘  └────────────────────┘  │
│       │                                               │
│  ┌────▼──────────────────────────────────────────┐    │
│  │         QueryEngine (THE BODY)                │    │
│  │  • Conversation loop (ReAct)                  │    │
│  │  • 47 tools                                   │    │
│  │  • Permission checks (4 modes)                │    │
│  │  • Pre/post hooks                             │    │
│  │  • Cost tracking                              │    │
│  │  • Auto-compaction                            │    │
│  │  • File state cache                           │    │
│  │  • Abort controller                           │    │
│  │  • MCP integration                            │    │
│  │  • Background review (self-improving)         │    │
│  └───────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────┘
```

**Flow:** User message → Brain decides intent → QueryEngine executes tools → results fed back → loop until done → background review saves learnings to memory.

---

## Tools (47)

NIA ships with 47 registered tools. Here's the full list by category:

| Category | Tools |
|---|---|
| **File ops** | `read_file`, `write_file`, `edit_file`, `notebook_edit`, `glob` |
| **Search** | `grep`, `lsp`, `tool_search` |
| **Shell & code** | `bash`, `run_code` |
| **Web** | `web_search`, `web_fetch`, `browser` |
| **Vision & voice** | `vision_analyze`, `speak` |
| **Skills** | `skill`, `skill_manage` |
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
- **`speak`** — Neural TTS via KittenTTS (Jasper voice, runs on CPU)
- **`vision_analyze`** — Multimodal image analysis (URLs or local files)
- **`skill_manage`** — Create/update/edit/delete skills at runtime
- **`session_search`** — FTS5-backed search over all past conversations

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

---

## Providers (15)

NIA supports 15 LLM providers through a unified registry:

| Provider | Env var | Notes |
|---|---|---|
| Anthropic | `ANTHROPIC_API_KEY` | Claude family |
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

---

## Self-Improving Learning Loop

After every turn, NIA spawns a background thread that:

1. Snapshots the last 20 messages
2. Makes a separate LLM call asking "is anything worth saving to memory?"
3. Parses the JSON response (`{"memories": [{"category": "preference|fact|pattern", ...}]}`)
4. Applies writes to NIA's `Memory` class (`add_preference`, `add_fact`, `add_pattern`)
5. Persists memory to disk

This runs **out-of-band** — it never blocks the main conversation and never breaks it if the review fails.

### Configuration

```bash
# Disable the loop entirely
export NIA_BACKGROUND_REVIEW=0

# Use a cheaper model for reviews
export NIA_BACKGROUND_REVIEW_MODEL=gpt-4o-mini

# Minimum seconds between reviews (anti-spam, default 30)
export NIA_BACKGROUND_REVIEW_INTERVAL=60
```

---

## Session Search

Every saved session is automatically indexed in a SQLite FTS5 database at `~/.niaharness/data/sessions.sqlite`. Search past conversations with the `session_search` tool:

```python
# Find sessions that mentioned "flask"
session_search(query="flask")

# Browse recent sessions
session_search(action="browse")

# Scroll inside a specific session
session_search(session_id="abc-123", around_message_idx=5)

# Rebuild the index from disk (admin)
session_search(action="rebuild")
```

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

Output is saved to `/home/z/my-project/download/` as WAV (24kHz) or MP3 (requires ffmpeg).

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

---

## Testing

```bash
# Run the full test suite (466 tests)
python -m pytest

# Run a specific test module
python -m pytest tests/test_tools/test_skill_manage_tool.py

# Run with verbose output
python -m pytest -v

# Skip UI tests (requires npm install)
python -m pytest --ignore=tests/test_ui/test_textual_app.py
```

**Current status:** 458 passing, 5 skipped, 1 xfailed, 2 pre-existing failures (network-dependent + React UI).

---

## Project Structure

```
N.I.A/
├── src/
│   ├── agents/nia/              # NIA (the soul)
│   │   ├── core/                # brain, personality, memory, context, react
│   │   ├── providers/           # 12 NIA-layer LLM providers
│   │   ├── communication/       # listener + speaker (voice)
│   │   ├── orchestration/       # coordinator, dispatcher, state
│   │   ├── ui/                  # backend_host, launcher, protocol
│   │   └── nia.py               # main NIA class
│   └── niaharness/              # niaharness (the body)
│       ├── engine/              # query_engine, query, messages, background_review
│       ├── tools/               # 47 tools (47 files)
│       ├── services/            # compact, session_storage, session_search, cron, lsp
│       ├── providers/           # 8 harness-layer LLM providers
│       ├── prompts/             # system_prompt, soul, environment, context
│       ├── permissions/         # checker, modes (4 permission modes)
│       ├── hooks/               # executor, loader, schemas
│       ├── mcp/                 # MCP client
│       ├── plugins/             # plugin loader + installer
│       ├── swarm/               # multi-agent team delegation
│       ├── tasks/               # background task management
│       ├── voice/               # speech-to-text
│       ├── coordinator/         # coordinator mode
│       ├── memory/              # file-based memory (memdir)
│       ├── config/              # settings, paths
│       ├── keybindings/         # customizable keybindings
│       ├── skills/              # 7 bundled skills + loader
│       ├── commands/            # 57 slash commands
│       ├── ui/                  # textual app, react launcher, backend
│       └── api/                 # API client, provider config, OpenAI shim
├── tests/                       # 466 tests
├── frontend/terminal/           # React/Ink TUI (13 components)
├── prompts/                     # system.md
├── docs/                        # developer docs
├── AUDIT_CHECKLIST.md           # original audit
├── FULL_AUDIT_REPORT.md         # full-spectrum audit
├── HERMES_VS_NIA_AUDIT.md       # Hermes vs NIA gap analysis
├── CHANGELOG.md                 # this project's changelog
└── pyproject.toml
```

---

## License

See the repository for license details.
