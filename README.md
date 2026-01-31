<div align="center">

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║    ███╗   ██╗   ██╗    █████╗                                             ║
║    ████╗  ██║   ██║   ██╔══██╗     Neural Intelligence Assistant          ║
║    ██╔██╗ ██║   ██║   ███████║     ─────────────────────────────          ║
║    ██║╚██╗██║   ██║   ██╔══██║     CLASSIFICATION: DIRECTOR_LEVEL_ACCESS  ║
║    ██║ ╚████║██╗██║██╗██║  ██║     DEVELOPER: SentArc Labs                ║
║    ╚═╝  ╚═══╝╚═╝╚═╝╚═╝╚═╝  ╚═╝     VERSION: 3.1.0 (Unknown Edition)       ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

### A Production-Ready, Multi-Modal AI System

**Voice • Vision • Tools • Reflexes**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-green.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 🎯 Overview

**N.I.A.** (Neural Intelligence Assistant) is a privacy-first, modular AI assistant designed for power users. It combines offline voice recognition, vision-based analysis, and automated desktop control into a unified system.

**v3.0.0 Release Highlights:**
- 🧹 **Codebase Hygiene** — Massive dead code removal & architectural cleanup
- 🔄 **TTS Optimization** — Persistent background loop for zero-latency speech
- 🛡️ **Error Hardening** — Comprehensive try/catch blocks & silent failure elimination
- 🎭 **Centralized Identity Engine** — Single source of truth for prompts
- 👁️ **Vision Hardening** — Robust initialization and explicit API key validation
- 🔥 **Multi-Provider LLM Hot-Swap** — Switch between NVIDIA, OpenAI, Groq, Ollama
- 🏛️ **ServiceContainer DI** — Clean dependency injection (Legacy container removed)
- 🛡️ **Diamond Security** — 3-Tier File System Protection with Path Traversal Locks
- 🔇 **Silent Core** — "Zero-Print" policy with Global Debug Mode (-d)
- 🧩 **Plugin Architecture** — Hot-loadable external tools support (ROOT/plugins/)
- 🏛️ **Unified Config** — Centralized configuration management (ROOT/config/)

---

## 🏗️ Architecture

N.I.A. uses a **LangGraph-based Supervisor Pattern** with four specialized units:

| Unit | Name     | Role                       | Technology                           |
|------|----------|--------------------------- |--------------------------------------|
| 🧠  | **NIA**  | Core Brain & Supervisor    | LangGraph + Multi-Provider LLM       |
| 🎤  | **NOLA** | Voice I/O (STT/TTS)        | Vosk (Offline) + Edge TTS            |
| 👁️  | **IRIS** | Vision & Screen Analysis   | Llama 3.2 Vision + mss               |
| 🛠️  | **TARA** | Tool Execution (v2.0)      | Unified Async Bridge + 50 Tools      |

### System Architecture (v3.0.0)

```
                    ┌──────────────────┐
                    │   USER INPUT     │
                    │  (Voice / Text)  │
                    └────────┬─────────┘
                             │
              ┌──────────────▼──────────────┐
              │     ⚡ REFLEX LAYER         │
              │  (Fuzzy Command Matching)   │
              └──────────────┬──────────────┘
                             │
     ┌───────────────────────▼───────────────────────┐
     │              🧠 SUPERVISOR (CEO)              │
     │  ┌─────────────────────────────────────────┐  │
     │  │         Protocol-based Routing          │  │
     │  │     + RoutingGatekeeper Validation      │  │
     │  │     + Identity Injection System         │  │
     │  └───────────────────┬─────────────────────┘  │
     │                      │                        │
     │  ┌───────────────────▼─────────────────────┐  │
     │  │   ⚡ CIRCUIT BREAKER (SafeLLM)          │  │
     │  │   └─> Retry Logic (Exponential)         │  │
     │  │   └─> Auto-Fallback on 429/503          │  │
     │  └───────────────────┬─────────────────────┘  │
     │                      │                        │
     │  ┌───────────────────▼─────────────────────┐  │
     │  │   🏭 MODEL FACTORY (ModelManager)       │  │
     │  │   ┌──────┬──────┬──────┬──────┐         │  │
     │  │   │NVIDIA│OpenAI│ Groq │Ollama│         │  │
     │  │   └──────┴──────┴──────┴──────┘         │  │
     │  └───────────────────┬─────────────────────┘  │
     │                      │                        │
     │  ┌───────────────────▼─────────────────────┐  │
     │  │          AGENT ROUTING                  │  │
     │  │ ╔════════╗  ╔════════╗  ╔═══════════╗   │  │
     │  │ ║  TARA  ║  ║  IRIS  ║  ║   CHAT    ║   │  │
     │  │ ║(Tools) ║  ║(Vision)║  ║ (General) ║   │  │
     │  │ ╚════╤═══╝  ╚════════╝  ╚═══════════╝   │  │
     │  │      │                                  │  │
     │  │ ┌────▼────────────────────────────┐     │  │
     │  │ │   🌊 UNIFIED ASYNC BRIDGE       │     │  │
     │  │ │  ThreadPool ↔ asyncio.run()     │     │  │
     │  │ └─────────────────────────────────┘     │  │
     └───────────────────────────────────────────────┘
```

### Key Design Patterns

1. **CEO → Circuit Breaker → Factory Flow**
   ```
   Supervisor.llm (property) → SafeLLM.invoke() → ModelManager.get_model() → Provider
                                    │
                                    └─> On 429: Switch provider, inject notice, retry
   ```

2. **Dynamic Provider Access** — Agents use `@property` for LLM access, not stored references
3. **Hot-Swap Capability** — Call `ModelManager.set_active_provider("openai")` at runtime

---

## ⚡ What's New in v3.0.0

### 🎭 Centralized Identity Engine
- Prompts are now loaded from `config/nia/prompts.json` rather than hardcoded.
- Ensures consistent persona across all interaction points (Chat, Supervisor).
- Supports instant personality updates without code changes.

### 👁️ Vision Hardening (IRIS)
- **Robust Initialization**: Explicitly validates `NVIDIA_API_KEY` on startup.
- **Fail-Safe**: If vision services are unavailable, IRIS gracefully degrades instead of crashing the graph.
- **Clear Feedback**: Returns specialized error messages helping users fix env config.

### 🔥 Multi-Provider LLM Support (Hot-Swap)
```python
from models.model_manager import get_model_manager

manager = get_model_manager()
manager.set_active_provider("openai")  # All agents now use OpenAI
manager.set_active_provider("groq")    # Switch to Groq for speed
```

Supported providers:
| Provider | Model | Use Case |
|----------|-------|----------|
| **nvidia** | Llama 3.1 70B | Primary (highest quality) |
| **openai** | GPT-4o | Fallback (widely available) |
| **groq** | Llama 3.1 70B | Speed (fastest inference) |
| **ollama** | Local models | Privacy (100% offline) |

### ⚡ Self-Healing Circuit Breaker (SafeLLM)
- **Auto-retry** with exponential backoff on rate limits
- **Auto-fallback** to alternative provider on 429/503 errors
- **Notice injection** — Agent knows when provider switched
- **Zero code changes** — Wrapped transparently by ModelManager

### 🏛️ ServiceContainer (Dependency Injection)
```python
from core.container import get_container

container = get_container()
memory = container.memory        # 4-Layer Memory
browser = container.browser_manager  # Playwright
```

---

## 🧠 4-Layer Hybrid Memory

| Layer | Storage | Purpose |
|-------|---------|---------|
| **Episodic** | ChromaDB | Semantic search over conversations |
| **Procedural** | NetworkX | Skill chains and tool sequences |
| **Preferences** | SQLite | User facts and settings |
| **Security** | SQLite | Audit logs and command history |

---

## 🛠️ TARA 2.0 Toolset (50+ Tools)

| Category | Tools |
|----------|-------|
| **Browser** | `browser_open_url`, `browser_click`, `browser_type`, `browser_scroll`, `browser_screenshot`, `browser_close`, `browser_new_tab`, `browser_get_content` |
| **Apps** | `launch_app`, `kill_app`, `list_processes` |
| **Windows** | `focus_window`, `minimize_window`, `maximize_window`, `snap_window`, `close_window`, `list_open_windows` |
| **File Ops** | `list_dir`, `read_file`, `write_file`, `delete_file`, `move_file`, `copy_file`, `search_files`, `get_file_info` |
| **System** | `system_power`, `set_volume`, `get_volume`, `system_stats`, `battery_status` |
| **Input** | `mouse_click`, `keyboard_type`, `keyboard_hotkey`, `mouse_scroll` |
| **Screen** | `take_screenshot`, `get_screen_resolution`, `get_mouse_position` |
| **Memory** | `save_user_preference`, `get_user_preference`, `list_user_preferences` |

---

## 🚀 Installation

### Prerequisites
- **Python 3.10+**
- **Windows 10/11** (Primary platform)
- **NVIDIA GPU** (Optional, for faster inference)

### Quick Start

```bash
# Clone
git clone https://github.com/Adi103-ETAI/N.I.A.git
cd N.I.A

# Create venv
python -m venv .venv
.venv\Scripts\activate

# Install
pip install -r requirements.txt

# Install Playwright browsers
playwright install chromium
```

### Environment Setup

Create `.env` in project root:

```env
# Primary Provider (required)
NVIDIA_API_KEY=nvapi-xxxx

# Fallback Providers (optional but recommended)
OPENAI_API_KEY=sk-xxxx
GROQ_API_KEY=gsk_xxxx

# Local Provider (optional)
OLLAMA_HOST=http://localhost:11434

# Runtime Configuration
ACTIVE_LLM_PROVIDER=nvidia    # Default provider on startup
DEBUG=false                   # Enable debug logging
```

---

## 📖 Usage

```bash
# Standard Run (Silent Mode)
python main.py

# Voice mode
python main.py --voice

# Always listening (no wake word)
python main.py --voice --no-wake

# Debug Run (Verbose Logs)
python main.py -d

# Check version
python main.py --version
```

### Example Commands

| Intent | Command | Handler |
|--------|---------|---------|
| Browser | "open google.com" | TARA → browser_open_url |
| App Launch | "open notepad" | TARA → launch_app |
| Vision | "what's on my screen" | IRIS → screen_capture |
| File | "create a file called notes.txt" | TARA → write_file |
| Memory | "remember I like dark mode" | TARA → save_user_preference |
| Provider | "switch to openai" | TARA → llm_switch_provider |

---

## 🔧 Tech Stack

| Component | Technology |
|-----------|------------|
| **Brain** | LangGraph, LangChain |
| **LLM Providers** | NVIDIA NIM, OpenAI, Groq, Ollama |
| **Voice STT** | Vosk (Offline) |
| **Voice TTS** | edge-tts (Microsoft Aria) |
| **Browser** | Playwright (Chromium) |
| **Desktop** | pyautogui, pywin32, pygetwindow |
| **Memory** | ChromaDB, NetworkX, SQLite |

---

## 🗺️ Roadmap

### v3.0.0 (Current)
- ✅ **Codebase Hygiene** — Removed orphaned files (`core/container.py`) & unused imports
- ✅ **TTS Latency Fix** — Implemented `run_coroutine_threadsafe` for NOLA
- ✅ **Robust Error Handling** — Added `check_db_health` & full stack trace logging
- ✅ Centralized Identity Engine & Prompts
- ✅ Vision Initialization Hardening

### v3.1 (Future)
- 🚀 **Native Async Graph** — Full async-first LangGraph
- 📦 **Poetry Migration** — Modern dependency management
- 🔌 **Plugin Architecture** — Dynamic tool loading
- 🧪 **Integration Tests** — End-to-end automation testing

---

## 📝 License

MIT License - See [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with ❤️ by SentArc Labs**

*"N.I.A. v3.1.0"*

</div>