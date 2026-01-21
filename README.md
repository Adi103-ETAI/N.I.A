<div align="center">

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║    ███╗   ██╗   ██╗    █████╗                                             ║
║    ████╗  ██║   ██║   ██╔══██╗     Neural Intelligence Assistant          ║
║    ██╔██╗ ██║   ██║   ███████║     ─────────────────────────────          ║
║    ██║╚██╗██║   ██║   ██╔══██║     CLASSIFICATION: DIRECTOR_LEVEL_ACCESS  ║
║    ██║ ╚████║██╗██║██╗██║  ██║     DEVELOPER: SentArc Labs                ║
║    ╚═╝  ╚═══╝╚═╝╚═╝╚═╝╚═╝  ╚═╝     VERSION: 2.5.2 (Velocity)              ║
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

**v2.5.2 "Velocity" Release Highlights:**
- 🔥 **Multi-Provider LLM Hot-Swap** — Switch between NVIDIA, OpenAI, Groq, Ollama at runtime
- ⚡ **Self-Healing Circuit Breaker** — Auto-fallback on rate limits (429/503 errors)
- 🏛️ **ServiceContainer DI** — Clean dependency injection
- 🔄 **Unified Async Bridge** — Stable browser automation
- 📝 **Protocol-based Routing** — Clean agent injection
- ✅ **Production Ready** — All critical issues resolved

---

## 🏗️ Architecture

N.I.A. uses a **LangGraph-based Supervisor Pattern** with four specialized units:

| Unit | Name     | Role                       | Technology                           |
|------|----------|--------------------------- |--------------------------------------|
| 🧠  | **NIA**  | Core Brain & Supervisor    | LangGraph + Multi-Provider LLM       |
| 🎤  | **NOLA** | Voice I/O (STT/TTS)        | Vosk (Offline) + Edge TTS            |
| 👁️  | **IRIS** | Vision & Screen Analysis   | Llama 3.2 Vision + mss               |
| 🛠️  | **TARA** | Tool Execution (v2.0)      | Unified Async Bridge + 50 Tools      |

### System Architecture (v2.5.2)

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

## ⚡ What's New in v2.5.2

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
# Text mode (default)
python main.py

# Voice mode
python main.py --voice

# Always listening (no wake word)
python main.py --voice --no-wake

# Debug mode
python main.py --debug

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

## 📁 Directory Structure

```
N.I.A/
├── main.py                     # Entry point (v2.5.2)
├── requirements.txt            # Dependencies
├── .env                        # API keys & config
│
├── core/                       # 🧠 Core services
│   ├── engine.py               # NIAAssistant orchestrator
│   ├── config.py               # Pydantic settings
│   ├── container.py            # ServiceContainer (DI)
│   ├── memory.py               # 4-Layer Memory
│   └── logger.py               # Centralized logging
│
├── models/                     # 🏭 LLM Factory
│   ├── model_manager.py        # Multi-Provider + Hot-Swap
│   └── safe_llm.py             # Circuit Breaker wrapper
│
├── nia/                        # 🧠 Brain module
│   ├── agent.py                # SupervisorAgent (Protocol-based)
│   ├── gatekeeper.py           # Routing validation
│   └── graph/                  # LangGraph
│       ├── builder.py          # Graph construction
│       └── nodes.py            # Node definitions
│
├── tara/                       # 🛠️ Tool Execution
│   ├── graph/                  # TARA 2.0 SubGraph
│   │   ├── nodes.py            # Unified Async Bridge
│   │   ├── state.py            # TaraState TypedDict
│   │   └── workflow.py         # Graph builder
│   └── tools/                  # 50+ Tools
│       ├── browser_ops.py      # Playwright browser
│       ├── app_launcher.py     # Application control
│       ├── file_ops.py         # File operations
│       ├── window_ops.py       # Window management
│       └── interface.py        # Tool discovery
│
├── nola/                       # 🎤 Voice I/O
│   ├── manager.py              # NOLAManager
│   ├── security.py             # InputSanitizer
│   └── io/
│       ├── speech.py           # Edge TTS
│       └── hearing.py          # Vosk STT
│
├── iris/                       # 👁️ Vision
│   ├── agent.py                # IrisAgent
│   └── tools.py                # Screen/Webcam capture
│
└── tests/                      # Unit tests
```

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

## 🛡️ Security

- **Offline Voice**: Vosk runs 100% locally
- **Local Vision**: Llama Vision on-device option
- **Hardware Mute**: Physical mic release
- **Input Sanitization**: InputSanitizer blocks dangerous patterns
- **Safety Locks**: delete_file requires `confirm=True`
- **Ghost Protocol**: Emergency privacy mode
- **No Telemetry**: Zero data collection

---

## 🗺️ Roadmap

### v2.5.2 (Current - Velocity)
- ✅ Multi-Provider LLM (NVIDIA, OpenAI, Groq, Ollama)
- ✅ SafeLLM Circuit Breaker with auto-fallback
- ✅ Dynamic provider hot-swap via ModelManager
- ✅ ServiceContainer Dependency Injection
- ✅ Unified Async Bridge for browser tools
- ✅ Protocol-based agent routing
- ✅ Strict type safety (TypedDict, Protocols)

### v3.0 (Future)
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

*"N.I.A. v2.5.2 — Velocity. Multi-Provider. Production Ready."*

</div>