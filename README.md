<div align="center">

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║    ███╗   ██╗   ██╗    █████╗                                             ║
║    ████╗  ██║   ██║   ██╔══██╗     Neural Intelligence Assistant          ║
║    ██╔██╗ ██║   ██║   ███████║     ─────────────────────────────          ║
║    ██║╚██╗██║   ██║   ██╔══██║     CLASSIFICATION: DIRECTOR_LEVEL_ACCESS  ║
║    ██║ ╚████║██╗██║██╗██║  ██║     DEVELOPER: SentArc Labs                ║
║    ╚═╝  ╚═══╝╚═╝╚═╝╚═╝╚═╝  ╚═╝     VERSION: 2.5.0 (Stable)                ║
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

**v2.5.0 "Resurrection" Release Highlights:**
- 🏛️ **ServiceContainer DI** — No more singleton hacks
- 🔄 **Unified Async Bridge** — Stable browser automation
- 📝 **Protocol-based Routing** — Clean agent injection
- ⚡ **Exponential Backoff** — Resilient retry logic
- ✅ **Production Ready** — All critical issues resolved

---

## 🏗️ Architecture

N.I.A. uses a **LangGraph-based Supervisor Pattern** with four specialized units:

| Unit | Name     | Role                       | Technology                           |
|------|----------|--------------------------- |--------------------------------------|
| 🧠  | **NIA**  | Core Brain & Supervisor    | LangGraph + NVIDIA NIM               |
| 🎤  | **NOLA** | Voice I/O (STT/TTS)        | Vosk (Offline) + Edge TTS            |
| 👁️  | **IRIS** | Vision & Screen Analysis   | Llama 3.2 Vision + mss               |
| 🛠️  | **TARA** | Tool Execution (v2.0)      | Unified Async Bridge + 50 Tools      |

### System Architecture (v2.5.0)

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
     │              🏛️ SERVICE CONTAINER             │
     │  ┌─────────────────────────────────────────┐  │
     │  │         🧠 NIA SUPERVISOR               │  │
     │  │     (Protocol-based Agent Routing)      │  │
     │  └───┬─────────────────────────────┬───────┘  │
     │      │                             │          │
     │ ╔════▼════╗   ╔════════╗   ╔═══════▼══════╗   │
     │ ║  TARA   ║   ║  IRIS  ║   ║    CHAT      ║   │
     │ ║(Tools)  ║   ║(Vision)║   ║  (General)   ║   │
     │ ╚════╤════╝   ╚════════╝   ╚══════════════╝   │
     │      │                                        │
     │ ┌────▼────────────────────────────────┐       │
     │ │      🌊 UNIFIED ASYNC BRIDGE        │       │
     │ │   ThreadPool ↔ asyncio.run() ↔ LLM  │       │
     │ └─────────────────────────────────────┘       │
     └───────────────────────────────────────────────┘
```

### Dependency Injection (v2.5.0)

The `ServiceContainer` replaces all singleton patterns:

```python
from core.container import get_container

container = get_container()
memory = container.memory        # 4-Layer Memory
browser = container.browser_manager  # Playwright
```

---

## ⚡ What's New in v2.5.0

### 🏛️ ServiceContainer (Dependency Injection)
- **Before**: `WindowRegistry.__new__()` singleton hacks
- **After**: Clean `ServiceContainer` with explicit injection
- Enables unit testing with mock services

### 🌊 Unified Async Bridge
- **Before**: Complex `is_async` detection with heuristics
- **After**: Single `await tool.ainvoke()` for all tools
- LangChain's polymorphic `ainvoke()` handles sync/async automatically

### 📝 Protocol-based Agent Routing
```python
@runtime_checkable
class AgentProtocol(Protocol):
    def process(self, state: Dict) -> Dict: ...
    def run(self, query: str) -> str: ...
```
- Agents can be swapped without changing Supervisor code
- Type-safe at IDE level, duck-typed at runtime

### ⚡ Exponential Backoff
```python
# Retry logic with jitter (prevents retry storms)
delay = min(0.5 * (2 ** attempt) + random.uniform(-0.125, 0.125), 5.0)
```

### ✅ Strict Type Safety
- All node functions return `TaraStateUpdate` TypedDict
- `from __future__ import annotations` throughout
- Full TYPE_CHECKING support for IDE hints

---

## 🧠 4-Layer Hybrid Memory

| Layer | Storage | Purpose |
|-------|---------|---------|
| **Episodic** | ChromaDB | Semantic search over conversations |
| **Procedural** | NetworkX | Skill chains and tool sequences |
| **Preferences** | SQLite | User facts and settings |
| **Security** | SQLite | Audit logs and command history |

---

## 🛠️ TARA 2.0 Toolset (50 Tools)

| Category | Tools |
|----------|-------|
| **Browser** | `browser_open_url`, `browser_click`, `browser_type`, `browser_scroll`, `browser_screenshot`, `browser_close`, `browser_new_tab`, `browser_get_content` |
| **Apps** | `launch_app`, `close_app`, `focus_app`, `windows_manager` |
| **File Ops** | `create_file`, `read_file`, `write_file`, `delete_file`, `list_directory`, `move_file`, `copy_file`, `search_files`, `zip_files`, `unzip_file`, `get_file_info` |
| **System** | `system_power`, `set_volume`, `get_volume`, `system_stats`, `battery_status` |
| **UI Automation** | `dump_ui_tree`, `find_ui_element`, `click_ui_element` |
| **Memory** | `save_user_preference`, `get_user_preference`, `list_user_preferences` |
| **Ghost** | `ghost_mode` |

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
# Required
NVIDIA_API_KEY=nvapi-xxxx

# Optional
GROQ_API_KEY=gsk_xxxx
OPENAI_API_KEY=sk-xxxx
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
```

### Example Commands

| Intent | Command | Handler |
|--------|---------|---------|
| Browser | "open google.com" | TARA → browser_open_url |
| App Launch | "open notepad" | TARA → launch_app |
| Vision | "what's on my screen" | IRIS → screen_capture |
| File | "create a file called notes.txt" | TARA → create_file |
| Memory | "remember I like dark mode" | TARA → save_user_preference |

---

## 📁 Directory Structure

```
N.I.A/
├── main.py                     # Entry point (v2.5.0)
├── requirements.txt            # Dependencies
├── .env                        # API keys
│
├── core/                       # 🧠 Core services
│   ├── engine.py               # NIAAssistant orchestrator
│   ├── config.py               # Pydantic settings
│   ├── container.py            # ServiceContainer (DI)
│   ├── memory.py               # 4-Layer Memory
│   └── logger.py               # Centralized logging
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
│   └── tools/                  # 50 Tools
│       ├── browser_ops.py      # Playwright browser
│       ├── app_launcher.py     # Application control
│       ├── file_ops.py         # File operations
│       └── interface.py        # Tool discovery
│
├── nola/                       # 🎤 Voice I/O
│   ├── manager.py              # NOLAManager
│   └── io/
│       ├── speech.py           # Edge TTS
│       └── hearing.py          # Vosk STT
│
├── iris/                       # 👁️ Vision
│   ├── agent.py                # IrisAgent
│   └── sentry.py               # Screen monitor
│
└── tests/                      # Unit tests
```

---

## 🔧 Tech Stack

| Component | Technology |
|-----------|------------|
| **Brain** | LangGraph, LangChain, NVIDIA NIM |
| **LLM** | Llama 3.1 70B (NVIDIA), Llama 3.2 Vision |
| **Voice STT** | Vosk (Offline) |
| **Voice TTS** | edge-tts (Microsoft Aria) |
| **Browser** | Playwright (Chromium) |
| **Desktop** | pyautogui, AppOpener, pycaw |

---

## 🛡️ Security

- **Offline Voice**: Vosk runs 100% locally
- **Local Vision**: Llama Vision on-device
- **Hardware Mute**: Physical mic release
- **Input Sanitization**: All inputs filtered
- **Ghost Protocol**: Emergency privacy mode
- **No Telemetry**: Zero data collection

---

## 🗺️ Roadmap

### v2.5.0 (Current - Stable)
- ✅ ServiceContainer Dependency Injection
- ✅ Unified Async Bridge for browser tools
- ✅ Protocol-based agent routing
- ✅ Exponential backoff with jitter
- ✅ Strict type safety (TypedDict, Protocols)

### v3.0 (Future)
- 🚀 **Native Async Graph** — Full async-first LangGraph
- 📦 **Poetry Migration** — Modern dependency management
- 🔌 **Plugin Architecture** — Dynamic tool loading
- 🌐 **Multi-Provider LLM** — Anthropic, Groq, Ollama support
- 🧪 **Integration Tests** — End-to-end automation testing

---

## 📝 License

MIT License - See [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with ❤️ by SentArc Labs**

*"N.I.A. v2.5.0 — Resurrected. Stable. Production Ready."*

</div>