<div align="center">

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║    ███╗   ██╗   ██╗    █████╗                                             ║
║    ████╗  ██║   ██║   ██╔══██╗     Neural Intelligence Assistant          ║
║    ██╔██╗ ██║   ██║   ███████║     ─────────────────────────────          ║
║    ██║╚██╗██║   ██║   ██╔══██║     CLASSIFICATION: DIRECTOR_LEVEL_ACCESS  ║
║    ██║ ╚████║██╗██║██╗██║  ██║     DEVELOPER: SentArc Labs                ║
║    ╚═╝  ╚═══╝╚═╝╚═╝╚═╝╚═╝  ╚═╝     VERSION: 4.0.0 (Velocity Edition)      ║
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

**v4.0.0 Release Highlights:**
- 🏗️ **Clean Architecture** — Complete restructuring with Domain-Driven Design
- 📁 **Unified `src/` Structure** — All code organized under `src/` with clear domains
- ⚙️ **YAML Configuration** — Human-readable configs with Pydantic validation
- 🧩 **Unified Capabilities** — Tools and skills merged into capability domains
- 🔌 **Extension System** — New extension architecture with v3.1.0 compatibility
- 🧪 **Improved Testing** — Test structure mirrors source (unit/integration/e2e)
- 📦 **Dependency Injection** — Enhanced ServiceRegistry with circular dependency detection

---

## 🏗️ Architecture

N.I.A. uses a **LangGraph-based Supervisor Pattern** with four specialized agents:

| Unit | Name     | Role                       | Technology                           |
|------|----------|--------------------------- |--------------------------------------|
| 🧠  | **NIA**  | Core Brain & Supervisor    | LangGraph + Multi-Provider LLM       |
| 🎤  | **NOLA** | Voice I/O (STT/TTS)        | Vosk (Offline) + Edge TTS            |
| 👁️  | **IRIS** | Vision & Screen Analysis   | Llama 3.2 Vision + mss               |
| 🛠️  | **TARA** | Tool Execution             | Unified Capabilities + 50+ Tools     |

### Directory Structure (v4.0.0)

```
N.I.A/
├── 📁 src/                          # All source code
│   ├── core/                        # Infrastructure (events, registry, platform)
│   ├── agents/                      # All agents unified
│   │   ├── nia/                     # Supervisor agent
│   │   ├── tara/                    # Tool execution agent
│   │   ├── iris/                    # Vision agent
│   │   ├── nola/                    # Voice agent
│   │   └── soldiers/                # Swarm nodes (code, browser, cmd)
│   ├── capabilities/                # Unified tool system
│   │   ├── desktop/                 # Apps, windows, input, screen
│   │   ├── system/                  # Files, clipboard, stats
│   │   ├── web/                     # Browser automation
│   │   └── execution/               # Lifecycle and environment
│   ├── infrastructure/              # Core systems
│   │   ├── container_engine/        # Docker sandbox
│   │   └── host_os/                 # AppIndex and OS adapters
│   ├── models/                      # LLM management
│   ├── persona/                     # Identity & personality
│   ├── interface/                   # CLI, API, GUI
│   └── extensions/                  # Extension system
│       └── compat/                  # v3.1.0 compatibility layer
│
├── 📁 data/                         # Runtime data (memory, cache, logs)
├── 📁 tests/                        # Mirrors src/ structure
│   ├── unit/
│   ├── integration/
│   └── e2e/
└── main.py                          # Entry point
```

### System Flow

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
     ┌───────────────────────▼─────────────────────────┐
     │              🧠 SUPERVISOR (NIA)               │
     │  ┌───────────────────────────────────────────┐  │
     │  │   ServiceRegistry (Dependency Injection)  │  │
     │  │   + Identity Injection System             │  │
     │  └───────────────────┬───────────────────────┘  │
     │                      │                          │
     │  ┌───────────────────▼───────────────────────┐  │
     │  │              AGENT ROUTING                │  │
     │  │   ╔════════╗  ╔════════╗  ╔═══════════╗   │  │
     │  │   ║  TARA  ║  ║  IRIS  ║  ║  SWARM    ║   │  │
     │  │   ║(Tools) ║  ║(Vision)║  ║(Soldiers) ║   │  │
     │  │   ╚════════╝  ╚════════╝  ╚═══════════╝   │  │
     │  │        │                                  │  │
     │  │   ┌────▼──────────────────────────────┐   │  │
     │  │   │   🧩 CAPABILITIES (Unified)       │   │  │
     │  │   │  desktop • system • web • ai      │   │  │
     │  │   └───────────────────────────────────┘   │  │  
     │  └───────────────────────────────────────────┘  │
     └─────────────────────────────────────────────────┘
```

---

## ⚙️ Configuration (v4.0.0 - YAML)

Configuration has been migrated from scattered JSON files to unified YAML:

### Agent Configuration

```yaml
# src/core/config/defaults/agents/nia.yaml
name: NIA
version: 4.0.0
debug_mode: false
log_level: INFO

routing_mode: hybrid
confidence_threshold: 0.7

memory:
  enabled: true
  max_conversation_length: 50
```

### Model Configuration

```yaml
# src/core/config/defaults/models.yaml
default_provider: nvidia

providers:
  nvidia:
    api_key: ${NVIDIA_API_KEY}
    model: meta/llama-3.1-70b-instruct
    temperature: 0.7
    max_tokens: 2000
  
  openai:
    api_key: ${OPENAI_API_KEY}
    model: gpt-4
    temperature: 0.7
    max_tokens: 2000

fallback_chain:
  - nvidia
  - openai
  - ollama
```

---

## 🧩 Capabilities (Unified Tool System)

The v4.0.0 capabilities system unifies tools and skills:

| Domain | Capabilities |
|--------|-------------|
| **Desktop** | `launch_app`, `kill_app`, `focus_window`, `minimize_window`, `maximize_window`, `snap_window`, `close_window`, `mouse_click`, `keyboard_type`, `keyboard_hotkey`, `take_screenshot` |
| **System** | `list_dir`, `read_file`, `write_file`, `delete_file`, `move_file`, `copy_file`, `search_files`, `system_stats`, `battery_status` |
| **Web** | `browser_open_url`, `browser_click`, `browser_type`, `browser_scroll`, `browser_screenshot`, `browser_close` |
| **Execution** | Application lifecycle and isolated command environments |

---

## 🧠 4-Layer Hybrid Memory

| Layer | Storage | Purpose |
|-------|---------|---------|
| **Episodic** | ChromaDB | Semantic search over conversations |
| **Procedural** | NetworkX | Skill chains and tool sequences |
| **Preferences** | SQLite | User facts and settings |
| **Security** | SQLite | Audit logs and command history |

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

# Install dependencies
uv sync

# Install Playwright browsers
uv run playwright install chromium
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
ACTIVE_LLM_PROVIDER=nvidia
DEBUG=false
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
| Code | "create a python snake game" | NIA → SWARM (Coding Agent) |

---

## 🔌 Extensions

### Creating an Extension

```python
# Create custom skills by adding to src/core/skills/library/
from src.core.skills.registry import SkillRegistry

def my_custom_skill(args):
    print(f"Executing skill with {args}")

# Register it using the core registry system
registry = SkillRegistry()
registry.register("my_custom_skill", my_custom_skill, "A custom tool")
```

See [docs/MIGRATION_GUIDE.md](docs/MIGRATION_GUIDE.md) for full migration details.

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
| **Config** | Pydantic, PyYAML |

---

## 🗺️ Roadmap

### v4.0.0 (Current)
- ✅ **Clean Architecture** — Domain-Driven Design restructuring
- ✅ **Unified src/ Structure** — All source under `src/`
- ✅ **YAML Configuration** — Pydantic-validated YAML configs
- ✅ **Unified Capabilities** — Tools and skills merged
- ✅ **Extension System** — Hot-loadable with v3.1.0 compat
- ✅ **Improved Testing** — unit/integration/e2e structure
- ✅ **ServiceRegistry DI** — Circular dependency detection

### v5.0.0 (Future)
- 🚀 **Multi-Platform** — macOS and Linux support
- 🌐 **REST API** — Remote control interface
- 🖼️ **GUI Interface** — Desktop application
- 🔐 **Enhanced Security** — Role-based permissions
- 🧠 **Agent Plugins** — Custom agent development

---

## 📝 License

MIT License - See [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with ❤️ by SentArc Labs**

*"N.I.A. v4.0.0 — Velocity Edition"*

</div>