<div align="center">

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║    ███╗   ██╗   ██╗    █████╗                                             ║
║    ████╗  ██║   ██║   ██╔══██╗     Neural Intelligence Assistant          ║
║    ██╔██╗ ██║   ██║   ███████║     ─────────────────────────────          ║
║    ██║╚██╗██║   ██║   ██╔══██║     CLASSIFICATION: DIRECTOR_LEVEL_ACCESS  ║
║    ██║ ╚████║██╗██║██╗██║  ██║     DEVELOPER: SentArc Labs                ║
║    ╚═╝  ╚═══╝╚═╝╚═╝╚═╝╚═╝  ╚═╝     VERSION: 2.1.0                         ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

### A Modular, Multi-Modal AI System

**Voice • Vision • Tools • Reflexes**

</div>

---

## 🎯 Overview

**N.I.A.** (Neural Intelligence Assistant) is a privacy-first, modular AI assistant designed for power users. It combines offline voice recognition, vision-based security monitoring, and automated tool execution into a unified system controlled by natural language or zero-latency reflexes.

```
┌─────────────────────────────┬────────────────────────────────────┐
│ N.I.A. SYSTEM DASHBOARD     │                2026-01-04 12:00:00 │
├─────────────────────────────┼────────────────────────────────────┤
│ 🧠 SUBSYSTEMS               │ 📊 RESOURCES                      │
│ • BRAIN (NIA) : [ON ]       │  CPU: [████░░░░░░]  42%            │
│ • VOICE (NOLA): [ON ]       │  RAM: [███████░░░]  76%            │
│ • SENTRY(IRIS): [OFF]       │  DSK: [██░░░░░░░░]  27%            │
│ • TOOLS (TARA): [ON ]       │                                    │
├─────────────────────────────┼────────────────────────────────────┤
│ 💾 MEMORY                   │ 🔐 SECURITY KEYS                  │
│  RAM : 7.6/10.0 GB          │  NVIDIA API: [LINKED ]             │
│  DISK: 680.5 GB Free        │  OPENAI API: [LINKED ]             │
└─────────────────────────────┴────────────────────────────────────┘
```

---

## 🏗️ Architecture

N.I.A. is composed of four specialized units working in concert:

| Unit | Name     |         Role               |              Technology              |
|------|----------|--------------------------  |--------------------------------------|
| 🧠  | **NIA**  | Core Brain & Supervisor    | LangGraph + NVIDIA NIM               |
| 🎤  | **NOLA** | Voice I/O (STT/TTS)        | Vosk (Offline) + Edge TTS (Aria)     |
| 👁️  | **IRIS** | Vision & Security Sentry   | Llama 3.2 Vision + mss               |
| 🛠️  | **TARA** | Tool Execution & Automation| Dynamic Registry + 14 Tools          |

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
         ┌───────────────────▼───────────────────┐
         │           🧠 NIA CORE                 │
         │      (LangGraph Supervisor)           │
         └───┬───────────┬───────────────┬───────┘
             │           │               │
      ┌──────▼──────┐ ┌──▼──────────┐ ┌──▼──────────┐
      │ 🎤 NOLA     │ │ 👁️ IRIS    │ │ 🛠️ TARA     │
      │ Voice I/O   │ │ Vision      │ │ Tools       │
      └─────────────┘ └─────────────┘ └─────────────┘
```

### Singleton Pattern

The Voice Manager (`NOLAManager`) uses a **Singleton Pattern** for stability:
- Single instance prevents hardware conflicts
- Thread-safe microphone access
- Consistent state across components

---

## ⚡ Features

### 🔇 True Hardware Mute (NEW)
The "Kill Mic" command **physically releases the microphone driver**:
- Closes the audio input stream entirely
- No "software mute" — the hardware is truly freed
- Say "mic on" to reopen the stream

### 🎙️ Edge TTS Integration (NEW)
High-quality neural voice synthesis:
- **Voice**: Microsoft `en-US-AriaNeural` (Cortana-like)
- **Fallback**: Piper TTS for offline operation
- Smooth playback via pygame

### 🧠 Smart Fuzzy Routing (NEW)
Natural language command recognition with **order-independent keyword matching**:
- "Kill the mic" → Mic Off
- "Turn off microphone" → Mic Off  
- "Disable voice" → Mic Off

The system extracts keywords (`mic` + `off`) regardless of phrasing.

### Zero-Latency Reflexes
Built-in command vocabulary bypasses the LLM for instant response:
- Voice control, mute/unmute, sentry toggle — all sub-50ms

### Privacy-First Sentry
- Screen monitoring runs **100% locally**
- LLama 3.2 Vision analyzes frames on-device
- No cloud uploads, no data leaves your machine

### Director-Level Dashboard
- Real-time system metrics (CPU, RAM, Disk)
- Subsystem status at a glance
- API key validation

---

## 🚀 Installation

### Prerequisites
- **Python 3.10+**
- **Windows 10/11** (Primary platform)
- **NVIDIA GPU** (Optional, for faster inference)
- **Nmap** (Optional, for network scanning tools)

### Steps

```bash
# Clone the repository
git clone https://github.com/Adi103-ETAI/N.I.A.git
cd N.I.A

# Create virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download Vosk model (required for voice)
# Place in: nola/vosk_model/
```

### Environment Setup

Create a `.env` file in the project root:

```env
# Required: At least one LLM provider
NVIDIA_API_KEY=nvapi-xxxx
OPENAI_API_KEY=sk-xxxx

# Optional
HUGGINGFACE_API_KEY=hf_xxxx
OLLAMA_HOST=http://localhost:11434
```

---

## 📖 Usage

### Starting N.I.A.

```bash
# Text mode (default)
python main.py

# Voice mode
python main.py --voice

# Voice mode (always listening, no wake word)
python main.py --voice --no-wake

# Check system status
python main.py --status

# Debug mode
python main.py --debug
```

### Command Reference

#### 🎤 Voice Control (NOLA) — with Fuzzy Matching

| Intent    | Example Phrases                                            | Action                  |
|-----------|------------------------------------------------------------|-------------------------|
| Mic On    | `mic on`,`enable microphone`,`start voice`,`activate mic`  | Open mic stream         |
| Mic Off   | `mic off`,`kill the mic`,`disable voice`,`mute microphone` | Release mic hardware    |
| Shh       | `quiet`,`shut up`,`hush`,`be quiet`,`stop talking`         | Stop TTS immediately    |

#### 👁️ Vision Control (IRIS)
| Command    | Aliases                                                  | Action                   |
|------------|----------------------------------------------------------|--------------------------|
| Sentry On  | `eyes on`, `guard mode`, `watch screen`, `start watching`| Enable screen monitoring |
| Sentry Off | `eyes off`, `standby`, `stop watching`                   | Disable sentry           |

#### 🔊 Audio Control (TARA Reflex)
| Command   | Aliases                                              | Action                   |
|-----------|------------------------------------------------------|--------------------------|
| Mute      | `mute speakers`, `kill sound`, `sound off`           | Mute system audio        |
| Unmute    | `sound on`, `restore audio`, `speakers on`           | Unmute audio             |

#### 🔒 Ghost Protocol (Emergency Privacy)
| Command         | Description                                          |
|-----------------|------------------------------------------------------|
| Ghost Layer 1   | Mute audio, disable TTS, minimize windows            |
| Ghost Layer 2   | + Kill distraction apps (browsers, media players)    |
| Ghost Layer 3   | + Lock workstation immediately                       |

#### ⚙️ System Commands
| Command   | Aliases                                              | Action                  |
|-----------|------------------------------------------------------|-------------------------|
| Status    | `report`, `stats`, `diagnostics`, `performance`      | Show dashboard          |
| Clear     | `cls`, `clean screen`                                | Clear terminal          |
| Exit      | `quit`, `bye`, `goodbye`, `shutdown`                 | Exit N.I.A.             |
| Help      | `commands`, `what can you do`                        | Show help               |

---

## 🛠️ TARA Toolset (14 Tools)

| Category           | Tools                                                              |
|--------------------|--------------------------------------------------------------------|
| **System Control** | `system_power`, `empty_recycle_bin`, `set_volume`, `mute_volume`,  |
|                    |   `get_volume`, `system_stats`, `battery_status`                   |
| **Desktop Control**| `app_control`, `browser_general`, `window_manager`, `file_manager` |
| **Web Search**     | `web_search`, `web_news`                                           |
| **Ghost Protocol** | `ghost_mode`                                                       |

---

## 📁 Directory Structure

```
N.I.A/
├── main.py                 # Entry point (Standard Logging)
├── requirements.txt        # Dependencies
├── .env                    # API keys (create this)
│
├── core/                   # Core engine & orchestration
│   ├── engine.py           # NIAAssistant + Fuzzy Reflex Layer
│   └── health.py           # System diagnostics
│
├── nia/                    # 🧠 Brain module
│   ├── graph.py            # LangGraph supervisor
│   └── state.py            # Conversation state
│
├── nola/                   # 🎤 Voice module
│   ├── io.py               # Edge TTS + Vosk STT (Singletons)
│   ├── manager.py          # NOLAManager (Hardware Mute Logic)
│   └── security.py         # Input sanitization
│
├── iris/                   # 👁️ Vision module
│   ├── agent.py            # Vision analysis agent
│   └── sentry.py           # Background screen monitor
│
├── tara/                   # 🛠️ Tools module
│   ├── agent.py            # Tool execution agent
│   ├── registry.py         # Dynamic tool discovery
│   └── units/              # Tool implementations (14 tools)
│       ├── system_control.py   # Power, Volume, Stats
│       ├── desktop_control.py  # Apps, Windows, Browser, Files
│       ├── web_search.py       # DuckDuckGo Search & News
│       └── ghost_protocol.py   # Emergency Privacy Mode
│
├── interface/              # UI components
│   ├── banner.py           # ASCII banner
│   └── chat.py             # Interactive prompt
│
├── persona/                # Personality config
│   └── default.py          # Default persona
│
├── models/                 # ML models (Vosk, etc.)
└── data/                   # Persistent state
```

---

## 🔧 Tech Stack

| Component     | Technology                                              |
|---------------|---------------------------------------------------------|
| **Brain**     | LangGraph, LangChain, NVIDIA NIM                        |
| **Voice STT** | Vosk (Offline), sounddevice                             |
| **Voice TTS** | edge-tts (`en-US-AriaNeural`), pygame                   |
| **Vision**    | Llama 3.2 Vision (Local), mss, Pillow                   |
| **Tools**     | pyautogui, AppOpener, pycaw, DuckDuckGo Search          |
| **UI**        | prompt_toolkit                                          |

---

## 🔧 Configuration

### Persona Customization
Edit `persona/profile.py` to change N.I.A.'s personality:

```python
SYSTEM_PROMPT = """You are N.I.A., a helpful AI assistant..."""
```

### Adding Custom Tools
Create a new file in `tara/units/` with `@tara_tool` decorated functions:

```python
from tara.protocols import tara_tool

@tara_tool(name="my_tool", category="custom", description="My custom tool")
def my_tool(arg: str) -> str:
    return f"Processed: {arg}"
```

Tools are automatically discovered on startup.

---

## 🛡️ Security

- **Offline Voice**: Vosk runs entirely on-device
- **Local Vision**: Sentry uses local Llama Vision model
- **Hardware Mute**: Mic stream is physically closed (not software muted)
- **Input Sanitization**: All voice input passes through security filters
- **Ghost Protocol**: Emergency privacy mode with hardware lock
- **No Telemetry**: Zero data collection or cloud uploads

---

## 📝 License

MIT License - See [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with ❤️ by SentArc Labs**

*"Your Intelligence, Augmented."*

</div>