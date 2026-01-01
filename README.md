<div align="center">

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║    ███╗   ██╗   ██╗    █████╗                                             ║
║    ████╗  ██║   ██║   ██╔══██╗     Neural Intelligence Assistant          ║
║    ██╔██╗ ██║   ██║   ███████║     ─────────────────────────────          ║
║    ██║╚██╗██║   ██║   ██╔══██║     CLASSIFICATION: DIRECTOR_LEVEL_ACCESS  ║
║    ██║ ╚████║██╗██║██╗██║  ██║     DEVELOPER: SentArc Labs                ║
║    ╚═╝  ╚═══╝╚═╝╚═╝╚═╝╚═╝  ╚═╝     VERSION: 2.0.0                         ║
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
│ N.I.A. SYSTEM DASHBOARD     │                2026-01-02 00:10:00 │
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
| 🧠  | **NIA**  | Core Brain & Supervisor    | LangGraph + NVIDIA NIM                |
| 🎤  | **NOLA** | Voice I/O (STT/TTS)        | Vosk (Offline) + Piper/Edge-TTS       |
| 👁️  | **IRIS** | Vision & Security Sentry   | Llama 3.2 Vision + mss                |
| 🛠️  | **TARA** | Tool Execution & Automation| Dynamic Registry + Function Calling   |

```
                    ┌──────────────────┐
                    │   USER INPUT     │
                    │  (Voice / Text)  │
                    └────────┬─────────┘
                             │
              ┌──────────────▼──────────────┐
              │     ⚡ REFLEX LAYER         │
              │  (Zero-Latency Commands)    │
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

---

## ⚡ Features

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

### Fuzzy Command Matching
- Say "turn mic on" or "activate voice" — both work
- Order-independent keyword detection

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
git clone https://github.com/yourusername/N.I.A.git
cd N.I.A

# Create virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download Vosk model (required for voice)
# Place in: models/vosk-model-small-en-us-0.15/

# Download Piper TTS binary (optional)
# Place in: nola/piper_bin/
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

# Check system status
python main.py --status
```

### Command Reference

#### 🎤 Voice Control (NOLA)
| Command   | Aliases                                              | Action               |
|-----------|------------------------------------------------------|----------------------|
| Voice On  | `mic on`, `wake up`, `ears on`, `start listening`    | Enable microphone    |
| Voice Off | `mic off`, `go silent`, `ears off`, `stop listening` | Mute microphone      |
| Shh       | `quiet`, `shut up`, `hush`, `be quiet`               | Stop TTS immediately |

#### 👁️ Vision Control (IRIS)
| Command    | Aliases                                                  | Action                   |
|------------|----------------------------------------------------------|--------------------------|
| Sentry On  | `eyes on`, `guard mode`, `watch screen`, `start watching`| Enable screen monitoring |
| Sentry Off | `eyes off`, `standby`, `stop watching`                   | Disable sentry           |

#### 🔊 Audio Control (TARA Reflex)
| Command   | Aliases                                              | Action                   |
|-----------|------------------------------------------------------|--------------------------|
| Mute      | `kill sound`, `silence speakers`, `sound off`        | Mute system audio        |
| Unmute    | `sound on`, `restore audio`, `speakers on`           | Unmute audio             |

#### ⚙️ System Commands
| Command   | Aliases                                              | Action                   |
|-----------|------------------------------------------------------|--------------------------|
| Status    | `report`, `stats`, `diagnostics`, `performance`      | Show dashboard           |
| Clear     | `cls`, `clean screen`                                | Clear terminal           |
| Exit      | `quit`, `bye`, `goodbye`, `shutdown`                 | Exit N.I.A.              |
| Help      | `commands`, `what can you do`                        | Show help                |

---

## 📁 Directory Structure

```
N.I.A/
├── main.py                 # Entry point
├── requirements.txt        # Dependencies
├── .env                    # API keys (create this)
│
├── core/                   # Core engine & orchestration
│   ├── engine.py           # Main assistant loop
│   └── health.py           # System diagnostics
│
├── nia/                    # 🧠 Brain module
│   ├── graph.py            # LangGraph supervisor
│   └── state.py            # Conversation state
│
├── nola/                   # 🎤 Voice module
│   ├── io.py               # AsyncEar (STT) + AsyncTTS
│   ├── manager.py          # NOLAManager orchestration
│   └── security.py         # Input sanitization
│
├── iris/                   # 👁️ Vision module
│   ├── agent.py            # Vision analysis agent
│   └── sentry.py           # Background screen monitor
│
├── tara/                   # 🛠️ Tools module
│   ├── agent.py            # Tool execution agent
│   ├── registry.py         # Dynamic tool discovery
│   └── units/              # Tool implementations
│       ├── system_control.py
│       ├── desktop_control.py
│       └── web_tools.py
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

## 🔧 Configuration

### Persona Customization
Edit `persona/default.py` to change N.I.A.'s personality:

```python
SYSTEM_PROMPT = """You are N.I.A., a helpful AI assistant..."""
```

### Adding Custom Tools
Create a new file in `tara/units/` with `@tool` decorated functions:

```python
from tara.protocols import tool

@tool(description="My custom tool")
def my_tool(arg: str) -> str:
    return f"Processed: {arg}"
```

Tools are automatically discovered on startup.

---

## 🛡️ Security

- **Offline Voice**: Vosk runs entirely on-device
- **Local Vision**: Sentry uses local Llama Vision model
- **Input Sanitization**: All voice input passes through security filters
- **No Telemetry**: Zero data collection or cloud uploads

---

## 📝 License

MIT License - See [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with ❤️ by SentArc Labs**

*"Your Intelligence, Augmented."*

</div>
