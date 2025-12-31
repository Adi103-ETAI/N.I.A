# N.I.A. - Neural Intelligence Assistant

A voice-enabled AI assistant combining **LangGraph-based reasoning** with **real-time speech I/O**.

```
╔═══════════════════════════════════════════════════════════════════════════╗
║    ███╗   ██╗   ██╗    █████╗                                             ║
║    ████╗  ██║   ██║   ██╔══██╗     Neural Intelligence Assistant          ║
║    ██╔██╗ ██║   ██║   ███████║     ─────────────────────────────          ║
║    ██║╚██╗██║   ██║   ██╔══██║     Voice-Enabled AI Companion             ║
║    ██║ ╚████║██╗██║██╗██║  ██║     Powered by LangGraph + NOLA            ║
║    ╚═╝  ╚═══╝╚═╝╚═╝╚═╝╚═╝  ╚═╝                                            ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

## Features

- **Dual Input Modes**: Text (keyboard) or Voice (speech recognition)
- **Wake Word Activation**: "Jarvis", "NIA", or custom phrases
- **Multi-Agent Architecture**: Supervisor routing to specialist agents
- **NVIDIA NIM Integration**: Primary inference via NVIDIA's cloud models
- **Conversation Persistence**: SQLite-backed memory across sessions
- **10 Built-in Tools**: System stats, app control, web search, YouTube, and more

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    N.I.A. Voice Assistant                        │
│                                                                  │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────────┐  │
│  │     NOLA     │────►│     NIA      │────►│       NOLA       │  │
│  │   AsyncEar   │     │  Supervisor  │     │     AsyncTTS     │  │
│  │  (Listen)    │     │   → IRIS     │     │     (Speak)      │  │
│  │              │     │   → TARA     │     │                  │  │
│  └──────────────┘     └──────────────┘     └──────────────────┘  │
│      Voice In              Brain               Voice Out         │
└──────────────────────────────────────────────────────────────────┘
```

| Component | Purpose                                                  |
|:----------|:---------------------------------------------------------|
| **NIA**   | LangGraph Supervisor - routes requests to specialists    |
| **TARA**  | Technical Agent - system control, apps, web search, math |
| **IRIS**  | Vision Agent - image analysis (placeholder)              |
| **NOLA**  | Voice I/O - speech recognition + text-to-speech          |

## Installation

### Requirements
- Python 3.10+
- Windows/Linux/macOS

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/N.I.A.git
cd N.I.A

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp env.example .env
```

### Voice Mode Dependencies (Optional)

```bash
pip install speechrecognition pyttsx3 pyaudio
```

## Configuration

Create a `.env` file with your API keys:

```env
# REQUIRED - Primary Inference Provider
NVIDIA_API_KEY=your_nvidia_api_key

# OPTIONAL - Fallback Providers
OPENAI_API_KEY=your_openai_api_key
HUGGINGFACE_API_KEY=your_hf_api_key
OLLAMA_HOST=http://localhost:11434

# OPTIONAL - Fallback order
MODEL_PROVIDER_FALLBACKS=nvidia,openai,ollama
```

| Variable                   | Required | Description                        |
|:---------------------------|:---------|:-----------------------------------|
| `NVIDIA_API_KEY`           | ✅ Yes   | Primary inference via NVIDIA NIM   |
| `OPENAI_API_KEY`           | ❌ No    | Fallback to OpenAI GPT models      |
| `HUGGINGFACE_API_KEY`      | ❌ No    | Fallback to HuggingFace Inference  |
| `OLLAMA_HOST`              | ❌ No    | Local Ollama server URL            |
| `MODEL_PROVIDER_FALLBACKS` | ❌ No    | Comma-separated provider order     |

## Usage

### Basic Commands

```bash
# Text mode (default)
python main.py

# Check system status
python main.py --status

# Voice mode with wake words
python main.py --voice

# Voice mode (always listening)
python main.py --voice --no-wake

# Custom wake words
python main.py --voice --wake-words "computer,assistant"

# Debug logging
python main.py --debug

# Custom conversation thread
python main.py --thread-id myproject
```

### In-App Commands

| Command         | Description               |
|:----------------|:--------------------------|
| `help`          | Show available commands   |
| `status`        | Check system status       |
| `history`       | View conversation history |
| `clear history` | Reset conversation        |
| `voice on`      | Enable voice mode         |
| `voice off`     | Disable voice mode        |
| `wake <word>`   | Set custom wake word      |
| `clear`         | Clear screen              |
| `exit` / `quit` | Exit application          |

### TARA Capabilities (Technical Agent)

| Tool            | Example Prompt                        |
|:----------------|:------------------------------------- |
| System Stats    | "Check my system health"              |
| Time/Date       | "What time is it?"                    |
| Open App        | "Open brave browser"                  |
| Close App       | "Close notepad"                       |
| Play YouTube    | "Play Bohemian Rhapsody on YouTube"   |
| Web Search      | "Search for Python tutorials"         |
| Clipboard       | "Copy this text: Hello World"         |

## Project Structure

```
N.I.A/
├── main.py              # Entry point & CLI
├── requirements.txt     # Python dependencies
├── .env                 # API keys (gitignored)
├── env.example          # Environment template
│
├── nia/                 # Brain (Supervisor Agent)
│   ├── __init__.py
│   ├── agent.py         # SupervisorAgent, routing logic
│   ├── graph.py         # LangGraph state machine
│   └── state.py         # AgentState definition
│
├── tara/                # Technical Agent
│   ├── __init__.py
│   ├── agent.py         # TaraAgent implementation
│   └── tools/
│       ├── __init__.py  # Tool registry (10 tools)
│       ├── system.py    # SystemStats, DiskStats, Time
│       ├── desktop.py   # OpenApp, CloseApp, Clipboard, YouTube
│       └── web.py       # WebSearch (DuckDuckGo)
│
├── nola/                # Voice I/O System
│   ├── __init__.py
│   ├── manager.py       # NOLAManager orchestrator
│   ├── io.py            # AsyncEar, AsyncTTS
│   ├── security.py      # Command sanitization
│   └── wakeword.py      # Wake word detection
│
├── models/              # LLM Provider Management
│   ├── __init__.py
│   └── model_manager.py # NVIDIA/OpenAI/Ollama routing
│
├── persona/             # Personality Configuration
│   ├── __init__.py
│   └── profile.py       # PersonaProfile, system prompts
│
├── core/                # Legacy Utilities
│   ├── memory.py        # Memory management
│   └── tool_manager.py  # Tool registration
│
├── interface/           # Alternative Interfaces
│   └── chat.py          # Standalone chat mode
│
└── tests/               # Test Suite
    └── *.py
```

## License

MIT License - See LICENSE file for details.

## Version

N.I.A. v1.0.0
