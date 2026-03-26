# N.I.A Setup Guide for Windows, Linux, and macOS

## Quick Start (All Platforms)

```bash
# 1. Clone the repository
git clone <repo-url>
cd N.I.A

# 2. Install dependencies with uv
uv sync --all-groups

# 3. Create .env file with your API keys
cp .env.example .env
# Edit .env with your API keys (NVIDIA, OpenAI, etc.)

# 4. Run the system
python main.py
# or: uv run nia
```

## Platform-Specific Setup

### Windows

#### Prerequisites
- Python 3.11+ (tested on 3.12.1)
- Visual Studio Build Tools (for native extensions)
- Git

#### Installation

```powershell
# PowerShell setup
pip install uv
uv sync --all-groups

# Optional: Enable Windows Defender protection (recommended)
# GUI automation works best with UAC disabled for automation tasks
```

#### Features Available
- ✅ Full desktop automation (UIAutomation)
- ✅ Native audio control (PyCaw)
- ✅ Window management (PyGetWindow)
- ✅ Screenshot capture (both PIL and MSS)
- ✅ Native file operations

#### Troubleshooting
- **UIAutomation not available**: Install `pywinauto`
  ```
  uv add pywinauto
  ```
- **Audio control fails**: Install `pycaw`
  ```
  uv add pycaw
  ```

### Linux (Ubuntu/Debian)

#### Prerequisites
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y \
    python3.11-dev \
    build-essential \
    git \
    libssl-dev \
    libffi-dev

# Fedora/RHEL
sudo dnf install -y \
    python3.11-devel \
    gcc \
    make \
    openssl-devel
```

#### Installation

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env

# Clone and setup
uv sync --all-groups

# Optional: Install GUI automation tools
sudo apt-get install -y xdotool xclip
```

#### Features Available
- ✅ Full LLM integration
- ✅ Docker execution
- ✅ File operations
- ✅ Screenshot capture (PIL)
- ✅ Browser automation (Playwright)
- ⚠️ Basic desktop automation (PyAutoGUI)
- ⚠️ GUI automation via xdotool (optional)

#### Troubleshooting
- **Display server issues**: Use Docker container if X11 not available
  ```bash
  docker-compose -f docker/docker-compose.yml up
  ```
- **Screenshot not working**: Install additional packages
  ```bash
  sudo apt-get install -y python3-tk python3-dev
  ```

### macOS

#### Prerequisites
- Python 3.11+ (via Homebrew or pyenv)
- Xcode Command Line Tools: `xcode-select --install`
- Homebrew (optional but recommended)

#### Installation

```bash
# Via Homebrew (recommended)
brew install python@3.12 git

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env

# Clone and setup
uv sync --all-groups

# Optional: Install GUI automation tools
brew install lua
```

#### Features Available
- ✅ Full LLM integration
- ✅ Docker execution (via Docker Desktop)
- ✅ File operations
- ✅ Screenshot capture (PIL)
- ✅ Browser automation (Playwright)
- ⚠️ Basic desktop automation (PyAutoGUI)
- ⚠️ macOS-native automation (Coming: PyObjC)

#### Troubleshooting
- **Script Editor access**: Grant Terminal access in Security Settings
- **Docker issues**: Ensure Docker Desktop is installed and running
- **Screenshot permissions**: Grant Screen Recording permission to Terminal

## Verify Installation

Run the platform detection to verify everything is installed:

```bash
# Show system info and available features
uv run python -c "
from src.core.os import get_os_context
from src.core.features import get_features

ctx = get_os_context()
feat = get_features()

print(ctx.summary())
print(feat.summary())
"
```

Or use the built-in health check:

```bash
uv run python -c "
from src.core.health import print_system_status
print_system_status()
"
```

## Docker Setup (All Platforms)

For headless execution or when desktop automation not available:

```bash
# Build the sandbox environment
docker-compose -f docker/docker-compose.yml build

# Run with Docker
docker-compose -f docker/docker-compose.yml up

# Run specific service
docker-compose -f docker/docker-compose.yml up sandbox
```

## Configuration (.env)

Required environment variables:

```bash
# AI Provider (choose one)
NVIDIA_API_KEY=nvapi-xxxxxxxxxxxx
# OR
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
# OR
GROQ_API_KEY=gsk-xxxxxxxxxxxxxxxx

# Optional configurations
DEBUG=false
VOICE_ENABLED=true
WAKE_WORDS=nia,hey nia

# LLM Models
LLM_MODEL=meta/llama-3.1-70b-instruct
LLM_MODEL_VISION=meta/llama-3.2-90b-vision-instruct
```

## Running N.I.A

```bash
# Standard execution
uv run nia

# With specific mode
uv run python main.py --help

# Debug mode
DEBUG=true uv run nia

# Specific LLM provider
ACTIVE_LLM_PROVIDER=openai uv run nia
```

## Testing

```bash
# Run all tests
uv run pytest

# Run cross-platform tests
uv run pytest tests/test_cross_platform.py -v

# Run with coverage
uv run pytest --cov=src

# Specific test file
uv run pytest tests/test_agents.py -v
```

## Platform-Specific Issues

### Linux Desktop Automation

If desktop automation is needed on Linux:

```bash
# Install xdotool and dependencies
sudo apt-get install -y xdotool xclip wmctrl

# Install pyautogui for mouse/keyboard
uv add pyautogui
```

### macOS Code Signing

If you encounter code signing issues:

```bash
# Check signature
codesign -v /usr/local/bin/python3

# Re-sign if needed
codesign -s - /usr/local/bin/python3
```

### Windows Console Encoding

UTF-8 encoding is automatically enabled for Windows console.
If issues occur, manually set:

```powershell
$env:PYTHONIOENCODING="utf-8"
```

## Performance Tips

- **Linux**: Use uvloop (automatically enabled) for 2-3x faster async performance
- **macOS**: Enable GPU acceleration if available
- **Windows**: Disable Windows Defender/antivirus scanning of project directory
- **All**: Use Docker for consistent performance across platforms

## Features by Platform (Quick Reference)

| Feature | Windows | Linux | macOS |
|---------|---------|-------|-------|
| LLM Integration | ✅ | ✅ | ✅ |
| File Operations | ✅ | ✅ | ✅ |
| Docker Execution | ✅ | ✅ | ✅ |
| Browser Automation | ✅ | ✅ | ✅ |
| Screenshots | ✅ | ✅ | ✅ |
| Desktop Automation | ✅ | ⚠️ | ⚠️ |
| Audio Control | ✅ | ⚠️ | ⚠️ |
| Window Management | ✅ | ⚠️ | ⚠️ |
| Native GUI Tools | ✅ | ✗ | ⚠️ |

**Legend:** ✅ = Full support, ⚠️ = Limited/Optional, ✗ = Not available

## Getting Help

- **Check logs**: See `logs/nia.log` for detailed error messages
- **Platform detection**: Run health check to see available features
- **Feature test**: Run `pytest tests/test_cross_platform.py -v`
- **Documentation**: See `PLATFORM_COMPATIBILITY.md` for architecture details
