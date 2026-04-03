# N.I.A Quick Start Guide - Running & Testing

## Prerequisites

✅ **Already Installed**:
- Python 3.11+ (checked)
- All 153 dependencies via `uv sync --all-groups`
- Virtual environment activated

---

## 🚀 Running N.I.A

### 1. Text Mode (Keyboard Input)
```bash
uv run python main.py
```
**Output**:
```
╭──────────────────────────────────────────╮
│  N.I.A. - Neural Intelligence Assistant  │
╰──────────────────────────────────────────╯

👤 You: [type your request]
🤖 NIA: [response from AI]
```

### 2. Voice Mode (with wake words)
```bash
uv run python main.py --voice
```
- Requires microphone
- Listens for wake words: "jarvis", "nia", "hey nia"
- Custom wake words: `--wake-words "alexa,computer,ai"`

### 3. Voice Mode (Always Listening)
```bash
uv run python main.py --voice --no-wake
```
- Starts listening immediately (no wake word needed)

### 4. Debug Mode
```bash
uv run python main.py --debug
# or
DEBUG=true uv run python main.py
```
Shows all internal logs and diagnostic information

### 5. System Status Check
```bash
uv run python main.py --status
```
Displays:
- Python version
- Installed dependencies
- Platform capabilities (xdotool, pycaw, tesseract, etc.)
- GPU support (CUDA)

---

## 🧪 Running Tests

### Run All Tests
```bash
uv run pytest tests/ -v
```

### Run Specific Test Categories

**Cross-Platform Tests** (Desktop automation, system operations):
```bash
uv run pytest tests/test_cross_platform.py -v
```

**Unit Tests** (Individual components):
```bash
uv run pytest tests/unit/ -v
```

**Integration Tests** (Component interactions):
```bash
uv run pytest tests/integration/ -v
```

**E2E Tests** (Full workflow):
```bash
uv run pytest tests/e2e/ -v
```

### Quick Verification

**Check Installation**:
```bash
uv run python -c "from src.core.features import get_features; print(get_features().summary())"
```
Output shows which features are available on your platform.

**Test Core Imports**:
```bash
uv run python -c "import langgraph; import chromadb; import docker; print('✅ Core dependencies OK')"
```

**Test Cross-Platform Detection**:
```bash
uv run python -c "from src.core.os import get_os_context; print(get_os_context())"
```

---

## 📊 Test Suites Explained

### test_cross_platform.py (15+ tests)
Tests all desktop automation features:
- ✅ Screenshot capture (multiple backends)
- ✅ Window management (focus, minimize, maximize, snap)
- ✅ Process management (launch, kill, list)
- ✅ Platform detection
- ✅ Feature availability

**Run it**:
```bash
uv run pytest tests/test_cross_platform.py -v
```

### test_memory.py
Tests ChromaDB vector memory:
- Memory manager initialization
- Namespace creation/retrieval
- Vector storage/search

### test_config.py
Tests configuration system:
- Config loading from YAML/JSON
- Environment variable overrides
- Default values

### test_model_manager.py
Tests LLM provider initialization:
- OpenAI
- Groq
- Ollama
- NVIDIA

### test_phase2_integration.py
Tests Docker Swarm execution:
- Container lifecycle
- Code execution in sandbox
- Output capture

---

## 🔧 Test Flags & Options

```bash
# Show print statements
uv run pytest tests/ -v -s

# Stop on first failure
uv run pytest tests/ -x

# Show slowest 10 tests
uv run pytest tests/ --durations=10

# Run tests matching pattern
uv run pytest tests/ -k "cross_platform" -v

# Run with coverage report
uv run pytest tests/ --cov=src --cov-report=html

# Run specific test function
uv run pytest tests/test_cross_platform.py::test_screenshot -v

# Run tests for specific OS (optional)
uv run pytest tests/test_cross_platform.py -v -m "not xdotool"  # Skip xdotool tests
```

---

## 📋 Expected Output - First Run

### Text Mode
```
╭──────────────────────────────────────────╮
│  N.I.A. - Neural Intelligence Assistant  │
╰──────────────────────────────────────────╯

👤 You: what is 2+2?
🤖 NIA: 2 + 2 = 4. This is basic arithmetic.

👤 You: take a screenshot
🤖 NIA: [Takes screenshot and analyzes it]

👤 You: exit
>> Shutting down N.I.A...
```

### Test Run
```
tests/test_cross_platform.py::test_platform_detection PASSED      [ 8%]
tests/test_cross_platform.py::test_screenshot PASSED              [16%]
tests/test_cross_platform.py::test_window_management PASSED       [24%]
tests/test_cross_platform.py::test_process_list PASSED            [32%]
...
====== 15 passed in 2.34s ======
```

---

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'langgraph'"
```bash
# Reinstall dependencies
uv sync --all-groups
```

### "xdotool not found" (Linux)
```bash
# Install window management tool
sudo apt install xdotool libxdo3
# Then test
uv run python main.py
```

### "ImportError: cannot import name 'pycaw'" (Windows audio)
```bash
# Already installed, but if missing:
pip install pycaw
```

### "Screenshot failed - no backend available"
```bash
# Install fallback: PIL/Pillow
pip install Pillow

# Or install platform-specific:
# Linux: sudo apt install gnome-screenshot scrot
# macOS: already built-in
```

### "pytest: command not found"
```bash
# Install pytest (already in dependencies, but can reinstall)
uv sync --all-groups
```

---

## 🎯 Testing Workflow

```
1. Install & Verify
   └─ uv run python main.py --status

2. Run Quick Tests
   └─ uv run pytest tests/test_cross_platform.py -v

3. Run Full Test Suite
   └─ uv run pytest tests/ -v --tb=short

4. Test Specific Feature
   └─ uv run python -c "from src.capabilities.desktop import screen; print(screen.take_screenshot())"

5. Run with Debug Logs
   └─ uv run pytest tests/test_cross_platform.py -v -s --log-cli-level=DEBUG

6. Run Full System
   └─ uv run python main.py --voice --debug
```

---

## 📈 Performance Check

**Measure startup time**:
```bash
time uv run python main.py --status
```
Expected: < 5 seconds (due to lazy loading)

**Count dependencies loaded**:
```bash
uv run python -c "import sys; from src.agents import nia; print(f'Modules loaded: {len(sys.modules)}')"
```
Expected: < 200 modules on startup (lazy loading works)

---

## ✅ Success Checklist

- [ ] `uv run python main.py --status` shows your platform
- [ ] `uv run pytest tests/test_cross_platform.py -v` passes
- [ ] `uv run python main.py` accepts text input
- [ ] `uv run python main.py --voice` detects microphone (if available)
- [ ] Screenshots work: `uv run python -c "from src.capabilities.desktop import screen; screen.take_screenshot()"`
- [ ] All tests pass: `uv run pytest tests/ -v`

---

## 📚 Next Steps

1. **Test Basic Features**: `uv run python main.py`
2. **Verify Desktop Automation**: `uv run pytest tests/test_cross_platform.py -v`
3. **Check System Status**: `uv run python main.py --status`
4. **Run Full Test Suite**: `uv run pytest tests/ -v`
5. **Try Voice Mode**: `uv run python main.py --voice` (if microphone available)

**System is ready. Execute and validate.** 🚀
