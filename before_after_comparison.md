# N.I.A. Structure: Before & After Comparison

## Visual Comparison

### BEFORE (v3.1.0) - Current Structure

```
N.I.A/
├── core/               ← Infrastructure mixed with business logic
├── nia/                ← Agent at root level
├── tara/               ← Agent at root level
│   └── tools/          ← 50+ tool files deeply nested
├── iris/               ← Agent at root level
├── nola/               ← Agent at root level
├── skills/             ← Empty placeholder (confusion!)
├── plugins/            ← User plugins
├── models/             ← LLM management
├── persona/            ← Personality
├── interface/          ← Incomplete UI
├── config/             ← Scattered JSON configs
│   ├── iris/
│   ├── nia/
│   ├── nola/
│   └── tara/
└── tests/              ← Flat test structure
```

**Problems:**
- 🔴 Agents scattered at root level
- 🔴 Tools vs Skills confusion
- 🔴 Config fragmentation (4 subdirectories)
- 🔴 Tests don't mirror source
- 🔴 No clear layer separation

---

### AFTER (v4.0.0) - Proposed Structure

```
nia/
├── src/                          ← All source code
│   ├── core/                     ← Pure infrastructure
│   ├── agents/                   ← ALL agents together
│   │   ├── nia/
│   │   ├── tara/
│   │   ├── iris/
│   │   └── nola/
│   ├── capabilities/             ← UNIFIED tools/skills
│   │   ├── desktop/
│   │   ├── system/
│   │   ├── web/
│   │   ├── vision/
│   │   ├── memory/
│   │   └── ai/
│   ├── models/
│   ├── persona/
│   ├── interface/
│   │   ├── cli/                  ← Complete CLI
│   │   ├── api/                  ← Future REST API
│   │   └── gui/                  ← Future GUI
│   └── extensions/               ← Extension infrastructure
├── config/                        ← Centralized YAML configs
│   ├── agents/
│   └── capabilities/
├── extensions/                    ← User extensions
├── data/                          ← Runtime data (logs, db, cache)
└── tests/                         ← Mirrors src/
    ├── unit/
    ├── integration/
    └── e2e/
```

**Benefits:**
- ✅ Clear domain separation (src/)
- ✅ All agents in one place
- ✅ Unified capability system
- ✅ Centralized configuration
- ✅ Proper test organization
- ✅ Scalable architecture

---

## Example 1: Tool/Capability Organization

### BEFORE: Window Management

```python
# File: tara/tools/desktop/window_ops.py
# Located deep in TARA's tool directory
# Mixed with other desktop tools

from tara.tools.decorators import tool

@tool(name="focus_window")
def focus_window(window_title: str) -> str:
    """Focus a window by title."""
    # Implementation
    pass

@tool(name="minimize_window")
def minimize_window(window_title: str) -> str:
    """Minimize a window."""
    pass

# ... 5 more window functions
```

**Problems:**
1. Tied to TARA agent (should be reusable)
2. No clear capability documentation
3. Mixed with unrelated tools
4. Hard to find all window operations

---

### AFTER: Window Management Capability

```python
# File: src/capabilities/desktop/windows.py
# Part of desktop capability domain
# Clear, focused module

"""
Window Management Capability

Provides cross-platform window control operations.
See CAPABILITY.md for usage examples.
"""

from nia.src.capabilities.base import BaseCapability, capability
from nia.src.capabilities.desktop.drivers import get_driver

class WindowManagement(BaseCapability):
    """Window control operations."""
    
    def __init__(self):
        super().__init__(domain="desktop", name="window_management")
        self.driver = get_driver()
    
    @capability(
        name="focus_window",
        description="Focus a window by title",
        parameters={"window_title": "str"},
        examples=["focus_window('Chrome')", "focus_window('VS Code')"]
    )
    async def focus(self, window_title: str) -> str:
        """Focus a window by title."""
        return await self.driver.focus_window(window_title)
    
    @capability(
        name="minimize_window",
        description="Minimize a window",
        parameters={"window_title": "str"}
    )
    async def minimize(self, window_title: str) -> str:
        """Minimize a window."""
        return await self.driver.minimize_window(window_title)
    
    # ... other window operations

# Auto-register when imported
__all__ = ['WindowManagement']
```

**File: src/capabilities/desktop/CAPABILITY.md**

```markdown
# Desktop Capability

## Window Management

### Available Operations

- `focus_window(title)` - Bring window to foreground
- `minimize_window(title)` - Minimize window
- `maximize_window(title)` - Maximize window
- `snap_window(title, position)` - Snap window to screen edge
- `close_window(title)` - Close window

### Usage Examples

```python
# In agent code
from nia.src.capabilities.desktop.windows import WindowManagement

wm = WindowManagement()
await wm.focus("Chrome")
await wm.snap("Chrome", "left")
```

### Security

- Requires `desktop.window_management` permission
- Cannot close system windows
- Rate limited: 10 ops/second
```

**Benefits:**
1. ✅ Reusable by any agent
2. ✅ Clear documentation
3. ✅ Focused responsibility
4. ✅ Easy to discover

---

## Example 2: Configuration

### BEFORE: NIA Configuration

```
config/
├── nia/
│   ├── general.json          ← General settings
│   ├── models.json           ← Model configs
│   ├── prompts.json          ← System prompts
│   ├── responses.json        ← Canned responses
│   └── routing.json          ← Routing rules
```

**File: config/nia/general.json**
```json
{
  "name": "NIA",
  "version": "3.1.0",
  "debug_mode": false,
  "log_level": "INFO",
  "max_conversation_length": 50
}
```

**File: config/nia/models.json**
```json
{
  "default_provider": "nvidia",
  "providers": {
    "nvidia": {
      "api_key": "${NVIDIA_API_KEY}",
      "model": "meta/llama-3.1-70b-instruct"
    }
  }
}
```

**Problems:**
1. Config spread across 5 files
2. JSON is verbose and less readable
3. No validation at load time
4. Environment variables awkward
5. Hard to override per-environment

---

### AFTER: Unified Configuration

```
config/
├── base/
│   ├── settings.py           ← Pydantic models
│   └── defaults.yaml         ← Base defaults
├── agents/
│   └── nia.yaml              ← All NIA config
├── capabilities/
│   └── desktop.yaml
└── models.yaml               ← Global model config
```

**File: config/base/settings.py**

```python
"""Configuration schemas using Pydantic."""

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from typing import Literal

class AgentConfig(BaseModel):
    """Base agent configuration."""
    name: str
    version: str
    debug_mode: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    max_conversation_length: int = Field(50, ge=1, le=1000)

class NIAConfig(AgentConfig):
    """NIA supervisor agent configuration."""
    routing_mode: Literal["llm", "rules", "hybrid"] = "hybrid"
    confidence_threshold: float = Field(0.7, ge=0.0, le=1.0)
    enable_memory: bool = True

class ModelProviderConfig(BaseModel):
    """LLM provider configuration."""
    api_key: str
    model: str
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(2000, ge=1)

class AppConfig(BaseSettings):
    """Application configuration with environment support."""
    
    # Loaded from config/agents/nia.yaml + environment
    nia: NIAConfig
    
    # Loaded from config/models.yaml + environment
    model_provider: str = "nvidia"
    models: dict[str, ModelProviderConfig]
    
    class Config:
        env_prefix = "NIA_"
        env_nested_delimiter = "__"
        
        @classmethod
        def settings_customise_sources(cls, *args, **kwargs):
            # Load YAML configs first, then env vars
            pass
```

**File: config/agents/nia.yaml**

```yaml
# NIA Supervisor Agent Configuration

name: NIA
version: 4.0.0
debug_mode: false
log_level: INFO
max_conversation_length: 50

# Routing configuration
routing_mode: hybrid
confidence_threshold: 0.7
enable_memory: true

# Gatekeeper settings
gatekeeper:
  enabled: true
  fallback_agent: chat
  
# LangGraph settings
graph:
  max_iterations: 10
  timeout_seconds: 30
```

**File: config/models.yaml**

```yaml
# LLM Model Configurations

default_provider: nvidia

providers:
  nvidia:
    api_key: ${NVIDIA_API_KEY}  # From environment
    model: meta/llama-3.1-70b-instruct
    temperature: 0.7
    max_tokens: 2000
  
  openai:
    api_key: ${OPENAI_API_KEY}
    model: gpt-4
    temperature: 0.7
    max_tokens: 2000
  
  ollama:
    api_key: null
    model: llama3.1
    temperature: 0.7
    max_tokens: 2000
    base_url: http://localhost:11434

# Fallback chain
fallback_chain:
  - nvidia
  - openai
  - ollama
```

**File: config/loader.py**

```python
"""Configuration loader with validation."""

from pathlib import Path
import yaml
from pydantic import ValidationError
from nia.src.core.logger import get_logger
from nia.src.config.base.settings import AppConfig

logger = get_logger(__name__)

def load_config() -> AppConfig:
    """Load and validate application configuration."""
    
    config_dir = Path("config")
    
    # Load all YAML files
    configs = {}
    for yaml_file in config_dir.rglob("*.yaml"):
        with open(yaml_file) as f:
            data = yaml.safe_load(f)
            configs[yaml_file.stem] = data
    
    try:
        # Validate with Pydantic
        app_config = AppConfig(**configs)
        logger.info("✓ Configuration loaded and validated")
        return app_config
    
    except ValidationError as e:
        logger.error(f"Configuration validation failed: {e}")
        raise

# Singleton instance
_config: AppConfig | None = None

def get_config() -> AppConfig:
    """Get configuration singleton."""
    global _config
    if _config is None:
        _config = load_config()
    return _config
```

**Benefits:**
1. ✅ Single file per agent
2. ✅ YAML is human-friendly
3. ✅ Validation at startup (Pydantic)
4. ✅ Environment variable support
5. ✅ Type-safe access
6. ✅ Easy to override (dev/prod/test)

**Usage:**

```python
from nia.src.config.loader import get_config

config = get_config()
print(config.nia.name)  # "NIA"
print(config.nia.log_level)  # "INFO"
print(config.models["nvidia"].model)  # "meta/llama-3.1-70b-instruct"
```

---

## Example 3: Agent Organization

### BEFORE: Agents at Root

```
N.I.A/
├── nia/              ← Supervisor agent
│   ├── agent.py
│   ├── gatekeeper.py
│   └── graph/
├── tara/             ← Tool agent
│   ├── graph/
│   └── tools/
├── iris/             ← Vision agent
│   ├── agent.py
│   └── sentry.py
└── nola/             ← Voice agent
    └── manager.py
```

**Problem:** Hard to see relationship between agents

---

### AFTER: Agents Together

```
src/
├── agents/
│   ├── base.py              ← Shared agent interface
│   ├── nia/                 ← Supervisor
│   │   ├── agent.py
│   │   ├── gatekeeper.py
│   │   └── graph/
│   ├── tara/                ← Tool executor
│   │   ├── agent.py
│   │   ├── security.py
│   │   └── graph/
│   ├── iris/                ← Vision
│   │   ├── agent.py
│   │   └── sentry.py
│   └── nola/                ← Voice
│       ├── manager.py
│       └── io/
```

**File: src/agents/base.py**

```python
"""Base agent interface."""

from abc import ABC, abstractmethod
from typing import Any
from nia.src.core.events import EventBus

class BaseAgent(ABC):
    """Base class for all NIA agents."""
    
    def __init__(self, name: str, event_bus: EventBus):
        self.name = name
        self.event_bus = event_bus
        self._initialized = False
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the agent."""
        pass
    
    @abstractmethod
    async def process(self, input: str, context: dict[str, Any]) -> str:
        """Process input and return response."""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Clean shutdown."""
        pass
    
    async def emit_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Emit an event to the event bus."""
        await self.event_bus.emit(f"agent.{self.name}.{event_type}", data)
```

**File: src/agents/nia/agent.py**

```python
"""NIA Supervisor Agent."""

from nia.src.agents.base import BaseAgent
from nia.src.agents.nia.gatekeeper import RoutingGatekeeper
from nia.src.agents.nia.graph.builder import build_graph

class SupervisorAgent(BaseAgent):
    """Routes queries to specialized agents."""
    
    def __init__(self, event_bus, config):
        super().__init__("nia", event_bus)
        self.config = config
        self.gatekeeper = RoutingGatekeeper(config.gatekeeper)
        self.graph = build_graph(config.graph)
    
    async def initialize(self) -> None:
        """Initialize supervisor."""
        await self.emit_event("initializing", {})
        # Setup logic
        self._initialized = True
        await self.emit_event("ready", {})
    
    async def process(self, input: str, context: dict) -> str:
        """Route input to appropriate agent."""
        if not self._initialized:
            raise RuntimeError("Agent not initialized")
        
        # Run LangGraph workflow
        result = await self.graph.ainvoke({
            "input": input,
            "context": context
        })
        
        return result["output"]
    
    async def shutdown(self) -> None:
        """Shutdown supervisor."""
        await self.emit_event("shutting_down", {})
        self._initialized = False
```

**Benefits:**
1. ✅ All agents follow same interface
2. ✅ Easy to add new agents
3. ✅ Clear agent discovery
4. ✅ Better code reuse

---

## Example 4: Testing

### BEFORE: Flat Test Structure

```
tests/
├── test_config.py
├── test_memory.py
├── test_model_manager.py
├── test_parser.py
├── test_safe_llm_fallback.py
├── test_tara.py
└── ...
```

**Problem:** 
- Hard to find tests for a module
- No separation of test types
- Doesn't scale well

---

### AFTER: Structured Tests

```
tests/
├── conftest.py                    ← Shared fixtures
├── unit/                          ← Fast, isolated tests
│   ├── core/
│   │   ├── test_config.py
│   │   ├── test_events.py
│   │   └── test_memory.py
│   ├── agents/
│   │   ├── nia/
│   │   │   └── test_gatekeeper.py
│   │   └── tara/
│   │       └── test_security.py
│   └── capabilities/
│       ├── desktop/
│       │   └── test_windows.py
│       └── system/
│           └── test_files.py
├── integration/                   ← Multi-component tests
│   ├── test_agent_flow.py
│   └── test_capability_execution.py
└── e2e/                          ← Full system tests
    └── test_user_scenarios.py
```

**File: tests/unit/capabilities/desktop/test_windows.py**

```python
"""Unit tests for window management capability."""

import pytest
from unittest.mock import Mock, AsyncMock
from nia.src.capabilities.desktop.windows import WindowManagement

@pytest.fixture
def window_mgmt():
    """Create WindowManagement instance with mocked driver."""
    wm = WindowManagement()
    wm.driver = Mock()
    wm.driver.focus_window = AsyncMock(return_value="Window focused")
    return wm

@pytest.mark.asyncio
async def test_focus_window(window_mgmt):
    """Test focusing a window."""
    result = await window_mgmt.focus("Chrome")
    
    assert result == "Window focused"
    window_mgmt.driver.focus_window.assert_called_once_with("Chrome")

@pytest.mark.asyncio
async def test_focus_nonexistent_window(window_mgmt):
    """Test focusing a non-existent window."""
    window_mgmt.driver.focus_window = AsyncMock(
        side_effect=ValueError("Window not found")
    )
    
    with pytest.raises(ValueError, match="Window not found"):
        await window_mgmt.focus("NonExistent")
```

**File: tests/integration/test_agent_flow.py**

```python
"""Integration tests for agent workflow."""

import pytest
from nia.src.core.engine.orchestrator import NIAAssistant
from nia.src.config.loader import get_config

@pytest.fixture
async def assistant():
    """Create and initialize NIA assistant."""
    config = get_config()
    assistant = NIAAssistant(config)
    await assistant.initialize()
    yield assistant
    await assistant.shutdown()

@pytest.mark.asyncio
@pytest.mark.integration
async def test_simple_query_routing(assistant):
    """Test that simple query routes to chat agent."""
    response = await assistant.process("Hello, how are you?")
    
    assert response is not None
    assert len(response) > 0
    # Verify it went through chat, not TARA
    assert not any(tool in response.lower() for tool in ["executed", "launched"])

@pytest.mark.asyncio
@pytest.mark.integration
async def test_tool_query_routing(assistant):
    """Test that tool query routes to TARA."""
    response = await assistant.process("Open Chrome browser")
    
    assert response is not None
    # Should mention tool execution
    assert any(word in response.lower() for word in ["opened", "launched", "started"])
```

**File: tests/e2e/test_user_scenarios.py**

```python
"""End-to-end user scenario tests."""

import pytest
from nia.src.core.engine.orchestrator import NIAAssistant
from nia.src.config.loader import get_config

@pytest.mark.e2e
@pytest.mark.asyncio
async def test_complete_user_workflow():
    """Test complete workflow: voice → routing → tool → response."""
    
    # Setup
    config = get_config()
    assistant = NIAAssistant(config)
    await assistant.initialize()
    
    try:
        # Step 1: User asks to open browser
        response1 = await assistant.process("Open Chrome and go to google.com")
        assert "chrome" in response1.lower()
        
        # Step 2: User asks about what's on screen
        response2 = await assistant.process("What's on my screen?")
        assert response2  # IRIS should respond
        
        # Step 3: User asks to close browser
        response3 = await assistant.process("Close Chrome")
        assert "closed" in response3.lower()
        
    finally:
        await assistant.shutdown()
```

**Benefits:**
1. ✅ Tests easy to find
2. ✅ Clear test types (unit/integration/e2e)
3. ✅ Can run subsets (pytest tests/unit/)
4. ✅ Scales with codebase

---

## Import Path Comparison

### BEFORE

```python
# Importing various components

from core.logger import setup_logger
from core.memory import MemoryManager
from nia.agent import SupervisorAgent
from tara.tools.desktop.window_ops import focus_window
from iris.agent import IrisAgent
from models.model_manager import ModelManager
```

**Problems:**
- Inconsistent depth
- No namespace
- Unclear what's a package

---

### AFTER

```python
# Importing various components

from nia.src.core.logger import get_logger
from nia.src.core.memory import MemoryManager
from nia.src.agents.nia.agent import SupervisorAgent
from nia.src.capabilities.desktop.windows import WindowManagement
from nia.src.agents.iris.agent import IrisAgent
from nia.src.models.manager import ModelManager
```

**Benefits:**
- ✅ Consistent depth
- ✅ Clear namespace (nia.src.*)
- ✅ Obvious what's what
- ✅ No ambiguity

---

## Directory Size Comparison

### BEFORE

```
N.I.A/
├── core/           (10 files)
├── nia/            (7 files)
├── tara/           (60+ files across tools/)
├── iris/           (4 files)
├── nola/           (4 files)
├── skills/         (9 empty directories)
├── plugins/        (3 files)
├── models/         (3 files)
├── config/         (15 files across 4 subdirs)
└── tests/          (8 files)

Total Python files: ~120
Total directories: ~30
```

---

### AFTER

```
nia/
├── src/
│   ├── core/           (10 files) ✓
│   ├── agents/         (20 files)
│   │   ├── nia/        (7 files)
│   │   ├── tara/       (6 files)
│   │   ├── iris/       (4 files)
│   │   └── nola/       (4 files)
│   ├── capabilities/   (15 files) ← Consolidated
│   │   ├── desktop/    (6 files)
│   │   ├── system/     (4 files)
│   │   ├── web/        (1 file)
│   │   ├── vision/     (1 file)
│   │   ├── memory/     (1 file)
│   │   └── ai/         (1 file)
│   ├── models/         (3 files) ✓
│   ├── persona/        (2 files) ✓
│   ├── interface/      (8 files) ← Expanded
│   └── extensions/     (5 files)
├── config/             (8 YAML files) ← Consolidated
├── extensions/         (user space)
├── data/               (runtime data)
└── tests/              (30 files) ← Expanded
    ├── unit/           (20 files)
    ├── integration/    (5 files)
    └── e2e/            (5 files)

Total Python files: ~120 (same)
Total directories: ~25 (fewer, better organized)
```

**Benefits:**
- ✅ Same number of files
- ✅ Fewer directories
- ✅ Better organization
- ✅ Clearer responsibility

---

## Decision Matrix: When to Use Each Approach

| Scenario | v3.1.0 (Current) | v4.0.0 (Proposed) |
|----------|------------------|-------------------|
| **Adding new tool** | Add file to `tara/tools/` | Add method to capability class |
| **Adding new agent** | Create root-level package | Create package in `src/agents/` |
| **Adding new config** | Add JSON to agent subdir | Add section to agent YAML |
| **Finding a test** | Search flat list | Navigate to `tests/{type}/{module}/` |
| **Importing a tool** | `from tara.tools.X import Y` | `from nia.src.capabilities.X import Y` |
| **Adding plugin** | Drop in `plugins/` | Drop in `extensions/custom/` |
| **Understanding flow** | Follow scattered files | Clear layer: core → agents → capabilities |

---

## Summary: Why This Matters

### Current Pain Points (v3.1.0)
1. "Where do I add a new tool?" → Unclear (tools vs skills)
2. "Where's the config for X?" → Scattered across subdirs
3. "How do I test Y?" → Hard to find tests
4. "What's the architecture?" → Implicit, not clear

### After Migration (v4.0.0)
1. "Where do I add a new capability?" → `src/capabilities/{domain}/`
2. "Where's the config for X?" → `config/{agents|capabilities}/{name}.yaml`
3. "How do I test Y?" → `tests/{unit|integration|e2e}/{module}/`
4. "What's the architecture?" → Explicit layers: core → agents → capabilities

---

**The goal isn't to change for change's sake—it's to make the codebase:**
- **Understandable** in 5 minutes
- **Navigable** by new developers
- **Scalable** to 10x the features
- **Maintainable** for years to come

---

**Next Steps:**
1. Review this comparison
2. Decide on migration approach (big bang vs gradual)
3. Start with lowest-risk phase (config consolidation)
4. Measure success (developer experience surveys, onboarding time)
