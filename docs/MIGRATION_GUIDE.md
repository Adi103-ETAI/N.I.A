# Migrating from N.I.A. v3.1.0 to v4.0.0

This guide helps you migrate from v3.1.0 to the new v4.0.0 architecture.

---

## Quick Reference

| What Changed | Before (v3.1.0) | After (v4.0.0) |
|--------------|-----------------|----------------|
| Source location | `core/`, `nia/`, `tara/`, `iris/`, `nola/` | `src/core/`, `src/agents/*` |
| Tools | `tara/tools/` | `src/capabilities/` |
| Config format | JSON files | YAML files |
| Config location | `config/nia/`, `config/tara/` | `config/agents/` |
| Plugins | `plugins/` | `extensions/custom/` |

---

## 1. Configuration Changes

### Before: Scattered JSON

```
config/
├── nia/
│   ├── general.json
│   ├── models.json
│   └── prompts.json
├── tara/
│   ├── apps.json
│   └── commands.json
└── iris/
    └── sentry.json
```

### After: Unified YAML

```
config/
├── agents/
│   ├── nia.yaml
│   ├── tara.yaml
│   ├── iris.yaml
│   └── nola.yaml
├── capabilities/
│   └── desktop.yaml
└── models.yaml
```

### Example Migration

**Before** (`config/nia/general.json`):
```json
{
  "name": "NIA",
  "version": "3.1.0",
  "debug_mode": false
}
```

**After** (`config/agents/nia.yaml`):
```yaml
name: NIA
version: 4.0.0
debug_mode: false
log_level: INFO

routing_mode: hybrid
confidence_threshold: 0.7
```

---

## 2. Import Path Changes

### Core Imports

```python
# Before
from core.logger import setup_logger
from core.memory import MemoryManager
from core.services import ServiceRegistry

# After
from src.core.logger import get_logger
from src.core.memory import MemoryManager
from src.core.registry import ServiceRegistry
```

### Agent Imports

```python
# Before
from nia.agent import SupervisorAgent
from tara.graph.workflow import create_workflow
from iris.agent import IrisAgent

# After
from src.agents.nia.agent import SupervisorAgent
from src.agents.tara.graph.workflow import create_workflow
from src.agents.iris.agent import IrisAgent
```

### Tool → Capability Imports

```python
# Before
from tara.tools.desktop.app_launcher import launch_app
from tara.tools.desktop.window_ops import focus_window
from tara.tools.system.file_ops import read_file

# After
from src.capabilities.desktop.apps import launch_app
from src.capabilities.desktop.windows import focus_window
from src.capabilities.system.files import read_file
```

---

## 3. Plugin → Extension Migration

### Option A: Keep Legacy Plugin (Automatic Compatibility)

Your v3.1.0 plugins work without changes:

```python
# plugins/my_plugin.py - STILL WORKS
from tara.tools.decorators import tool

@tool(name="my_tool")
def my_tool():
    return "Works via compatibility layer!"
```

You'll see a deprecation warning, but it works.

### Option B: Update to v4.0.0 Extensions (Recommended)

```python
# extensions/custom/my_extension.py
from src.extensions.base import BaseExtension
from src.capabilities.decorators import capability

class MyExtension(BaseExtension):
    """My custom extension for v4.0.0."""
    
    def initialize(self):
        """Called when extension loads."""
        pass
    
    def cleanup(self):
        """Called on shutdown."""
        pass

# Register capabilities
@capability(
    name="my_custom_tool",
    description="Does something custom"
)
def my_custom_tool(param: str) -> str:
    return f"Result: {param}"
```

---

## 4. Import Mapping Reference

| v3.1.0 Import | v4.0.0 Import |
|---------------|---------------|
| `from core.logger` | `from src.core.logger` |
| `from core.memory` | `from src.core.memory` |
| `from core.services` | `from src.core.registry` |
| `from core.event_bus` | `from src.core.events` |
| `from core.os_context` | `from src.core.platform` |
| `from nia.agent` | `from src.agents.nia.agent` |
| `from tara.graph` | `from src.agents.tara.graph` |
| `from iris.agent` | `from src.agents.iris.agent` |
| `from nola.manager` | `from src.agents.nola.manager` |
| `from tara.tools.decorators` | `from src.capabilities.decorators` |
| `from tara.tools.desktop.*` | `from src.capabilities.desktop.*` |
| `from tara.tools.system.*` | `from src.capabilities.system.*` |
| `from tara.tools.web.*` | `from src.capabilities.web.*` |
| `from models.model_manager` | `from src.models.manager` |
| `from persona.profile` | `from src.persona.profile` |

---

## 5. Breaking Changes

### Removed Files
- `core/container.py` — Use `src/core/registry.py`
- `tara/tools/base.py` — Use `src/capabilities/decorators.py`
- `plugins/loader.py` — Use `src/extensions/loader.py`

### Renamed Functions
- `setup_logger()` → `get_logger()`
- `get_service()` → `ServiceRegistry.get()`
- `@tool` decorator → `@capability` decorator

### Config Loading
```python
# Before
from core.config import load_config
config = load_config("nia/general.json")

# After
from config.loader import get_settings
settings = get_settings()
print(settings.nia.version)  # Type-safe access
```

---

## 6. Verification Checklist

After migration, verify:

- [ ] `python main.py --version` shows 4.0.0
- [ ] All custom plugins/extensions load without errors
- [ ] Configuration loads from YAML files
- [ ] Tests pass: `pytest tests/`

---

## Need Help?

- Check `config/agents/` for YAML config examples
- See `src/extensions/compat/` for compatibility layer code
- Review `src/capabilities/` for new capability structure
