# N.I.A. v3.1.0 → v4.0.0 Restructuring Plan
## From Functional to Clean Architecture

---

## Executive Summary

**Goal**: Transform N.I.A. from a functional but scattered codebase into a clean, scalable architecture following Domain-Driven Design and SOLID principles.

**Key Improvements**:
- ✅ Clear separation of domains (agents, capabilities, infrastructure)
- ✅ Unified plugin/skill architecture
- ✅ Centralized configuration management
- ✅ Proper test organization mirroring source structure
- ✅ Improved dependency management
- ✅ Future-proof extensibility

---

## Proposed Structure (v4.0.0)

```
nia/                                    # Root package (lowercase, PEP 8)
├── 📁 .agent/                          # AI Agent Development Rules
│   └── rules/                          # (Keep as-is)
│
├── 📁 config/                          # UNIFIED Configuration Hub
│   ├── base/                           # Base configurations
│   │   ├── __init__.py
│   │   ├── settings.py                 # Pydantic settings models
│   │   └── defaults.yaml               # Default values
│   ├── agents/                         # Agent-specific configs
│   │   ├── nia.yaml                    # NIA supervisor config
│   │   ├── tara.yaml                   # TARA tool config
│   │   ├── iris.yaml                   # IRIS vision config
│   │   └── nola.yaml                   # NOLA voice config
│   ├── capabilities/                   # Capability configs
│   │   ├── desktop.yaml                # Desktop automation
│   │   ├── web.yaml                    # Web automation
│   │   └── vision.yaml                 # Vision triggers
│   ├── models.yaml                     # LLM model definitions
│   ├── apps.yaml                       # Application registry
│   └── loader.py                       # Config loader with validation
│
├── 📁 src/                             # Source Code (Clean separation)
│   ├── __init__.py
│   │
│   ├── 📁 core/                        # Core Infrastructure (Unchanged mostly)
│   │   ├── __init__.py
│   │   ├── config.py                   # Settings manager
│   │   ├── events.py                   # Event bus (renamed from event_bus.py)
│   │   ├── health.py
│   │   ├── logger.py
│   │   ├── memory.py
│   │   ├── registry.py                 # Service registry (renamed from services.py)
│   │   ├── platform.py                 # OS context (renamed from os_context.py)
│   │   └── engine/
│   │       ├── __init__.py
│   │       ├── orchestrator.py         # NIAAssistant (renamed from system.py)
│   │       └── commands.py
│   │
│   ├── 📁 agents/                      # ALL Agents in one place
│   │   ├── __init__.py
│   │   ├── base.py                     # BaseAgent abstract class
│   │   │
│   │   ├── 📁 nia/                     # Supervisor Agent
│   │   │   ├── __init__.py
│   │   │   ├── agent.py
│   │   │   ├── gatekeeper.py
│   │   │   ├── state.py
│   │   │   └── graph/
│   │   │       ├── __init__.py
│   │   │       ├── builder.py
│   │   │       └── nodes.py
│   │   │
│   │   ├── 📁 tara/                    # Tool Execution Agent
│   │   │   ├── __init__.py
│   │   │   ├── agent.py                # TARA agent interface
│   │   │   ├── protocols.py
│   │   │   ├── security.py             # Warden moved here
│   │   │   └── graph/
│   │   │       ├── __init__.py
│   │   │       ├── workflow.py
│   │   │       ├── nodes.py
│   │   │       ├── prompts.py
│   │   │       └── state.py
│   │   │
│   │   ├── 📁 iris/                    # Vision Agent
│   │   │   ├── __init__.py
│   │   │   ├── agent.py
│   │   │   ├── sentry.py
│   │   │   └── capture.py              # Screen capture tools (renamed from tools.py)
│   │   │
│   │   └── 📁 nola/                    # Voice Agent
│   │       ├── __init__.py
│   │       ├── manager.py
│   │       ├── security.py
│   │       └── io/
│   │           ├── __init__.py
│   │           ├── tts.py              # Text-to-speech
│   │           └── stt.py              # Speech-to-text
│   │
│   ├── 📁 capabilities/                # UNIFIED Capability System (Tools + Skills)
│   │   ├── __init__.py
│   │   ├── base.py                     # BaseCapability class
│   │   ├── decorators.py               # @capability decorator
│   │   ├── loader.py                   # Dynamic capability discovery
│   │   ├── registry.py                 # Capability registry
│   │   │
│   │   ├── 📁 desktop/                 # Desktop Automation
│   │   │   ├── __init__.py
│   │   │   ├── CAPABILITY.md           # Documentation
│   │   │   ├── apps.py                 # launch_app, kill_app, list_processes
│   │   │   ├── windows.py              # Window management
│   │   │   ├── input.py                # Mouse/keyboard control
│   │   │   ├── screen.py               # Screenshots, screen info
│   │   │   ├── uia.py                  # UI automation
│   │   │   └── drivers/                # Platform drivers
│   │   │       ├── __init__.py
│   │   │       ├── base.py
│   │   │       ├── factory.py
│   │   │       ├── universal.py
│   │   │       └── windows.py
│   │   │
│   │   ├── 📁 system/                  # System Operations
│   │   │   ├── __init__.py
│   │   │   ├── CAPABILITY.md
│   │   │   ├── files.py                # File operations (3-tier security)
│   │   │   ├── clipboard.py            # Clipboard operations
│   │   │   ├── stats.py                # System stats, battery
│   │   │   └── processes.py            # Process management
│   │   │
│   │   ├── 📁 web/                     # Web Automation
│   │   │   ├── __init__.py
│   │   │   ├── CAPABILITY.md
│   │   │   └── browser.py              # Playwright automation
│   │   │
│   │   ├── 📁 vision/                  # Vision Capabilities
│   │   │   ├── __init__.py
│   │   │   ├── CAPABILITY.md
│   │   │   ├── analysis.py             # Image analysis
│   │   │   └── ocr.py                  # Text extraction
│   │   │
│   │   ├── 📁 memory/                  # Memory Capabilities
│   │   │   ├── __init__.py
│   │   │   ├── CAPABILITY.md
│   │   │   └── preferences.py          # User preferences
│   │   │
│   │   └── 📁 ai/                      # AI Operations
│   │       ├── __init__.py
│   │       ├── CAPABILITY.md
│   │       └── llm.py                  # LLM operations
│   │
│   ├── 📁 models/                      # LLM Management (Unchanged)
│   │   ├── __init__.py
│   │   ├── manager.py                  # ModelManager
│   │   └── safe_llm.py                 # SafeLLM wrapper
│   │
│   ├── 📁 persona/                     # Identity & Personality (Unchanged)
│   │   ├── __init__.py
│   │   └── profile.py
│   │
│   ├── 📁 interface/                   # User Interfaces
│   │   ├── __init__.py
│   │   ├── banner.py
│   │   ├── 📁 cli/                     # Command Line Interface
│   │   │   ├── __init__.py
│   │   │   ├── app.py                  # CLI application
│   │   │   ├── commands/               # CLI commands
│   │   │   │   ├── __init__.py
│   │   │   │   ├── chat.py
│   │   │   │   ├── config.py
│   │   │   │   └── tools.py
│   │   │   └── prompts.py              # Interactive prompts
│   │   ├── 📁 api/                     # REST API (Future)
│   │   │   ├── __init__.py
│   │   │   ├── app.py
│   │   │   └── routes/
│   │   └── 📁 gui/                     # Graphical UI (Future)
│   │       └── __init__.py
│   │
│   └── 📁 extensions/                  # Extension System (Unified Plugins)
│       ├── __init__.py
│       ├── base.py                     # BaseExtension class
│       ├── loader.py                   # Extension loader
│       ├── watcher.py                  # Hot-reload watcher
│       ├── registry.py                 # Extension registry
│       └── examples/
│           ├── __init__.py
│           ├── example_capability.py   # Example capability extension
│           └── example_agent.py        # Example agent extension
│
├── 📁 extensions/                      # User Extensions (Hot-loadable)
│   ├── README.md                       # User guide
│   └── custom/                         # User's custom extensions
│       └── .gitkeep
│
├── 📁 data/                            # Runtime Data
│   ├── .gitignore                      # Ignore db files
│   ├── memory/                         # Memory databases
│   │   ├── chroma/                     # Vector store
│   │   └── sqlite/                     # Relational data
│   ├── cache/                          # Cache files
│   └── logs/                           # Log files (moved from logs/)
│       └── nia.log
│
├── 📁 tests/                           # Test Suite (Mirrors src/)
│   ├── __init__.py
│   ├── conftest.py                     # Shared fixtures
│   ├── 📁 unit/                        # Unit tests
│   │   ├── __init__.py
│   │   ├── core/
│   │   │   ├── test_config.py
│   │   │   ├── test_events.py
│   │   │   ├── test_memory.py
│   │   │   └── test_registry.py
│   │   ├── agents/
│   │   │   ├── nia/
│   │   │   │   └── test_gatekeeper.py
│   │   │   └── tara/
│   │   │       └── test_security.py
│   │   ├── capabilities/
│   │   │   ├── test_desktop.py
│   │   │   └── test_system.py
│   │   └── models/
│   │       ├── test_manager.py
│   │       └── test_safe_llm.py
│   ├── 📁 integration/                 # Integration tests
│   │   ├── __init__.py
│   │   ├── test_agent_flow.py
│   │   └── test_capability_execution.py
│   └── 📁 e2e/                         # End-to-end tests
│       ├── __init__.py
│       └── test_user_scenarios.py
│
├── 📁 scripts/                         # Development & Ops Scripts
│   ├── dev/                            # Development tools
│   │   ├── verify_config.py
│   │   ├── verify_hot_swap.py
│   │   └── verify_shadow_config.py
│   ├── ops/                            # Operations tools
│   │   ├── reset_brain.py
│   │   └── backup_memory.py
│   └── setup/                          # Setup scripts
│       ├── install_dependencies.py
│       └── setup_environment.py
│
├── 📁 docs/                            # Documentation
│   ├── architecture/
│   │   ├── overview.md
│   │   ├── agents.md
│   │   └── capabilities.md
│   ├── guides/
│   │   ├── getting_started.md
│   │   ├── creating_extensions.md
│   │   └── configuration.md
│   └── api/
│       └── reference.md
│
├── 📁 sounds/                          # Audio Assets
│   └── .gitkeep
│
├── .env.example                        # Environment template
├── .gitignore
├── pyproject.toml                      # Project metadata (Poetry/modern)
├── requirements.txt                    # Dependencies (fallback)
├── README.md
├── LICENSE
└── main.py                             # Entry point
```

---

## Key Architectural Changes

### 1. **Unified Capability System** (Biggest Change)

**Before:**
- Tools in `tara/tools/` (50+ files)
- Empty skill placeholders in `skills/`
- Confusion about tools vs skills

**After:**
- Single `src/capabilities/` directory
- Each capability is a domain module (desktop, system, web, etc.)
- Tools and skills are the same thing: **capabilities**
- Clear CAPABILITY.md documentation per domain

**Benefits:**
- No duplication between tools/skills
- Easy to add new capability domains
- Clear responsibility boundaries
- Better for testing and documentation

---

### 2. **Centralized Configuration**

**Before:**
```
config/
├── apps.json
├── iris/
│   ├── prompt.txt
│   ├── sentry.json
│   └── triggers.json
├── nia/
│   ├── general.json
│   ├── models.json
│   ├── prompts.json
│   ├── responses.json
│   └── routing.json
├── nola/
│   └── voice.json
└── tara/
    ├── apps.json
    ├── commands.json
    ├── help.txt
    ├── plugins.json
    └── uia.json
```

**After:**
```
config/
├── base/
│   ├── settings.py      # Pydantic models
│   └── defaults.yaml    # Default values
├── agents/
│   ├── nia.yaml
│   ├── tara.yaml
│   ├── iris.yaml
│   └── nola.yaml
├── capabilities/
│   ├── desktop.yaml
│   ├── web.yaml
│   └── vision.yaml
├── models.yaml
├── apps.yaml
└── loader.py
```

**Benefits:**
- Single YAML format (more readable than JSON)
- Logical grouping (agents vs capabilities)
- Type-safe with Pydantic models
- Environment-based overrides (.env)
- Validation at startup

---

### 3. **Proper Source Organization**

**Before:**
- Mix of agent code at root level (`nia/`, `tara/`, etc.)
- Unclear what's core vs agent-specific

**After:**
- All code in `src/`
- Clear domains: `core/`, `agents/`, `capabilities/`
- Infrastructure separate from business logic

**Benefits:**
- Easier imports (`from nia.src.agents.tara import ...`)
- Clear dependency flow (core → agents → capabilities)
- Better IDE navigation

---

### 4. **Test Structure Mirrors Source**

**Before:**
```
tests/
├── test_config.py
├── test_memory.py
├── test_model_manager.py
├── test_tara.py
└── ...
```

**After:**
```
tests/
├── unit/
│   ├── core/
│   ├── agents/
│   └── capabilities/
├── integration/
└── e2e/
```

**Benefits:**
- Easy to find tests for any module
- Separation of unit/integration/e2e tests
- Scales as codebase grows

---

### 5. **Unified Extension System**

**Before:**
- `plugins/` directory
- `tara/plugin_system/` directory
- Confusion about where plugins go

**After:**
- Single `src/extensions/` infrastructure
- User extensions in `extensions/` (root level)
- Support for both capability and agent extensions

**Benefits:**
- One extension API
- Hot-reload for all extension types
- Clear user vs system boundary

---

## Migration Guide

### Phase 1: Preparation (Week 1)

1. **Create Feature Branch**
   ```bash
   git checkout -b refactor/v4-architecture
   ```

2. **Backup Critical Data**
   ```bash
   python scripts/ops/backup_memory.py
   ```

3. **Document Current Behavior**
   - Run full test suite
   - Document any custom configurations
   - Note any user extensions

### Phase 2: Configuration Migration (Week 1)

1. **Convert JSON to YAML**
   ```bash
   python scripts/migration/convert_configs.py
   ```

2. **Create Pydantic Models**
   - Define settings classes
   - Add validation rules
   - Test with existing configs

3. **Update Config Loader**
   - Implement new loader.py
   - Support legacy formats temporarily
   - Add deprecation warnings

### Phase 3: Move Source Files (Week 2)

1. **Create New Structure**
   ```bash
   mkdir -p src/{core,agents,capabilities,models,persona,interface,extensions}
   ```

2. **Move Core**
   ```bash
   mv core/* src/core/
   # Rename files
   mv src/core/event_bus.py src/core/events.py
   mv src/core/os_context.py src/core/platform.py
   mv src/core/services.py src/core/registry.py
   mv src/core/engine/system.py src/core/engine/orchestrator.py
   ```

3. **Move Agents**
   ```bash
   mkdir -p src/agents/{nia,tara,iris,nola}
   mv nia/* src/agents/nia/
   mv tara/* src/agents/tara/ (except tools/)
   mv iris/* src/agents/iris/
   mv nola/* src/agents/nola/
   
   # Move security
   mv tara/security/warden.py src/agents/tara/security.py
   ```

4. **Consolidate Capabilities**
   ```bash
   # This is the big one - merge tara/tools/ and skills/
   python scripts/migration/merge_capabilities.py
   ```

### Phase 4: Update Imports (Week 2-3)

1. **Update All Imports**
   - Use find/replace for common patterns
   - Update __init__.py files
   - Fix circular dependencies

2. **Update Tests**
   - Reorganize test files
   - Update test imports
   - Ensure all tests pass

### Phase 5: Extension System (Week 3)

1. **Implement New Extension API**
   ```python
   # src/extensions/base.py
   class BaseExtension(ABC):
       @abstractmethod
       def initialize(self) -> None: ...
       
       @abstractmethod
       def cleanup(self) -> None: ...
   ```

2. **Migrate Existing Plugins**
   - Update plugin format
   - Test hot-reload
   - Update documentation

### Phase 6: Documentation & Cleanup (Week 4)

1. **Update Documentation**
   - New architecture diagrams
   - Updated API reference
   - Migration guide for users

2. **Remove Old Files**
   - Delete legacy directories
   - Clean up deprecated code
   - Update .gitignore

3. **Final Testing**
   - Full regression test
   - Performance testing
   - User acceptance testing

---

## Detailed Migration Scripts

### Script 1: Config Converter

```python
# scripts/migration/convert_configs.py
import json
import yaml
from pathlib import Path

def convert_json_to_yaml(json_path: Path, yaml_path: Path):
    """Convert JSON config to YAML."""
    with open(json_path) as f:
        data = json.load(f)
    
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

def main():
    config_dir = Path("config")
    
    # Convert agent configs
    mappings = {
        "nia/general.json": "agents/nia.yaml",
        "nia/models.json": "models.yaml",
        "tara/apps.json": "capabilities/desktop.yaml",
        # ... more mappings
    }
    
    for old_path, new_path in mappings.items():
        convert_json_to_yaml(
            config_dir / old_path,
            config_dir / new_path
        )

if __name__ == "__main__":
    main()
```

### Script 2: Capability Merger

```python
# scripts/migration/merge_capabilities.py
from pathlib import Path
import shutil

def merge_capabilities():
    """Merge tara/tools/ and skills/ into src/capabilities/."""
    
    # Mapping of old tool files to new capability modules
    tool_mapping = {
        "tara/tools/desktop/app_launcher.py": "src/capabilities/desktop/apps.py",
        "tara/tools/desktop/window_ops.py": "src/capabilities/desktop/windows.py",
        "tara/tools/desktop/screen_ops.py": "src/capabilities/desktop/screen.py",
        "tara/tools/desktop/uia_ops.py": "src/capabilities/desktop/uia.py",
        "tara/tools/system/file_ops.py": "src/capabilities/system/files.py",
        "tara/tools/system/input_ops.py": "src/capabilities/desktop/input.py",
        "tara/tools/system/system_ops.py": "src/capabilities/system/stats.py",
        "tara/tools/web/browser_ops.py": "src/capabilities/web/browser.py",
        "tara/tools/memory/preferences.py": "src/capabilities/memory/preferences.py",
        "tara/tools/ai/llm_ops.py": "src/capabilities/ai/llm.py",
    }
    
    for old_path, new_path in tool_mapping.items():
        old_file = Path(old_path)
        new_file = Path(new_path)
        
        new_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(old_file, new_file)
        print(f"✓ Moved {old_path} → {new_path}")

if __name__ == "__main__":
    merge_capabilities()
```

### Script 3: Import Updater

```python
# scripts/migration/update_imports.py
import re
from pathlib import Path

def update_imports_in_file(file_path: Path):
    """Update import statements in a Python file."""
    
    replacements = {
        r'from core\.': 'from nia.src.core.',
        r'from nia\.': 'from nia.src.agents.nia.',
        r'from tara\.': 'from nia.src.agents.tara.',
        r'from iris\.': 'from nia.src.agents.iris.',
        r'from nola\.': 'from nia.src.agents.nola.',
        r'from models\.': 'from nia.src.models.',
        r'from persona\.': 'from nia.src.persona.',
        # Add more patterns
    }
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    for pattern, replacement in replacements.items():
        content = re.sub(pattern, replacement, content)
    
    with open(file_path, 'w') as f:
        f.write(content)

def main():
    # Update all Python files
    for py_file in Path('src').rglob('*.py'):
        update_imports_in_file(py_file)
        print(f"✓ Updated imports in {py_file}")

if __name__ == "__main__":
    main()
```

---

## Benefits Summary

### Developer Experience
- ✅ **Clearer mental model**: Domain-driven structure
- ✅ **Faster navigation**: Logical file organization
- ✅ **Easier testing**: Tests mirror source
- ✅ **Better IDE support**: Proper package structure

### Maintainability
- ✅ **Reduced duplication**: Unified capabilities system
- ✅ **Clear dependencies**: Layered architecture
- ✅ **Type safety**: Pydantic configs
- ✅ **Self-documenting**: CAPABILITY.md files

### Scalability
- ✅ **Easy to extend**: Plugin capability domains
- ✅ **Modular agents**: Add new agents easily
- ✅ **Multiple interfaces**: CLI, API, GUI ready
- ✅ **Configuration scaling**: YAML hierarchies

### Performance
- ✅ **Lazy loading**: Import only what's needed
- ✅ **Better caching**: Centralized config loading
- ✅ **Optimized imports**: Reduced circular dependencies

---

## Risk Assessment

### Low Risk
- Config format changes (backward compatible loader)
- File moves within same domain
- Test reorganization

### Medium Risk
- Import path updates (automated script)
- Capability system merger (comprehensive mapping)
- Extension API changes (v1 compatibility layer)

### High Risk
- Database schema changes (none planned)
- Breaking API changes (none if done right)
- User data migration (backup strategy in place)

---

## Timeline

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Phase 1: Preparation | 3 days | Backups, documentation |
| Phase 2: Configuration | 4 days | New config system |
| Phase 3: Source Migration | 5 days | New directory structure |
| Phase 4: Import Updates | 7 days | All imports fixed, tests pass |
| Phase 5: Extensions | 5 days | New extension API |
| Phase 6: Cleanup | 4 days | Documentation, release |
| **Total** | **4 weeks** | **v4.0.0 Release** |

---

## Rollback Plan

If migration fails:

1. **Immediate Rollback**
   ```bash
   git checkout main
   git branch -D refactor/v4-architecture
   ```

2. **Restore Backups**
   ```bash
   python scripts/ops/restore_memory.py
   ```

3. **Post-Mortem**
   - Document what went wrong
   - Update migration plan
   - Try again with lessons learned

---

## Next Steps

1. **Review this plan** with the team
2. **Create detailed task breakdown** in your project management tool
3. **Set up migration branch** with CI/CD
4. **Start with Phase 1** (low-risk preparation)
5. **Iterate and adapt** based on discoveries

---

## Questions to Consider

1. **Do you have active users?** If yes, need gradual migration with v3/v4 compatibility
2. **Is there a deadline?** Might need to prioritize certain phases
3. **Database migrations needed?** Memory system changes require careful planning
4. **Extension ecosystem?** If users have custom plugins, need compatibility layer

---

**Author**: Claude (Sonnet 4.5)  
**Date**: 2026-02-03  
**Version**: 1.0  
**Status**: Proposal - Awaiting Review
