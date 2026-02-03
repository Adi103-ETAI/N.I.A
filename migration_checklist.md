# N.I.A. v4.0.0 Migration Checklist
## Step-by-Step Implementation Guide

---

## Pre-Migration (Day 1)

### ✅ Checkpoint 1: Backup Everything

```bash
# Create backup branch
git checkout -b backup/v3.1.0-$(date +%Y%m%d)
git push origin backup/v3.1.0-$(date +%Y%m%d)

# Backup memory databases
cp -r data/ data.backup/

# Backup configs
tar -czf config_backup_$(date +%Y%m%d).tar.gz config/

# Export current environment
pip freeze > requirements_v3.1.0.txt
```

### ✅ Checkpoint 2: Document Current State

```bash
# Run all tests and save results
pytest -v > test_results_v3.1.0.txt

# Generate coverage report
pytest --cov=. --cov-report=html
mv htmlcov/ coverage_v3.1.0/

# List all custom plugins
ls -la plugins/ > custom_plugins_list.txt

# Save current config values
python scripts/verify_config.py > config_snapshot.txt
```

### ✅ Checkpoint 3: Create Migration Branch

```bash
git checkout -b refactor/v4-architecture
git push -u origin refactor/v4-architecture
```

---

## Phase 1: Configuration Migration (Days 2-5)

### ✅ Step 1.1: Install YAML Support

```bash
pip install pyyaml pydantic pydantic-settings
pip freeze > requirements.txt
```

### ✅ Step 1.2: Create New Config Structure

```bash
# Create new config directories
mkdir -p config/{base,agents,capabilities}

# Create settings model
cat > config/base/settings.py << 'EOF'
"""Configuration schemas."""

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal, Optional

class AgentConfig(BaseModel):
    """Base agent configuration."""
    name: str
    version: str
    debug_mode: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

class NIAConfig(AgentConfig):
    """NIA supervisor configuration."""
    routing_mode: Literal["llm", "rules", "hybrid"] = "hybrid"
    confidence_threshold: float = Field(0.7, ge=0.0, le=1.0)

class TARAConfig(AgentConfig):
    """TARA tool agent configuration."""
    max_tool_retries: int = Field(3, ge=1)
    tool_timeout: int = Field(30, ge=1)

class IRISConfig(AgentConfig):
    """IRIS vision agent configuration."""
    vision_model: str = "llama-3.2-90b-vision-instruct"
    confidence_threshold: float = 0.6

class NOLAConfig(AgentConfig):
    """NOLA voice agent configuration."""
    wake_word: str = "hey nia"
    tts_voice: str = "af_sarah"
    stt_model: str = "whisper-large-v3"

class ModelProviderConfig(BaseModel):
    """LLM provider configuration."""
    api_key: Optional[str] = None
    model: str
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(2000, ge=1)

class Settings(BaseSettings):
    """Application settings."""
    
    model_config = SettingsConfigDict(
        env_prefix="NIA_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    # Agent configs
    nia: NIAConfig
    tara: TARAConfig
    iris: IRISConfig
    nola: NOLAConfig
    
    # Model configs
    default_provider: str = "nvidia"
    providers: dict[str, ModelProviderConfig]
EOF
```

### ✅ Step 1.3: Convert JSON to YAML

```bash
# Create conversion script
cat > scripts/migration/convert_json_to_yaml.py << 'EOF'
"""Convert JSON configs to YAML."""

import json
import yaml
from pathlib import Path

def convert_file(json_path: Path, yaml_path: Path):
    """Convert a JSON file to YAML."""
    with open(json_path) as f:
        data = json.load(f)
    
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    
    print(f"✓ Converted {json_path} → {yaml_path}")

def main():
    """Convert all config files."""
    
    conversions = [
        # NIA configs
        ("config/nia/general.json", "config/agents/nia.yaml"),
        ("config/nia/models.json", "config/models.yaml"),
        ("config/nia/routing.json", "config/agents/nia_routing.yaml"),
        
        # TARA configs
        ("config/tara/apps.json", "config/capabilities/desktop.yaml"),
        ("config/tara/commands.json", "config/agents/tara.yaml"),
        
        # IRIS configs
        ("config/iris/sentry.json", "config/agents/iris.yaml"),
        
        # NOLA configs
        ("config/nola/voice.json", "config/agents/nola.yaml"),
    ]
    
    for json_path, yaml_path in conversions:
        json_file = Path(json_path)
        yaml_file = Path(yaml_path)
        
        if json_file.exists():
            convert_file(json_file, yaml_file)
        else:
            print(f"⚠ Skipping {json_path} (not found)")

if __name__ == "__main__":
    main()
EOF

# Run conversion
python scripts/migration/convert_json_to_yaml.py
```

### ✅ Step 1.4: Manually Consolidate Configs

```bash
# Example: Create config/agents/nia.yaml
cat > config/agents/nia.yaml << 'EOF'
# NIA Supervisor Agent Configuration

name: NIA
version: 4.0.0
debug_mode: false
log_level: INFO

# Routing
routing_mode: hybrid
confidence_threshold: 0.7

# Gatekeeper
gatekeeper:
  enabled: true
  fallback_agent: chat
  
# LangGraph
graph:
  max_iterations: 10
  timeout_seconds: 30

# Memory
memory:
  enabled: true
  max_conversation_length: 50
EOF

# Example: Create config/models.yaml
cat > config/models.yaml << 'EOF'
# LLM Model Configurations

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
EOF
```

### ✅ Step 1.5: Create Config Loader

```bash
cat > config/loader.py << 'EOF'
"""Configuration loader with validation."""

import os
import yaml
from pathlib import Path
from typing import Any
from config.base.settings import Settings

def _load_yaml(file_path: Path) -> dict[str, Any]:
    """Load YAML file with environment variable substitution."""
    with open(file_path) as f:
        content = f.read()
        
    # Replace ${VAR} with environment variables
    import re
    def replace_env(match):
        var_name = match.group(1)
        return os.getenv(var_name, match.group(0))
    
    content = re.sub(r'\$\{([^}]+)\}', replace_env, content)
    return yaml.safe_load(content)

def load_config() -> Settings:
    """Load and validate application configuration."""
    config_dir = Path("config")
    
    # Load all YAML configs
    config_data = {}
    
    # Load agent configs
    for agent in ["nia", "tara", "iris", "nola"]:
        agent_file = config_dir / "agents" / f"{agent}.yaml"
        if agent_file.exists():
            config_data[agent] = _load_yaml(agent_file)
    
    # Load model config
    models_file = config_dir / "models.yaml"
    if models_file.exists():
        model_data = _load_yaml(models_file)
        config_data.update(model_data)
    
    # Validate with Pydantic
    return Settings(**config_data)

# Singleton
_settings: Settings | None = None

def get_settings() -> Settings:
    """Get settings singleton."""
    global _settings
    if _settings is None:
        _settings = load_config()
    return _settings
EOF
```

### ✅ Step 1.6: Test Config Loading

```bash
# Create test script
cat > scripts/test_config_loading.py << 'EOF'
"""Test new config loading."""

from config.loader import get_settings

def main():
    try:
        settings = get_settings()
        print("✓ Config loaded successfully!")
        print(f"  NIA version: {settings.nia.version}")
        print(f"  Default provider: {settings.default_provider}")
        print(f"  Log level: {settings.nia.log_level}")
    except Exception as e:
        print(f"✗ Config loading failed: {e}")
        raise

if __name__ == "__main__":
    main()
EOF

python scripts/test_config_loading.py
```

---

## Phase 2: Directory Restructure (Days 6-10)

### ✅ Step 2.1: Create New Source Structure

```bash
# Create src/ directory structure
mkdir -p src/{core,agents,capabilities,models,persona,interface,extensions}
mkdir -p src/agents/{nia,tara,iris,nola}
mkdir -p src/capabilities/{desktop,system,web,vision,memory,ai}
mkdir -p src/interface/{cli,api,gui}
```

### ✅ Step 2.2: Move Core Files

```bash
# Move core files
cp -r core/* src/core/

# Rename files
mv src/core/event_bus.py src/core/events.py
mv src/core/os_context.py src/core/platform.py
mv src/core/services.py src/core/registry.py
mv src/core/engine/system.py src/core/engine/orchestrator.py

# Verify
ls -la src/core/
```

### ✅ Step 2.3: Move Agent Files

```bash
# Move NIA
cp -r nia/* src/agents/nia/

# Move TARA (without tools for now)
mkdir -p src/agents/tara/graph
cp tara/__init__.py src/agents/tara/
cp tara/protocols.py src/agents/tara/
cp -r tara/graph/* src/agents/tara/graph/
cp tara/security/warden.py src/agents/tara/security.py

# Move IRIS
cp -r iris/* src/agents/iris/
mv src/agents/iris/tools.py src/agents/iris/capture.py

# Move NOLA
cp -r nola/* src/agents/nola/

# Verify
tree src/agents/
```

### ✅ Step 2.4: Consolidate Capabilities (Critical Step)

```bash
# Create capability merger script
cat > scripts/migration/merge_capabilities.py << 'EOF'
"""Merge tara/tools/ into src/capabilities/."""

import shutil
from pathlib import Path

def merge_tools():
    """Merge tool files into capability domains."""
    
    # Define mappings: old file → new file
    mappings = {
        # Desktop capabilities
        "tara/tools/desktop/app_launcher.py": "src/capabilities/desktop/apps.py",
        "tara/tools/desktop/window_ops.py": "src/capabilities/desktop/windows.py",
        "tara/tools/desktop/screen_ops.py": "src/capabilities/desktop/screen.py",
        "tara/tools/desktop/uia_ops.py": "src/capabilities/desktop/uia.py",
        "tara/tools/system/input_ops.py": "src/capabilities/desktop/input.py",
        
        # Desktop drivers
        "tara/tools/desktop/drivers/": "src/capabilities/desktop/drivers/",
        "tara/tools/desktop/window_manager.py": "src/capabilities/desktop/window_manager.py",
        
        # System capabilities
        "tara/tools/system/file_ops.py": "src/capabilities/system/files.py",
        "tara/tools/system/system_ops.py": "src/capabilities/system/stats.py",
        
        # Web capabilities
        "tara/tools/web/browser_ops.py": "src/capabilities/web/browser.py",
        
        # Memory capabilities
        "tara/tools/memory/preferences.py": "src/capabilities/memory/preferences.py",
        
        # AI capabilities
        "tara/tools/ai/llm_ops.py": "src/capabilities/ai/llm.py",
    }
    
    for old_path, new_path in mappings.items():
        old = Path(old_path)
        new = Path(new_path)
        
        if old.is_dir():
            shutil.copytree(old, new, dirs_exist_ok=True)
            print(f"✓ Copied {old_path}/ → {new_path}/")
        elif old.exists():
            new.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(old, new)
            print(f"✓ Moved {old_path} → {new_path}")
        else:
            print(f"⚠ Skipped {old_path} (not found)")

if __name__ == "__main__":
    merge_tools()
EOF

# Run merger
python scripts/migration/merge_capabilities.py
```

### ✅ Step 2.5: Create __init__.py Files

```bash
# Create capability __init__.py files
for dir in src/capabilities/*; do
    if [ -d "$dir" ]; then
        touch "$dir/__init__.py"
        echo "# $(basename $dir) capability" > "$dir/__init__.py"
    fi
done

# Create agent __init__.py files
for dir in src/agents/*; do
    if [ -d "$dir" ]; then
        if [ ! -f "$dir/__init__.py" ]; then
            touch "$dir/__init__.py"
        fi
    fi
done
```

### ✅ Step 2.6: Move Other Packages

```bash
# Move models
cp -r models/* src/models/
mv src/models/model_manager.py src/models/manager.py

# Move persona
cp -r persona/* src/persona/

# Move interface
cp -r interface/* src/interface/
```

---

## Phase 3: Update Imports (Days 11-15)

### ✅ Step 3.1: Create Import Update Script

```bash
cat > scripts/migration/update_imports.py << 'EOF'
"""Update import statements across codebase."""

import re
from pathlib import Path
from typing import Dict

# Import replacements
REPLACEMENTS: Dict[str, str] = {
    # Core imports
    r'from core\.': 'from nia.src.core.',
    r'import core\.': 'import nia.src.core.',
    
    # Agent imports
    r'from nia\.agent': 'from nia.src.agents.nia.agent',
    r'from nia\.graph': 'from nia.src.agents.nia.graph',
    r'from tara\.': 'from nia.src.agents.tara.',
    r'from iris\.': 'from nia.src.agents.iris.',
    r'from nola\.': 'from nia.src.agents.nola.',
    
    # Tool imports → Capability imports
    r'from tara\.tools\.desktop\.app_launcher': 'from nia.src.capabilities.desktop.apps',
    r'from tara\.tools\.desktop\.window_ops': 'from nia.src.capabilities.desktop.windows',
    r'from tara\.tools\.desktop\.screen_ops': 'from nia.src.capabilities.desktop.screen',
    r'from tara\.tools\.desktop\.uia_ops': 'from nia.src.capabilities.desktop.uia',
    r'from tara\.tools\.system\.file_ops': 'from nia.src.capabilities.system.files',
    r'from tara\.tools\.system\.input_ops': 'from nia.src.capabilities.desktop.input',
    r'from tara\.tools\.system\.system_ops': 'from nia.src.capabilities.system.stats',
    r'from tara\.tools\.web\.browser_ops': 'from nia.src.capabilities.web.browser',
    r'from tara\.tools\.memory\.preferences': 'from nia.src.capabilities.memory.preferences',
    r'from tara\.tools\.ai\.llm_ops': 'from nia.src.capabilities.ai.llm',
    
    # Model imports
    r'from models\.model_manager': 'from nia.src.models.manager',
    r'from models\.': 'from nia.src.models.',
    
    # Persona imports
    r'from persona\.': 'from nia.src.persona.',
    
    # Interface imports
    r'from interface\.': 'from nia.src.interface.',
}

def update_file(file_path: Path) -> int:
    """Update imports in a single file. Returns number of changes."""
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"⚠ Error reading {file_path}: {e}")
        return 0
    
    original_content = content
    changes = 0
    
    for pattern, replacement in REPLACEMENTS.items():
        new_content = re.sub(pattern, replacement, content)
        if new_content != content:
            changes += 1
            content = new_content
    
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return changes
    
    return 0

def main():
    """Update all Python files."""
    src_dir = Path('src')
    total_files = 0
    total_changes = 0
    
    for py_file in src_dir.rglob('*.py'):
        changes = update_file(py_file)
        if changes > 0:
            print(f"✓ {py_file}: {changes} import(s) updated")
            total_changes += changes
        total_files += 1
    
    print(f"\nSummary: {total_changes} imports updated in {total_files} files")

if __name__ == "__main__":
    main()
EOF

# Run import updater
python scripts/migration/update_imports.py
```

### ✅ Step 3.2: Update main.py

```bash
# Update main entry point
cat > main.py << 'EOF'
"""N.I.A. v4.0.0 - Main entry point."""

import asyncio
from nia.src.core.engine.orchestrator import NIAAssistant
from nia.src.core.logger import get_logger
from config.loader import get_settings

logger = get_logger(__name__)

async def main():
    """Initialize and run NIA."""
    try:
        # Load configuration
        settings = get_settings()
        logger.info(f"Starting N.I.A. v{settings.nia.version}")
        
        # Create and initialize assistant
        assistant = NIAAssistant(settings)
        await assistant.initialize()
        
        # Run assistant
        await assistant.run()
        
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        if 'assistant' in locals():
            await assistant.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
EOF
```

### ✅ Step 3.3: Fix Circular Imports

```bash
# Common circular import fixes:

# 1. Use TYPE_CHECKING
cat > src/core/example_fix.py << 'EOF'
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nia.src.agents.nia.agent import SupervisorAgent

def some_function(agent: "SupervisorAgent"):
    pass
EOF

# 2. Move imports into functions (if needed)
# 3. Use forward references
```

---

## Phase 4: Testing (Days 16-18)

### ✅ Step 4.1: Reorganize Tests

```bash
# Create new test structure
mkdir -p tests/{unit,integration,e2e}
mkdir -p tests/unit/{core,agents,capabilities,models}

# Move existing tests
mv tests/test_config.py tests/unit/core/
mv tests/test_memory.py tests/unit/core/
mv tests/test_model_manager.py tests/unit/models/
# ... move others
```

### ✅ Step 4.2: Update Test Imports

```bash
# Same script as before but for tests/
python scripts/migration/update_imports.py  # Add tests/ to search path
```

### ✅ Step 4.3: Run Test Suite

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Compare with baseline
diff test_results_v3.1.0.txt test_results_v4.0.0.txt
```

### ✅ Step 4.4: Fix Broken Tests

```bash
# Common issues:
# 1. Import errors → Update import paths
# 2. Config loading → Use new config loader
# 3. Path changes → Update file paths
```

---

## Phase 5: Documentation (Days 19-20)

### ✅ Step 5.1: Update README

```bash
cat > README.md << 'EOF'
# N.I.A. v4.0.0

Next-generation AI assistant with modular architecture.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env with your API keys

# Run
python main.py
```

## Architecture

- `src/core/` - Infrastructure
- `src/agents/` - AI agents (NIA, TARA, IRIS, NOLA)
- `src/capabilities/` - Tool implementations
- `config/` - YAML configurations

See `docs/architecture/` for details.
EOF
```

### ✅ Step 5.2: Create Migration Guide for Users

```bash
cat > docs/MIGRATION_GUIDE.md << 'EOF'
# Migrating from v3.1.0 to v4.0.0

## For Users

### Configuration Changes

Old:
```json
// config/nia/general.json
{
  "name": "NIA",
  "debug_mode": false
}
```

New:
```yaml
# config/agents/nia.yaml
name: NIA
debug_mode: false
```

### Plugin Changes

Old location: `plugins/my_plugin.py`
New location: `extensions/custom/my_plugin.py`

Update your plugin:
```python
# Old
from plugins import BasePlugin

# New
from nia.src.extensions.base import BaseExtension
```

## For Developers

### Import Changes

```python
# Old imports
from core.logger import setup_logger
from tara.tools.desktop.window_ops import focus_window

# New imports
from nia.src.core.logger import get_logger
from nia.src.capabilities.desktop.windows import WindowManagement
```

See full migration guide in `/docs/migration/`.
EOF
```

---

## Phase 6: Cleanup & Release (Days 21-22)

### ✅ Step 6.1: Remove Old Files

```bash
# Create cleanup script
cat > scripts/migration/cleanup_old_files.py << 'EOF'
"""Remove old directory structure after verification."""

import shutil
from pathlib import Path

def cleanup():
    """Remove old directories."""
    old_dirs = [
        "core",  # Moved to src/core
        "nia",   # Moved to src/agents/nia
        "tara",  # Moved to src/agents/tara
        "iris",  # Moved to src/agents/iris
        "nola",  # Moved to src/agents/nola
        "models",  # Moved to src/models
        "persona",  # Moved to src/persona
        "interface",  # Moved to src/interface
        "plugins",  # Moved to extensions
        "skills",  # Merged into capabilities
    ]
    
    print("⚠ This will DELETE old directories. Ensure migration is complete!")
    print("Old directories to remove:")
    for d in old_dirs:
        print(f"  - {d}/")
    
    response = input("\nProceed? (yes/no): ")
    if response.lower() != 'yes':
        print("Aborted.")
        return
    
    for d in old_dirs:
        path = Path(d)
        if path.exists():
            shutil.rmtree(path)
            print(f"✓ Removed {d}/")

if __name__ == "__main__":
    cleanup()
EOF

# Run after manual verification
# python scripts/migration/cleanup_old_files.py
```

### ✅ Step 6.2: Update Version

```bash
# Update version in code
echo "4.0.0" > VERSION

# Update setup.py or pyproject.toml
cat > pyproject.toml << 'EOF'
[tool.poetry]
name = "nia"
version = "4.0.0"
description = "Next-generation AI assistant"
authors = ["Your Name <you@example.com>"]

[tool.poetry.dependencies]
python = "^3.11"
# ... dependencies

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
EOF
```

### ✅ Step 6.3: Final Testing

```bash
# Full test suite
pytest tests/ -v --cov=src

# Manual smoke tests
python main.py  # Should start without errors

# Test key features
# - Voice input
# - Tool execution
# - Vision analysis
# - Memory storage
```

### ✅ Step 6.4: Create Release

```bash
# Commit changes
git add .
git commit -m "refactor: Migrate to v4.0.0 clean architecture"

# Merge to main (after review)
git checkout main
git merge refactor/v4-architecture

# Tag release
git tag -a v4.0.0 -m "Release v4.0.0 - Clean Architecture"
git push origin main --tags
```

---

## Rollback Procedure (If Needed)

### If Something Goes Wrong

```bash
# Immediate rollback
git checkout main
git reset --hard backup/v3.1.0-YYYYMMDD

# Restore data
rm -rf data/
mv data.backup/ data/

# Restore configs
tar -xzf config_backup_YYYYMMDD.tar.gz

# Verify
python main.py
```

---

## Post-Migration Checklist

### ✅ Verify Everything Works

- [ ] Configuration loads without errors
- [ ] All tests pass
- [ ] Main application starts
- [ ] Voice input/output works
- [ ] Tool execution works
- [ ] Vision analysis works
- [ ] Memory persistence works
- [ ] Plugins load correctly
- [ ] No import errors in logs
- [ ] Performance is comparable

### ✅ Documentation Complete

- [ ] README updated
- [ ] Architecture docs updated
- [ ] API reference updated
- [ ] Migration guide written
- [ ] CHANGELOG updated

### ✅ User Communication

- [ ] Announce v4.0.0 release
- [ ] Share migration guide
- [ ] Provide support for migration issues
- [ ] Collect feedback

---

## Estimated Timeline Summary

| Phase | Days | Checkpoints |
|-------|------|-------------|
| **Pre-Migration** | 1 | Backup, document, branch |
| **Phase 1: Config** | 4 | YAML conversion, validation |
| **Phase 2: Structure** | 5 | Move files, consolidate |
| **Phase 3: Imports** | 5 | Update all imports |
| **Phase 4: Testing** | 3 | Fix and verify tests |
| **Phase 5: Docs** | 2 | Update documentation |
| **Phase 6: Cleanup** | 2 | Remove old files, release |
| **Total** | **22 days** | **Production ready** |

---

## Tips for Success

1. **Test frequently** - After each phase, run tests
2. **Commit often** - Small commits are easier to debug
3. **Use branches** - Keep main stable during migration
4. **Document issues** - Note any problems for later reference
5. **Ask for help** - Review complex changes with team
6. **Take breaks** - Large refactors are mentally taxing

---

## Emergency Contacts

- Migration lead: [Your name]
- Backup: [Backup person]
- Escalation: [Manager/Lead]

**Remember:** This is a significant refactor. Take your time, test thoroughly, and don't hesitate to pause if issues arise.

---

**Good luck with the migration! 🚀**
