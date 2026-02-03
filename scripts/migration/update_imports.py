"""Update import statements across the src/ codebase."""

import re
from pathlib import Path
from typing import Dict, List, Tuple

# Import replacements (old pattern -> new pattern)
# These are applied IN ORDER so more specific patterns should come first
REPLACEMENTS: List[Tuple[str, str]] = [
    # Specific tool -> capability mappings (MUST come first)
    (r'from tara\.tools\.desktop\.app_launcher', 'from src.capabilities.desktop.apps'),
    (r'from tara\.tools\.desktop\.window_ops', 'from src.capabilities.desktop.windows'),
    (r'from tara\.tools\.desktop\.screen_ops', 'from src.capabilities.desktop.screen'),
    (r'from tara\.tools\.desktop\.uia_ops', 'from src.capabilities.desktop.uia'),
    (r'from tara\.tools\.desktop\.window_manager', 'from src.capabilities.desktop.window_manager'),
    (r'from tara\.tools\.system\.file_ops', 'from src.capabilities.system.files'),
    (r'from tara\.tools\.system\.input_ops', 'from src.capabilities.desktop.input'),
    (r'from tara\.tools\.system\.system_ops', 'from src.capabilities.system.stats'),
    (r'from tara\.tools\.web\.browser_ops', 'from src.capabilities.web.browser'),
    (r'from tara\.tools\.memory\.preferences', 'from src.capabilities.memory.preferences'),
    (r'from tara\.tools\.ai\.llm_ops', 'from src.capabilities.ai.llm'),
    (r'from tara\.tools\.decorators', 'from src.capabilities.decorators'),
    (r'from tara\.tools\.interface', 'from src.capabilities.interface'),
    (r'from tara\.tools\.', 'from src.capabilities.'),
    (r'import tara\.tools\.', 'import src.capabilities.'),
    
    # Model manager rename
    (r'from models\.model_manager', 'from src.models.manager'),
    (r'import models\.model_manager', 'import src.models.manager'),
    
    # Core file renames
    (r'from core\.event_bus', 'from src.core.events'),
    (r'from core\.os_context', 'from src.core.platform'),
    (r'from core\.services', 'from src.core.registry'),
    (r'from core\.engine\.system', 'from src.core.engine.orchestrator'),
    (r'import core\.event_bus', 'import src.core.events'),
    (r'import core\.os_context', 'import src.core.platform'),
    (r'import core\.services', 'import src.core.registry'),
    (r'import core\.engine\.system', 'import src.core.engine.orchestrator'),
    
    # IRIS tools.py rename
    (r'from iris\.tools', 'from src.agents.iris.capture'),
    (r'import iris\.tools', 'import src.agents.iris.capture'),
    
    # General package migrations (MUST come after specific patterns)
    (r'from core\.', 'from src.core.'),
    (r'import core\.', 'import src.core.'),
    (r'from nia\.', 'from src.agents.nia.'),
    (r'import nia\.', 'import src.agents.nia.'),
    (r'from tara\.', 'from src.agents.tara.'),
    (r'import tara\.', 'import src.agents.tara.'),
    (r'from iris\.', 'from src.agents.iris.'),
    (r'import iris\.', 'import src.agents.iris.'),
    (r'from nola\.', 'from src.agents.nola.'),
    (r'import nola\.', 'import src.agents.nola.'),
    (r'from models\.', 'from src.models.'),
    (r'import models\.', 'import src.models.'),
    (r'from persona\.', 'from src.persona.'),
    (r'import persona\.', 'import src.persona.'),
    (r'from interface\.', 'from src.interface.'),
    (r'import interface\.', 'import src.interface.'),
]


def update_file(file_path: Path) -> int:
    """Update imports in a single file. Returns number of changes."""
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"[WARN] Error reading {file_path}: {e}")
        return 0
    
    original_content = content
    changes = 0
    
    for pattern, replacement in REPLACEMENTS:
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
    """Update all Python files in src/."""
    src_dir = Path('src')
    total_files = 0
    total_changes = 0
    
    print("=" * 60)
    print("N.I.A. v4.0.0 Import Update Script")
    print("=" * 60)
    print()
    
    for py_file in src_dir.rglob('*.py'):
        changes = update_file(py_file)
        if changes > 0:
            print(f"[OK] {py_file}: {changes} import(s) updated")
            total_changes += changes
        total_files += 1
    
    print()
    print("=" * 60)
    print(f"Summary: {total_changes} imports updated across {total_files} files")
    print("=" * 60)


if __name__ == "__main__":
    main()
