"""Update import statements across the tests/ codebase."""

import re
from pathlib import Path
from typing import List, Tuple

# Import replacements (old pattern -> new pattern)
# These are applied IN ORDER so more specific patterns should come first
REPLACEMENTS: List[Tuple[str, str]] = [
    # Model imports
    (r'from models\.model_manager', 'from src.models.manager'),
    (r'import models\.model_manager', 'import src.models.manager'),
    (r'from models\.', 'from src.models.'),
    (r'import models\.', 'import src.models.'),
    
    # Core imports
    (r'from core\.event_bus', 'from src.core.events'),
    (r'from core\.os_context', 'from src.core.platform'),
    (r'from core\.services', 'from src.core.registry'),
    (r'from core\.engine\.system', 'from src.core.engine.orchestrator'),
    (r'from core\.', 'from src.core.'),
    (r'import core\.', 'import src.core.'),
    
    # Agent imports
    (r'from nia\.', 'from src.agents.nia.'),
    (r'import nia\.', 'import src.agents.nia.'),
    (r'from tara\.tools\.desktop\.app_launcher', 'from src.capabilities.desktop.apps'),
    (r'from tara\.tools\.desktop\.window_ops', 'from src.capabilities.desktop.windows'),
    (r'from tara\.tools\.system\.file_ops', 'from src.capabilities.system.files'),
    (r'from tara\.tools\.', 'from src.capabilities.'),
    (r'from tara\.', 'from src.agents.tara.'),
    (r'import tara\.', 'import src.agents.tara.'),
    (r'from iris\.', 'from src.agents.iris.'),
    (r'import iris\.', 'import src.agents.iris.'),
    (r'from nola\.', 'from src.agents.nola.'),
    (r'import nola\.', 'import src.agents.nola.'),
    
    # Persona imports
    (r'from persona\.', 'from src.persona.'),
    (r'import persona\.', 'import src.persona.'),
    
    # Interface imports
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
    """Update all Python files in tests/."""
    tests_dir = Path('tests')
    total_files = 0
    total_changes = 0
    
    print("=" * 60)
    print("N.I.A. v4.0.0 Test Import Update Script")
    print("=" * 60)
    print()
    
    for py_file in tests_dir.rglob('*.py'):
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
