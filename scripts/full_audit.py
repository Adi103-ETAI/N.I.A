"""Comprehensive N.I.A. v4.0.0 Structural Audit.

Checks:
1. Import validation (every .py in src/)
2. Missing __init__.py files
3. Hardcoded config path references vs actual files
4. Missing files referenced in code
5. PYTHONPATH / naming collision issues
"""
import importlib
import sys
import os
import re
import logging
import builtins
from pathlib import Path

# Write output to UTF-8 file
_output_file = open('scripts/audit_report.txt', 'w', encoding='utf-8')
_original_print = builtins.print
def print(*args, **kwargs):
    _original_print(*args, **kwargs)
    kwargs['file'] = _output_file
    _original_print(*args, **kwargs)
    _output_file.flush()

logging.disable(logging.CRITICAL)
sys.path.insert(0, '.')

PROJECT_ROOT = Path('.')
SRC_DIR = PROJECT_ROOT / 'src'
CONFIG_DIR = PROJECT_ROOT / 'config'
DATA_DIR = PROJECT_ROOT / 'data'

results = {
    'import_errors': [],
    'missing_init': [],
    'missing_config_files': [],
    'hardcoded_paths': [],
    'broken_references': [],
}

# =========================================================================
# 1. IMPORT VALIDATION
# =========================================================================
print("=" * 60)
print("1. IMPORT PATH AUDIT")
print("=" * 60)

import_count = 0
for root, dirs, files in os.walk('src'):
    dirs[:] = [d for d in dirs if d != '__pycache__']
    for f in files:
        if f.endswith('.py'):
            full = os.path.join(root, f)
            if f == '__init__.py':
                # Also try importing package init
                mod = root.replace(os.sep, '.')
            else:
                mod = full.replace(os.sep, '.').replace('.py', '')
            import_count += 1
            try:
                importlib.import_module(mod)
            except Exception as e:
                err_msg = str(e).split('\n')[0][:200]
                results['import_errors'].append({
                    'module': mod,
                    'file': full,
                    'error_type': type(e).__name__,
                    'message': err_msg,
                })

print(f"Scanned: {import_count} modules")
print(f"Errors: {len(results['import_errors'])}")
for err in results['import_errors']:
    print(f"  FAIL: {err['file']}")
    print(f"        {err['error_type']}: {err['message']}")
    print()

# =========================================================================
# 2. MISSING __init__.py FILES
# =========================================================================
print("=" * 60)
print("2. MISSING __init__.py FILES")
print("=" * 60)

for root, dirs, files in os.walk('src'):
    dirs[:] = [d for d in dirs if d != '__pycache__']
    for d in dirs:
        dir_path = os.path.join(root, d)
        init_path = os.path.join(dir_path, '__init__.py')
        has_py_files = any(f.endswith('.py') for f in os.listdir(dir_path) if f != '__init__.py')
        if has_py_files and not os.path.exists(init_path):
            results['missing_init'].append(dir_path)
            print(f"  MISSING: {init_path}")

if not results['missing_init']:
    print("  All directories have __init__.py ✓")

# =========================================================================
# 3. HARDCODED CONFIG / FILE PATH REFERENCES
# =========================================================================
print()
print("=" * 60)
print("3. HARDCODED PATH REFERENCES vs ACTUAL FILES")
print("=" * 60)

# Patterns to search for
path_patterns = [
    r'["\']([^"\']*(?:general|triggers|sentry|models|voice|engine)\.(?:json|yaml|yml))["\']',
    r'["\']([^"\']*\.json)["\']',
    r'["\']([^"\']*\.yaml)["\']',
    r'["\']([^"\']*\.yml)["\']',
]

config_refs = {}  # file -> list of referenced paths

for root, dirs, files in os.walk('src'):
    dirs[:] = [d for d in dirs if d != '__pycache__']
    for f in files:
        if f.endswith('.py'):
            filepath = os.path.join(root, f)
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as fh:
                    content = fh.read()
                    for pattern in path_patterns:
                        matches = re.findall(pattern, content)
                        for match in matches:
                            if any(skip in match for skip in ['http', 'schema', 'application/', 'text/', '.pyc', 'test']):
                                continue
                            if match not in config_refs:
                                config_refs[match] = []
                            config_refs[match].append(filepath)
            except Exception:
                pass

# Check each referenced config file
for ref_path, source_files in sorted(config_refs.items()):
    # Try to resolve the path
    candidates = [
        PROJECT_ROOT / ref_path,
        CONFIG_DIR / ref_path,
        PROJECT_ROOT / 'config' / ref_path,
    ]
    
    # Also check if it's a relative reference like 'config/nola/voice.json'
    found = False
    for candidate in candidates:
        if candidate.exists():
            found = True
            break
    
    # Also do a broader search
    if not found:
        basename = os.path.basename(ref_path)
        for search_root, _, search_files in os.walk('.'):
            if basename in search_files:
                found = True
                break
    
    status = "✓ FOUND" if found else "✗ MISSING"
    if not found:
        results['missing_config_files'].append({
            'path': ref_path,
            'referenced_by': source_files,
        })
    
    # Only print missing or important ones
    if not found or any(kw in ref_path for kw in ['general', 'triggers', 'sentry', 'models', 'voice', 'engine']):
        print(f"  {status}: {ref_path}")
        for sf in source_files[:3]:
            print(f"           ← {sf}")

# =========================================================================
# 4. SPECIFIC FILE EXISTENCE CHECKS
# =========================================================================
print()
print("=" * 60)
print("4. CRITICAL FILE EXISTENCE CHECKS")
print("=" * 60)

critical_files = [
    'config/general.json',
    'config/general.yaml',
    'config/triggers.json',
    'config/triggers.yaml', 
    'config/sentry.json',
    'config/sentry.yaml',
    'config/models.json',
    'config/models.yaml',
    'config/nola/voice.json',
    'config/nola/voice.yaml',
    'config/engine.yaml',
    'src/interface/__init__.py',
    'src/interface/cli/__init__.py',
    'src/interface/cli/banner.py',
    'src/models/__init__.py',
    'src/models/manager.py',
    'src/agents/iris/__init__.py',
    'src/agents/iris/agent.py',
    'src/core/engine/__init__.py',
    'src/core/engine/orchestrator.py',
]

for fpath in critical_files:
    full = PROJECT_ROOT / fpath
    status = "✓" if full.exists() else "✗ MISSING"
    print(f"  {status}  {fpath}")
    if not full.exists():
        results['broken_references'].append(fpath)

# =========================================================================
# 5. INTERFACE / CLI MODULE CHECK
# =========================================================================
print()
print("=" * 60)
print("5. INTERFACE & CLI MODULE CHECK")
print("=" * 60)

# Check if interface directory has proper structure
interface_dir = SRC_DIR / 'interface'
if interface_dir.exists():
    print(f"  src/interface/ exists")
    for item in sorted(interface_dir.rglob('*')):
        if '__pycache__' not in str(item):
            rel = item.relative_to(SRC_DIR)
            print(f"    {'📁' if item.is_dir() else '📄'} {rel}")
else:
    print(f"  ✗ src/interface/ DOES NOT EXIST")

# Try importing src.interface.cli.banner specifically
print()
try:
    import importlib
    mod = importlib.import_module('src.interface.cli.banner')
    print(f"  ✓ src.interface.cli.banner imports OK")
except Exception as e:
    print(f"  ✗ src.interface.cli.banner FAILS: {type(e).__name__}: {e}")

# =========================================================================
# 6. MODEL MANAGER ANALYSIS
# =========================================================================
print()
print("=" * 60)
print("6. MODEL MANAGER ANALYSIS")
print("=" * 60)

models_manager = SRC_DIR / 'models' / 'manager.py'
if models_manager.exists():
    with open(models_manager, 'r', encoding='utf-8', errors='ignore') as fh:
        content = fh.read()
    
    # Find config file references
    json_refs = re.findall(r'["\']([^"\']*\.(?:json|yaml|yml))["\']', content)
    print(f"  Config references in manager.py: {json_refs}")
    
    # Find Path() references
    path_refs = re.findall(r'Path\(([^)]+)\)', content)
    print(f"  Path() calls: {len(path_refs)}")
    for pr in path_refs[:10]:
        print(f"    Path({pr})")
    
    # Check for provider logic
    provider_lines = [l.strip() for l in content.split('\n') if 'provider' in l.lower() and not l.strip().startswith('#')]
    print(f"\n  Provider-related lines ({len(provider_lines)}):")
    for pl in provider_lines[:15]:
        print(f"    {pl[:120]}")
else:
    print(f"  ✗ src/models/manager.py NOT FOUND")

# =========================================================================
# 7. CONFIG DIRECTORY TREE
# =========================================================================
print()
print("=" * 60)
print("7. ACTUAL CONFIG DIRECTORY TREE")
print("=" * 60)

config_dir = PROJECT_ROOT / 'config'
if config_dir.exists():
    for item in sorted(config_dir.rglob('*')):
        if '__pycache__' not in str(item):
            rel = item.relative_to(PROJECT_ROOT)
            size = item.stat().st_size if item.is_file() else ''
            print(f"  {'📁' if item.is_dir() else '📄'} {rel} {f'({size}B)' if size else ''}")
else:
    print("  ✗ config/ directory NOT FOUND")

# =========================================================================
# SUMMARY
# =========================================================================
print()
print("=" * 60)
print("AUDIT SUMMARY")
print("=" * 60)
print(f"  Import errors:      {len(results['import_errors'])}")
print(f"  Missing __init__.py: {len(results['missing_init'])}")
print(f"  Missing config files: {len(results['missing_config_files'])}")
print(f"  Missing critical files: {len(results['broken_references'])}")
