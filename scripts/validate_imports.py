"""Validate all imports in src/ by attempting to import each module. Writes results to a file."""
import importlib
import sys
import os
import logging

# Suppress all logging output during import testing
logging.disable(logging.CRITICAL)

sys.path.insert(0, '.')
errors = []
success = 0

for root, dirs, files in os.walk('src'):
    dirs[:] = [d for d in dirs if d != '__pycache__']
    for f in files:
        if f.endswith('.py') and f != '__init__.py':
            full = os.path.join(root, f)
            mod = full.replace(os.sep, '.').replace('.py', '')
            try:
                importlib.import_module(mod)
                success += 1
            except Exception as e:
                err_msg = str(e).split('\n')[0][:200]
                errors.append((mod, type(e).__name__, err_msg))

with open('scripts/import_report.txt', 'w', encoding='utf-8') as out:
    out.write(f"Scanned: {success + len(errors)} modules\n")
    out.write(f"OK: {success}\n")
    out.write(f"ERRORS: {len(errors)}\n")
    out.write("---\n")
    for m, t, e in errors:
        out.write(f"FAIL: {m}\n")
        out.write(f"      {t}: {e}\n\n")

print(f"Done. {len(errors)} errors found. See scripts/import_report.txt")
