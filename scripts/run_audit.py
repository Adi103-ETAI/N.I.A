"""Run the audit and write results to a file."""
import subprocess
import sys

result = subprocess.run(
    [sys.executable, 'scripts/full_audit.py'],
    capture_output=True,
    text=True,
    cwd='.',
    timeout=120,
)

with open('scripts/audit_report.txt', 'w', encoding='utf-8') as f:
    f.write(result.stdout)
    if result.stderr:
        f.write("\n\nSTDERR:\n")
        f.write(result.stderr[:2000])

print(f"Audit complete. Output written to scripts/audit_report.txt ({len(result.stdout)} bytes)")
if result.returncode != 0:
    print(f"Exit code: {result.returncode}")
