"""Entry point for `python -m niaharness`.

Loads ~/.nia/.env first, then delegates to the unified NIA CLI.
"""

# Load ~/.nia/.env BEFORE anything else.
try:
    from niaharness.config.env_loader import load_nia_env
    load_nia_env()
except Exception:
    pass

from niaharness.cli import app

if __name__ == "__main__":
    app()
