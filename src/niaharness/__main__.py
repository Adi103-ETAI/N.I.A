"""Entry point for `python -m niaharness`.

Redirects to the unified NIA CLI (niaharness.cli:app).
"""

from niaharness.cli import app

if __name__ == "__main__":
    app()
