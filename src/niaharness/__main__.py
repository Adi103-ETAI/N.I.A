"""Entry point for `python -m niaharness`.

Redirects to the NIA agent CLI (agents.nia.__main__) which provides
the new medical-themed UI with caduceus logo, session info, and
flicker-free streaming.
"""

from agents.nia.__main__ import app

if __name__ == "__main__":
    app()
