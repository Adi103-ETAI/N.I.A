"""NIA CLI package — re-exports from the legacy ``cli.py`` module.

The ``cli/`` package contains new modules (``doctor.py``, ``update.py``)
ported from Hermes. The legacy Typer ``app`` still lives in the
top-level ``cli.py`` file; this ``__init__.py`` re-exports it so
existing ``from niaharness.cli import app`` imports keep working.
"""

# Re-export the Typer app from the legacy cli.py module.
# The cli.py file is loaded via importlib to avoid a circular import
# (cli.py imports from niaharness.* which may import from niaharness.cli).
import importlib.util as _importlib_util
import sys as _sys
from pathlib import Path as _Path

_cli_py = _Path(__file__).resolve().parent.parent / "cli.py"
if _cli_py.exists():
    _spec = _importlib_util.spec_from_file_location("niaharness._legacy_cli", str(_cli_py))
    if _spec is not None and _spec.loader is not None:
        _module = _importlib_util.module_from_spec(_spec)
        _spec.loader.exec_module(_module)
        app = getattr(_module, "app", None)
        _sys.modules["niaharness._legacy_cli"] = _module
    else:
        app = None
else:
    app = None

__all__ = ["app"]
