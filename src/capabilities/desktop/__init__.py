# src/capabilities/desktop/__init__.py
"""Desktop Automation Capabilities.

Provides all host-side desktop control tools used by TARA.
Imports are star-exported so the executor node can discover them via
``get_tara_tools()``.

Modules:
    apps.py          — Application launching (``launch_app``)
    windows.py       — Window management (focus, resize, move)
    screen.py        — Screen capture and inspection
    uia.py           — UIAutomation element interaction (click by text/role)
    input.py         — Keyboard & mouse input (``type_text``, ``click``, hotkeys)
    window_manager.py — Named window registry for multi-window workflows

Security: most tools carry ``@security_level("host_standard")``;
file-destructive operations are tagged ``"high_risk"``.
"""

from .apps import *
from .windows import *
from .screen import *
from .uia import *
from .input import *
from .window_manager import *
