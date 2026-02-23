# src/capabilities/system/__init__.py
"""System Operations Capabilities.

Host-level system tools used by TARA for file I/O, process management,
and hardware monitoring.

Modules:
    files.py          — File read/write/delete/search (``read_file``, ``write_file``, etc.)
    stats.py          — CPU, RAM, disk, and GPU usage reporting
    process_tools.py  — Process listing, finding by name, and safe termination
"""

from .files import *
from .stats import *
from .process_tools import *
