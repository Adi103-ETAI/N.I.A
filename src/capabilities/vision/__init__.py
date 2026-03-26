# src/capabilities/vision/__init__.py
"""Vision Helper Capabilities.

Shared screenshot and capture utilities used by both IRIS (the vision agent)
and TARA (when it needs to inspect the screen).

Currently provides no direct tool exports — capture functions are defined in
``src.agents.iris.capture`` and re-exported through the IRIS package.
This subpackage is reserved for future shared vision utilities.
"""

__all__ = []
