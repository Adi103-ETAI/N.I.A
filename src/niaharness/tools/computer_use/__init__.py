"""Computer Use tool package — cua-driver backend only.

Mirrors Hermes Agent's tools/computer_use/ architecture:
- backend.py: abstract ComputerUseBackend interface
- cua_backend.py: CUADriverBackend (the only implementation — no fallback)
- schema.py: ComputerUseInput Pydantic model (13 actions)
- tool.py: ComputerUseTool (agent-facing tool)
"""

from niaharness.tools.computer_use.tool import ComputerUseTool

__all__ = ["ComputerUseTool"]
