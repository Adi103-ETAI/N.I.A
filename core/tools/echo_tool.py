"""A simple echo tool for development and demos.

This tool follows the lightweight tool contract:
  - name: str
  - description: str
  - run(params: Dict) -> Dict

Used by the ToolManager in demos and tests.
"""
from typing import Any, Dict


class EchoTool:
    """A simple demo tool that echoes back whatever is passed."""

    name = "echo"
    description = "Returns whatever text or message you provide."

    def run(self, text: str = None, message: str = None):
        # Normalize input regardless of key name
        if message is not None:
            return message
        if text is not None:
            return text
        return "(nothing to echo)"
