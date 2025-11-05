"""A simple echo tool for development and demos.

This tool follows the lightweight tool contract:
  - name: str
  - description: str
  - run(params: Dict) -> Dict

Used by the ToolManager in demos and tests.
"""
from typing import Any
from core.base_tool import BaseTool


class EchoTool(BaseTool):
    """A simple demo tool that echoes back whatever is passed.

    The run method accepts any keyword args and normalizes on
    `message` or `text` keys.
    """

    name = "echo"
    description = "Returns whatever text or message you provide."

    def run(self, **params) -> Any:
        message = params.get('message')
        if message is not None:
            return message
        text = params.get('text')
        if text is not None:
            return text
        return "(nothing to echo)"
