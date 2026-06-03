"""N.I.A Execution Bridge - Connects Head to Hands.

This bridge wires N.I.A's dispatcher to niaharness's tool execution system.
N.I.A decides WHAT to do, this bridge makes it HAPPEN.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from niaharness.tools import create_default_tool_registry
from niaharness.tools.base import ToolExecutionContext

logger = logging.getLogger(__name__)


class HarnessExecutorBridge:
    """Bridges N.I.A's dispatcher to niaharness's tool execution.

    This is the critical connection between:
    - N.I.A's brain (decides what to do)
    - niaharness tools (actually does the work)

    Usage:
        bridge = HarnessExecutorBridge(workspace_dir="/path/to/project")
        result = await bridge.execute_tool("write_file", {"file_path": "test.py", "content": "print('hello')"})
    """

    def __init__(self, workspace_dir: str | None = None) -> None:
        self._workspace_dir = workspace_dir or str(Path.cwd())
        self._tool_registry = create_default_tool_registry()
        self._initialized = True

        logger.info(f"Execution bridge initialized for: {self._workspace_dir}")

    async def execute_tool_call(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        """Execute a tool call (compatible with dispatcher's expected signature).

        Args:
            tool_call: Dict with 'tool' and 'arguments' keys

        Returns:
            Dict with 'output' and 'is_error' keys
        """
        tool_name = tool_call.get("tool", "")
        arguments = tool_call.get("arguments", {})
        return await self.execute_tool(tool_name, arguments)

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute a single tool call.

        Args:
            tool_name: Name of the tool to execute (e.g., "write_file", "bash")
            arguments: Tool arguments (dict or Pydantic model)

        Returns:
            Dict with 'output' and 'is_error' keys
        """
        try:
            # 1. Get the tool from registry
            tool = self._tool_registry.get(tool_name)
            if tool is None:
                return {
                    "output": f"Tool '{tool_name}' not found",
                    "is_error": True,
                }

            # 2. Build execution context
            context = ToolExecutionContext(
                cwd=Path(self._workspace_dir),
            )

            # 3. Convert arguments to the tool's input model
            input_model = tool.input_model
            if input_model and isinstance(arguments, dict):
                # Convert dict to Pydantic model
                try:
                    typed_args = input_model(**arguments)
                except Exception as e:
                    # If conversion fails, try with default values
                    logger.warning(f"Failed to convert args to {input_model.__name__}: {e}")
                    typed_args = arguments
            else:
                typed_args = arguments

            # 4. Execute the tool
            logger.info(f"Executing tool: {tool_name} with args: {arguments}")

            # Run the tool's execute method
            result = await tool.execute(typed_args, context)

            # Format the result
            if hasattr(result, 'output'):
                output = result.output
            elif isinstance(result, dict):
                output = result.get('output', str(result))
            else:
                output = str(result)

            return {
                "output": output,
                "is_error": False,
            }

        except Exception as e:
            logger.error(f"Tool execution failed: {tool_name} - {e}")
            return {
                "output": f"Error executing {tool_name}: {str(e)}",
                "is_error": True,
            }

    async def execute_batch(self, tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Execute multiple tool calls.

        Args:
            tool_calls: List of dicts with 'tool' and 'arguments' keys

        Returns:
            List of result dicts
        """
        results = []
        for call in tool_calls:
            tool_name = call.get("tool", "")
            arguments = call.get("arguments", {})
            result = await self.execute_tool(tool_name, arguments)
            results.append(result)
        return results

    def get_available_tools(self) -> list[str]:
        """List all available tools."""
        return list(self._tool_registry._tools.keys())

    def get_status(self) -> dict[str, Any]:
        """Get bridge status."""
        return {
            "initialized": self._initialized,
            "workspace": self._workspace_dir,
            "tools_available": len(self._tool_registry._tools),
        }
