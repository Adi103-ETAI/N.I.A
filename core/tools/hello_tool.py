from typing import Any
from core.base_tool import BaseTool


class HelloTool(BaseTool):
    """Simple greeting tool.

    Expects keyword args and returns a dict with a greeting message.
    """
    name = 'hello'
    description = 'Return a greeting.'

    def run(self, *args, **params) -> Any:
        # Backwards-compatible: support either a single positional dict
        # (legacy tests/tools call `run({'who': 'NIA'})`) or keyword args.
        if args and isinstance(args[0], dict):
            params = dict(args[0])
        who = params.get('who', 'world')
        return {'greeting': f'Hello, {who}!'}
