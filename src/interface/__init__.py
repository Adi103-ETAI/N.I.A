"""Interface package exports.

Expose the CLI entrypoint `main` from `interface.chat` lazily to avoid
import-time side effects when importing the `interface` package. Call
`interface.main()` to invoke the real CLI entrypoint.
"""

__all__ = ["main"]


def main(*args, **kwargs):
	"""Lazily import and call `interface.chat.main` to avoid importing
	`interface.chat` at module import time.
	"""
	from .chat import main as _main
	return _main(*args, **kwargs)

