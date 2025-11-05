"""Interface package exports.

Expose the CLI entrypoint `main` from `interface.chat` as `interface.main`.
"""

try:
	from .chat import main
	__all__ = ["main"]
except Exception:
	main = None
	__all__ = []

