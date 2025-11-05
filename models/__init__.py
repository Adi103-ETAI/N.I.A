"""Models package public exports.

Expose ModelManager for convenient import: `from models import ModelManager`.
"""

try:
	from .model_manager import ModelManager
	__all__ = ["ModelManager"]
except Exception:
	ModelManager = None
	__all__ = []

