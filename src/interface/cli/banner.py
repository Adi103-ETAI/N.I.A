"""CLI Banner shim — re-exports from src.interface.banner.

This module exists because code imports `from src.interface.cli.banner import BANNER`
but the actual banner lives at `src/interface/banner.py`.
"""
from src.interface.banner import BANNER, MINI_BANNER, VERSION

__all__ = ["BANNER", "MINI_BANNER", "VERSION"]
