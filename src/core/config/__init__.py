"""src/core/config — N.I.A. Centralized Configuration Package.

All imports from this package are identical to the old flat src.core.config module.
Zero changes required in any consumer — this package is a drop-in replacement.

Data files (JSON/YAML) are now at src/core/config/defaults/ instead of root config/.

Public API:
    from src.core.config import Settings, settings, get_settings
    from src.core.config import get_embedding_function, get_browser_path
    from src.core.config import CONFIG_DATA_DIR
"""
from src.core.config.settings import (
    Settings,
    settings,
    get_settings,
    get_embedding_function,
    get_browser_path,
    CONFIG_DATA_DIR,
    _get_base_dir,
)

__all__ = [
    "Settings",
    "settings",
    "get_settings",
    "get_embedding_function",
    "get_browser_path",
    "CONFIG_DATA_DIR",
    "_get_base_dir",
]
