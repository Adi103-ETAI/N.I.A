"""Extension loader with v3.1.0 compatibility."""

import sys
from pathlib import Path
from typing import List, Optional
import importlib.util
import logging

from src.extensions.base import BaseExtension
from src.extensions.compat import enable_compatibility_mode

logger = logging.getLogger(__name__)


class ExtensionLoader:
    """Load user extensions with backward compatibility."""
    
    def __init__(
        self, 
        extensions_dir: Optional[Path] = None, 
        enable_compat: bool = True
    ):
        """
        Initialize extension loader.
        
        Args:
            extensions_dir: Directory containing extensions.
                            Defaults to project_root/extensions/custom/
            enable_compat: Enable v3.1.0 plugin compatibility mode
        """
        if extensions_dir is None:
            # Default to project root's extensions/custom/ directory
            extensions_dir = Path(__file__).parent.parent.parent / "extensions" / "custom"
        
        self.extensions_dir = Path(extensions_dir)
        self.loaded_extensions: List[BaseExtension] = []
        
        # Enable v3.1.0 plugin compatibility
        if enable_compat:
            enable_compatibility_mode()
    
    def discover_extensions(self) -> List[Path]:
        """Find all extension files."""
        if not self.extensions_dir.exists():
            logger.warning(f"Extensions directory not found: {self.extensions_dir}")
            return []
        
        extensions = []
        
        # Find Python files in custom/ subdirectory
        extensions.extend(self.extensions_dir.glob("*.py"))
        
        # Also check subdirectories for packages
        for subdir in self.extensions_dir.iterdir():
            if subdir.is_dir() and (subdir / "__init__.py").exists():
                extensions.append(subdir / "__init__.py")
        
        # Filter out __init__.py from root
        return [e for e in extensions if e.name != "__init__.py" or e.parent != self.extensions_dir]
    
    def load_extension(self, extension_path: Path) -> Optional[BaseExtension]:
        """Load a single extension with error handling."""
        try:
            # Determine module name
            if extension_path.name == "__init__.py":
                module_name = extension_path.parent.name
            else:
                module_name = extension_path.stem
            
            # Load module
            spec = importlib.util.spec_from_file_location(
                module_name,
                extension_path
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            
            # Find extension class
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and 
                    issubclass(attr, BaseExtension) and 
                    attr is not BaseExtension):
                    
                    # Instantiate and initialize
                    instance = attr()
                    instance.initialize()
                    instance._initialized = True
                    
                    logger.info(f"[OK] Loaded extension: {instance.name} v{instance.version}")
                    return instance
            
            logger.warning(f"No extension class found in {extension_path.name}")
            return None
            
        except Exception as e:
            logger.error(f"Failed to load extension {extension_path.name}: {e}", exc_info=True)
            return None
    
    def load_all(self) -> List[BaseExtension]:
        """Load all extensions."""
        extension_files = self.discover_extensions()
        logger.info(f"Found {len(extension_files)} extension(s) to load")
        
        for ext_file in extension_files:
            ext = self.load_extension(ext_file)
            if ext:
                self.loaded_extensions.append(ext)
        
        logger.info(f"[OK] Loaded {len(self.loaded_extensions)} extension(s)")
        return self.loaded_extensions
    
    def get_extension(self, name: str) -> Optional[BaseExtension]:
        """Get a loaded extension by name."""
        for ext in self.loaded_extensions:
            if ext.name == name:
                return ext
        return None
    
    def shutdown_all(self) -> None:
        """Shutdown all loaded extensions."""
        for ext in self.loaded_extensions:
            try:
                ext.shutdown()
                ext._initialized = False
                logger.info(f"[OK] Shutdown extension: {ext.name}")
            except Exception as e:
                logger.error(f"Error shutting down extension {ext.name}: {e}")
        
        self.loaded_extensions.clear()


# =============================================================================
# Plugin Watcher Compatibility (for main.py backward compat)
# =============================================================================

_loader_instance = None

def start_plugin_watcher() -> Optional[ExtensionLoader]:
    """Start the extension loader (backward compat for old plugin watcher)."""
    global _loader_instance
    if _loader_instance is None:
        _loader_instance = ExtensionLoader()
        _loader_instance.load_all()
    return _loader_instance


def stop_plugin_watcher(loader: Optional[ExtensionLoader] = None) -> None:
    """Stop the extension loader (backward compat for old plugin watcher)."""
    global _loader_instance
    target = loader or _loader_instance
    if target:
        target.shutdown_all()
    _loader_instance = None
