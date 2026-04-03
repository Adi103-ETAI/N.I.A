"""
Compatibility layer for v3.1.0 plugins.

Provides import aliases for old paths so that existing plugins
continue to work without modification.
"""

import sys
import importlib
import logging

logger = logging.getLogger(__name__)

# Mapping of old imports to new locations
IMPORT_ALIASES = {
    # Tool decorators
    'tara.tools.decorators': 'src.capabilities.decorators',
    'tara.tools.interface': 'src.capabilities.interface',
    
    # Desktop tools
    'tara.tools.desktop.app_launcher': 'src.capabilities.desktop.app_launcher',
    'tara.tools.desktop.window_ops': 'src.capabilities.desktop.window_ops',
    'tara.tools.desktop.screen_ops': 'src.capabilities.desktop.screen',
    'tara.tools.desktop.uia_ops': 'src.capabilities.desktop.uia',
    'tara.tools.desktop.window_manager': 'src.capabilities.desktop.window_registry',
    
    # System tools
    'tara.tools.system.file_ops': 'src.capabilities.system.files',
    'tara.tools.system.input_ops': 'src.capabilities.desktop.input',
    'tara.tools.system.system_ops': 'src.capabilities.system.stats',
    
    # Web tools
    'tara.tools.web.browser_ops': 'src.capabilities.web.browser',
    
    # Memory tools
    'tara.tools.memory.preferences': 'src.core.memory',
    
    # AI tools
    'tara.tools.ai.llm_ops': 'src.models.manager',
    
    # Core imports
    'core.logger': 'src.core.logger',
    'core.memory': 'src.core.memory',
    'core.config': 'src.core.config',
    'core.services': 'src.core.di.service_registry',
    'core.event_bus': 'src.core.bus.events',
    'core.os_context': 'src.core.os.platform',
    'core.engine.system': 'src.core.engine.orchestrator',
    
    # Agent imports
    'nia.agent': 'src.agents.nia.agent',
    'nia.graph': 'src.agents.nia.graph',
    'tara.protocols': 'src.agents.tara.protocols',
    'tara.graph': 'src.agents.tara.graph',
    'iris.agent': 'src.agents.iris.agent',
    'iris.sentry': 'src.agents.iris.sentry',
    'iris.tools': 'src.agents.iris.capture',
    'nola.manager': 'src.agents.nola.manager',
    
    # Model imports
    'models.model_manager': 'src.models.manager',
    'models.safe_llm': 'src.models.safe_llm',
    
    # Persona imports
    'persona.profile': 'src.persona.profile',
    
    # Interface imports
    'interface.banner': 'src.interface.banner',
    
    # Plugins -> Extensions
    'plugins': 'src.extensions',
}


class CompatibilityImporter:
    """
    Custom import hook that redirects old imports to new locations.
    
    This allows v3.1.0 plugins to work without modification.
    """
    
    def __init__(self):
        self.warned = set()
    
    def find_module(self, fullname, path=None):
        """Check if this is a legacy import we should redirect."""
        if fullname in IMPORT_ALIASES:
            return self
        
        # Check parent modules
        parts = fullname.split('.')
        for i in range(len(parts)):
            parent = '.'.join(parts[:i+1])
            if parent in IMPORT_ALIASES:
                return self
        
        return None
    
    def load_module(self, fullname):
        """Redirect import to new location."""
        if fullname in sys.modules:
            return sys.modules[fullname]
        
        # Find redirect target
        redirect_to = IMPORT_ALIASES.get(fullname)
        
        if not redirect_to:
            # Check if parent is aliased
            parts = fullname.split('.')
            for i in range(len(parts), 0, -1):
                parent = '.'.join(parts[:i])
                if parent in IMPORT_ALIASES:
                    # Construct new path
                    suffix = '.'.join(parts[i:])
                    redirect_to = f"{IMPORT_ALIASES[parent]}.{suffix}" if suffix else IMPORT_ALIASES[parent]
                    break
        
        if not redirect_to:
            raise ImportError(f"Cannot redirect {fullname}")
        
        # Warn user (once per import)
        if fullname not in self.warned:
            logger.warning(
                f"Legacy import detected: {fullname}\n"
                f"  -> Redirected to: {redirect_to}\n"
                f"  -> Please update your plugin to use new imports"
            )
            self.warned.add(fullname)
        
        # Import from new location
        try:
            module = importlib.import_module(redirect_to)
            sys.modules[fullname] = module
            return module
        except ImportError as e:
            logger.error(f"Failed to redirect {fullname} -> {redirect_to}: {e}")
            raise


_importer_instance = None


def enable_compatibility_mode():
    """Enable import compatibility for v3.1.0 plugins."""
    global _importer_instance
    if _importer_instance is None:
        _importer_instance = CompatibilityImporter()
        sys.meta_path.insert(0, _importer_instance)
        logger.info("[OK] Plugin compatibility mode enabled for v3.1.0 plugins")


def disable_compatibility_mode():
    """Disable compatibility mode."""
    global _importer_instance
    if _importer_instance is not None:
        sys.meta_path = [
            imp for imp in sys.meta_path 
            if imp is not _importer_instance
        ]
        _importer_instance = None
        logger.info("[OK] Plugin compatibility mode disabled")


def is_compatibility_mode_enabled() -> bool:
    """Check if compatibility mode is currently enabled."""
    return _importer_instance is not None
