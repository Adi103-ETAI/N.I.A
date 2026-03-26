"""N.I.A. Tool Registry — Sprint 1.

Provides a queryable registry of tool manifests (YAML-declared),
exposing scope, timeout, and reversibility metadata for every tool.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    yaml = None  # type: ignore
    _HAS_YAML = False

from pydantic import BaseModel, ConfigDict
from src.core.policy.scopes import CapabilityScope
from src.core.logger import setup_logger

logger = setup_logger("ToolRegistry")

class ToolManifest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    name: str
    scope: CapabilityScope
    reversible: bool = False
    description: str
    timeout: int = 300

class ToolRegistry:
    def __init__(self, manifests_dir: str = "src/capabilities/manifests"):
        self.manifests_dir = manifests_dir
        self._manifests: Dict[str, ToolManifest] = {}
        self._loaded = False
        
    def _load_manifests(self):
        if self._loaded:
            return
            
        if not os.path.exists(self.manifests_dir):
            os.makedirs(self.manifests_dir, exist_ok=True)
            logger.warning(f"Manifests directory created: {self.manifests_dir}")
            self._loaded = True
            return
            
        for filename in os.listdir(self.manifests_dir):
            if filename.endswith((".yaml", ".yml")):
                path = os.path.join(self.manifests_dir, filename)
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        data = yaml.safe_load(f)
                        
                    if not data:
                        continue
                        
                    manifest = ToolManifest(**data)
                    self._manifests[manifest.name] = manifest
                except Exception as e:
                    logger.error(f"Failed to load manifest {path}: {e}")
                    
        self._loaded = True

    def get_manifest(self, name: str) -> Optional[ToolManifest]:
        self._load_manifests()
        return self._manifests.get(name)
        
    def get_scope(self, name: str) -> CapabilityScope:
        self._load_manifests()
        manifest = self._manifests.get(name)
        # Default to a highly restricted scope 'EXECUTE' if no manifest exists
        return manifest.scope if manifest else CapabilityScope.EXECUTE
        
    def get_all_by_scope(self, scope: CapabilityScope) -> List[ToolManifest]:
        self._load_manifests()
        return [m for m in self._manifests.values() if m.scope == scope]

# Global registry instance
global_registry = ToolRegistry()

def get_tool(name: str):
    """Retrieve the LangChain tool by name (lazy-loads interface)."""
    from src.capabilities.interface import get_tool_by_name
    return get_tool_by_name(name)

def get_tool_manifest(name: str) -> Optional[ToolManifest]:
    """Retrieve the declarative manifest for a tool."""
    return global_registry.get_manifest(name)

def get_scope(name: str) -> CapabilityScope:
    return global_registry.get_scope(name)

def get_all_by_scope(scope: CapabilityScope) -> List[ToolManifest]:
    return global_registry.get_all_by_scope(scope)
