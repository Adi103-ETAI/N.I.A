"""
Session Factory.

Helpers for managing session-specific paths and configurations.
"""
import os
from pathlib import Path
from typing import Dict, Any

from src.core.config import get_settings

settings = get_settings()

class SessionBuilder:
    """Helper to build session configurations."""
    
    @staticmethod
    def get_session_mounts(session_id: str) -> Dict[str, Dict[str, str]]:
        """
        Prepare host directory and return Docker volume configuration.
        
        Args:
            session_id: Unique session identifier.
            
        Returns:
            Dictionary suitable for the `volumes` argument in docker-py.
        """
        # Ensure host path exists
        # Path: data/sandbox_mounts/<session_id>/
        # relative to project root
        
        host_mount_base = settings.BASE_DIR / "data" / "sandbox_mounts" / session_id
        host_mount_base.mkdir(parents=True, exist_ok=True)
        
        # Return docker-py volumes dict
        # {host_path: {'bind': container_path, 'mode': 'rw'}}
        return {
            str(host_mount_base.absolute()): {
                'bind': '/workspace',
                'mode': 'rw'
            }
        }
