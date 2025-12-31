"""System monitoring tools for TARA.

Provides CPU, RAM, and disk usage monitoring via psutil.
"""
from __future__ import annotations

from typing import Any, Type

# Try to import psutil
try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False
    psutil = None  # type: ignore

# Try to import LangChain tools
try:
    from langchain_core.tools import BaseTool
    from pydantic import BaseModel, Field
    _HAS_LANGCHAIN = True
except ImportError:
    _HAS_LANGCHAIN = False
    BaseTool = object  # type: ignore
    BaseModel = object  # type: ignore


class SystemStatsInput(BaseModel):
    """Input for SystemStatsTool - no input required."""
    pass


class SystemStatsTool(BaseTool):
    """Tool to get current system statistics (CPU, RAM usage).
    
    Returns a formatted string with CPU load and RAM usage.
    """
    
    name: str = "system_stats"
    description: str = (
        "Get current system health statistics including CPU load and RAM usage. "
        "Use this when asked about system health, performance, or resource usage."
    )
    args_schema: Type[BaseModel] = SystemStatsInput
    
    def _run(self, **kwargs: Any) -> str:
        """Get system statistics."""
        if not _HAS_PSUTIL:
            return "Error: psutil not installed. Run: pip install psutil"
        
        try:
            # CPU usage (average over 1 second)
            cpu_percent = psutil.cpu_percent(interval=0.5)
            
            # RAM usage
            memory = psutil.virtual_memory()
            ram_percent = memory.percent
            ram_used_gb = memory.used / (1024 ** 3)
            ram_total_gb = memory.total / (1024 ** 3)
            
            return (
                f"CPU Load: {cpu_percent}% | "
                f"RAM Usage: {ram_percent}% ({ram_used_gb:.1f}GB / {ram_total_gb:.1f}GB)"
            )
            
        except Exception as exc:
            return f"Error getting system stats: {exc}"
    
    async def _arun(self, **kwargs: Any) -> str:
        """Async version - just calls sync."""
        return self._run(**kwargs)


class DiskStatsTool(BaseTool):
    """Tool to get disk usage statistics."""
    
    name: str = "disk_stats"
    description: str = (
        "Get disk space usage for the system drive. "
        "Use when asked about storage, disk space, or free space."
    )
    args_schema: Type[BaseModel] = SystemStatsInput
    
    def _run(self, **kwargs: Any) -> str:
        """Get disk statistics."""
        if not _HAS_PSUTIL:
            return "Error: psutil not installed. Run: pip install psutil"
        
        try:
            # Get disk usage for root drive
            disk = psutil.disk_usage('/')
            
            used_gb = disk.used / (1024 ** 3)
            total_gb = disk.total / (1024 ** 3)
            free_gb = disk.free / (1024 ** 3)
            
            return (
                f"Disk Usage: {disk.percent}% | "
                f"Used: {used_gb:.1f}GB | Free: {free_gb:.1f}GB | Total: {total_gb:.1f}GB"
            )
            
        except Exception as exc:
            return f"Error getting disk stats: {exc}"
    
    async def _arun(self, **kwargs: Any) -> str:
        """Async version - just calls sync."""
        return self._run(**kwargs)


class TimeTool(BaseTool):
    """Tool to get current local time and date.
    
    Returns the current time in a human-readable format.
    Much faster than web search for time queries.
    """
    
    name: str = "current_time"
    description: str = (
        "Get the current local time and date. "
        "Use this when asked about the time, date, day of week, or 'what time is it'."
    )
    args_schema: Type[BaseModel] = SystemStatsInput  # No input needed
    
    def _run(self, **kwargs: Any) -> str:
        """Get current time and date."""
        from datetime import datetime
        
        try:
            now = datetime.now()
            
            # Format: "Monday, Dec 30, 2025 at 6:45 PM"
            day_name = now.strftime("%A")
            date_str = now.strftime("%b %d, %Y")
            time_str = now.strftime("%I:%M %p").lstrip("0")
            
            return f"{day_name}, {date_str} at {time_str}"
            
        except Exception as exc:
            return f"Error getting time: {exc}"
    
    async def _arun(self, **kwargs: Any) -> str:
        """Async version - just calls sync."""
        return self._run(**kwargs)
