"""Sandboxed Shell Tool — Docker-Backed Code Execution.

Provides the ``SandboxedShell`` LangChain tool that routes bash command
execution through the ``DockerEngine``.  TARA uses this as its primary
execution mechanism for file operations, git commands, and code runs —
keeping all such work inside a containerised Linux environment and off
the host filesystem.

Tools:
    SandboxedShell  — Execute bash in a persistent ``/workspace`` Linux container

Usage via TARA::

    # TARA calls this automatically when the user asks to run code or write files.
    result = sandboxed_shell.invoke({
        "command": "echo 'hello' > hello.txt && cat hello.txt",
        "session_id": "abc-123",
    })
"""
import shlex
import logging
from typing import Type, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

from src.infrastructure.container_engine.manager import DockerEngine
from src.core.config import get_settings

logger = logging.getLogger("TARA.Tools.Execution")
settings = get_settings()

class SandboxedShellInput(BaseModel):
    command: str = Field(description="The bash script/command to execute.")
    session_id: str = Field(
        default="default", 
        description="The persistent session identifier (UUID). Use the EXACT ID provided in your system prompt. Do NOT use '{session_id}' or placeholders."
    )
    timeout: int = Field(default=60, description="Execution timeout in seconds.")
    background: bool = Field(default=False, description="Run in background (detached).")

class SandboxedShell(BaseTool):
    name: str = "sandboxed_shell"
    description: str = "Execute bash commands in a secure Linux sandbox. Use this for ALL file operations, git commands, and code execution. Input should be a valid bash script."
    args_schema: Type[BaseModel] = SandboxedShellInput
    
    # Metadata for TARA 2.0 — use Field(default_factory) to avoid Pydantic v2 mutable default error
    metadata: Optional[dict] = Field(
        default_factory=lambda: {"security_level": "sandboxed", "type": "execution"}
    )

    def _run(self, command: str, session_id: str = "default", timeout: int = 60, background: bool = False) -> str:
        """Execute the command in the sandbox."""
        try:
            # Get Engine Singleton
            engine = DockerEngine()
            
            # Critical Security Wrapper: Ensure proper shell handling
            # We use sh -c to allow pipes, redirects, and compound commands
            safe_command = f"sh -c {shlex.quote(command)}"
            
            # Execute
            # Default to python:3.11-slim as per Phase 1
            image = "python:3.11-slim" 
            
            logger.info(f"Executing in sandbox [{session_id}]: {command[:50]}...")
            if settings.DEBUG:
                print(f"DEBUG SHELL COMMAND: {command}")
            
            # Phase 3 Update: Pass session_id to engine
            exit_code, stdout, stderr = engine.run_command(
                image=image,
                command=safe_command,
                session_id=session_id,
                # Mounts are now handled internally by manager fallback if session inactive
                # mounts=mounts, 
            )
            
            # Format Output
            output_parts = []
            if stdout:
                output_parts.append(f"STDOUT:\n{stdout}")
            if stderr:
                output_parts.append(f"STDERR:\n{stderr}")
            if not stdout and not stderr:
                output_parts.append("(No Output)")
            
            output_str = "\n".join(output_parts)
            
            if exit_code != 0:
                return f"❌ Command failed (Exit Code {exit_code}):\n{output_str}"
            
            return f"✅ Success:\n{output_str}"

        except Exception as e:
            logger.error(f"Sandbox execution failed: {e}")
            return f"❌ Execution Error: {str(e)}"

    async def _arun(self, command: str, session_id: str = "default", timeout: int = 60, background: bool = False) -> str:
        """Async support (Run in threadpool)."""
        # Docker SDK is synchronous, so we run it in a thread
        import asyncio
        return await asyncio.to_thread(self._run, command, session_id, timeout, background)
