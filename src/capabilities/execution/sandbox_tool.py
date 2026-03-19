import logging
from typing import Optional, Type
from pydantic import BaseModel, Field

from langchain_core.tools import BaseTool
from src.infrastructure.container_engine.sandbox import StaticSandbox

logger = logging.getLogger("N.I.A.Execution")

class RunInSandboxInput(BaseModel):
    command: str = Field(description="The bash script or command to execute.")
    timeout: int = Field(default=120, description="Execution timeout in seconds.")

class RunInSandboxTool(BaseTool):
    """Langchain tool to securely route bash commands to the static N.I.A Docker sandbox."""
    name: str = "run_in_sandbox"
    description: str = (
        "Execute bash commands in a secure Linux sandbox container. "
        "Use this for running scripts, installing dependencies (npm/pip), and executing code. "
        "The current working directory is mounted to /workspace inside the sandbox, "
        "so files created here will appear on your local filesystem."
    )
    args_schema: Type[BaseModel] = RunInSandboxInput
    
    metadata: Optional[dict] = Field(
        default_factory=lambda: {"security_level": "sandboxed"}
    )

    def _run(self, command: str, timeout: int = 120) -> str:
        """Synchronous implementation, but we override _arun for actual use."""
        import asyncio
        return asyncio.run(self._arun(command, timeout))

    async def _arun(self, command: str, timeout: int = 120) -> str:
        """Execute the command in the persistent static sandbox."""
        sandbox = StaticSandbox.get_instance()
        
        # Ensure it's running before executing
        if not sandbox.container:
            started = sandbox.start()
            if not started:
                return "❌ Error: Failed to start the static sandbox environment."

        logger.info(f"Running in sandbox: {command[:50]}...")
        
        exit_code, output = await sandbox.execute(command, timeout=timeout)
        
        if exit_code != 0:
            return f"❌ Command failed (Exit Code {exit_code}):\\n{output}"
            
        return f"✅ Success:\\n{output}"
