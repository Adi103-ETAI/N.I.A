"""Session Lifecycle Tools — Docker Workspace Management.

Provides LangChain ``BaseTool`` wrappers for starting and stopping persistent
Docker workspace sessions.  These tools are used by TARA to manage the
container lifecycle for complex, multi-step tasks that require a persistent
``/workspace`` directory across tool calls.

Tools:
    StartSession  — Calls ``DockerEngine.start_session(session_id)``
    EndSession    — Calls ``DockerEngine.stop_session(session_id)`` and cleans up

Usage via TARA::

    # TARA will automatically call start_session at the beginning of a task
    # and end_session when the task is complete or on error.
"""
from typing import Type, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

from src.infrastructure.container_engine.manager import DockerEngine
from src.core.logger import setup_logger

logger = setup_logger("TARA.Tools.Lifecycle")

class StartSessionInput(BaseModel):
    session_id: str = Field(description="Unique identifier for the session.")

class StartSession(BaseTool):
    name: str = "start_session"
    description: str = "Initialize a persistent Docker workspace session. Use this at the start of complex tasks."
    args_schema: Type[BaseModel] = StartSessionInput
    
    metadata: Optional[dict] = Field(
        default_factory=lambda: {"security_level": "standard", "type": "lifecycle"}
    )

    def _run(self, session_id: str) -> str:
        engine = DockerEngine()
        try:
            result = engine.start_session(session_id)
            return f"✅ {result}"
        except Exception as e:
            return f"❌ Failed to start session: {e}"

    async def _arun(self, session_id: str) -> str:
        # Simple wrapper, engine calls are sync but fast enough?
        # Ideally threading
        import asyncio
        return await asyncio.to_thread(self._run, session_id)


class EndSessionInput(BaseModel):
    session_id: str = Field(description="Unique identifier for the session.")

class EndSession(BaseTool):
    name: str = "end_session"
    description: str = "Destroy a persistent Docker workspace session. Cleanup after task completion."
    args_schema: Type[BaseModel] = EndSessionInput
    
    metadata: Optional[dict] = Field(
        default_factory=lambda: {"security_level": "standard", "type": "lifecycle"}
    )

    def _run(self, session_id: str) -> str:
        engine = DockerEngine()
        try:
            success = engine.stop_session(session_id)
            if success:
                return f"✅ Session {session_id} ended (Container removed)"
            else:
                return f"⚠️ Session {session_id} not found or failed to stop"
        except Exception as e:
            return f"❌ Failed to end session: {e}"

    async def _arun(self, session_id: str) -> str:
        import asyncio
        return await asyncio.to_thread(self._run, session_id)
