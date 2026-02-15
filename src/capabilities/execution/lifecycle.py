"""
TARA 2.0 Lifecycle Tools.

Phase 3: Session Management.
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
    
    metadata: Optional[dict] = {
        "security_level": "standard",
        "type": "lifecycle"
    }

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
    
    metadata: Optional[dict] = {
        "security_level": "standard",
        "type": "lifecycle"
    }

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
