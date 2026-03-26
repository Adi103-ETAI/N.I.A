# ----------------------------------------------------------------------
# FILE: nia/agent.py
# VERSION: 2.5.2
# STATUS: SYSTEM HUB - Core Supervisor Implementation
# FEATURES: Dynamic Provider Access, Protocol-based Routing, SafeLLM Integration
# ----------------------------------------------------------------------
from __future__ import annotations

import asyncio
import time
import random
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

from src.core.logger import setup_logger

# --- CRITICAL IMPORTS ---
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
# Removed top-level get_smart_model to prevent circular import/mocking issues
from src.core.config import settings as config

# =============================================================================
# TYPE_CHECKING Block (IDE-only, no runtime cost)
# =============================================================================

if TYPE_CHECKING:
    from src.agents.iris.agent import IrisAgent
    from typing import Optional


# =============================================================================
# Protocol for Agent Interface (Duck Typing Support)
# =============================================================================

@runtime_checkable
class AgentProtocol(Protocol):
    """Protocol defining the expected interface for pluggable agents.
    
    Agents can be IrisAgent, TaraAgent, or any class implementing:
    - process(state: Dict) -> Dict
    - run(query: str) -> str
    """
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]: ...
    def run(self, query: str) -> str: ...


logger = setup_logger("NIA.Supervisor")

class SupervisorAgent:
    """NIA Supervisor Agent — Persona & Chat Handler.
    
    Phase 3: This agent is now purely responsible for conversation.
    Routing is handled upstream by the Decision Core (router_node).
    If you are seeing this node execute, the Router has already
    classified the user's intent as 'chat'.
    
    v2.5.2: Dynamic Provider Access
        The LLM is fetched via `@property` on each access, NOT stored at init.
        This enables hot-swap provider switching via ModelManager.set_active_provider()
    
    Design Patterns:
        - **Protocol-based DI**: Agents injected via AgentProtocol interface
        - **Dynamic LLM Access**: Uses @property for hot-swap support
    
    Attributes:
        iris_agent: Optional IrisAgent instance.
        llm: Property - fetches model dynamically from ModelManager.
    """
    
    
    def __init__(
        self,
        iris_agent: AgentProtocol | None = None,  # Vision agent
        model_type: str = "smart",
        temperature: float = 0.7,
    ) -> None:
        """Initialize the SupervisorAgent with typed dependencies.
        
        Args:
            iris_agent: IrisAgent instance for vision tasks.
                       Must implement AgentProtocol if provided.
            model_type: LLM model type ('smart' or 'fast').
            temperature: LLM temperature setting (0.0-2.0).
        """
        self.iris_agent: AgentProtocol | None = iris_agent
        # Phase 3: Routing handled upstream by Decision Core
        
        # v2.5.2: Store temperature for dynamic LLM access
        self._temperature = temperature
        
        # Verify LLM access works at startup (fail-fast)
        try:
            _ = self.llm  # Access property to verify connectivity
            logger.info(f"🧠 SupervisorAgent ready (dynamic LLM via ModelManager)")
        except Exception as e:
            logger.error(f"Failed to access LLM: {e}")
            raise RuntimeError(f"SupervisorAgent cannot start without LLM: {e}") from e
        
        # System Prompt: Prefer Persona module, fallback to text file
        try:
            from src.persona.profile import get_system_prompt
            prompt_text = get_system_prompt()
            logger.debug("System prompt loaded from Persona module")
        except Exception as persona_err:
            logger.warning(f"Persona module failed ({persona_err}), falling back to file")
            try:
                with open(config.SUPERVISOR_PROMPT_FILE, "r", encoding="utf-8") as f:
                    prompt_text = f.read()
            except FileNotFoundError:
                prompt_text = "You are NIA. I am ready to chat."
            
        self.system_prompt = prompt_text
    
    @property
    def llm(self):
        """Get LLM dynamically from ModelManager.
        
        v2.5.2: Fetched on each access to support hot-swap provider switching.
        When ModelManager.set_active_provider() is called, subsequent accesses
        will automatically use the new provider.
        """
        from src.models.manager import get_smart_model
        return get_smart_model(temperature=self._temperature)

    async def _build_context(self, state: Dict[str, Any]) -> Tuple[List[BaseMessage], List[BaseMessage]]:
        """
        Build the message context for LLM invocation.
        
        Injects relevant memory context between system prompt and conversation history.
        Regenerates system prompt each call for dynamic 80/20 personality.
        
        Args:
            state: The current agent state containing messages.
            
        Returns:
            Tuple of (current_messages, retry_buffer).
            - current_messages: System prompt + memory context + conversation history.
            - retry_buffer: Empty list for retry attempts.
        """
        # 🎲 DYNAMIC PERSONA: Regenerate system prompt each call for fresh 80/20 roll
        from src.persona.profile import get_system_prompt
        fresh_system_prompt = get_system_prompt()
        
        # 🛠️ SKILL INJECTION (Phase 2)
        from src.core.skills.loader import get_skills_prompt
        skills_doc = get_skills_prompt()
        if skills_doc:
            fresh_system_prompt += f"\n\n{skills_doc}"
        
        current_messages: List[BaseMessage] = [SystemMessage(content=fresh_system_prompt)]
        
        # Extract messages from state
        state_messages = state.get("messages", []) if isinstance(state.get("messages"), list) else []
        
        # 🧠 MEMORY INJECTION: Get relevant context for the last user message
        # RIPPLE FIX: Await the async memory retrieval
        memory_content = await self._get_memory_context(state_messages)
        if memory_content:
            logger.debug(f"🧠 [AGENT] Injected Memory Context: {len(memory_content)} chars")
            current_messages.append(SystemMessage(content=memory_content))
        else:
            logger.debug("🧠 [AGENT] No memory context found for this query")
        
        # Append conversation history
        current_messages.extend(state_messages)
            
        return current_messages, []
    
    async def _get_memory_context(self, messages: List[BaseMessage]) -> Optional[str]:
        """
        Retrieve relevant memory context for the last user message.
        
        Args:
            messages: List of conversation messages.
            
        Returns:
            Formatted memory context string, or None if unavailable.
        """
        # Get memory from ServiceRegistry
        from src.core.di import ServiceRegistry
        memory = ServiceRegistry.get("memory")
        
        if memory is None:
            return None
        
        # Find the last user message
        user_query = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                user_query = msg.content
                break
        
        if not user_query:
            return None
        
        try:
            # Async call to memory (ChromaDB semantic search)
            # Use asyncio.to_thread for blocking sync calls until MemoryManager is fully async
            # But wait! If MemoryManager HAS async methods, use them.
            # Assuming MemoryManager might still be sync-heavy, we wrap in to_thread if needed.
            # However, the user request says: "await memory.get_all_preferences()".
            # This implies get_all_preferences might NOT be async yet, OR it IS async.
            # If it IS async, we await it. If it's sync, we shouldn't await it unless we wrap it.
            # Inspecting common patterns: if user says "un-awaited coroutine", it means
            # get_all_preferences IS ALREADY async.
            
            # CRITICAL: User said "Issue: ... calling the newly asynchronous MemoryManager.get_all_preferences() synchronously."
            # This means get_all_preferences IS ASYNC.
            
            # Check for arecall_episodes or wrap recall_episodes
            if hasattr(memory, "arecall_episodes"):
                episodes = await memory.arecall_episodes(user_query, n=3)
            else:
                # Fallback: Wrap sync recall
                episodes = await asyncio.to_thread(memory._recall_episodes_sync, user_query, n=3)
            
            # User specifically mentioned get_all_preferences is async
            if asyncio.iscoroutinefunction(memory.get_all_preferences):
                preferences = await memory.get_all_preferences()
            else:
                # If it's not async (unexpected given the report), run it sync
                preferences = memory.get_all_preferences()
            
            # Build context string
            parts = []
            
            if episodes:
                parts.append("### Relevant Past Conversations:")
                for i, ep in enumerate(episodes[:3], 1):
                    parts.append(f"  {i}. {ep[:200]}...")  # Truncate long episodes
                    
            if preferences:
                parts.append("### User Preferences:")
                for key, val in list(preferences.items())[:5]:  # Limit to 5 prefs
                    parts.append(f"  - {key}: {val}")
            
            if parts:
                return "🧠 MEMORY CONTEXT (for reference):\n" + "\n".join(parts)
                
        except Exception as e:
            logger.warning(f"Memory retrieval failed: {e}")
        
        return None
    
    def _append_retry(
        self, 
        retry_buffer: List[BaseMessage], 
        content: str, 
        error: str
    ) -> None:
        """Append retry messages to buffer (mutates in-place)."""
        retry_buffer.append(AIMessage(content=content))
        retry_buffer.append(HumanMessage(content=f"SYSTEM ERROR: {error}"))
    
    def _save_to_memory(self, user_query: Optional[str], ai_response: str) -> None:
        """
        Persist the conversation turn to MemoryManager.
        
        Stores both user input (if available) and AI response to ChromaDB
        for future semantic retrieval.
        
        Args:
            user_query: The user's input message (may be None).
            ai_response: The AI's response content.
        """
        from src.core.di import ServiceRegistry
        memory = ServiceRegistry.get("memory")
        
        if memory is None:
            return
        
        try:
            # Store user message
            if user_query:
                memory._store_episode_sync(user_query, role="user", metadata=None)
                logger.debug(f"Saved user episode: {user_query[:50]}...")
            
            # Store AI response
            if ai_response:
                memory._store_episode_sync(ai_response, role="assistant", metadata=None)
                logger.debug(f"Saved AI episode: {ai_response[:50]}...")
                
        except Exception as e:
            logger.warning(f"Memory save failed: {e}")

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synchronous Chat Execution (fallback for non-async contexts).
        
        Phase 3: Simplified to chat-only. No routing, no gatekeeper.
        """
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                current_messages, _ = loop.run_until_complete(self._build_context(state))
            else:
                current_messages, _ = asyncio.run(self._build_context(state))
        except RuntimeError:
            logger.warning("Sync process() could not resolve async context, proceeding without memory")
            from src.persona.profile import get_system_prompt
            current_messages = [SystemMessage(content=get_system_prompt())]
            current_messages.extend(state.get("messages", []))
        
        try:
            response = self.llm.invoke(current_messages)
            content = response.content
            return {"messages": [AIMessage(content=content)]}
        except Exception as e:
            logger.error(f"Chat LLM Failed (sync): {e}")
            return {"messages": [AIMessage(content="I'm having a bit of trouble thinking right now.")]}

    async def aprocess(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Async Chat Execution Logic (Native Async).
        
        Now simplified to purely chat processing. Routing is handled upstream by the Router Node.
        """
        # 1. Build Context (Messages + Memory)
        current_messages, _ = await self._build_context(state)
        
        # Extract last user query for memory persistence logic (if needed)
        user_query = None
        state_messages = state.get("messages", [])
        if isinstance(state_messages, list):
            for msg in reversed(state_messages):
                if isinstance(msg, HumanMessage):
                    user_query = msg.content
                    break
        
        try:
            # 2. Invoke LLM (Chat Mode)
            response = await self.llm.ainvoke(current_messages)
            content = response.content
            
            # 3. Simple Memory Save (if configured to save every turn)
            self._save_to_memory(user_query, content)
            
            # 4. Return Chat Response (Next -> END)
            # No routing needed here. The graph connects supervisor -> END by default for chat.
            return {"messages": [AIMessage(content=content)]}
            
        except Exception as e:
            logger.error(f"Chat LLM Failed: {e}")
            return {"messages": [AIMessage(content="I'm having a bit of trouble thinking right now.")]}

