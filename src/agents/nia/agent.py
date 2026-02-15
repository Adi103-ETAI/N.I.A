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
from src.agents.nia.gatekeeper import RoutingGatekeeper
# ADAPTED: import settings as config to match user variable name but use correct source
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
    """NIA Supervisor Agent - Orchestrates TARA and IRIS via Protocol-based routing.
    
    v2.5.2: Dynamic Provider Access
        The LLM is fetched via `@property` on each access, NOT stored at init.
        This enables hot-swap provider switching via ModelManager.set_active_provider()
        without restarting the application.
    
    Data Flow:
        User Input -> SupervisorAgent -> SafeLLM -> ModelManager -> Active Provider
                                            ^
                                            |__ Auto-fallback on 429/503 errors
    
    Routing Targets:
        - **TARA**: Desktop automation, browser control, file operations
        - **IRIS**: Vision tasks (screen analysis, webcam capture)
        - **CHAT**: General conversation, information queries
    
    Design Patterns:
        - **Protocol-based DI**: Agents injected via AgentProtocol interface
        - **Gated Routing**: RoutingGatekeeper validates LLM decisions
        - **Dynamic LLM Access**: Uses @property for hot-swap support
        - **SafeLLM Wrapped**: All LLM calls protected by circuit breaker
    
    Attributes:
        iris_agent: IrisAgent instance for vision tasks.
        iris_agent: IrisAgent instance for vision tasks.
        gatekeeper: RoutingGatekeeper for LLM response validation.
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
        self.gatekeeper: RoutingGatekeeper = RoutingGatekeeper()
        
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
                prompt_text = "You are NIA. Route commands to TARA or IRIS."
            
        self.system_prompt = prompt_text + "\n\n### CRITICAL: ROUTE SILENTLY. Example: 'ROUTE:TARA: kill notepad'"
    
    @property
    def llm(self):
        """Get LLM dynamically from ModelManager.
        
        v2.5.2: Fetched on each access to support hot-swap provider switching.
        When ModelManager.set_active_provider() is called, subsequent accesses
        will automatically use the new provider.
        """
        from src.models.manager import get_smart_model
        return get_smart_model(temperature=self._temperature)
    

    
    def _decompose_command(self, command: str) -> List[str]:
        """Decompose a compound command into individual sub-commands.
        
        Handles patterns like:
        - "kill notepad and brave" -> ["kill notepad", "kill brave"]
        - "open chrome, then open notepad" -> ["open chrome", "open notepad"]
        - "1. do X 2. do Y" -> ["do X", "do Y"]
        
        Args:
            command: The raw command string.
            
        Returns:
            List of individual commands to execute.
        """
        import re
        
        # Already a single command? Return as-is
        if not any(delim in command.lower() for delim in [' and ', ' then ', ', ', '\n', '1.', '2.']):
            return [command.strip()]
        
        sub_commands = []
        
        # Pattern 1: "kill X and Y" -> expand verb to each target
        # Look for pattern: <verb> <target1> and <target2>
        and_match = re.match(r'^(\w+)\s+(.+?)\s+and\s+(.+)$', command, re.IGNORECASE)
        if and_match:
            verb = and_match.group(1)
            target1 = and_match.group(2).strip()
            target2 = and_match.group(3).strip()
            return [f"{verb} {target1}", f"{verb} {target2}"]
        
        # Pattern 2: Numbered list "1. do X 2. do Y"
        numbered = re.findall(r'\d+\.\s*(.+?)(?=\d+\.|$)', command)
        if numbered:
            return [cmd.strip() for cmd in numbered if cmd.strip()]
        
        # Pattern 3: Comma or "then" separated
        if ', then ' in command.lower():
            parts = re.split(r',\s*then\s*', command, flags=re.IGNORECASE)
            return [p.strip() for p in parts if p.strip()]
        
        if ', ' in command:
            parts = command.split(', ')
            return [p.strip() for p in parts if p.strip()]
        
        # Pattern 4: Newline separated
        if '\n' in command:
            parts = command.split('\n')
            return [p.strip() for p in parts if p.strip()]
        
        # Fallback: return as single command
        return [command.strip()]

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
        from src.core.registry import ServiceRegistry
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
    
    def _handle_validation(
        self, 
        validation: Dict[str, Any], 
        content: str,
        base_metadata: Dict[str, Any] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Handle gatekeeper validation result and return routing decision.
        
        Args:
            validation: Gatekeeper validation result dict.
            content: Raw LLM response content.
            base_metadata: Current state metadata (to merge into).
            
        Returns:
            State dict if routing decision made, None if retry needed.
        """
        if not validation["valid"]:
            return None
            
        target = validation["target"]
        command = validation["command"]
        
        # Merge metadata
        current_meta = (base_metadata or {}).copy()
        
        # --- TARA ROUTE ---
        if target == "TARA":
            logger.info(f"🛠️ TARA 2.0: Returning TARA route for: {command}")
            return {
                "messages": [HumanMessage(content=command)],
                "next": "tara",
                "user_input": command,
            }
            
        # --- IRIS ROUTE ---
        elif target == "IRIS":
            logger.info(f"👁️ Routing to IRIS: {command}")
            return {
                "messages": [HumanMessage(content=command)],
                "next": "iris",
                "user_input": command,
            }
        
        # --- DOCKER ROUTE (Phase 2) ---
        elif target == "DOCKER":
            logger.info(f"🐳 Routing to Docker Swarm: {command}")
            
            # Parse "skill query"
            parts = command.strip().split(" ", 1)
            skill_name = parts[0]
            query = parts[1] if len(parts) > 1 else ""
            
            current_meta["target_skill"] = skill_name
            current_meta["skill_query"] = query
            
            return {
                # We forward the command as a HumanMessage so the node has context 
                # (though it reads metadata)
                "messages": [HumanMessage(content=command)],
                "next": "docker",
                "user_input": command,
                "metadata": current_meta
            }

        # --- CHAT (Direct Response) ---
        else:
            return {"messages": [AIMessage(content=content)]}
    
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
        from src.core.registry import ServiceRegistry
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
        Synchronous Main Execution Logic.
        
        Args:
            state: LangGraph state dict with 'messages' key.
            
        Returns:
            Updated state dict with routing decision.
        """
        # RIPPLE FIX: _build_context is now async, so we must wrap it for sync execution
        # process() is rarely used in production (mostly tests/fallback), so overhead is acceptable
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're already in a loop, can't use run_until_complete easily without nesting
                # Ideally, use aprocess() instead. But for fallback support:
                current_messages, retry_buffer = loop.run_until_complete(self._build_context(state))
            else:
                current_messages, retry_buffer = asyncio.run(self._build_context(state))
        except RuntimeError:
            # Fallback for complex loop states
            logger.warning("Could not run async memory context in sync process(), proceeding without memory")
            from langchain_core.messages import SystemMessage
            from src.persona.profile import get_system_prompt
            
            current_messages = [SystemMessage(content=get_system_prompt())]
            state_messages = state.get("messages", [])
            current_messages.extend(state_messages)
            retry_buffer = []
        
        # Extract last user query for memory persistence
        user_query = None
        state_messages = state.get("messages", [])
        if isinstance(state_messages, list):
            for msg in reversed(state_messages):
                if isinstance(msg, HumanMessage):
                    user_query = msg.content
                    break

        for attempt in range(config.MAX_RETRIES + 1):
            full_context = current_messages + retry_buffer
            
            # LLM Invoke (SYNC)
            try:
                response = self.llm.invoke(full_context)
                content = response.content
            except Exception as e:
                logger.error(f"LLM Invocation Failed: {e}")
                content = "I'm having trouble connecting to my brain."
            
            # Gatekeeper Check
            validation = self.gatekeeper.validate(content)
            result = self._handle_validation(validation, content, state.get("metadata", {}))
            
            if result is not None:
                # 🧠 MEMORY SAVE: Persist this turn
                self._save_to_memory(user_query, content)
                return result
            
            # Retry with backoff
            logger.warning(f"🔄 Retry {attempt+1}/{config.MAX_RETRIES}: {validation['error']}")
            self._append_retry(retry_buffer, content, validation['error'])
            
            if attempt == config.MAX_RETRIES:
                logger.error(f"❌ Gatekeeper failed after {config.MAX_RETRIES + 1} attempts.")
                return {"messages": [AIMessage(content="ERROR: Unable to process your request. The routing validation failed repeatedly.")]}
            
            # Exponential backoff with jitter
            base_delay = 0.5 * (2 ** attempt)
            jitter = base_delay * 0.25 * (2 * random.random() - 1)
            delay = min(base_delay + jitter, 5.0)
            logger.info(f"💤 Backoff: Sleeping {delay:.2f}s before retry...")
            time.sleep(delay)
                
        return {"messages": [AIMessage(content="I am having trouble processing your request.")]}

    async def aprocess(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Async Main Execution Logic (Native Async).
        
        Args:
            state: LangGraph state dict with 'messages' key.
            
        Returns:
            Updated state dict with routing decision.
        """
        current_messages, retry_buffer = await self._build_context(state)
        
        # Extract last user query for memory persistence
        user_query = None
        state_messages = state.get("messages", [])
        if isinstance(state_messages, list):
            for msg in reversed(state_messages):
                if isinstance(msg, HumanMessage):
                    user_query = msg.content
                    break
        
        # Track last response for memory save on any exit path
        last_content = None

        for attempt in range(config.MAX_RETRIES + 1):
            full_context = current_messages + retry_buffer
            
            # LLM Invoke (ASYNC)
            try:
                response = await self.llm.ainvoke(full_context)
                content = response.content
                last_content = content
            except Exception as e:
                logger.error(f"LLM Async Invocation Failed: {e}")
                content = "I'm having trouble connecting to my brain."
                last_content = content
            
            # Gatekeeper Check (sync - pure CPU logic, negligible blocking)
            validation = self.gatekeeper.validate(content)
            result = self._handle_validation(validation, content, state.get("metadata", {}))
            
            if result is not None:
                # Memory is saved in core/engine.py after graph execution (single source of truth)
                return result
            
            # Retry with backoff
            logger.warning(f"🔄 Retry {attempt+1}/{config.MAX_RETRIES}: {validation['error']}")
            self._append_retry(retry_buffer, content, validation['error'])
            
            if attempt == config.MAX_RETRIES:
                logger.error(f"❌ Gatekeeper failed after {config.MAX_RETRIES + 1} attempts.")
                # Memory is saved in core/engine.py after graph execution
                return {"messages": [AIMessage(content="ERROR: Unable to process your request.")]}
            
            # Async sleep with backoff
            base_delay = 0.5 * (2 ** attempt)
            jitter = base_delay * 0.25 * (2 * random.random() - 1)
            delay = min(base_delay + jitter, 5.0)
            await asyncio.sleep(delay)
        
        # Memory is saved in core/engine.py after graph execution
        return {"messages": [AIMessage(content="I am having trouble processing your request.")]}

