"""TARA Agent - Technical Agent for Reasoning & Analysis.

The executive operator that handles system, desktop, and web tasks for NIA.
Uses dynamic tool discovery via ToolRegistry.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

# Centralized logging and config
from core.logger import setup_logger
from core.config import settings

logger = setup_logger("TARA")

# Import ToolRegistry for dynamic tool discovery
try:
    from .registry import ToolRegistry
    _HAS_REGISTRY = True
except ImportError:
    _HAS_REGISTRY = False
    ToolRegistry = None  # type: ignore
    logger.warning("ToolRegistry not available")

# Import ModelManager
try:
    from models.model_manager import ModelManager
    _HAS_MODEL_MANAGER = True
except ImportError:
    _HAS_MODEL_MANAGER = False
    ModelManager = None  # type: ignore
    logger.warning("ModelManager not available")

# Import LangChain messages
try:
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    _HAS_LANGCHAIN = True
except ImportError:
    _HAS_LANGCHAIN = False


# =============================================================================
# TARA System Prompt Template
# =============================================================================

TARA_SYSTEM_PROMPT = """You are TARA (Tool-Augmented Robotic Assistant).

=== PROTOCOL (READ THIS FIRST) ===
1. You are NOT a conversationalist. You are a DOER.
2. If the user's request matches a tool, you MUST execute that tool immediately.
3. Do NOT say "I will do that" or "Understood" or "I have noted that". JUST RUN THE TOOL.
4. Do NOT make up results. Do NOT pretend to execute. Actually call the tool.
5. If the user asks to 'remember', 'save', 'note', or 'set' something → MUST use 'save_user_preference'.
6. If no tool matches the request, respond with exactly: NO_TOOL_MATCH

=== AVAILABLE TOOLS ===
{tool_descriptions}

=== TOOL CALL FORMAT ===
When executing a tool, use EXACTLY this format:
TOOL: tool_name
ARGS: {{"param1": "value1", "param2": "value2"}}

=== MEMORY COMMANDS (MANDATORY TOOL EXECUTION) ===
These user phrases REQUIRE the save_user_preference tool:
- "remember that I...", "save this...", "note that...", "don't forget..."
- "I prefer...", "I like...", "I hate...", "I love...", "I am a..."
→ You MUST call: TOOL: save_user_preference ARGS: {{"key": "...", "value": "..."}}

=== BANNED RESPONSES ===
NEVER say these phrases without executing a tool:
- "I will remember that" ❌
- "Noted" ❌
- "I have saved that" ❌
- "Understood, I'll keep that in mind" ❌
→ These are LIES. You have NO memory. CALL THE TOOL.
"""


# =============================================================================
# TARA Agent Class
# =============================================================================

class TaraAgent:
    """TARA - Technical Agent for Reasoning & Analysis.
    
    A dynamic tool-calling agent that auto-discovers tools from tara/units/.
    
    Example:
        agent = TaraAgent()
        result = agent.run("Check system health")
        print(result)  # "CPU Load: 12.5% | RAM Usage: 45.2%..."
    """
    
    def __init__(self, temperature: float = 0.3, memory: Any = None) -> None:
        """Initialize TARA agent.
        
        Args:
            temperature: LLM temperature (lower = more deterministic)
            memory: MemoryManager instance for skill tracking (optional)
        """
        self.temperature = temperature
        self.memory = memory  # For skill mastery tracking
        self._llm = None
        self._initialized = False
        
        # Initialize dynamic tool registry
        self.registry: Optional[ToolRegistry] = None
        if _HAS_REGISTRY:
            self.registry = ToolRegistry()
            discovered = self.registry.discover_tools()
            logger.info("🛠️ TARA discovered %d tools from units/", discovered)
        
        self._initialize()
    
    def _initialize(self) -> bool:
        """Initialize the agent."""
        if not self.registry or len(self.registry) == 0:
            logger.warning("No tools discovered - TARA will have limited functionality")
        
        try:
            # Get LLM from ModelManager
            if _HAS_MODEL_MANAGER:
                manager = ModelManager()
                self._llm = manager.get_smart_model(self.temperature)
            else:
                # Fallback to direct import
                try:
                    from langchain_openai import ChatOpenAI
                    if settings.has_openai_key:
                        self._llm = ChatOpenAI(
                            model="gpt-4o",
                            temperature=self.temperature,
                            api_key=settings.OPENAI_API_KEY.get_secret_value()
                        )
                except ImportError:
                    logger.error("No LLM provider available")
                    return False
            
            if not self._llm:
                logger.error("Failed to initialize LLM")
                return False
            
            self._initialized = True
            tool_count = len(self.registry) if self.registry else 0
            logger.info("🛠️ TARA agent initialized with %d tools", tool_count)
            return True
            
        except Exception as exc:
            logger.exception("Failed to initialize TARA: %s", exc)
            return False
    
    def _build_system_prompt(self) -> str:
        """Build system prompt from registry's tool descriptions."""
        if self.registry:
            tool_descriptions = self.registry.build_system_prompt()
        else:
            tool_descriptions = "No tools available."
        
        return TARA_SYSTEM_PROMPT.format(tool_descriptions=tool_descriptions)
    
    def _parse_tool_call(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse tool call from LLM response.
        
        Returns:
            Dict with 'tool' and 'args' keys, or None if no tool call.
        """
        # Look for TOOL: and ARGS: pattern
        tool_match = re.search(r'TOOL:\s*(\w+)', response, re.IGNORECASE)
        if not tool_match:
            return None
        
        tool_name = tool_match.group(1).lower()
        
        # Look for ARGS
        args_match = re.search(r'ARGS:\s*(\{.*?\})', response, re.IGNORECASE | re.DOTALL)
        if args_match:
            try:
                args = json.loads(args_match.group(1))
            except json.JSONDecodeError:
                args = {}
        else:
            args = {}
        
        return {"tool": tool_name, "args": args}
    
    def _execute_tool(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Execute a tool dynamically via registry.
        
        Args:
            tool_name: Name of the tool to execute.
            args: Arguments to pass to the tool.
            
        Returns:
            String result of tool execution.
        """
        if not self.registry:
            return "Error: Tool registry not available."
        
        # Get tool from registry
        tool = self.registry.get_tool(tool_name)
        
        if tool is None:
            available = self.registry.list_tools()
            return f"Error: Tool '{tool_name}' not found. Available: {available}"
        
        # Execute with error handling
        try:
            result = tool.func(**args)
            result_str = str(result)
            
            # Record successful execution for skill mastery tracking
            if self.memory and not result_str.startswith("Error:") and not result_str.startswith("Tool error"):
                self.memory.record_skill_usage(tool_name)
            
            return result_str
        except TypeError as e:
            # Argument mismatch
            return f"Error: Invalid arguments for '{tool_name}': {e}"
        except Exception as e:
            # Runtime error in tool
            return f"Tool error ({tool_name}): {e}"
    
    def run(self, task: str, max_iterations: int = 3) -> str:
        """Execute a task with automatic backup model fallback.
        
        Args:
            task: The task description.
            max_iterations: Maximum number of tool call iterations.
            
        Returns:
            The result of the task execution.
        """
        if not self._initialized or not self._llm:
            return "TARA is not initialized. Check dependencies."
        
        # Track execution chain for procedural memory (skill chains)
        execution_chain = []
        
        try:
            messages = [
                SystemMessage(content=self._build_system_prompt()),
                HumanMessage(content=task)
            ]
            
            # Track which LLM to use (can switch to backup)
            current_llm = self._llm
            
            for iteration in range(max_iterations):
                # Get LLM response with fallback
                try:
                    response = current_llm.invoke(messages)
                    response_text = response.content if hasattr(response, 'content') else str(response)
                except Exception as llm_error:
                    # Primary model failed - try backup
                    logger.warning("TARA primary model failed: %s", llm_error)
                    print(f"⚠️  TARA Primary Brain Failed. Engaging Backup (70B)...")
                    
                    backup_llm = self._get_backup_llm()
                    if backup_llm:
                        try:
                            current_llm = backup_llm  # Switch for remaining iterations
                            response = backup_llm.invoke(messages)
                            response_text = response.content if hasattr(response, 'content') else str(response)
                            logger.info("TARA backup model succeeded")
                        except Exception as backup_error:
                            logger.exception("TARA backup also failed: %s", backup_error)
                            return f"Error: TARA models unavailable. {backup_error}"
                    else:
                        return f"Error: No backup LLM available. {llm_error}"
                
                # Check for tool call
                tool_call = self._parse_tool_call(response_text)
                
                if tool_call:
                    # Execute the tool
                    tool_result = self._execute_tool(tool_call["tool"], tool_call["args"])
                    
                    # Track successful tool execution for skill chain
                    if not tool_result.startswith("Error:") and not tool_result.startswith("Tool error"):
                        execution_chain.append(tool_call["tool"])
                    
                    # Add to conversation and let LLM summarize
                    messages.append(AIMessage(content=response_text))
                    messages.append(HumanMessage(content=f"Tool result: {tool_result}"))
                    
                    # If this is the last iteration, return the tool result directly
                    if iteration == max_iterations - 1:
                        break  # Exit loop to save skill chain
                else:
                    # No tool call - return the response
                    # Save skill chain before returning
                    if self.memory and execution_chain:
                        logger.info("🧠 Learning Skill Chain: %s", execution_chain)
                        self.memory.add_skill_path(goal=task[:100], steps=execution_chain)
                    return response_text
            
            # Save skill chain for multi-tool tasks
            if self.memory and execution_chain:
                logger.info("🧠 Learning Skill Chain: %s", execution_chain)
                self.memory.add_skill_path(goal=task[:100], steps=execution_chain)
            
            return tool_result if 'tool_result' in dir() else "Task completed."
            
        except Exception as exc:
            logger.exception("TARA execution error: %s", exc)
            return f"Error executing task: {exc}"
    
    def _get_backup_llm(self):
        """Get backup LLM (70B) for when primary fails."""
        try:
            from langchain_nvidia_ai_endpoints import ChatNVIDIA
            backup = ChatNVIDIA(
                model=settings.LLM_MODEL,
                temperature=self.temperature,
                max_tokens=1024,
                api_key=settings.NVIDIA_API_KEY.get_secret_value(),
            )
            logger.info(f"TARA initialized backup LLM: {settings.LLM_MODEL}")
            return backup
        except Exception as e:
            logger.debug(f"NVIDIA backup failed: {e}")
        
        # Try OpenAI as second backup
        try:
            from langchain_openai import ChatOpenAI
            if settings.has_openai_key:
                backup = ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=self.temperature,
                    api_key=settings.OPENAI_API_KEY.get_secret_value(),
                )
                logger.info("TARA initialized backup LLM: gpt-4o-mini")
                return backup
        except Exception as e:
            logger.debug(f"OpenAI backup failed: {e}")
        
        return None
    
    async def arun(self, task: str) -> str:
        """Async version of run."""
        return self.run(task)
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process a state dict (for NIA graph integration).
        
        Args:
            state: AgentState dict with messages and routing info.
            
        Returns:
            Updated state with response.
        """
        # Get the task from the state
        messages = state.get("messages", [])
        if not messages:
            response = "No task provided to TARA."
            return {
                **state,
                "messages": [AIMessage(content=response)],
                "next": "__end__",
            }
        
        # Get the last message as the task
        task = ""
        for msg in reversed(messages):
            if hasattr(msg, "type") and msg.type == "human":
                task = msg.content
                break
            elif hasattr(msg, "content"):
                task = msg.content
                break
        
        if not task:
            response = "Could not extract task from messages."
            return {
                **state,
                "messages": [AIMessage(content=response)],
                "next": "__end__",
            }
        
        # =============================================================
        # ⚡ JSON PROTOCOL RECEIVER: Bulletproof command execution
        # =============================================================
        if task.strip().startswith("JSON_CMD:"):
            import json
            logger.info("⚡ TARA: Received JSON Command - Bypassing LLM")
            
            try:
                # Extract JSON string after prefix
                json_str = task.replace("JSON_CMD:", "", 1).strip()
                
                # Parse the JSON payload
                data = json.loads(json_str)
                tool_name = data["tool"]
                tool_args = data["args"]
                
                logger.info("⚡ TARA: Executing %s with args: %s", tool_name, list(tool_args.keys()))
                
                # Execute immediately without LLM
                result = self._execute_tool(tool_name, tool_args)
                
                # Return to supervisor to continue with original user request
                return {
                    **state,
                    "messages": list(messages) + [AIMessage(content=f"✅ SYSTEM: Preference saved ({result}). Now answer the user's original request.")],
                    "next": "supervisor",  # LOOP BACK to fulfill main task
                }
                
            except json.JSONDecodeError as e:
                logger.error("⚡ TARA: JSON Parse Error: %s", e)
                return {
                    **state,
                    "messages": list(messages) + [AIMessage(content=f"❌ JSON Protocol Error: {e}")],
                    "next": "supervisor",  # Let supervisor handle the error
                }
            except KeyError as e:
                logger.error("⚡ TARA: Missing key in JSON: %s", e)
                return {
                    **state,
                    "messages": list(messages) + [AIMessage(content=f"❌ Missing field in command: {e}")],
                    "next": "supervisor",  # Let supervisor handle the error
                }
            except Exception as e:
                logger.error("⚡ TARA: Execution Error: %s", e)
                return {
                    **state,
                    "messages": list(messages) + [AIMessage(content=f"❌ Tool Error: {e}")],
                    "next": "supervisor",  # Let supervisor handle the error
                }
        # =============================================================
        # END JSON PROTOCOL RECEIVER
        # =============================================================
        
        # Standard execution path (invoke LLM)
        response = self.run(task)
        
        # Update state
        new_messages = list(messages) + [AIMessage(content=response)]
        
        return {
            **state,
            "messages": new_messages,
            "next": "__end__",
        }
    
    @property
    def is_ready(self) -> bool:
        """Check if TARA is ready to accept tasks."""
        return self._initialized and self._llm is not None
    
    def list_tools(self) -> list:
        """List available tools from registry."""
        if self.registry:
            return self.registry.list_tools()
        return []
