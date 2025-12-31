"""TARA Agent - Technical Agent for Reasoning & Analysis.

The executive operator that handles system, desktop, and web tasks for NIA.
Uses a simple tool-calling agent pattern (no langgraph.prebuilt dependency).
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Import tools
try:
    from .tools import TARA_TOOLS, get_tool_names
    _HAS_TOOLS = True
except ImportError:
    TARA_TOOLS = []
    _HAS_TOOLS = False
    get_tool_names = lambda: []
    logger.warning("TARA tools not available")

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
# TARA System Prompt
# =============================================================================

TARA_SYSTEM_PROMPT = """You are TARA, the Technical Agent for Reasoning & Analysis.
You are the Executive Operator of NIA, handling system-level tasks.

AVAILABLE TOOLS:
{tool_descriptions}

HOW TO USE TOOLS:
When you need to use a tool, respond with EXACTLY this format:
TOOL: <tool_name>
ARGS: <json_arguments>

Example for checking system stats:
TOOL: system_stats
ARGS: {{}}

Example for opening an app:
TOOL: open_app
ARGS: {{"app_name": "brave"}}

Example for web search:
TOOL: web_search
ARGS: {{"query": "current weather in Mumbai"}}

RULES:
1. Act IMMEDIATELY. Do not ask for confirmation.
2. Use the tool format above when you need to perform an action.
3. After a tool executes, provide a brief summary of the result.
4. For browsers, default to 'brave' unless specified.
5. Be concise. One sentence responses when possible.
6. If no tool is needed, just respond directly.
"""


# =============================================================================
# TARA Agent Class
# =============================================================================

class TaraAgent:
    """TARA - Technical Agent for Reasoning & Analysis.
    
    A simple tool-calling agent that handles system, desktop, and web tasks.
    
    Example:
        agent = TaraAgent()
        result = agent.run("Check system health")
        print(result)  # "CPU Load: 12.5% | RAM Usage: 45.2%..."
    """
    
    def __init__(self, temperature: float = 0.3) -> None:
        """Initialize TARA agent.
        
        Args:
            temperature: LLM temperature (lower = more deterministic)
        """
        self.temperature = temperature
        self._llm = None
        self._tools: Dict[str, Any] = {}
        self._initialized = False
        
        self._initialize()
    
    def _initialize(self) -> bool:
        """Initialize the agent."""
        if not _HAS_TOOLS or not TARA_TOOLS:
            logger.warning("No tools available - TARA will have limited functionality")
        
        # Build tool registry
        for tool in TARA_TOOLS:
            self._tools[tool.name] = tool
        
        try:
            # Get LLM from ModelManager
            if _HAS_MODEL_MANAGER:
                manager = ModelManager()
                self._llm = manager.get_smart_model(self.temperature)
            else:
                # Fallback to direct import
                try:
                    from langchain_openai import ChatOpenAI
                    import os
                    api_key = os.environ.get("OPENAI_API_KEY")
                    if api_key:
                        self._llm = ChatOpenAI(
                            model="gpt-4o",
                            temperature=self.temperature,
                            api_key=api_key
                        )
                except ImportError:
                    logger.error("No LLM provider available")
                    return False
            
            if not self._llm:
                logger.error("Failed to initialize LLM")
                return False
            
            self._initialized = True
            logger.info("TARA agent initialized with %d tools", len(self._tools))
            return True
            
        except Exception as exc:
            logger.exception("Failed to initialize TARA: %s", exc)
            return False
    
    def _build_system_prompt(self) -> str:
        """Build system prompt with tool descriptions."""
        tool_descriptions = []
        for name, tool in self._tools.items():
            desc = f"- {name}: {tool.description}"
            tool_descriptions.append(desc)
        
        return TARA_SYSTEM_PROMPT.format(
            tool_descriptions="\n".join(tool_descriptions) if tool_descriptions else "No tools available"
        )
    
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
        """Execute a tool and return the result."""
        if tool_name not in self._tools:
            return f"Error: Unknown tool '{tool_name}'. Available: {list(self._tools.keys())}"
        
        tool = self._tools[tool_name]
        
        try:
            result = tool._run(**args)
            return str(result)
        except Exception as exc:
            return f"Tool error: {exc}"
    
    def run(self, task: str, max_iterations: int = 3) -> str:
        """Execute a task and return the result.
        
        Args:
            task: The task description.
            max_iterations: Maximum number of tool call iterations.
            
        Returns:
            The result of the task execution.
        """
        if not self._initialized or not self._llm:
            return "TARA is not initialized. Check dependencies."
        
        try:
            messages = [
                SystemMessage(content=self._build_system_prompt()),
                HumanMessage(content=task)
            ]
            
            for iteration in range(max_iterations):
                # Get LLM response
                response = self._llm.invoke(messages)
                response_text = response.content if hasattr(response, 'content') else str(response)
                
                # Check for tool call
                tool_call = self._parse_tool_call(response_text)
                
                if tool_call:
                    # Execute the tool
                    tool_result = self._execute_tool(tool_call["tool"], tool_call["args"])
                    
                    # Add to conversation and let LLM summarize
                    messages.append(AIMessage(content=response_text))
                    messages.append(HumanMessage(content=f"Tool result: {tool_result}"))
                    
                    # If this is the last iteration, return the tool result directly
                    if iteration == max_iterations - 1:
                        return tool_result
                else:
                    # No tool call - return the response
                    return response_text
            
            return "Task completed."
            
        except Exception as exc:
            logger.exception("TARA execution error: %s", exc)
            return f"Error executing task: {exc}"
    
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
        else:
            # Get the last human message as the task
            task = ""
            for msg in reversed(messages):
                if hasattr(msg, "type") and msg.type == "human":
                    task = msg.content
                    break
                elif hasattr(msg, "content"):
                    task = msg.content
                    break
            
            if task:
                response = self.run(task)
            else:
                response = "Could not extract task from messages."
        
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
        """List available tools."""
        return get_tool_names()
