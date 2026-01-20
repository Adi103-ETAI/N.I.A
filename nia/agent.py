# ----------------------------------------------------------------------
# FILE: nia/agent.py
# STATUS: SYSTEM HUB - Core Supervisor Implementation
# ----------------------------------------------------------------------
from __future__ import annotations

import time
import random
from typing import TYPE_CHECKING, Any, Dict, List, Protocol, runtime_checkable

from core.logger import setup_logger

# --- CRITICAL IMPORTS ---
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from nia.gatekeeper import RoutingGatekeeper
# ADAPTED: import settings as config to match user variable name but use correct source
from core.config import settings as config

# =============================================================================
# TYPE_CHECKING Block (IDE-only, no runtime cost)
# =============================================================================

if TYPE_CHECKING:
    from iris.agent import IrisAgent
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
    
    Central coordinator that receives user input, classifies intent, and routes
    to the appropriate specialist agent. Uses LLM-based intent classification
    with validation gating to ensure clean routing decisions.
    
    Routing Targets:
        - **TARA**: Desktop automation, browser control, file operations
        - **IRIS**: Vision tasks (screen analysis, webcam capture)
        - **CHAT**: General conversation, information queries
    
    Design Pattern:
        - **Protocol-based DI**: Agents injected via AgentProtocol interface
        - **Gated Routing**: RoutingGatekeeper validates LLM decisions
        - **Exponential Backoff**: Retry logic with jitter for resilience
    
    Attributes:
        tara_agent: Optional TaraAgent (TARA 2.0 uses graph node instead).
        iris_agent: IrisAgent instance for vision tasks.
        gatekeeper: RoutingGatekeeper for LLM response validation.
        llm: ChatNVIDIA LLM instance for intent classification.
    """
    
    
    def __init__(
        self,
        tara_agent: AgentProtocol | None = None,  # TARA 2.0: Now optional
        iris_agent: AgentProtocol | None = None,  # Vision agent
        model_type: str = "smart",
        temperature: float = 0.7,
    ) -> None:
        """Initialize the SupervisorAgent with typed dependencies.
        
        Args:
            tara_agent: Optional TaraAgent. TARA 2.0 uses call_tara_2 node.
                       Must implement AgentProtocol if provided.
            iris_agent: IrisAgent instance for vision tasks.
                       Must implement AgentProtocol if provided.
            model_type: LLM model type ('smart' or 'fast').
            temperature: LLM temperature setting (0.0-2.0).
        """
        self.tara_agent: AgentProtocol | None = tara_agent
        self.iris_agent: AgentProtocol | None = iris_agent
        self.gatekeeper: RoutingGatekeeper = RoutingGatekeeper()
        
        
        self._verify_wiring()
        
        # Initialize NVIDIA NIM LLM
        try:
            if model_type == "smart":
                llm_model = config.LLM_MODEL_SMART
            else:
                llm_model = config.LLM_MODEL_FAST
            
            # Use NVIDIA API credentials
            api_key_val = config.NVIDIA_API_KEY.get_secret_value() if config.NVIDIA_API_KEY else None
            base_url_val = config.NVIDIA_BASE_URL
            
            # Fail explicitly if no API key
            if not api_key_val or not api_key_val.startswith("nvapi-"):
                raise ValueError("NVIDIA_API_KEY is not configured or invalid (must start with 'nvapi-')")
            
            self.llm = ChatNVIDIA(
                model=llm_model,
                api_key=api_key_val,
                base_url=base_url_val,
                temperature=temperature,
                max_tokens=2048,
            )
            logger.info(f"🧠 SupervisorAgent LLM initialized: {llm_model} (NVIDIA NIM)")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise RuntimeError(f"SupervisorAgent cannot start without LLM: {e}") from e
        
        # System Prompt
        try:
            with open("nia/config/supervisor_prompt.txt", "r", encoding="utf-8") as f:
                prompt_text = f.read()
        except FileNotFoundError:
            prompt_text = "You are NIA. Route commands to TARA or IRIS."
            
        self.system_prompt = prompt_text + "\n\n### CRITICAL: ROUTE SILENTLY. Example: 'ROUTE:TARA: kill notepad'"
    
    def _verify_wiring(self) -> None:
        """Verify that critical dependencies are properly wired.
        
        Note: With TARA 2.0, tara_agent is optional (handled by graph node).
        """
        if self.tara_agent is None:
            logger.info("ℹ️ TARA 2.0 mode: Tools handled by call_tara_2 graph node")
        else:
            logger.info("ℹ️ Legacy TARA mode: Using TaraAgent instance")
        
        if self.iris_agent is None:
            # IRIS is optional - warn but don't fail
            logger.warning("⚠️ SupervisorAgent has no IRIS agent. Vision routing disabled.")
    
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

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main Execution Logic.
        """
        # 1. Build Context (Stateless)
        current_messages: List[BaseMessage] = [SystemMessage(content=self.system_prompt)]
        
        if "messages" in state and isinstance(state["messages"], list):
            current_messages.extend(state["messages"])
            
        # 2. Gatekeeper Loop
        retry_buffer: List[BaseMessage] = []

        for attempt in range(config.MAX_RETRIES + 1):
            full_context = current_messages + retry_buffer
            
            # Brain Think
            try:
                response = self.llm.invoke(full_context)
                content = response.content
            except Exception as e:
                logger.error(f"LLM Invocation Failed: {e}")
                content = "I'm having trouble connecting to my brain."
            
            # Gatekeeper Check
            validation = self.gatekeeper.validate(content)
            
            if validation["valid"]:
                target = validation["target"]
                command = validation["command"]
                
                # --- TARA ROUTE ---
                if target == "TARA":
                    # Legacy tara_agent path has been removed
                    logger.info(f"🛠️ TARA 2.0: Returning TARA route for: {command}")
                    return {
                        "messages": [HumanMessage(content=command)],
                        "next": "tara",
                        "user_input": command,
                    }
                    
                # --- IRIS ROUTE ---
                elif target == "IRIS":
                    logger.info(f"👁️ Routing to IRIS: {command}")
                    try:
                        if hasattr(self.iris_agent, 'run'):
                            tool_result = self.iris_agent.run(command)
                        else:
                            tool_result = self.iris_agent.process({"messages": [HumanMessage(content=command)]})
                            if isinstance(tool_result, dict):
                                msgs = tool_result.get("messages", [])
                                tool_result = msgs[-1].content if msgs else str(tool_result)
                    except Exception as e:
                        tool_result = f"Error executing IRIS command: {e}"
                    
                    if tool_result is None: tool_result = "✅ Visual check completed."
                    
                    # Contract: Return Dictionary
                    return {"messages": [AIMessage(content=str(tool_result))]}
                
                # --- CHAT ---
                else:
                    return {"messages": [AIMessage(content=content)]}
            
            else:
                # --- RETRY WITH BACKOFF ---
                logger.warning(f"🔄 Retry {attempt+1}/{config.MAX_RETRIES}: {validation['error']}")
                retry_buffer.append(AIMessage(content=content))
                retry_buffer.append(HumanMessage(content=f"SYSTEM ERROR: {validation['error']}"))
                
                if attempt == config.MAX_RETRIES:
                    logger.error(f"❌ Gatekeeper failed after {config.MAX_RETRIES + 1} attempts. Last error: {validation['error']}")
                    return {"messages": [AIMessage(content="ERROR: Unable to process your request. The routing validation failed repeatedly.")]}
                
                # Exponential backoff: 0.5s, 1s, 2s... with ±25% jitter
                base_delay = 0.5 * (2 ** attempt)
                jitter = base_delay * 0.25 * (2 * random.random() - 1)  # ±25%
                delay = min(base_delay + jitter, 5.0)  # Cap at 5 seconds
                logger.info(f"💤 Backoff: Sleeping {delay:.2f}s before retry...")
                time.sleep(delay)
                
        # 3. Fallback (should not reach here due to above exit, but kept for safety)
        return {"messages": [AIMessage(content="I am having trouble processing your request.")]}
