"""N.I.A. Agent Module - Supervisor and Specialist Agents.

This module implements the agent classes for the NIA supervisor architecture:
- SupervisorAgent: Routes queries and handles general conversation
- Placeholder agents for IRIS (Vision) and TARA (Logic)

The supervisor uses the ModelManager to get the best available model
(NVIDIA NIM, OpenAI, or Ollama) to decide whether to:
1. Handle the query directly (general chat)
2. Route to IRIS for vision/image tasks
3. Route to TARA for logic/reasoning tasks
"""
from __future__ import annotations

import os

from dotenv import load_dotenv

# Import state definitions
from .state import (
    AgentState,
    AGENT_IRIS,
    AGENT_TARA,
    AGENT_END,
)

# Centralized logging and config
from core.logger import setup_logger
from core.config import settings

logger = setup_logger("NIA.AGENT")

# Load environment variables
load_dotenv()

# Try to import LangChain core
try:
    from langchain_core.messages import AIMessage
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    _HAS_LANGCHAIN = True
except ImportError:
    _HAS_LANGCHAIN = False
    AIMessage = dict  # type: ignore
    logger.warning("langchain-core not installed. Install with: pip install langchain-core")

# Try to import ModelManager
try:
    from models.model_manager import ModelManager, get_smart_model
    _HAS_MODEL_MANAGER = True
except ImportError:
    _HAS_MODEL_MANAGER = False
    ModelManager = None  # type: ignore
    get_smart_model = None  # type: ignore
    logger.warning("models.model_manager not available")

# Try to import PersonaProfile for dynamic system prompt
try:
    from persona.profile import get_system_prompt, PersonaProfile
    _HAS_PERSONA = True
except ImportError:
    _HAS_PERSONA = False
    get_system_prompt = None  # type: ignore
    PersonaProfile = None  # type: ignore
    logger.debug("persona.profile not available, using default prompts")


# =============================================================================
# System Prompts (Fallback - used if persona.profile not available)
# =============================================================================

SUPERVISOR_SYSTEM_PROMPT_FALLBACK = """You are NIA (Neural Intelligence Assistant), a unified AI with specialized internal capabilities.

## CRITICAL ROUTING RULES - READ CAREFULLY

You have TWO internal modules. You MUST route tasks to them instead of answering yourself:

### TARA (Technical Agent) - ROUTE:TARA:
TARA handles ALL of these tasks. You MUST route to TARA for:
- System health, CPU, RAM, disk usage, system stats
- Opening or closing applications (browser, notepad, spotify, etc.)
- Media playback: playing songs, videos, YouTube, Spotify, streaming content
- Clipboard operations (copy, paste)
- Web searches, current information, real-time data (weather, prices, news, stocks)
- Math calculations, equations, analysis
- Any technical or factual queries requiring external data
- **MEMORY OPERATIONS: "remember that", "save this", "note that", "don't forget"**

### IRIS (Vision Agent) - ROUTE:IRIS:
IRIS handles:
- Image analysis, photo description
- Visual recognition, OCR
- Anything requiring seeing an image

## ABSOLUTE CONSTRAINTS - YOU MUST FOLLOW THESE

1. DO NOT answer technical questions yourself
2. DO NOT simulate actions (like "Opening Notepad..." without actually routing)
3. DO NOT make up prices, weather, or any real-time data
4. DO NOT pretend to perform system operations
5. **DO NOT say "I will remember that" - you have NO memory. Route to TARA!**
6. ALWAYS route to TARA for: search, open, close, check, system, calculate, weather, price, stats

=== 🧠 ACTIVE LISTENING (CRITICAL) ===
You are a Long-Term Memory System. Your goal is to learn about the user.

RULES FOR PREFERENCE EXTRACTION:
1.  **SCAN FIRST:** Before answering, scan the user's input for any personal facts, preferences, or traits (e.g., "I love simple code", "My name is Raj", "I hate comments").
2.  **ACTION PRIORITY:** If you find a preference, you MUST call `save_user_preference` **IMMEDIATELY**.
    * ❌ DO NOT write the code yet.
    * ❌ DO NOT say "I will save this."
    * ✅ ROUTE:TARA: save_user_preference(key="...", value="...")
3.  **THE LOOP:** Understand that calling a tool does NOT end the conversation.
    * Step 1: You call `save_user_preference` via ROUTE:TARA.
    * Step 2: The system executes it.
    * Step 3: You will be called AGAIN to generate the final answer (the Python code).
    * THEREFORE: Save the data FIRST. The code can wait 1 second.

EXAMPLE:
User: "Write a script, I hate comments."
✅ Correct: ROUTE:TARA: save_user_preference(key="code_style", value="hates comments")
❌ Wrong: Immediately outputting the script without saving the preference

## HOW TO ROUTE (CLEAN HANDOFF)

When routing, send CLEAN COMMANDS. Do NOT add conversational filler.
Format: "ROUTE:TARA: [exact command]"

**CORRECT:**
User: "Remember that I love Python"
→ "ROUTE:TARA: Save preference: user loves Python"

User: "Open brave browser"
→ "ROUTE:TARA: Open brave browser"

User: "Write code in Rust, my favorite"
→ "ROUTE:TARA: Save preference: favorite language is Rust" (then continue with the code)

**WRONG:**
→ "I'll remember that for you!" (NO! Route to TARA!)
→ "Sure, let me go ahead and open that for you..." (Too verbose!)

## GENERAL CONVERSATION

For simple greetings, opinions, or non-technical chat, respond directly:
- "Hello" -> Greet warmly
- "Tell me a joke" -> Tell a joke
- "Who are you?" -> Introduce yourself as NIA

Remember: When in doubt, ROUTE TO TARA. Never guess or hallucinate data."""


# Keywords that MUST trigger TARA routing
TARA_ROUTING_KEYWORDS = [
    # System & Time
    "system", "cpu", "ram", "memory", "disk", "health", "stats", "usage", "performance",
    "time", "date", "day", "clock", "hour", "minute",
    # Apps
    "open", "close", "launch", "start", "run", "application", "app", "browser", "notepad",
    "chrome", "brave", "firefox", "spotify", "discord", "code", "vscode",
    # Media playback
    "play", "watch", "stream", "listen", "youtube", "video", "song", "music", "audio",
    "movie", "podcast", "radio", "pause", "stop", "resume", "skip", "next", "previous",
    # Volume & Audio Control
    "volume", "mute", "unmute", "sound", "louder", "quieter", "speaker",
    # Power & Battery
    "battery", "power", "charge", "lock", "shutdown", "restart", "reboot", "hibernate",
    # Maintenance
    "recycle", "bin", "trash", "empty",
    # Ghost Protocol
    "ghost", "conceal", "hide", "panic",
    # Clipboard
    "copy", "paste", "clipboard",
    # Web/Search/URLs
    "search", "google", "look up", "find", "weather", "price", "cost", "news", "stock",
    "bitcoin", "crypto", "current", "today", "now", "latest",
    "url", "link", "website", "webpage", "http", "www", "go to", "navigate",
    # Math
    "calculate", "solve", "math", "equation", "compute", "analyze", "sum", "multiply",
    # Memory/Preferences (CRITICAL: Forces routing for save operations)
    "remember", "save", "note", "forget", "preference", "prefer", "like", "hate", "love",
    "store", "record", "i am a", "i'm a", "my name", "call me",
]

# Keywords that trigger IRIS routing
IRIS_ROUTING_KEYWORDS = [
    # Image analysis
    "image", "photo", "picture", "visual", "camera",
    # Screen analysis
    "screen", "screenshot", "what do you see", "look at this", "analyze screen",
    "read this error", "what's on my screen", "look at the screen",
    # OCR/Text
    "read text", "ocr", "read this", "what does this say",
    # General vision
    "see", "look", "describe this", "what's in this", "analyze this",
]


IRIS_PLACEHOLDER_RESPONSE = """I analyzed the image, but my vision capabilities are still being enhanced.

Currently, I can help you with:
- 🖼️ General image descriptions
- 👁️ Object identification concepts
- 📝 Understanding what you're looking for

My full visual analysis features are coming soon! For now, could you describe what you're trying to understand about the image? I'll do my best to help."""


TARA_PLACEHOLDER_RESPONSE = """I processed that request, but my advanced reasoning module is still being developed.

I can currently help with:
- 🧮 Basic calculations and math
- 🧩 Logical thinking and problem-solving approaches
- 💻 Code explanations and concepts
- 📊 Structured analysis

My full computational capabilities will be available soon! In the meantime, let me share what I can about your question."""


# =============================================================================
# Supervisor Agent
# =============================================================================

class SupervisorAgent:
    """The main supervisor agent that routes queries and handles general chat.
    
    The supervisor uses the ModelManager to get the best available model
    (NVIDIA NIM, OpenAI, or Ollama) to decide how to handle each user query:
    - Direct response for general conversation
    - Route to IRIS for vision tasks
    - Route to TARA for logic tasks
    
    Example:
        supervisor = SupervisorAgent()
        state = supervisor.process(state)
    """
    
    # Sliding window limit to prevent context overflow
    MAX_HISTORY = 10
    
    def __init__(
        self,
        temperature: float = 0.7,
        model_type: str = "smart",
    ) -> None:
        """Initialize the supervisor agent.
        
        Args:
            temperature: Sampling temperature for responses.
            model_type: Type of model to use ('smart', 'fast').
        """
        self.temperature = temperature
        self.model_type = model_type
        
        self._llm = None
        self._prompt = None
        self._model_manager = None
        self._model_name = "unknown"
        self._system_prompt = None
        
        # Get system prompt from PersonaProfile or use fallback
        if _HAS_PERSONA and get_system_prompt:
            self._system_prompt = get_system_prompt()
            logger.debug("Using PersonaProfile system prompt")
        else:
            self._system_prompt = SUPERVISOR_SYSTEM_PROMPT_FALLBACK
            logger.debug("Using fallback system prompt")
        
        if _HAS_LANGCHAIN:
            self._init_llm()
        else:
            logger.warning("LangChain not available. Supervisor will use fallback mode.")
    
    def _init_llm(self) -> None:
        """Initialize the LLM using ModelManager."""
        try:
            if _HAS_MODEL_MANAGER:
                # Use ModelManager to get the best available model
                self._model_manager = ModelManager()
                
                if self.model_type == "fast":
                    self._llm = self._model_manager.get_fast_model(self.temperature)
                else:
                    self._llm = self._model_manager.get_smart_model(self.temperature)
                
                # Get model name for logging
                self._model_name = getattr(self._llm, "model", "unknown")
            else:
                # Fallback to direct import if ModelManager not available
                logger.warning("ModelManager not available, trying direct import")
                try:
                    from langchain_openai import ChatOpenAI
                    api_key = os.environ.get("OPENAI_API_KEY")
                    if api_key:
                        self._llm = ChatOpenAI(
                            model="gpt-4o",
                            temperature=self.temperature,
                            api_key=api_key,
                        )
                        self._model_name = "gpt-4o"
                except ImportError:
                    logger.warning("No LLM providers available")
            
            # Create prompt template with dynamic system prompt
            self._prompt = ChatPromptTemplate.from_messages([
                ("system", self._system_prompt),
                MessagesPlaceholder(variable_name="messages"),
            ])
            
            if self._llm:
                logger.info("SupervisorAgent initialized with model: %s", self._model_name)
            else:
                logger.warning("SupervisorAgent running in fallback mode (no LLM)")
                
        except Exception as exc:
            logger.exception("Failed to initialize LLM: %s", exc)
            self._llm = None
    
    def process(self, state: AgentState) -> AgentState:
        """Process the current state and decide on action.
        
        Args:
            state: Current agent state with messages.
            
        Returns:
            Updated state with supervisor's decision.
        """
        messages = state.get("messages", [])
        
        if not messages:
            return {
                **state,
                "next": AGENT_END,
                "final_response": "I didn't receive any input. How can I help you?",
            }
        
        # =================================================================
        # ⚡ SHORT-CIRCUIT TRAP: Send raw TOOL: command to bypass TARA's LLM
        # This constructs regex-ready command that TARA executes immediately
        # =================================================================
        user_input_raw = ""
        user_input_lower = ""
        for msg in reversed(messages):
            if _HAS_LANGCHAIN and hasattr(msg, "type") and msg.type == "human":
                user_input_raw = msg.content
                user_input_lower = msg.content.lower()
                break
            elif isinstance(msg, dict) and msg.get("role") == "user":
                user_input_raw = msg.get("content", "")
                user_input_lower = user_input_raw.lower()
                break
        
        # ONLY explicit "Hard" commands trigger Short-Circuit
        # Soft preferences ("I like", "prefer") are handled by LLM Active Listening
        memory_triggers = [
            "remember that", 
            "save this", 
            "note that", 
            "don't forget",
            "set my preference",
            "set my setting",
            "store this",
        ]
        
        # 🛑 LOOP BREAKER: Check if we JUST came back from saving
        last_msg = messages[-1] if messages else None
        just_saved_preference = False
        if _HAS_LANGCHAIN and hasattr(last_msg, "content"):
            msg_content = str(last_msg.content)
            if "Preference saved" in msg_content or "SAVED TO DATABASE" in msg_content:
                just_saved_preference = True
                logger.info("🛑 Loop Breaker: Action already completed. Skipping re-trigger.")
        
        # 🛑 GUARD: Do NOT short-circuit if it's a question (read request)
        read_indicators = ["what", "show", "list", "tell me", "how many", "do you know"]
        is_question = (
            user_input_lower.endswith("?") or
            any(user_input_lower.startswith(ind) for ind in read_indicators) or
            "?" in user_input_lower
        )
        
        if any(trigger in user_input_lower for trigger in memory_triggers) and not is_question and not just_saved_preference:
            import json  # Import locally for bulletproof JSON serialization
            
            logger.info("⚡ Supervisor: Short-circuiting TARA (Clean Input Mode)")
            
            # =============================================================
            # EXTRACT CLEAN INPUT (Strip Memory Context)
            # The engine injects memory with "\n\nUser Input: " delimiter
            # =============================================================
            if "User Input:" in user_input_raw:
                # Split and take the last part (the actual user message)
                clean_text = user_input_raw.split("User Input:")[-1].strip()
                logger.debug("Extracted clean input from augmented prompt")
            else:
                clean_text = user_input_raw.strip()
            
            # Create full payload with tool name AND args
            payload = {
                "tool": "save_user_preference",
                "args": {
                    "key": "user_fact",
                    "value": clean_text  # Use clean text, not raw
                }
            }
            
            # Strict serialization: Compact mode removes all internal spaces
            json_clean = json.dumps(payload, separators=(',', ':'))
            
            # Construct command with tight prefix (no space after colon)
            command_text = f"JSON_CMD:{json_clean}"
            
            # Build as HumanMessage so TARA sees it as a task
            if _HAS_LANGCHAIN:
                from langchain_core.messages import HumanMessage
                command_message = HumanMessage(content=command_text)
            else:
                command_message = {"role": "user", "content": command_text}
            
            return {
                **state,
                "messages": [command_message],
                "next": AGENT_TARA,
                "final_response": None,
                "route_reason": "Memory operation (JSON Protocol)",
            }
        # =================================================================
        # END SHORT-CIRCUIT TRAP
        # =================================================================
        
        # Get supervisor response
        response_text = self._get_response(messages)
        
        # Parse routing decision
        next_agent, final_response, route_reason = self._parse_response(response_text)
        
        # Build AI message
        if _HAS_LANGCHAIN:
            ai_message = AIMessage(content=response_text)
        else:
            ai_message = {"role": "assistant", "content": response_text}
        
        return {
            **state,
            "messages": [ai_message],
            "next": next_agent,
            "final_response": final_response,
            "route_reason": route_reason,
        }
    
    def _prune_messages(self, messages: list) -> list:
        """Apply sliding window to prevent context overflow.
        
        Keeps system message (if present) + last MAX_HISTORY messages.
        Old messages are safely discarded since they're stored in ChromaDB.
        
        Args:
            messages: Full message history.
            
        Returns:
            Pruned message list within context limits.
        """
        if len(messages) <= self.MAX_HISTORY:
            return messages
        
        # Separate system message from conversation
        system_msg = None
        conversation = []
        
        for msg in messages:
            if _HAS_LANGCHAIN and hasattr(msg, "type"):
                if msg.type == "system":
                    system_msg = msg
                else:
                    conversation.append(msg)
            elif isinstance(msg, dict) and msg.get("role") == "system":
                system_msg = msg
            else:
                conversation.append(msg)
        
        # Keep last MAX_HISTORY messages
        recent = conversation[-self.MAX_HISTORY:]
        
        # Rebuild with system message first
        if system_msg:
            pruned = [system_msg] + recent
        else:
            pruned = recent
        
        logger.debug("Pruned messages: %d -> %d", len(messages), len(pruned))
        return pruned
    
    def _get_response(self, messages: list) -> str:
        """Get response from LLM with automatic backup model fallback.
        
        If the primary model (405B) fails, automatically switches to 70B backup.
        """
        # Apply sliding window pruning to prevent context overflow
        pruned_messages = self._prune_messages(messages)
        
        # =====================================================================
        # 👂 ACTIVE LISTENING: Dynamic Input Injection
        # If user input contains preference keywords, inject a "whisper" to force
        # the LLM to save the preference BEFORE answering.
        # =====================================================================
        soft_triggers = ["i love", "i like", "i prefer", "i hate", "my favorite", "i am a", "my name is"]
        user_input_lower = ""
        
        # Extract user input for soft trigger check
        for msg in reversed(pruned_messages):
            if _HAS_LANGCHAIN and hasattr(msg, "type") and msg.type == "human":
                user_input_lower = msg.content.lower()
                break
            elif isinstance(msg, dict) and msg.get("role") == "user":
                user_input_lower = msg.get("content", "").lower()
                break
        
        # =====================================================================
        # 🛑 LOOP BREAKER: Check if we JUST came back from saving
        # Prevents infinite loop when preference keywords remain in history
        # =====================================================================
        just_saved_preference = False
        if pruned_messages:
            last_msg = pruned_messages[-1]
            if _HAS_LANGCHAIN and hasattr(last_msg, "content"):
                msg_content = str(last_msg.content)
                if "Preference saved" in msg_content or "Preference Saved" in msg_content:
                    just_saved_preference = True
                    logger.info("🛑 Active Listening: Preference already saved. Skipping injection.")
        
        # Inject whisper instruction if preference detected AND not just saved
        if any(trigger in user_input_lower for trigger in soft_triggers) and not just_saved_preference:
            logger.info("👂 Active Listening: Detected preference. Injecting JSON instruction.")
            
            if _HAS_LANGCHAIN:
                from langchain_core.messages import HumanMessage
                # Use HumanMessage to maintain chat structure
                # Instruct to use JSON_CMD format which router already handles
                injection = HumanMessage(content="""
[SYSTEM INTERVENTION]
The user just expressed a personal preference. 
You MUST save this immediately using the JSON Short-Circuit Protocol.
OUTPUT EXACTLY THIS FORMAT:
JSON_CMD:{"tool":"save_user_preference","args":{"key":"preference_key","value":"preference_value"}}

Replace "preference_key" with a descriptive key (e.g., "code_style", "language_preference").
Replace "preference_value" with what the user expressed.
Do not answer the user's question yet. Save the preference data FIRST.
""")
                # Append to messages for this inference only
                pruned_messages = list(pruned_messages) + [injection]
        
        response = None
        
        # ATTEMPT 1: Primary Model
        if self._llm and self._prompt:
            try:
                chain = self._prompt | self._llm
                result = chain.invoke({"messages": pruned_messages})
                response = result.content
            except Exception as exc:
                logger.warning("⚠️ Primary Brain Failed: %s. Engaging Backup (70B)...", exc)
                
                # ATTEMPT 2: Backup Model (70B)
                try:
                    backup_llm = self._get_backup_llm()
                    if backup_llm:
                        backup_chain = self._prompt | backup_llm
                        result = backup_chain.invoke({"messages": pruned_messages})
                        response = result.content
                        logger.info("Backup model succeeded")
                    else:
                        response = self._fallback_response(pruned_messages)
                except Exception as backup_exc:
                    logger.exception("Backup model also failed: %s", backup_exc)
                    response = self._fallback_response(pruned_messages)
        else:
            response = self._fallback_response(pruned_messages)
        
        # APPLY THE ROUTING SAFETY NET
        # Extract last user message for keyword checking
        user_input = ""
        for msg in reversed(messages):
            if _HAS_LANGCHAIN and hasattr(msg, "content"):
                if hasattr(msg, "type") and msg.type == "human":
                    user_input = msg.content
                    break
            elif isinstance(msg, dict) and msg.get("role") == "user":
                user_input = msg.get("content", "")
                break
        
        # Apply safety net to force routing if needed
        response = self._enforce_routing(user_input, response)
        
        return response
    
    def _get_backup_llm(self):
        """Get backup LLM (70B) for when primary fails."""
        try:
            # Try NVIDIA 70B first
            from langchain_nvidia_ai_endpoints import ChatNVIDIA
            backup = ChatNVIDIA(
                model=settings.LLM_MODEL,
                temperature=self.temperature,
                max_tokens=1024,
                api_key=settings.NVIDIA_API_KEY.get_secret_value(),
            )
            logger.info(f"Initialized backup LLM: {settings.LLM_MODEL}")
            return backup
        except Exception as e:
            logger.debug("NVIDIA backup failed: %s", e)
        
        # Try OpenAI as second backup
        try:
            from langchain_openai import ChatOpenAI
            if settings.has_openai_key:
                backup = ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=self.temperature,
                    api_key=settings.OPENAI_API_KEY.get_secret_value(),
                )
                logger.info("Initialized backup LLM: gpt-4o-mini")
                return backup
        except Exception as e:
            logger.debug(f"OpenAI backup failed: {e}")
        
        return None
    
    def _enforce_routing(self, user_input: str, llm_response: str) -> str:
        """Safety net: Force routing to TARA for action keywords.
        
        If the user asked for an action but the LLM forgot to route,
        this method appends the routing instruction.
        
        Args:
            user_input: The user's original message.
            llm_response: The LLM's response.
            
        Returns:
            The response, possibly with ROUTE:TARA appended.
        """
        # Strong trigger keywords that demand an action
        strong_triggers = [
            # Media
            "play", "watch", "stream", "listen",
            # Apps
            "open", "launch", "close", "kill", "start",
            # Web/Search
            "search", "weather", "price", "news", "google",
            # System
            "system", "cpu", "ram", "disk", "check",
            # Time
            "time", "date",
            # Clipboard
            "copy", "paste",
        ]
        
        lower_input = user_input.lower()
        
        # Check if user asked for action but LLM didn't route
        if any(trigger in lower_input for trigger in strong_triggers):
            if "ROUTE:TARA" not in llm_response and "ROUTE:IRIS" not in llm_response:
                logger.info("Safety Net: Forcing routing to TARA based on keywords")
                # Append routing instruction with the user's original request
                return f"{llm_response}\nROUTE:TARA: {user_input}"
        
        return llm_response
    
    def _fallback_response(self, messages: list) -> str:
        """Fallback response when LLM is unavailable - uses keyword routing."""
        # Extract user message
        user_input = ""
        for msg in reversed(messages):
            if _HAS_LANGCHAIN and hasattr(msg, "content"):
                if hasattr(msg, "type") and msg.type == "human":
                    user_input = msg.content
                    break
            elif isinstance(msg, dict) and msg.get("role") == "user":
                user_input = msg.get("content", "")
                break
        
        # Keyword-based routing using the defined lists
        lower_input = user_input.lower()
        
        # Check for IRIS keywords first (vision)
        if any(word in lower_input for word in IRIS_ROUTING_KEYWORDS):
            return f"Let me analyze that.\nROUTE:IRIS: {user_input}"
        
        # Check for TARA keywords (technical/search/system)
        if any(word in lower_input for word in TARA_ROUTING_KEYWORDS):
            return f"I'll handle that.\nROUTE:TARA: {user_input}"
        
        # Default: general conversation
        return f"Hello! I'm NIA, your Neural Intelligence Assistant. I received: '{user_input}'. I'm in fallback mode - please ensure your API key is configured."
    
    def _parse_response(self, response: str) -> tuple:
        """Parse the supervisor's response for routing instructions.
        
        Searches for ROUTE:IRIS: or ROUTE:TARA: anywhere in the response,
        not just at the beginning. This allows the LLM to include a brief
        acknowledgment before the routing instruction.
        
        Returns:
            Tuple of (next_agent, final_response, route_reason)
        """
        import re
        
        # Look for ROUTE:IRIS: anywhere in response
        iris_match = re.search(r'ROUTE:IRIS:\s*(.+?)(?:\n|$)', response, re.IGNORECASE | re.DOTALL)
        if iris_match:
            task = iris_match.group(1).strip()
            # Get text before the ROUTE as the user-facing message
            user_msg = response[:iris_match.start()].strip()
            return AGENT_IRIS, user_msg if user_msg else None, f"Vision task: {task}"
        
        # Look for ROUTE:TARA: anywhere in response
        tara_match = re.search(r'ROUTE:TARA:\s*(.+?)(?:\n|$)', response, re.IGNORECASE | re.DOTALL)
        if tara_match:
            task = tara_match.group(1).strip()
            # Get text before the ROUTE as the user-facing message
            user_msg = response[:tara_match.start()].strip()
            return AGENT_TARA, user_msg if user_msg else None, f"Technical task: {task}"
        
        # No routing - direct response from supervisor
        return AGENT_END, response, "Direct response"

# =============================================================================
# Placeholder Agents (TaraAgent only - IrisAgent removed, use iris.agent)
# =============================================================================

class TaraAgent:
    """TARA - Tactical Analysis & Reasoning Agent (Placeholder).
    
    This is a placeholder for when the real tara.agent import fails.
    The real TaraAgent lives in tara/agent.py.
    """
    
    name = AGENT_TARA
    description = "Logic specialist for reasoning and calculations"
    
    def process(self, state: AgentState) -> AgentState:
        """Process logic request (placeholder implementation)."""
        logger.info("TARA placeholder agent invoked")
        
        if _HAS_LANGCHAIN:
            ai_message = AIMessage(content=TARA_PLACEHOLDER_RESPONSE)
        else:
            ai_message = {"role": "assistant", "content": TARA_PLACEHOLDER_RESPONSE}
        
        return {
            **state,
            "messages": [ai_message],
            "next": AGENT_END,
            "final_response": TARA_PLACEHOLDER_RESPONSE,
        }


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "SupervisorAgent",
    "TaraAgent",
    "SUPERVISOR_SYSTEM_PROMPT_FALLBACK",
]

