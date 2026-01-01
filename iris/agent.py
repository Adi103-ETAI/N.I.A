"""IRIS Agent - Intelligent Recognition & Image System.

Vision specialist agent using NVIDIA Llama 3.2 Vision for screen analysis.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Try to import NVIDIA vision model
try:
    from langchain_nvidia_ai_endpoints import ChatNVIDIA
    _HAS_NVIDIA = True
except ImportError:
    _HAS_NVIDIA = False
    ChatNVIDIA = None  # type: ignore
    logger.warning("langchain-nvidia-ai-endpoints not installed")

# Try to import LangChain messages
try:
    from langchain_core.messages import HumanMessage, AIMessage
    _HAS_LANGCHAIN = True
except ImportError:
    _HAS_LANGCHAIN = False
    HumanMessage = None  # type: ignore
    AIMessage = None  # type: ignore

# Import our screen capture tool
try:
    from .tools import capture_screen_raw
    _HAS_TOOLS = True
except ImportError:
    _HAS_TOOLS = False
    capture_screen_raw = None  # type: ignore


# =============================================================================
# IRIS Vision Agent
# =============================================================================

class IrisAgent:
    """IRIS - Intelligent Recognition & Image System.
    
    Vision specialist that captures the screen and analyzes it using
    NVIDIA's Llama 3.2 Vision model.
    
    Example:
        agent = IrisAgent()
        result = agent.run("What is on my screen?")
    """
    
    # NVIDIA Llama 3.2 Vision model (11B instruct)
    MODEL_NAME = "meta/llama-3.2-11b-vision-instruct"
    
    def __init__(self, temperature: float = 0.1) -> None:
        """Initialize IRIS agent.
        
        Args:
            temperature: LLM temperature (lower = more deterministic).
        """
        self.temperature = temperature
        self._llm = None
        self._initialized = False
        
        self._initialize()
    
    def _initialize(self) -> bool:
        """Initialize the vision LLM."""
        if not _HAS_NVIDIA:
            logger.error("NVIDIA AI endpoints not available")
            return False
        
        if not os.environ.get("NVIDIA_API_KEY"):
            logger.error("NVIDIA_API_KEY not set")
            return False
        
        try:
            self._llm = ChatNVIDIA(
                model=self.MODEL_NAME,
                temperature=self.temperature,
                max_tokens=1024,
            )
            self._initialized = True
            logger.info("IRIS agent initialized with %s", self.MODEL_NAME)
            return True
        except Exception as exc:
            logger.exception("Failed to initialize IRIS: %s", exc)
            return False
    
    def run(self, query: str) -> str:
        """Capture screen and analyze with vision model.
        
        Args:
            query: User's question about what they see.
            
        Returns:
            Description/analysis of the screen content.
        """
        if not self._initialized or not self._llm:
            return "IRIS is not initialized. Check NVIDIA_API_KEY."
        
        if not _HAS_TOOLS:
            return "Screen capture tools not available."
        
        try:
            print("👁️ 📸 IRIS: Capturing visual data...")
            
            # 1. Capture screen as Base64
            b64_image = capture_screen_raw()
            
            # 2. Prepare multimodal message
            # IMPROVED PROMPT: Strict Observation - Discourages guessing
            prompt_text = (
                f"User Query: {query}\n\n"
                "INSTRUCTIONS:\n"
                "1. Identify the MAIN active windows in focus (e.g., Code Editor, Terminal, Browser).\n"
                "2. Read window titles or tabs to identify specific apps (e.g., 'Visual Studio Code', 'Brave', 'Chrome').\n"
                "3. Describe the screen layout (Left vs Right).\n"
                "4. CRITICAL: Do NOT guess about small icons or minimized windows if you cannot clearly read their text.\n"
                "5. Be concise and factual."
            )
            
            message = HumanMessage(content=[
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}}
            ])
            
            # 3. Inference
            print("👁️ 🧠 IRIS: Analyzing image...")
            response = self._llm.invoke([message])
            
            return response.content
            
        except Exception as exc:
            logger.exception("IRIS analysis failed: %s", exc)
            return f"Visual analysis failed: {exc}"
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process state dict for NIA graph integration.
        
        Args:
            state: AgentState dict with messages.
            
        Returns:
            Updated state with IRIS response.
        """
        messages = state.get("messages", [])
        
        # Extract the task from messages
        query = ""
        for msg in reversed(messages):
            if hasattr(msg, "type") and msg.type == "human":
                query = msg.content
                break
            elif hasattr(msg, "content"):
                query = msg.content
                break
        
        if query:
            response = self.run(query)
        else:
            response = "No visual query provided."
        
        # Build response
        if _HAS_LANGCHAIN:
            ai_message = AIMessage(content=response)
        else:
            ai_message = {"role": "assistant", "content": response}
        
        new_messages = list(messages) + [ai_message]
        
        return {
            **state,
            "messages": new_messages,
            "next": "__end__",
        }
    
    def start_sentry(self) -> bool:
        """Start the Sentry background monitoring thread.
        
        Returns:
            True if sentry was started, False if already running or failed.
        """
        # Lazy import to avoid circular dependency
        try:
            from .sentry import SentryThread
        except ImportError:
            print("👁️ Sentry module not available")
            return False
        
        # Check if already running
        if hasattr(self, '_sentry') and self._sentry is not None:
            if self._sentry.is_alive():
                print("👁️ Sentry is already running")
                return False
        
        try:
            self._sentry = SentryThread(self)
            self._sentry.start()
            print("👁️ ✅ Sentry: ENABLED")
            return True
        except Exception as e:
            print(f"❌ Failed to start Sentry: {e}")
            return False
    
    def stop_sentry(self) -> bool:
        """Stop the Sentry background monitoring thread.
        
        Returns:
            True if sentry was stopped, False if not running.
        """
        if not hasattr(self, '_sentry') or self._sentry is None:
            print("⚠️  Sentry is not active")
            return False
        
        try:
            self._sentry.stop()
            self._sentry = None
            print("👁️ ❌ Sentry: DISABLED")
            return True
        except Exception as e:
            print(f"❌ Failed to stop Sentry: {e}")
            return False
    
    @property
    def is_ready(self) -> bool:
        """Check if IRIS is ready."""
        return self._initialized and self._llm is not None


def run_iris_agent(state: dict) -> dict:
    """IRIS LangGraph Node function.
    
    1. Captures the screen.
    2. Sends Image + User Query to NVIDIA.
    3. Returns the observation.
    
    Args:
        state: LangGraph state dict.
        
    Returns:
        Updated state with IRIS response.
    """
    agent = IrisAgent()
    return agent.process(state)
