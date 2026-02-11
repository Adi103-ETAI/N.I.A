"""IRIS Agent - Intelligent Recognition & Image System.

v2.5.2 "Velocity" - Vision Specialist Agent

Uses NVIDIA Llama 3.2 Vision (or active provider's vision model) for image 
analysis. Autonomously captures screen or webcam based on user intent.

Dynamic Provider Access:
    The vision LLM is fetched via `@property` on each access, NOT stored at init.
    This enables hot-swap provider switching via ModelManager.set_active_provider()
    without restarting. All calls are wrapped with SafeLLM circuit breaker.

Data Flow:
    Supervisor -> ROUTE:IRIS -> IrisAgent.process()
                                    |
                                    v
                             Screen/Webcam Capture
                                    |
                                    v
                             SafeLLM -> Vision LLM -> Analysis Result

Usage:
    from src.agents.iris.agent import IrisAgent
    
    agent = IrisAgent()
    
    # Direct string input
    result = agent.process("What's on my screen?")
    
    # LangGraph state dict input
    result = agent.process({"messages": [HumanMessage(content="Look at screen")]})

Version: 2.5.2
"""
from __future__ import annotations

import base64
import os
from typing import Any, Dict, Optional, Union

from src.core.logger import setup_logger
from src.core.config import settings

logger = setup_logger("IRIS")

# v2.5.2: Import vision model from ModelManager (enables dynamic provider switching)
from src.models.manager import get_vision_model

# Import LangChain messages
try:
    from langchain_core.messages import HumanMessage, AIMessage
    _HAS_LANGCHAIN = True
except ImportError:
    _HAS_LANGCHAIN = False
    HumanMessage = None  # type: ignore
    AIMessage = None  # type: ignore

# Import capture tools
try:
    from src.agents.iris.capture import capture_screen, capture_webcam
    _HAS_TOOLS = True
except ImportError:
    _HAS_TOOLS = False
    capture_screen = None  # type: ignore
    capture_webcam = None  # type: ignore


# =============================================================================
# Config Loader (Dynamic from External Files)
# =============================================================================

import json
from pathlib import Path

def _load_iris_config() -> dict:
    """Load IRIS configuration from centralized ROOT/config/iris/.
    
    Returns:
        Dictionary with intent keywords and vision prompt.
    """
    # Centralized config path: iris -> ROOT (1 level up via .parents[1])
    config_dir = Path(__file__).resolve().parents[3] / "config" / "iris"
    config = {}
    
    vision_config_path = config_dir / "triggers.json"
    try:
        with open(vision_config_path, "r", encoding="utf-8") as f:
            vision_cfg = json.load(f)
            triggers = vision_cfg.get("triggers", {})
            config["screen_keywords"] = triggers.get("screen", [])
            config["webcam_keywords"] = triggers.get("camera", [])
            logger.debug("Loaded keywords from centralized triggers.json")
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Failed to load triggers.json: {e}. Using defaults.")
        config["screen_keywords"] = ["screen", "window", "monitor", "display"]
        config["webcam_keywords"] = ["camera", "webcam", "photo", "picture"]
    
    # Load vision prompt template from centralized config
    prompt_path = config_dir / "prompt.txt"
    if prompt_path.exists():
        with open(prompt_path, "r", encoding="utf-8") as f:
            config["vision_prompt"] = f.read()
    else:
        config["vision_prompt"] = "User Query: {query}\n\nDescribe what you see."
    
    return config


# Load at module level (cached)
_IRIS_CONFIG = _load_iris_config()
SCREEN_KEYWORDS = _IRIS_CONFIG["screen_keywords"]
WEBCAM_KEYWORDS = _IRIS_CONFIG["webcam_keywords"]
VISION_PROMPT_TEMPLATE = _IRIS_CONFIG["vision_prompt"]


# =============================================================================
# IRIS Vision Agent
# =============================================================================

class IrisAgent:
    """IRIS - Intelligent Recognition & Image System.
    
    Vision specialist that:
    1. Detects user intent (screen vs webcam)
    2. Captures image automatically or uses provided path
    3. Analyzes with NVIDIA Llama 3.2 Vision
    
    Handles both string input and LangGraph state dict input.
    """
    
    # Vision model from settings
    MODEL_NAME = settings.LLM_MODEL_VISION
    
    def __init__(self, temperature: float = 0.1) -> None:
        """Initialize IRIS agent.
        
        Args:
            temperature: LLM temperature (lower = more deterministic).
        """
        self.model = settings.LLM_MODEL_VISION
        self.temperature = temperature
        self._llm = None
        self._initialized = False
        self._sentry = None
        
        self._initialize()
    
    def _initialize(self) -> bool:
        """Verify LLM access at startup (fail-fast check).
        
        v2.5.2: LLM is now fetched dynamically via the llm property.
        This just verifies we can access the ModelManager at startup.
        """
        # Ensure env is loaded
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass

        # === ROOT FIX: Explicitly check for API Key ===
        if not os.getenv("NVIDIA_API_KEY"):
            logger.warning("❌ IRIS Setup Error: NVIDIA_API_KEY missing in .env")
            self._initialized = False
            return False

        try:
            _ = self.llm  # Access property to verify connectivity
            self._initialized = True
            logger.info("IRIS agent ready (dynamic LLM via ModelManager)")
            return True
        except Exception as exc:
            logger.exception("Failed to initialize IRIS: %s", exc)
            self._initialized = False
            return False
    
    @property
    def llm(self):
        """Get vision LLM dynamically from ModelManager.
        
        v2.5.2: Fetched on each access to support hot-swap provider switching.
        When ModelManager.set_active_provider() is called, subsequent accesses
        will automatically use the new provider's vision model.
        """
        return get_vision_model(temperature=self.temperature)

    
    # =========================================================================
    # Input Extraction
    # =========================================================================
    
    def _extract_user_input(self, input_data: Union[str, Dict[str, Any]]) -> str:
        """Extract user text from string or LangGraph state dict.
        
        Args:
            input_data: Either a string or a LangGraph state dict.
            
        Returns:
            The user's text input as a string.
        """
        # If it's already a string, return directly
        if isinstance(input_data, str):
            return input_data
        
        # If it's a dict (LangGraph state), extract from messages
        if isinstance(input_data, dict):
            messages = input_data.get("messages", [])
            if messages:
                # Get the last message
                last_msg = messages[-1]
                # Extract content
                if hasattr(last_msg, "content"):
                    return last_msg.content
                elif isinstance(last_msg, dict):
                    return last_msg.get("content", "")
        
        # Fallback: convert to string
        return str(input_data)
    
    # =========================================================================
    # Intent Detection
    # =========================================================================
    
    def _detect_intent(self, text: str) -> Optional[callable]:
        """Detect capture intent from user text.
        
        Args:
            text: User's query text.
            
        Returns:
            Capture function (capture_screen or capture_webcam), or None.
        """
        if not _HAS_TOOLS:
            return None
        
        text_lower = text.lower()
        
        # Check for webcam keywords first (more specific)
        for keyword in WEBCAM_KEYWORDS:
            if keyword in text_lower:
                return capture_webcam
        
        # Check for screen keywords
        for keyword in SCREEN_KEYWORDS:
            if keyword in text_lower:
                return capture_screen
        
        return None
    
    # =========================================================================
    # Image Encoding
    # =========================================================================
    
    def _encode_image(self, image_path: str) -> Optional[str]:
        """Encode image file to base64."""
        try:
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception as exc:
            logger.error("Failed to encode image: %s", exc)
            return None
    
    def _get_mime_type(self, image_path: str) -> str:
        """Get MIME type from file extension."""
        ext = os.path.splitext(image_path)[1].lower()
        return {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }.get(ext, "image/png")
    
    # =========================================================================
    # Main Processing
    # =========================================================================
    
    def process(
        self,
        input_data: Union[str, Dict[str, Any]],
        image_path: str = None
    ) -> Union[str, Dict[str, Any]]:
        """Process vision query with auto-capture or provided image.
        
        Handles both string input and LangGraph state dict input.
        
        Args:
            input_data: Either a string query or LangGraph state dict.
            image_path: Optional path to existing image file.
            
        Returns:
            String response (if input was string) or updated state dict.
        """
        # Track if input was a dict (for return format)
        is_langgraph = isinstance(input_data, dict)
        
        # Step 1: Extract user text
        user_input = self._extract_user_input(input_data)
        
        if not self._initialized:
            # === ROOT FIX: Specific Error Message ===
            if not os.getenv("NVIDIA_API_KEY"):
                response = "❌ IRIS Setup Error: NVIDIA_API_KEY not found in environment. Please check your .env file."
            else:
                response = "❌ IRIS is not initialized. Check logs for ModelManager errors."
            return self._format_response(response, input_data, is_langgraph)
        
        path = image_path
        
        # Step 2: Auto-Capture if no image provided
        if path is None:
            intent_func = self._detect_intent(user_input)
            
            if intent_func is not None:
                logger.debug("👁️ 📸 IRIS: Capturing visual data...")
                result = intent_func()
                
                # Check if tool returned an error
                if result.startswith("Error"):
                    response = f"❌ {result}"
                    return self._format_response(response, input_data, is_langgraph)
                
                path = result
        
        # Step 3: Validate - must have an image
        if path is None:
            response = (
                "👁️ I need an image to answer that. "
                "Tell me to 'look at the screen' or 'take a photo'."
            )
            return self._format_response(response, input_data, is_langgraph)
        
        # Verify file exists
        if not os.path.exists(path):
            response = f"❌ Image file not found: {path}"
            return self._format_response(response, input_data, is_langgraph)
        
        # Step 4: Encode image and run inference
        try:
            logger.debug("👁️ 🤔 IRIS: Analyzing image...")
            
            b64_image = self._encode_image(path)
            if not b64_image:
                response = "❌ Failed to encode image."
                return self._format_response(response, input_data, is_langgraph)
            
            mime_type = self._get_mime_type(path)
            
            # Build prompt from external template
            prompt_text = VISION_PROMPT_TEMPLATE.format(query=user_input)
            
            # Create multimodal message
            message = HumanMessage(content=[
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64_image}"}}
            ])
            
            # Invoke vision model (dynamic LLM access)
            response_obj = self.llm.invoke([message])
            response = response_obj.content
            
        except Exception as exc:
            logger.exception("IRIS analysis failed: %s", exc)
            response = f"❌ Visual analysis failed: {exc}"
        
        return self._format_response(response, input_data, is_langgraph)
    
    def _format_response(
        self,
        response: str,
        original_input: Union[str, Dict[str, Any]],
        is_langgraph: bool
    ) -> Union[str, Dict[str, Any]]:
        """Format response based on input type.
        
        Args:
            response: The text response.
            original_input: The original input (for state preservation).
            is_langgraph: Whether input was a LangGraph state dict.
            
        Returns:
            String (if input was string) or updated state dict.
        """
        if not is_langgraph:
            return response
        
        # Build LangGraph-compatible response
        if _HAS_LANGCHAIN:
            ai_message = AIMessage(content=response)
        else:
            ai_message = {"role": "assistant", "content": response}
        
        # Preserve original state and add response
        if isinstance(original_input, dict):
            messages = list(original_input.get("messages", []))
            messages.append(ai_message)
            return {
                **original_input,
                "messages": messages,
                "next": "__end__",
            }
        
        return {"messages": [ai_message], "next": "__end__"}
    
    def run(self, query: str) -> str:
        """Convenience method - same as process() with string input.
        
        Args:
            query: User's question about what they see.
            
        Returns:
            Description/analysis of the visual content.
        """
        result = self.process(query)
        # Ensure we return a string
        if isinstance(result, dict):
            messages = result.get("messages", [])
            if messages:
                last = messages[-1]
                if hasattr(last, "content"):
                    return last.content
                elif isinstance(last, dict):
                    return last.get("content", "")
        return result
    
    # =========================================================================
    # Sentry Control
    # =========================================================================
    
    def start_sentry(self) -> bool:
        """Start the Sentry background monitoring thread."""
        try:
            from src.agents.iris.sentry import start_sentry
        except ImportError:
            print("👁️ Sentry module not available")
            return False
        
        if self._sentry is not None and hasattr(self._sentry, 'is_alive'):
            if self._sentry.is_alive():
                print("👁️ Sentry is already running")
                return False
        
        try:
            self._sentry = start_sentry()
            print("👁️ ✅ Sentry: ENABLED")
            return True
        except Exception as e:
            print(f"❌ Failed to start Sentry: {e}")
            return False
    
    def stop_sentry(self) -> bool:
        """Stop the Sentry background monitoring thread."""
        if self._sentry is None:
            print("⚠️  Sentry is not active")
            return False
        
        try:
            from src.agents.iris.sentry import stop_sentry
            stop_sentry()
            self._sentry = None
            print("👁️ ❌ Sentry: DISABLED")
            return True
        except Exception as e:
            print(f"❌ Failed to stop Sentry: {e}")
            return False
    
    @property
    def is_ready(self) -> bool:
        """Check if IRIS is ready."""
        return self._initialized


# =============================================================================
# LangGraph Node Function
# =============================================================================

def run_iris_agent(state: dict) -> dict:
    """IRIS LangGraph Node function.
    
    Args:
        state: LangGraph state dict.
        
    Returns:
        Updated state with IRIS response.
    """
    agent = IrisAgent()
    return agent.process(state)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "IrisAgent",
    "run_iris_agent",
]
