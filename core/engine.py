"""N.I.A. Core Engine - Central Nervous System.

Contains the NIAAssistant class that orchestrates all components.
"""
from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple

# Centralized Logging (lightweight - loads quickly)
from core.logger import setup_logger

# =============================================================================
# TYPE_CHECKING Block: IDE-only imports (no runtime cost)
# =============================================================================
if TYPE_CHECKING:
    # 🌊 RIPPLE SAFE: These imports are for type hints only, not loaded at runtime
    from core.memory import MemoryManager

# =============================================================================
# Lazy Import Markers (actual imports happen in _init_nia)
# =============================================================================
# The following were previously top-level imports that blocked startup:
# - from core.memory import get_memory_manager  (ChromaDB, SQLite, NetworkX)
# Now deferred to: _init_nia()

# Import banner (lightweight - just a string constant)
try:
    from interface.banner import MINI_BANNER
except ImportError:
    MINI_BANNER = "N.I.A. - Neural Intelligence Assistant"

# Import Terminal UI (lazy-ish - only UI components)
try:
    from interface.chat import TerminalUI
except ImportError:
    TerminalUI = None


# =============================================================================
# Config Loader (Dynamic from External Files)
# =============================================================================

def _load_engine_config() -> dict:
    """Load engine configuration from external files.
    
    Returns:
        Dictionary with command vocabularies and help text.
    """
    config_dir = Path(__file__).parent / "config"
    config = {}
    
    # Load command vocabularies
    commands_path = config_dir / "commands.json"
    if commands_path.exists():
        with open(commands_path, "r", encoding="utf-8") as f:
            config["commands"] = json.load(f)
    else:
        config["commands"] = {}
    
    # Load help text
    help_path = config_dir / "help.txt"
    if help_path.exists():
        with open(help_path, "r", encoding="utf-8") as f:
            config["help_text"] = f.read()
    else:
        config["help_text"] = "Type 'exit' to quit."
    
    return config


# Load at module level (cached)
_ENGINE_CONFIG = _load_engine_config()
_COMMANDS = _ENGINE_CONFIG.get("commands", {})
_HELP_TEXT = _ENGINE_CONFIG.get("help_text", "")


# =============================================================================
# NIAAssistant Class
# =============================================================================

class NIAAssistant:
    """Main NIA Voice Assistant application.
    
    The NIAAssistant orchestrates all N.I.A. components including:
    - NIA Brain (LangGraph-based reasoning)
    - NOLA (Voice I/O with wake word detection)
    - IRIS (Vision analysis)
    - TARA (Tool execution)
    
    Attributes:
        voice_mode: Whether voice I/O is enabled.
        wake_word_enabled: Whether wake word detection is active.
        wake_words: List of wake word triggers.
        thread_id: Conversation thread identifier.
        debug: Enable debug logging.
        
    Example:
        >>> assistant = NIAAssistant(voice_mode=True)
        >>> assistant.run()  # Blocking main loop
    """
    
    def __init__(
        self,
        voice_mode: bool = False,
        wake_word_enabled: bool = True,
        wake_words: Optional[list[str]] = None,
        thread_id: str = "root",
        debug: bool = False,
    ) -> None:
        """Initialize the NIAAssistant.
        
        NOTE: This constructor is LIGHTWEIGHT for instant startup.
        Heavy components (NIA brain, Memory) are loaded in start() -> _init_nia().
        
        Args:
            voice_mode: Enable voice input/output via NOLA.
            wake_word_enabled: Require wake word before accepting commands.
            wake_words: List of wake word triggers (default: ["jarvis", "nia", "hey nia"]).
            thread_id: Unique identifier for conversation thread persistence.
            debug: Enable verbose debug logging.
        """
        # 🌊 RIPPLE SAFE: Only lightweight assignments here - NO heavy imports
        self.voice_mode: bool = voice_mode
        self.wake_word_enabled: bool = wake_word_enabled
        self.wake_words: list[str] = wake_words or ["jarvis", "nia", "hey nia"]
        self.thread_id: str = thread_id
        self.debug: bool = debug
        
        # Centralized logger (lightweight - already imported at module level)
        self.logger = setup_logger("BRAIN")
        
        # Components (lazy initialization in start() -> _init_*)
        self._nia_process: Optional[callable] = None
        self._nola: Optional[object] = None
        self.sentry_thread: Optional[object] = None
        self._running: bool = False
        
        # 4-Layer Memory System (initialized in _init_nia, typed for IDE)
        self.memory: Optional['MemoryManager'] = None
    
    def _init_nia(self) -> bool:
        """Initialize the NIA brain (LangGraph reasoning engine).
        
        NOTE: This is where ALL heavy imports happen (lazy loading).
        
        Returns:
            True if NIA brain initialized successfully, False otherwise.
        """
        import time
        try:
            self.logger.debug("Step 1: Starting NIA import...")
            t0 = time.perf_counter()
            
            # 🌊 LAZY LOAD: NIA brain (LangGraph, LangChain, NVIDIA API)
            from nia import process_input
            
            t1 = time.perf_counter()
            self.logger.debug(f"Step 2: NIA import complete ({t1-t0:.2f}s)")
            
            self._nia_process = process_input
            
            t2 = time.perf_counter()
            self.logger.debug(f"Step 3: Process function assigned ({t2-t1:.2f}s)")
            
            self.logger.info("🧠 NIA brain initialized (total: %.2fs)", t2-t0)
            
            # 🌊 LAZY LOAD: 4-Layer Memory System (ChromaDB, SQLite, NetworkX)
            try:
                from core.memory import get_memory_manager
                self.memory = get_memory_manager()
                self.logger.info("💾 Memory connected: %s", self.memory.get_stats())
                
                # Housekeeping: vacuum SQLite databases on startup
                if hasattr(self.memory, '_vacuum_memory_db'):
                    self.memory._vacuum_memory_db()
            except Exception as mem_exc:
                self.logger.warning("Memory init failed (continuing without): %s", mem_exc)
                self.memory = None
            
            return True
        except ImportError as exc:
            self.logger.error("❌ Failed to import NIA: %s", exc)
            return False

    
    def _init_nola(self) -> bool:
        """Initialize NOLA voice system via singleton.
        
        Creates the NOLAManager with wake word configuration and starts
        the voice processing loop if voice_mode is enabled.
        
        Returns:
            True if NOLA initialized and started, False otherwise.
        """
        if not self.voice_mode:
            return True
        
        try:
            from nola.manager import get_nola_manager, NOLAConfig
            
            config = NOLAConfig(
                wake_word_enabled=self.wake_word_enabled,
                wake_words=self.wake_words,
                wake_word_timeout=30.0,
                security_enabled=True,
                pause_ear_while_speaking=True,
            )
            
            # Use singleton manager
            self._nola = get_nola_manager(config=config)
            
            if self._nola.start():
                self.logger.info("🎤 NOLA voice system initialized")
                return True
            else:
                self.logger.error("❌ NOLA failed to start")
                return False
                
        except ImportError as exc:
            self.logger.error("❌ Failed to import NOLA: %s", exc)
            return False
    
    # NOTE: IRIS is now managed by NIAGraph singleton - no separate init needed
    
    def _check_ghost_state(self) -> Tuple[bool, int]:
        """Check if ghost mode is active by reading state file.
        
        Returns:
            Tuple of (is_active, layer). Defaults to (False, 0) on any error.
        """
        try:
            state_file = "data/ghost_state.json"
            if not os.path.exists(state_file):
                return (False, 0)
            
            with open(state_file, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            active = state.get("active", False)
            layer = state.get("layer", 0)
            return (active, layer)
            
        except json.JSONDecodeError as e:
            self.logger.warning(f"Ghost state file corrupted: {e}")
            return (False, 0)
        except (OSError, IOError) as e:
            self.logger.debug(f"Ghost state file not readable: {e}")
            return (False, 0)
        except (KeyError, TypeError) as e:
            self.logger.warning(f"Invalid ghost state format: {e}")
            return (False, 0)
    
    def _init_sentry(self, headless: bool = False) -> None:
        """Initialize IRIS Sentry for Security & Communications monitoring.
        
        Args:
            headless: If True, run in stealth mode (no visible window).
        """
        try:
            from iris.sentry import start_sentry
            
            def sentry_callback(alert_type: str, found_keyword: str):
                """Handle sentry alerts via voice."""
                if alert_type == "SECURITY":
                    print("🚨", end="", flush=True)
                    self.speak("Security alert. Sensitive information visible.")
                elif alert_type == "COMMS":
                    print("📩", end="", flush=True)
                    self.speak("You have a new message.")
            
            self.sentry_thread = start_sentry(
                callback=sentry_callback,
                interval=8,
                headless=headless
            )
            if self.sentry_thread:
                if headless:
                    self.logger.debug("👁️ Sentry started (Stealth Mode)")
                else:
                    self.logger.debug("👁️ IRIS Sentry started")
                
        except ImportError:
            self.logger.debug("IRIS Sentry not available")
    
    def start(self) -> bool:
        """Start the assistant and initialize all components.
        
        Initializes NIA brain, IRIS vision (optional), and NOLA voice (if enabled).
        Must be called before run() or process().
        
        Returns:
            True if core components initialized successfully.
            
        Example:
            >>> assistant = NIAAssistant()
            >>> if assistant.start():
            ...     response = assistant.process("Hello")
        """
        print(MINI_BANNER)
        self.logger.info("Initializing N.I.A. Core Engine...")
        
        # Initialize NIA
        if not self._init_nia():
            self.logger.error("Failed to initialize NIA brain")
            return False
        
        # NOTE: IRIS is initialized via NIAGraph singleton (no duplicate needed)
        
        # Initialize NOLA (if voice mode)
        if self.voice_mode:
            if not self._init_nola():
                self.logger.warning("Voice mode unavailable, continuing in text mode")
                self.voice_mode = False
        
        self._running = True
        
        # Sentry now manual-start only (use 'sentry on')
        self.logger.debug("Sentry in standby mode (use 'sentry on' to activate)")
        
        # Log mode info
        if self.voice_mode:
            if self.wake_word_enabled:
                self.logger.info(f"Voice mode active | Wake words: {', '.join(self.wake_words)}")
            else:
                self.logger.info("Voice mode active (always listening)")
        else:
            self.logger.info("Text mode active")
        
        print("\nType 'help' for commands, 'exit' to quit.\n")
        
        return True
    
    def stop(self) -> None:
        """Stop the assistant and cleanup all resources.
        
        Gracefully shuts down NOLA voice system and IRIS Sentry.
        Safe to call multiple times.
        """
        self._running = False
        
        # Stop IRIS Sentry
        try:
            from iris.sentry import stop_sentry
            stop_sentry()
        except ImportError:
            pass
        
        if self._nola:
            self.logger.info("🔇 Stopping NOLA...")
            self._nola.stop()
            self._nola = None
        
        self.logger.info("👋 NIA shutdown complete")
    
    def process(self, text: str) -> str:
        """Process user input through the NIA brain.
        
        Handles wake word detection, fast-path responses (time/date),
        and routes complex queries to the LangGraph reasoning engine.
        
        Args:
            text: User input text to process.
            
        Returns:
            AI response string.
            
        Raises:
            ConnectionError: Network issues with AI service.
            TimeoutError: Request timeout.
        """
        if not text:
            return ""
        
        # The wake-up-only signal handling is now done exclusively in NOLA before calling process()
        
        # Handle wake words in commands
        text_lower = text.lower().strip()
        for wake_word in self.wake_words:
            if text_lower.startswith(wake_word):
                cleaned = text[len(wake_word):].strip()
                if cleaned:
                    print(f"⚡ One-Shot: '{cleaned}'")
                    text = cleaned
                else:
                    print("🎤 Wake Word Detected. Listening...")
                    self.speak("Yes, Director?")
                    return "Listening..."
                break
        
        # Fast path (time/date)
        fast_response = self._handle_fast_path(text)
        if fast_response:
            print(f"⚡ Reflex: {fast_response}")
            return fast_response
        
        if not self._nia_process:
            return "❌ NIA brain not initialized."
        
        # Security Check (Layer 4)
        if self.memory and self.memory.is_blocked(text):
            self.logger.warning("🚫 Blocked by security: %s", text[:50])
            return "🚫 Blocked by security protocol."
        
        # Get Memory Context (Layers 1-3)
        memory_context = ""
        if self.memory:
            try:
                ctx = self.memory.get_full_context(text)
                if ctx.get("relevant_episodes") or ctx.get("relevant_skills") or ctx.get("preferences"):
                    memory_context = self._format_memory_context(ctx)
                    self.logger.debug("Memory context: %d episodes, %d skills", 
                                      len(ctx.get('relevant_episodes', [])),
                                      len(ctx.get('relevant_skills', [])))
            except Exception as mem_exc:
                self.logger.debug("Memory context error: %s", mem_exc)
        
        try:
            # Inject memory context into prompt
            if memory_context:
                augmented_input = f"{memory_context}\n\nUser Input: {text}"
                self.logger.debug("Injecting memory context (%d chars) into prompt", len(memory_context))
            else:
                augmented_input = text
            
            response = self._nia_process(augmented_input, thread_id=self.thread_id)
            
            # Store Episodes (Layer 1) - store original text, not augmented
            if self.memory:
                try:
                    self.memory.store_episode(text, role="user")
                    self.memory.store_episode(response, role="assistant")
                except Exception as e:
                    self.logger.debug(f"Memory storage failed: {e}")
            
            return response
        except ConnectionError as exc:
            self.logger.error(f"Network error during NIA processing: {exc}", exc_info=True)
            return "I couldn't connect to the AI service. Please check your network."
        except TimeoutError as exc:
            self.logger.error(f"Timeout during NIA processing: {exc}", exc_info=True)
            return "The request timed out. Please try again."
        except Exception as exc:
            self.logger.error(f"Unexpected NIA processing error: {exc}", exc_info=True)
            return f"I encountered an error: {exc}"
    
    def _format_memory_context(self, ctx: dict) -> str:
        """Format memory context for LLM injection.
        
        Args:
            ctx: Context dictionary from memory.get_full_context().
            
        Returns:
            Formatted context string.
        """
        parts = ["[MEMORY CONTEXT]"]
        
        if ctx.get("preferences"):
            parts.append(f"- User Preferences: {ctx['preferences']}")
        
        if ctx.get("relevant_episodes"):
            episodes = ctx["relevant_episodes"][:3]  # Limit to 3
            parts.append(f"- Relevant Past Chats: {episodes}")
        
        if ctx.get("relevant_skills"):
            parts.append(f"- Known Skills: {ctx['relevant_skills']}")
        
        return "\n".join(parts)
    
    def _handle_fast_path(self, text: str) -> Optional[str]:
        """Handle simple utility queries locally without LLM.
        
        Provides instant responses for time and date queries,
        bypassing the AI for zero-latency user experience.
        
        Args:
            text: User input to check for fast-path patterns.
            
        Returns:
            Response string if fast-path matched, None otherwise.
        """
        query = text.lower().strip()
        now = datetime.now()
        
        # TIME queries
        if any(kw in query for kw in ["time", "clock", "hour"]):
            if any(q in query for q in ["what", "tell", "current", "now"]):
                return f"The current time is {now.strftime('%I:%M %p')}."
        
        # DATE queries
        if any(kw in query for kw in ["date", "day", "today"]):
            if any(q in query for q in ["what", "tell", "current", "today"]):
                suffix = self._get_day_suffix(now.day)
                return f"Today is {now.strftime(f'%A, %B {now.day}{suffix}, %Y')}."
        
        return None
    
    def _get_day_suffix(self, day: int) -> str:
        """Get ordinal suffix for a day number (st, nd, rd, th).
        
        Args:
            day: Day of month (1-31).
            
        Returns:
            Ordinal suffix string.
        """
        if 11 <= day <= 13:
            return "th"
        return {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    
    def speak(self, text: str) -> None:
        """Speak text through NOLA voice system.
        
        Respects Ghost Protocol - suppresses audio when ghost mode is active
        to maintain operational security.
        
        Args:
            text: Text to speak aloud.
            
        Note:
            Silent operation when ghost mode is active (layer 1+).
        """
        # Check ghost mode before speaking
        is_ghost, layer = self._check_ghost_state()
        if is_ghost:
            self.logger.debug(f"Ghost mode active (layer {layer}), audio suppressed")
            print(f"🤫 [Ghost Mode: Audio Suppressed] NIA: {text}")
            return
        
        if self._nola and text:
            try:
                self._nola.speak(text)
            except OSError as exc:
                self.logger.error(f"Audio device error: {exc}", exc_info=True)
            except RuntimeError as exc:
                self.logger.error(f"TTS runtime error: {exc}", exc_info=True)
    
    def run(self) -> None:
        """Main application loop (synchronous blocking).
        
        Starts the assistant, displays the terminal UI, and processes
        user input in a loop until 'exit' or Ctrl+C.
        
        This is the primary entry point for running NIA interactively.
        
        Example:
            >>> assistant = NIAAssistant(voice_mode=True)
            >>> assistant.run()  # Blocks until exit
        """
        if not self.start():
            return
        
        # Initialize Terminal UI
        if TerminalUI:
            ui = TerminalUI()
        else:
            # Fallback
            class FallbackUI:
                def context(self):
                    import contextlib
                    return contextlib.nullcontext()
                def get_input(self, prompt: str) -> str:
                    return input(prompt)
                def print(self, *args, **kwargs):
                    print(*args, **kwargs)
            ui = FallbackUI()
        
        # Main loop
        with ui.context():
            while self._running:
                try:
                    # Ghost Protocol Watchdog - Auto-engage stealth sentry on Layer 3
                    is_ghost, layer = self._check_ghost_state()
                    if is_ghost and layer >= 3 and self.sentry_thread is None:
                        self.logger.info("Ghost Layer 3: Auto-engaging Sentry (Stealth Mode)")
                        self._init_sentry(headless=True)
                    
                    user_input = ui.get_input("You: ")
                    
                    if not user_input or not user_input.strip():
                        continue
                    
                    user_input = user_input.strip()
                    
                    # Handle commands locally
                    if self._handle_command(user_input):
                        continue
                    
                    # Process through NIA brain
                    self.logger.debug(f"Processing: {user_input[:50]}...")
                    response = self.process(user_input)
                    ui.print(f"\n💬 NIA: {response}\n")
                    self.speak(response)
                    
                except KeyboardInterrupt:
                    ui.print("\n\n👋 Goodbye!")
                    break
                except EOFError:
                    continue
        
        self.stop()
    
    def _handle_command(self, text: str) -> bool:
        """Handle built-in commands using vocabulary-based reflex matching.
        
        Reflex commands bypass the NIA brain for zero-latency response.
        Supports fuzzy keyword matching for natural variations.
        
        Args:
            text: User input to check for commands.
            
        Returns:
            True if command was handled, False to continue to NIA brain.
        """
        cmd = text.lower().strip()
        
        # =================================================================
        # FUZZY KEYWORD MATCHING (catches natural variations)
        # =================================================================
        
        # 🎤 MIC OFF: "turn off mic", "disable microphone", "kill the mic"
        # Load mic control words from config
        mic_cfg = _COMMANDS.get("mic_control", {})
        mic_words = mic_cfg.get("mic_words", ["mic", "microphone"])
        off_words = mic_cfg.get("off_words", ["off", "mute", "stop"])
        on_words = mic_cfg.get("on_words", ["on", "unmute", "start"])
        
        has_mic = any(w in cmd for w in mic_words)
        has_off = any(w in cmd for w in off_words)
        has_on = any(w in cmd for w in on_words)
        
        if has_mic and has_off and not has_on:
            # MIC OFF command
            if self._nola:
                self._nola.pause_listening()
            try:
                from main import print_mic_off
                print_mic_off()
            except ImportError:
                print("🔇 Microphone Paused")
            return True
        
        if has_mic and has_on and not has_off:
            # MIC ON command
            if not self.voice_mode:
                self.voice_mode = True
                if self._init_nola():
                    try:
                        from main import print_mic_on
                        print_mic_on()
                    except ImportError:
                        print("🎙️ Microphone Active")
                    self.speak("Voice mode enabled.")
            else:
                if self._nola:
                    self._nola.resume_listening()
                try:
                    from main import print_mic_on
                    print_mic_on()
                except ImportError:
                    print("🎙️ Microphone Active")
            return True
        
        # =================================================================
        # VOCABULARY DEFINITIONS (Loaded from config)
        # =================================================================
        
        # 👁️ IRIS (Sentry/Vision Control)
        iris_cfg = _COMMANDS.get("iris_control", {})
        IRIS_ON = iris_cfg.get("on", [])
        IRIS_OFF = iris_cfg.get("off", [])
        
        # 🔊 TARA (Speaker Mute - Zero Latency Reflex)
        speaker_cfg = _COMMANDS.get("speaker_control", {})
        SPEAKER_MUTE = speaker_cfg.get("mute", [])
        SPEAKER_UNMUTE = speaker_cfg.get("unmute", [])
        
        # 🔇 TTS (Stop Speaking)
        tts_cfg = _COMMANDS.get("tts_control", {})
        TTS_STOP = tts_cfg.get("stop", [])
        
        # ⚙️ SYSTEM (Maintenance)
        sys_cfg = _COMMANDS.get("system_control", {})
        SYS_STATUS = sys_cfg.get("status", [])
        SYS_CLEAR = sys_cfg.get("clear", [])
        SYS_EXIT = sys_cfg.get("exit", [])
        SYS_HELP = sys_cfg.get("help", [])
        
        # =================================================================
        # HELPER: Check if any phrase matches the command
        # =================================================================
        def matches(phrases):
            """Returns True if any phrase is found in cmd."""
            return any(phrase in cmd for phrase in phrases)
        
        def match(*words):
            """Returns True if ALL keywords are present (order-independent)."""
            return all(w in cmd for w in words)
        
        # =================================================================
        # 1. 👁️ IRIS SENTRY CONTROL (Priority)
        # =================================================================
        if matches(IRIS_ON):
            if not self.sentry_thread:
                self._init_sentry()
                print("👁️ ✅ Sentry: ONLINE")
            else:
                print("⚠️  Sentry is already running")
            return True
        
        if matches(IRIS_OFF):
            if self.sentry_thread:
                self.sentry_thread.stop()
                self.sentry_thread = None
                print("👁️ ❌ Sentry: OFFLINE")
            else:
                print("⚠️  Sentry is not active")
            return True
        
        # =================================================================
        # 2. 🔊 SPEAKER MUTE/UNMUTE (Zero-Latency TARA Reflex)
        # =================================================================
        # UNMUTE first (to avoid "unmute" matching "mute")
        if matches(SPEAKER_UNMUTE) and "mic" not in cmd and "microphone" not in cmd:
            try:
                from tara.tools.system_ops import mute_volume
                result = mute_volume(mute=False)
                print(result)  # Tool returns "🔊 System Unmuted"
            except Exception as e:
                print(f"⚠️  Audio control error: {e}")
            return True
        
        # MUTE SPEAKERS (exclude "mic" to prevent false matches)
        if matches(SPEAKER_MUTE) and "mic" not in cmd and "microphone" not in cmd:
            try:
                from tara.tools.system_ops import mute_volume
                result = mute_volume(mute=True)
                print(result)  # Tool returns "🔇 System Muted"
            except Exception as e:
                print(f"⚠️  Audio control error: {e}")
            return True
        
        # Microphone commands now handled by fuzzy matching above
        
        # =================================================================
        # 4. 🔇 TTS CONTROL (Stop Speaking)
        # =================================================================
        if matches(TTS_STOP):
            if self._nola:
                self._nola.stop_speaking()
            print("🔇 Silenced")
            return True
        
        # =================================================================
        # 5. ⚙️ SYSTEM COMMANDS
        # =================================================================
        # EXIT
        if matches(SYS_EXIT):
            if self.voice_mode:
                self.speak("Goodbye!")
            print("👋 Goodbye!")
            self._running = False
            return True
        
        # HELP
        if matches(SYS_HELP):
            self._print_help()
            return True
        
        # STATUS
        if matches(SYS_STATUS):
            self._print_status()
            return True
        
        # CLEAR SCREEN
        if matches(SYS_CLEAR):
            # 🌊 MODERNIZED: subprocess.run instead of os.system (cross-platform)
            subprocess.run(['cls' if os.name == 'nt' else 'clear'], shell=True, check=False)
            print(MINI_BANNER)
            return True
        
        # =================================================================
        # 6. 📋 PREFERENCES (God Mode - Direct DB Access)
        # =================================================================
        if cmd == "prefs":
            try:
                import json
                prefs = self.memory.get_all_preferences() if self.memory else {}
                if prefs:
                    print(f"\n📋 [User Preferences]:\n{json.dumps(prefs, indent=2)}\n")
                else:
                    print("\n📋 No preferences saved yet.\n")
            except Exception as exc:
                print(f"⚠️  Could not retrieve preferences: {exc}")
            return True
        
        # =================================================================
        # 7. 📜 HISTORY MANAGEMENT
        # =================================================================
        if cmd == "history":
            try:
                from nia import get_conversation_history
                history = get_conversation_history(self.thread_id)
                if history:
                    print(f"📜 History ({len(history)} messages):")
                    for msg in history[-10:]:
                        role = getattr(msg, 'type', 'unknown')
                        content = getattr(msg, 'content', str(msg))[:100]
                        print(f"   [{role}]: {content}")
                else:
                    print("📜 History empty")
            except Exception as exc:
                print(f"⚠️  Could not retrieve history: {exc}")
            return True
        
        if match("clear", "history"):
            try:
                from nia import clear_conversation
                if clear_conversation(self.thread_id):
                    print("✅ History cleared")
                else:
                    print("⚠️  Could not clear history")
            except Exception as exc:
                print(f"❌ Error: {exc}")
            return True
        
        # =================================================================
        # 7. 🔧 AUDIO RESET
        # =================================================================
        if match("reset", "audio"):
            print("⚙️  Resetting audio engine...")
            if self._nola:
                self._nola.stop_speaking()
                self._nola.resume_listening()
            print("✅ Audio reset complete")
            return True
        
        # =================================================================
        # NOT A REFLEX - Pass to Brain/TARA
        # =================================================================
        return False
    
    def _print_help(self) -> None:
        """Print help information from external file."""
        print(f"\n{_HELP_TEXT}\n")
    
    def _draw_bar(self, percent: float) -> str:
        """Returns a strict 17-character progress bar string."""
        bar_len = 10
        filled = int((percent / 100) * bar_len)
        bar = "█" * filled + "░" * (bar_len - filled)
        return f"[{bar}] {percent:>3.0f}%"

    def _print_status(self) -> None:
        """Displays the Precision Aligned Dashboard (Strict Grid Layout)."""
        import psutil
        from datetime import datetime

        # 1. Gather Data
        cpu_p = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        dsk = psutil.disk_usage('/')
        
        # 2. Prepare Strings
        time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Status Flags (Strict 3 chars: "ON ", "OFF")
        s_nia  = "ON " 
        s_nola = "ON " if self._nola else "OFF"
        s_iris = "ON " if self.sentry_thread else "OFF"
        s_tara = "ON "
        
        # API Keys (Strict 7 chars)
        k_nv = "LINKED " if os.environ.get("NVIDIA_API_KEY") else "MISSING"
        k_oa = "LINKED " if os.environ.get("OPENAI_API_KEY") else "MISSING"

        # Resource Bars (Strict 17 chars)
        bar_cpu = self._draw_bar(cpu_p)
        bar_ram = self._draw_bar(mem.percent)
        bar_dsk = self._draw_bar(dsk.percent)
        
        # Memory Strings
        mem_used = f"{mem.used / (1024**3):.1f}"
        mem_tot  = f"{mem.total / (1024**3):.1f}"
        mem_str  = f"{mem_used}/{mem_tot} GB"
        dsk_free = f"{dsk.free / (1024**3):.1f} GB Free"

        # 3. Render Dashboard (Grid: Left=29, Right=36)
        print("\n┌" + "─"*29 + "┬" + "─"*36 + "┐")
        print(f"│ N.I.A. SYSTEM DASHBOARD     │ {time_str:>34} │")
        print("├─────────────────────────────┼────────────────────────────────────┤")
        print(f"│ 🧠 SUBSYSTEMS               │ 📊 RESOURCES                       │")
        print(f"│ • BRAIN (NIA) : [{s_nia}]       │  CPU: {bar_cpu:<25}    │")
        print(f"│ • VOICE (NOLA): [{s_nola}]       │  RAM: {bar_ram:<25}    │")
        print(f"│ • SENTRY(IRIS): [{s_iris}]       │  DSK: {bar_dsk:<25}    │")
        print(f"│ • TOOLS (TARA): [{s_tara}]       │                                    │")
        print("├─────────────────────────────┼────────────────────────────────────┤")
        print(f"│ 💾 MEMORY                   │ 🔐 SECURITY KEYS                   │")
        print(f"│  RAM : {mem_str:<21}│  NVIDIA API: [{k_nv:<7}]             │")
        print(f"│  DISK: {dsk_free:<21}│  OPENAI API: [{k_oa:<7}]             │")
        print("└─────────────────────────────┴────────────────────────────────────┘\n")
