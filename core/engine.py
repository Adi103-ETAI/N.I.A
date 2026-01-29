"""N.I.A. Core Engine - Central Nervous System.

v2.5.2 "Velocity" - The Orchestrator
    Contains the NIAAssistant class that coordinates all components:
    - NOLA (Voice I/O) with wake word detection
    - NIA Supervisor (LLM-based routing) with SafeLLM protection
    - TARA (Tool Execution) via LangGraph SubGraph
    - IRIS (Vision) for screen/webcam analysis
    - Memory (4-Layer Hybrid) for context injection

Data Flow:
    User -> NOLA/Terminal -> Reflex Layer -> NIAAssistant
                                    |
                                    v
    Supervisor <-> SafeLLM <-> ModelManager <-> [NVIDIA|OpenAI|Groq|Ollama]
        |
        +-> ROUTE:TARA -> Tool Execution
        +-> ROUTE:IRIS -> Vision Analysis  
        +-> ROUTE:CHAT -> Direct Response
        |
        v
    Memory Storage -> Response -> NOLA/Terminal -> User

Version: 2.5.2
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", message=".*timeout is not default parameter.*")

import json
import os
import time
import sys
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple

# Centralized Logging (lightweight - loads quickly)
from core.logger import setup_logger
import asyncio
import aioconsole
from core.services import ServiceRegistry
from core.event_bus import get_event_bus

# =============================================================================
# TYPE_CHECKING Block: IDE-only imports (no runtime cost)
# =============================================================================
if TYPE_CHECKING:
    # 🌊 RIPPLE SAFE: These imports are for type hints only, not loaded at runtime
    from core.memory import MemoryManager

    # 🌊 LAZY LOAD: core.logger functions (for runtime switching)
    from core.logger import set_console_level, logging

# =============================================================================
# Lazy Import Markers (actual imports happen in _init_nia)
# =============================================================================
# The following were previously top-level imports that blocked startup:
# - from core.memory import get_memory_manager  (ChromaDB, SQLite, NetworkX)
# Now deferred to: _init_nia()

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
    config_dir = Path(__file__).parent.parent / "config" / "tara"
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
        
        # Components (Now managed via ServiceRegistry)
        self._nia_process: Optional[callable] = None
        self._running: bool = False
        
        # 4-Layer Memory System (initialized in _init_nia, typed for IDE)
        self.memory: Optional['MemoryManager'] = None
        
        # Plugin Hot-Reload Watcher (v3.0)
        self._plugin_observer: Optional[object] = None

        # Ghost Mode Cache (TTL: 1.0s) to prevent blocking I/O
        self._ghost_cache: Tuple[bool, int] = (False, 0)
        self._ghost_last_check: float = 0.0

        # Input Queue for serialization (Voice + Text)
        self.processing_queue = asyncio.Queue()
        
        # Event Bus Subscription
        self.bus = get_event_bus()
        self.bus.subscribe("voice_command", self._handle_voice_command)

        # UI Lock (Semaphore) for Turn-Taking
        # Green (Set) = Ready for Input
        # Red (Clear) = System Busy / Processing
        self.ui_lock = asyncio.Event()
        self.ui_lock.set()  # Start Green

        self.system_commands = {}
        self._init_command_registry()

    def _handle_voice_command(self, text: str) -> None:
        """Callback for EventBus voice commands."""
        if not text or not text.strip():
            self.logger.warning("[DEBUG] Ignored empty voice input")
            return
        self.logger.info(f"🎤 Voice Command Queued: '{text}'")
        self.processing_queue.put_nowait(text)

    def _init_command_registry(self) -> None:
        """Initialize the strict command dispatch registry."""
        self.system_commands = {
            "reload": self._cmd_reload,
            "debug": self._cmd_toggle_debug,
            "mic": self._cmd_mic_control,
            "ghost": self._cmd_ghost_control,
            "exit": self._cmd_shutdown,
            "quit": self._cmd_shutdown,
            # Standby Triggers (Soft Stop)
            "bye": self._cmd_standby,
            "goodbye": self._cmd_standby,
            "goodnight": self._cmd_standby,
            "standby": self._cmd_standby,
            "sleep": self._cmd_standby,
            "rest": self._cmd_standby,
            
            "help": self._handle_help,     # Keeping existing helper
            "status": self._handle_status, # Keeping existing helper
            "clear": self._handle_clear,   # Keeping existing helper
            "cls": self._handle_clear,
            "history": self._handle_history,
            "reset": self._handle_reset,
        }
    
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
            from nia import aprocess_input
            
            t1 = time.perf_counter()
            self.logger.debug(f"Step 2: NIA import complete ({t1-t0:.2f}s)")
            
            self._nia_process = aprocess_input
            
            t2 = time.perf_counter()
            self.logger.debug(f"Step 3: Process function assigned ({t2-t1:.2f}s)")
            
            # self.logger.info("🧠 NIA brain initialized (total: %.2fs)", t2-t0)
            
            # 🌊 LAZY LOAD: 4-Layer Memory System (ChromaDB, SQLite, NetworkX)
            try:
                from core.memory import get_memory_manager
                self.memory = get_memory_manager()
                
                # Register in ServiceRegistry for centralized access
                ServiceRegistry.register("memory", self.memory)
                
                # Housekeeping: vacuum SQLite databases on startup
                if hasattr(self.memory, '_vacuum_memory_db'):
                    self.memory._vacuum_memory_db()
            except Exception as mem_exc:
                self.logger.warning("Memory init failed (continuing without): %s", mem_exc)
                self.memory = None
            
            # 🔌 PLUGIN SYSTEM: Start hot-reload watcher (v3.0)
            try:
                from tara.plugin_system.watcher import start_plugin_watcher
                self._plugin_observer = start_plugin_watcher()
            except ImportError as plugin_exc:
                self.logger.debug("Plugin watcher not available: %s", plugin_exc)
            except Exception as plugin_exc:
                self.logger.warning("Plugin watcher failed (continuing without): %s", plugin_exc)
            
            return True
        except ImportError as exc:
            self.logger.error("❌ Failed to import NIA: %s", exc)
            return False

    
    # NOLA is now initialized externally and registered as "voice"
    # IRIS is now initialized externally and registered as "vision"
    
    # NOTE: IRIS is now managed by NIAGraph singleton - no separate init needed
    
    def _check_ghost_state(self) -> Tuple[bool, int]:
        """Check if ghost mode is active by reading state file.
        
        Optimized with TTL Cache (1.0s) to prevent blocking I/O in main loop.
        
        Returns:
            Tuple of (is_active, layer). Defaults to (False, 0) on any error.
        """
        # ⚡ Fast Path: Return cached result if within TTL
        if time.time() - self._ghost_last_check < 1.0:
            return self._ghost_cache
            
        try:
            from core.config import settings
            state_file = settings.GHOST_STATE_FILE
            if not os.path.exists(state_file):
                self._update_ghost_cache((False, 0))
                return (False, 0)
            
            with open(state_file, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            active = state.get("active", False)
            layer = state.get("layer", 0)
            
            # Update cache
            self._update_ghost_cache((active, layer))
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

    def _update_ghost_cache(self, state: Tuple[bool, int]) -> None:
        """Helper to update ghost cache and timestamp."""
        self._ghost_cache = state
        self._ghost_last_check = time.time()
    
    # Sentry moved to external IRIS agent
    
    async def start(self) -> bool:
        """Start the assistant and initialize all components."""
        
        # 🌊 RIPPLE FIX: Check if we are recovering from a reload
        if os.environ.get("NIA_RELOADED"):
             self.logger.info("🔄 NIA Reloaded successfully")
             # Clean env
             os.environ.pop("NIA_RELOADED", None)
        

        self.logger.debug("Initializing N.I.A. Core Engine...")
        
        # Initialize NIA
        if not self._init_nia():
            self.logger.error("Failed to initialize NIA brain")
            return False
        
        # NOTE: IRIS is initialized via NIAGraph singleton (no duplicate needed)
        
        self._running = True
        
        # Sentry checks are now delegated to "vision" service
        self.logger.debug("Sentry managed by Vision Service (if active)")
        
        # Log mode info (Silent Boot: DEBUG only)
        if self.voice_mode:
            if self.wake_word_enabled:
                self.logger.debug(f"Voice mode active | Wake words: {', '.join(self.wake_words)}")
            else:
                self.logger.debug("Voice mode active (always listening)")
        else:
            self.logger.debug("Text mode active")
        
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
        
        if self._running:
            # Stop NOLA via Registry
            nola = ServiceRegistry.get("voice")
            if nola:
                self.logger.info("🔇 Stopping NOLA...")
                nola.stop()
        
        # Stop Plugin Watcher (v3.0)
        if self._plugin_observer:
            try:
                from tara.plugin_system.watcher import stop_plugin_watcher
                stop_plugin_watcher(self._plugin_observer)
                self._plugin_observer = None
            except Exception as e:
                self.logger.debug(f"Plugin watcher stop error: {e}")
        
        self.logger.info("👋 NIA shutdown complete")
    
    async def process(self, text: str) -> str:
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
                    self.logger.debug(f"⚡ One-Shot: '{cleaned}'")
                    text = cleaned
                else:
                    self.logger.debug("🎤 Wake Word Detected. Listening...")
                    self.speak("Yes, Director?")
                    return "Listening..."
                break
        
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
                # 🌊 ASYNC MEMORY CALL
                ctx = await self.memory.get_full_context(text)
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
            
            # Run Async NIA process directly (Native)
            response = await self._nia_process(augmented_input, thread_id=self.thread_id)
            
            # Store Episodes (Layer 1) - store original text, not augmented
            if self.memory:
                try:
                    # 🌊 ASYNC MEMORY CALLS with loud logging
                    user_preview = text[:50] if len(text) > 50 else text
                    response_preview = response[:50] if len(response) > 50 else response
                    self.logger.debug(f"🧠 [ENGINE] Saving to Memory -> User: '{user_preview}...' | AI: '{response_preview}...'")
                    await self.memory.store_episode(text, role="user")
                    await self.memory.store_episode(response, role="assistant")
                    self.logger.debug("🧠 [ENGINE] Memory save complete!")
                except Exception as e:
                    self.logger.error(f"🧠 [ENGINE] Memory storage FAILED: {e}")
            
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
            self.logger.debug(f"🤫 [Ghost Mode: Audio Suppressed] NIA: {text}")
            return
        
        if text:
            nola = ServiceRegistry.get("voice")
            if nola:
                try:
                    nola.speak(text)
                except OSError as exc:
                    self.logger.error(f"Audio device error: {exc}", exc_info=True)
                except RuntimeError as exc:
                    self.logger.error(f"TTS runtime error: {exc}", exc_info=True)
    
    async def run(self) -> None:
        """Main application loop (Concurrent).
        
        Handles both voice events (via Queue) and keyboard input (via asyncio task)
        concurrently without blocking.
        """
        print("\033[0m", end="", flush=True) # Reset terminal colors
        
        # Show Splash Screen (High-Quality ASCII)
        import os

        from interface.banner import BANNER
        print("\n" + BANNER + "\n")
        
        if not await self.start():
            return
        
        # We no longer use TerminalUI's context manager strictly if we want async input
        ui_print = print
        if TerminalUI:
            try:
                # Basic instantiation if needed for other things, but input is replaced
                _ui = TerminalUI()
                ui_print = _ui.print
            except:
                pass

        print("\nType 'help' for commands, 'exit' to quit.\n")
        
        # Capture the loop for EventBus thread-safety
        self.bus.set_loop(asyncio.get_running_loop())

        # Start Keyboard Listener Task
        keyboard_task = asyncio.create_task(self._keyboard_listener())

        # Main Processing Loop
        self.logger.info("🚀 Main Loop Started (Concurrent Mode)")
        
        try:
            while self._running:
                # Ghost Protocol Watchdog - Auto-engage stealth sentry on Layer 3
                # 🌊 ASYNC OFFLOAD: Prevent blocking I/O in main loop
                is_ghost, layer = await asyncio.to_thread(self._check_ghost_state)
                if is_ghost and layer >= 3:
                     # Delegate Sentry toggle to Vision Service
                     vision = ServiceRegistry.get("vision")
                     if vision and hasattr(vision, "start_sentry"):
                          # Only log once periodically? keeping simple
                          pass 
                
                # Wait for next input (Voice OR Text)
                input_text = await self.processing_queue.get()
                
                # Validation: Ghostbuster (Silence Filter)
                if not input_text or not input_text.strip():
                    self.ui_lock.set()  # Release lock immediately
                    continue
                
                # Handle commands locally
                if self._handle_command(input_text):
                    self.ui_lock.set() # Commands are fast, release immediately
                    continue
                
                # Process through NIA brain (Async)
                self.logger.debug(f"Processing: {input_text[:50]}...")
                
                try:
                    response = await self.process(input_text)
                    
                    # UI Output
                    if ui_print:
                        ui_print(f"\n💬 NIA: {response}\n")
                    else:
                        print(f"\n💬 NIA: {response}\n")
                    
                    # Speak Response (Blocking I/O wrapped in thread)
                    await asyncio.to_thread(self.speak, response)
                finally:
                    # 🌊 RIPPLE SAFE: Always release lock, even on error
                    self.ui_lock.set()
                
        except asyncio.CancelledError:
            print("\n👋 Goodbye!")
        except KeyboardInterrupt:
            print("\n👋 Interrupted by user")
        except Exception as e:
            self.logger.error(f"Loop error: {e}", exc_info=True)
        finally:
            keyboard_task.cancel()
            self.stop()

    async def _keyboard_listener(self) -> None:
        """Background task for keyboard input."""
        while self._running:
            try:
                # 🚦 TURN-TAKING: Wait for Green Light (System Ready)
                await self.ui_lock.wait()
                
                # \033[1;92m = Bold Bright Green, \033[0m = Reset
                user_input = await aioconsole.ainput("\033[1;92mYou\033[0m: ")
                
                if user_input and user_input.strip():
                    # 🔴 Stop Light: System Busy (Lock UI immediately)
                    self.ui_lock.clear()
                    await self.processing_queue.put(user_input.strip())
            except asyncio.CancelledError:
                break
            except EOFError:
                break
            except Exception as e:
                self.logger.error(f"Keyboard input error: {e}")
                await asyncio.sleep(1.0)
    
    def _handle_command(self, text: str) -> bool:
        """Handle built-in commands using Dispatcher Pattern.
        
        Args:
            text: User input to check for commands.
            
        Returns:
            True if command was handled, False to continue to NIA brain.
        """
        cmd = text.lower().strip()
        verb = cmd.split()[0] if cmd else ""
        
        # 1. Strict Dispatch (Administrator Override)
        if verb in self.system_commands:
            return self.system_commands[verb](cmd)
            
        # 2. Reflex Layer (Fuzzy Hardware Control)
        if self._handle_reflex(cmd):
            return True
            
        return False
        
    # =========================================================================
    # COMMAND HANDLERS (Priority 0)
    # =========================================================================

    def _cmd_toggle_debug(self, text: str) -> bool:
        """Handle debug on/off."""
        if "on" in text:
            from core.logger import set_console_level, logging
            set_console_level(logging.DEBUG)
            self.debug = True
            print("🐞 Debug Mode: ON")
        elif "off" in text:
            from core.logger import set_console_level, logging
            set_console_level(logging.WARNING)
            self.debug = False
            print("🐞 Debug Mode: OFF")
        else:
            print(f"Debug is {'ON' if self.debug else 'OFF'}")
        return True

    def _cmd_reload(self, text: str) -> bool:
        """Handle system reload (restart script)."""
        print("[SYSTEM] 🔄 Reloading...")
        self.stop()
        
        # Mark reload in env to detect on startup
        os.environ["NIA_RELOADED"] = "1"
        
        # Re-execute the current script
        os.execv(sys.executable, ['python'] + sys.argv)
        return True

    def _cmd_ghost_control(self, text: str) -> bool:
        """Handle ghost mode on/off."""
        from core.config import settings
        state_file = settings.GHOST_STATE_FILE
        try:
             # Default to False
             new_state = False
             
             if "on" in text:
                 new_state = True
             elif "off" in text:
                 new_state = False
             else:
                 # Toggle if no arg? No, safer to just show status
                 active, layer = self._check_ghost_state()
                 print(f"👻 Ghost Mode: {'ON' if active else 'OFF'} (Layer {layer})")
                 return True

             # Write state
             layer_val = 3 if new_state else 0
             with open(state_file, "w") as f:
                 json.dump({"active": new_state, "layer": layer_val}, f)
             
             # 🌊 RIPPLE FIX: Update cache immediately to reflect change
             self._update_ghost_cache((new_state, layer_val))
             
             if new_state:
                 print("👻 [Ghost Mode ENABLED] Audio Suppressed. Stealth Sentry Active.")
             else:
                 print("👻 [Ghost Mode DISABLED] Systems Normal.")
                 
        except Exception as e:
            self.logger.error(f"Ghost toggle failed: {e}")
            print(f"⚠️ Failed to toggle Ghost Mode: {e}")
            
        return True
        
    def _cmd_mic_control(self, text: str) -> bool:
        """Handle mic on/off."""
        if "off" in text or "mute" in text:
            nola = ServiceRegistry.get("voice")
            if nola:
                nola.pause_listening()
            try:
                from main import print_mic_off
                print_mic_off()
            except ImportError:
                print("🔇 Microphone Paused")
            self.speak("Voice system offline.")
            return True
            
        if "on" in text or "unmute" in text:
            if not self.voice_mode:
                self.voice_mode = True
                if self.voice_mode: # We still track the boolean preference
                    print("[SYSTEM] 🎙️ Voice System ONLINE")
                    self.speak("Voice system online.")
            else:
                nola = ServiceRegistry.get("voice")
                if nola:
                    nola.resume_listening()
                print("[SYSTEM] 🎙️ Voice System ONLINE")
                self.speak("Voice system online.")
            return True
            
        print("Usage: mic [on|off]")
        return True

    def _cmd_shutdown(self, text: str) -> bool:
        """Handle exit/quit."""
        if self.voice_mode:
            self.speak("Goodbye!")
        print("👋 Goodbye!")
        self._running = False
        return True

    def _cmd_standby(self, text: str) -> bool:
        """
        Enter Standby Mode (End current conversation, keep system running).
        Acts as a 'Soft Stop' - the system stops processing the current turn 
        and returns to the main loop to await the next wake word.
        """
        # Visual Feedback
        print("\n[SYSTEM] 🌙 Entering Standby Mode...")

        # Audio Feedback (Non-blocking if possible, or short)
        self.speak("Standing by, Director.")

        # Return True to signal 'Command Handled'. 
        # This prevents the text from falling through to the AI Brain.
        return True

    def _handle_help(self, text: str) -> bool:
        self._print_help()
        return True

    def _handle_clear(self, text: str) -> bool:
        subprocess.run(['cls' if os.name == 'nt' else 'clear'], shell=True, check=False)
        return True

    def _handle_status(self, text: str) -> bool:
        self._print_status()
        return True

    def _handle_prefs(self, text: str) -> bool:
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

    def _handle_history(self, text: str) -> bool:
        """Handle history commands (view/clear)."""
        # "history clear" or "clear history"
        if "clear" in text:
            try:
                from nia import clear_conversation
                if clear_conversation(self.thread_id):
                    print("✅ History cleared")
                else:
                    print("⚠️  Could not clear history")
            except Exception as exc:
                print(f"❌ Error: {exc}")
            return True
            
        # Default: View history
        try:
            from nia import get_conversation_history
            history = get_conversation_history(self.thread_id)
            if history:
                print(f"📜 History ({len(history)} messages):")
                # Show last 5 interactions (10 messages roughly)
                for msg in history[-10:]:
                    role = getattr(msg, 'type', 'unknown')
                    content = getattr(msg, 'content', str(msg))[:100]
                    content = content.replace('\n', ' ')
                    print(f"   [{role}]: {content}")
            else:
                print("📜 History empty")
        except Exception as exc:
            print(f"⚠️  Could not retrieve history: {exc}")
        return True

    def _handle_reset(self, text: str) -> bool:
        """Handle component resets."""
        if "audio" in text:
            print("⚙️  Resetting audio engine...")
            if self._nola:
                self._nola.stop_speaking()
                self._nola.resume_listening()
            print("✅ Audio reset complete")
            return True
        return False
         
    # =========================================================================
    # REFLEX HANDLERS (Priority 1 - Fuzzy)
    # =========================================================================

    def _handle_reflex(self, text: str) -> bool:
        """Handle hardware reflex commands (mic, speaker, sentry)."""
        cmd = text.lower()
        
        # 🎤 MIC CONTROL
        # Note: 'mic on'/'mic off' are handled by strict dispatcher now.
        # This block catches legacy variations if needed, or we can just return False.
        # For safety/consistency with user request, we'll keep the logic but delegate to strict if match found?
        # Actually user said "If no match, check _handle_reflex".
        
        if "sentry on" in cmd or "activate sentry" in cmd:
            if not self.sentry_thread:
                self._init_sentry()
                print("👁️ ✅ Sentry: ONLINE")
            else:
                print("⚠️  Sentry is already running")
            return True
        
        if "sentry off" in cmd or "deactivate sentry" in cmd:
            if self.sentry_thread:
                self.sentry_thread.stop()
                self.sentry_thread = None
                print("👁️ ❌ Sentry: OFFLINE")
            else:
                print("⚠️  Sentry is not active")
            return True

        # 🔇 TTS STOP
        if "shut up" in cmd or "stop talking" in cmd or "silence" in cmd:
             if self._nola:
                self._nola.stop_speaking()
             print("🔇 Silenced")
             return True

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
        # 🔧 FIX: Use ServiceRegistry instead of dead instance variables
        from core.services import ServiceRegistry
        
        nola_manager = ServiceRegistry.get("voice")
        iris_manager = ServiceRegistry.get("iris")
        
        s_nia  = "ON " 
        s_nola = "ON " if (nola_manager and getattr(nola_manager, 'is_active', lambda: False)()) else "OFF"
        s_iris = "ON " if iris_manager else "OFF"
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
