"""N.I.A. Core Engine — Central Orchestrator.

The ``NIAAssistant`` class is the single integration point for all N.I.A.
components. It owns the main run loop, voice/keyboard I/O, and delegates
all AI reasoning to the NIA LangGraph brain.

Component map:
    NOLA   — Voice I/O layer (TTS + STT + wake-word detection)
    NIA    — LangGraph brain (Router → Supervisor / TARA / Docker)
    TARA   — Tool execution sub-graph (50+ desktop automation tools)
    IRIS   — Vision agent (screen + webcam analysis)
    Memory — 4-layer hybrid store (SQLite, ChromaDB, NetworkX, in-memory)

Data flow::

    User
      │
      ├─ Voice ──► NOLA ──► EventBus ──► NIAAssistant.process()
      └─ Text  ──► Terminal            ──► NIAAssistant.process()
                                                   │
                              ┌────────────────────┘
                              ▼
                   NIA LangGraph Brain
                    ├─ router_node  → decide: chat / system / swarm
                    ├─ supervisor   → conversational response
                    ├─ call_tara_2  → desktop automation
                    └─ docker_node  → Docker swarm execution
                              │
                              ▼
              Memory.store() + NOLA.speak() / Terminal.print()

Lifecycle::

    assistant = NIAAssistant(voice_mode=True)
    await assistant.start()   # lazy-loads all heavy components
    await assistant.run()     # blocks: voice + keyboard loop
    await assistant.stop()    # graceful shutdown

Version: 4.0.0
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", message=".*timeout is not default parameter.*")

import json

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore
import os
import time
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple

# Centralized Logging (lightweight - loads quickly)
from src.core.logger import setup_logger
import asyncio
import aioconsole
from src.core.di import ServiceRegistry
from src.core.bus import get_event_bus

# =============================================================================
# TYPE_CHECKING Block: IDE-only imports (no runtime cost)
# =============================================================================
if TYPE_CHECKING:
    # 🌊 RIPPLE SAFE: These imports are for type hints only, not loaded at runtime
    from src.core.memory import MemoryManager

    # 🌊 LAZY LOAD: core.logger functions (for runtime switching)
    from src.core.logger import set_console_level, logging

# =============================================================================
# Lazy Import Markers (actual imports happen in _init_nia)
# =============================================================================
# The following were previously top-level imports that blocked startup:
# - from src.core.memory import get_memory_manager  (ChromaDB, SQLite, NetworkX)
# Now deferred to: _init_nia()

# Import Terminal UI (lazy-ish - only UI components)
try:
    from src.interface.chat import TerminalUI
except ImportError:
    TerminalUI = None


# =============================================================================
# Config Loader (Dynamic from External Files)
# =============================================================================

def _load_engine_config() -> dict:
    """Load engine configuration from centralized defaults."""
    config_path = Path(__file__).resolve().parents[1] / "config" / "defaults" / "agents" / "tara.yaml"

    if yaml is None:
        return {"commands": {}, "help_text": "Type 'exit' to quit."}

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        data = {}

    return {
        "commands": data.get("commands", {}),
        "help_text": "Type 'exit' to quit.",
    }


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
        self._nola: Optional[object] = None  # Reference for direct control
        
        # 4-Layer Memory System (initialized in _init_nia, typed for IDE)
        self.memory: Optional['MemoryManager'] = None
        
        # Plugin Hot-Reload Watcher (v3.0)
        self._plugin_observer: Optional[object] = None

        # Ghost Mode Cache (TTL: 1.0s) to prevent blocking I/O
        self._ghost_cache: Tuple[bool, int] = (False, 0)
        self._ghost_last_check: float = 0.0

        # Sentry thread reference (for legacy reflex handlers)
        self.sentry_thread: Optional[object] = None

        # Input Queue for serialization (Voice + Text)
        self.processing_queue = asyncio.Queue()
        
        # Event Bus Subscription
        self.bus = get_event_bus()
        self.bus.subscribe("voice_command", self._handle_voice_command)
        
        # 🌊 Event-Driven Logger Init
        from src.core.logger import start_log_listener
        start_log_listener()
        
        # NOTE: Warden init moved to start() - fixes core→tara coupling

        # UI Lock (Semaphore) for Turn-Taking
        # Green (Set) = Ready for Input
        # Red (Clear) = System Busy / Processing
        self.ui_lock = asyncio.Event()
        self.ui_lock.set()  # Start Green

        # Command Registry - Built from commands.py
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
        # Import command registry builder from commands module
        from .commands import build_command_registry
        self.system_commands = build_command_registry(self)
    
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
            from src.agents.nia import aprocess_input
            
            t1 = time.perf_counter()
            self.logger.debug(f"Step 2: NIA import complete ({t1-t0:.2f}s)")
            
            self._nia_process = aprocess_input
            
            t2 = time.perf_counter()
            self.logger.debug(f"Step 3: Process function assigned ({t2-t1:.2f}s)")
            
            # 🌊 LAZY LOAD: 4-Layer Memory System (ChromaDB, SQLite, NetworkX)
            try:
                from src.core.memory import get_memory_manager
                self.memory = get_memory_manager()
                
                # Register in ServiceRegistry for centralized access
                ServiceRegistry.register("memory", self.memory)
                
                # Housekeeping: vacuum SQLite databases on startup
                if hasattr(self.memory, '_vacuum_memory_db'):
                    self.memory._vacuum_memory_db()
            except Exception as mem_exc:
                self.logger.warning("Memory init failed (continuing without): %s", mem_exc)
                self.memory = None
            
            # 🔌 PLUGIN SYSTEM: Get from ServiceRegistry (v3.1 Decoupled)
            plugins = ServiceRegistry.get("plugins")
            if plugins:
                self._plugin_observer = plugins
                self.logger.debug("Plugin watcher attached from ServiceRegistry")
            else:
                self.logger.debug("Plugin watcher not registered (optional)")
            
            return True
        except ImportError as exc:
            self.logger.error("❌ Failed to import NIA: %s", exc)
            return False

    
    # NOLA is now initialized externally and registered as "voice"
    # IRIS is now initialized externally and registered as "vision"
    
    def _init_sentry(self) -> None:
        """Initialize IRIS Sentry (placeholder for legacy reflex handler)."""
        try:
            vision = ServiceRegistry.get("vision")
            if vision and hasattr(vision, "start_sentry"):
                vision.start_sentry()
                self.sentry_thread = vision
        except Exception as e:
            self.logger.warning(f"Sentry init failed: {e}")
    
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
            from src.core.config import settings
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
        
        # 🛡️ Operation Iron Cage: Warden via ServiceRegistry (v3.1 Decoupled)
        warden = ServiceRegistry.get("security")
        if warden and hasattr(warden, "start"):
            warden.start()
            self.logger.debug("Warden security service started via ServiceRegistry")
        else:
            self.logger.debug("Warden not registered (security features limited)")
        
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
        
        return True
    
    def stop(self) -> None:
        """Stop the assistant and cleanup all resources.
        
        Gracefully shuts down NOLA voice system and IRIS Sentry.
        Safe to call multiple times.
        """
        self._running = False
        
        # Stop IRIS Sentry via ServiceRegistry (v3.1 Decoupled)
        vision = ServiceRegistry.get("vision")
        if vision and hasattr(vision, "stop_sentry"):
            vision.stop_sentry()
        else:
            self.logger.debug("Vision service not available for sentry stop")
        
        # Stop NOLA via Registry
        nola = ServiceRegistry.get("voice")
        if nola:
            self.logger.info("🔇 Stopping NOLA...")
            nola.stop()
        
        # Stop Plugin Watcher via ServiceRegistry (v3.1 Decoupled)
        plugins = ServiceRegistry.get("plugins")
        if plugins and hasattr(plugins, "stop"):
            plugins.stop()
            self._plugin_observer = None
            self.logger.debug("Plugin watcher stopped via ServiceRegistry")
        
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
            
            # Store Episodes (Layer 1)
            if self.memory:
                try:
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
        from src.interface.banner import render_banner, render_hint, style

        print("\n" + render_banner() + "\n")
        
        if not await self.start():
            return
        
        # We no longer use TerminalUI's context manager strictly if we want async input
        ui_print = print
        if TerminalUI:
            try:
                _ui = TerminalUI()
                ui_print = _ui.print
            except:
                pass

        print(style("Ready for directives.", "1;92"))
        print(render_hint() + "\n")
        
        # Capture the loop for EventBus thread-safety
        self.bus.set_loop(asyncio.get_running_loop())

        # Start Keyboard Listener Task
        keyboard_task = asyncio.create_task(self._keyboard_listener())

        # Main Processing Loop
        self.logger.info("🚀 Main Loop Started (Concurrent Mode)")
        
        from src.interface.banner import style

        try:
            while self._running:
                # Ghost Protocol Watchdog - Auto-engage stealth sentry on Layer 3
                # 🌊 ASYNC OFFLOAD: Prevent blocking I/O in main loop
                is_ghost, layer = await asyncio.to_thread(self._check_ghost_state)
                if is_ghost and layer >= 3:
                     # Delegate Sentry toggle to Vision Service
                     vision = ServiceRegistry.get("vision")
                     if vision and hasattr(vision, "start_sentry"):
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
                        ui_print(f"\n{style('💬 NIA', '1;96')}: {response}\n")
                    else:
                        print(f"\n{style('💬 NIA', '1;96')}: {response}\n")
                    
                    # Speak Response (Blocking I/O wrapped in thread)
                    await asyncio.to_thread(self.speak, response)
                finally:
                    # 🌊 RIPPLE SAFE: Always release lock, even on error
                    self.ui_lock.set()
                
        except asyncio.CancelledError:
            print(style("\n👋 Goodbye!", "1;95"))
        except KeyboardInterrupt:
            print(style("\n👋 Interrupted by user", "1;93"))
        except Exception as e:
            self.logger.error(f"Loop error: {e}", exc_info=True)
        finally:
            keyboard_task.cancel()
            self.stop()

    async def _keyboard_listener(self) -> None:
        """Background task for keyboard input."""
        from src.interface.banner import style

        while self._running:
            try:
                # 🚦 TURN-TAKING: Wait for Green Light (System Ready)
                await self.ui_lock.wait()
                
                user_input = await aioconsole.ainput(f"{style('You', '1;92')}: ")
                
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
        from .commands import dispatch_reflex
        if dispatch_reflex(self, cmd):
            return True
            
        return False
