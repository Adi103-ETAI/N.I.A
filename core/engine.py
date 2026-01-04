"""N.I.A. Core Engine - Central Nervous System.

Contains the NIAAssistant class that orchestrates all components.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Optional, Tuple

# Import banner
try:
    from interface.banner import MINI_BANNER
except ImportError:
    MINI_BANNER = "N.I.A. - Neural Intelligence Assistant"

# Import Terminal UI
try:
    from interface.chat import TerminalUI
except ImportError:
    TerminalUI = None


# =============================================================================
# NIAAssistant Class
# =============================================================================

class NIAAssistant:
    """Main NIA Voice Assistant application.
    
    Combines NIA (brain) and NOLA (voice I/O) into a unified interface.
    """
    
    def __init__(
        self,
        voice_mode: bool = False,
        wake_word_enabled: bool = True,
        wake_words: Optional[list] = None,
        thread_id: str = "root",
        debug: bool = False,
    ) -> None:
        """Initialize the assistant."""
        self.voice_mode = voice_mode
        self.wake_word_enabled = wake_word_enabled
        self.wake_words = wake_words or ["jarvis", "nia", "hey nia"]
        self.thread_id = thread_id
        self.debug = debug
        
        # Get logger (main.py configures logging)
        self.logger = logging.getLogger("NIA")
        if debug:
            self.logger.setLevel(logging.DEBUG)
        
        # Components (lazy initialization)
        self._nia_process = None
        self._nola = None
        self.iris = None
        self.sentry_thread = None  # Track sentry thread
        self._running = False
    
    def _init_nia(self) -> bool:
        """Initialize the NIA brain."""
        try:
            from nia import process_input
            self._nia_process = process_input
            self.logger.info("🧠 NIA brain initialized")
            return True
        except ImportError as exc:
            self.logger.error("❌ Failed to import NIA: %s", exc)
            return False
    
    def _init_nola(self) -> bool:
        """Initialize NOLA voice system via singleton."""
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
    
    def _init_iris(self) -> bool:
        """Initialize IRIS vision agent."""
        try:
            from iris.agent import IrisAgent
            self.iris = IrisAgent()
            self.logger.info("👁️ IRIS vision agent initialized")
            return True
        except ImportError as exc:
            self.logger.debug("IRIS not available: %s", exc)
            return False
    
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
            
        except (json.JSONDecodeError, IOError, KeyError, TypeError):
            # File missing, empty, or corrupted - default to normal mode
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
        """Start the assistant."""
        print(MINI_BANNER)
        print("🚀 Initializing N.I.A. Core...")
        
        # Initialize NIA
        if not self._init_nia():
            print("❌ Failed to initialize NIA brain.")
            return False
        
        # Initialize IRIS
        self._init_iris()
        
        # Initialize NOLA (if voice mode)
        if self.voice_mode:
            if not self._init_nola():
                print("⚠️  Voice mode unavailable. Continuing in text mode.")
                self.voice_mode = False
        
        self._running = True
        
        # Sentry now manual-start only (use 'sentry on')
        # self._init_sentry()
        print("👁️ 💤 Sentry: Standby (Use 'sentry on' to activate)")
        
        # Print mode info
        if self.voice_mode:
            if self.wake_word_enabled:
                print(f"🎤 Voice mode active. Wake words: {', '.join(self.wake_words)}")
            else:
                print("🎤 Voice mode active (always listening)")
        else:
            print("⌨️  Text mode active")
        
        print("\nType 'help' for commands, 'exit' to quit.\n")
        
        return True
    
    def stop(self) -> None:
        """Stop the assistant gracefully."""
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
        """Process user input through NIA."""
        if not text:
            return ""
        
        # Handle wake-up-only signal (single space from Vosk)
        if text.strip() == "":
            print("🎤 Wake Word Detected. Listening...")
            self.speak("Yes, Director?")
            return "Listening..."
        
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
        
        try:
            return self._nia_process(text, thread_id=self.thread_id)
        except Exception as exc:
            self.logger.exception("❌ NIA processing error: %s", exc)
            return f"I encountered an error: {exc}"
    
    def _handle_fast_path(self, text: str) -> Optional[str]:
        """Handle simple utility queries locally (no LLM needed)."""
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
        """Get ordinal suffix for day."""
        if 11 <= day <= 13:
            return "th"
        return {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    
    def speak(self, text: str) -> None:
        """Speak text through NOLA.
        
        Respects Ghost Protocol - suppresses audio when ghost mode is active.
        """
        # Check ghost mode before speaking
        is_ghost, layer = self._check_ghost_state()
        if is_ghost:
            print(f"🤫 [Ghost Mode: Audio Suppressed] NIA: {text}")
            return
        
        if self._nola and text:
            try:
                self._nola.speak(text)
            except Exception as exc:
                self.logger.debug("❌ TTS error: %s", exc)
    
    def run(self) -> None:
        """Main loop (synchronous blocking)."""
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
                def get_input(self, prompt):
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
                        print("\n👁️ 🔒 Ghost Layer 3: Auto-Engaging Sentry (Stealth Mode)")
                        self._init_sentry(headless=True)
                    
                    user_input = ui.get_input("You: ")
                    
                    if not user_input or not user_input.strip():
                        continue
                    
                    user_input = user_input.strip()
                    
                    # Handle commands locally
                    if self._handle_command(user_input):
                        continue
                    
                    # Process through NIA brain
                    ui.print("🧠 Processing...")
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
        
        Reflex commands bypass the brain for zero-latency response.
        """
        cmd = text.lower().strip()
        
        # =================================================================
        # FUZZY KEYWORD MATCHING (catches natural variations)
        # =================================================================
        
        # 🎤 MIC OFF: "turn off mic", "disable microphone", "kill the mic"
        mic_words = ["mic", "microphone"]
        off_words = ["off", "mute", "stop", "kill", "disable", "pause", "silence"]
        on_words = ["on", "unmute", "start", "enable", "resume", "activate"]
        
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
        # VOCABULARY DEFINITIONS (Synonyms for each intent)
        # =================================================================
        
        # 👁️ IRIS (Sentry/Vision Control)
        IRIS_ON = ["sentry on", "activate sentry", "enable sentry", "guard mode",
                   "watch screen", "eyes on", "start sentry", "start watching"]
        IRIS_OFF = ["sentry off", "disable sentry", "stop sentry", "standby",
                    "eyes off", "stop watching", "sentry standby"]
        
        # 🔊 TARA (Speaker Mute - Zero Latency Reflex)
        # Note: "mic" is excluded to prevent speaker commands on mic phrases
        SPEAKER_MUTE = ["mute speakers", "mute system", "mute audio", "kill sound", 
                        "silence speakers", "speakers off", "sound off", "mute volume"]
        SPEAKER_UNMUTE = ["unmute speakers", "unmute system", "sound on", "restore audio", 
                          "speakers on", "audio on", "turn on speakers", "enable sound",
                          "unmute volume", "unmute audio"]
        
        # 🔇 TTS (Stop Speaking)
        TTS_STOP = ["stop talking", "shh", "quiet", "shut up", "be quiet", "hush"]
        
        # ⚙️ SYSTEM (Maintenance)
        SYS_STATUS = ["status", "report", "system check", "diagnostics", "health",
                      "stats", "specs", "performance", "usage", "sys stats", "system stats"]
        SYS_CLEAR = ["clear", "cls", "clean screen"]
        SYS_EXIT = ["exit", "quit", "bye", "goodbye", "terminate", "close"]
        SYS_HELP = ["help", "commands", "what can you do"]
        
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
                from tara.units.system_control import mute_volume
                result = mute_volume(mute=False)
                print(result)  # Tool returns "🔊 System Unmuted"
            except Exception as e:
                print(f"⚠️  Audio control error: {e}")
            return True
        
        # MUTE SPEAKERS (exclude "mic" to prevent false matches)
        if matches(SPEAKER_MUTE) and "mic" not in cmd and "microphone" not in cmd:
            try:
                from tara.units.system_control import mute_volume
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
            os.system('cls' if os.name == 'nt' else 'clear')
            print(MINI_BANNER)
            return True
        
        # =================================================================
        # 6. 📜 HISTORY MANAGEMENT
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
        """Print help information."""
        help_text = """
╭────────────────────────────────────────────────────────────╮
│                      NIA Commands                          │
├────────────────────────────────────────────────────────────┤
│  General:                                                  │
│    help           - Show this help                         │
│    status         - Show system status                     │
│    clear          - Clear the screen                       │
│    exit/quit      - Exit the assistant                     │
│                                                            │
│  Voice:                                                    │
│    voice on       - Enable voice mode                      │
│    voice off      - Mute microphone                        │
│    sentry on/off  - Toggle vision monitoring               │
│                                                            │
│  Memory:                                                   │
│    history        - Show conversation history              │
│    clear history  - Clear conversation history             │
╰────────────────────────────────────────────────────────────╯
"""
        print(help_text)
    
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
