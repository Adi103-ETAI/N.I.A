#!/usr/bin/env python
"""N.I.A. - Neural Intelligence Assistant

A complete voice-enabled AI assistant combining:
- NIA: LangGraph-based supervisor with specialist routing
- NOLA: Non-blocking voice I/O (speech recognition + text-to-speech)

Usage:
    python main.py                    # Text mode (default)
    python main.py --voice            # Voice mode with wake words
    python main.py --voice --no-wake  # Voice mode without wake words
    python main.py --status           # Check system status
    python main.py --help             # Show all options

Architecture:
    ┌──────────────────────────────────────────────────────────────────┐
    │                    N.I.A. Voice Assistant                        │
    │                                                                  │
    │  ┌──────────────┐     ┌──────────────┐     ┌──────────────────┐  │
    │  │     NOLA     │────►│     NIA      │────►│       NOLA       │  │
    │  │   AsyncEar   │     │  Supervisor  │     │     AsyncTTS     │  │
    │  │  (Listen)    │     │   → IRIS     │     │     (Speak)      │  │
    │  │              │     │   → TARA     │     │                  │  │
    │  └──────────────┘     └──────────────┘     └──────────────────┘  │
    │      Voice In              Brain               Voice Out         │
    └──────────────────────────────────────────────────────────────────┘

Version: 1.0.0
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Ensure data directory exists for persistence
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)


# =============================================================================
# ASCII Art Banner
# =============================================================================

BANNER = r"""
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║    ███╗   ██╗   ██╗    █████╗                                             ║
║    ████╗  ██║   ██║   ██╔══██╗     Neural Intelligence Assistant          ║
║    ██╔██╗ ██║   ██║   ███████║     ─────────────────────────────          ║
║    ██║╚██╗██║   ██║   ██╔══██║     Voice-Enabled AI Companion             ║
║    ██║ ╚████║██╗██║██╗██║  ██║     Powered by LangGraph + NOLA            ║
║    ╚═╝  ╚═══╝╚═╝╚═╝╚═╝╚═╝  ╚═╝                                            ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""

MINI_BANNER = """
╭──────────────────────────────────────────╮
│  N.I.A. - Neural Intelligence Assistant  │
╰──────────────────────────────────────────╯
"""


# =============================================================================
# Dependency Imports (with graceful fallback)
# =============================================================================

def _check_dependencies() -> dict:
    """Check all required dependencies."""
    deps = {}
    
    # NIA dependencies
    try:
        from nia import process_input, print_status
        deps["nia"] = True
    except ImportError as e:
        deps["nia"] = False
        deps["nia_error"] = str(e)
    
    # NOLA dependencies
    try:
        from nola import NOLAManager, NOLAConfig
        deps["nola"] = True
    except ImportError as e:
        deps["nola"] = False
        deps["nola_error"] = str(e)
    
    # LangGraph
    try:
        import langgraph
        deps["langgraph"] = True
    except ImportError:
        deps["langgraph"] = False
    
    # OpenAI API key
    deps["openai_key"] = bool(os.environ.get("OPENAI_API_KEY"))
    
    # Speech dependencies
    try:
        import speech_recognition
        deps["speech_recognition"] = True
    except ImportError:
        deps["speech_recognition"] = False
    
    try:
        import pyttsx3
        deps["pyttsx3"] = True
    except ImportError:
        deps["pyttsx3"] = False
    
    return deps


def print_system_status():
    """Print detailed system status."""
    deps = _check_dependencies()
    
    print(BANNER)
    print("System Status")
    print("=" * 50)
    
    # Core components
    print("\n📦 Core Components:")
    for name in ["nia", "nola", "langgraph"]:
        status = "✓" if deps.get(name) else "✗"
        print(f"   {status} {name}")
        if not deps.get(name) and deps.get(f"{name}_error"):
            print(f"      └─ Error: {deps.get(f'{name}_error')}")
    
    # Voice components
    print("\n🎤 Voice Components:")
    for name in ["speech_recognition", "pyttsx3"]:
        status = "✓" if deps.get(name) else "✗"
        print(f"   {status} {name}")
    
    # Configuration
    print("\n🔑 Configuration:")
    
    # List of keys: (Key Name, Is Required)
    env_vars = [
        ("NVIDIA_API_KEY", True),           # Primary Provider (Required)
        ("OPENAI_API_KEY", False),          # Optional
        ("HUGGINGFACE_API_KEY", False),     # Optional
        ("OLLAMA_HOST", False),             # Optional
        ("MODEL_PROVIDER_FALLBACKS", False) # Optional
    ]

    for key, required in env_vars:
        value = os.environ.get(key)
        is_set = bool(value and value.strip())
        
        if is_set:
            # Show values for non-secrets, hide actual keys
            if key in ["OLLAMA_HOST", "MODEL_PROVIDER_FALLBACKS"]:
                print(f"   ✓ {key}: {value}")
            else:
                print(f"   ✓ {key}: Set")
        else:
            if required:
                print(f"   ✗ {key}: Not set (Required!)")
            else:
                print(f"   ○ {key}: Not set (Optional)")
    
    # Summary
    print("\n" + "=" * 50)
    
    # Check if we have NIA, NOLA, and the Primary Key (NVIDIA)
    has_primary_key = bool(os.environ.get("NVIDIA_API_KEY"))
    all_core = deps.get("nia") and deps.get("nola") and has_primary_key
    voice_ready = deps.get("speech_recognition") and deps.get("pyttsx3")
    
    if all_core and voice_ready:
        print("✅ All systems ready! Voice mode available.")
    elif all_core:
        print("⚠️  Core ready, but voice dependencies missing.")
        print("   Install: pip install speechrecognition pyttsx3 pyaudio")
    else:
        print("❌ System not ready. Check missing components above.")
        if not has_primary_key:
            print("   Set NVIDIA_API_KEY in .env file (Primary Provider)")
    
    print()


# =============================================================================
# Main Application
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
        """Initialize the assistant.
        
        Args:
            voice_mode: Enable voice input/output.
            wake_word_enabled: Require wake word for voice activation.
            wake_words: Custom wake words (default: jarvis, nia).
            thread_id: Conversation thread ID for persistence.
            debug: Enable debug logging.
        """
        self.voice_mode = voice_mode
        self.wake_word_enabled = wake_word_enabled
        self.wake_words = wake_words or ["jarvis", "nia", "hey nia"]
        self.thread_id = thread_id
        self.debug = debug
        
        # Configure logging
        log_level = logging.DEBUG if debug else logging.INFO
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )
        self.logger = logging.getLogger("NIA")
        
        # Components (lazy initialization)
        self._nia_process = None
        self._nola = None
        self._running = False
    
    def _init_nia(self) -> bool:
        """Initialize the NIA brain."""
        try:
            from nia import process_input
            self._nia_process = process_input
            self.logger.info("NIA brain initialized")
            return True
        except ImportError as exc:
            self.logger.error("Failed to import NIA: %s", exc)
            return False
    
    def _init_nola(self) -> bool:
        """Initialize NOLA voice system."""
        if not self.voice_mode:
            return True
        
        try:
            from nola import NOLAManager, NOLAConfig
            
            config = NOLAConfig(
                wake_word_enabled=self.wake_word_enabled,
                wake_words=self.wake_words,
                wake_word_timeout=30.0,
                security_enabled=True,
                pause_ear_while_speaking=True,
            )
            
            self._nola = NOLAManager(config=config)
            
            if self._nola.start():
                self.logger.info("NOLA voice system initialized")
                return True
            else:
                self.logger.error("NOLA failed to start")
                return False
                
        except ImportError as exc:
            self.logger.error("Failed to import NOLA: %s", exc)
            return False
    
    def start(self) -> bool:
        """Start the assistant.
        
        Returns:
            True if started successfully.
        """
        print(MINI_BANNER)
        
        # Initialize NIA
        if not self._init_nia():
            print("❌ Failed to initialize NIA brain.")
            return False
        
        # Initialize NOLA (if voice mode)
        if self.voice_mode:
            if not self._init_nola():
                print("⚠️  Voice mode unavailable. Continuing in text mode.")
                self.voice_mode = False
        
        self._running = True
        
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
        
        if self._nola:
            self.logger.info("Stopping NOLA...")
            self._nola.stop()
            self._nola = None
        
        self.logger.info("NIA shutdown complete")
    
    def process(self, text: str) -> str:
        """Process user input through NIA.
        
        Args:
            text: User input text.
            
        Returns:
            Assistant response.
        """
        if not text:
            return ""
        
        if not self._nia_process:
            return "NIA brain not initialized."
        
        try:
            return self._nia_process(text, thread_id=self.thread_id)
        except Exception as exc:
            self.logger.exception("NIA processing error: %s", exc)
            return f"I encountered an error: {exc}"
    
    def speak(self, text: str) -> None:
        """Speak text through NOLA.
        
        Args:
            text: Text to speak.
        """
        if self._nola and text:
            try:
                self._nola.speak(text)
            except Exception as exc:
                self.logger.debug("TTS error: %s", exc)
    
    def run(self) -> None:
        """Run the main interaction loop."""
        if not self.start():
            return
        
        try:
            while self._running:
                try:
                    # Get input (voice or text)
                    user_input = self._get_input()
                    
                    if user_input is None:
                        continue
                    
                    # Handle commands
                    if self._handle_command(user_input):
                        continue
                    
                    # Process through NIA
                    print("🤔 Thinking...")
                    response = self.process(user_input)
                    
                    # Output response
                    print(f"\n💬 NIA: {response}\n")
                    self.speak(response)
                    
                except KeyboardInterrupt:
                    print("\n\n👋 Goodbye!")
                    break
                except EOFError:
                    continue
                    
        finally:
            self.stop()
    
    def _get_input(self) -> Optional[str]:
        """Get input from voice or keyboard.
        
        Returns:
            User input text, or None if no input.
        """
        if self.voice_mode and self._nola:
            # Try voice input first
            result = self._nola.get_input(timeout=0.3)
            
            if result:
                if result.is_blocked:
                    print(f"🔒 Security blocked: {result.blocked_reason}")
                    return None
                
                text = result.text.strip()
                if text:
                    print(f"\n🎤 You: {text}")
                    return text
            
            # Check for keyboard input (non-blocking would be ideal)
            # For simplicity, we'll alternate between voice and allowing text input
            try:
                # Use select on stdin if available
                import select
                if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                    line = sys.stdin.readline().strip()
                    if line:
                        return line
            except Exception:
                pass
            
            return None
        else:
            # Text mode
            try:
                return input("You: ").strip()
            except EOFError:
                return None
    
    def _handle_command(self, text: str) -> bool:
        """Handle built-in commands.
        
        Args:
            text: User input.
            
        Returns:
            True if command was handled (don't process further).
        """
        cmd = text.lower().strip()
        
        if cmd in ("exit", "quit", "bye", "goodbye"):
            if self.voice_mode:
                self.speak("Goodbye! Have a great day!")
            print("👋 Goodbye!")
            self._running = False
            return True
        
        if cmd == "help":
            print("\n💬 NIA: Here is the list of available commands:\n")
            self._print_help()
            return True
        
        if cmd == "status":
            print("\n💬 NIA: Checking system status...\n")
            print_system_status()
            return True
        
        if cmd == "voice on":
            if not self.voice_mode:
                self.voice_mode = True
                if self._init_nola():
                    print("🎤 Voice mode enabled")
                else:
                    self.voice_mode = False
            else:
                print("🎤 Voice mode already active")
            return True
        
        if cmd == "voice off":
            if self.voice_mode and self._nola:
                self._nola.stop()
                self._nola = None
                self.voice_mode = False
                print("🔇 Voice mode disabled")
            return True
        
        if cmd == "clear":
            os.system('cls' if os.name == 'nt' else 'clear')
            print(MINI_BANNER)
            return True
        
        if cmd.startswith("wake "):
            # Set custom wake word
            new_wake = cmd[5:].strip()
            if new_wake:
                self.wake_words = [new_wake]
                if self._nola:
                    self._nola.set_wake_words(self.wake_words)
                print(f"🎤 Wake word set to: {new_wake}")
            return True
        
        if cmd == "history":
            # Show conversation history
            print("\n💬 NIA: Here is your conversation history:\n")
            try:
                from nia import get_conversation_history
                history = get_conversation_history(self.thread_id)
                if history:
                    print(f"📜 Conversation history ({len(history)} messages):")
                    for msg in history[-10:]:  # Show last 10
                        role = getattr(msg, 'type', 'unknown')
                        content = getattr(msg, 'content', str(msg))[:100]
                        print(f"   [{role}]: {content}..." if len(content) == 100 else f"   [{role}]: {content}")
                else:
                    print("📜 No conversation history yet.")
            except Exception as exc:
                print(f"❌ Could not retrieve history: {exc}")
            return True
        
        if cmd == "clear history":
            # Clear conversation history
            try:
                from nia import clear_conversation
                if clear_conversation(self.thread_id):
                    print("🗑️ Conversation history cleared.")
                else:
                    print("⚠️ Could not clear history.")
            except Exception as exc:
                print(f"❌ Error clearing history: {exc}")
            return True
        
        return False
    
    def _print_help(self) -> None:
        """Print help information with correct alignment."""
        # Width configuration (inner width)
        w = 58  
        
        # Helper to format a line with borders
        def line(text="", align="<"):
            # Truncate if too long to prevent breaking
            if len(text) > w: text = text[:w-3] + "..."
            return f"│ {text:{align}{w}} │"

        help_text = [
            f"╭{'─' * (w+2)}╮",
            line("NIA Commands", "^"),
            f"├{'─' * (w+2)}┤",
            line(),
            line("General:"),
            line("  help             - Show this help message"),
            line("  status           - Show system status"),
            line("  clear            - Clear the screen"),
            line("  exit/quit        - Exit the assistant"),
            line(),
            line("Memory:"),
            line("  history          - Show conversation history"),
            line("  clear history    - Clear conversation history"),
            line(f"  Thread ID:       {self.thread_id}"), 
            line(),
            line("Voice:"),
            line("  voice on         - Enable voice mode"),
            line("  voice off        - Disable voice mode"),
            line("  wake <word>      - Set custom wake word"),
            line(),
            line("Specialists:"),
            line("  IRIS - Vision/Image tasks (say 'analyze image')"),
            line("  TARA - Logic/Math tasks (say 'solve equation')"),
            line(),
            line("Tips:"),
            line("  • Conversations are saved across runs"),
            line("  • In voice mode, say wake word first"),
            line("  • Dangerous commands are blocked for safety"),
            line(),
            f"╰{'─' * (w+2)}╯"
        ]
        
        print("\n".join(help_text))


# =============================================================================
# CLI Entry Point
# =============================================================================

def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="N.I.A. - Neural Intelligence Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                     Text mode (keyboard input)
  python main.py --voice             Voice mode with wake words
  python main.py --voice --no-wake   Voice mode (always listening)
  python main.py --status            Check system dependencies
  python main.py --debug             Enable debug logging
        """
    )
    
    parser.add_argument(
        "--voice", "-v",
        action="store_true",
        help="Enable voice input/output"
    )
    parser.add_argument(
        "--no-wake",
        action="store_true",
        help="Disable wake word requirement (always listening)"
    )
    parser.add_argument(
        "--wake-words", "-w",
        type=str,
        default="jarvis,nia,hey nia",
        help="Comma-separated wake words (default: jarvis,nia,hey nia)"
    )
    parser.add_argument(
        "--status", "-s",
        action="store_true",
        help="Print system status and exit"
    )
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--thread-id", "-t",
        type=str,
        default="root",
        help="Conversation thread ID for persistence (default: root)"
    )
    parser.add_argument(
        "--version",
        action="version",
        version="N.I.A. v1.0.0"
    )
    
    args = parser.parse_args()
    
    # Status check mode
    if args.status:
        print_system_status()
        return 0
    
    # Parse wake words
    wake_words = [w.strip() for w in args.wake_words.split(",") if w.strip()]
    
    # Create and run assistant
    assistant = NIAAssistant(
        voice_mode=args.voice,
        wake_word_enabled=not args.no_wake,
        wake_words=wake_words,
        thread_id=args.thread_id,
        debug=args.debug,
    )
    
    try:
        assistant.run()
        return 0
    except Exception as exc:
        logging.exception("Unexpected error: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
