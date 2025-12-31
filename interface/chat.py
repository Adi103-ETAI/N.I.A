"""NIA CLI - Alternative interface using the NIA brain.

This script provides a CLI interface to the NIA assistant, with optional
voice mode powered by NOLA (Non-blocking Operator for Language & Audio).

This is the legacy CLI interface. For the main application, use main.py.

Voice I/O is powered by the NOLA package, which provides non-blocking
async speech recognition and synthesis.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Import NIA brain (the new architecture)
try:
    from nia import process_input, get_conversation_history, clear_conversation
    _HAS_NIA = True
except ImportError:
    _HAS_NIA = False
    process_input = None  # type: ignore
    get_conversation_history = None  # type: ignore
    clear_conversation = None  # type: ignore

# Import NOLA for voice I/O
try:
    from nola import NOLAManager, NOLAConfig
    _HAS_NOLA = True
except ImportError:
    _HAS_NOLA = False
    NOLAManager = None  # type: ignore
    NOLAConfig = None  # type: ignore


def main(argv: Optional[list] = None) -> None:
    """Entry point for the NIA CLI."""
    parser = argparse.ArgumentParser(
        description="NIA CLI - Neural Intelligence Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Voice Mode:
  When --voice is enabled, NIA uses NOLA for speech recognition and synthesis.
  Wake words: 'jarvis', 'nia', 'hey nia'
  
  You can also type commands while in voice mode.
  Prefix with 'type ' or 't ' to force typed input.

Examples:
  python -m interface.chat                    # Text-only mode
  python -m interface.chat --voice            # Voice mode with wake words
  python -m interface.chat --voice --no-wake  # Voice mode without wake words
  python -m interface.chat --thread-id user1  # Use specific conversation thread
        """
    )
    parser.add_argument(
        '--voice', action='store_true',
        help='Enable voice mode (requires microphone and speakers)'
    )
    parser.add_argument(
        '--no-wake', action='store_true',
        help='Disable wake word requirement in voice mode'
    )
    parser.add_argument(
        '--wake-words', type=str, default='jarvis,nia,hey nia',
        help='Comma-separated wake words (default: jarvis,nia,hey nia)'
    )
    parser.add_argument(
        '--thread-id', '-t', type=str, default='cli',
        help='Conversation thread ID for persistence (default: cli)'
    )
    parser.add_argument(
        '--debug', action='store_true',
        help='Enable debug logging'
    )
    args = parser.parse_args(args=argv)

    # Load .env at runtime to avoid import-time side effects
    load_dotenv()

    # Ensure data directory exists
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)

    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    logger = logging.getLogger(__name__)

    # Check if NIA is available
    if not _HAS_NIA:
        logger.error("NIA package not available. Install dependencies first.")
        print("[ERROR] NIA brain not available. Please check installation.")
        return

    # =========================================================================
    # Voice Mode Setup (using NOLA)
    # =========================================================================
    voice_mode = False
    nola: Optional[NOLAManager] = None
    thread_id = args.thread_id

    if args.voice:
        if not _HAS_NOLA:
            logger.error("NOLA package not available. Install with: pip install speechrecognition pyttsx3 pyaudio")
            print("[WARN] Voice mode unavailable. Starting in text-only mode.")
        else:
            # Parse wake words
            wake_words = [w.strip() for w in args.wake_words.split(',') if w.strip()]
            
            # Configure NOLA
            nola_config = NOLAConfig(
                wake_word_enabled=not args.no_wake,
                wake_words=wake_words,
                wake_word_timeout=30.0,
                security_enabled=True,  # Enable command sanitization
                pause_ear_while_speaking=True,  # Echo cancellation
            )
            
            nola = NOLAManager(config=nola_config)
            
            if nola.start():
                voice_mode = True
                logger.info("NOLA voice system started")
                if not args.no_wake:
                    print(f"[VOICE] Enabled. Wake words: {', '.join(wake_words)}")
                else:
                    print("[VOICE] Enabled (no wake word required)")
            else:
                logger.error("Failed to start NOLA voice system")
                print("[WARN] Voice mode failed to start. Starting in text-only mode.")
                nola = None

    # =========================================================================
    # Main Chat Loop
    # =========================================================================
    print("\n" + "=" * 50)
    print("  NIA - Neural Intelligence Assistant")
    print(f"  Thread: {thread_id}")
    print("  Type 'help' for commands, 'exit' to quit")
    print("=" * 50 + "\n")

    try:
        while True:
            user_input = None
            
            try:
                if voice_mode and nola:
                    # =========================================================
                    # Voice Mode: Use NOLA for input
                    # =========================================================
                    # Try to get voice input with a short timeout
                    result = nola.get_input(timeout=0.3)
                    
                    if result and result.text:
                        # Got voice input
                        user_input = result.text
                        print(f"[VOICE] {user_input}")
                        
                        # Check for security blocks
                        if result.is_blocked:
                            print(f"[BLOCKED] Security: {result.blocked_reason}")
                            continue
                    else:
                        # No voice input, check for keyboard input
                        try:
                            if sys.stdin in _select_stdin():
                                typed = input()
                                if typed.strip():
                                    # Handle 'type ' or 't ' prefix
                                    if typed.strip().startswith('type '):
                                        user_input = typed.strip()[5:]
                                    elif typed.strip().startswith('t '):
                                        user_input = typed.strip()[2:]
                                    else:
                                        user_input = typed.strip()
                        except Exception:
                            pass
                        
                        if not user_input:
                            continue
                else:
                    # =========================================================
                    # Text Mode: Standard input
                    # =========================================================
                    user_input = input('> ')
                    
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except EOFError:
                print("\n(EOF received - ignoring)")
                continue

            if not user_input:
                continue

            # Process commands
            cmd = user_input.strip().lower()
            
            if cmd in ("exit", "quit", "bye"):
                if voice_mode and nola:
                    nola.speak("Goodbye!")
                print("Goodbye!")
                break

            if cmd == "help":
                _print_help(voice_mode, thread_id)
                continue

            if cmd == "history":
                # Show conversation history
                if get_conversation_history:
                    try:
                        history = get_conversation_history(thread_id)
                        if history:
                            print(f"\n[HISTORY] Thread '{thread_id}' ({len(history)} messages):")
                            for msg in history[-10:]:  # Show last 10
                                role = getattr(msg, 'type', 'unknown')
                                content = getattr(msg, 'content', str(msg))[:80]
                                print(f"  [{role}]: {content}...")
                        else:
                            print("[HISTORY] No conversation history yet.")
                    except Exception as exc:
                        print(f"[ERROR] Could not retrieve history: {exc}")
                else:
                    print("[WARN] History function not available")
                continue

            if cmd == "clear":
                # Clear conversation history
                if clear_conversation:
                    try:
                        if clear_conversation(thread_id):
                            print(f"[OK] Cleared history for thread '{thread_id}'")
                        else:
                            print("[WARN] Could not clear history")
                    except Exception as exc:
                        print(f"[ERROR] {exc}")
                else:
                    print("[WARN] Clear function not available")
                continue

            if cmd == "voice on" and _HAS_NOLA and not voice_mode:
                # Enable voice mode mid-session
                nola_config = NOLAConfig(wake_word_enabled=False)
                nola = NOLAManager(config=nola_config)
                if nola.start():
                    voice_mode = True
                    print("[VOICE] Enabled")
                continue

            if cmd == "voice off" and voice_mode and nola:
                # Disable voice mode mid-session
                nola.stop()
                nola = None
                voice_mode = False
                print("[VOICE] Disabled")
                continue

            # =================================================================
            # Process with NIA Brain
            # =================================================================
            print("[THINKING]...")
            
            try:
                response = process_input(user_input, thread_id=thread_id)
                print(f"\nNIA: {response}\n")
            except Exception as exc:
                logger.exception("NIA processing error: %s", exc)
                print(f"\n[ERROR] {exc}\n")
                continue

            # Speak the response in voice mode
            if voice_mode and nola:
                try:
                    nola.speak(response)
                except Exception as exc:
                    logger.debug("TTS failed: %s", exc)

    finally:
        # Cleanup
        if nola:
            logger.info("Stopping NOLA...")
            nola.stop()


def _select_stdin():
    """Check if stdin has data available (cross-platform best-effort)."""
    try:
        import select
        readable, _, _ = select.select([sys.stdin], [], [], 0)
        return readable
    except Exception:
        return []


def _print_help(voice_mode: bool, thread_id: str) -> None:
    """Print help information."""
    print(f"""
NIA CLI Help
============
Thread ID: {thread_id}

Commands:
  exit, quit, bye  - End the session
  help             - Show this help
  history          - Show conversation history
  clear            - Clear conversation history
""")
    if voice_mode:
        print("""Voice Mode Commands:
  voice off        - Disable voice mode
  type <text>      - Force typed input
  t <text>         - Short form of 'type'
""")
    else:
        print("""  voice on         - Enable voice mode (if NOLA available)
""")


if __name__ == "__main__":
    main()
