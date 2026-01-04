#!/usr/bin/env python3
"""N.I.A. - Neural Intelligence Assistant.

Clean CLI entry point using standard Python logging.

Usage:
    python main.py                     Text mode (keyboard input)
    python main.py --voice             Voice mode with wake words
    python main.py --voice --no-wake   Voice mode (always listening)
    python main.py --status            Check system dependencies
    python main.py --debug             Enable debug logging
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time

# Suppress pygame welcome message before any imports
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# =============================================================================
# Logging Configuration (Single Source of Truth)
# =============================================================================

def setup_logging(debug: bool = False) -> None:
    """Configure standard Python logging."""
    log_level = logging.DEBUG if debug else logging.INFO
    
    logging.basicConfig(
        level=log_level,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    
    # Silence noisy third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("vosk").setLevel(logging.ERROR)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)


# =============================================================================
# UI Helper Functions
# =============================================================================

def print_header() -> None:
    """Print the ASCII logo header."""
    header = """
╭──────────────────────────────────────────╮
│  N.I.A. - Neural Intelligence Assistant  │
╰──────────────────────────────────────────╯
"""
    print(header)


def print_user(text: str) -> None:
    """Print user input."""
    print(f"\n👤 You: {text}")


def print_nia(response: str) -> None:
    """Print NIA response."""
    print(f"🤖 NIA: {response}\n")


def print_system(message: str) -> None:
    """Print system message."""
    print(f">> {message}")


def print_mic_on() -> None:
    """Print microphone active message."""
    print("🎙️ Microphone Active")


def print_mic_off() -> None:
    """Print microphone paused message."""
    print("🔇 Microphone Paused")


# =============================================================================
# Main Entry Point
# =============================================================================

def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="N.I.A. - Neural Intelligence Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--voice", "-v", action="store_true", help="Enable voice mode")
    parser.add_argument("--no-wake", action="store_true", help="Disable wake word requirement")
    parser.add_argument("--wake-words", "-w", type=str, default="jarvis,nia,hey nia", help="Comma-separated wake words")
    parser.add_argument("--status", "-s", action="store_true", help="Print system status and exit")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug logging")
    parser.add_argument("--thread-id", "-t", type=str, default="root", help="Conversation thread ID")
    parser.add_argument("--version", action="version", version="N.I.A. v2.1.0")
    
    args = parser.parse_args()
    
    # Setup logging BEFORE any other imports
    setup_logging(debug=args.debug)
    
    # Status check mode
    if args.status:
        from core.health import print_system_status
        print_system_status()
        return 0
    
    # Import and run engine
    from core.engine import NIAAssistant
    
    wake_words = [w.strip() for w in args.wake_words.split(",") if w.strip()]
    
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
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
        return 0
    except Exception as exc:
        logging.exception("Unexpected error: %s", exc)
        return 1


# =============================================================================
# Exports (for use by other modules)
# =============================================================================

__all__ = [
    "print_header",
    "print_user",
    "print_nia",
    "print_system",
    "print_mic_on",
    "print_mic_off",
]


if __name__ == "__main__":
    sys.exit(main())
