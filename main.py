#!/usr/bin/env python3
"""N.I.A. - Neural Intelligence Assistant.

Minimal CLI entry point. The actual logic is in core/engine.py.

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
import sys

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


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
    parser.add_argument("--version", action="version", version="N.I.A. v2.0.0")
    
    args = parser.parse_args()
    
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
    except Exception as exc:
        logging.exception("Unexpected error: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
