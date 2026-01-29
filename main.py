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

import asyncio

# Suppress pygame welcome message before any imports
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


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

async def main() -> int:
    """Main entry point (Async)."""
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
    parser.add_argument("--version", action="version", version="N.I.A. v3.0.0")
    
    args = parser.parse_args()
    
    # Initialize global logging BEFORE any other imports
    # This ensures all modules pick up the correct debug level
    from core.logger import init_logging, setup_logger
    init_logging(debug=args.debug)
    
    # Get logger for this module
    logger = setup_logger("MAIN")
    
    # Status check mode
    if args.status:
        from core.health import print_system_status
        print_system_status()
        return 0
    
    # Log startup configuration
    logger.debug("N.I.A. v3.0.0 starting (Async Native)...")
    logger.debug(f"Mode: {'Voice' if args.voice else 'Text'} | Debug: {args.debug} | Thread: {args.thread_id}")
    if args.voice:
        logger.debug(f"Wake words: {args.wake_words} | Wake required: {not args.no_wake}")
    
    # Import and run engine
    # Import components for Dependency Injection
    from core.services import ServiceRegistry
    from core.engine import NIAAssistant
    from nola.manager import get_nola_manager, NOLAConfig
    from iris.agent import IrisAgent
    
    wake_words = [w.strip() for w in args.wake_words.split(",") if w.strip()]
    
    # 0. Event Bus (The Spine)
    from core.event_bus import get_event_bus
    ServiceRegistry.register("events", get_event_bus())
    
    # --- SERVICE REGISTRY WIRING ---
    
    # 1. Voice Service (NOLA)
    if args.voice:
        try:
            nola_config = NOLAConfig(
                wake_word_enabled=not args.no_wake,
                wake_words=wake_words,
                wake_word_timeout=30.0,
                security_enabled=True,
                pause_ear_while_speaking=True,
            )
            # Initialize singleton
            nola = get_nola_manager(config=nola_config)
            
            # Start NOLA hardware
            if nola.start():
                ServiceRegistry.register("voice", nola)
                logger.info("🎤 Registered Service: 'voice' -> NOLAManager")
            else:
                logger.error("❌ Failed to start NOLA voice service")
        except ImportError as e:
            logger.error(f"❌ Failed to load Voice Service: {e}")

    # 2. Vision Service (IRIS)
    try:
        # Initialize IrisAgent
        iris = IrisAgent()
        if iris.is_ready:
            ServiceRegistry.register("vision", iris)
            logger.info("👁️ Registered Service: 'vision' -> IrisAgent")
        else:
             logger.warning("Vision Service not ready (check API key)")
    except Exception as e:
        logger.warning(f"Failed to load Vision Service: {e}")

    # --- END WIRING ---
    
    assistant = NIAAssistant(
        voice_mode=args.voice,
        wake_word_enabled=not args.no_wake,
        wake_words=wake_words,
        thread_id=args.thread_id,
        debug=args.debug,
    )
    
    try:
        logger.debug("Entering async main loop")
        # Run the async engine
        await assistant.run()
        logger.debug("N.I.A. shutdown complete")
        return 0
    except asyncio.CancelledError:
        logger.info("Async task cancelled")
        return 0
    except KeyboardInterrupt:
        logger.info("Interrupted by user (Ctrl+C)")
        print("\n👋 Interrupted by user")
        return 0
    except Exception as exc:
        logger.error(f"Unexpected error: {exc}", exc_info=True)
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
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Fatal error: {e}")
