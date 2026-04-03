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
import atexit
import logging
import os
import sys

import asyncio

# Suppress pygame welcome message before any imports
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"
# Reduce ONNX Runtime non-actionable warnings in normal CLI sessions.
os.environ.setdefault("ORT_LOG_SEVERITY_LEVEL", "3")

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Initialize telemetry if endpoint configured
otel_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
if otel_endpoint:
    try:
        from src.core.telemetry.tracer import init_tracer
        init_tracer(service_name="nia-core", endpoint=otel_endpoint)
        print(f"✅ Telemetry enabled: {otel_endpoint}")
    except ImportError:
        print("⚠️ OpenTelemetry not installed, tracing disabled")
    except Exception as e:
        print(f"⚠️ Tracer init failed: {e}")


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
    parser.add_argument("--version", action="version", version="N.I.A. v4.0.0")
    
    args = parser.parse_args()
    
    # Initialize global logging BEFORE any other imports
    # This ensures all modules pick up the correct debug level
    from src.core.logger import init_logging, setup_logger, set_debug_mode
    init_logging(debug=args.debug)
    
    if args.debug:
        set_debug_mode(True)
        print(">> [SYSTEM] Debug Mode Enabled: Showing all internal logs.")
    
    # Get logger for this module
    logger = setup_logger("MAIN")
    
    # Status check mode
    if args.status:
        from src.core.health import print_system_status
        print_system_status()
        return 0
    
    # Log startup configuration
    logger.debug("N.I.A. v4.0.0 starting (Async Native)...")
    logger.debug(f"Mode: {'Voice' if args.voice else 'Text'} | Debug: {args.debug} | Thread: {args.thread_id}")
    if args.voice:
        logger.debug(f"Wake words: {args.wake_words} | Wake required: {not args.no_wake}")
    
    # Register cleanup
    from src.infrastructure.container_engine.manager import DockerEngine
    atexit.register(DockerEngine().cleanup)

    # Import and run engine
    # Import components for Dependency Injection
    from src.core.di import ServiceRegistry
    from src.core.engine import NIAAssistant
    from src.agents.nola.manager import get_nola_manager, NOLAConfig
    from src.agents.iris.agent import IrisAgent
    
    wake_words = [w.strip() for w in args.wake_words.split(",") if w.strip()]
    
    # 0. Event Bus (The Spine)
    from src.core.bus import get_event_bus
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
            ServiceRegistry.register("vision", iris, description="Screen analysis and OCR")
            logger.info("👁️ Registered Service: 'vision' -> IrisAgent")
        else:
             logger.warning("Vision Service not ready (check API key)")
    except Exception as e:
        logger.warning(f"Failed to load Vision Service: {e}")

    # 3. Security Service (Warden) - v3.1 Decoupled
    try:
        from src.agents.tara.security import start_warden_service
        
        # Create a wrapper that matches the ServiceRegistry pattern
        class WardenServiceWrapper:
            """Wrapper to adapt Warden to ServiceRegistry lifecycle."""
            def __init__(self):
                self._started = False
            
            def start(self):
                if not self._started:
                    start_warden_service()
                    self._started = True
            
            def stop(self):
                # Warden doesn't need explicit stop
                self._started = False
        
        warden = WardenServiceWrapper()
        ServiceRegistry.register("security", warden, description="Security escalation handler", priority=10)
        logger.info("🛡️ Registered Service: 'security' -> WardenService")
    except ImportError as e:
        logger.debug(f"Warden not available (optional): {e}")

    # 4. Plugin System (Hot-Reload Watcher) - v3.1 Decoupled
    try:
        from src.extensions.loader import start_plugin_watcher, stop_plugin_watcher
        
        class PluginWatcherWrapper:
            """Wrapper to adapt Plugin Watcher to ServiceRegistry lifecycle."""
            def __init__(self):
                self._observer = None
            
            def start(self):
                if not self._observer:
                    self._observer = start_plugin_watcher()
            
            def stop(self):
                if self._observer:
                    stop_plugin_watcher(self._observer)
                    self._observer = None
        
        plugins = PluginWatcherWrapper()
        ServiceRegistry.register("plugins", plugins, description="Plugin hot-reload watcher", priority=50)
        logger.info("🔌 Registered Service: 'plugins' -> PluginWatcher")
    except ImportError as e:
        logger.debug(f"Plugin watcher not available (optional): {e}")

    # --- END WIRING ---
    
    # Log service status
    logger.debug(f"Services registered: {ServiceRegistry.list_services()}")
    
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
