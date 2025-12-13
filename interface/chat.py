"""Minimal CLI to demo the CognitiveLoop.

This script demonstrates how to wire the core components together with
the default in-memory/tool/model stubs. It is intentionally simple so
developers can replace pieces with real implementations as they build.
"""
import logging
from core.brain import CognitiveLoop
from core.memory import InMemoryMemory
from core.tool_manager import ToolManager
from core.tools import register_dev_tools
from typing import Optional
from core.voice_manager import BackgroundListener, normalize_listen_result
import os
from pathlib import Path
from models.model_manager import ModelManager
from persona.profile import build_persona_config
import argparse
import asyncio
import threading
import queue
import time
import sys
try:
    import msvcrt
    _HAS_MSVCRT = True
except Exception:
    _HAS_MSVCRT = False
from dotenv import load_dotenv


def _listen_result_to_text(res: object) -> Optional[str]:
    """Normalize listen tool results into a plain text string or None.

    Accepts strings or dict-like responses from plugins and returns the
    recognized text, or None when no usable text is present.
    """
    if res is None:
        return None
    if isinstance(res, str):
        return res
    if isinstance(res, dict):
        # Legacy flags that indicate a failed listen
        if 'ok' in res and res.get('ok') is False:
            return None
        if 'success' in res and res.get('success') is False:
            return None

        for key in ('text', 'output', 'transcript', 'message'):
            if key in res and res.get(key):
                return res.get(key)

    return None


def main(argv: Optional[list] = None) -> None:
    """Entry point for the simple CLI demo."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--voice', action='store_true', help='Enable voice mode')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    args = parser.parse_args(args=argv)

    # Load .env at runtime to avoid import-time side effects
    load_dotenv()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    logger = logging.getLogger(__name__)

    memory = InMemoryMemory()
    tools = ToolManager()

    # Optional dev tool registration — non-fatal if it fails
    try:
        register_dev_tools(tools)
    except Exception as exc:
        logger.debug("register_dev_tools failed: %s", exc)

    # Plugin directory and initial load
    PLUGIN_DIR = Path(__file__).resolve().parent.parent / "plugins"
    try:
        tools.reload_plugins(PLUGIN_DIR)
    except Exception as exc:
        logger.debug("initial plugin load failed: %s", exc)

    persona_config = build_persona_config()
    model_config = {
        "fallback_providers": ["openai"],
        "provider_models": {
            "nvidia": "meta/llama-4-maverick-17b-128e-instruct",
            "openai": "gpt-4o",
        },
        **persona_config,
    }
    models = ModelManager(
        provider="nvidia",
        model_name=model_config["provider_models"]["nvidia"],
        config=model_config,
    )

    brain = CognitiveLoop(memory=memory, tool_manager=tools, model_manager=models, logger=logger)

    print("NIA CognitiveLoop demo (type 'exit' to quit)")
    # Voice mode is enabled only if both 'listen' and 'speak' tools are available
    voice_mode = args.voice and tools.has_tool('listen') and tools.has_tool('speak')

    # Background listener queue and thread (used only in voice mode)
    voice_queue: "queue.Queue[str]" = queue.Queue()
    listener = None
    if voice_mode:
        # Use the VoiceManager-agnostic BackgroundListener. We pass a tiny
        # adapter that exposes `listen(**kwargs)` using the underlying tools
        class _ToolAdapter:
            def __init__(self, tools):
                self._tools = tools

            def listen(self, **kwargs):
                # Keep compatibility with both sync execute and voice manager
                try:
                    return self._tools.execute('listen', kwargs)
                except Exception:
                    # Fallback to async execution if plugin expects it
                    try:
                        return asyncio.run(self._tools.execute_async('listen', kwargs, timeout=kwargs.get('_timeout', 2)))
                    except Exception:
                        return None

        listener = BackgroundListener(_ToolAdapter(tools), output_queue=voice_queue, poll_interval=0.1, timeout=2)
        listener.start()

    try:
        while True:
            try:
                if voice_mode:
                    # run the listen tool with a short timeout and fall back to stdin
                    # if it fails or returns nothing. Use execute_async with a timeout
                    # to allow the plugin to be interruptible and non-blocking.
                    try:
                        li = asyncio.run(tools.execute_async('listen', {}, timeout=3))
                    except Exception:
                        li = None
                    # Normalize plugin responses (dict/object) to a plain string
                    text = _listen_result_to_text(li)
                    if text:
                        user_input = text
                    else:
                        # fallback to typed input if listen produced no text
                        # Accept 'type' or 't' prefix as a keyboard override while in voice mode
                        typed = input('[voice] Press Enter to type or wait for speech:\n> ')
                        if typed.strip().startswith('type '):
                            user_input = typed.strip()[5:]
                        elif typed.strip().startswith('t '):
                            user_input = typed.strip()[2:]
                        else:
                            user_input = typed
                else:
                    user_input = input('> ')
            except (KeyboardInterrupt, EOFError):
                print("\nExiting.")
                break

            if not user_input:
                continue
            cmd = user_input.strip().lower()
            if cmd in ("exit", "quit"):
                print("Goodbye")
                break
            if cmd == "list plugins":
                print("Plugin tool(s):", tools.plugin_tools())
                continue
            if cmd == "reload plugins":
                try:
                    count = tools.reload_plugins(PLUGIN_DIR)
                    logger.info("Reloaded %d plugins", count)
                except Exception as exc:
                    logger.warning("Reloading plugins failed: %s", exc)
                print("Plugins reloaded. Plugin tool(s):", tools.plugin_tools())
                continue
            if cmd.startswith("unload plugin"):
                parts = cmd.split()
                if len(parts) == 3:
                    try:
                        ok = tools.unload_plugin(parts[2])
                    except Exception as exc:
                        ok = False
                        logger.warning("Failed to unload plugin %s: %s", parts[2], exc)
                    print(f"Plugin '{parts[2]}' unloaded: {ok}. Plugin tools:", tools.plugin_tools())
                else:
                    print("Usage: unload plugin <tool_name>")
                continue

            response = brain.run(user_input)
            print(response)
            if voice_mode:
                # call the speak tool and ignore errors
                try:
                    tools.execute('speak', {'text': response})
                except Exception as exc:
                    logger.debug("speak tool failed: %s", exc)
    finally:
        if listener:
            listener.stop()
    # finally:
    #     if listener:
    #         listener.stop()


if __name__ == "__main__":
    main()
