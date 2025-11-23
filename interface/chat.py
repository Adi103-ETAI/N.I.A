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
import os
from pathlib import Path
from models.model_manager import ModelManager
from persona.profile import build_persona_config
import argparse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("nia")

    parser = argparse.ArgumentParser()
    parser.add_argument('--voice', action='store_true', help='Use microphone and speech output')
    args = parser.parse_args()

    memory = InMemoryMemory()
    tools = ToolManager(logger=logger)
    try:
        tools.discover_and_register()
    except Exception as exc:
        logger.warning(f'Tool auto-discovery failed: {exc}')

    # Project root and plugin directory
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    PLUGIN_DIR = os.path.join(PROJECT_ROOT, "plugins")

    # Load any plugins from the plugins directory (best-effort)
    try:
        tools.load_plugins_from_directory(PLUGIN_DIR)
    except Exception as exc:
        logger.debug("Plugin load failed at startup: %s", exc)

    # Register a small set of development/demo tools. In production this
    # would be dynamic and policy-driven. Use explicit helper to avoid
    # import-time side-effects and keep registration explicit.
    try:
        register_dev_tools(tools)
    except Exception as exc:
        logger.debug("register_dev_tools failed: %s", exc)

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

    while True:
        try:
            if voice_mode:
                # run the listen tool and fall back to stdin if it fails or returns falsy
                try:
                    li = tools.execute('listen', {})
                except Exception:
                    li = None
                user_input = li or input('> ')
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


if __name__ == "__main__":
    main()
