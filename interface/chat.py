"""Minimal CLI to demo the CognitiveLoop.

This script demonstrates how to wire the core components together with
the default in-memory/tool/model stubs. It is intentionally simple so
developers can replace pieces with real implementations as they build.
"""
import logging
from core.brain import CognitiveLoop
from core.memory import InMemoryMemory
from core.tool_manager import ToolManager
from core.tools.echo_tool import EchoTool
from models.model_manager import ModelManager
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

    # Register a small set of development/demo tools. In production this
    # would be dynamic and policy-driven.
    tools.register_tool(EchoTool)
    models = ModelManager()

    brain = CognitiveLoop(memory=memory, tool_manager=tools, model_manager=models, logger=logger)

    print("NIA CognitiveLoop demo (type 'exit' to quit)")
    voice_mode = args.voice and hasattr(tools, 'listen') and hasattr(tools, 'speak')

    while True:
        try:
            if voice_mode:
                user_input = tools.listen() or input('> ')
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
            tools.reload_plugins()
            print("Plugins reloaded. Plugin tool(s):", tools.plugin_tools())
            continue
        if cmd.startswith("unload plugin"):
            parts = cmd.split()
            if len(parts)==3:
                ok = tools.unload_plugin(parts[2])
                print(f"Plugin '{parts[2]}' unloaded: {ok}. Plugin tools:", tools.plugin_tools())
            else:
                print("Usage: unload plugin <tool_name>")
            continue

        response = brain.run(user_input)
        print(response)
        if voice_mode:
            tools.speak(response)


if __name__ == "__main__":
    main()
