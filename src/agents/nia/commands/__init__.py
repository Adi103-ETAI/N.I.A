"""N.I.A Commands - Slash commands for the interactive session."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.nia.nia import NIA


async def handle_command(nia: "NIA", command: str, args: dict[str, str]) -> str:
    """Handle a slash command from the user."""
    handlers = {
        "connect": cmd_connect,
        "models": cmd_models,
        "status": cmd_status,
        "provider": cmd_provider,
        "clear": cmd_clear,
        "help": cmd_help,
    }

    handler = handlers.get(command)
    if handler:
        return await handler(nia, args)

    return f"Unknown command: /{command}. Type /help for available commands."


async def cmd_connect(nia: "NIA", args: dict[str, str]) -> str:
    """Handle /connect command - configure a provider."""
    provider_id = args.get("0") or args.get("provider")

    if not provider_id:
        # List available providers
        providers = nia._provider_registry.list_providers()
        lines = ["Available providers:\n"]
        for p in providers:
            status = "configured" if p.api_key_configured else "not configured"
            lines.append(f"  {p.id:<15} {p.name:<25} [{status}]")
        lines.append("\nUsage: /connect <provider_id>")
        lines.append("Example: /connect anthropic")
        return "\n".join(lines)

    # Try to configure the provider
    provider = nia._provider_registry.get_provider(provider_id)
    if not provider:
        return f"Provider '{provider_id}' not found. Use /connect to see available providers."

    # Check if already configured
    if provider.is_configured():
        return f"Provider '{provider_id}' is already configured."

    # Get API key from args or prompt
    api_key = args.get("api_key") or args.get("key")
    base_url = args.get("base_url") or args.get("url")

    if api_key:
        provider.configure(api_key=api_key, base_url=base_url)
        nia._config_manager.add_provider(provider_id, api_key=api_key, base_url=base_url)
        return f"Provider '{provider_id}' configured successfully."

    return (
        f"To configure {provider.name}:\n"
        f"  /connect {provider_id} api_key=YOUR_API_KEY\n"
        f"  /connect {provider_id} api_key=YOUR_API_KEY base_url=CUSTOM_URL\n"
        f"\nOr set environment variable: export {provider_id.upper()}_API_KEY=YOUR_KEY"
    )


async def cmd_models(nia: "NIA", args: dict[str, str]) -> str:
    """Handle /models command - list and select models."""
    provider_id = args.get("0") or args.get("provider")

    if provider_id:
        # List models for specific provider
        provider = nia._provider_registry.get_provider(provider_id)
        if not provider:
            return f"Provider '{provider_id}' not found."

        models = provider.list_models()
        if not models:
            return f"No models available for '{provider_id}'."

        lines = [f"Models for {provider.name}:\n"]
        current = nia._provider_registry.get_active_model()
        for m in models:
            marker = " -> " if m.id == current else "    "
            lines.append(f"{marker}{m.id:<40} {m.name}")
        lines.append(f"\nSelect: /models {provider_id} <model_id>")
        return "\n".join(lines)

    # List all models from all providers
    all_models = nia._provider_registry.get_all_models()
    if not all_models:
        return "No models available. Use /connect to configure a provider first."

    # Group by provider
    by_provider: dict[str, list] = {}
    for m in all_models:
        if m.provider_id not in by_provider:
            by_provider[m.provider_id] = []
        by_provider[m.provider_id].append(m)

    current_model = nia._provider_registry.get_active_model()
    current_provider = nia._provider_registry._active_provider_id

    lines = ["Available models:\n"]
    for pid, models in by_provider.items():
        lines.append(f"  {pid}:")
        for m in models:
            is_current = (pid == current_provider and m.id == current_model)
            marker = " * " if is_current else "   "
            lines.append(f"    {marker}{m.id}")
        lines.append("")

    lines.append("Select: /models <provider> <model_id>")
    lines.append("Example: /models anthropic claude-sonnet-4-20250514")

    return "\n".join(lines)


async def cmd_status(nia: "NIA", args: dict[str, str]) -> str:
    """Handle /status command - show system status."""
    status = nia.get_status()
    lines = [
        "N.I.A Status:",
        f"  System: {status['state']}",
        f"  Provider: {status.get('active_provider', 'none')}",
        f"  Model: {status.get('active_model', 'none')}",
        f"  Decisions: {status.get('total_decisions', 0)}",
        f"  Memory: {status.get('memory_entries', 0)} entries",
    ]
    return "\n".join(lines)


async def cmd_provider(nia: "NIA", args: dict[str, str]) -> str:
    """Handle /provider command - switch active provider."""
    provider_id = args.get("0")
    model = args.get("1") or args.get("model")

    if not provider_id:
        current = nia._provider_registry._active_provider_id or "none"
        return f"Current provider: {current}\nUsage: /provider <provider_id> [model_id]"

    success = nia._provider_registry.set_active(provider_id, model)
    if success:
        msg = f"Active provider set to: {provider_id}"
        if model:
            msg += f"/{model}"
        return msg
    return f"Failed to set provider '{provider_id}'. Is it configured?"


async def cmd_clear(nia: "NIA", args: dict[str, str]) -> str:
    """Handle /clear command - clear conversation history."""
    nia._brain.clear_history()
    return "Conversation history cleared."


async def cmd_help(nia: "NIA", args: dict[str, str]) -> str:
    """Handle /help command."""
    return """N.I.A Commands:

  /connect              List or configure providers
  /connect <provider>   Configure a provider
  /models               List all available models
  /models <provider>    List models for a provider
  /provider <id> [model] Switch active provider/model
  /status               Show system status
  /clear                Clear conversation history
  /help                 Show this help

Examples:
  /connect anthropic api_key=sk-ant-...
  /connect ollama
  /models anthropic
  /provider openai gpt-4o
"""
