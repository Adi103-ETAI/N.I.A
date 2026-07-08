"""N.I.A CLI — Jarvis-style entry point.

Usage:
    python -m agents.nia              # Start N.I.A (interactive REPL)
    python -m agents.nia --print "..." # Single prompt, print and exit
    python -m agents.nia --help       # Show help

This is the primary entry point for NIA. It boots the NIA agent, which
owns identity (SOUL.md), memory, and personality, and uses niaharness
as its runtime (tools, permissions, hooks, MCP).

The niaharness CLI (niaharness.cli) remains available for backward
compatibility and power users who want the raw harness without NIA's
identity layer.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import typer

app = typer.Typer(
    name="nia",
    help="N.I.A — Neural Intelligence Assistant. Your Jarvis-style AI partner.",
    add_completion=False,
    rich_markup_mode="rich",
    invoke_without_command=True,
)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    cwd: str = typer.Option(
        str(Path.cwd()),
        "--cwd",
        help="Working directory for the session.",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        "-m",
        help="Model name (e.g. 'claude-3-opus', 'gpt-4o'). Defaults to config.",
    ),
    api_key: str | None = typer.Option(
        None,
        "--api-key",
        "-k",
        help="API key (overrides config and environment).",
    ),
    base_url: str | None = typer.Option(
        None,
        "--base-url",
        help="API base URL override.",
    ),
    provider: str | None = typer.Option(
        None,
        "--provider",
        "-p",
        help=(
            "LLM provider (anthropic, openai, opencode, xai, perplexity, "
            "openrouter, groq, deepseek, ollama, etc.). Use --list-providers."
        ),
    ),
    print_mode: str | None = typer.Option(
        None,
        "--print",
        help="Non-interactive: print response and exit. Pass prompt as value.",
    ),
    list_providers: bool = typer.Option(
        False,
        "--list-providers",
        help="List all available LLM providers and exit.",
    ),
    profile: str | None = typer.Option(
        None,
        "--profile",
        help=(
            "NIA profile name (e.g. 'work', 'personal'). Uses ~/.nia/profiles/<name>/. "
            "Defaults to the active profile or 'default'."
        ),
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        "-d",
        help="Enable debug logging.",
    ),
) -> None:
    """Start N.I.A — your Jarvis-style AI assistant."""
    if ctx.invoked_subcommand is not None:
        return

    if debug:
        import logging
        logging.basicConfig(level=logging.DEBUG)

    # --list-providers: delegate to niaharness's provider registry.
    if list_providers:
        from niaharness.providers.registry import ProviderRegistry

        registry = ProviderRegistry()
        registry._register_builtin_providers()
        print(f"\nAvailable LLM providers ({len(registry._providers)} total):\n")
        for name, prov in sorted(registry._providers.items()):
            cfg = prov.config
            env_vars = cfg.auth.api_key_env_vars
            env_hint = env_vars[0] if env_vars else "(no key needed)"
            print(f"  {name:<15} {cfg.label:<22} key: {env_hint}")
        raise typer.Exit(0)

    # If --provider was given, resolve credentials from niaharness's registry.
    resolved_api_key = api_key
    resolved_base_url = base_url
    resolved_model = model
    if provider:
        from niaharness.providers.registry import ProviderRegistry as _PR

        _reg = _PR()
        _reg._register_builtin_providers()
        _prov = _reg.get_provider(provider)
        if _prov is None:
            print(
                f"Unknown provider: {provider!r}. Use --list-providers to see options.",
                file=sys.stderr,
            )
            raise typer.Exit(1)
        _cfg = _prov.config
        try:
            resolved_api_key = _prov.resolve_api_key(api_key)
        except Exception:
            resolved_api_key = api_key
        resolved_base_url = base_url or _prov.resolve_base_url()
        resolved_model = model or _cfg.auth.default_model
        if not resolved_api_key:
            env_hint = _cfg.auth.api_key_env_vars[0] if _cfg.auth.api_key_env_vars else "(none)"
            print(
                f"Provider {provider!r} requires an API key. Set {env_hint} env var or use --api-key.",
                file=sys.stderr,
            )
            raise typer.Exit(1)

    # Boot NIA.
    from agents.nia.nia import NIA
    from niaharness.engine.stream_events import (
        AssistantTextDelta,
        AssistantTurnComplete,
        ToolExecutionStarted,
        ToolExecutionCompleted,
    )

    # P1 fix: set the active profile before NIA boots, so all path
    # resolution (SOUL.md, memory, sessions.db, credentials, skills) uses
    # the profile-scoped directory.
    if profile:
        from niaharness.profiles import switch_profile

        try:
            switch_profile(profile)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            raise typer.Exit(1)

    async def run() -> int:
        nia = NIA(working_directory=cwd)
        try:
            greeting = await nia.initialize(
                api_key=resolved_api_key,
                model=resolved_model,
                base_url=resolved_base_url,
            )
        except Exception as exc:
            print(f"Failed to initialize N.I.A: {exc}", file=sys.stderr)
            return 1

        if print_mode is not None:
            # Non-interactive mode: print response and exit.
            prompt = print_mode.strip()
            if not prompt:
                print("Error: --print requires a prompt value.", file=sys.stderr)
                return 1
            try:
                async for event in nia.chat(prompt):
                    if isinstance(event, AssistantTextDelta):
                        sys.stdout.write(event.text)
                        sys.stdout.flush()
                    elif isinstance(event, AssistantTurnComplete):
                        sys.stdout.write("\n")
                return 0
            finally:
                await nia.shutdown()
        else:
            # Interactive REPL with the new NIA UI.
            from agents.nia.cli_ui import run_interactive

            await run_interactive(nia)
            await nia.shutdown()
            return 0

    exit_code = asyncio.run(run())
    if exit_code != 0:
        raise typer.Exit(exit_code)


if __name__ == "__main__":
    app()
