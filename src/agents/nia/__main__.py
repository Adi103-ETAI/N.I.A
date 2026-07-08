"""N.I.A CLI — Jarvis-style entry point.

Usage:
    python -m agents.nia              # Start N.I.A (React TUI or Python fallback)
    python -m agents.nia --print "..." # Single prompt, print and exit
    python -m agents.nia --help       # Show help

This is the primary entry point for NIA. It boots the NIA agent, which
owns identity (SOUL.md), memory, and personality, and uses niaharness
as its runtime (tools, permissions, hooks, MCP).

NIA tries to launch the React+Ink terminal frontend (same tech stack
as Hermes Agent) for the best UI experience. If the frontend isn't
available (no Node.js, no node_modules), it falls back to the Python
CLI with the caduceus banner.

First-run API key setup: if no API key is found, NIA shows an
interactive setup prompt instead of crashing.
"""

from __future__ import annotations

import asyncio
import os
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


def _check_api_key(api_key: str | None) -> str | None:
    """Check if an API key is available. Returns the key or None.

    Checks (in order):
    1. Explicitly passed api_key
    2. Environment variables (ANTHROPIC_API_KEY, OPENAI_API_KEY, etc.)
    3. Settings file (~/.niaharness/settings.json)
    """
    if api_key:
        return api_key

    # Check environment variables.
    env_vars = [
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "NVIDIA_API_KEY",
        "DEEPSEEK_API_KEY",
        "GROQ_API_KEY",
        "OPENROUTER_API_KEY",
        "TOGETHER_API_KEY",
        "FIREWORKS_API_KEY",
        "CEREBRAS_API_KEY",
        "XAI_API_KEY",
        "PERPLEXITY_API_KEY",
        "MISTRAL_API_KEY",
        "DEEPINFRA_API_KEY",
        "HF_API_KEY",
        "GOOGLE_API_KEY",
        "AZURE_OPENAI_API_KEY",
    ]
    for var in env_vars:
        value = os.environ.get(var, "").strip()
        if value:
            return value

    # Check settings file.
    try:
        from niaharness.config.settings import load_settings

        settings = load_settings()
        if settings.api_key:
            return settings.api_key
    except Exception:
        pass

    return None


def _run_first_run_setup() -> str | None:
    """Interactive first-run setup. Returns an API key or None.

    Shows the NIA banner, explains what's needed, and prompts for
    an API key. Saves the key to settings.json for future runs.
    """
    from agents.nia.cli_ui import render_banner, console
    from rich.rule import Rule
    from rich.text import Text
    from rich.panel import Panel

    console.clear()
    console.print(render_banner())
    console.print()
    console.print(Rule(style="#B8860B"))
    console.print()

    # Setup prompt
    setup_text = Text()
    setup_text.append("  Welcome to N.I.A!\n\n", style="bold #FFD700")
    setup_text.append("  To get started, you need an API key from one of these providers:\n\n", style="#F8F8F2")
    setup_text.append("  • Anthropic  (ANTHROPIC_API_KEY)  — Claude models\n", style="#00CED1")
    setup_text.append("  • OpenAI     (OPENAI_API_KEY)      — GPT models\n", style="#00CED1")
    setup_text.append("  • DeepSeek   (DEEPSEEK_API_KEY)    — DeepSeek models\n", style="#00CED1")
    setup_text.append("  • Groq       (GROQ_API_KEY)        — Fast inference\n", style="#00CED1")
    setup_text.append("  • OpenRouter (OPENROUTER_API_KEY)  — Multi-provider\n", style="#00CED1")
    setup_text.append("  • Or any of 15+ other providers\n\n", style="#6272A4")
    setup_text.append("  Get a key from: https://console.anthropic.com/ or https://platform.openai.com/\n\n", style="#6272A4")
    setup_text.append("  Paste your API key below (or press Ctrl+C to exit):\n", style="#FF8C00")

    console.print(Panel(setup_text, border_style="#B8860B", title="[bold #FFD700] First-Time Setup [/]", title_align="left", padding=(1, 2)))
    console.print()

    try:
        key = input("  API Key: ").strip()
    except (KeyboardInterrupt, EOFError):
        console.print(Text("\n  Goodbye.", style="#FFD700"))
        return None

    if not key:
        console.print(Text("\n  No key provided. Set ANTHROPIC_API_KEY env var and try again.", style="#FF5555"))
        return None

    # Save to settings.
    try:
        from niaharness.config.settings import load_settings, save_settings

        settings = load_settings()
        settings.api_key = key
        save_settings(settings)
        console.print(Text("\n  ✓ API key saved to ~/.niaharness/settings.json", style="#50FA7B"))
        console.print(Text("  Starting N.I.A...\n", style="#6272A4"))
    except Exception as exc:
        # If we can't save, just use it for this session.
        console.print(Text(f"\n  ⚠ Could not save key ({exc}), using for this session only.", style="#FF8C00"))

    return key


def _try_launch_react_tui(
    *,
    cwd: str,
    model: str | None,
    base_url: str | None,
    api_key: str,
    system_prompt: str | None = None,
) -> bool:
    """Try to launch the React+Ink terminal frontend.

    Returns True if launched, False if unavailable.
    """
    try:
        from niaharness.ui.react_launcher import launch_react_tui, get_frontend_dir

        # Check if frontend exists and Node.js is available.
        frontend_dir = get_frontend_dir()
        if not (frontend_dir / "package.json").exists():
            return False

        import shutil

        if not shutil.which("npm"):
            return False

        # Launch the React TUI (this blocks until the user exits).
        exit_code = asyncio.run(launch_react_tui(
            prompt=None,
            cwd=cwd,
            model=model,
            base_url=base_url,
            system_prompt=system_prompt,
            api_key=api_key,
        ))
        return exit_code == 0
    except Exception:
        return False


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
    no_frontend: bool = typer.Option(
        False,
        "--no-frontend",
        help="Skip the React TUI and use the Python CLI fallback.",
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

    # Check for API key — run first-run setup if missing.
    resolved_api_key = _check_api_key(resolved_api_key)
    if not resolved_api_key:
        # No API key found — run interactive setup.
        resolved_api_key = _run_first_run_setup()
        if not resolved_api_key:
            raise typer.Exit(1)

    # P1 fix: set the active profile before NIA boots.
    if profile:
        from niaharness.profiles import switch_profile

        try:
            switch_profile(profile)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            raise typer.Exit(1)

    # --print mode: non-interactive, use Python CLI.
    if print_mode is not None:
        from agents.nia.nia import NIA
        from niaharness.engine.stream_events import AssistantTextDelta, AssistantTurnComplete

        prompt = print_mode.strip()
        if not prompt:
            print("Error: --print requires a prompt value.", file=sys.stderr)
            raise typer.Exit(1)

        async def run_print() -> int:
            nia = NIA(working_directory=cwd)
            try:
                await nia.initialize(api_key=resolved_api_key, model=resolved_model, base_url=resolved_base_url)
                async for event in nia.chat(prompt):
                    if isinstance(event, AssistantTextDelta):
                        sys.stdout.write(event.text)
                        sys.stdout.flush()
                    elif isinstance(event, AssistantTurnComplete):
                        sys.stdout.write("\n")
                return 0
            finally:
                await nia.shutdown()

        exit_code = asyncio.run(run_print())
        if exit_code != 0:
            raise typer.Exit(exit_code)
        return

    # Interactive mode: try React TUI first, fall back to Python CLI.
    if not no_frontend:
        launched = _try_launch_react_tui(
            cwd=cwd,
            model=resolved_model,
            base_url=resolved_base_url,
            api_key=resolved_api_key,
        )
        if launched:
            return
        # Fall through to Python CLI if React TUI unavailable.

    # Python CLI fallback (with caduceus banner + flicker-free streaming).
    from agents.nia.nia import NIA
    from agents.nia.cli_ui import run_interactive

    async def run_python_cli() -> int:
        nia = NIA(working_directory=cwd)
        try:
            await nia.initialize(api_key=resolved_api_key, model=resolved_model, base_url=resolved_base_url)
        except Exception as exc:
            print(f"Failed to initialize N.I.A: {exc}", file=sys.stderr)
            return 1
        await run_interactive(nia)
        await nia.shutdown()
        return 0

    exit_code = asyncio.run(run_python_cli())
    if exit_code != 0:
        raise typer.Exit(exit_code)


if __name__ == "__main__":
    app()
