"""N.I.A CLI - Command line interface for Neural Intelligence Assistant.

Usage:
    python -m agents.nia              # Start N.I.A with React TUI
    python -m agents.nia --text       # Start text-only mode
    python -m agents.nia --backend-only  # Run as backend for React frontend
    python -m agents.nia --help       # Show help
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer

app = typer.Typer(
    name="nia",
    help="N.I.A - Neural Intelligence Assistant\n\nThe head that listens, speaks, and divides tasks.",
    add_completion=False,
    invoke_without_command=True,
)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    cwd: str = typer.Option(
        str(Path.cwd()),
        "--cwd",
        help="Working directory",
    ),
    text_mode: bool = typer.Option(
        False,
        "--text",
        "-t",
        help="Text-only mode (no React TUI)",
    ),
    backend_only: bool = typer.Option(
        False,
        "--backend-only",
        help="Run as backend for React frontend (internal use)",
    ),
    provider: str | None = typer.Option(
        None,
        "--provider",
        "-p",
        help="LLM provider (anthropic, openai, ollama, etc.)",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        "-m",
        help="Model to use",
    ),
    api_key: str | None = typer.Option(
        None,
        "--api-key",
        "-k",
        help="API key for the provider",
    ),
    base_url: str | None = typer.Option(
        None,
        "--base-url",
        help="Custom API base URL",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging",
    ),
    prompt: str | None = typer.Option(
        None,
        "--prompt",
        help="Initial prompt to send",
    ),
) -> None:
    """Start N.I.A interactive session."""
    if ctx.invoked_subcommand is not None:
        return

    if verbose:
        import logging
        logging.basicConfig(level=logging.INFO)

    if backend_only:
        asyncio.run(_run_backend(cwd, provider, model, api_key, base_url))
    elif text_mode:
        asyncio.run(_run_interactive(cwd))
    else:
        asyncio.run(_run_tui(cwd, provider, model, api_key, base_url, prompt))


@app.command()
def status() -> None:
    """Show N.I.A system status."""
    import json
    from agents.nia.nia import NIA
    nia = NIA()
    asyncio.run(nia.initialize())
    print(json.dumps(nia.get_status(), indent=2))
    nia.shutdown()


@app.command()
def greet(
    time: str = typer.Option(
        None,
        "--time",
        help="Time of day (morning, afternoon, evening, night)",
    ),
) -> None:
    """Get a greeting from N.I.A."""
    from agents.nia.core.personality import Personality
    personality = Personality()
    print(personality.greet(time or "afternoon"))


async def _run_tui(
    cwd: str,
    provider: str | None,
    model: str | None,
    api_key: str | None,
    base_url: str | None,
    prompt: str | None,
) -> None:
    """Run N.I.A with the React terminal UI."""
    from agents.nia.ui.launcher import launch_nia_tui

    exit_code = await launch_nia_tui(
        prompt=prompt,
        cwd=cwd,
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
    )
    if exit_code != 0:
        raise SystemExit(exit_code)


async def _run_backend(
    cwd: str,
    provider: str | None,
    model: str | None,
    api_key: str | None,
    base_url: str | None,
) -> None:
    """Run N.I.A as a backend for the React frontend."""
    from agents.nia.ui.backend_host import BackendHostConfig, run_nia_backend

    config = BackendHostConfig(
        working_directory=cwd,
        provider=provider or "",
        model=model or "",
        api_key=api_key or "",
        base_url=base_url or "",
    )
    await run_nia_backend(config)


async def _run_interactive(cwd: str) -> None:
    """Run N.I.A in text-only interactive mode."""
    from agents.nia.nia import NIA

    nia = NIA(working_directory=cwd)
    greeting = await nia.initialize()
    print(f"\n{greeting}\n")
    print("Type 'exit' or 'quit' to leave.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nShutting down N.I.A...")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "bye"):
            print("\nN.I.A: Goodbye. All systems secured.")
            break

        response = await nia.process(user_input)
        print(f"\nN.I.A: {response}\n")

    nia.shutdown()


if __name__ == "__main__":
    app()
