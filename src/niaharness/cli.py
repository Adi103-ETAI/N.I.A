"""CLI entry point using typer."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

# Load ~/.nia/.env BEFORE anything else — this is the single most critical
# startup step. Without it, API keys saved by `nia setup` never reach
# os.environ and every provider probe / API call fails with "no key".
try:
    from niaharness.config.env_loader import load_nia_env
    load_nia_env()
except Exception:
    pass  # Best-effort — don't crash if .env doesn't exist yet.

import typer

app = typer.Typer(
    name="nia",
    help=(
        "N.I.A — Neural Intelligence Assistant\n\n"
        "An AI partner inspired by J.A.R.V.I.S. — thinks, plans, and executes with calm authority.\n\n"
        "Starts an interactive session by default. Use subcommands for management tasks:\n"
        "  nia setup     — First-time setup wizard\n"
        "  nia doctor    — Diagnose and auto-repair issues\n"
        "  nia update    — Check for and install updates\n"
        "  nia gateway   — Manage chat platform gateway\n"
        "  nia profile   — Manage profiles and aliases\n"
        "  nia auth      — Manage authentication\n"
        "  nia cron      — Manage scheduled jobs\n"
        "  nia status    — Show system status\n"
        "  nia version   — Show version info"
    ),
    add_completion=False,
    rich_markup_mode="rich",
    invoke_without_command=True,
    no_args_is_help=False,
)


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

mcp_app = typer.Typer(name="mcp", help="Manage MCP servers")
plugin_app = typer.Typer(name="plugin", help="Manage plugins")
auth_app = typer.Typer(name="auth", help="Manage authentication")
cron_app = typer.Typer(name="cron", help="Manage cron scheduler and jobs")
gateway_app = typer.Typer(name="gateway", help="Manage chat platform gateway")
profile_app = typer.Typer(name="profile", help="Manage profiles and aliases")
memory_app = typer.Typer(name="memory", help="Manage persistent memory store")

app.add_typer(mcp_app)
app.add_typer(plugin_app)
app.add_typer(auth_app)
app.add_typer(cron_app)
app.add_typer(gateway_app)
app.add_typer(profile_app)
app.add_typer(memory_app)


# ---------------------------------------------------------------------------
# nia setup — first-time setup wizard
# ---------------------------------------------------------------------------

@app.command("setup")
def setup_command() -> None:
    """First-time setup wizard — configure API key, model, and identity."""
    print()
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║              🤖 N.I.A Setup Wizard                        ║")
    print("╚═══════════════════════════════════════════════════════════╝")
    print()

    # Step 1: API key
    print("Step 1: API Key")
    print("  NIA needs an API key to talk to an LLM provider.")
    print("  Supported: Anthropic, OpenAI, OpenRouter, and 17+ more.")
    print()
    key = typer.prompt("  Paste your API key (or press Enter to skip)", default="", show_default=False)
    if key:
        # Detect provider from key prefix.
        if key.startswith("sk-ant-"):
            env_var = "ANTHROPIC_API_KEY"
            provider = "Anthropic"
        elif key.startswith("sk-or-"):
            env_var = "OPENROUTER_API_KEY"
            provider = "OpenRouter"
        elif key.startswith("sk-"):
            env_var = "OPENAI_API_KEY"
            provider = "OpenAI"
        else:
            # Generic — ask which env var to use.
            env_var = "OPENAI_API_KEY"
            provider = "auto-detect"

        # Save to ~/.nia/.env using the env_loader (also sets in os.environ immediately).
        from niaharness.config.env_loader import save_env_value, get_env_path
        save_env_value(env_var, key)
        env_path = get_env_path()
        print(f"\n  ✓ API key saved to {env_path} ({provider})")
        print(f"  ✓ Available in this session immediately (no restart needed)")
    else:
        print("  ⚠ Skipped — set ANTHROPIC_API_KEY or OPENAI_API_KEY manually later.")
    print()

    # Step 2: Model selection
    print("Step 2: Model")
    print("  Recommended models:")
    print("    1. claude-sonnet-4-6     (Anthropic — best balance)")
    print("    2. claude-opus-4-7       (Anthropic — most capable)")
    print("    3. gpt-4o                (OpenAI)")
    print("    4. deepseek-chat         (DeepSeek — cheapest)")
    model_choice = typer.prompt("  Choose (1-4) or type a model name", default="1")
    models = {
        "1": "claude-sonnet-4-6",
        "2": "claude-opus-4-7",
        "3": "gpt-4o",
        "4": "deepseek-chat",
    }
    model = models.get(model_choice, model_choice)
    print(f"\n  ✓ Model: {model}")
    print()

    # Step 3: SOUL.md
    print("Step 3: Identity (SOUL.md)")
    from niaharness.prompts.soul import get_nia_home
    soul_path = get_nia_home() / "SOUL.md"
    if not soul_path.exists():
        soul_path.parent.mkdir(parents=True, exist_ok=True)
        soul_path.write_text(
            "# NIA Agent Persona\n\n"
            "You are NIA, a helpful AI assistant with a calm, professional "
            "demeanor inspired by J.A.R.V.I.S.\n",
            encoding="utf-8",
        )
        print(f"  ✓ Created {soul_path}")
    else:
        print(f"  ✓ SOUL.md already exists at {soul_path}")
    print()

    # Step 4: Directory structure
    print("Step 4: Directory structure")
    nia_home = get_nia_home()
    for subdir in ["cron", "sessions", "skills", "memories", "mcp-tokens"]:
        d = nia_home / subdir
        d.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {d}")
    print()

    # Step 5: Done
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║  ✅ Setup complete!                                       ║")
    print("║                                                           ║")
    print("║  Start NIA with:  nia                                     ║")
    print("║  Or one-shot:     nia -p 'Hello NIA'                     ║")
    print("║  Run diagnostics: nia doctor                              ║")
    print("╚═══════════════════════════════════════════════════════════╝")


# ---------------------------------------------------------------------------
# nia doctor — run diagnostics
# ---------------------------------------------------------------------------

@app.command("doctor")
def doctor_command(
    fix: bool = typer.Option(False, "--fix", help="Auto-repair fixable issues"),
    ack: str = typer.Option(None, "--ack", help="Acknowledge a security advisory by ID"),
) -> None:
    """Run diagnostics and auto-repair issues."""
    from niaharness.cli.doctor import run_doctor
    result = run_doctor(fix=fix, ack=ack)
    print(result.report)


# ---------------------------------------------------------------------------
# nia update — check for and install updates
# ---------------------------------------------------------------------------

@app.command("update")
def update_command(
    check: bool = typer.Option(False, "--check", help="Check only, don't install"),
    no_backup: bool = typer.Option(False, "--no-backup", help="Skip pre-update backup"),
) -> None:
    """Check for and install NIA updates."""
    from niaharness.cli.update import run_update
    result = run_update(check=check, no_backup=no_backup)
    print(result.report)


# ---------------------------------------------------------------------------
# nia status — show system status
# ---------------------------------------------------------------------------

@app.command("status")
def status_command() -> None:
    """Show system status — version, model, provider, sessions, profile."""
    import importlib.metadata
    from niaharness.config.settings import load_settings

    try:
        version = importlib.metadata.version("niaharness")
    except Exception:
        version = "unknown"

    settings = load_settings()
    print()
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║                   📊 N.I.A Status                         ║")
    print("╚═══════════════════════════════════════════════════════════╝")
    print(f"  Version:    {version}")
    print(f"  Model:      {settings.model}")
    print(f"  Max tokens: {settings.max_tokens}")
    print(f"  Permission: {settings.permission.mode}")
    print(f"  API format: {settings.api_format}")

    # Check API key.
    import os
    has_key = bool(
        os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("OPENROUTER_API_KEY")
    )
    print(f"  API key:    {'✓ configured' if has_key else '✗ not set'}")

    # Check SOUL.md.
    from niaharness.prompts.soul import get_nia_home
    soul = get_nia_home() / "SOUL.md"
    print(f"  SOUL.md:    {'✓ exists' if soul.exists() else '✗ missing'}")
    print(f"  NIA home:   {get_nia_home()}")
    print()


# ---------------------------------------------------------------------------
# nia version — show version
# ---------------------------------------------------------------------------

@app.command("version")
def version_command() -> None:
    """Show NIA version information."""
    import importlib.metadata
    try:
        version = importlib.metadata.version("niaharness")
    except Exception:
        version = "unknown"
    print(f"N.I.A — Neural Intelligence Assistant v{version}")


# ---------------------------------------------------------------------------
# nia gateway subcommands
# ---------------------------------------------------------------------------

@gateway_app.command("run")
def gateway_run(
    platform: str = typer.Option("telegram", "--platform", help="Chat platform (telegram)"),
) -> None:
    """Start the gateway (Telegram bot long-polling)."""
    import asyncio
    import os

    token = os.environ.get("NIA_TELEGRAM_BOT_TOKEN", "")
    if not token:
        print("Error: NIA_TELEGRAM_BOT_TOKEN not set.", file=sys.stderr)
        print("Get a token from @BotFather, then:", file=sys.stderr)
        print("  export NIA_TELEGRAM_BOT_TOKEN='your-token'", file=sys.stderr)
        raise typer.Exit(1)

    from niaharness.gateway import GatewayRouter, TelegramAdapter

    router = GatewayRouter()
    adapter = TelegramAdapter(token=token, router=router)
    router.register_adapter(adapter)

    print(f"Starting NIA gateway on {platform}...")
    asyncio.run(router.start_all())


@gateway_app.command("status")
def gateway_status() -> None:
    """Check if the gateway is running."""
    from niaharness.services.cron_scheduler import is_scheduler_running
    print(f"Gateway: {'running' if is_scheduler_running() else 'stopped'}")


# ---------------------------------------------------------------------------
# nia profile subcommands
# ---------------------------------------------------------------------------

@profile_app.command("list")
def profile_list() -> None:
    """List all profiles."""
    from niaharness.profiles import list_profiles, get_active_profile
    profiles = list_profiles()
    active = get_active_profile()
    print(f"\nProfiles ({len(profiles)}):")
    for p in profiles:
        marker = " *" if p.name == active.name else "  "
        print(f"{marker} {p.name:<20} {'(default)' if p.is_default else ''}")
    print()


@profile_app.command("create")
def profile_create(
    name: str = typer.Argument(..., help="Profile name"),
    clone: bool = typer.Option(False, "--clone", help="Copy config/.env/SOUL.md from default"),
    alias: bool = typer.Option(False, "--alias", help="Create a wrapper script alias"),
) -> None:
    """Create a new profile."""
    from niaharness.profiles import create_profile, get_profile
    try:
        create_profile(name, seed_from_default=clone)
        print(f"✓ Created profile: {name}")
    except Exception as exc:
        print(f"✗ Failed: {exc}", file=sys.stderr)
        raise typer.Exit(1)

    if alias:
        from niaharness.profiles.aliases import create_wrapper_script
        path = create_wrapper_script(name)
        if path:
            print(f"✓ Alias created: {path}")
            print(f"  Type '{name}' to launch NIA under this profile.")
        else:
            print("⚠ Could not create alias (check ~/.local/bin is writable)")


@profile_app.command("delete")
def profile_delete(
    name: str = typer.Argument(..., help="Profile name to delete"),
) -> None:
    """Delete a profile."""
    from niaharness.profiles import delete_profile
    try:
        delete_profile(name)
        # Also remove alias.
        from niaharness.profiles.aliases import remove_wrapper_script
        remove_wrapper_script(name)
        print(f"✓ Deleted profile: {name}")
    except Exception as exc:
        print(f"✗ Failed: {exc}", file=sys.stderr)
        raise typer.Exit(1)


@profile_app.command("switch")
def profile_switch(
    name: str = typer.Argument(..., help="Profile to switch to"),
) -> None:
    """Switch the active profile."""
    from niaharness.profiles import switch_profile
    try:
        switch_profile(name)
        print(f"✓ Active profile: {name}")
    except Exception as exc:
        print(f"✗ Failed: {exc}", file=sys.stderr)
        raise typer.Exit(1)


@profile_app.command("alias")
def profile_alias(
    profile: str = typer.Argument(..., help="Profile to create an alias for"),
    name: str = typer.Option(None, "--name", help="Custom alias name (defaults to profile name)"),
) -> None:
    """Create a wrapper script alias for a profile."""
    from niaharness.profiles.aliases import create_wrapper_script, check_alias_collision
    alias_name = name or profile
    collision = check_alias_collision(alias_name)
    if collision:
        print(f"✗ {collision}", file=sys.stderr)
        raise typer.Exit(1)
    path = create_wrapper_script(alias_name, target=profile)
    if path:
        print(f"✓ Alias created: {path}")
        print(f"  Type '{alias_name}' to launch NIA under profile '{profile}'.")
    else:
        print("✗ Could not create alias", file=sys.stderr)
        raise typer.Exit(1)


# ---- mcp subcommands ----

@mcp_app.command("list")
def mcp_list() -> None:
    """List configured MCP servers."""
    from niaharness.config import load_settings
    from niaharness.mcp.config import load_mcp_server_configs
    from niaharness.plugins import load_plugins

    settings = load_settings()
    plugins = load_plugins(settings, str(Path.cwd()))
    configs = load_mcp_server_configs(settings, plugins)
    if not configs:
        print("No MCP servers configured.")
        return
    for name, cfg in configs.items():
        transport = cfg.get("transport", cfg.get("command", "unknown"))
        print(f"  {name}: {transport}")


@mcp_app.command("add")
def mcp_add(
    name: str = typer.Argument(..., help="Server name"),
    config_json: str = typer.Argument(..., help="Server config as JSON string"),
) -> None:
    """Add an MCP server configuration."""
    from niaharness.config import load_settings, save_settings

    settings = load_settings()
    try:
        cfg = json.loads(config_json)
    except json.JSONDecodeError as exc:
        print(f"Invalid JSON: {exc}", file=sys.stderr)
        raise typer.Exit(1)
    if not isinstance(settings.mcp_servers, dict):
        settings.mcp_servers = {}
    settings.mcp_servers[name] = cfg
    save_settings(settings)
    print(f"Added MCP server: {name}")


@mcp_app.command("remove")
def mcp_remove(
    name: str = typer.Argument(..., help="Server name to remove"),
) -> None:
    """Remove an MCP server configuration."""
    from niaharness.config import load_settings, save_settings

    settings = load_settings()
    if not isinstance(settings.mcp_servers, dict) or name not in settings.mcp_servers:
        print(f"MCP server not found: {name}", file=sys.stderr)
        raise typer.Exit(1)
    del settings.mcp_servers[name]
    save_settings(settings)
    print(f"Removed MCP server: {name}")


# ---- plugin subcommands ----

@plugin_app.command("list")
def plugin_list() -> None:
    """List installed plugins."""
    from niaharness.config import load_settings
    from niaharness.plugins import load_plugins

    settings = load_settings()
    plugins = load_plugins(settings, str(Path.cwd()))
    if not plugins:
        print("No plugins installed.")
        return
    for plugin in plugins:
        status = "enabled" if plugin.enabled else "disabled"
        print(f"  {plugin.name} [{status}] - {plugin.description or ''}")


@plugin_app.command("install")
def plugin_install(
    source: str = typer.Argument(..., help="Plugin source (path or URL)"),
) -> None:
    """Install a plugin from a source path."""
    from niaharness.plugins.installer import install_plugin_from_path

    result = install_plugin_from_path(source)
    print(f"Installed plugin: {result}")


@plugin_app.command("uninstall")
def plugin_uninstall(
    name: str = typer.Argument(..., help="Plugin name to uninstall"),
) -> None:
    """Uninstall a plugin."""
    from niaharness.plugins.installer import uninstall_plugin

    uninstall_plugin(name)
    print(f"Uninstalled plugin: {name}")


# ---- cron subcommands ----

@cron_app.command("start")
def cron_start() -> None:
    """Start the cron scheduler daemon."""
    from niaharness.services.cron_scheduler import is_scheduler_running, start_daemon

    if is_scheduler_running():
        print("Cron scheduler is already running.")
        return
    pid = start_daemon()
    print(f"Cron scheduler started (pid={pid})")


@cron_app.command("stop")
def cron_stop() -> None:
    """Stop the cron scheduler daemon."""
    from niaharness.services.cron_scheduler import stop_scheduler

    if stop_scheduler():
        print("Cron scheduler stopped.")
    else:
        print("Cron scheduler is not running.")


@cron_app.command("status")
def cron_status_cmd() -> None:
    """Show cron scheduler status and job summary."""
    from niaharness.services.cron_scheduler import scheduler_status

    status = scheduler_status()
    state = "running" if status["running"] else "stopped"
    print(f"Scheduler: {state}" + (f" (pid={status['pid']})" if status["pid"] else ""))
    print(f"Jobs:      {status['enabled_jobs']} enabled / {status['total_jobs']} total")
    print(f"Log:       {status['log_file']}")


@cron_app.command("list")
def cron_list_cmd() -> None:
    """List all registered cron jobs with schedule and status."""
    from niaharness.services.cron import load_cron_jobs

    jobs = load_cron_jobs()
    if not jobs:
        print("No cron jobs configured.")
        return
    for job in jobs:
        enabled = "on " if job.get("enabled", True) else "off"
        last = job.get("last_run", "never")
        if last != "never":
            last = last[:19]  # trim to readable datetime
        last_status = job.get("last_status", "")
        status_indicator = f" [{last_status}]" if last_status else ""
        print(f"  [{enabled}] {job['name']}  {job.get('schedule', '?')}")
        print(f"        cmd: {job['command']}")
        print(f"        last: {last}{status_indicator}  next: {job.get('next_run', 'n/a')[:19]}")


@cron_app.command("toggle")
def cron_toggle_cmd(
    name: str = typer.Argument(..., help="Cron job name"),
    enabled: bool = typer.Argument(..., help="true to enable, false to disable"),
) -> None:
    """Enable or disable a cron job."""
    from niaharness.services.cron import set_job_enabled

    if not set_job_enabled(name, enabled):
        print(f"Cron job not found: {name}")
        raise typer.Exit(1)
    state = "enabled" if enabled else "disabled"
    print(f"Cron job '{name}' is now {state}")


@cron_app.command("history")
def cron_history_cmd(
    name: str | None = typer.Argument(None, help="Filter by job name"),
    limit: int = typer.Option(20, "--limit", "-n", help="Number of entries"),
) -> None:
    """Show cron execution history."""
    from niaharness.services.cron_scheduler import load_history

    entries = load_history(limit=limit, job_name=name)
    if not entries:
        print("No execution history.")
        return
    for entry in entries:
        ts = entry.get("started_at", "?")[:19]
        status = entry.get("status", "?")
        rc = entry.get("returncode", "?")
        print(f"  {ts}  {entry.get('name', '?')}  {status} (rc={rc})")
        stderr = entry.get("stderr", "").strip()
        if stderr and status != "success":
            for line in stderr.splitlines()[:3]:
                print(f"    stderr: {line}")


@cron_app.command("logs")
def cron_logs_cmd(
    lines: int = typer.Option(30, "--lines", "-n", help="Number of lines to show"),
) -> None:
    """Show recent cron scheduler log output."""
    from niaharness.config.paths import get_logs_dir

    log_path = get_logs_dir() / "cron_scheduler.log"
    if not log_path.exists():
        print("No scheduler log found. Start the scheduler with: oh cron start")
        return
    content = log_path.read_text(encoding="utf-8", errors="replace")
    tail = content.splitlines()[-lines:]
    for line in tail:
        print(line)


# ---- auth subcommands ----

@auth_app.command("status")
def auth_status_cmd() -> None:
    """Show authentication status."""
    from niaharness.api.provider import auth_status, detect_provider
    from niaharness.config import load_settings

    settings = load_settings()
    provider = detect_provider(settings)
    status = auth_status(settings)
    print(f"Provider: {provider}")
    print(f"Status:   {status}")


@auth_app.command("login")
def auth_login(
    api_key: str | None = typer.Option(None, "--api-key", "-k", help="API key"),
) -> None:
    """Configure authentication."""
    from niaharness.config import load_settings, save_settings

    if not api_key:
        api_key = typer.prompt("Enter your API key", hide_input=True)
    settings = load_settings()
    settings.api_key = api_key
    save_settings(settings)
    print("API key saved.")


@auth_app.command("logout")
def auth_logout() -> None:
    """Remove stored authentication."""
    from niaharness.config import load_settings, save_settings

    settings = load_settings()
    settings.api_key = None
    save_settings(settings)
    print("Authentication cleared.")


# ---- memory subcommands ----

@memory_app.command("setup")
def memory_setup(
    non_interactive: bool = typer.Option(
        False,
        "--non-interactive",
        "--yes",
        "-y",
        help="Skip prompts and create an empty store non-interactively",
    ),
) -> None:
    """Initialize the persistent memory store for the current project."""
    from niaharness.memory import run_memory_setup_wizard

    result = run_memory_setup_wizard(interactive=not non_interactive)
    print()
    print("Memory Setup Complete")
    print("=" * 40)
    print(f"Memory directory: {result['memory_dir']}")
    print(f"Entrypoint:       {result['entrypoint']}")
    print(f"Store:            {result['store_path']}")
    if result["created"]:
        print(f"Created:          {', '.join(result['created'])}")
    print(f"Seeded entries:   {result['seeded_entries']}")
    stats = result.get("stats", {})
    if stats:
        print(f"Total entries:    {stats.get('entry_count', 0)}")
        print(f"Total chars:      {stats.get('total_chars', 0)}/{stats.get('max_total_chars', 0)}")
    print()


@memory_app.command("stats")
def memory_stats() -> None:
    """Show statistics about the persistent memory store."""
    from niaharness.memory import MemoryStore
    from niaharness.memory.paths import get_project_memory_dir

    import os
    store_path = get_project_memory_dir(os.getcwd()) / "STORE.md"
    if not store_path.exists():
        print("No memory store found. Run `nia memory setup` first.")
        return
    store = MemoryStore(path=store_path)
    stats = store.stats()
    print(f"Path:             {stats['path']}")
    print(f"Entries:          {stats['entry_count']}")
    print(f"Total chars:      {stats['total_chars']}/{stats['max_total_chars']}")
    print(f"Max entry chars:  {stats['max_entry_chars']}")
    print(f"Write-gate blocked:   {stats['write_gate_blocked']}")
    print(f"Write-gate approved:  {stats['write_gate_approved']}")
    if stats["categories"]:
        print("Categories:")
        for cat, count in sorted(stats["categories"].items()):
            print(f"  {cat}: {count}")


@memory_app.command("list")
def memory_list(
    category: str | None = typer.Option(
        None, "--category", "-c", help="Filter by category"
    ),
    limit: int = typer.Option(
        20, "--limit", "-n", help="Max results"
    ),
) -> None:
    """List memory entries."""
    from niaharness.memory import MemoryStore
    from niaharness.memory.paths import get_project_memory_dir

    import os
    store_path = get_project_memory_dir(os.getcwd()) / "STORE.md"
    if not store_path.exists():
        print("No memory store found. Run `nia memory setup` first.")
        return
    store = MemoryStore(path=store_path)
    entries = store.get_entries(category=category, limit=limit)
    if not entries:
        print("No entries found.")
        return
    for i, entry in enumerate(entries):
        print(f"[{i}] [{entry.category}] {entry.content[:100]}")
        print(f"    source={entry.source} ts={entry.timestamp:.0f}")


@memory_app.command("search")
def memory_search(
    query: str = typer.Argument(..., help="Search query"),
    limit: int = typer.Option(
        5, "--limit", "-n", help="Max results"
    ),
) -> None:
    """Search memory entries."""
    from niaharness.memory import MemoryStore
    from niaharness.memory.paths import get_project_memory_dir

    import os
    store_path = get_project_memory_dir(os.getcwd()) / "STORE.md"
    if not store_path.exists():
        print("No memory store found. Run `nia memory setup` first.")
        return
    store = MemoryStore(path=store_path)
    entries = store.get_entries(query=query, limit=limit)
    if not entries:
        print("No matching entries found.")
        return
    for i, entry in enumerate(entries):
        print(f"[{i}] [{entry.category}] {entry.content[:200]}")


@memory_app.command("add")
def memory_add(
    content: str = typer.Argument(..., help="Memory entry content"),
    category: str = typer.Option(
        "note", "--category", "-c", help="Entry category"
    ),
) -> None:
    """Add a memory entry."""
    from niaharness.memory import MemoryEntry, MemoryStore
    from niaharness.memory.paths import get_project_memory_dir

    import os
    store_path = get_project_memory_dir(os.getcwd()) / "STORE.md"
    store = MemoryStore(path=store_path)
    entry = MemoryEntry(content=content, category=category, source="user")
    if store.add_entry(entry):
        print(f"Added [{category}]: {content}")
    else:
        print("Write blocked by gate (threat scan or approval).")


@memory_app.command("remove")
def memory_remove(
    index: int = typer.Argument(..., help="Entry index (from `nia memory list`)"),
) -> None:
    """Remove a memory entry by index."""
    from niaharness.memory import MemoryStore
    from niaharness.memory.paths import get_project_memory_dir

    import os
    store_path = get_project_memory_dir(os.getcwd()) / "STORE.md"
    if not store_path.exists():
        print("No memory store found.")
        return
    store = MemoryStore(path=store_path)
    if store.remove_entry(index):
        print(f"Removed entry {index}.")
    else:
        print(f"Could not remove entry {index} (not found or blocked).")


@memory_app.command("drift")
def memory_drift() -> None:
    """Check if the memory store was modified externally."""
    from niaharness.memory import MemoryStore
    from niaharness.memory.paths import get_project_memory_dir

    import os
    store_path = get_project_memory_dir(os.getcwd()) / "STORE.md"
    if not store_path.exists():
        print("No memory store found.")
        return
    store = MemoryStore(path=store_path)
    report = store.detect_drift()
    if report.changed:
        print("DRIFT DETECTED — the memory store was modified externally.")
        if report.old_size is not None:
            print(f"  Old size: {report.old_size} bytes")
        print(f"  New size: {report.new_size} bytes")
        if report.old_mtime is not None:
            print(f"  Old mtime: {report.old_mtime:.0f}")
        print(f"  New mtime: {report.new_mtime:.0f}")
    else:
        print("No drift detected — the store is unchanged since last access.")


@memory_app.command("clear")
def memory_clear() -> None:
    """Remove all memory entries (requires confirmation)."""
    from niaharness.memory import MemoryStore
    from niaharness.memory.paths import get_project_memory_dir

    import os
    confirmed = typer.confirm(
        "This will remove ALL memory entries. Are you sure?",
        default=False,
    )
    if not confirmed:
        print("Cancelled.")
        return
    store_path = get_project_memory_dir(os.getcwd()) / "STORE.md"
    if not store_path.exists():
        print("No memory store found.")
        return
    store = MemoryStore(path=store_path)
    if store.clear():
        print("Cleared all memory entries.")
    else:
        print("Clear blocked by gate.")


# ---------------------------------------------------------------------------
# Main command
# ---------------------------------------------------------------------------

@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    # --- Session ---
    continue_session: bool = typer.Option(
        False,
        "--continue",
        "-c",
        help="Continue the most recent conversation in the current directory",
        rich_help_panel="Session",
    ),
    resume: str | None = typer.Option(
        None,
        "--resume",
        "-r",
        help="Resume a conversation by session ID, or open picker",
        rich_help_panel="Session",
    ),
    name: str | None = typer.Option(
        None,
        "--name",
        "-n",
        help="Set a display name for this session",
        rich_help_panel="Session",
    ),
    # --- Model & Effort ---
    model: str | None = typer.Option(
        None,
        "--model",
        "-m",
        help="Model alias (e.g. 'sonnet', 'opus') or full model ID",
        rich_help_panel="Model & Effort",
    ),
    effort: str | None = typer.Option(
        None,
        "--effort",
        help="Effort level for the session (low, medium, high, max)",
        rich_help_panel="Model & Effort",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Override verbose mode setting from config",
        rich_help_panel="Model & Effort",
    ),
    max_turns: int | None = typer.Option(
        None,
        "--max-turns",
        help="Maximum number of agentic turns (useful with --print)",
        rich_help_panel="Model & Effort",
    ),
    # --- Output ---
    print_mode: str | None = typer.Option(
        None,
        "--print",
        "-p",
        help="Print response and exit. Pass your prompt as the value: -p 'your prompt'",
        rich_help_panel="Output",
    ),
    output_format: str | None = typer.Option(
        None,
        "--output-format",
        help="Output format with --print: text (default), json, or stream-json",
        rich_help_panel="Output",
    ),
    # --- Permissions ---
    permission_mode: str | None = typer.Option(
        None,
        "--permission-mode",
        help="Permission mode: default, plan, or full_auto",
        rich_help_panel="Permissions",
    ),
    dangerously_skip_permissions: bool = typer.Option(
        False,
        "--dangerously-skip-permissions",
        help="Bypass all permission checks (only for sandboxed environments)",
        rich_help_panel="Permissions",
    ),
    allowed_tools: Optional[list[str]] = typer.Option(
        None,
        "--allowed-tools",
        help="Comma or space-separated list of tool names to allow",
        rich_help_panel="Permissions",
    ),
    disallowed_tools: Optional[list[str]] = typer.Option(
        None,
        "--disallowed-tools",
        help="Comma or space-separated list of tool names to deny",
        rich_help_panel="Permissions",
    ),
    # --- System & Context ---
    system_prompt: str | None = typer.Option(
        None,
        "--system-prompt",
        "-s",
        help="Override the default system prompt",
        rich_help_panel="System & Context",
    ),
    append_system_prompt: str | None = typer.Option(
        None,
        "--append-system-prompt",
        help="Append text to the default system prompt",
        rich_help_panel="System & Context",
    ),
    settings_file: str | None = typer.Option(
        None,
        "--settings",
        help="Path to a JSON settings file or inline JSON string",
        rich_help_panel="System & Context",
    ),
    base_url: str | None = typer.Option(
        None,
        "--base-url",
        help="Anthropic-compatible API base URL",
        rich_help_panel="System & Context",
    ),
    api_key: str | None = typer.Option(
        None,
        "--api-key",
        "-k",
        help="API key (overrides config and environment)",
        rich_help_panel="System & Context",
    ),
    bare: bool = typer.Option(
        False,
        "--bare",
        help="Minimal mode: skip hooks, plugins, MCP, and auto-discovery",
        rich_help_panel="System & Context",
    ),
    api_format: str | None = typer.Option(
        None,
        "--api-format",
        help="API format: 'anthropic' (default) or 'openai' (for DashScope, GitHub Models, etc.)",
        rich_help_panel="System & Context",
    ),
    provider: str | None = typer.Option(
        None,
        "--provider",
        "-P",
        help=(
            "LLM provider name (e.g. 'anthropic', 'openai', 'opencode', 'xai', "
            "'perplexity', 'openrouter', 'groq', 'deepseek', 'ollama', etc.). "
            "Routes through the ProviderRegistry (20 providers). "
            "Use --list-providers to see all options."
        ),
        rich_help_panel="System & Context",
    ),
    list_providers: bool = typer.Option(
        False,
        "--list-providers",
        help="List all available LLM providers and exit",
        rich_help_panel="System & Context",
    ),
    list_models: bool = typer.Option(
        False,
        "--list-models",
        help="Fetch and list all available models from configured providers, then exit",
        rich_help_panel="System & Context",
    ),
    # --- Advanced ---
    debug: bool = typer.Option(
        False,
        "--debug",
        "-d",
        help="Enable debug logging",
        rich_help_panel="Advanced",
    ),
    mcp_config: Optional[list[str]] = typer.Option(
        None,
        "--mcp-config",
        help="Load MCP servers from JSON files or strings",
        rich_help_panel="Advanced",
    ),
    cwd: str = typer.Option(
        str(Path.cwd()),
        "--cwd",
        help="Working directory for the session",
        hidden=True,
    ),
    backend_only: bool = typer.Option(
        False,
        "--backend-only",
        help="Run the structured backend host for the React terminal UI",
        hidden=True,
    ),
) -> None:
    """Start an interactive session or run a single prompt."""
    if ctx.invoked_subcommand is not None:
        return

    import asyncio

    # Handle --list-providers
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
            print(f"                  base_url: {cfg.auth.default_base_url}")
            print(f"                  default model: {cfg.auth.default_model}")
            print()
        raise typer.Exit(0)

    # Handle --list-models — fetch models from all configured providers
    if list_models:
        import asyncio as _asyncio
        from niaharness.providers.registry import ProviderRegistry

        registry = ProviderRegistry()
        registry._register_builtin_providers()
        registry._auto_detect_providers()

        configured = [
            name for name, state in registry._states.items()
            if state.configured or name == "ollama"
        ]
        if not configured:
            print(
                "No providers configured. Set an API key env var "
                "(e.g. ANTHROPIC_API_KEY, OPENAI_API_KEY, OPENCODE_API_KEY) "
                "and try again.",
                file=sys.stderr,
            )
            raise typer.Exit(1)

        print(f"\nFetching models from {len(configured)} configured provider(s)...\n")
        all_models = _asyncio.run(registry.fetch_all_models())

        total = 0
        for name in sorted(all_models.keys()):
            models = all_models[name]
            if not models:
                continue
            print(f"=== {name} ({len(models)} models) ===")
            for m in models:
                marker = " *" if m.get("active") else "  "
                ctx_window = m.get("context_window", 0)
                ctx_str = f"{ctx_window // 1000}K" if ctx_window >= 1000 else str(ctx_window)
                print(f"{marker} {m['id']:<50} ({ctx_str} ctx)")
            print()
            total += len(models)

        print(f"Total: {total} models from {len(configured)} provider(s).")
        print("\nUse --provider <name> --model <id> to select one.")
        raise typer.Exit(0)

    if dangerously_skip_permissions:
        permission_mode = "full_auto"

    from niaharness.ui.app import run_print_mode, run_repl

    # Handle --continue and --resume flags
    if continue_session or resume is not None:
        from niaharness.services.session_storage import (
            list_session_snapshots,
            load_session_by_id,
            load_session_snapshot,
        )

        session_data = None
        if continue_session:
            session_data = load_session_snapshot(cwd)
            if session_data is None:
                print("No previous session found in this directory.", file=sys.stderr)
                raise typer.Exit(1)
            print(f"Continuing session: {session_data.get('summary', '(untitled)')[:60]}")
        elif resume == "" or resume is None:
            # --resume with no value: show session picker
            sessions = list_session_snapshots(cwd, limit=10)
            if not sessions:
                print("No saved sessions found.", file=sys.stderr)
                raise typer.Exit(1)
            print("Saved sessions:")
            for i, s in enumerate(sessions, 1):
                print(f"  {i}. [{s['session_id']}] {s.get('summary', '?')[:50]} ({s['message_count']} msgs)")
            choice = typer.prompt("Enter session number or ID")
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(sessions):
                    session_data = load_session_by_id(cwd, sessions[idx]["session_id"])
                else:
                    print("Invalid selection.", file=sys.stderr)
                    raise typer.Exit(1)
            except ValueError:
                session_data = load_session_by_id(cwd, choice)
            if session_data is None:
                print(f"Session not found: {choice}", file=sys.stderr)
                raise typer.Exit(1)
        else:
            session_data = load_session_by_id(cwd, resume)
            if session_data is None:
                print(f"Session not found: {resume}", file=sys.stderr)
                raise typer.Exit(1)

        # Pass restored session to the REPL
        asyncio.run(
            run_repl(
                prompt=None,
                cwd=cwd,
                model=session_data.get("model") or model,
                backend_only=backend_only,
                base_url=base_url,
                system_prompt=session_data.get("system_prompt") or system_prompt,
                api_key=api_key,
                restore_messages=session_data.get("messages"),
            )
        )
        return

    if print_mode is not None:
        prompt = print_mode.strip()
        if not prompt:
            print("Error: -p/--print requires a prompt value, e.g. -p 'your prompt'", file=sys.stderr)
            raise typer.Exit(1)

        # If --provider was given, resolve credentials from the ProviderRegistry.
        resolved_api_key = api_key
        resolved_base_url = base_url
        resolved_model = model
        resolved_api_format = api_format
        if provider:
            from niaharness.providers.registry import ProviderRegistry as _PR

            _reg = _PR()
            _reg._register_builtin_providers()
            _prov = _reg.get_provider(provider)
            if _prov is None:
                print(
                    f"Unknown provider: {provider!r}. "
                    f"Use --list-providers to see options.",
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
            # All providers except Anthropic use the OpenAI-compatible format.
            resolved_api_format = api_format or (
                "anthropic" if provider == "anthropic" else "openai"
            )
            if not resolved_api_key:
                env_hint = _cfg.auth.api_key_env_vars[0] if _cfg.auth.api_key_env_vars else "(none)"
                print(
                    f"Provider {provider!r} requires an API key. Set {env_hint} env var "
                    f"or use --api-key.",
                    file=sys.stderr,
                )
                raise typer.Exit(1)

        asyncio.run(
            run_print_mode(
                prompt=prompt,
                output_format=output_format or "text",
                cwd=cwd,
                model=resolved_model,
                base_url=resolved_base_url,
                system_prompt=system_prompt,
                append_system_prompt=append_system_prompt,
                api_key=resolved_api_key,
                api_format=resolved_api_format,
                permission_mode=permission_mode,
                max_turns=max_turns,
            )
        )
        return

    # Interactive REPL
    resolved_api_key = api_key
    resolved_base_url = base_url
    resolved_model = model
    resolved_api_format = api_format
    if provider:
        from niaharness.providers.registry import ProviderRegistry as _PR

        _reg = _PR()
        _reg._register_builtin_providers()
        _prov = _reg.get_provider(provider)
        if _prov is None:
            print(
                f"Unknown provider: {provider!r}. "
                f"Use --list-providers to see options.",
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
        resolved_api_format = api_format or (
            "anthropic" if provider == "anthropic" else "openai"
        )
        if not resolved_api_key:
            env_hint = _cfg.auth.api_key_env_vars[0] if _cfg.auth.api_key_env_vars else "(none)"
            print(
                f"Provider {provider!r} requires an API key. Set {env_hint} env var "
                f"or use --api-key.",
                file=sys.stderr,
            )
            raise typer.Exit(1)

    asyncio.run(
        run_repl(
            prompt=None,
            cwd=cwd,
            model=resolved_model,
            backend_only=backend_only,
            base_url=resolved_base_url,
            system_prompt=system_prompt,
            api_key=resolved_api_key,
            api_format=resolved_api_format,
        )
    )


if __name__ == "__main__":
    app()
