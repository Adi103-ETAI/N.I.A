"""Slash command registry."""

from __future__ import annotations

import importlib.metadata
import json
import shutil
import subprocess
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Awaitable, Callable, Literal, get_args

import pyperclip

from niaharness.config.paths import (
    get_config_dir,
    get_data_dir,
    get_feedback_log_path,
    get_project_config_dir,
    get_project_issue_file,
    get_project_pr_comments_file,
)
from niaharness.bridge import get_bridge_manager
from niaharness.bridge.types import WorkSecret
from niaharness.bridge.work_secret import build_sdk_url, decode_work_secret, encode_work_secret
from niaharness.api.provider import auth_status, detect_provider
from niaharness.config.settings import Settings, load_settings, save_settings
from niaharness.engine.messages import ConversationMessage
from niaharness.engine.query_engine import QueryEngine
from niaharness.memory import (
    add_memory_entry,
    get_memory_entrypoint,
    get_project_memory_dir,
    list_memory_files,
    remove_memory_entry,
)
from niaharness.output_styles import load_output_styles
from niaharness.permissions import PermissionChecker, PermissionMode
from niaharness.plugins import load_plugins
from niaharness.prompts import build_runtime_system_prompt
from niaharness.plugins.installer import install_plugin_from_path, uninstall_plugin
from niaharness.services import (
    compact_messages,
    estimate_conversation_tokens,
    export_session_markdown,
    save_session_snapshot,
    summarize_messages,
)
from niaharness.services.session_storage import get_project_session_dir, load_session_snapshot
from niaharness.tools.skills_loader import load_skill_registry
from niaharness.tasks import get_task_manager

if TYPE_CHECKING:
    from niaharness.state import AppStateStore
    from niaharness.tools.base import ToolRegistry


@dataclass
class CommandResult:
    """Result returned by a slash command."""

    message: str | None = None
    should_exit: bool = False
    clear_screen: bool = False
    replay_messages: list | None = None  # ConversationMessage list to replay in TUI


@dataclass
class CommandContext:
    """Context available to command handlers."""

    engine: QueryEngine
    hooks_summary: str = ""
    mcp_summary: str = ""
    plugin_summary: str = ""
    cwd: str = "."
    tool_registry: ToolRegistry | None = None
    app_state: AppStateStore | None = None


CommandHandler = Callable[[str, CommandContext], Awaitable[CommandResult]]


@dataclass
class SlashCommand:
    """Definition of a slash command."""

    name: str
    description: str
    handler: CommandHandler


class CommandRegistry:
    """Map slash commands to handlers."""

    def __init__(self) -> None:
        self._commands: dict[str, SlashCommand] = {}

    def register(self, command: SlashCommand) -> None:
        """Register a command."""
        self._commands[command.name] = command

    def lookup(self, raw_input: str) -> tuple[SlashCommand, str] | None:
        """Parse a slash command and return its handler plus raw args.

        If the command isn't a registered slash command, checks whether it
        matches a skill name (P0 #5: skill slash commands). Skill commands
        are resolved dynamically by scanning the skills directory.
        """
        if not raw_input.startswith("/"):
            return None
        name, _, args = raw_input[1:].partition(" ")
        command = self._commands.get(name)
        if command is not None:
            return command, args.strip()

        # P0 #5: Check if this is a skill slash command (/<skill-name>).
        skill_result = self._lookup_skill_command(name, args.strip())
        if skill_result is not None:
            return skill_result

        return None

    def _lookup_skill_command(self, name: str, args: str) -> tuple[SlashCommand, str] | None:
        """Check if ``name`` matches a skill and return a synthetic command.

        Scans the user skills directory + bundled skills for a matching name.
        If found, returns a SlashCommand whose handler loads the skill content
        and injects it as a user message (preserves prompt cache).

        Adapted from Hermes Agent's agent/skill_commands.py.
        """
        # Normalize: hyphens and underscores are interchangeable.
        normalized = name.lower().replace("_", "-")

        # Scan skills.
        try:
            from niaharness.tools.skills_loader import load_skill_registry

            registry = load_skill_registry()
            # Try exact, normalized, and title-case matches.
            skill = (
                registry.get(normalized)
                or registry.get(name)
                or registry.get(name.title())
            )
            if skill is None:
                return None

            # Build a synthetic slash command that injects the skill content.
            async def _skill_handler(_args: str, context: CommandContext) -> CommandResult:
                activation_note = (
                    f'[IMPORTANT: The user has invoked the "{skill.name}" skill, '
                    "indicating they want you to follow its instructions. "
                    "The full skill content is loaded below.]"
                )
                user_instruction = f"\n\nUser instruction: {_args}" if _args else ""

                # Enumerate supporting files (adapted from reference _build_skill_message).
                support_files_hint = ""
                if skill.path:
                    from pathlib import Path
                    from niaharness.tools.skill_utils import SKILL_SUPPORT_DIRS

                    skill_dir = Path(skill.path).parent
                    linked = []
                    for support_dir_name in sorted(SKILL_SUPPORT_DIRS):
                        support_dir = skill_dir / support_dir_name
                        if support_dir.is_dir():
                            for f in sorted(support_dir.rglob("*")):
                                if f.is_file() and not f.name.startswith(".") and f.suffix != ".pyc":
                                    rel = f.relative_to(skill_dir)
                                    linked.append(str(rel))
                    if linked:
                        support_files_hint = "\n\n[This skill has supporting files. Use skill(action=\"view\", name=\"" + skill.name + "\", file_path=\"<path>\") to read them:]\n"
                        for lf in linked:
                            support_files_hint += f"  {lf}\n"
                    if skill_dir.exists():
                        support_files_hint += f"\n[Skill directory: {skill_dir}]"

                # Inject as a user message (preserves prompt cache).
                from niaharness.engine.messages import ConversationMessage, TextBlock

                context.engine._messages.append(
                    ConversationMessage(
                        role="user",
                        content=[TextBlock(text=f"{activation_note}\n\n{skill.content}{support_files_hint}{user_instruction}")],
                    )
                )
                return CommandResult(
                    message=f"Loaded skill '{skill.name}'. The skill instructions are now in the conversation.",
                )

            return SlashCommand(
                name=normalized,
                description=f"Invoke the {skill.name} skill",
                handler=_skill_handler,
            ), args
        except Exception:
            return None

    def help_text(self) -> str:
        """Return a formatted summary of all registered commands."""
        lines = ["Available commands:"]
        for command in sorted(self._commands.values(), key=lambda item: item.name):
            lines.append(f"/{command.name:<12} {command.description}")
        return "\n".join(lines)

    def list_commands(self) -> list[SlashCommand]:
        """Return commands in registration order."""
        return list(self._commands.values())


def _run_git_command(cwd: str, *args: str) -> tuple[bool, str]:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return False, "git is not installed."
    output = (completed.stdout or completed.stderr).strip()
    if completed.returncode != 0:
        return False, output or f"git {' '.join(args)} failed"
    return True, output


def _copy_to_clipboard(text: str) -> tuple[bool, str]:
    try:
        pyperclip.copy(text)
        return True, "clipboard"
    except Exception:
        for command in (["pbcopy"], ["wl-copy"], ["xclip", "-selection", "clipboard"], ["xsel", "--clipboard"]):
            try:
                subprocess.run(command, input=text, text=True, check=True, capture_output=True)
                return True, "clipboard"
            except Exception:
                continue
    fallback = get_data_dir() / "last_copy.txt"
    fallback.write_text(text, encoding="utf-8")
    return False, str(fallback)


def _last_message_text(messages: list[ConversationMessage]) -> str:
    for message in reversed(messages):
        if message.text.strip():
            return message.text.strip()
    return ""


def _rewind_turns(messages: list[ConversationMessage], turns: int) -> list[ConversationMessage]:
    updated = list(messages)
    for _ in range(max(0, turns)):
        if not updated:
            break
        while updated:
            popped = updated.pop()
            if popped.role == "user" and popped.text.strip():
                break
    return updated


def _coerce_setting_value(settings: Settings, key: str, raw: str):
    field = Settings.model_fields.get(key)
    if field is None:
        raise KeyError(key)
    annotation = field.annotation
    if annotation is bool:
        lowered = raw.lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"Invalid boolean value for {key}: {raw}")
    if annotation is int:
        return int(raw)
    if annotation is str:
        return raw
    if annotation is Literal or getattr(annotation, "__origin__", None) is Literal:
        allowed = get_args(annotation)
        if raw not in allowed:
            raise ValueError(f"Invalid value for {key}: {raw}")
        return raw
    return raw


def create_default_command_registry() -> CommandRegistry:
    """Create the built-in command registry."""
    registry = CommandRegistry()

    async def _help_handler(_: str, context: CommandContext) -> CommandResult:
        del context
        return CommandResult(message=registry.help_text())

    async def _exit_handler(_: str, context: CommandContext) -> CommandResult:
        del context
        return CommandResult(should_exit=True)

    async def _clear_handler(_: str, context: CommandContext) -> CommandResult:
        context.engine.clear()
        return CommandResult(message="Conversation cleared.", clear_screen=True)

    async def _status_handler(_: str, context: CommandContext) -> CommandResult:
        usage = context.engine.total_usage
        state = context.app_state.get() if context.app_state is not None else None
        return CommandResult(
            message=(
                f"Messages: {len(context.engine.messages)}\n"
                f"Usage: input={usage.input_tokens} output={usage.output_tokens}\n"
                f"Effort: {state.effort if state is not None else load_settings().effort}\n"
                f"Passes: {state.passes if state is not None else load_settings().passes}"
            )
        )

    async def _version_handler(_: str, context: CommandContext) -> CommandResult:
        del context
        try:
            version = importlib.metadata.version("niaharness")
        except importlib.metadata.PackageNotFoundError:
            version = "0.1.0"
        return CommandResult(message=f"N.I.A {version}")

    async def _context_handler(_: str, context: CommandContext) -> CommandResult:
        settings = load_settings()
        prompt = build_runtime_system_prompt(settings, cwd=context.cwd)
        return CommandResult(message=prompt)

    async def _summary_handler(args: str, context: CommandContext) -> CommandResult:
        max_messages = 8
        if args:
            try:
                max_messages = max(1, int(args))
            except ValueError:
                return CommandResult(message="Usage: /summary [MAX_MESSAGES]")
        summary = summarize_messages(context.engine.messages, max_messages=max_messages)
        return CommandResult(message=summary or "No conversation content to summarize.")

    async def _compact_handler(args: str, context: CommandContext) -> CommandResult:
        preserve_recent = 6
        if args:
            try:
                preserve_recent = max(1, int(args))
            except ValueError:
                return CommandResult(message="Usage: /compact [PRESERVE_RECENT]")
        before = len(context.engine.messages)
        compacted = compact_messages(context.engine.messages, preserve_recent=preserve_recent)
        context.engine.load_messages(compacted)
        return CommandResult(
            message=f"Compacted conversation from {before} messages to {len(compacted)}."
        )

    async def _usage_handler(_: str, context: CommandContext) -> CommandResult:
        usage = context.engine.total_usage
        estimated = estimate_conversation_tokens(context.engine.messages)
        return CommandResult(
            message=(
                f"Actual usage: input={usage.input_tokens} output={usage.output_tokens}\n"
                f"Estimated conversation tokens: {estimated}\n"
                f"Messages: {len(context.engine.messages)}"
            )
        )

    async def _cost_handler(_: str, context: CommandContext) -> CommandResult:
        usage = context.engine.total_usage
        model = context.app_state.get().model if context.app_state is not None else load_settings().model
        estimated_cost = "unavailable"
        if model.startswith("claude-3-5-sonnet"):
            estimated = (usage.input_tokens * 3.0 + usage.output_tokens * 15.0) / 1_000_000
            estimated_cost = f"${estimated:.4f} (estimated)"
        elif model.startswith("claude-3-7-sonnet"):
            estimated = (usage.input_tokens * 3.0 + usage.output_tokens * 15.0) / 1_000_000
            estimated_cost = f"${estimated:.4f} (estimated)"
        elif model.startswith("claude-3-opus"):
            estimated = (usage.input_tokens * 15.0 + usage.output_tokens * 75.0) / 1_000_000
            estimated_cost = f"${estimated:.4f} (estimated)"
        return CommandResult(
            message=(
                f"Model: {model}\n"
                f"Input tokens: {usage.input_tokens}\n"
                f"Output tokens: {usage.output_tokens}\n"
                f"Total tokens: {usage.total_tokens}\n"
                f"Estimated cost: {estimated_cost}"
            )
        )

    async def _stats_handler(_: str, context: CommandContext) -> CommandResult:
        settings = load_settings()
        memory_count = len(list_memory_files(context.cwd))
        task_count = len(get_task_manager().list_tasks())
        tool_count = len(context.tool_registry.list_tools()) if context.tool_registry is not None else 0
        style = settings.output_style
        if context.app_state is not None:
            state = context.app_state.get()
            style = state.output_style
        return CommandResult(
            message=(
                "Session stats:\n"
                f"- messages: {len(context.engine.messages)}\n"
                f"- estimated_tokens: {estimate_conversation_tokens(context.engine.messages)}\n"
                f"- tools: {tool_count}\n"
                f"- memory_files: {memory_count}\n"
                f"- background_tasks: {task_count}\n"
                f"- output_style: {style}"
            )
        )

    async def _memory_handler(args: str, context: CommandContext) -> CommandResult:
        tokens = args.split(maxsplit=1)
        if not tokens:
            memory_dir = get_project_memory_dir(context.cwd)
            entrypoint = get_memory_entrypoint(context.cwd)
            return CommandResult(
                message=f"Memory directory: {memory_dir}\nEntrypoint: {entrypoint}"
            )
        action = tokens[0]
        rest = tokens[1] if len(tokens) == 2 else ""
        if action == "list":
            memory_files = list_memory_files(context.cwd)
            if not memory_files:
                return CommandResult(message="No memory files.")
            return CommandResult(message="\n".join(path.name for path in memory_files))
        if action == "show" and rest:
            memory_dir = get_project_memory_dir(context.cwd)
            path = memory_dir / rest
            if not path.exists():
                path = memory_dir / f"{rest}.md"
            if not path.exists():
                return CommandResult(message=f"Memory entry not found: {rest}")
            return CommandResult(message=path.read_text(encoding="utf-8"))
        if action == "add" and rest:
            title, separator, content = rest.partition("::")
            if not separator or not title.strip() or not content.strip():
                return CommandResult(message="Usage: /memory add TITLE :: CONTENT")
            path = add_memory_entry(context.cwd, title.strip(), content.strip())
            return CommandResult(message=f"Added memory entry {path.name}")
        if action == "remove" and rest:
            if remove_memory_entry(context.cwd, rest.strip()):
                return CommandResult(message=f"Removed memory entry {rest.strip()}")
            return CommandResult(message=f"Memory entry not found: {rest.strip()}")
        return CommandResult(message="Usage: /memory [list|show NAME|add TITLE :: CONTENT|remove NAME]")

    async def _hooks_handler(_: str, context: CommandContext) -> CommandResult:
        return CommandResult(message=context.hooks_summary or "No hooks configured.")

    async def _resume_handler(args: str, context: CommandContext) -> CommandResult:
        from niaharness.services.session_storage import list_session_snapshots, load_session_by_id

        tokens = args.strip().split()

        # /resume <session_id> — load a specific session
        if tokens:
            sid = tokens[0]
            snapshot = load_session_by_id(context.cwd, sid)
            if snapshot is None:
                return CommandResult(message=f"Session not found: {sid}")
            messages = [
                ConversationMessage.model_validate(item)
                for item in snapshot.get("messages", [])
            ]
            context.engine.load_messages(messages)
            summary = snapshot.get("summary", "")[:60]
            return CommandResult(
                message=f"Restored {len(messages)} messages from session {sid}"
                + (f" ({summary})" if summary else ""),
                replay_messages=messages,
            )

        # /resume — list sessions (for the TUI to show a picker)
        sessions = list_session_snapshots(context.cwd, limit=10)
        if not sessions:
            # Fall back to latest.json
            snapshot = load_session_snapshot(context.cwd)
            if snapshot is None:
                return CommandResult(message="No saved sessions found for this project.")
            messages = [
                ConversationMessage.model_validate(item)
                for item in snapshot.get("messages", [])
            ]
            context.engine.load_messages(messages)
            return CommandResult(
                message=f"Restored {len(messages)} messages from the latest session.",
                replay_messages=messages,
            )

        # Format session list for display / picker
        import time
        lines = ["Saved sessions:"]
        for s in sessions:
            ts = time.strftime("%m/%d %H:%M", time.localtime(s["created_at"]))
            summary = s["summary"][:50] or "(no summary)"
            lines.append(f"  {s['session_id']}  {ts}  {s['message_count']}msg  {summary}")
        lines.append("")
        lines.append("Use /resume <session_id> to restore a specific session.")
        return CommandResult(message="\n".join(lines))

    async def _export_handler(_: str, context: CommandContext) -> CommandResult:
        path = export_session_markdown(cwd=context.cwd, messages=context.engine.messages)
        return CommandResult(message=f"Exported transcript to {path}")

    async def _share_handler(_: str, context: CommandContext) -> CommandResult:
        path = export_session_markdown(cwd=context.cwd, messages=context.engine.messages)
        return CommandResult(message=f"Created shareable transcript snapshot at {path}")

    async def _copy_handler(args: str, context: CommandContext) -> CommandResult:
        text = args.strip() or _last_message_text(context.engine.messages)
        if not text:
            return CommandResult(message="Nothing to copy.")
        copied, target = _copy_to_clipboard(text)
        if copied:
            return CommandResult(message=f"Copied {len(text)} characters to the clipboard.")
        return CommandResult(message=f"Clipboard unavailable. Saved copied text to {target}")

    async def _session_handler(args: str, context: CommandContext) -> CommandResult:
        session_dir = get_project_session_dir(context.cwd)
        tokens = args.split()
        if not tokens or tokens[0] == "show":
            latest = session_dir / "latest.json"
            transcript = session_dir / "transcript.md"
            lines = [
                f"Session directory: {session_dir}",
                f"Latest snapshot: {'present' if latest.exists() else 'missing'}",
                f"Transcript export: {'present' if transcript.exists() else 'missing'}",
                f"Message count: {len(context.engine.messages)}",
            ]
            return CommandResult(message="\n".join(lines))
        if tokens[0] == "ls":
            files = sorted(path.name for path in session_dir.iterdir())
            return CommandResult(message="\n".join(files) if files else "(empty)")
        if tokens[0] == "path":
            return CommandResult(message=str(session_dir))
        if tokens[0] == "tag" and len(tokens) == 2:
            safe_name = "".join(character for character in tokens[1] if character.isalnum() or character in {"-", "_"})
            if not safe_name:
                return CommandResult(message="Usage: /session tag NAME")
            snapshot_path = save_session_snapshot(
                cwd=context.cwd,
                model=context.app_state.get().model if context.app_state is not None else load_settings().model,
                system_prompt=build_runtime_system_prompt(load_settings(), cwd=context.cwd),
                messages=context.engine.messages,
                usage=context.engine.total_usage,
            )
            export_path = export_session_markdown(cwd=context.cwd, messages=context.engine.messages)
            tagged_json = session_dir / f"{safe_name}.json"
            tagged_md = session_dir / f"{safe_name}.md"
            shutil.copy2(snapshot_path, tagged_json)
            shutil.copy2(export_path, tagged_md)
            return CommandResult(message=f"Tagged session as {safe_name}:\n- {tagged_json}\n- {tagged_md}")
        if tokens[0] == "clear":
            if session_dir.exists():
                shutil.rmtree(session_dir)
            session_dir.mkdir(parents=True, exist_ok=True)
            return CommandResult(message=f"Cleared session storage at {session_dir}")
        return CommandResult(message="Usage: /session [show|ls|path|tag NAME|clear]")

    async def _rewind_handler(args: str, context: CommandContext) -> CommandResult:
        turns = 1
        if args.strip():
            try:
                turns = max(1, int(args.strip()))
            except ValueError:
                return CommandResult(message="Usage: /rewind [TURNS]")
        before = len(context.engine.messages)
        updated = _rewind_turns(context.engine.messages, turns)
        context.engine.load_messages(updated)
        removed = before - len(updated)
        return CommandResult(message=f"Rewound {turns} turn(s); removed {removed} message(s).")

    async def _tag_handler(args: str, context: CommandContext) -> CommandResult:
        name = args.strip()
        if not name:
            return CommandResult(message="Usage: /tag NAME")
        return await _session_handler(f"tag {name}", context)

    async def _files_handler(args: str, context: CommandContext) -> CommandResult:
        raw = args.strip()
        root = Path(context.cwd)
        max_items = 30
        tokens = raw.split(maxsplit=1)
        if tokens and tokens[0] == "dirs":
            dirs = [
                path
                for path in sorted(root.rglob("*"))
                if path.is_dir() and ".git" not in path.parts and ".venv" not in path.parts
            ]
            lines = [str(path.relative_to(root)) for path in dirs[:max_items]]
            if len(dirs) > max_items:
                lines.append(f"... {len(dirs) - max_items} more")
            return CommandResult(message="\n".join(lines) if lines else "(no directories)")
        if tokens and tokens[0].isdigit():
            max_items = max(1, min(int(tokens[0]), 200))
            raw = tokens[1] if len(tokens) == 2 else ""
        needle = raw.lower()
        files = [
            path
            for path in sorted(root.rglob("*"))
            if path.is_file() and ".git" not in path.parts and ".venv" not in path.parts
        ]
        if needle:
            files = [path for path in files if needle in str(path.relative_to(root)).lower()]
        lines = [str(path.relative_to(root)) for path in files[:max_items]]
        if len(files) > max_items:
            lines.append(f"... {len(files) - max_items} more")
        return CommandResult(
            message="\n".join(lines) if lines else "(no matching files)"
        )

    async def _agents_handler(args: str, context: CommandContext) -> CommandResult:
        tokens = args.split(maxsplit=1)
        if tokens and tokens[0] == "show" and len(tokens) == 2:
            task = get_task_manager().get_task(tokens[1])
            if task is None or task.type not in {"local_agent", "remote_agent", "in_process_teammate"}:
                return CommandResult(message=f"No agent found with ID: {tokens[1]}")
            output = get_task_manager().read_task_output(task.id)
            return CommandResult(
                message=(
                    f"{task.id} {task.type} {task.status} {task.description}\n"
                    f"metadata={task.metadata}\n"
                    f"output:\n{output or '(no output)'}"
                )
            )
        tasks = [
            task
            for task in get_task_manager().list_tasks()
            if task.type in {"local_agent", "remote_agent", "in_process_teammate"}
        ]
        if not tasks:
            return CommandResult(message="No active or recorded agents.")
        lines = [
            f"{task.id} {task.type} {task.status} {task.description}"
            for task in tasks
        ]
        return CommandResult(message="\n".join(lines))

    async def _init_handler(args: str, context: CommandContext) -> CommandResult:
        del args
        project_dir = get_project_config_dir(context.cwd)
        created: list[str] = []

        claudemd = Path(context.cwd) / "CLAUDE.md"
        if not claudemd.exists():
            claudemd.write_text(
                "# Project Instructions\n\n"
                "- Use NiaHarness tools deliberately.\n"
                "- Keep changes minimal and verify with tests when possible.\n",
                encoding="utf-8",
            )
            created.append(str(claudemd.relative_to(Path(context.cwd))))

        for relative, content in (
            (
                project_dir / "README.md",
                "# Project NiaHarness Config\n\nThis directory stores project-specific NiaHarness state.\n",
            ),
            (
                project_dir / "memory" / "MEMORY.md",
                "# Project Memory\n\nAdd reusable project knowledge here.\n",
            ),
            (
                project_dir / "plugins" / ".gitkeep",
                "",
            ),
            (
                project_dir / "skills" / ".gitkeep",
                "",
            ),
        ):
            relative.parent.mkdir(parents=True, exist_ok=True)
            if not relative.exists():
                relative.write_text(content, encoding="utf-8")
                created.append(str(relative.relative_to(Path(context.cwd))))

        if not created:
            return CommandResult(message="Project already initialized for NiaHarness.")
        return CommandResult(message="Initialized project files:\n" + "\n".join(f"- {item}" for item in created))

    async def _bridge_handler(args: str, context: CommandContext) -> CommandResult:
        tokens = args.split()
        if not tokens or tokens[0] == "show":
            sessions = get_bridge_manager().list_sessions()
            lines = [
                "Bridge summary:",
                "- backend host: available",
                f"- cwd: {context.cwd}",
                f"- sessions: {len(sessions)}",
                "- utilities: encode, decode, sdk, spawn, list, output, stop",
            ]
            return CommandResult(message="\n".join(lines))
        if tokens[0] == "encode" and len(tokens) == 3:
            encoded = encode_work_secret(
                WorkSecret(version=1, session_ingress_token=tokens[2], api_base_url=tokens[1])
            )
            return CommandResult(message=encoded)
        if tokens[0] == "decode" and len(tokens) == 2:
            secret = decode_work_secret(tokens[1])
            return CommandResult(message=json.dumps(secret.__dict__, indent=2))
        if tokens[0] == "sdk" and len(tokens) == 3:
            return CommandResult(message=build_sdk_url(tokens[1], tokens[2]))
        if tokens[0] == "spawn" and len(tokens) >= 2:
            command = args[len("spawn ") :]
            handle = await get_bridge_manager().spawn(
                session_id=f"bridge-{datetime.now(timezone.utc).strftime('%H%M%S')}",
                command=command,
                cwd=context.cwd,
            )
            return CommandResult(
                message=f"Spawned bridge session {handle.session_id} pid={handle.process.pid}"
            )
        if tokens[0] == "list":
            sessions = get_bridge_manager().list_sessions()
            if not sessions:
                return CommandResult(message="No bridge sessions.")
            return CommandResult(
                message="\n".join(
                    f"{item.session_id} [{item.status}] pid={item.pid} {item.command}"
                    for item in sessions
                )
            )
        if tokens[0] == "output" and len(tokens) == 2:
            return CommandResult(message=get_bridge_manager().read_output(tokens[1]) or "(no output)")
        if tokens[0] == "stop" and len(tokens) == 2:
            try:
                await get_bridge_manager().stop(tokens[1])
            except ValueError as exc:
                return CommandResult(message=str(exc))
            return CommandResult(message=f"Stopped bridge session {tokens[1]}")
        return CommandResult(
            message="Usage: /bridge [show|encode API_BASE_URL TOKEN|decode SECRET|sdk API_BASE_URL SESSION_ID|spawn CMD|list|output SESSION_ID|stop SESSION_ID]"
        )

    async def _reload_plugins_handler(_: str, context: CommandContext) -> CommandResult:
        settings = load_settings()
        plugins = load_plugins(settings, context.cwd)
        if not plugins:
            return CommandResult(message="No plugins discovered.")
        lines = ["Reloaded plugins:"]
        for plugin in plugins:
            state = "enabled" if plugin.enabled else "disabled"
            lines.append(f"- {plugin.manifest.name} [{state}]")
        return CommandResult(message="\n".join(lines))

    async def _skills_handler(args: str, context: CommandContext) -> CommandResult:
        del context
        # Subcommands: browse, search, install, uninstall
        parts = args.split(maxsplit=1) if args else []
        subcommand = parts[0].lower() if parts else ""
        sub_args = parts[1] if len(parts) > 1 else ""

        if subcommand == "browse":
            from niaharness.tools.skills_hub import list_optional_skills

            skills = list_optional_skills()
            if not skills:
                return CommandResult(message="No optional skills available.")
            installed = sum(1 for s in skills if s.installed)
            lines = [f"Optional skills ({len(skills)} total, {installed} installed):", ""]
            for s in sorted(skills, key=lambda x: (x.category, x.name)):
                status = "✓" if s.installed else " "
                lines.append(f"  [{status}] {s.category}/{s.name}: {s.description[:60]}")
            lines.append("")
            lines.append("Use: /skills install <name> to install.")
            return CommandResult(message="\n".join(lines))

        if subcommand == "search":
            from niaharness.tools.skills_hub import search_optional_skills

            if not sub_args:
                return CommandResult(message="Usage: /skills search <query>")
            results = search_optional_skills(sub_args)
            if not results:
                return CommandResult(message=f"No skills found matching '{sub_args}'.")
            lines = [f"Found {len(results)} skill(s):", ""]
            for s in results:
                status = "✓" if s.installed else " "
                lines.append(f"  [{status}] {s.category}/{s.name}: {s.description[:60]}")
            return CommandResult(message="\n".join(lines))

        if subcommand == "install":
            from niaharness.tools.skills_hub import install_skill

            if not sub_args:
                return CommandResult(message="Usage: /skills install <name>")
            success, message = install_skill(sub_args.strip())
            return CommandResult(message=message)

        if subcommand == "uninstall":
            from niaharness.tools.skills_hub import uninstall_skill

            if not sub_args:
                return CommandResult(message="Usage: /skills uninstall <name>")
            success, message = uninstall_skill(sub_args.strip())
            return CommandResult(message=message)

        # Default: list installed skills (or show a specific skill).
        skill_registry = load_skill_registry()
        if args:
            skill = skill_registry.get(args) or skill_registry.get(args.lower())
            if skill is None:
                return CommandResult(message=f"Skill not found: {args}")
            return CommandResult(message=skill.content)
        skills = skill_registry.list_skills()
        if not skills:
            return CommandResult(message="No skills available. Use /skills browse to see optional skills.")
        lines = ["Available skills:"]
        for skill in skills:
            source = f" [{skill.source}]"
            lines.append(f"- {skill.name}{source}: {skill.description}")
        lines.append("")
        lines.append("Use /skills browse to see optional skills, /skills install <name> to add.")
        return CommandResult(message="\n".join(lines))

    async def _config_handler(args: str, context: CommandContext) -> CommandResult:
        del context
        settings = load_settings()
        tokens = args.split(maxsplit=2)
        if not tokens or tokens[0] == "show":
            return CommandResult(message=settings.model_dump_json(indent=2))
        if tokens[0] == "set" and len(tokens) == 3:
            key, value = tokens[1], tokens[2]
            if key not in Settings.model_fields:
                return CommandResult(message=f"Unknown config key: {key}")
            try:
                coerced = _coerce_setting_value(settings, key, value)
            except ValueError as exc:
                return CommandResult(message=str(exc))
            setattr(settings, key, coerced)
            save_settings(settings)
            return CommandResult(message=f"Updated {key}")
        return CommandResult(message="Usage: /config [show|set KEY VALUE]")

    async def _login_handler(args: str, context: CommandContext) -> CommandResult:
        del context
        settings = load_settings()
        provider = detect_provider(settings)
        api_key = args.strip()
        if not api_key:
            masked = (
                f"{settings.api_key[:6]}...{settings.api_key[-4:]}"
                if settings.api_key
                else "(not configured)"
            )
            return CommandResult(
                message=(
                    f"Auth status:\n"
                    f"- provider: {provider.name}\n"
                    f"- auth_status: {auth_status(settings)}\n"
                    f"- base_url: {settings.base_url or '(default)'}\n"
                    f"- model: {settings.model}\n"
                    f"- api_key: {masked}\n"
                    "Usage: /login API_KEY"
                )
            )
        settings.api_key = api_key
        save_settings(settings)
        return CommandResult(message="Stored API key in ~/.niaharness/settings.json")

    async def _logout_handler(_: str, context: CommandContext) -> CommandResult:
        del context
        settings = load_settings()
        settings.api_key = ""
        save_settings(settings)
        return CommandResult(message="Cleared stored API key.")

    async def _oauth_handler(args: str, context: CommandContext) -> CommandResult:
        """Run the Anthropic PKCE OAuth login flow (Claude Pro/Max)."""
        del context
        try:
            from niaharness.providers.anthropic import OAuthTokenManager

            manager = OAuthTokenManager()

            if args.strip() == "status":
                token = manager.get_valid_token()
                if token:
                    return CommandResult(
                        message="OAuth: valid token found.\n"
                        f"Token file: {manager._token_path}"
                    )
                return CommandResult(message="OAuth: no valid token. Run /oauth login.")

            if args.strip() == "logout":
                manager.clear()
                return CommandResult(message="OAuth tokens cleared.")

            if args.strip() in ("login", ""):
                # Run the interactive PKCE OAuth flow.
                tokens = manager.login()
                if tokens:
                    return CommandResult(
                        message="OAuth login successful! Tokens saved.\n"
                        "You can now use Claude Pro/Max with NIA."
                    )
                return CommandResult(message="OAuth login failed.")

            return CommandResult(
                message="Usage:\n"
                "  /oauth login    Run PKCE OAuth flow (Claude Pro/Max)\n"
                "  /oauth status   Check if OAuth tokens are configured\n"
                "  /oauth logout   Clear stored OAuth tokens"
            )
        except Exception as exc:
            return CommandResult(message=f"OAuth error: {exc}")

    async def _feedback_handler(args: str, context: CommandContext) -> CommandResult:
        del context
        path = get_feedback_log_path()
        if not args.strip():
            return CommandResult(message=f"Feedback log: {path}\nUsage: /feedback TEXT")
        timestamp = datetime.now(timezone.utc).isoformat()
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"[{timestamp}] {args.strip()}\n")
        return CommandResult(message=f"Saved feedback to {path}")

    async def _onboarding_handler(_: str, context: CommandContext) -> CommandResult:
        del context
        return CommandResult(
            message=(
                "NiaHarness quickstart:\n"
                "1. Ask for a coding task in plain language.\n"
                "2. Use /help to inspect commands.\n"
                "3. Use /doctor to inspect runtime state.\n"
                "4. Use /tasks for background work and /memory for project memory.\n"
                "5. Use /login to store an API key if needed."
            )
        )

    async def _fast_handler(args: str, context: CommandContext) -> CommandResult:
        settings = load_settings()
        current = (
            context.app_state.get().fast_mode
            if context.app_state is not None
            else settings.fast_mode
        )
        action = args.strip() or "show"
        if action == "show":
            return CommandResult(message=f"Fast mode: {'on' if current else 'off'}")
        enabled = {"on": True, "off": False, "toggle": not current}.get(action)
        if enabled is None:
            return CommandResult(message="Usage: /fast [show|on|off|toggle]")
        settings.fast_mode = enabled
        save_settings(settings)
        if context.app_state is not None:
            context.app_state.set(fast_mode=enabled)
        return CommandResult(message=f"Fast mode {'enabled' if enabled else 'disabled'}.")

    async def _effort_handler(args: str, context: CommandContext) -> CommandResult:
        settings = load_settings()
        current = context.app_state.get().effort if context.app_state is not None else settings.effort
        value = args.strip() or "show"
        if value == "show":
            return CommandResult(message=f"Reasoning effort: {current}")
        if value not in {"low", "medium", "high"}:
            return CommandResult(message="Usage: /effort [show|low|medium|high]")
        settings.effort = value
        save_settings(settings)
        context.engine.set_system_prompt(build_runtime_system_prompt(settings, cwd=context.cwd))
        if context.app_state is not None:
            context.app_state.set(effort=value)
        return CommandResult(message=f"Reasoning effort set to {value}.")

    async def _passes_handler(args: str, context: CommandContext) -> CommandResult:
        settings = load_settings()
        current = context.app_state.get().passes if context.app_state is not None else settings.passes
        value = args.strip() or "show"
        if value == "show":
            return CommandResult(message=f"Passes: {current}")
        try:
            passes = max(1, min(int(value), 8))
        except ValueError:
            return CommandResult(message="Usage: /passes [show|COUNT]")
        settings.passes = passes
        save_settings(settings)
        context.engine.set_system_prompt(build_runtime_system_prompt(settings, cwd=context.cwd))
        if context.app_state is not None:
            context.app_state.set(passes=passes)
        return CommandResult(message=f"Pass count set to {passes}.")

    async def _issue_handler(args: str, context: CommandContext) -> CommandResult:
        path = get_project_issue_file(context.cwd)
        tokens = args.split(maxsplit=1)
        action = tokens[0] if tokens else "show"
        rest = tokens[1] if len(tokens) == 2 else ""
        if action == "show":
            if not path.exists():
                return CommandResult(message=f"No issue context. File path: {path}")
            return CommandResult(message=path.read_text(encoding="utf-8"))
        if action == "set" and rest:
            title, separator, body = rest.partition("::")
            if not separator or not title.strip() or not body.strip():
                return CommandResult(message="Usage: /issue set TITLE :: BODY")
            content = f"# {title.strip()}\n\n{body.strip()}\n"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            return CommandResult(message=f"Saved issue context to {path}")
        if action == "clear":
            if path.exists():
                path.unlink()
                return CommandResult(message="Cleared issue context.")
            return CommandResult(message="No issue context to clear.")
        return CommandResult(message="Usage: /issue [show|set TITLE :: BODY|clear]")

    async def _pr_comments_handler(args: str, context: CommandContext) -> CommandResult:
        path = get_project_pr_comments_file(context.cwd)
        tokens = args.split(maxsplit=1)
        action = tokens[0] if tokens else "show"
        rest = tokens[1] if len(tokens) == 2 else ""
        if action == "show":
            if not path.exists():
                return CommandResult(message=f"No PR comments context. File path: {path}")
            return CommandResult(message=path.read_text(encoding="utf-8"))
        if action == "add" and rest:
            location, separator, comment = rest.partition("::")
            if not separator or not location.strip() or not comment.strip():
                return CommandResult(message="Usage: /pr_comments add FILE[:LINE] :: COMMENT")
            existing = path.read_text(encoding="utf-8") if path.exists() else "# PR Comments\n"
            if not existing.endswith("\n"):
                existing += "\n"
            existing += f"- {location.strip()}: {comment.strip()}\n"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(existing, encoding="utf-8")
            return CommandResult(message=f"Added PR comment to {path}")
        if action == "clear":
            if path.exists():
                path.unlink()
                return CommandResult(message="Cleared PR comments context.")
            return CommandResult(message="No PR comments context to clear.")
        return CommandResult(message="Usage: /pr_comments [show|add FILE[:LINE] :: COMMENT|clear]")

    async def _mcp_handler(args: str, context: CommandContext) -> CommandResult:
        settings = load_settings()
        tokens = args.split()
        if tokens and tokens[0] == "auth" and len(tokens) >= 3:
            server_name = tokens[1]
            config = settings.mcp_servers.get(server_name)
            if config is None:
                return CommandResult(message=f"Unknown MCP server: {server_name}")

            if len(tokens) == 3:
                mode = "bearer"
                key = None
                value = tokens[2]
            elif len(tokens) == 4:
                mode = tokens[2]
                key = None
                value = tokens[3]
            elif len(tokens) == 5:
                mode = tokens[2]
                key = tokens[3]
                value = tokens[4]
            else:
                return CommandResult(
                    message="Usage: /mcp auth SERVER TOKEN | /mcp auth SERVER [bearer|env] VALUE | /mcp auth SERVER header KEY VALUE"
                )

            if hasattr(config, "headers"):
                if mode not in {"bearer", "header"}:
                    return CommandResult(message="HTTP/WS MCP auth supports bearer or header modes.")
                header_key = key or "Authorization"
                header_value = (
                    f"Bearer {value}" if mode == "bearer" and header_key == "Authorization" else value
                )
                headers = dict(getattr(config, "headers", {}) or {})
                headers[header_key] = header_value
                settings.mcp_servers[server_name] = config.model_copy(update={"headers": headers})
            elif hasattr(config, "env"):
                if mode not in {"bearer", "env"}:
                    return CommandResult(message="stdio MCP auth supports bearer or env modes.")
                env_key = key or "MCP_AUTH_TOKEN"
                env_value = f"Bearer {value}" if mode == "bearer" else value
                env = dict(getattr(config, "env", {}) or {})
                env[env_key] = env_value
                settings.mcp_servers[server_name] = config.model_copy(update={"env": env})
            else:
                return CommandResult(message=f"Server {server_name} does not support auth updates")
            save_settings(settings)
            return CommandResult(message=f"Saved MCP auth for {server_name}. Restart session to reconnect.")
        return CommandResult(message=context.mcp_summary or "No MCP servers configured.")

    async def _plugin_handler(args: str, context: CommandContext) -> CommandResult:
        settings = load_settings()
        tokens = args.split()
        if not tokens or tokens[0] == "list":
            return CommandResult(message=context.plugin_summary or "No plugins discovered.")
        if tokens[0] == "enable" and len(tokens) == 2:
            settings.enabled_plugins[tokens[1]] = True
            save_settings(settings)
            return CommandResult(message=f"Enabled plugin '{tokens[1]}'. Restart session to reload.")
        if tokens[0] == "disable" and len(tokens) == 2:
            settings.enabled_plugins[tokens[1]] = False
            save_settings(settings)
            return CommandResult(message=f"Disabled plugin '{tokens[1]}'. Restart session to reload.")
        if tokens[0] == "install" and len(tokens) == 2:
            path = install_plugin_from_path(tokens[1])
            return CommandResult(message=f"Installed plugin to {path}")
        if tokens[0] == "uninstall" and len(tokens) == 2:
            if uninstall_plugin(tokens[1]):
                return CommandResult(message=f"Uninstalled plugin '{tokens[1]}'")
            return CommandResult(message=f"Plugin '{tokens[1]}' not found")
        plugins = load_plugins(settings, context.cwd)
        if plugins:
            return CommandResult(message=context.plugin_summary)
        return CommandResult(message="Usage: /plugin [list|enable NAME|disable NAME|install PATH|uninstall NAME]")

    _MODE_LABELS = {"default": "Default", "plan": "Plan Mode", "full_auto": "Auto"}

    async def _permissions_handler(args: str, context: CommandContext) -> CommandResult:
        settings = load_settings()
        tokens = args.split()
        if not tokens or tokens[0] == "show":
            permission = settings.permission
            label = _MODE_LABELS.get(permission.mode.value, permission.mode.value)
            return CommandResult(
                message=(
                    f"Mode: {label}\n"
                    f"Allowed tools: {permission.allowed_tools}\n"
                    f"Denied tools: {permission.denied_tools}"
                )
            )
        if tokens[0] == "set" and len(tokens) == 2:
            settings.permission.mode = PermissionMode(tokens[1])
            save_settings(settings)
            context.engine.set_permission_checker(PermissionChecker(settings.permission))
            if context.app_state is not None:
                context.app_state.set(permission_mode=settings.permission.mode.value)
            label = _MODE_LABELS.get(tokens[1], tokens[1])
            return CommandResult(message=f"Permission mode set to {label}")
        return CommandResult(message="Usage: /permissions [show|set MODE]")

    async def _plan_handler(args: str, context: CommandContext) -> CommandResult:
        settings = load_settings()
        mode = args.strip() or "on"
        if mode in {"on", "enter"}:
            settings.permission.mode = PermissionMode.PLAN
            save_settings(settings)
            context.engine.set_permission_checker(PermissionChecker(settings.permission))
            if context.app_state is not None:
                context.app_state.set(permission_mode=settings.permission.mode.value)
            return CommandResult(message="Plan mode enabled.")
        if mode in {"off", "exit"}:
            settings.permission.mode = PermissionMode.DEFAULT
            save_settings(settings)
            context.engine.set_permission_checker(PermissionChecker(settings.permission))
            if context.app_state is not None:
                context.app_state.set(permission_mode=settings.permission.mode.value)
            return CommandResult(message="Plan mode disabled.")
        return CommandResult(message="Usage: /plan [on|off]")

    async def _model_handler(args: str, context: CommandContext) -> CommandResult:
        settings = load_settings()
        tokens = args.split(maxsplit=1)
        if not tokens or tokens[0] == "show":
            return CommandResult(message=f"Model: {settings.model}")
        if tokens[0] == "list":
            # List models from the active provider's API (or hardcoded defaults).
            from niaharness.providers.registry import ProviderRegistry

            registry = ProviderRegistry()
            registry._register_builtin_providers()
            registry._auto_detect_providers()

            # Find the provider whose base_url or api_format matches current settings.
            active_name = None
            for name, prov in registry._providers.items():
                cfg = prov.config
                try:
                    base = prov.resolve_base_url()
                    if settings.base_url and base and base.rstrip("/") == (settings.base_url or "").rstrip("/"):
                        active_name = name
                        break
                except Exception:
                    pass
            if not active_name:
                # Fall back to api_format heuristic.
                active_name = "anthropic" if settings.api_format == "anthropic" else "openai"

            prov = registry._providers.get(active_name)
            if not prov:
                return CommandResult(message=f"Active provider not found: {active_name}")

            import asyncio as _aio

            try:
                models = _aio.run(prov.fetch_models())
            except Exception as exc:
                models = prov.config.models

            lines = [f"Models from {active_name} ({len(models)} total):", ""]
            for m in models:
                marker = " *" if m.id == settings.model else "  "
                ctx_w = m.context_window
                ctx_str = f"{ctx_w // 1000}K" if ctx_w >= 1000 else str(ctx_w)
                lines.append(f"{marker} {m.id:<50} ({ctx_str} ctx)")
            lines.append("")
            lines.append(f"Active model: {settings.model}")
            lines.append("Use: /model set <id>")
            return CommandResult(message="\n".join(lines))
        if tokens[0] == "set" and len(tokens) == 2:
            settings.model = tokens[1]
            save_settings(settings)
            context.engine.set_model(tokens[1])
            if context.app_state is not None:
                context.app_state.set(model=tokens[1])
            return CommandResult(message=f"Model set to {tokens[1]}.")
        return CommandResult(message="Usage: /model [show|list|set MODEL]")

    async def _provider_handler(args: str, context: CommandContext) -> CommandResult:
        """Provider management slash command.

        Usage:
            /provider              — show current provider + model
            /provider list         — list all 20 providers with env vars
            /provider <name>       — switch to a provider (resolves credentials)
            /provider <name> <model> — switch to a provider + specific model
            /provider models       — fetch models from the active provider
        """
        settings = load_settings()
        tokens = args.split()

        # No args → show current.
        if not tokens:
            base = settings.base_url or "(default)"
            fmt = settings.api_format
            key_set = "yes" if settings.resolve_api_key() else "no"
            return CommandResult(
                message=(
                    f"Current provider config:\n"
                    f"  Model:    {settings.model}\n"
                    f"  Base URL: {base}\n"
                    f"  Format:   {fmt}\n"
                    f"  API key:  {key_set}\n"
                    f"\n"
                    f"Use /provider list to see all 20 providers.\n"
                    f"Use /provider <name> [model] to switch."
                )
            )

        # /provider list
        if tokens[0] == "list":
            from niaharness.providers.registry import ProviderRegistry

            registry = ProviderRegistry()
            registry._register_builtin_providers()
            lines = [f"Available providers ({len(registry._providers)}):", ""]
            for name in sorted(registry._providers.keys()):
                cfg = registry._providers[name].config
                env = cfg.auth.api_key_env_vars[0] if cfg.auth.api_key_env_vars else "(none)"
                # Check if configured (env var set).
                import os

                configured = "✓" if any(os.environ.get(v) for v in cfg.auth.api_key_env_vars) else " "
                lines.append(f"  [{configured}] {name:<15} {cfg.label:<22} key: {env}")
            lines.append("")
            lines.append("✓ = API key detected in environment.")
            lines.append("Use /provider <name> [model] to switch.")
            return CommandResult(message="\n".join(lines))

        # /provider models — fetch from active provider
        if tokens[0] == "models":
            from niaharness.providers.registry import ProviderRegistry

            registry = ProviderRegistry()
            registry._register_builtin_providers()
            registry._auto_detect_providers()
            # Find active provider by matching base_url.
            active_name = None
            for name, prov in registry._providers.items():
                try:
                    base = prov.resolve_base_url()
                    if settings.base_url and base and base.rstrip("/") == (settings.base_url or "").rstrip("/"):
                        active_name = name
                        break
                except Exception:
                    pass
            if not active_name:
                active_name = "anthropic" if settings.api_format == "anthropic" else "openai"
            prov = registry._providers.get(active_name)
            if not prov:
                return CommandResult(message=f"Provider not found: {active_name}", )
            import asyncio as _aio

            try:
                models = _aio.run(prov.fetch_models())
            except Exception as exc:
                models = prov.config.models
            lines = [f"Models from {active_name} ({len(models)}):", ""]
            for m in models:
                marker = " *" if m.id == settings.model else "  "
                lines.append(f"{marker} {m.id}")
            lines.append("")
            lines.append(f"Active: {settings.model}")
            lines.append("Use: /model set <id>")
            return CommandResult(message="\n".join(lines))

        # /provider <name> [model] — switch
        provider_name = tokens[0]
        model_override = tokens[1] if len(tokens) >= 2 else None

        from niaharness.providers.registry import ProviderRegistry

        registry = ProviderRegistry()
        registry._register_builtin_providers()
        prov = registry.get_provider(provider_name)
        if prov is None:
            return CommandResult(
                message=f"Unknown provider: {provider_name!r}. Use /provider list to see options."
            )
        cfg = prov.config
        try:
            api_key = prov.resolve_api_key()
        except Exception:
            api_key = ""
        if not api_key:
            env_hint = cfg.auth.api_key_env_vars[0] if cfg.auth.api_key_env_vars else "(none)"
            return CommandResult(
                message=(
                    f"Provider {provider_name!r} requires an API key. "
                    f"Set {env_hint} env var first, then restart the session."
                )
            )
        base_url = prov.resolve_base_url()
        model = model_override or cfg.auth.default_model
        api_format = "anthropic" if provider_name == "anthropic" else "openai"

        # Update settings.
        settings.api_key = api_key
        settings.base_url = base_url
        settings.model = model
        settings.api_format = api_format
        save_settings(settings)

        # Update engine + app state.
        context.engine.set_model(model)
        if context.app_state is not None:
            context.app_state.set(model=model, provider=provider_name, base_url=base_url or "")

        return CommandResult(
            message=(
                f"Switched to provider: {provider_name} ({cfg.label})\n"
                f"  Model:    {model}\n"
                f"  Base URL: {base_url}\n"
                f"  Format:   {api_format}\n"
                f"\n"
                f"Settings saved. Use /model list to see available models."
            )
        )

    async def _theme_handler(args: str, context: CommandContext) -> CommandResult:
        settings = load_settings()
        tokens = args.split(maxsplit=1)
        if not tokens or tokens[0] == "show":
            return CommandResult(message=f"Theme: {settings.theme}")
        if tokens[0] == "set" and len(tokens) == 2:
            settings.theme = tokens[1]
            save_settings(settings)
            if context.app_state is not None:
                context.app_state.set(theme=tokens[1])
            return CommandResult(message=f"Theme set to {tokens[1]}")
        return CommandResult(message="Usage: /theme [show|set THEME]")

    async def _output_style_handler(args: str, context: CommandContext) -> CommandResult:
        settings = load_settings()
        tokens = args.split(maxsplit=1)
        styles = load_output_styles()
        available = {style.name: style for style in styles}
        current = (
            context.app_state.get().output_style
            if context.app_state is not None
            else settings.output_style
        )
        if not tokens or tokens[0] == "show":
            return CommandResult(message=f"Output style: {current}")
        if tokens[0] == "list":
            return CommandResult(
                message="\n".join(f"{style.name} [{style.source}]" for style in styles)
            )
        if tokens[0] == "set" and len(tokens) == 2:
            if tokens[1] not in available:
                return CommandResult(message=f"Unknown output style: {tokens[1]}")
            settings.output_style = tokens[1]
            save_settings(settings)
            if context.app_state is not None:
                context.app_state.set(output_style=tokens[1])
            return CommandResult(message=f"Output style set to {tokens[1]}")
        return CommandResult(message="Usage: /output-style [show|list|set NAME]")

    async def _keybindings_handler(_: str, context: CommandContext) -> CommandResult:
        from niaharness.keybindings import get_keybindings_path, load_keybindings

        bindings = (
            context.app_state.get().keybindings
            if context.app_state is not None and context.app_state.get().keybindings
            else load_keybindings()
        )
        lines = [f"Keybindings file: {get_keybindings_path()}"]
        lines.extend(f"{key} -> {command}" for key, command in sorted(bindings.items()))
        return CommandResult(message="\n".join(lines))

    async def _vim_handler(args: str, context: CommandContext) -> CommandResult:
        settings = load_settings()
        current = (
            context.app_state.get().vim_enabled
            if context.app_state is not None
            else settings.vim_mode
        )
        action = args.strip() or "show"
        if action == "show":
            return CommandResult(message=f"Vim mode: {'on' if current else 'off'}")
        enabled = {"on": True, "off": False, "toggle": not current}.get(action)
        if enabled is None:
            return CommandResult(message="Usage: /vim [show|on|off|toggle]")
        settings.vim_mode = enabled
        save_settings(settings)
        if context.app_state is not None:
            context.app_state.set(vim_enabled=enabled)
        return CommandResult(message=f"Vim mode {'enabled' if enabled else 'disabled'}.")

    async def _voice_handler(args: str, context: CommandContext) -> CommandResult:
        from niaharness.voice import extract_keyterms, inspect_voice_capabilities

        settings = load_settings()
        diagnostics = inspect_voice_capabilities(detect_provider(settings))
        current = (
            context.app_state.get().voice_enabled
            if context.app_state is not None
            else settings.voice_mode
        )
        tokens = args.split(maxsplit=1)
        if not tokens or tokens[0] == "show":
            return CommandResult(
                message=(
                    f"Voice mode: {'on' if current else 'off'}\n"
                    f"Available: {'yes' if diagnostics.available else 'no'}\n"
                    f"Recorder: {diagnostics.recorder or '(none)'}\n"
                    f"Reason: {diagnostics.reason}"
                )
            )
        if tokens[0] == "keyterms" and len(tokens) == 2:
            keyterms = extract_keyterms(tokens[1])
            return CommandResult(message="\n".join(keyterms) if keyterms else "(no keyterms)")
        enabled = {"on": True, "off": False, "toggle": not current}.get(tokens[0])
        if enabled is None:
            return CommandResult(message="Usage: /voice [show|on|off|toggle|keyterms TEXT]")
        settings.voice_mode = enabled
        save_settings(settings)
        if context.app_state is not None:
            context.app_state.set(
                voice_enabled=enabled,
                voice_available=diagnostics.available,
                voice_reason=diagnostics.reason,
            )
        return CommandResult(message=f"Voice mode {'enabled' if enabled else 'disabled'}.")

    async def _doctor_handler(args: str, context: CommandContext) -> CommandResult:
        """Run NIA Doctor diagnostics.

        Supports:
          - ``/doctor`` — dry-run (report only)
          - ``/doctor --fix`` — auto-repair fixable issues
          - ``/doctor --ack <id>`` — acknowledge a security advisory
        """
        del context
        try:
            from niaharness.cli.doctor import run_doctor

            # Parse args.
            fix = "--fix" in args
            ack_id = None
            if "--ack" in args:
                parts = args.split()
                idx = parts.index("--ack")
                if idx + 1 < len(parts):
                    ack_id = parts[idx + 1]

            result = run_doctor(fix=fix, ack=ack_id)
            return CommandResult(message=result.report)
        except Exception as exc:
            return CommandResult(message=f"Doctor failed: {exc}")

    async def _privacy_settings_handler(_: str, context: CommandContext) -> CommandResult:
        settings = load_settings()
        session_dir = get_project_session_dir(context.cwd)
        lines = [
            "Privacy settings:",
            f"- user_config_dir: {get_config_dir()}",
            f"- project_config_dir: {get_project_config_dir(context.cwd)}",
            f"- session_dir: {session_dir}",
            f"- feedback_log: {get_feedback_log_path()}",
            f"- api_base_url: {settings.base_url or '(default Anthropic-compatible endpoint)'}",
            "- network: enabled only for provider and explicit web/MCP calls",
            "- storage: local files under ~/.niaharness and project .niaharness",
        ]
        return CommandResult(message="\n".join(lines))

    async def _rate_limit_options_handler(_: str, context: CommandContext) -> CommandResult:
        settings = load_settings()
        provider = "moonshot-compatible" if (settings.base_url and "moonshot" in settings.base_url) else "anthropic-compatible"
        lines = [
            "Rate limit options:",
            f"- provider: {provider}",
            "- reduce /passes or switch /effort low for lighter requests",
            "- enable /fast for shorter responses and less tool churn",
            "- use /compact to shrink long transcripts before retrying",
            "- prefer background /tasks for long-running local work",
        ]
        return CommandResult(message="\n".join(lines))

    async def _release_notes_handler(_: str, context: CommandContext) -> CommandResult:
        path = Path(context.cwd) / "RELEASE_NOTES.md"
        if path.exists():
            return CommandResult(message=path.read_text(encoding="utf-8"))
        return CommandResult(
            message=(
                "# Release Notes\n\n"
                "- React TUI is now the default `oh` interface.\n"
                "- Added richer session, files, bridge, agent, copy, rewind, effort, passes, and privacy commands.\n"
                "- Expanded real-model validation across tools, MCP, tasks, plugins, notebook, LSP, cron, and worktree flows.\n"
            )
        )

    async def _upgrade_handler(args: str, context: CommandContext) -> CommandResult:
        """Check for and execute NIA updates.

        Supports:
          - ``/upgrade`` — check for update + execute if available
          - ``/upgrade --check`` — check only, don't install
          - ``/upgrade --no-backup`` — skip the pre-update backup
        """
        del context
        try:
            from niaharness.cli.update import run_update

            check_only = "--check" in args
            no_backup = "--no-backup" in args

            result = run_update(check=check_only, no_backup=no_backup)
            return CommandResult(message=result.report)
        except Exception as exc:
            return CommandResult(message=f"Upgrade failed: {exc}")

    async def _diff_handler(args: str, context: CommandContext) -> CommandResult:
        if args.strip() == "full":
            ok, output = _run_git_command(context.cwd, "diff", "HEAD")
            return CommandResult(message=output or "(no diff)")
        ok, output = _run_git_command(context.cwd, "diff", "--stat")
        if not ok:
            return CommandResult(message=output)
        return CommandResult(message=output or "(no diff)")

    async def _branch_handler(args: str, context: CommandContext) -> CommandResult:
        action = args.strip() or "show"
        if action == "show":
            ok, current = _run_git_command(context.cwd, "branch", "--show-current")
            if not ok:
                return CommandResult(message=current)
            return CommandResult(message=f"Current branch: {current or '(detached HEAD)'}")
        if action == "list":
            ok, branches = _run_git_command(context.cwd, "branch", "--format", "%(refname:short)")
            return CommandResult(message=branches if ok else branches)
        return CommandResult(message="Usage: /branch [show|list]")

    async def _commit_handler(args: str, context: CommandContext) -> CommandResult:
        message = args.strip()
        if not message:
            ok, status = _run_git_command(context.cwd, "status", "--short")
            return CommandResult(message=status if ok and status else "(working tree clean)")
        ok, status = _run_git_command(context.cwd, "status", "--short")
        if not ok:
            return CommandResult(message=status)
        if not status.strip():
            return CommandResult(message="Nothing to commit.")
        ok, output = _run_git_command(context.cwd, "add", "-A")
        if not ok:
            return CommandResult(message=output)
        ok, output = _run_git_command(context.cwd, "commit", "-m", message)
        return CommandResult(message=output if ok else output)

    async def _tasks_handler(args: str, context: CommandContext) -> CommandResult:
        manager = get_task_manager()
        tokens = args.split(maxsplit=2)
        if not tokens or tokens[0] == "list":
            tasks = manager.list_tasks()
            if not tasks:
                return CommandResult(message="No background tasks.")
            return CommandResult(
                message="\n".join(f"{task.id} {task.type} {task.status} {task.description}" for task in tasks)
            )
        if tokens[0] == "run" and len(tokens) >= 2:
            command = args[len("run ") :]
            task = await manager.create_shell_task(
                command=command,
                description=command[:80],
                cwd=context.cwd,
            )
            return CommandResult(message=f"Started task {task.id}")
        if tokens[0] == "stop" and len(tokens) == 2:
            task = await manager.stop_task(tokens[1])
            return CommandResult(message=f"Stopped task {task.id}")
        if tokens[0] == "show" and len(tokens) == 2:
            task = manager.get_task(tokens[1])
            if task is None:
                return CommandResult(message=f"No task found with ID: {tokens[1]}")
            return CommandResult(message=str(task))
        if tokens[0] == "update" and len(tokens) == 3:
            task_id = tokens[1]
            rest = tokens[2]
            field, _, value = rest.partition(" ")
            if not value.strip():
                return CommandResult(
                    message="Usage: /tasks update ID [description TEXT|progress NUMBER|note TEXT]"
                )
            try:
                if field == "description":
                    task = manager.update_task(task_id, description=value)
                    return CommandResult(message=f"Updated task {task.id} description")
                if field == "progress":
                    try:
                        progress = int(value)
                    except ValueError:
                        return CommandResult(message="Progress must be an integer between 0 and 100.")
                    task = manager.update_task(task_id, progress=progress)
                    return CommandResult(message=f"Updated task {task.id} progress to {progress}%")
                if field == "note":
                    task = manager.update_task(task_id, status_note=value)
                    return CommandResult(message=f"Updated task {task.id} note")
            except ValueError as exc:
                return CommandResult(message=str(exc))
            return CommandResult(
                message="Usage: /tasks update ID [description TEXT|progress NUMBER|note TEXT]"
            )
        if tokens[0] == "output" and len(tokens) == 2:
            return CommandResult(message=manager.read_task_output(tokens[1]) or "(no output)")
        return CommandResult(
            message=(
                "Usage: /tasks "
                "[list|run CMD|stop ID|show ID|update ID description TEXT|update ID progress NUMBER|update ID note TEXT|output ID]"
            )
        )

    registry.register(SlashCommand("help", "Show available commands", _help_handler))
    registry.register(SlashCommand("exit", "Exit NiaHarness", _exit_handler))
    registry.register(SlashCommand("clear", "Clear conversation history", _clear_handler))
    registry.register(SlashCommand("version", "Show the installed NiaHarness version", _version_handler))
    registry.register(SlashCommand("status", "Show session status", _status_handler))
    registry.register(SlashCommand("context", "Show the active runtime system prompt", _context_handler))
    registry.register(SlashCommand("summary", "Summarize conversation history", _summary_handler))
    registry.register(SlashCommand("compact", "Compact older conversation history", _compact_handler))
    registry.register(SlashCommand("cost", "Show token usage and estimated cost", _cost_handler))
    registry.register(SlashCommand("usage", "Show usage and token estimates", _usage_handler))
    registry.register(SlashCommand("stats", "Show session statistics", _stats_handler))
    registry.register(SlashCommand("memory", "Inspect and manage project memory", _memory_handler))
    registry.register(SlashCommand("hooks", "Show configured hooks", _hooks_handler))
    registry.register(SlashCommand("resume", "Restore the latest saved session", _resume_handler))
    registry.register(SlashCommand("session", "Inspect the current session storage", _session_handler))
    registry.register(SlashCommand("export", "Export the current transcript", _export_handler))
    registry.register(SlashCommand("share", "Create a shareable transcript snapshot", _share_handler))
    registry.register(SlashCommand("copy", "Copy the latest response or provided text", _copy_handler))
    registry.register(SlashCommand("tag", "Create a named snapshot of the current session", _tag_handler))
    registry.register(SlashCommand("rewind", "Remove the latest conversation turn(s)", _rewind_handler))
    registry.register(SlashCommand("files", "List files in the current workspace", _files_handler))
    registry.register(SlashCommand("init", "Initialize project NiaHarness files", _init_handler))
    registry.register(SlashCommand("bridge", "Inspect bridge helpers and spawn bridge sessions", _bridge_handler))
    registry.register(SlashCommand("login", "Show auth status or store an API key", _login_handler))
    registry.register(SlashCommand("logout", "Clear the stored API key", _logout_handler))
    registry.register(SlashCommand("oauth", "Anthropic PKCE OAuth login (Claude Pro/Max)", _oauth_handler))
    registry.register(SlashCommand("feedback", "Save CLI feedback to the local feedback log", _feedback_handler))
    registry.register(SlashCommand("onboarding", "Show the quickstart guide", _onboarding_handler))
    registry.register(SlashCommand("skills", "List or show available skills", _skills_handler))
    registry.register(SlashCommand("config", "Show or update configuration", _config_handler))
    registry.register(SlashCommand("mcp", "Show MCP status", _mcp_handler))
    registry.register(SlashCommand("plugin", "Manage plugins", _plugin_handler))
    registry.register(SlashCommand("reload-plugins", "Reload plugin discovery for this workspace", _reload_plugins_handler))
    registry.register(SlashCommand("permissions", "Show or update permission mode", _permissions_handler))
    registry.register(SlashCommand("plan", "Toggle plan permission mode", _plan_handler))
    registry.register(SlashCommand("fast", "Show or update fast mode", _fast_handler))
    registry.register(SlashCommand("effort", "Show or update reasoning effort", _effort_handler))
    registry.register(SlashCommand("passes", "Show or update reasoning pass count", _passes_handler))
    registry.register(SlashCommand("model", "Show, list, or set the active model", _model_handler))
    registry.register(SlashCommand("provider", "Show, list, or switch LLM provider", _provider_handler))
    registry.register(SlashCommand("theme", "Show or update the theme", _theme_handler))
    registry.register(SlashCommand("output-style", "Show or update output style", _output_style_handler))
    registry.register(SlashCommand("keybindings", "Show resolved keybindings", _keybindings_handler))
    registry.register(SlashCommand("vim", "Show or update Vim mode", _vim_handler))
    registry.register(SlashCommand("voice", "Show or update voice mode", _voice_handler))
    registry.register(SlashCommand("doctor", "Run diagnostics and auto-repair (optional: --fix, --ack <id>)", _doctor_handler))
    registry.register(SlashCommand("diff", "Show git diff output", _diff_handler))
    registry.register(SlashCommand("branch", "Show git branch information", _branch_handler))
    registry.register(SlashCommand("commit", "Show status or create a git commit", _commit_handler))
    registry.register(SlashCommand("issue", "Show or update project issue context", _issue_handler))
    registry.register(SlashCommand("pr_comments", "Show or update project PR comments context", _pr_comments_handler))
    registry.register(SlashCommand("privacy-settings", "Show local privacy and storage settings", _privacy_settings_handler))
    registry.register(SlashCommand("rate-limit-options", "Show ways to reduce provider rate pressure", _rate_limit_options_handler))
    registry.register(SlashCommand("release-notes", "Show recent NiaHarness release notes", _release_notes_handler))
    registry.register(SlashCommand("upgrade", "Check for and install updates (optional: --check, --no-backup)", _upgrade_handler))
    registry.register(SlashCommand("agents", "List or inspect agent and teammate tasks", _agents_handler))
    registry.register(SlashCommand("tasks", "Manage background tasks", _tasks_handler))

    # ── SOUL.md identity ─────────────────────────────────────────────
    async def _soul_handler(args: str, context: CommandContext) -> CommandResult:
        del context
        from niaharness.prompts.soul import (
            DEFAULT_SOUL_MD,
            get_soul_md_path,
            is_default_soul,
            load_soul_md,
        )

        soul_path = get_soul_md_path()
        subcommand = args.split()[0].lower() if args.strip() else "show"

        if subcommand == "show":
            content = load_soul_md()
            using_default = is_default_soul(content)
            header = (
                f"SOUL.md path: {soul_path}\n"
                f"Status: {'default (never customized)' if using_default else 'custom'}\n"
                f"---\n"
            )
            return CommandResult(message=header + content)

        if subcommand == "path":
            return CommandResult(message=str(soul_path))

        if subcommand == "reset":
            try:
                soul_path.write_text(DEFAULT_SOUL_MD, encoding="utf-8")
                return CommandResult(
                    message=f"Reset SOUL.md to default at {soul_path}"
                )
            except OSError as exc:
                return CommandResult(
                    message=f"Could not reset SOUL.md: {exc}", should_exit=False
                )

        if subcommand == "edit":
            # Hint for the UI to open the file in $EDITOR.
            import os

            editor = os.environ.get("EDITOR") or os.environ.get("VISUAL") or "nano"
            return CommandResult(
                message=(
                    f"Open this file in your editor:\n"
                    f"  {editor} {soul_path}\n\n"
                    f"Or run: $EDITOR {soul_path}\n"
                    f"Changes are picked up on the next message — no restart needed."
                )
            )

        return CommandResult(
            message=(
                "Usage:\n"
                "  /soul          Show current SOUL.md content + path\n"
                "  /soul show     (same as above)\n"
                "  /soul path     Print the SOUL.md file path\n"
                "  /soul edit     Show how to open SOUL.md in $EDITOR\n"
                "  /soul reset    Reset SOUL.md to the default (overwrites custom content)"
            )
        )

    registry.register(
        SlashCommand("soul", "Show or manage NIA's SOUL.md identity file", _soul_handler)
    )

    # ── /insights — usage analytics + cost estimation ───────────────
    async def _insights_handler(args: str, context: CommandContext) -> CommandResult:
        del context
        try:
            from niaharness.insights import InsightsEngine

            engine = InsightsEngine()
            # Parse args: optional days (default 30) and optional --source <name>
            # and --gateway flag for chat-delivery format.
            args_stripped = args.strip()
            days = 30
            source = None
            use_gateway_format = False

            if args_stripped:
                parts = args_stripped.split()
                for part in parts:
                    if part == "--gateway":
                        use_gateway_format = True
                    elif part.startswith("--source="):
                        source = part[len("--source="):]
                    elif part == "--source":
                        # Skip the flag; the next part is the value.
                        continue
                    elif part.isdigit():
                        days = max(1, min(int(part), 365))
                    # Skip the value of a preceding --source flag.
                # Handle "--source <name>" (space-separated).
                if "--source" in parts:
                    idx = parts.index("--source")
                    if idx + 1 < len(parts) and not parts[idx + 1].startswith("--"):
                        source = parts[idx + 1]

            report = engine.generate(days=days, source=source)
            if use_gateway_format:
                message = engine.format_gateway(report)
            else:
                message = engine.format_terminal(report)
            return CommandResult(message=message)
        except Exception as exc:
            return CommandResult(message=f"Failed to generate insights: {exc}")

    registry.register(
        SlashCommand(
            "insights",
            "Show usage analytics and cost estimation (optional: days, --source <name>, --gateway)",
            _insights_handler,
        )
    )

    # ── /profile — profile management ───────────────────────────────
    async def _profile_handler(args: str, context: CommandContext) -> CommandResult:
        del context
        from niaharness.profiles import (
            get_active_profile,
            list_profiles,
            switch_profile,
            create_profile,
            delete_profile,
        )

        parts = args.strip().split(None, 1)
        subcommand = parts[0] if parts else ""

        if subcommand == "" or subcommand == "show":
            active = get_active_profile()
            profiles = list_profiles()
            lines = [f"Active profile: {active.name}", "", "Available profiles:"]
            for p in profiles:
                marker = " *" if p.name == active.name else "  "
                lines.append(f"{marker} {p.name}  ({p.home})")
            lines.append("\nUsage:\n  /profile list           List profiles\n  /profile switch <name>  Switch active profile\n  /profile create <name>  Create a new profile\n  /profile delete <name>  Delete a profile")
            return CommandResult(message="\n".join(lines))

        if subcommand == "list":
            profiles = list_profiles()
            lines = [f"Profiles ({len(profiles)}):"]
            for p in profiles:
                lines.append(f"  {p.name}  ({p.home})")
            return CommandResult(message="\n".join(lines))

        if subcommand == "switch" and len(parts) > 1:
            name = parts[1].strip()
            try:
                profile = switch_profile(name)
                return CommandResult(
                    message=f"Switched to profile '{profile.name}'.\nRestart NIA for the change to take full effect."
                )
            except ValueError as exc:
                return CommandResult(message=f"Error: {exc}")

        if subcommand == "create" and len(parts) > 1:
            name = parts[1].strip()
            try:
                profile = create_profile(name)
                return CommandResult(message=f"Created profile '{profile.name}' at {profile.home}")
            except ValueError as exc:
                return CommandResult(message=f"Error: {exc}")

        if subcommand == "delete" and len(parts) > 1:
            name = parts[1].strip()
            try:
                if delete_profile(name):
                    return CommandResult(message=f"Deleted profile '{name}'.")
                return CommandResult(message=f"Profile '{name}' not found.")
            except ValueError as exc:
                return CommandResult(message=f"Error: {exc}")

        return CommandResult(
            message=(
                "Usage:\n"
                "  /profile                 Show active profile + list\n"
                "  /profile list            List all profiles\n"
                "  /profile switch <name>   Switch active profile\n"
                "  /profile create <name>   Create a new profile\n"
                "  /profile delete <name>   Delete a profile"
            )
        )

    registry.register(
        SlashCommand("profile", "Manage NIA profiles (isolated identity/memory/sessions)", _profile_handler)
    )

    return registry
