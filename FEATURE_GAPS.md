# NIA vs Hermes — Feature Gaps Tracker

> **Purpose:** This file is a *living tracker* of features we discover are missing
> while we work through the porting/repair effort. **Do not pre-fill it.**
>
> As we work on a subsystem and notice a real feature gap (something Hermes has
> that NIA lacks, or a behavior we can't reproduce), append a new entry under the
> appropriate section. Each entry should be one line plus optional context.
>
> When we come back to "see to those features and start adding", we'll work from
> this list in priority order.
>
> Format per entry:
> ```
> - [ ] <subsystem>: <feature name> — <one-line impact statement>
>       Reference: hermes-agent/<path>:<approx-line>
>       Context: <optional 1-2 sentences>
> ```

---

## Engine

<!-- Append engine gaps here as you find them -->

---

## Auxiliary Models

<!-- Append auxiliary gaps here as you find them -->

---

## Memory

<!-- Append memory gaps here as you find them -->

---

## Context Engine

<!-- Append context-engine gaps here as you find them -->

---

## Session DB

<!-- Append session-db gaps here as you find them -->

---

## Permissions

<!-- Append permissions gaps here as you find them -->

---

## Gateway

<!-- Append gateway gaps here as you find them -->

---

## Cron

<!-- Append cron gaps here as you find them -->

---

## TUI Gateway (WebSocket bridge)

<!-- Append tui-gateway gaps here as you find them -->

- [ ] TUI Gateway: `set_nia_home_override` / `reset_nia_home_override` — per-turn NIA_HOME override for resumed remote profiles is a no-op
      Reference: hermes-agent/tui_gateway/server.py:8572
      Context: When a session is resumed from a remote profile, Hermes swaps HERMES_HOME for the duration of the turn so the agent loads the profile's config/SOUL.md. NIA has `get_nia_home` (in `niaharness.prompts.soul`) but no setter/resetter pair. The deep-port of `prompt.submit` guards the call with try/except ImportError so the turn still runs, but profile-scoped config is silently ignored.

- [ ] TUI Gateway: `gateway.session_context.set_session_vars` / `clear_session_vars` — session-key/source/cwd contextvars not bound per-turn
      Reference: hermes-agent/tui_gateway/server.py:1884
      Context: Hermes binds session-key/source/cwd into a contextvar stack at turn start so downstream tools (terminal sudo, async delegation, approval routing) can read the active session without explicit threading. NIA's `niaharness.gateway` package has no `session_context` module. The deep-port calls the helpers via try/except ImportError — contextvars stay empty, so tools that would consult them fall back to defaults.

- [ ] TUI Gateway: `tools.terminal_tool.register_task_env_overrides` — per-session cwd override for terminal sudo / async tasks is missing
      Reference: hermes-agent/tui_gateway/server.py:1552
      Context: Lets the TUI session register its workspace as the cwd for terminal sudo prompts and async-delegation tasks, so they don't fall back to the gateway launch dir. NIA's `niaharness.tools` has no `terminal_tool` module exposing this API. The deep-port guards the call — terminal tasks use the process cwd.

- [ ] TUI Gateway: `engine.context_references.preprocess_context_references` — `@`-mention context injection is skipped
      Reference: hermes-agent/tui_gateway/server.py:8588
      Context: When the user's prompt contains `@path/to/file`, Hermes preprocesses the mention to inject the file's contents (subject to context-length budget and allowed-root checks). NIA's `niaharness.engine` has no `context_references` module. The deep-port guards the call — `@` mentions pass through to the agent as raw text.

- [ ] TUI Gateway: `engine.image_routing.decide_image_input_mode` / `build_native_content_parts` — native image input mode decision is hardcoded to "text"
      Reference: hermes-agent/tui_gateway/server.py:8626
      Context: Hermes decides per-turn whether to pass attached images to the main model as native OpenAI-style content parts (Anthropic/Gemini/Bedrock adapters translate) or pre-analyze with vision_analyze and prepend the description. NIA's `niaharness.engine` has no `image_routing` module. The deep-port always falls back to text-mode (`_enrich_with_attached_images`), which is correct but slower and loses pixel detail for vision-capable models.

- [ ] TUI Gateway: Mixture-of-Agents (MoA) one-shot restore path — `/moa` one-shot model swap has no restore hook
      Reference: hermes-agent/tui_gateway/server.py:8693
      Context: Hermes supports a `/moa` slash command that swaps the live agent to a Mixture-of-Agents model for one turn, then restores the previous model. The restore happens in `_run_prompt_submit` after the turn completes. NIA has no MoA subsystem. The deep-port pops any `moa_one_shot_restore` session key without restoring (no-op).

- [ ] TUI Gateway: `niaharness.goals.GoalManager` — `/goal` Ralph-style continuation loop is missing
      Reference: hermes-agent/tui_gateway/server.py:8820
      Context: After every TUI turn, if a `/goal` is active, Hermes asks a judge whether the goal is done and — if not and still under budget — queues a continuation prompt. NIA has no `niaharness.goals` package. The deep-port guards the call — `goal_followup` stays None and the continuation branch is a no-op.

- [ ] TUI Gateway: `engine.title_generator.maybe_auto_title` — LLM-backed auto-title generation falls back to first-60-chars heuristic
      Reference: hermes-agent/tui_gateway/server.py:8892
      Context: Hermes generates a session title asynchronously via an LLM call after the first turn, pushing it live to the sidebar via a callback. NIA's `niaharness.engine` has no `title_generator` module. The deep-port falls back to the existing simple heuristic (first 60 chars of the user's prompt).

- [ ] TUI Gateway: `tools.process_registry` (process_registry singleton, completion_queue, drain_notifications, format_process_notification) — background-process notification drain is missing
      Reference: hermes-agent/tui_gateway/server.py:8366
      Context: Hermes runs a per-session poller thread that drains a process-wide `completion_queue` of background-process events (completion, watch_match, async_delegation) and chains an agent turn for each. NIA has no `niaharness.tools.process_registry` module. The deep-port's `_notification_poller_loop` and the post-turn `_drain_notifications` block are guarded by try/except ImportError — background-process completions never surface as agent turns.

- [ ] TUI Gateway: `engine.auxiliary_client._read_main_provider` / `_read_main_model` — auxiliary-client metadata readers for image routing are missing
      Reference: hermes-agent/tui_gateway/server.py:8630
      Context: Image-routing decision uses these readers to determine the active provider/model. NIA has no `niaharness.engine.auxiliary_client` module. The deep-port's image-routing block falls back to text mode when the import fails.

- [ ] TUI Gateway: `tools.async_delegation.active_count` — live subagent count for the usage payload is missing
      Reference: hermes-agent/tui_gateway/server.py:3084
      Context: Hermes's `_get_usage` reports the live count of background/async subagents (delegate_task batches + background single delegations) as `usage["active_subagents"]`. NIA has no `niaharness.tools.async_delegation` module. The deep-port of `_get_usage` (already present in NIA) does not populate this field — the desktop status bar can't show the ⛓ indicator.

- [ ] TUI Gateway: `tools.vision_tools.vision_analyze_tool` — text-mode image enrichment falls back to "analysis unavailable" hint
      Reference: hermes-agent/tui_gateway/server.py:4574
      Context: When image_routing decides on text mode (or as a fallback), Hermes pre-analyzes each attached image via `vision_analyze_tool` and prepends the description. NIA has no `niaharness.tools.vision_tools` module exposing this function. The deep-port's `_enrich_with_attached_images` guards the import — attached images get a "[analysis unavailable]" placeholder hint instead of a description.

- [ ] TUI Gateway: `engine.model_metadata.get_model_context_length` — context-length resolver for `@`-mention budgeting is missing
      Reference: hermes-agent/tui_gateway/server.py:8589
      Context: Used by `preprocess_context_references` to cap injected file content at the model's context window. NIA's `niaharness.engine.model_metadata` module exists but does not export `get_model_context_length`. The deep-port guards the call — `@`-mention preprocessing is skipped entirely (see context_references gap above).

- [ ] TUI Gateway: `hermes_cli.active_sessions.transfer_active_session` — lease transfer on compression session_key rotation is a no-op
      Reference: hermes-agent/tui_gateway/server.py:2982
      Context: When the agent's compression path rotates `session_id`, Hermes transfers the active-session lease to the new id (so a concurrent gateway at the session cap can't grab the freed slot). NIA has no lease registry. The deep-port's `_transfer_active_session_slot` returns True unconditionally — the in-process `_active_session_sid` slot is session-id-agnostic so this is correct for now, but a future lease registry would need the transfer path.

- [ ] TUI Gateway: `hermes_cli.voice.speak_text` — voice TTS for agent replies uses NIA's SpeakTool wrapper (best-effort)
      Reference: hermes-agent/tui_gateway/server.py:8922
      Context: Hermes's TTS path is `hermes_cli.voice.speak_text(text)`. NIA has `niaharness.tools.speak_tool.SpeakTool` (a BaseTool with an async `execute`), but no module-level `speak_text(text)` function. The deep-port adds a `_speak_text` wrapper that invokes SpeakTool via `asyncio.run` — functionally equivalent but with a different audio backend (KittenTTS/espeak vs Hermes's TTS stack).

- [ ] TUI Gateway: `tools.terminal_tool.cleanup_vm` — terminal VM cleanup on cwd change is a no-op
      Reference: hermes-agent/tui_gateway/server.py:1785
      Context: When the user changes a session's cwd mid-session, Hermes calls `tools.terminal_tool.cleanup_vm(session_key)` to tear down any persistent terminal VM bound to the old workspace so subsequent terminal commands run in the new dir. NIA has no `niaharness.tools.terminal_tool` module. The deep-port of `_set_session_cwd` guards the call with try/except ImportError — terminal VMs (if any) are not cleaned up on cwd change.

- [ ] TUI Gateway: Profile-scoped `state.db` — per-profile session DB routing is missing
      Reference: hermes-agent/tui_gateway/server.py:1586
      Context: Hermes supports app-global remote mode where each local profile has its own `state.db` under the profile's home dir. A session created under a non-launch profile must persist against THAT profile's db, not the dashboard's launch profile. NIA's `_session_db` context manager (deep-ported) opens a fresh `SessionDB(db_path=Path(profile_home) / "state.db")` when `session["profile_home"]` is set — but no caller in NIA currently sets `profile_home` on the session dict, so the profile-scoped branch is dead code. The `profile` param on session.create / session.resume is also not honored.

- [ ] Session DB: `set_session_title` rowcount semantics — fixed (was using cumulative `conn.total_changes`)
      Reference: hermes-agent/tui_gateway/server.py:6016 (caller depends on truthy = row updated)
      Context: NIA's `SessionDB.set_session_title` was using `conn.total_changes > 0` to decide whether the UPDATE matched a row, but `total_changes` is cumulative across the connection's lifetime — so once any prior write succeeded on the same connection, every subsequent `set_session_title` would falsely report success even when the row didn't exist. This broke the `session.title` deep-port's ensure-row-then-retry fallback (the first call would return True, skipping the row-creation path, and the title would never persist). Fixed in this port by switching to `cursor.rowcount > 0`. Logged here so we remember to audit other `_execute_write` callers for the same anti-pattern.

- [ ] TUI Gateway: `cli.clipboard.has_clipboard_image` / `save_clipboard_image` — clipboard image paste falls back to xclip
      Reference: hermes-agent/tui_gateway/server.py:9037
      Context: Hermes's clipboard.paste uses `hermes_cli.clipboard.has_clipboard_image` + `save_clipboard_image` (cross-platform: macOS pbpaste, Windows PowerShell, Linux xclip). NIA has no `niaharness.cli.clipboard` module. The deep-port guards the import and falls back to invoking `xclip -selection clipboard -t image/png -o` directly — works on Linux but not macOS/Windows.

- [ ] TUI Gateway: `cli._detect_file_drop` / `_resolve_attachment_path` / `_split_path_input` / `_IMAGE_EXTENSIONS` — file-drop detection + path resolution helpers are missing
      Reference: hermes-agent/tui_gateway/server.py:9080
      Context: image.attach + input.detect_drop use these CLI helpers to parse pasted text that contains a file path (with optional remainder text). NIA's `niaharness.cli` module doesn't expose them. The deep-ports fall back to simple `Path(raw).expanduser()` resolution — works for basic cases but loses the file-drop pattern parsing + workspace-relative resolution.

- [ ] TUI Gateway: `tools.voice_mode.check_voice_requirements` — voice mode availability probe is missing
      Reference: hermes-agent/tui_gateway/server.py:12822
      Context: `/voice status` probes STT/TTS provider availability (audio device, transcription deps) so the user can tell why voice isn't working. NIA has no `niaharness.tools.voice_mode` module. The deep-port of voice.toggle returns `available: False, details: "voice_mode module not available"` instead.

- [ ] TUI Gateway: `cli.voice.start_continuous` / `stop_continuous` — voice recording loop is missing
      Reference: hermes-agent/tui_gateway/server.py:12913
      Context: voice.record start/stop use these to run a VAD-bounded push-to-talk capture loop with configurable silence_threshold / silence_duration. NIA has no `niaharness.cli.voice` module. The deep-port returns 5025 "voice module not available" on start/stop.

- [ ] TUI Gateway: `engine.verification_evidence.verification_status` — verification ledger read is missing
      Reference: hermes-agent/tui_gateway/server.py:5235
      Context: verification.status is a read-only consumer of the core verification ledger (test pass/fail evidence per session/cwd). NIA has no `niaharness.engine.verification_evidence` module. The deep-port returns `{status: "unknown", evidence: None}` instead.

- [ ] TUI Gateway: `engine.checkpoint_manager.CheckpointManager` — git-based rollback checkpoints are missing
      Reference: hermes-agent/tui_gateway/server.py:4559
      Context: rollback.list / rollback.restore / rollback.diff use a CheckpointManager to list/restore/diff git-based checkpoints created during agent turns. NIA has no `niaharness.engine.checkpoint_manager` module. The deep-ports' `_with_checkpoints` helper falls back to `{enabled: False, checkpoints: []}` — rollback RPCs return "disabled" instead of crashing.

- [ ] TUI Gateway: `engine.preview_restart.ephemeral_preview_agent_kwargs` / `preview_restart_callbacks` / `preview_restart_history` — preview restart agent is missing
      Reference: hermes-agent/tui_gateway/server.py:9772
      Context: preview.restart spawns a background AIAgent with the parent session's history to recover a broken local preview URL. NIA has no `niaharness.engine.preview_restart` module. The deep-port builds the restart prompt correctly but returns 5030 "preview.restart not available" instead of spawning the agent.

- [ ] TUI Gateway: `cli.runtime_provider.resolve_runtime_provider` / `cli.auth.has_usable_secret` / `cli.main._has_any_provider_configured` — provider runtime resolution is missing
      Reference: hermes-agent/tui_gateway/server.py:10998
      Context: setup.runtime_check runs the same resolve_runtime_provider() call the agent uses on session creation to verify the configured model actually resolves to a usable runtime. NIA has no `niaharness.cli.runtime_provider` / `cli.auth` / `cli.main` modules. The deep-port falls back to scanning common API key env vars (ANTHROPIC_API_KEY, OPENAI_API_KEY, etc.) — works for basic key-based providers but misses OAuth / Bedrock IAM / Copilot gh-auth-token paths.

- [ ] TUI Gateway: `cli.auth.PROVIDER_REGISTRY` / `clear_provider_auth` / `cli.config.remove_env_value` — provider credential management is missing
      Reference: hermes-agent/tui_gateway/server.py:12500
      Context: model.disconnect removes API key env vars from .env + process env, and clears OAuth / credential-pool state. NIA has no `niaharness.cli.auth` / `cli.config` modules. The deep-port falls back to clearing common env var naming patterns (`{SLUG}_API_KEY`, `NIA_{SLUG}_API_KEY`, `HERMES_{SLUG}_API_KEY`) — works for simple key-based providers but misses OAuth token clearing.

- [ ] TUI Gateway: `cli.projects_db` (connect_closing, list_projects, get_project, create_project, update_project, add_folder, remove_folder, set_primary_folder, archive_project, delete_project, set_active, get_active_id, find_for_cwd) — projects DB is missing
      Reference: hermes-agent/tui_gateway/server.py:10426
      Context: The 11 projects.* RPCs (list / get / create / update / add_folder / remove_folder / set_primary / archive / delete / set_active / for_cwd) all use a per-profile projects DB to manage first-class multi-folder workspaces. NIA has no `niaharness.cli.projects_db` module. The deep-ports' `_projects_connect` helper returns None when the module is unavailable, so every projects RPC returns an empty / "no such project" response. The RPC structure + error codes (5061/5062/5063) are in place — only the DB backend is missing.

- [ ] TUI Gateway: `engine.redact.redact_sensitive_text` — secret redaction in command output is missing
      Reference: hermes-agent/tui_gateway/server.py:11484
      Context: command.dispatch's quick-command exec path redacts GitHub/OpenAI/Anthropic/Bearer/AWS tokens + password= assignments from subprocess output before returning it to the TUI. NIA has no `niaharness.engine.redact` module. The deep-port's `_redact_sensitive_text` helper falls back to a regex-based redaction covering the same patterns — functionally equivalent but not centralized.

- [ ] TUI Gateway: `engine.learn_prompt.build_learn_prompt` — /learn prompt builder is missing
      Reference: hermes-agent/tui_gateway/server.py:11546
      Context: /learn is an open-ended skill-authoring command that builds a standards-guided prompt for the agent. NIA has no `niaharness.engine.learn_prompt` module. The deep-port falls back to a generic "Gather relevant context and write a new skill via skill_manage" message.

- [ ] TUI Gateway: `goals.GoalManager` — /goal Ralph-style continuation loop is missing (duplicate of earlier entry, listed here for completeness)
      Reference: hermes-agent/tui_gateway/server.py:11670
      Context: /goal set/pause/resume/clear/status + the post-turn goal continuation hook in _run_prompt_submit both depend on a GoalManager that tracks active goals + turn budgets + a judge. NIA has no `niaharness.goals` package. Both the command.dispatch /goal handler and the prompt.submit goal-followup path return clear "goals unavailable" errors.

- [ ] TUI Gateway: `skills.skill_commands.scan_skill_commands` / `build_skill_invocation_message` — skill slash-command discovery is missing
      Reference: hermes-agent/tui_gateway/server.py:11509
      Context: command.dispatch checks for skill-defined slash commands (e.g. `/commit`, `/review`) before falling through to built-in handlers. NIA has no `niaharness.skills.skill_commands` module. The deep-port's skill-commands block is guarded by try/except ImportError — skill slash-commands fall through to the "not a quick/plugin/skill command" error.

- [ ] TUI Gateway: `plugins.get_plugin_command_handler` / `resolve_plugin_command_result` — plugin command dispatch is missing
      Reference: hermes-agent/tui_gateway/server.py:11496
      Context: command.dispatch checks for plugin-registered slash commands. NIA has no `niaharness.plugins` module with these functions. The deep-port's plugin-commands block is guarded by try/except ImportError — plugin commands fall through.

- [ ] TUI Gateway: `engine.replay_cleanup.sanitize_replay_history` — dangling tool-call tail cleanup is missing
      Reference: hermes-agent/tui_gateway/server.py:5612
      Context: session.resume sanitizes the replayed history to strip a dangling assistant(tool_calls) tail (a session killed mid-tool-loop would otherwise replay the unanswered call forever). NIA has no `niaharness.engine.replay_cleanup` module. The deep-port's `_sanitize_replay_history` helper falls back to the raw history — sessions killed mid-tool-loop may replay the dangling call.

- [ ] TUI Gateway: `profiles.get_profile_dir` / `current_profile_name` — profile name resolution is missing
      Reference: hermes-agent/tui_gateway/server.py:920
      Context: _profile_home + _current_profile_name resolve named profiles' home dirs (for per-profile state.db / config.yaml / pets dir). NIA's `niaharness.profiles` package doesn't expose these. The deep-ports return None / "" so the launch profile path is used — no per-profile isolation yet.

- [ ] TUI Gateway: `config.reasoning.parse_reasoning_effort` — reasoning effort parser is missing
      Reference: hermes-agent/tui_gateway/server.py:10185
      Context: config.set reasoning uses parse_reasoning_effort to validate + normalize effort levels (low/medium/high/none). NIA has no `niaharness.config.reasoning` module. The deep-port stores the raw arg as-is when the parser is unavailable — works for valid values but doesn't reject invalid ones.

---

## UI / TUI / Web App

<!-- Append UI gaps here as you find them -->

---

## Doctor

<!-- Append doctor gaps here as you find them -->

---

## Profiles

<!-- Append profiles gaps here as you find them -->

---

## Skills

<!-- Append skills gaps here as you find them -->

---

## Tools

<!-- Append tools gaps here as you find them -->

---

## Providers

<!-- Append providers gaps here as you find them -->

---

## CLI

<!-- Append CLI gaps here as you find them -->

---

## Other / Cross-cutting

<!-- Append cross-cutting gaps here as you find them -->
