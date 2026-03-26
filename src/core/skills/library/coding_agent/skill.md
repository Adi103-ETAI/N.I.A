---
name: coding-agent
emoji: 🤖
description: Spawns an AI coding agent (Codex, Claude Code, etc.) inside the Docker Sandbox to write, debug, and execute code autonomously.
runtime: python
pty: true
workdir: /workspace/project
metadata:
  nia:
    requires: ["codex"]
    compatibility: ["gpt-4-turbo", "claude-3-opus"]
    version: "1.0.0"
---

# 🤖 Coding Agent Skill (Bash-First)

Enables the General to delegate complex coding tasks to a sandboxed AI coding agent (e.g., OpenAI Codex CLI, Claude Code, Aider) running inside Docker.

## When to Use

- User asks to **write**, **debug**, or **refactor** code
- User provides a coding task that requires file creation and execution
- User wants a sub-agent to autonomously solve a programming problem

## Execution Model

```
General → MissionManifest (pty: true) → DockerBridge → Container
                                                        ↓
                                              Universal Bash Wrapper (source.py)
                                                        ↓
                                              Checks PTY requirements
                                                        ↓
                                              Spawns Interactive Shell (/bin/bash)
                                                        ↓
                                              Executes Command (e.g., "codex 'Build a flask app'")
```

## Requirements

- **PTY Required**: `pty: true`
- **Binary Required**: `codex` (or equivalent CLI)
- **Workdir**: `/workspace/project`
