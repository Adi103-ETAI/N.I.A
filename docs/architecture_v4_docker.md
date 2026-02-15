# N.I.A. v4.0.0 Architecture: Docker + TARA + Ephemeral

**Status:** DRAFT
**Author:** Antigravity
**Date:** 2026-02-14

## 1. The 'Sandboxed Execution' Pattern

We are introducing a strictly sandboxed execution layer to allow N.I.A. to safely execute code and terminal commands without risking the host system.

### System Diagram

```mermaid
graph TD
    subgraph Host["Host System (Windows)"]
        Supervisor[N.I.A. Supervisor]
        Tara[TARA 2.0 Agent]
        Warden[Warden (Security)]
        DM[DockerManager]
        Mounts[Shared Volume (workspace/docker_mounts)]
    end

    subgraph Docker["Docker Environment (Linux)"]
        Container[Ephemeral Container]
        Shell[Bash Shell]
        Tools[Linux Tools (git, python, pip)]
    end

    Supervisor -->|Delegates Task| Tara
    Tara -->|Generates Command| Warden
    Warden -->|Approves/Intercepts| DM
    DM -->|Spawns/Executes| Container
    Container -->|Runs| Shell
    Shell -->|Output/Files| Mounts
    Mounts -->|Syncs| Tara
```

### Data Exchange
Data transfer between the Host and the Container is handled via a shared volume mount.
- **Host Path:** `workspace/docker_mounts/<session_id>/`
- **Container Path:** `/workspace/`

1.  **Input:** TARA writes code/scripts to `workspace/docker_mounts/<session_id>/`.
2.  **Execution:** DockerManager commands the container to run scripts located in `/workspace/`.
3.  **Output:** The container writes results/artifacts back to `/workspace/`, which are immediately visible to TARA on the host.

---

## 2. TARA 2.0: The Direct Execution Model

We are moving from a "Tool-First" approach (specific tools for every little action) to a "Bash-First" approach (direct shell interaction), effectively giving TARA a computer terminal.

### The SandboxedShell Tool

Instead of `git_clone`, `write_file`, `install_package`, TARA will use a single, powerful tool:

```python
class SandboxedShell(BaseTool):
    name = "sandboxed_shell"
    description = "Execute bash commands in a secure Linux sandbox. Use this for ALL file operations, git commands, and code execution."
    
    class Input(BaseModel):
        command: str = Field(description="The bash command(s) to execute.")
        timeout: int = Field(default=60, description="Execution timeout in seconds.")
        background: bool = Field(default=False, description="Run in background (detached).")

    def _run(self, command: str, timeout: int = 60, background: bool = False):
        return docker_manager.execute(command, timeout, background)
```

### Warden Interception
The `Warden` currently intercepts "high-risk" tools. In v4.0.0, **ALL** usage of `SandboxedShell` is considered high-risk but "Pre-Approved" for the Sandbox.
However, if TARA attempts to use *Legacy Local Tools* (e.g., `local_file_write` on the Windows host), the Warden must **BLOCK** it and instruct TARA to use `SandboxedShell` instead.

---

## 3. Ephemeral Agent Lifecycle

We introduce two modes of container operation handled by `DockerManager`.

### 3.1 Task Containers (Standard Mode)
- **Concept:** One container per command.
- **Lifecycle:** 
    1.  `docker run ... command`
    2.  Container starts, runs command, exits.
    3.  Container is removed.
- **Use Case:** Simple, stateless checks (e.g., `python --version`, `ls -la`).

### 3.2 Resident Mode (Session Mode)
- **Concept:** A long-running container acting as a persistent session.
- **Lifecycle:**
    1.  `docker run -d ... sleep infinity` (Started at task beginning)
    2.  `docker exec ... command_1`
    3.  `docker exec ... command_2`
    4.  `docker stop/rm` (At task completion)
- **Use Case:** Complex coding tasks, maintaining git state, running a dev server, installing dependencies that need to persist across steps.

*For v4.0.0, we will prioritize **Resident Mode** to support multi-step workflows.*

---

## 4. Proposed Folder Structure

New directories and files to implement this architecture:

```text
N.I.A/
├── docs/
│   └── architecture_v4_docker.md   <-- (This Document)
├── src/
│   ├── agents/
│   │   └── tara/
│   │       └── tools/              <-- [NEW] Specialized TARA tools
│   │           ├── __init__.py
│   │           └── shell.py        <-- The SandboxedShell tool
│   └── core/
│       └── sandbox/                <-- [NEW] Docker handling Logic
│           ├── __init__.py
│           ├── docker_manager.py   <-- Docker API wrapper
│           └── session.py          <-- Manages Resident Sessions
└── workspace/
    └── docker_mounts/              <-- [NEW] Shared data volume
```

---

## 5. The Ripple Effect Analysis

Integrating this layer will require changes in the following existing files:

| File | Impact | Description |
| :--- | :--- | :--- |
| `src/core/config.py` | **HIGH** | Add `DOCKER_IMAGE`, `DOCKER_TIMEOUT`, `USE_DOCKER_SANDBOX` settings. |
| `src/agents/tara/graph/nodes.py` | **MEDIUM** | Update `tool_executor` to prefer `SandboxedShell` and handle Docker errors differently. |
| `src/capabilities/interface.py` | **MEDIUM** | Register the new `SandboxedShell` tool so TARA can see it. |
| `src/agents/tara/security.py` | **MEDIUM** | Update `Warden` to allow Docker execution but stricter blocking on Host execution. |
| `src/agents/nia/graph/builder.py` | **LOW** | Ensure the graph passes the correct context (Session ID) to TARA for volume mounting. |
| `main.py` | **LOW** | Initialize `DockerManager` cleanup on shutdown. |
