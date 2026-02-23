"""Coding Agent — Direct LLM code generation + execution.

Strategy:
    1. Read the objective from mission.json
    2. Call NVIDIA's API (OpenAI-compatible) to generate Python code
    3. Execute the generated code inside the container
    4. Write result.json (or update the global 'result' dict for the entrypoint)

This avoids Pi-Mono CLI dependency entirely and directly uses the openai SDK
which is pip-installable and works reliably with NVIDIA's endpoint.
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


# Absolute paths — the bridge sets workdir=/workspace/project when host_workdir is mounted,
# but mission.json and result.json always live at /workspace/ (the mount root).
WORKSPACE = Path("/workspace")


def _get_llm_code(objective: str, api_key: str) -> str:
    """Ask NVIDIA LLM to generate Python code for the given objective."""
    try:
        from openai import OpenAI
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "openai", "--break-system-packages", "-q"], check=True)
        from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        base_url="https://integrate.api.nvidia.com/v1",
    )

    system_prompt = (
        "You are a Python coding assistant running inside a non-interactive Docker container. "
        "Respond ONLY with a complete, runnable Python script — no markdown fences, no explanation. "
        "The script must print its output to stdout. "
        "IMPORTANT: NEVER use input(), sys.stdin, or any interactive prompts — there is no terminal. "
        "If the task says 'accept input from user' or 'read from user', use hardcoded example values instead "
        "and add a comment showing where the user would provide input."
    )

    response = client.chat.completions.create(
        model="meta/llama-3.1-70b-instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": objective},
        ],
        temperature=0.1,
        max_tokens=2048,
    )

    code = response.choices[0].message.content.strip()

    # Strip markdown fences if the model ignores instructions
    if code.startswith("```"):
        lines = code.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        code = "\n".join(lines).strip()

    return code


def main(mission: dict = None):
    """Main entry point for the coding agent soldier.
    
    Args:
        mission: Pre-loaded mission dict (when called from _entrypoint.py exec()).
                 If None, reads from /workspace/mission.json.
    """

    # -----------------------------------------------------------------------
    # 1. Read Mission
    # -----------------------------------------------------------------------
    if mission is None:
        mission_path = WORKSPACE / "mission.json"
        if not mission_path.exists():
            print(f"❌ mission.json not found at {mission_path}")
            sys.exit(1)
        mission = json.loads(mission_path.read_text(encoding="utf-8"))

    objective = (
        mission.get("objective", "").strip()
        or mission.get("user_query", "").strip()
        or mission.get("command", "").strip()
    )

    # Strip [MEMORY CONTEXT] prefix if injected by graph
    if "[MEMORY CONTEXT]" in objective:
        # Extract the actual user query from the context block
        parts = objective.split("- User Input:")
        if len(parts) > 1:
            objective = parts[-1].split("\n")[0].strip().strip('"').strip("'")
        else:
            # fallback: take first line that isn't context metadata
            lines = [l.strip() for l in objective.split("\n") if l.strip() and not l.startswith("-") and "MEMORY" not in l]
            objective = lines[0] if lines else objective

    if not objective:
        print("❌ No objective found in mission.json")
        sys.exit(1)

    print(f"🎯 Objective: {objective}")

    # -----------------------------------------------------------------------
    # 2. Get API Key
    # -----------------------------------------------------------------------
    api_key = (
        os.environ.get("NVIDIA_API_KEY", "")
        or os.environ.get("OPENAI_API_KEY", "")
    )
    if not api_key:
        print("❌ No API key found — set NVIDIA_API_KEY in environment")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # 3. Generate Code via LLM
    # -----------------------------------------------------------------------
    print("🤖 Asking LLM to generate code...")
    try:
        generated_code = _get_llm_code(objective, api_key)
    except Exception as e:
        print(f"❌ LLM call failed: {e}")
        sys.exit(1)

    print(f"\n📝 Generated Code:\n{'='*40}")
    print(generated_code)
    print('='*40)

    # -----------------------------------------------------------------------
    # 4. Execute the Generated Code
    # -----------------------------------------------------------------------
    print("\n🚀 Executing code...")
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(generated_code)
        tmp_path = tmp.name

    try:
        proc = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=60,
        )
        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()
        exit_code = proc.returncode

        if stdout:
            print(f"\n✅ Output:\n{stdout}")
        if stderr:
            print(f"\n⚠️ Stderr:\n{stderr}")

    except subprocess.TimeoutExpired:
        stdout = ""
        stderr = "Execution timed out (60s)"
        exit_code = 1
        print(f"❌ {stderr}")
    except Exception as e:
        stdout = ""
        stderr = str(e)
        exit_code = 1
        print(f"❌ Execution error: {e}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    # -----------------------------------------------------------------------
    # 5. Write result (for standalone run) & update globals for entrypoint
    # -----------------------------------------------------------------------
    combined_output = f"Generated Code:\n{generated_code}\n\nOutput:\n{stdout}"
    if stderr:
        combined_output += f"\n\nStderr:\n{stderr}"

    result_data = {
        "status": "success" if exit_code == 0 else "error",
        "exit_code": exit_code,
        "output": combined_output,
        "artifacts": [],
        "generated_code": generated_code,
    }

    print(f"\n🏁 Finished with exit code {exit_code}")
    return result_data


if __name__ == "__main__":
    # When called via exec() from _entrypoint.py, the `mission` dict is injected
    # into globals. Pass it to main() so it doesn't try to re-read mission.json.
    _injected_mission = globals().get("mission", None)
    result = main(mission=_injected_mission)
