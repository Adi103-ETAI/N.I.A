"""Universal Bash Wrapper for Coding Agent.

This script acts as the driver for the coding-agent skill.
It handles:
1. Reading the mission objective/command from mission.json.
2. Checking PTY requirements.
3. Spawning the appropriate shell process (interactive vs non-interactive).
4. Capturing output and writing result.json.
"""
import json
import os
import sys
import subprocess
import pty
import select
from pathlib import Path

def main():
    # 1. Read Mission
    try:
        mission_path = Path("mission.json")
        if not mission_path.exists():
            print("❌ mission.json not found")
            sys.exit(1)
            
        mission = json.loads(mission_path.read_text(encoding="utf-8"))
        command = mission.get("objective", "").strip() or mission.get("command", "").strip()
        
        # Fallback for raw command injection if 'code' was used to pass arguments
        # In this skill, 'code' isn't Python code to run, but might be CLI args
        # But per skill design, 'objective' usually holds the prompt "Write a flask app"
        # We need to construct the actual CLI command here if it wasn't passed fully.
        # For now, assume 'objective' IS the command or prompt for the default agent.
        
        # If the binary is 'codex', we might wrap the objective: codex "objective"
        # But to be 'Universal', we expect the General to pass the full command string
        # OR we default to a specific binary.
        # Let's assume the manifest 'objective' is the Prompt, and we default to 'codex'
        # unless 'command' is explicitly set in logic I don't see yet.
        # Actually, let's treat the 'objective' as the argument to the agent.
        
        # Check if we have a specific binary command
        binary = "codex" # Default
        full_command = f"{binary} {json.dumps(command)}" # Quote the prompt
        
    except Exception as e:
        print(f"❌ Failed to parse mission: {e}")
        sys.exit(1)

    print(f"🚀 Launching Coding Agent: {full_command}")
    
    # 2. Check PTY Requirement via Environment or Manifest inference
    # (The Bridge sets PTY at Docker level, so we are ALREADY in a PTY if requested)
    # However, Python subprocess doesn't automatically attach to it unless we use pty.spawn
    # or just subprocess.run if the parent is already a PTY? 
    # Actually, DockerBridge.run_command_pty attaches the container's TTY to the socket.
    # We are running INSIDE the container. 
    # If we are inside, `sys.stdout.isatty()` should be True if Bridge did its job.
    
    is_interactive = sys.stdout.isatty()
    output_text = ""
    exit_code = 0
    
    # 3. Execution
    if is_interactive:
        print("💻 PTY Detected - Spawning interactive shell...")
        
        # pty.spawn fits perfectly for interactive CLIs that need a TTY
        # It forks, runs the command, and connects parent FD to child TTY
        # But capturing output from pty.spawn is tricky as it writes to stdout directly.
        # We need to read from the master fd if we want to capture it for result.json.
        
        # Simple approach: Use subprocess with standard streams, relying on caller (Docker) TTY
        # For 'codex', it might need a real TTY.
        # Let's use a simpler robust method: subprocess.run, but we force a TTY allocation if needed?
        # If sys.stdout is already a TTY, subprocess.run inherits it by default usually.
        
        try:
            # We want to capture output AND stream it.
            process = subprocess.Popen(
                ["/bin/bash", "-c", full_command],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, # Merge stderr
                stdin=sys.stdin, # Pass stdin transparently
                text=True,
                bufsize=1 # Line buffered
            )
            
            captured_lines = []
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    sys.stdout.write(line) # Stream to Docker logs
                    sys.stdout.flush()
                    captured_lines.append(line)
            
            exit_code = process.poll()
            output_text = "".join(captured_lines)
            
        except Exception as e:
            output_text = f"Execution Error: {e}"
            exit_code = 1
            
    else:
        print("⚠️ No PTY Detected - Running in non-interactive mode")
        result = subprocess.run(
            ["/bin/bash", "-c", full_command],
            capture_output=True,
            text=True
        )
        output_text = result.stdout + "\n" + result.stderr
        exit_code = result.returncode

    # 4. Write Result
    result_data = {
        "status": "success" if exit_code == 0 else "error",
        "exit_code": exit_code,
        "output": output_text,
        "artifacts": [] # TODO: Scan workspace for new files?
    }
    
    # Write for standalone usage
    Path("result.json").write_text(json.dumps(result_data, indent=2), encoding="utf-8")
    print(f"\n🏁 Finished with exit code {exit_code}")
    
    return result_data

if __name__ == "__main__":
    # Assign to global 'result' so DockerBridge's _entrypoint.py picks it up
    result = main()
