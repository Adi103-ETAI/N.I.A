import docker
import os
import asyncio
from typing import Optional, Dict, Any, Tuple
import logging

logger = logging.getLogger("N.I.A.StaticSandbox")

class StaticSandbox:
    """
    Manages a long-running, static Docker container session for secure task execution.
    Maintains a persistent container across N.I.A execution so commands share state.
    """
    _instance: Optional['StaticSandbox'] = None
    
    def __init__(self, image_name: str = "nia-sandbox-common:latest"):
        self.image_name = image_name
        self.container_name = "nia-active-sandbox"
        self.workspace_dir = "/workspace"
        
        try:
            self.client = docker.from_env()
        except Exception as e:
            logger.error(f"Failed to connect to Docker daemon. Is Docker Desktop running? Error: {e}")
            self.client = None
            
        self.container = None
        
    @classmethod
    def get_instance(cls) -> 'StaticSandbox':
        """Singleton pattern so all tools use the same sandbox instance."""
        if cls._instance is None:
            cls._instance = StaticSandbox()
        return cls._instance

    def _get_existing_container(self):
        """Find the sandbox container if it's already running or stopped."""
        if not self.client:
            return None
            
        try:
            return self.client.containers.get(self.container_name)
        except docker.errors.NotFound:
            return None
        except Exception as e:
            logger.error(f"Error checking for existing container: {e}")
            return None

    def start(self) -> bool:
        """Starts the static sandbox container. Mounts current working directory."""
        if not self.client:
            logger.error("Cannot start sandbox: Docker client not initialized")
            return False
            
        local_cwd = os.getcwd()
        logger.info(f"Starting sandbox with volume mount: {local_cwd} -> {self.workspace_dir}")
        
        # Check if already running or stopped
        container = self._get_existing_container()
        
        if container:
            if container.status != 'running':
                logger.info("Restarting stopped sandbox container...")
                container.start()
            self.container = container
            return True
            
        # Create fresh sandbox from static image
        try:
            # We run it detached with a generic entrypoint tailored to keep it alive
            # 'tail -f /dev/null' is the easiest way to keep a container running forever
            self.container = self.client.containers.run(
                image=self.image_name,
                name=self.container_name,
                entrypoint=["tail", "-f", "/dev/null"],
                detach=True,
                volumes={
                    local_cwd: {'bind': self.workspace_dir, 'mode': 'rw'}
                },
                working_dir=self.workspace_dir,
                # Run as the `sandbox` user created in Layer 1
                user="1000:1000",
                network_mode="bridge"
            )
            logger.info(f"Started new sandbox container {self.container.short_id}")
            return True
        except docker.errors.ImageNotFound:
            logger.error(f"Sandbox image '{self.image_name}' not found. Did you run docker-setup.sh?")
            return False
        except Exception as e:
            logger.error(f"Failed to start sandbox container: {e}")
            return False

    def stop(self):
        """Stops and removes the sandbox container."""
        if self.container:
            logger.info("Stopping sandbox container...")
            try:
                self.container.stop(timeout=2)
                self.container.remove(force=True)
                self.container = None
                logger.info("Sandbox container removed.")
            except Exception as e:
                logger.error(f"Error stopping sandbox container: {e}")

    async def execute(self, command: str, timeout: int = 120) -> Tuple[int, str]:
        """
        Execute a bash command securely inside the active sandbox.
        Returns: (exit_code, output_text)
        """
        if not self.client or not self.container:
            return 1, "Error: Sandbox container is not running."
            
        logger.debug(f"Sandbox Exec: {command}")
        
        try:
            # Using asyncio.to_thread because docker SDK run/exec is blocking
            # We wrap the underlying exec_run to prevent asyncio loop blocking
            def _run_sync():
                # We wrap the command in bash -c to support pipes, redirection, etc.
                exec_result = self.container.exec_run(
                    cmd=["bash", "-c", command],
                    workdir=self.workspace_dir,
                    user="1000"
                )
                output = exec_result.output.decode('utf-8', errors='replace')
                return exec_result.exit_code, output

            # Run in a separate thread so N.I.A stays responsive
            exit_code, output = await asyncio.wait_for(
                asyncio.to_thread(_run_sync), 
                timeout=timeout
            )
            
            return exit_code, output

        except asyncio.TimeoutError:
            return 124, f"Command timed out after {timeout} seconds."
        except Exception as e:
            return 1, f"Execution failed: {e}"

# Cleanup hook (can be registered in main/app startup)
def cleanup_sandbox():
    sandbox = StaticSandbox.get_instance()
    sandbox.stop()
