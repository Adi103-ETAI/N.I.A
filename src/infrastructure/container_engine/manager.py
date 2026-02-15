"""
Docker Engine Manager.

Handles low-level Docker operations for the Sandbox.
"""
import docker
import logging
from typing import Optional, Tuple, Dict, Any, List

logger = logging.getLogger("NIA.Infrastructure.DockerEngine")

class DockerEngine:
    """Singleton wrapper for Docker SDK operations."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DockerEngine, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        try:
            self.client = docker.from_env()
            self.client.ping()
            logger.info("🐳 Connected to Docker Engine")
        except docker.errors.DockerException as e:
            logger.warning(f"⚠️ Docker Desktop not running or not installed: {e}")
            self.client = None
            
        self._initialized = True
        
    def ensure_network(self, network_name: str = "nia-sandbox-net"):
        """Ensure the isolated bridge network exists."""
        if not self.client:
            return

        try:
            self.client.networks.get(network_name)
        except docker.errors.NotFound:
            logger.info(f"Creating internal network: {network_name}")
            self.client.networks.create(
                network_name,
                driver="bridge",
                internal=True,  # No internet access by default (secure)
                check_duplicate=True
            )

    def pull_image(self, image_name: str = "python:3.11-slim"):
        """Ensure the target image exists locally."""
        if not self.client:
            return

        try:
            self.client.images.get(image_name)
        except docker.errors.ImageNotFound:
            logger.info(f"Pulling sandbox image: {image_name}...")
            self.client.images.pull(image_name)
            logger.info(f"Successfully pulled {image_name}")

    def _get_session_container_name(self, session_id: str) -> str:
        return f"nia-session-{session_id}"

    def _get_container(self, session_id: str):
        """Get active container for session if exists."""
        if not self.client:
            return None
        try:
            name = self._get_session_container_name(session_id)
            return self.client.containers.get(name)
        except docker.errors.NotFound:
            return None

    def start_session(self, session_id: str) -> str:
        """Start a persistent session container."""
        if not self.client:
            return "Docker client unavailable"
            
        name = self._get_session_container_name(session_id)
        existing = self._get_container(session_id)
        
        if existing:
            if existing.status != "running":
                existing.start()
            return f"Session {session_id} already active ({existing.short_id})"
            
        try:
            from src.infrastructure.container_engine.factory import SessionBuilder
            mounts = SessionBuilder.get_session_mounts(session_id)
            
            # Start persistent container
            container = self.client.containers.run(
                "python:3.11-slim",
                command="tail -f /dev/null", # Keep alive
                name=name,
                detach=True,
                remove=False, # Persistent
                volumes=mounts,
                network_mode="bridge"
            )
            return f"Session {session_id} started ({container.short_id})"
        except Exception as e:
            logger.error(f"Failed to start session {session_id}: {e}")
            raise

    def stop_session(self, session_id: str) -> bool:
        """Stop and remove a session container."""
        container = self._get_container(session_id)
        if container:
            try:
                container.kill()
                container.remove()
                return True
            except Exception as e:
                logger.error(f"Error stopping session {session_id}: {e}")
                return False
        return False

    def run_command(
        self, 
        image: str, 
        command: str, 
        session_id: str = "default",
        environment: Optional[Dict[str, str]] = None,
        mounts: Optional[Dict[str, Dict[str, str]]] = None,
        network: str = "nia-sandbox-net"
    ) -> Tuple[int, str, str]:
        """
        Run a command in the sandbox (Ephemeral or Persistent).
        """
        if not self.client:
            return -1, "", "Docker client not available"

        # Check for active session container
        container = self._get_container(session_id)
        
        if container and container.status == "running":
            # RESIDENT EXECUTION
            try:
                # exec_run returns (exit_code, output)
                # output is combined bytes in default configuration?
                # or we can use socket?
                # SDK exec_run: returns ExecResult(exit_code, output)
                exec_result = container.exec_run(
                    cmd=["bash", "-c", command],
                    environment=environment,
                    workdir="/workspace"
                )
                
                # Output is mixed stdout/stderr usually
                output = exec_result.output.decode('utf-8', errors='replace')
                return exec_result.exit_code, output, "" # Stderr mixed in stdout usually
                
            except Exception as e:
                logger.error(f"Session execution failed: {e}")
                return -1, "", str(e)
        else:
            # EPHEMERAL EXECUTION (Fallback)
            try:
                # Use provided mounts OR build them for this session
                if not mounts:
                    from src.infrastructure.container_engine.factory import SessionBuilder
                    mounts = SessionBuilder.get_session_mounts(session_id)

                container_ephemeral = self.client.containers.run(
                    image,
                    command=["bash", "-c", command],
                    detach=True,
                    environment=environment or {},
                    volumes=mounts or {},
                    network_mode="bridge",
                    remove=False,
                    working_dir="/workspace"  # FIX: Ensure we write to the mounted volume
                )
                
                result = container_ephemeral.wait()
                exit_code = result.get('StatusCode', -1)
                stdout = container_ephemeral.logs(stdout=True, stderr=False).decode('utf-8', errors='replace')
                stderr = container_ephemeral.logs(stdout=False, stderr=True).decode('utf-8', errors='replace')
                container_ephemeral.remove(force=True)
                
                return exit_code, stdout, stderr

            except Exception as e:
                logger.error(f"Ephemeral execution failed: {e}")
                return -1, "", str(e)

    def cleanup(self):
        """Kill all session containers on shutdown."""
        if not self.client:
            return
            
        try:
            sessions = self.client.containers.list(filters={"name": "nia-session-"})
            for s in sessions:
                try:
                    logger.info(f"🛑 Stopping session container: {s.name}")
                    s.kill()
                    s.remove()
                except Exception as e:
                    logger.warning(f"Failed to cleanup {s.name}: {e}")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
