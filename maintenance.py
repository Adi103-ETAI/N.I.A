import os
import shutil
import argparse
import subprocess
import sys
from pathlib import Path

# --- Configuration ---
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
LOG_DIR = ROOT_DIR / "logs"
MEMORY_DB_FILE = DATA_DIR / "memory.db" 
STATE_DB_FILE = DATA_DIR / "state.db"
GHOST_FILE = DATA_DIR / "ghost_state.json"
LOG_FILE = LOG_DIR / "nia.log"
SANDBOX_MOUNTS = DATA_DIR / "sandbox_mounts"

def log(message, verbose=False, force=False):
    """Log helper. Always print if not forcing, or if verbose."""
    if verbose or not force:
        print(message)

def run_command(command, ignore_errors=True):
    """Run a shell command silently."""
    try:
        subprocess.run(command, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        if not ignore_errors:
            raise

def clean_pycache(force=False, verbose=False):
    """Removes __pycache__ and .pytest_cache folders."""
    log("🧹 Scanning for caches...", verbose, force)
    
    # 1. __pycache__
    count = 0
    for path in ROOT_DIR.rglob("__pycache__"):
        try:
            shutil.rmtree(path)
            count += 1
            if verbose: print(f"   Deleted: {path}")
        except Exception as e:
            if verbose: print(f"   ⚠️ Could not remove {path}: {e}")
    
    # 2. .pytest_cache
    pytest_cache = ROOT_DIR / ".pytest_cache"
    if pytest_cache.exists():
        try:
            shutil.rmtree(pytest_cache)
            count += 1
            if verbose: print(f"   Deleted: {pytest_cache}")
        except Exception as e:
            if verbose: print(f"   ⚠️ Could not remove {pytest_cache}: {e}")

    log(f"   ✅ Removed {count} cache directories.", verbose, force)

def cleanup_docker(force=False, verbose=False):
    """Kills session containers and wipes mount data."""
    log("🐳 Cleaning Docker Environment...", verbose, force)
    
    # 1. Kill Session Containers
    try:
        # Check for docker
        subprocess.check_call(["docker", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # List nia-session containers
        cmd = "docker ps -a --filter name=nia-session- -q"
        ids = subprocess.check_output(cmd, shell=True).decode().strip().split()
        
        if ids:
            log(f"   Found {len(ids)} session containers.", verbose, force)
            for container_id in ids:
                if verbose: print(f"   Stopping {container_id}...")
                run_command(f"docker rm -f {container_id}")
            log("   ✅ Removed session containers.", verbose, force)
        else:
            log("   ℹ️ No active session containers found.", verbose, force)
            
        # Optional: Prune if force is high? Let's just do orphans if requested? 
        # Requirement said: "Orphans: Run docker container prune -f (optional)"
        # We'll do it if force is on, to be thorough.
        if force:
             run_command("docker container prune -f")
             log("   ✅ Pruned stopped containers.", verbose, force)

    except (subprocess.CalledProcessError, FileNotFoundError):
        log("   ⚠️ Docker not available or error during cleanup.", verbose, force)

    # 2. Wipe Sandbox Mounts
    if SANDBOX_MOUNTS.exists():
        try:
            shutil.rmtree(SANDBOX_MOUNTS)
            log("   ✅ Wiped sandbox data mounts.", verbose, force)
        except Exception as e:
            log(f"   ⚠️ Failed to wipe mounts: {e}", verbose, force)

def wipe_logs(force=False, verbose=False):
    """Clears log files."""
    log("📝 Cleaning Logs...", verbose, force)
    
    # Main Log
    if LOG_FILE.exists():
        try:
            with open(LOG_FILE, 'w') as f: f.write("")
            log(f"   ✅ Truncated {LOG_FILE.name}", verbose, force)
        except Exception as e:
            if verbose: print(f"   ⚠️ Error: {e}")
            
    # Other .log files
    for log_file in LOG_DIR.glob("*.log"):
        if log_file != LOG_FILE:
            try:
                os.remove(log_file)
                if verbose: print(f"   Deleted: {log_file.name}")
            except Exception:
                pass

def reset_ghost_state(force=False, verbose=False):
    """Resets Ghost Mode to OFF."""
    log("👻 Resetting Ghost State...", verbose, force)
    if GHOST_FILE.exists():
        try:
            with open(GHOST_FILE, 'w') as f:
                f.write('{"active": false, "layer": 0}')
            log("   ✅ Ghost Mode set to OFF", verbose, force)
        except Exception:
            pass

def clean_dbs(force=False, verbose=False):
    """Deletes databases."""
    log("🧠 Checking Databases...", verbose, force)
    
    targets = [MEMORY_DB_FILE, STATE_DB_FILE]
    
    for db in targets:
        if db.exists():
            should_delete = force
            if not force:
                choice = input(f"   ❓ Found {db.name}. Delete? (y/n): ").lower()
                should_delete = (choice == 'y')
            
            if should_delete:
                try:
                    os.remove(db)
                    log(f"   ✅ Deleted {db.name}", verbose, force)
                except Exception as e:
                    if verbose: print(f"   ⚠️ Error deleting {db.name}: {e}")
            else:
                log(f"   ℹ️ Preserved {db.name}", verbose, force)

def run_verification(verbose=False):
    """Run post-cleanup verification if script is available."""
    print("\n🔎 Running Post-Cleanup Verification...")
    script_path = ROOT_DIR / "scripts" / "verify_integration.py"
    if script_path.exists():
        try:
            # Use same python interpreter
            cmd = [sys.executable, str(script_path)]
            result = subprocess.run(cmd, capture_output=not verbose, text=True)
            
            if result.returncode == 0:
                print("   ✅ Integration Verified (System Healthy)")
            else:
                print("   ❌ Verification Failed!")
                if not verbose:
                    print(result.stdout)
                    print(result.stderr)
        except Exception as e:
            print(f"   ⚠️ Failed to run verification: {e}")
    else:
        print("   ℹ️ Verification script not available (scripts/verify_integration.py missing).")

def main():
    parser = argparse.ArgumentParser(description="N.I.A. System Maintenance Tool")
    parser.add_argument("-f", "--force", action="store_true", help="Auto-confirm all deletions")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show detailed logs")
    parser.add_argument("--verify", action="store_true", help="Run verification after cleanup")
    
    args = parser.parse_args()
    
    # Always show header
    print("\n╔════════════════════════════════════╗")
    print("║   N.I.A. SYSTEM MAINTENANCE TOOL   ║")
    print("╚════════════════════════════════════╝\n")
    
    if args.force:
        print("🚀 Force Mode Enabled: Silent Cleanup Initiated...")
    
    clean_pycache(args.force, args.verbose)
    cleanup_docker(args.force, args.verbose)
    wipe_logs(args.force, args.verbose)
    reset_ghost_state(args.force, args.verbose)
    clean_dbs(args.force, args.verbose)
    
    if args.verify:
        run_verification(args.verbose)
    
    print("\n✨ Maintenance Complete.\n")

if __name__ == "__main__":
    main()
