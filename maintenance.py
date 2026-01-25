import os
import shutil
from pathlib import Path

# --- Configuration ---
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
LOG_DIR = ROOT_DIR / "logs"
# Updated to match actual DB names found in config.py
MEMORY_DB_FILE = DATA_DIR / "memory.db" 
STATE_DB_FILE = DATA_DIR / "state.db"
GHOST_FILE = DATA_DIR / "ghost_state.json"
LOG_FILE = LOG_DIR / "nia.log"

def clean_pycache():
    """Removes all __pycache__ folders recursively."""
    print("🧹 Scanning for __pycache__...")
    count = 0
    for path in ROOT_DIR.rglob("__pycache__"):
        try:
            shutil.rmtree(path)
            count += 1
        except Exception as e:
            print(f"   ⚠️ Could not remove {path}: {e}")
    print(f"   ✅ Removed {count} cache directories.")

def wipe_logs():
    """Clears the content of the log file without deleting it."""
    print("🧹 Cleaning Log Files...")
    if LOG_FILE.exists():
        try:
            # Open in write mode to truncate (empty) the file
            with open(LOG_FILE, 'w') as f:
                f.write("") 
            print(f"   ✅ Wiped {LOG_FILE.name} (Fresh start)")
        except Exception as e:
            print(f"   ⚠️ Failed to wipe log: {e}")
    else:
        print("   ℹ️ No log file found.")

def reset_ghost_state():
    """Resets Ghost Mode to OFF."""
    print("👻 Resetting Ghost State...")
    if GHOST_FILE.exists():
        try:
            with open(GHOST_FILE, 'w') as f:
                f.write('{"active": false, "layer": 0}')
            print("   ✅ Ghost Mode set to OFF")
        except Exception as e:
            print(f"   ⚠️ Failed to reset ghost state: {e}")
    else:
        print("   ℹ️ No ghost state file found.")

def clear_memory_db():
    """Optional: Deletes the conversation memory."""
    print("🧠 Checking Memory Databases...")
    
    # Check Memory DB
    if MEMORY_DB_FILE.exists():
        choice = input(f"   ❓ Found semantic memory ({MEMORY_DB_FILE.name}). Delete it? (y/n): ").lower()
        if choice == 'y':
            try:
                os.remove(MEMORY_DB_FILE)
                print("   ✅ Memory DB wiped.")
            except Exception as e:
                print(f"   ⚠️ Failed to delete Memory DB: {e}")
        else:
            print("   ℹ️ Memory DB preserved.")
            
    # Check State DB
    if STATE_DB_FILE.exists():
        choice = input(f"   ❓ Found state DB ({STATE_DB_FILE.name}). Delete it? (y/n): ").lower()
        if choice == 'y':
            try:
                os.remove(STATE_DB_FILE)
                print("   ✅ State DB wiped.")
            except Exception as e:
                print(f"   ⚠️ Failed to delete State DB: {e}")
        else:
            print("   ℹ️ State DB preserved.")

def main():
    print("\n╔════════════════════════════════════╗")
    print("║   N.I.A. SYSTEM MAINTENANCE TOOL   ║")
    print("╚════════════════════════════════════╝\n")
    
    clean_pycache()
    print("-" * 30)
    wipe_logs()
    print("-" * 30)
    reset_ghost_state()
    print("-" * 30)
    clear_memory_db()
    
    print("\n✨ Maintenance Complete. System is polished and ready.\n")

if __name__ == "__main__":
    main()
