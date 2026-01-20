#!/usr/bin/env python3
"""
N.I.A. Brain Reset Script - "Brainwash" Utility

Surgically cleans LangGraph state while preserving database structure
and the skill/preference memory layer.

Target: data/state.db (conversation checkpoints)
Preserve: data/memory.db (skills, preferences, security logs)

Usage:
    python scripts/reset_brain.py
    python scripts/reset_brain.py --dry-run  # Preview without changes
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
STATE_DB = PROJECT_ROOT / "data" / "state.db"
MEMORY_DB = PROJECT_ROOT / "data" / "memory.db"  # DO NOT TOUCH


def reset_state_db(dry_run: bool = False) -> dict:
    """
    Clear LangGraph checkpoints and writes.
    
    Args:
        dry_run: If True, only report what would be deleted.
        
    Returns:
        Dict with deletion counts.
    """
    if not STATE_DB.exists():
        print(f"[ERROR] State DB not found: {STATE_DB}")
        return {"error": "Database not found"}
    
    conn = sqlite3.connect(str(STATE_DB))
    cursor = conn.cursor()
    
    # Get counts before deletion
    cursor.execute("SELECT COUNT(*) FROM checkpoints")
    checkpoint_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM writes")
    writes_count = cursor.fetchone()[0]
    
    print(f"\n{'='*60}")
    print("  N.I.A. BRAIN RESET UTILITY")
    print(f"{'='*60}")
    print(f"\n[INFO] Current State:")
    print(f"   - Checkpoints: {checkpoint_count}")
    print(f"   - Writes: {writes_count}")
    print(f"   - Total zombie entries: {checkpoint_count + writes_count}")
    
    if dry_run:
        print(f"\n[DRY RUN] No changes made")
        conn.close()
        return {
            "dry_run": True,
            "checkpoints": checkpoint_count,
            "writes": writes_count,
        }
    
    # Execute surgical deletion
    print(f"\n[EXEC] Executing brain wipe...")
    
    cursor.execute("DELETE FROM checkpoints")
    cursor.execute("DELETE FROM writes")
    conn.commit()
    
    # Verify deletion
    cursor.execute("SELECT COUNT(*) FROM checkpoints")
    remaining_checkpoints = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM writes")
    remaining_writes = cursor.fetchone()[0]
    
    # Vacuum to reclaim space
    print("[EXEC] Vacuuming database...")
    cursor.execute("VACUUM")
    conn.close()
    
    print(f"\n[SUCCESS] BRAIN WIPE COMPLETE")
    print(f"   - Checkpoints deleted: {checkpoint_count}")
    print(f"   - Writes deleted: {writes_count}")
    print(f"   - Remaining entries: {remaining_checkpoints + remaining_writes}")
    print(f"\n[PRESERVED] {MEMORY_DB.name} (skills/preferences intact)")
    print(f"\n>>> Memory Wiped. System Ready for Fresh Boot. <<<\n")
    
    return {
        "success": True,
        "checkpoints_deleted": checkpoint_count,
        "writes_deleted": writes_count,
    }


def main():
    parser = argparse.ArgumentParser(
        description="N.I.A. Brain Reset - Clear stale conversation state"
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Show what would be deleted without making changes"
    )
    args = parser.parse_args()
    
    result = reset_state_db(dry_run=args.dry_run)
    
    if result.get("error"):
        exit(1)
    exit(0)


if __name__ == "__main__":
    main()
