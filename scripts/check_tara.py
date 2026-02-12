#!/usr/bin/env python3
"""
TARA 2.0 Pre-Flight Test Script.

This standalone script tests TARA's subsystems in isolation before
integrating with the NIA master graph.

Usage:
    python scripts/test_tara.py
    
Tests:
    1. Tool auto-discovery
    2. Context building
    3. Full reasoning loop (with real LLM)
    
Safety:
    - 3-second countdown before execution
    - Move mouse to corner to abort (pyautogui failsafe)
"""
from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

# =============================================================================
# Path Setup (Ensure project root is in sys.path)
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# Imports (After path setup)
# =============================================================================

from src.core.logger import setup_logger
from src.core.config import get_settings

logger = setup_logger("TEST_FLIGHT")
settings = get_settings()


# =============================================================================
# Test Functions
# =============================================================================

def test_tool_discovery():
    """Test 1: Verify tool auto-discovery works."""
    logger.info("=" * 60)
    logger.info("TEST 1: Tool Auto-Discovery")
    logger.info("=" * 60)
    
    try:
        from src.capabilities.interface import get_tara_tools, get_tools_by_category
        
        # Get all tools
        tools = get_tara_tools()
        logger.info(f"✅ Discovered {len(tools)} tools")
        
        # Show by category
        categories = get_tools_by_category()
        for category, tool_names in categories.items():
            logger.info(f"   📦 {category}: {len(tool_names)} tools")
            for name in tool_names[:3]:  # Show first 3
                logger.info(f"      • {name}")
            if len(tool_names) > 3:
                logger.info(f"      ... and {len(tool_names) - 3} more")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Tool discovery failed: {e}")
        return False


def test_context_builder():
    """Test 2: Verify context builder formats correctly."""
    logger.info("=" * 60)
    logger.info("TEST 2: Context Builder")
    logger.info("=" * 60)
    
    try:
        from src.agents.tara.graph.prompts import build_tara_context
        
        # Mock state
        mock_state = {
            "user_goal": "Test goal",
            "screen_context": "[1] {Button} \"Save File\"\n[2] {Edit} \"filename.txt\"",
            "active_app": "notepad_1",
            "clipboard": "Test clipboard content",
            "last_error": "None",
            "iteration_count": 1,
        }
        
        context = build_tara_context(mock_state)
        
        logger.info("✅ Context built successfully:")
        for line in context.split("\n")[:10]:
            logger.info(f"   {line}")
        logger.info("   ...")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Context builder failed: {e}")
        return False


def test_state_creation():
    """Test 3: Verify state factory works."""
    logger.info("=" * 60)
    logger.info("TEST 3: State Factory")
    logger.info("=" * 60)
    
    try:
        from src.agents.tara.graph.state import create_initial_tara_state, TaraState
        
        state = create_initial_tara_state("Test goal")
        
        logger.info("✅ State created successfully:")
        logger.info(f"   user_goal: {state.get('user_goal')}")
        logger.info(f"   iteration_count: {state.get('iteration_count')}")
        logger.info(f"   messages: {len(state.get('messages', []))} items")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ State factory failed: {e}")
        return False


def test_graph_compilation():
    """Test 4: Verify graph compiles without error."""
    logger.info("=" * 60)
    logger.info("TEST 4: Graph Compilation")
    logger.info("=" * 60)
    
    try:
        from src.agents.tara.graph.workflow import build_tara_graph
        
        app = build_tara_graph()
        
        if app is not None:
            logger.info("✅ Graph compiled successfully")
            logger.info(f"   Type: {type(app).__name__}")
            return True
        else:
            logger.warning("⚠️ Graph is None (langgraph may not be installed)")
            return False
        
    except Exception as e:
        logger.error(f"❌ Graph compilation failed: {e}")
        return False


def test_full_execution(goal: str):
    """Test 5: Full reasoning loop with real LLM."""
    logger.info("=" * 60)
    logger.info("TEST 5: Full Execution (LIVE)")
    logger.info("=" * 60)
    logger.info(f"Goal: {goal}")
    
    # Safety countdown
    logger.warning("⚠️  SAFETY: Move mouse to screen corner to abort!")
    logger.warning("⚠️  Starting in 3 seconds...")
    for i in range(3, 0, -1):
        logger.info(f"   {i}...")
        time.sleep(1)
    
    try:
        from src.agents.tara.graph import run_tara
        
        start_time = time.time()
        
        logger.info("🚀 Executing TARA reasoning loop...")
        result = run_tara(goal)
        
        elapsed = time.time() - start_time
        
        logger.info("=" * 60)
        logger.info("✅ EXECUTION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"⏱️  Time: {elapsed:.2f} seconds")
        logger.info(f"📝 Result:")
        logger.info("-" * 40)
        print(result)
        logger.info("-" * 40)
        
        return True
        
    except KeyboardInterrupt:
        logger.warning("🛑 Aborted by user")
        return False
    except Exception as e:
        logger.error(f"❌ Execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Main
# =============================================================================

def main():
    """Run all tests."""
    print()
    print("=" * 60)
    print("   TARA 2.0 PRE-FLIGHT CHECK")
    print("=" * 60)
    print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Project: {PROJECT_ROOT}")
    print("=" * 60)
    print()
    
    results = {}
    
    # Test 1: Tool Discovery
    results["Tool Discovery"] = test_tool_discovery()
    print()
    
    # Test 2: Context Builder
    results["Context Builder"] = test_context_builder()
    print()
    
    # Test 3: State Factory
    results["State Factory"] = test_state_creation()
    print()
    
    # Test 4: Graph Compilation
    results["Graph Compilation"] = test_graph_compilation()
    print()
    
    # Summary before live test
    print("=" * 60)
    print("   PRE-FLIGHT SUMMARY")
    print("=" * 60)
    all_passed = True
    for test_name, passed in results.items():
        status = "✅" if passed else "❌"
        print(f"   {status} {test_name}")
        if not passed:
            all_passed = False
    print()
    
    if not all_passed:
        logger.error("❌ Pre-flight checks failed. Aborting live test.")
        return 1
    
    # Ask before live test
    print("=" * 60)
    print("   READY FOR LIVE TEST")
    print("=" * 60)
    print()
    
    GOAL = (
        "Open 'https://www.google.com' using the browser. "
        "Wait for the page to load. "
        "Type 'Agentic AI' into the search bar. "
        "Wait 3 seconds to verify typing. "
        "Finally, CLOSE the browser window."
    )
    
    print(f"   Mission: {GOAL}")
    print()
    response = input("   Proceed with live test? [y/N]: ").strip().lower()
    
    if response != "y":
        logger.info("Live test skipped.")
        return 0
    
    print()
    
    # Test 5: Full Execution
    success = test_full_execution(GOAL)
    
    print()
    print("=" * 60)
    print("   FINAL STATUS")
    print("=" * 60)
    if success:
        print("   🎉 ALL SYSTEMS GO!")
    else:
        print("   ⚠️  TEST INCOMPLETE")
    print("=" * 60)
    print()
    
    return 0 if success else 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n🛑 Aborted.")
        sys.exit(1)
