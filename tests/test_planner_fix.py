#!/usr/bin/env python3
"""Quick test of improved planner prompt with better scope classification."""

import asyncio
import sys

async def test_planner():
    from src.agents.nia.planner import MissionPlanner
    from src.core.policy.scopes import CapabilityScope

    planner = MissionPlanner()

    # Test cases
    test_cases = [
        ("hello", CapabilityScope.READ_ONLY),
        ("help", CapabilityScope.READ_ONLY),
        ("what is Python?", CapabilityScope.READ_ONLY),
        ("write a Python script", CapabilityScope.EXECUTE),
        ("write and run a script", [CapabilityScope.WRITE, CapabilityScope.EXECUTE]),
        ("delete old files", CapabilityScope.DESTRUCTIVE),
        ("fetch weather data", CapabilityScope.NETWORK),
    ]

    print("=" * 70)
    print("Testing Improved Mission Planner Prompts")
    print("=" * 70)

    for user_input, expected in test_cases:
        manifest = await planner.plan(user_input)
        actual_scopes = manifest.required_scopes

        # Check if expected scopes are present
        if isinstance(expected, list):
            match = all(scope in actual_scopes for scope in expected)
            expected_str = ", ".join(s.value for s in expected)
        else:
            match = expected in actual_scopes
            expected_str = expected.value

        actual_str = ", ".join(s.value for s in actual_scopes)
        status = "✅" if match else "❌"

        print(f"\n{status} Input: '{user_input}'")
        print(f"   Expected: {expected_str}")
        print(f"   Actual:   {actual_str}")
        print(f"   Steps:    {len(manifest.steps)} | Mode: {manifest.execution_mode}")

    print("\n" + "=" * 70)
    print("Test complete!")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(test_planner())
