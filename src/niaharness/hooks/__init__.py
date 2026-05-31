"""Hooks exports."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from niaharness.hooks.events import HookEvent
    from niaharness.hooks.executor import HookExecutionContext, HookExecutor
    from niaharness.hooks.loader import HookRegistry
    from niaharness.hooks.types import AggregatedHookResult, HookResult

__all__ = [
    "AggregatedHookResult",
    "HookEvent",
    "HookExecutionContext",
    "HookExecutor",
    "HookRegistry",
    "HookResult",
    "load_hook_registry",
]


def __getattr__(name: str):
    if name == "HookEvent":
        from niaharness.hooks.events import HookEvent

        return HookEvent
    if name in {"HookExecutionContext", "HookExecutor"}:
        from niaharness.hooks.executor import HookExecutionContext, HookExecutor

        return {
            "HookExecutionContext": HookExecutionContext,
            "HookExecutor": HookExecutor,
        }[name]
    if name in {"HookRegistry", "load_hook_registry"}:
        from niaharness.hooks.loader import HookRegistry, load_hook_registry

        return {
            "HookRegistry": HookRegistry,
            "load_hook_registry": load_hook_registry,
        }[name]
    if name in {"AggregatedHookResult", "HookResult"}:
        from niaharness.hooks.types import AggregatedHookResult, HookResult

        return {
            "AggregatedHookResult": AggregatedHookResult,
            "HookResult": HookResult,
        }[name]
    raise AttributeError(name)
