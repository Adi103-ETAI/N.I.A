"""Span helpers for coordinator lifecycle events."""
from __future__ import annotations
import logging
from contextlib import contextmanager
from src.core.telemetry.tracer import get_tracer, NoOpSpan

logger = logging.getLogger("NIA.Telemetry.Spans")

@contextmanager
def coordinator_span(mission_id: str, node_name: str):
    """Create a span for a coordinator node execution."""
    tracer = get_tracer("nia.coordinator")
    span = tracer.start_span(f"coordinator.{node_name}")
    try:
        span.set_attribute("mission_id", mission_id)
        span.set_attribute("node_name", node_name)
        yield span
    except Exception as exc:
        span.record_exception(exc)
        raise
    finally:
        span.end()

def record_dispatch_event(span, batch_size: int, total_spawned: int, mission_id: str) -> None:
    """Record a dispatch event on the current span."""
    span.add_event("dispatch", attributes={
        "batch_size": batch_size,
        "total_spawned": total_spawned,
        "mission_id": mission_id,
    })

def record_evaluate_event(span, status: str, results_count: int, needs_reflection: bool) -> None:
    """Record an evaluate event on the current span."""
    span.add_event("evaluate", attributes={
        "status": status,
        "results_count": results_count,
        "needs_reflection": needs_reflection,
    })

def record_reflect_event(span, step_index: str, attempt: int, original: str, reformulated: str) -> None:
    """Record a reflect/reformulate event."""
    span.add_event("reflect", attributes={
        "step_index": step_index,
        "attempt": attempt,
        "original_objective": original[:200],
        "reformulated_objective": reformulated[:200],
    })

def record_subagent_result(span, agent_id: str, status: str, tokens_used: int, role: str) -> None:
    """Record a subagent completion event."""
    span.add_event("subagent_result", attributes={
        "agent_id": agent_id,
        "status": status,
        "tokens_used": tokens_used,
        "role": role,
    })

__all__ = [
    "coordinator_span", "record_dispatch_event", "record_evaluate_event",
    "record_reflect_event", "record_subagent_result",
]
