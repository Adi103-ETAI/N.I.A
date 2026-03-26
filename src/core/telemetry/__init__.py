"""OTEL Telemetry package for N.I.A. Sprint 6."""
from __future__ import annotations

from src.core.telemetry.tracer import init_tracer, get_tracer, NoOpSpan, NoOpTracer
from src.core.telemetry.middleware import TokenCounter, get_token_counter
from src.core.telemetry.spans import (
    coordinator_span, record_dispatch_event, record_evaluate_event,
    record_reflect_event, record_subagent_result,
)

__all__ = [
    "init_tracer", "get_tracer", "NoOpSpan", "NoOpTracer",
    "TokenCounter", "get_token_counter",
    "coordinator_span", "record_dispatch_event", "record_evaluate_event",
    "record_reflect_event", "record_subagent_result",
]
