"""OTEL Tracer — optional observability for N.I.A.

If opentelemetry packages are installed, traces are exported to a local
OTLP endpoint (Jaeger/Grafana). If not installed, all tracing calls
become no-ops — zero runtime cost.
"""
from __future__ import annotations
import logging
import os

logger = logging.getLogger("NIA.Telemetry")

# Detect OTEL availability
_HAS_OTEL = False
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    _HAS_OTEL = True
except ImportError:
    trace = None  # type: ignore

# ── No-Op Fallbacks ──────────────────────────────────────────────────

class NoOpSpan:
    """Drop-in span replacement when OTEL is unavailable."""
    def set_attribute(self, key, value): pass
    def set_status(self, status): pass
    def record_exception(self, exc): pass
    def add_event(self, name, attributes=None): pass
    def end(self): pass
    def __enter__(self): return self
    def __exit__(self, *args): pass

class NoOpTracer:
    """Drop-in tracer replacement when OTEL is unavailable."""
    def start_span(self, name, **kwargs):
        return NoOpSpan()
    def start_as_current_span(self, name, **kwargs):
        return NoOpSpan()

# ── Module State ─────────────────────────────────────────────────────

_tracer_provider = None
_initialized = False

def init_tracer(
    service_name: str = "nia-framework",
    endpoint: str | None = None,
) -> None:
    """Initialize the OTEL tracer provider. No-op if OTEL is not installed."""
    global _tracer_provider, _initialized
    if not _HAS_OTEL:
        logger.debug("OpenTelemetry not installed — tracing disabled.")
        _initialized = True
        return
    if _initialized:
        return

    ep = endpoint or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
    resource = Resource.create({"service.name": service_name})
    _tracer_provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=ep, insecure=True)
    _tracer_provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(_tracer_provider)
    _initialized = True
    logger.info("OTEL tracer initialized — exporting to %s", ep)

def get_tracer(name: str = "nia") -> NoOpTracer:
    """Return an OTEL tracer, or NoOpTracer if OTEL is unavailable."""
    if _HAS_OTEL and _initialized and _tracer_provider is not None:
        return trace.get_tracer(name)  # type: ignore
    return NoOpTracer()

__all__ = ["init_tracer", "get_tracer", "NoOpSpan", "NoOpTracer"]
