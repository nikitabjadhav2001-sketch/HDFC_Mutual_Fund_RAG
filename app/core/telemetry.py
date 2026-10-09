"""Request tracing: trace_id context, access logs, OpenTelemetry spans (Phase 6).

Spans follow api → retrieve → llm: the HTTP middleware opens the server span,
`Retriever.retrieve` and `LLMProvider.stream` open children via
`start_child_span`. Export is enabled by setting OTEL_EXPORTER_OTLP_ENDPOINT.
"""

from __future__ import annotations

import logging
import uuid
from contextvars import ContextVar
from time import perf_counter

from app.config import get_settings
from app.core.metrics import observe_http

logger = logging.getLogger(__name__)

_trace_id: ContextVar[str] = ContextVar("trace_id", default="")
_tracing_configured = False


def get_trace_id() -> str:
    return _trace_id.get()


def setup_tracing() -> None:
    """Configure an OTLP tracer provider once, when an endpoint is configured."""
    global _tracing_configured
    if _tracing_configured:
        return
    _tracing_configured = True
    endpoint = get_settings().otel_exporter_otlp_endpoint
    if not endpoint:
        return

    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider(resource=Resource.create({"service.name": "rag-api"}))
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{endpoint.rstrip('/')}/v1/traces"))
    )
    trace.set_tracer_provider(provider)
    logger.info("otel tracing enabled", extra={"otel_endpoint": endpoint})


def start_child_span(name: str, **attributes):
    """Open a child span of the current request span (no-op without a provider)."""
    from opentelemetry import trace

    tracer = trace.get_tracer("app")
    span = tracer.start_span(name)
    for key, value in attributes.items():
        span.set_attribute(key, value)
    return span


def _route_template(request) -> str:
    route = request.scope.get("route")
    return getattr(route, "path", None) or "unmatched"


def install_request_middleware(app) -> None:
    """trace_id assignment + structured access log + server span + /metrics."""
    from starlette.middleware.base import BaseHTTPMiddleware

    class RequestContextMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            incoming = request.headers.get("x-request-id", "").strip()
            trace_id = incoming[:64] if incoming else uuid.uuid4().hex
            token = _trace_id.set(trace_id)
            span = start_child_span(
                "http.request",
                **{
                    "http.method": request.method,
                    "http.target": request.url.path,
                },
            )
            started = perf_counter()
            status_code = 500
            try:
                response = await call_next(request)
                status_code = response.status_code
                span.set_attribute("http.status_code", status_code)
                response.headers["X-Request-ID"] = trace_id
                return response
            finally:
                duration = perf_counter() - started
                route = _route_template(request)
                span.set_attribute("http.route", route)
                span.end()
                observe_http(request.method, route, status_code, duration)
                logger.info(
                    "request",
                    extra={
                        "event": "http_request",
                        "method": request.method,
                        "route": route,
                        "status": status_code,
                        "duration_ms": round(duration * 1000, 1),
                    },
                )
                _trace_id.reset(token)

    app.add_middleware(RequestContextMiddleware)
