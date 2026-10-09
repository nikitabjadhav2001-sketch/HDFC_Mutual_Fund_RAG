"""Prometheus metrics (Phase 6): latency histograms, request/error/token counters."""

from __future__ import annotations

from fastapi.responses import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)

HTTP_REQUESTS = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "route", "status"],
)
HTTP_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency",
    ["method", "route"],
    buckets=(0.005, 0.025, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)
RETRIEVAL_DURATION = Histogram(
    "retrieval_duration_seconds",
    "End-to-end retrieval latency (embed + dense + sparse + fusion)",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 3.0),
)
LLM_TOKENS = Counter(
    "llm_tokens_total",
    "LLM token usage reported by the provider",
    ["type"],
)
LLM_FAILURES = Counter(
    "llm_failures_total",
    "LLM stream failures",
    ["reason"],
)


def observe_http(method: str, route: str, status: int, duration_seconds: float) -> None:
    HTTP_REQUESTS.labels(method=method, route=route, status=str(status)).inc()
    HTTP_DURATION.labels(method=method, route=route).observe(duration_seconds)


def observe_llm_usage(prompt_tokens: int | None, completion_tokens: int | None) -> None:
    if prompt_tokens:
        LLM_TOKENS.labels(type="prompt").inc(prompt_tokens)
    if completion_tokens:
        LLM_TOKENS.labels(type="completion").inc(completion_tokens)


def metrics_response() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
