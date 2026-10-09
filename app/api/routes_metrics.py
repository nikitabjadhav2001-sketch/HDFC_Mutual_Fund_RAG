"""Prometheus scrape endpoint (Phase 6), intentionally unauthenticated."""

from __future__ import annotations

from fastapi import APIRouter, Response

from app.core.metrics import metrics_response

router = APIRouter(tags=["metrics"])


@router.get("/metrics", include_in_schema=False)
def metrics_endpoint() -> Response:
    return metrics_response()
