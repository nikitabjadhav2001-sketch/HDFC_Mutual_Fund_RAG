"""Structured JSON logging with trace correlation (Phase 6).

Every record carries the request `trace_id` (from `app.core.telemetry`), so a
single id ties the access log line, retrieval snapshot and LLM errors together.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime

from app.config import get_settings
from app.core.telemetry import get_trace_id

_RESERVED = frozenset(
    {
        "name", "msg", "args", "levelname", "levelno", "pathname", "filename", "module",
        "exc_info", "exc_text", "stack_info", "lineno", "funcName", "created", "msecs",
        "relativeCreated", "thread", "threadName", "processName", "process", "taskName",
        "message", "asctime",
    }
)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        trace_id = get_trace_id()
        if trace_id:
            payload["trace_id"] = trace_id
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging() -> None:
    """Install the JSON formatter on the root logger (idempotent)."""
    settings = get_settings()
    root = logging.getLogger()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.handlers = [handler]
    root.setLevel(settings.log_level.upper())
    # The middleware emits a structured access log per request instead.
    access = logging.getLogger("uvicorn.access")
    access.handlers = []
    access.propagate = False
