"""Structured logging + request/turn correlation.

Console output stays human-readable in dev; set `json_output=True` for
single-line JSON records suitable for log aggregation in production. Every
record picks up the current `request_id`/`turn_id` from context automatically
— call sites don't need to thread them through function signatures.

Never log raw audio, full prompt/response text, tokens, or secrets — only
lengths/ids/labels. See callers in api/*.py for the pattern.
"""
from __future__ import annotations

import json
import logging
import logging.handlers
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

_request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
_turn_id_var: ContextVar[str | None] = ContextVar("turn_id", default=None)

_EXTRA_FIELDS = (
    "operation", "component", "route", "method", "status_code",
    "duration_ms", "outcome", "error_code", "error_type", "retryable",
)


def get_request_id() -> str | None:
    return _request_id_var.get()


def get_turn_id() -> str | None:
    return _turn_id_var.get()


@contextmanager
def bind_request_id(request_id: str) -> Iterator[None]:
    token = _request_id_var.set(request_id)
    try:
        yield
    finally:
        _request_id_var.reset(token)


@contextmanager
def bind_turn_id(turn_id: str) -> Iterator[None]:
    """Scopes `turn_id` on every log record emitted inside the block —
    including ones from a `BackgroundTaskRegistry.spawn()` coroutine created
    while the block is active, since asyncio.Task snapshots the current
    context at creation time. This is what lets a turn be grepped end-to-end,
    browser to background memory job.
    """
    token = _turn_id_var.set(turn_id)
    try:
        yield
    finally:
        _turn_id_var.reset(token)


class _CorrelationFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id_var.get()
        record.turn_id = _turn_id_var.get()
        return True


class _JsonFormatter(logging.Formatter):
    def __init__(self, service: str, environment: str) -> None:
        super().__init__()
        self._service = service
        self._environment = environment

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "service": self._service,
            "environment": self._environment,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.request_id:  # type: ignore[attr-defined]
            payload["request_id"] = record.request_id  # type: ignore[attr-defined]
        if record.turn_id:  # type: ignore[attr-defined]
            payload["turn_id"] = record.turn_id  # type: ignore[attr-defined]
        for key in _EXTRA_FIELDS:
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        if record.exc_info:
            payload["traceback"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(
    *,
    service: str = "mitsuka-api",
    environment: str = "development",
    json_output: bool = False,
    level: str = "INFO",
    log_file: str | None = None,
) -> None:
    """Call once, before any service construction, so startup itself is logged
    with the final format. Idempotent — safe to call again (e.g. in tests).
    """
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    handler: logging.Handler
    if log_file:
        # Bounded so a long-running process can't grow the log file forever.
        handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8",
        )
    else:
        handler = logging.StreamHandler()

    if json_output:
        handler.setFormatter(_JsonFormatter(service, environment))
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        ))
    handler.addFilter(_CorrelationFilter())
    root.addHandler(handler)


@contextmanager
def log_duration(logger: logging.Logger, operation: str, **extra_fields: Any) -> Iterator[None]:
    """Log start/end of a key stage (decode, Whisper, RAG, LLM, TTS,
    background job) with `duration_ms` and `outcome`, without every call site
    hand-rolling timing + try/except. Re-raises whatever the block raises.
    """
    start = time.perf_counter()
    try:
        yield
    except Exception as exc:
        duration_ms = round((time.perf_counter() - start) * 1000, 1)
        logger.exception(
            "%s failed",
            operation,
            extra={
                "operation": operation, "duration_ms": duration_ms, "outcome": "error",
                "error_type": type(exc).__name__, **extra_fields,
            },
        )
        raise
    else:
        duration_ms = round((time.perf_counter() - start) * 1000, 1)
        logger.info(
            "%s completed",
            operation,
            extra={"operation": operation, "duration_ms": duration_ms, "outcome": "success", **extra_fields},
        )
