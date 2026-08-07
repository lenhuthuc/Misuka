"""Pure-ASGI (not Starlette's BaseHTTPMiddleware) request-context middleware.

`@app.middleware("http")` is sugar over `BaseHTTPMiddleware`, which has a
known interaction bug with a catch-all `Exception` handler: the handler's
response gets produced correctly, but `BaseHTTPMiddleware.call_next()` can
still re-raise the original exception past it instead of returning that
response (https://github.com/encode/starlette/issues/1678). Plain ASGI
middleware doesn't go through that mechanism, so `/v1/chat` etc. can raise
freely and still get the clean `{"error": {...}}` shape from
`main.py`'s exception handler.
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Awaitable, Callable

from core.logging import bind_request_id

logger = logging.getLogger("main")

Scope = dict[str, Any]
Receive = Callable[[], Awaitable[dict[str, Any]]]
Send = Callable[[dict[str, Any]], Awaitable[None]]


class RequestContextMiddleware:
    """Assigns/echoes `X-Request-Id`, binds it for the duration of the
    request, logs one structured line per request (route, method,
    status_code, duration_ms, outcome), and cancels any in-flight TTS stream
    when a new-input path is hit — replaces three previously separate
    concerns (a request-id middleware, a logging middleware, and the
    TTS-interrupt middleware) that all needed the same "wrap the whole
    request" shape.
    """

    def __init__(self, app: Callable, interrupt_paths: frozenset[str]) -> None:
        self.app = app
        self._interrupt_paths = interrupt_paths

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope["headers"])
        request_id = headers.get(b"x-request-id", b"").decode() or str(uuid.uuid4())
        method = scope["method"]
        path = scope["path"]
        status_code = 500
        start = time.perf_counter()

        # Also stashed on scope["state"] (Starlette's Request.state), not just
        # the contextvar: ServerErrorMiddleware sits *outside* this middleware
        # and calls the catch-all exception handler after unwinding back out
        # of the `with bind_request_id(...)` block below — by then the
        # contextvar has already reset. scope["state"] is the same dict
        # Starlette re-wraps into a fresh Request there, so it survives.
        scope.setdefault("state", {})["request_id"] = request_id

        async def send_wrapper(message: dict[str, Any]) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                message["headers"] = [*message.get("headers", []), (b"x-request-id", request_id.encode())]
            await send(message)

        with bind_request_id(request_id):
            if method == "POST" and path in self._interrupt_paths:
                scope["app"].state.container.tts_coordinator.cancel_active()
            await self.app(scope, receive, send_wrapper)

        duration_ms = round((time.perf_counter() - start) * 1000, 1)
        logger.info(
            "http request",
            extra={
                "operation": "http_request",
                "component": "http",
                "route": path,
                "method": method,
                "status_code": status_code,
                "duration_ms": duration_ms,
                "outcome": "success" if status_code < 500 else "error",
            },
        )
