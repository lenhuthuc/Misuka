"""Registry for fire-and-forget background tasks (memory save, fact
extraction) so they can be logged on failure and drained on shutdown instead
of leaking as untracked `asyncio.create_task` calls.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from typing import Any

logger = logging.getLogger(__name__)


class BackgroundTaskRegistry:
    """Use when: spawning work that must outlive the HTTP response but should
    still be logged on failure and cancelled/awaited during shutdown.

    Expects: `spawn()` is called with a fresh coroutine per call.

    Returns: `spawn()` returns the created `asyncio.Task`; `drain()` waits
    for all currently-tracked tasks (cancelling stragglers past `timeout`).
    """

    def __init__(self) -> None:
        self._tasks: set[asyncio.Task] = set()

    def spawn(self, coro: Coroutine[Any, Any, Any], *, name: str) -> asyncio.Task:
        task = asyncio.create_task(coro, name=name)
        self._tasks.add(task)
        task.add_done_callback(self._on_done)
        return task

    def _on_done(self, task: asyncio.Task) -> None:
        self._tasks.discard(task)
        if task.cancelled():
            return
        if exc := task.exception():
            # Not inside the task's own except block (this runs from the
            # event loop's callback dispatch), so sys.exc_info() wouldn't
            # find it — pass the exception object directly to keep the
            # traceback logger.exception() would normally capture.
            logger.error(
                "background task '%s' failed", task.get_name(),
                exc_info=exc,
                extra={"operation": task.get_name(), "outcome": "error", "error_type": type(exc).__name__},
            )

    async def drain(self, timeout: float = 5.0) -> None:
        """Await in-flight tasks, cancelling whatever hasn't finished after `timeout`."""
        if not self._tasks:
            return
        pending = set(self._tasks)
        _done, still_pending = await asyncio.wait(pending, timeout=timeout)
        for task in still_pending:
            task.cancel()
        if still_pending:
            await asyncio.gather(*still_pending, return_exceptions=True)
