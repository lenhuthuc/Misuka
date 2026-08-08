"""Owns the "background LLM work must never overlap a user-facing turn"
invariant.

Ollama serves one request at a time per model, so a background
`llm.generate()` queued ahead of the next chat turn delays that turn's *first*
token by the background call's full duration. Measured on this project's CPU
setup (qwen2.5:3b, DDR4-3200): time-to-first-token was 16-19s while the
previous turn's memory tasks were still queued, versus 3.3s once the queue had
drained.

Serving them concurrently is not an alternative. Decode here is
memory-bandwidth-bound, so two concurrent generations do not share the machine
gracefully -- measured aggregate throughput *dropped* to 0.46x of a single
request. The only way to protect the foreground turn is to keep background
work strictly off the wire while a turn is in flight.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Coroutine
from typing import Any

logger = logging.getLogger(__name__)


class LLMPriorityGate:
    """Keeps background LLM calls out of the way of user-facing turns.

    Use when: a request path serves a user who is waiting (chat, streaming
    chat) and some other path issues LLM calls that nobody is waiting for
    (fact extraction, summarisation).

    Expects: every user-facing turn is wrapped in `foreground()`, and every
    background LLM call site awaits `wait_until_idle()` first. Background
    callers should re-await it between successive calls rather than assuming
    the gate stayed open.

    Returns: `foreground()` is an async context manager holding the gate open
    for its body; `wait_until_idle()` resolves once no turn is in flight and a
    short settle window has passed with the gate still clear.
    """

    def __init__(self, settle_seconds: float = 0.25) -> None:
        # Nested/overlapping foreground turns are possible (a second client
        # request arriving mid-stream), so track a depth rather than a bool.
        self._active = 0
        self._idle = asyncio.Event()
        self._idle.set()
        self._busy = asyncio.Event()
        self._settle_seconds = settle_seconds

    def foreground(self) -> "_ForegroundTurn":
        return _ForegroundTurn(self)

    async def run_when_idle(
        self,
        work: "Callable[[], Coroutine[Any, Any, Any]]",
        timeout: float | None = None,
    ) -> bool:
        """Run background LLM work in the gap between turns, abandoning it the
        moment a turn arrives.

        Waiting for idle is not enough on its own: a user pausing to read looks
        exactly like a user who has gone away, so background work legitimately
        starts during the pause and then holds the queue when they type again.
        Cancelling the coroutine drops the HTTP request, which is what makes
        Ollama stop generating rather than finish into a queue nobody is
        reading.

        Expects: `work` builds a *fresh* coroutine each call, since a cancelled
        one cannot be retried.

        Returns: True if the work ran to completion, False if it never got a
        quiet moment or was abandoned partway.
        """
        if not await self.wait_until_idle(timeout):
            return False

        task = asyncio.ensure_future(work())
        interrupted = asyncio.ensure_future(self._busy.wait())
        try:
            await asyncio.wait({task, interrupted}, return_when=asyncio.FIRST_COMPLETED)

            if task.done():
                await task  # surface any exception to the caller's handler
                return True

            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            logger.info("background LLM work abandoned: a turn started")
            return False
        finally:
            interrupted.cancel()

    async def wait_until_idle(self, timeout: float | None = None) -> bool:
        """Block until no foreground turn is in flight.

        Re-checks after the settle window because a turn that starts during
        the wait would otherwise race past the gate. Returns False if
        `timeout` elapsed first, so callers can decide whether to skip their
        work rather than pile onto a busy queue.
        """
        loop = asyncio.get_running_loop()
        deadline = None if timeout is None else loop.time() + timeout

        while True:
            remaining = None if deadline is None else deadline - loop.time()
            if remaining is not None and remaining <= 0:
                return False

            try:
                await asyncio.wait_for(self._idle.wait(), timeout=remaining)
            except asyncio.TimeoutError:
                return False

            await asyncio.sleep(self._settle_seconds)
            if self._idle.is_set():
                return True

    def _enter(self) -> None:
        self._active += 1
        self._idle.clear()
        self._busy.set()

    def _exit(self) -> None:
        self._active = max(0, self._active - 1)
        if self._active == 0:
            self._idle.set()
            self._busy.clear()


class _ForegroundTurn:
    """Async context manager returned by `LLMPriorityGate.foreground()`."""

    def __init__(self, gate: LLMPriorityGate) -> None:
        self._gate = gate

    async def __aenter__(self) -> "_ForegroundTurn":
        self._gate._enter()
        return self

    async def __aexit__(self, *_) -> None:
        self._gate._exit()
