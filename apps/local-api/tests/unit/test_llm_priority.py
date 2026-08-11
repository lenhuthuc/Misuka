"""Tests for the gate that keeps background LLM work off the wire while a
user-facing turn is in flight.

ROOT CAUSE these tests pin down:

Ollama serves one request at a time per model, so the two `llm.generate()`
calls `run_memory_tasks` fires after every turn were queued in front of the
next turn's chat request. Measured on this project's CPU setup, that pushed
time-to-first-token from 3.3s (drained queue) to 16-19s (queued). Running them
concurrently instead is not an option -- decode is memory-bandwidth-bound and
two concurrent generations measured 0.46x the aggregate throughput of one.

The fix is `LLMPriorityGate`: foreground turns hold it, background callers wait
for it to clear.

The tests below pass `quiet_seconds=0.0` unless they are specifically about the
quiet window, so they exercise the foreground/interrupt mechanics in isolation.
`TestQuietWindow` covers the rest.
"""
from __future__ import annotations

import asyncio

import pytest

from core.llm_priority import LLMPriorityGate


@pytest.mark.asyncio
async def test_idle_gate_lets_background_work_start_immediately():
    """@example: no turn in flight -> `wait_until_idle()` returns True fast."""
    gate = LLMPriorityGate(settle_seconds=0.01, quiet_seconds=0.0)

    granted = await gate.wait_until_idle(timeout=1.0)

    assert granted is True


@pytest.mark.asyncio
async def test_background_waits_until_foreground_turn_finishes():
    """@example: background asks while a turn holds the gate -> it only proceeds
    after the turn's context manager exits."""
    gate = LLMPriorityGate(settle_seconds=0.01, quiet_seconds=0.0)
    order: list[str] = []

    async def foreground_turn():
        async with gate.foreground():
            await asyncio.sleep(0.1)
            order.append("foreground-done")

    async def background_work():
        granted = await gate.wait_until_idle(timeout=2.0)
        order.append("background-start")
        return granted

    fg = asyncio.create_task(foreground_turn())
    await asyncio.sleep(0.01)  # let the turn take the gate first
    granted = await background_work()
    await fg

    assert granted is True
    assert order == ["foreground-done", "background-start"]


@pytest.mark.asyncio
async def test_background_gives_up_when_turns_keep_arriving():
    """@example: the gate never clears within the timeout -> `wait_until_idle()`
    returns False so the caller can skip rather than queue behind the user."""
    gate = LLMPriorityGate(settle_seconds=0.01, quiet_seconds=0.0)

    async def busy_forever():
        async with gate.foreground():
            await asyncio.sleep(5.0)

    fg = asyncio.create_task(busy_forever())
    await asyncio.sleep(0.01)

    granted = await gate.wait_until_idle(timeout=0.2)

    assert granted is False
    fg.cancel()
    await asyncio.gather(fg, return_exceptions=True)


@pytest.mark.asyncio
async def test_overlapping_turns_keep_the_gate_closed_until_the_last_one_ends():
    """@example: a second turn starts mid-stream -> the gate stays closed until
    both have exited, not just the first."""
    gate = LLMPriorityGate(settle_seconds=0.01, quiet_seconds=0.0)
    released: list[str] = []

    async def turn(name: str, duration: float):
        async with gate.foreground():
            await asyncio.sleep(duration)
            released.append(name)

    first = asyncio.create_task(turn("first", 0.05))
    await asyncio.sleep(0.01)
    second = asyncio.create_task(turn("second", 0.15))

    granted = await gate.wait_until_idle(timeout=2.0)
    await asyncio.gather(first, second)

    assert granted is True
    assert released == ["first", "second"]


@pytest.mark.asyncio
async def test_run_when_idle_completes_work_while_nobody_is_talking():
    """@example: no turn arrives -> the work runs to completion and reports True."""
    gate = LLMPriorityGate(settle_seconds=0.01, quiet_seconds=0.0)
    done: list[str] = []

    async def work():
        await asyncio.sleep(0.05)
        done.append("finished")

    assert await gate.run_when_idle(work, timeout=1.0) is True
    assert done == ["finished"]


@pytest.mark.asyncio
async def test_run_when_idle_abandons_work_when_a_turn_arrives():
    """@example: a turn starts mid-generation -> the background coroutine is
    cancelled rather than left holding Ollama's queue, and reports False.

    This is the case a plain `wait_until_idle()` cannot cover: the work had
    already legitimately started during the user's pause."""
    gate = LLMPriorityGate(settle_seconds=0.01, quiet_seconds=0.0)
    progress: list[str] = []

    async def slow_work():
        progress.append("started")
        await asyncio.sleep(5.0)
        progress.append("should-not-reach")

    runner = asyncio.create_task(gate.run_when_idle(slow_work, timeout=1.0))
    await asyncio.sleep(0.1)  # let the work get going in the gap

    async with gate.foreground():
        completed = await runner
        await asyncio.sleep(0.01)

    assert completed is False
    assert progress == ["started"]


@pytest.mark.asyncio
async def test_run_when_idle_reports_false_when_never_quiet():
    """@example: a turn holds the gate for the whole timeout -> the work is
    never started at all."""
    gate = LLMPriorityGate(settle_seconds=0.01, quiet_seconds=0.0)
    started: list[str] = []

    async def work():
        started.append("started")

    async def busy():
        async with gate.foreground():
            await asyncio.sleep(5.0)

    fg = asyncio.create_task(busy())
    await asyncio.sleep(0.01)

    assert await gate.run_when_idle(work, timeout=0.2) is False
    assert started == []

    fg.cancel()
    await asyncio.gather(fg, return_exceptions=True)


@pytest.mark.asyncio
async def test_run_when_idle_propagates_work_failures():
    """@example: the background work raises -> the exception reaches the caller
    so it can be logged, rather than vanishing into a cancelled task."""
    gate = LLMPriorityGate(settle_seconds=0.01, quiet_seconds=0.0)

    async def failing_work():
        raise RuntimeError("ollama unreachable")

    with pytest.raises(RuntimeError, match="ollama unreachable"):
        await gate.run_when_idle(failing_work, timeout=1.0)


@pytest.mark.asyncio
async def test_gate_reopens_for_the_next_background_call():
    """@example: background work runs, a new turn comes and goes, background asks
    again -> the gate is reusable rather than one-shot."""
    gate = LLMPriorityGate(settle_seconds=0.01, quiet_seconds=0.0)

    assert await gate.wait_until_idle(timeout=1.0) is True

    async with gate.foreground():
        assert await gate.wait_until_idle(timeout=0.05) is False

    assert await gate.wait_until_idle(timeout=1.0) is True


class TestQuietWindow:
    """ROOT CAUSE these pin down:

    Holding the gate for the LLM call alone left it open from the last
    generated token onwards -- while the reply was still being spoken and the
    user had not answered it yet. Background work started in exactly that gap,
    every turn, and cancelling it when the turn arrived did not refund the
    cached-prefix eviction it had already caused in Ollama.
    """

    @pytest.mark.asyncio
    async def test_a_finished_generation_does_not_count_as_a_finished_turn(self):
        """@example: the foreground block exits -> background still waits, because
        the reply is only just starting to play."""
        gate = LLMPriorityGate(settle_seconds=0.01, quiet_seconds=0.3)

        async with gate.foreground():
            pass

        assert await gate.wait_until_idle(timeout=0.1) is False
        assert await gate.wait_until_idle(timeout=1.0) is True

    @pytest.mark.asyncio
    async def test_reply_playback_extends_the_wait_past_its_own_duration(self):
        """@example: 0.3s of audio handed to the client -> the quiet window starts
        after it has played, not when the bytes were sent."""
        gate = LLMPriorityGate(settle_seconds=0.01, quiet_seconds=0.2)

        gate.hold_active(0.3)

        assert await gate.wait_until_idle(timeout=0.35) is False
        assert await gate.wait_until_idle(timeout=1.0) is True

    @pytest.mark.asyncio
    async def test_queued_sentences_accumulate_rather_than_overwrite(self):
        """@example: three clips synthesised ahead of playback -> the reservation
        covers all three end to end, since the client plays them serially."""
        gate = LLMPriorityGate(settle_seconds=0.01, quiet_seconds=0.0)

        loop = asyncio.get_running_loop()
        before = loop.time()
        for _ in range(3):
            gate.hold_active(0.2)

        assert gate._quiet_at() - before == pytest.approx(0.6, abs=0.05)

    @pytest.mark.asyncio
    async def test_user_speaking_drops_the_playback_reservation(self):
        """@example: the user talks over a long reply -> the reservation for audio
        that is no longer playing is discarded instead of blocking curation for
        its full length."""
        gate = LLMPriorityGate(settle_seconds=0.01, quiet_seconds=0.0)

        gate.hold_active(30.0)
        gate.mark_active()

        assert await gate.wait_until_idle(timeout=1.0) is True

    @pytest.mark.asyncio
    async def test_speech_detected_abandons_work_already_running(self):
        """@example: `/emotion-vad` lands mid-generation -> the background call is
        cancelled there, rather than surviving until `/v1/chat` opens the gate a
        transcription later."""
        gate = LLMPriorityGate(settle_seconds=0.01, quiet_seconds=0.0)
        progress: list[str] = []

        async def slow_work():
            progress.append("started")
            await asyncio.sleep(5.0)
            progress.append("should-not-reach")

        runner = asyncio.create_task(gate.run_when_idle(slow_work, timeout=1.0))
        await asyncio.sleep(0.1)

        gate.mark_active()

        assert await runner is False
        assert progress == ["started"]

    @pytest.mark.asyncio
    async def test_marking_activity_does_not_latch_the_gate_shut(self):
        """@example: activity is a pulse, not a turn -> the gate is usable again
        once the quiet window passes, with no matching 'inactive' call."""
        gate = LLMPriorityGate(settle_seconds=0.01, quiet_seconds=0.0)

        gate.mark_active()

        assert await gate.wait_until_idle(timeout=1.0) is True
