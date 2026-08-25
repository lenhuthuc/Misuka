"""Tests for the gate that keeps heavy background work out of the way of the
reply the user is waiting to hear.

The suite used to also cover `run_when_idle` and the quiet window -- the wider
invariant that background *LLM* calls must never overlap a turn. Both went with
the memory curator, the only component that ever issued them; what is left is
the narrower speech question below.
"""
from __future__ import annotations

import asyncio

import pytest

from core.llm_priority import LLMPriorityGate


class TestSpeechFlag:
    """ROOT CAUSE these pin down:

    `run_memory_tasks` never went through the gate at all -- only LLM work did.
    So the exchange's embedding was spawned the moment the last token was
    generated, which is the moment Piper starts rendering the reply, and the
    two fought over the same cores. Observed in the server log: "saved
    conversation turn" at 15:48:23, "indexed exchange" 3s later at 15:48:26,
    and the speech response landing at 15:48:26 -- the user waited through the
    embedding to hear anything.

    `wait_until_spoken` is the gate for work like that: not "is the LLM runner
    free" (it is, that is the problem) but "has the user heard the reply yet".
    """

    @pytest.mark.asyncio
    async def test_the_flag_is_raised_by_the_user_speaking(self):
        """@example: `/emotion-vad` marks the turn -> a reply is owed and heavy
        background work waits for it, rather than running against synthesis."""
        gate = LLMPriorityGate()

        gate.mark_active()

        assert await gate.wait_until_spoken(timeout=0.1) is False

    @pytest.mark.asyncio
    async def test_the_client_reporting_a_drained_queue_releases_the_work(self):
        """@example: the last clip finished playing -> POST /speech/finished, and
        the embedding starts immediately rather than at the timeout."""
        gate = LLMPriorityGate()
        gate.mark_active()

        waiter = asyncio.create_task(gate.wait_until_spoken(timeout=2.0))
        await asyncio.sleep(0.05)
        assert waiter.done() is False

        gate.speech_finished()

        assert await waiter is True

    @pytest.mark.asyncio
    async def test_reserved_playback_releases_the_work_without_any_signal(self):
        """@example: the page was closed mid-reply, so no signal ever comes -> the
        durations reserved by each synthesised clip stand in for it, once the
        client has also stopped asking for clips."""
        gate = LLMPriorityGate(speech_lull_seconds=0.15)
        gate.mark_active()
        gate.hold_active(0.2)

        assert await gate.wait_until_spoken(timeout=0.1) is False
        assert await gate.wait_until_spoken(timeout=1.0) is True

    @pytest.mark.asyncio
    async def test_the_gap_between_two_sentences_is_not_the_end_of_the_reply(self):
        """@example: a two-sentence reply whose second clip is still rendering.

        ROOT CAUSE this pins down. `hold_active` only ever covers the clips the
        client has already asked for, so on a multi-sentence reply the
        reservation runs dry in every gap -- and the gap is exactly the moment
        the next sentence is being rendered. Observed in the server log: clip
        one returned at 16:47:12, "indexed exchange" fired at 16:47:18 and clip
        two landed in the same second, six seconds after the one before it.
        The embedding was released by the gap and then made the gap longer.

        A lull with no new clip, not an elapsed reservation, is what says the
        client has stopped.
        """
        gate = LLMPriorityGate(speech_lull_seconds=0.4)
        gate.mark_active()
        gate.hold_active(0.05)  # a short first sentence

        # The reservation is long spent, but the reply is not over.
        assert await gate.wait_until_spoken(timeout=0.2) is False

        gate.hold_active(0.05)  # the second sentence finally rendered
        assert await gate.wait_until_spoken(timeout=0.2) is False

        gate.speech_finished()
        assert await gate.wait_until_spoken(timeout=1.0) is True

    @pytest.mark.asyncio
    async def test_nothing_owed_means_no_wait(self):
        """@example: a text-only client that never speaks a reply -> background
        work is not held behind a signal that will never come."""
        gate = LLMPriorityGate()

        assert await gate.wait_until_spoken(timeout=1.0) is True

    @pytest.mark.asyncio
    async def test_a_turn_in_flight_holds_the_work_even_once_spoken(self):
        """@example: the user answered before the previous exchange was indexed ->
        the embedding stays off the CPU the new turn is generating on."""
        gate = LLMPriorityGate()
        gate.speech_finished()

        async with gate.foreground():
            assert await gate.wait_until_spoken(timeout=0.1) is False

        assert await gate.wait_until_spoken(timeout=1.0) is True

    @pytest.mark.asyncio
    async def test_the_next_utterance_re_arms_the_flag(self):
        """@example: the user talks over the reply -> the wait is now for the reply
        to *that*, not for the abandoned one. This is what makes an interrupted
        turn's indexing wait for the conversation to settle instead of landing
        in the middle of the next answer.
        """
        gate = LLMPriorityGate()
        gate.mark_active()
        gate.speech_finished()

        gate.mark_active()

        assert await gate.wait_until_spoken(timeout=0.1) is False
