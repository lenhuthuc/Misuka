"""Answers one question for background work: "has the user heard the reply yet?"

Embedding a finished exchange is CPU-heavy but never touches the LLM runner,
so the thing it must stay out of the way of is *synthesis*, not generation.
Waiting for the generation to finish is the wrong gate and actively harmful:
it opens the moment the last token is produced, which is precisely when Piper
starts rendering the reply, so the embedding and the synthesis then fight over
the same cores and the user waits longer to hear anything. Measured on this
project's CPU, the exchange embedding ran for ~3s starting the same second the
client asked for speech.

So the gate tracks the conversation rather than the generation. Call sites feed
it what the server already knows: `/emotion-vad` means the user is speaking now
(`mark_active`), a `/v1/audio/speech` response means that many seconds of reply
are about to be played (`hold_active`), and `/v1/audio/speech/finished` means
the client's playback queue drained (`speech_finished`).

This module used to carry a second, wider invariant -- background *LLM* calls
must never overlap a user-facing turn -- for the memory curator, which mined
finished exchanges for durable facts on the same Ollama runner the chat turn
needs. That component is gone, and with it `run_when_idle`, the quiet-window
machinery it waited on, and the interrupt channel that abandoned its work when
a turn arrived. Nothing on the server issues background LLM calls any more; if
something ever does, that gate has to come back with it.
"""
from __future__ import annotations

import asyncio


class LLMPriorityGate:
    """Keeps heavy background work out of the way of reply synthesis.

    Use when: a request path serves a user who is waiting (chat, streaming
    chat) and some other path does CPU-heavy work nobody is waiting for
    (embedding a finished exchange).

    Expects: every user-facing turn is wrapped in `foreground()`, and every
    other sign of an ongoing conversation is reported through `mark_active()` /
    `hold_active()` / `speech_finished()`.

    Returns: `foreground()` is an async context manager holding the gate open
    for its body; `wait_until_spoken()` resolves as soon as the reply has
    finished playing.
    """

    def __init__(self, speech_lull_seconds: float = 6.0) -> None:
        # Nested/overlapping foreground turns are possible (a second client
        # request arriving mid-stream), so track a depth rather than a bool.
        self._active = 0
        self._idle = asyncio.Event()
        self._idle.set()
        self._speech_lull_seconds = speech_lull_seconds
        # Loop-clock timestamps. 0.0 reads as "long ago", which is what a
        # freshly started process should mean: nothing to stay out of the way of.
        self._playback_until = 0.0
        # "The reply to what the user just said has not finished being spoken."
        # Set the moment the user's voice arrives, cleared when the client says
        # its playback queue drained. Starts set (nothing is owed at boot).
        self._spoken = asyncio.Event()
        self._spoken.set()
        # Whether any audio has been synthesised since the flag was raised. Until
        # one clip exists, `_playback_until` is a stale reservation from an older
        # turn and says nothing about this one.
        self._playback_held = False
        # When the most recent clip was handed over. A reply is spoken a
        # sentence at a time, so the reservation running dry means "the client
        # has not asked for the next clip *yet*", not "there is no next clip".
        self._last_hold = 0.0

    def foreground(self) -> "_ForegroundTurn":
        return _ForegroundTurn(self)

    def mark_active(self) -> None:
        """Report conversation activity that is not itself an LLM turn -- the
        user has started speaking.

        Also drops any outstanding playback hold: getting here means the user
        talked over the reply, so the audio the hold was reserving time for is
        no longer playing.

        This is also where the "a reply is owed and unspoken" flag is raised.
        Every new utterance re-arms it, including one that interrupts a reply
        still being spoken -- that reply is now dead, and the one being waited
        for is the one this utterance will produce.
        """
        self._playback_until = 0.0
        self._playback_held = False
        self._last_hold = 0.0
        self._spoken.clear()

    def hold_active(self, seconds: float) -> None:
        """Reserve conversation time that has not elapsed yet -- reply audio
        just handed to the client, which will play it back.

        Sentences are synthesised ahead of playback, so several holds can be
        placed while earlier audio is still sounding. They accumulate onto the
        end of the existing reservation rather than overwriting it, which is
        how a serial playback queue actually behaves.
        """
        if seconds <= 0:
            return
        now = self._now()
        self._playback_until = max(self._playback_until, now) + seconds
        self._playback_held = True
        self._last_hold = now

    def speech_finished(self) -> None:
        """The client's playback queue has drained -- the reply has been heard.

        This is the signal `wait_until_spoken()` exists for. The reservations
        `hold_active` accumulates are only an estimate of the same moment
        (they assume playback starts the instant synthesis returns), so the
        estimate is collapsed to now rather than left running past the audio.
        """
        self._playback_until = min(self._playback_until, self._now())
        self._last_hold = 0.0
        self._spoken.set()

    @staticmethod
    def _now() -> float:
        return asyncio.get_running_loop().time()

    def _reservation_ends_at(self) -> float:
        """The earliest moment the *unsignalled* end of a reply may be assumed.

        A reply is synthesised one sentence at a time, so `_playback_until`
        covers only the clips that have been asked for so far. Treating it as
        the end of the reply is wrong in the one case that matters: the client
        is mid-reply and its next clip is still rendering, which is precisely
        when the reservation runs dry. Measured on a two-sentence reply, the
        exchange embedding started 2s after the first clip and 4s before the
        second one arrived -- so it competed with the render it was waiting to
        avoid, and made the gap between the two sentences it was sitting in
        longer still.

        Hence the second term: a lull with no new clip is what says the client
        has stopped, and the reservation only expires once both have. The
        explicit `speech_finished()` signal short-circuits all of it, so this
        only ever delays a client that went away, which `timeout` already
        covers.
        """
        return max(self._playback_until, self._last_hold + self._speech_lull_seconds)

    async def wait_until_spoken(self, timeout: float | None = None) -> bool:
        """Block until the reply to the user's last utterance has been spoken.

        Use when: background work is CPU-heavy but does *not* touch the LLM
        runner -- embedding a finished exchange, say. Waiting on the
        generation instead is the wrong gate for that work: it opens when the
        last token is produced, which is precisely when Piper starts rendering
        the reply, so the embedding and the synthesis then fight over the same
        cores and the user waits longer to hear anything. Measured on this
        project's CPU: the exchange embedding ran for ~3s starting at the same
        second the client asked for speech.

        Expects: `mark_active()` on every user utterance (that raises the flag)
        and `speech_finished()` from the client when its playback queue drains.
        A client that goes away mid-reply never sends the latter, so the
        reservations from `hold_active()` are the backstop, and `timeout` is
        the backstop behind that.

        Returns: True once the reply has been spoken and no foreground turn is
        in flight; False if `timeout` elapsed first. Callers holding durable
        work should still run it on False -- late is better than dropped.
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

            if self._spoken.is_set():
                return True

            # No signal yet. Fall back on the reservations -- but only once
            # the client has also stopped asking for clips; see
            # `_reservation_ends_at`.
            if self._playback_held and self._reservation_ends_at() <= loop.time():
                return True

            remaining = None if deadline is None else deadline - loop.time()
            if remaining is not None and remaining <= 0:
                return False
            # Re-check when the reservation runs out, or as soon as the client
            # reports the queue drained -- whichever happens first.
            # Nothing synthesised yet: there is no reservation to time out on,
            # so poll slowly and let `_spoken.wait()` do the real waking.
            reserved = self._reservation_ends_at() - loop.time() if self._playback_held else 1.0
            wake_in = max(reserved, 0.05)
            if remaining is not None:
                wake_in = min(wake_in, remaining)
            try:
                await asyncio.wait_for(self._spoken.wait(), timeout=wake_in)
            except asyncio.TimeoutError:
                continue

    def _enter(self) -> None:
        self._active += 1
        self._idle.clear()

    def _exit(self) -> None:
        self._active = max(0, self._active - 1)
        if self._active == 0:
            self._idle.set()


class _ForegroundTurn:
    """Async context manager returned by `LLMPriorityGate.foreground()`."""

    def __init__(self, gate: LLMPriorityGate) -> None:
        self._gate = gate

    async def __aenter__(self) -> "_ForegroundTurn":
        self._gate._enter()
        return self

    async def __aexit__(self, *_) -> None:
        self._gate._exit()
