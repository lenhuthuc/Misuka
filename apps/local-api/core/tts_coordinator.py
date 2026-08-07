"""Owns the "which TTS generation is currently allowed to keep streaming?"
invariant that used to be a bare `app.state.active_tts_id` counter mutated
from three different call sites (middleware + two TTS routes).
"""


class TTSInterruptCoordinator:
    """Tracks the current TTS generation id.

    Use when: a new turn of user input (or a new TTS request) should
    invalidate any TTS stream already in flight — the classic "barge-in"
    interrupt.

    Expects: `begin_turn()` is called once per new TTS-eligible request;
    `is_cancelled(my_id)` is polled by the in-flight stream between chunks.

    Returns: `begin_turn()` returns the generation id the caller now owns.
    """

    def __init__(self) -> None:
        self._active_id = 0

    def begin_turn(self) -> int:
        self._active_id += 1
        return self._active_id

    def is_cancelled(self, my_id: int) -> bool:
        return self._active_id != my_id

    def cancel_active(self) -> None:
        self._active_id += 1
