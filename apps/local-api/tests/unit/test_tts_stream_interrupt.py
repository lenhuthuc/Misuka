"""Deterministic unit test for the TTS stream interrupt guard.

Goes through `_stream_chunks` + `TTSInterruptCoordinator` directly rather
than the HTTP layer: neither httpx.ASGITransport nor Starlette's TestClient
preserve per-`yield` chunk boundaries in this environment (both collect the
streaming response before returning it to the test), so interruption timing
can't be observed through a real request. This pairing is exactly what
REFACTOR_PLAN.md Phase 3 turned the old bare `active_tts_id` counter into.
"""
from api.tts import _stream_chunks
from core.tts_coordinator import TTSInterruptCoordinator


async def test_stream_chunks_delivers_everything_when_uninterrupted():
    coordinator = TTSInterruptCoordinator()
    my_id = coordinator.begin_turn()
    payload = b"0123456789" * 10  # 100 bytes

    chunks = [c async for c in _stream_chunks(payload, coordinator, my_id, chunk_size=10)]

    assert b"".join(chunks) == payload
    assert len(chunks) == 10


async def test_stream_chunks_stops_once_a_newer_generation_claims_the_id():
    coordinator = TTSInterruptCoordinator()
    my_id = coordinator.begin_turn()
    payload = b"0123456789" * 10

    chunks = []
    async for chunk in _stream_chunks(payload, coordinator, my_id, chunk_size=10):
        chunks.append(chunk)
        if len(chunks) == 2:
            coordinator.begin_turn()  # a newer request claims a new generation

    assert b"".join(chunks) == payload[:20]


async def test_stream_chunks_yields_nothing_if_already_stale_on_entry():
    coordinator = TTSInterruptCoordinator()
    my_id = coordinator.begin_turn()
    coordinator.begin_turn()  # someone else already claimed a newer generation
    payload = b"0123456789"

    chunks = [c async for c in _stream_chunks(payload, coordinator, my_id, chunk_size=10)]

    assert chunks == []


def test_cancel_active_invalidates_the_current_generation_without_claiming_a_new_one():
    coordinator = TTSInterruptCoordinator()
    my_id = coordinator.begin_turn()

    coordinator.cancel_active()

    assert coordinator.is_cancelled(my_id) is True
