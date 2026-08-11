"""Tests for the WAV duration probe that feeds the LLM priority gate.

Why the duration matters: a synthesised reply handed to the client occupies the
user for exactly its playback length, and that is the window background LLM work
must stay out of. Reading it wrong in either direction is a real cost --
too short and curation starts while the assistant is still talking, too long and
the queue only drains between sessions.
"""
from __future__ import annotations

import io
import wave

import pytest

from api.tts import wav_duration_seconds


def _wav(frames: int, rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(b"\x00\x00" * frames)
    return buf.getvalue()


def test_reads_playback_length_from_the_header():
    """@example: 24000 frames at 16 kHz -> 1.5s."""
    assert wav_duration_seconds(_wav(24000)) == pytest.approx(1.5)


def test_scales_with_sample_rate_rather_than_byte_count():
    """@example: the same frame count at 22.05 kHz plays for less time.

    Piper and Kokoro voices run at different rates, so byte length alone would
    mis-size the hold whenever the configured voice changed.
    """
    assert wav_duration_seconds(_wav(22050, rate=22050)) == pytest.approx(1.0)


def test_empty_audio_reserves_nothing():
    """@example: a zero-frame clip -> 0.0, so `hold_active` short-circuits."""
    assert wav_duration_seconds(_wav(0)) == 0.0


def test_unreadable_bytes_skip_the_reservation_instead_of_raising():
    """@example: not a WAV at all -> 0.0.

    This runs on the success path of a synthesis request that already produced
    audio; a header the `wave` module dislikes must not turn that into a 500.
    """
    assert wav_duration_seconds(b"not a wav file") == 0.0
