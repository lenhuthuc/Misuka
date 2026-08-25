import io
import wave

import numpy as np
import pytest

from service.audio_preprocessing import (
    AudioDecodeError,
    center_crop_or_pad,
    normalize_for_wavlm,
    prepare_mono_16k,
)


def _wav_bytes(duration_sec: float = 1.0, sample_rate: int = 16000, channels: int = 1) -> bytes:
    n_samples = int(duration_sec * sample_rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        frame = b"\x00\x01" * channels
        wf.writeframes(frame * n_samples)
    return buf.getvalue()


def test_prepare_mono_16k_downmixes_stereo_to_mono_float32():
    wav = _wav_bytes(duration_sec=0.5, sample_rate=16000, channels=2)
    audio = prepare_mono_16k(wav)
    assert audio.dtype == np.float32
    assert audio.ndim == 1
    assert audio.shape[0] == 8000


def test_prepare_mono_16k_resamples_to_16khz():
    wav = _wav_bytes(duration_sec=1.0, sample_rate=44100, channels=1)
    audio = prepare_mono_16k(wav)
    assert audio.shape[0] == 16000


def test_prepare_mono_16k_does_not_trim_long_audio():
    """Regression for the Whisper -> Sherpa-ONNX swap: full audio must reach
    ASR, never cut down to a fixed duration."""
    wav = _wav_bytes(duration_sec=10.0, sample_rate=16000)
    audio = prepare_mono_16k(wav)
    assert audio.shape[0] == 160000


def test_prepare_mono_16k_rejects_empty_bytes():
    with pytest.raises(AudioDecodeError):
        prepare_mono_16k(b"")


def test_prepare_mono_16k_rejects_undecodable_bytes():
    with pytest.raises(AudioDecodeError):
        prepare_mono_16k(b"not a real audio file")


def test_center_crop_or_pad_crops_longer_audio_to_exact_target_length():
    audio = np.arange(160000, dtype=np.float32)  # 10s @ 16kHz
    cropped, mask = center_crop_or_pad(audio, target_length=64000)
    assert cropped.shape[0] == 64000
    assert mask.shape[0] == 64000
    assert mask.all()


def test_center_crop_or_pad_pads_shorter_audio_with_a_matching_mask():
    audio = np.ones(8000, dtype=np.float32)  # 0.5s @ 16kHz
    padded, mask = center_crop_or_pad(audio, target_length=64000)
    assert padded.shape[0] == 64000
    assert mask.shape[0] == 64000
    assert mask[:8000].all()
    assert not mask[8000:].any()
    assert not padded[8000:].any()


def test_normalize_for_wavlm_zero_means_and_unit_stds():
    audio = np.random.RandomState(0).randn(16000).astype(np.float32) * 5 + 3
    normalized = normalize_for_wavlm(audio)
    assert abs(float(normalized.mean())) < 1e-4
    assert abs(float(normalized.std()) - 1.0) < 1e-3
