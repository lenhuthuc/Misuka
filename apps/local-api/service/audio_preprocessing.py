"""Shared audio decode/resample/crop helpers for the ASR + multimodal-VAD
pipeline (`service/emotion_pipeline.py`).

Both branches start from the same mono/16kHz/float32 PCM (`prepare_mono_16k`):
Sherpa-ONNX gets the full-length signal (it needs the whole utterance to
produce a transcript), while WavLM gets a fixed-length center crop
(`center_crop_or_pad`) matching how the multimodal checkpoint was validated
during training.
"""
from __future__ import annotations

import io

import numpy as np
import soundfile as sf
import torch
import torchaudio


class AudioDecodeError(Exception):
    """Raised when the uploaded bytes are not a decodable audio file, or decode
    to an empty signal."""


def _decode(raw: bytes) -> tuple[np.ndarray, int]:
    try:
        data, sample_rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    except Exception as exc:
        raise AudioDecodeError(f"Cannot decode audio: {exc}") from exc
    return data, sample_rate


def _to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32)


def _resample_to_16k(audio: np.ndarray, orig_sr: int, target_sr: int = 16000) -> np.ndarray:
    if orig_sr == target_sr:
        return audio.astype(np.float32)
    waveform = torch.from_numpy(audio).unsqueeze(0)
    resampled = torchaudio.functional.resample(waveform, orig_sr, target_sr)
    return resampled.squeeze(0).numpy().astype(np.float32)


def prepare_mono_16k(raw: bytes) -> np.ndarray:
    """Decode raw uploaded audio bytes into mono float32 PCM at 16 kHz, valid
    sample range. Shared starting point for both the ASR branch (full length)
    and the multimodal-VAD branch (further center-cropped)."""
    if not raw:
        raise AudioDecodeError("Empty audio file.")
    audio, sample_rate = _decode(raw)
    if audio.size == 0:
        raise AudioDecodeError("Decoded audio is empty.")
    audio = _to_mono(audio)
    audio = _resample_to_16k(audio, sample_rate)
    return np.clip(audio, -1.0, 1.0)


def center_crop_or_pad(audio: np.ndarray, target_length: int = 64000) -> tuple[np.ndarray, np.ndarray]:
    """Center-crop to `target_length` samples (matches the multimodal
    checkpoint's validation/test preprocessing — 4s at 16kHz by default).
    Shorter audio is right-padded with zeros. Returns `(samples, attention_mask)`,
    where the mask is 0 over padded (not real audio) positions."""
    n = audio.shape[0]
    if n >= target_length:
        start = (n - target_length) // 2
        cropped = audio[start:start + target_length].astype(np.float32)
        mask = np.ones(target_length, dtype=np.int64)
        return cropped, mask

    padded = np.zeros(target_length, dtype=np.float32)
    padded[:n] = audio
    mask = np.zeros(target_length, dtype=np.int64)
    mask[:n] = 1
    return padded, mask


def normalize_for_wavlm(audio: np.ndarray) -> np.ndarray:
    """`(x - mean) / std.clamp(min=1e-7)` — must match the normalization used
    to train the multimodal checkpoint's audio tower."""
    tensor = torch.from_numpy(audio.astype(np.float32))
    normalized = (tensor - tensor.mean()) / tensor.std().clamp(min=1e-7)
    return normalized.numpy()
