"""Vietnamese speech-to-text via Sherpa-ONNX (replaces Whisper).

Loads one `OfflineRecognizer` at construction (see `core/container.py` —
constructed once inside the FastAPI lifespan, never per-request) and
transcribes full-length mono 16kHz PCM. Sherpa's offline transducer recognizer
decodes an entire utterance in one call, so long audio needs no manual
chunking — that's the "handle audio of any length" behaviour the recognizer
already provides.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import sherpa_onnx

logger = logging.getLogger(__name__)


def normalize_transcript(text: str) -> str:
    """Fold Sherpa's ALL-CAPS output down to ordinary sentence case.

    The Vietnamese zipformer's BPE vocabulary is entirely uppercase (every one
    of its 2000 tokens), so *every* transcript comes out shouting. That is not
    cosmetic. Downstream, ALL CAPS is a different token sequence than normal
    text for both consumers:

    - The LLM: measured on `qwen2.5:1.5b` and `qwen2.5:3b`, "ĐỐ BẠN LÀ MỘT
      CỘNG MỘT BẰNG BAO NHIÊU" was answered as a question about the
      assistant's identity by both models, while the same sentence in
      lowercase was answered "một cộng một bằng hai" by both. Four out of four
      caps runs wrong, four out of four lowercase runs right.
    - PhoBERT: `vinai/phobert-base-v2` is trained on lowercase-dominant text,
      so an all-caps transcript is out of distribution for the V/A/D heads too.

    Capitalising only the first letter (rather than title-casing) keeps
    Vietnamese diacritics intact and matches how the training corpora of both
    downstream models are written.
    """
    folded = text.lower()
    for index, char in enumerate(folded):
        if char.isalpha():
            return folded[:index] + char.upper() + folded[index + 1:]
    return folded


class SherpaModelFilesMissing(Exception):
    """One or more of the configured ONNX/tokens files does not exist."""


class SherpaRecognizerInitError(Exception):
    """`sherpa_onnx.OfflineRecognizer.from_transducer` itself failed."""


def _require_file(label: str, path: str) -> None:
    if not path or not Path(path).is_file():
        raise SherpaModelFilesMissing(
            f"{label} not found at {path!r}. Configure SHERPA_ONNX_{label} to point at a "
            f"downloaded sherpa-onnx-zipformer-vi model — see README for download/setup."
        )


class SherpaASRService:
    """Loads a Sherpa-ONNX offline transducer recognizer once and transcribes
    full audio arrays. Vietnamese-only — the configured model has no
    `language`/`task` switches (those were Whisper-specific)."""

    def __init__(
        self,
        tokens: str,
        encoder: str,
        decoder: str,
        joiner: str,
        num_threads: int = 4,
        sample_rate: int = 16000,
        feature_dim: int = 80,
        provider: str = "cpu",
    ) -> None:
        _require_file("TOKENS", tokens)
        _require_file("ENCODER", encoder)
        _require_file("DECODER", decoder)
        _require_file("JOINER", joiner)

        try:
            self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
                tokens=tokens,
                encoder=encoder,
                decoder=decoder,
                joiner=joiner,
                num_threads=num_threads,
                sample_rate=sample_rate,
                feature_dim=feature_dim,
                decoding_method="greedy_search",
                provider=provider,
            )
        except Exception as exc:
            raise SherpaRecognizerInitError(f"Failed to initialize Sherpa-ONNX recognizer: {exc}") from exc

        self._sample_rate = sample_rate
        logger.info("SherpaASRService ready (vi, sample_rate=%d, threads=%d)", sample_rate, num_threads)

    def transcribe(self, samples: np.ndarray) -> str:
        """`samples`: mono float32 PCM at `sample_rate`, full utterance length
        (not trimmed to any fixed duration)."""
        stream = self._recognizer.create_stream()
        stream.accept_waveform(self._sample_rate, samples)
        self._recognizer.decode_stream(stream)
        text = stream.result.text.strip()
        return normalize_transcript(" ".join(text.split()))
