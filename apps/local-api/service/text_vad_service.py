"""Text-only V/A/D inference: word-segment -> PhoBERT -> V/A/D in [0, 1].

Used for the agent's own response text (`agent_vad`) and, as a legacy-contract
wrapper (`predict_signed`), for anything still built on the app's older
signed [-1, 1] convention (`/vad`, `brain/emotion_service.py`, response
policy, the Live2D emotion driver).
"""
from __future__ import annotations

import math

import torch
from transformers import PreTrainedTokenizerBase
from underthesea import word_tokenize

from model.text_vad import TextVAD


class VADOutputInvalid(Exception):
    """The model produced a NaN/Inf score."""


def finite_or_raise(values: list[float]) -> None:
    if any(math.isnan(v) or math.isinf(v) for v in values):
        raise VADOutputInvalid(f"VAD model produced a non-finite score: {values}")


class TextVADService:
    def __init__(self, model: TextVAD, tokenizer: PreTrainedTokenizerBase, max_length: int = 128) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._max_length = max_length

    def predict_raw(self, text: str) -> tuple[float, float, float]:
        """V/A/D in [0, 1] — the checkpoint's native trained range, clamped."""
        segmented = word_tokenize(text, format="text")
        enc = self._tokenizer(
            segmented,
            max_length=self._max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        with torch.inference_mode():
            out = self._model(enc["input_ids"], enc["attention_mask"])
        v, a, d = out[0].tolist()
        finite_or_raise([v, a, d])
        return (
            min(1.0, max(0.0, v)),
            min(1.0, max(0.0, a)),
            min(1.0, max(0.0, d)),
        )

    def predict_signed(self, text: str) -> tuple[float, float, float]:
        """V/A/D rescaled to [-1, 1] — the convention every existing
        consumer (`EmotionService`, response policy, Live2D driver) expects."""
        v, a, d = self.predict_raw(text)
        return (v * 2.0 - 1.0, a * 2.0 - 1.0, d * 2.0 - 1.0)

    # Back-compat alias: `service/vad_service.py`'s old `.predict()` name,
    # kept so `/vad`'s legacy signed-VAD contract needs no route-level change.
    def predict(self, text: str) -> tuple[float, float, float]:
        return self.predict_signed(text)
