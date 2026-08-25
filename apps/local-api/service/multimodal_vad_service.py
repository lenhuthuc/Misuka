"""Multimodal (WavLM audio + PhoBERT text) V/A/D inference for the user's
spoken input. Output range is [0, 1] — the checkpoint's native trained range.
"""
from __future__ import annotations

import numpy as np
import torch
from transformers import PreTrainedTokenizerBase
from underthesea import word_tokenize

from model.multimodal_vad import MultimodalVAD
from service.audio_preprocessing import normalize_for_wavlm
from service.text_vad_service import VADOutputInvalid, finite_or_raise


class MultimodalVADService:
    def __init__(self, model: MultimodalVAD, tokenizer: PreTrainedTokenizerBase, max_text_length: int = 128) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._max_text_length = max_text_length

    def predict(
        self,
        audio: np.ndarray,
        audio_attention_mask: np.ndarray,
        text: str,
    ) -> tuple[float, float, float]:
        """`audio`/`audio_attention_mask`: the multimodal checkpoint's fixed-length
        center-cropped (or zero-padded) window — see
        `service.audio_preprocessing.center_crop_or_pad`, *not* yet normalized.
        `text`: the transcript for this same audio (word-segmented here)."""
        normalized = normalize_for_wavlm(audio)
        input_values = torch.from_numpy(normalized).unsqueeze(0)
        amask = torch.from_numpy(audio_attention_mask.astype(np.int64)).unsqueeze(0)

        segmented = word_tokenize(text, format="text") if text.strip() else ""
        enc = self._tokenizer(
            segmented,
            max_length=self._max_text_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        with torch.inference_mode():
            out = self._model(input_values, enc["input_ids"], enc["attention_mask"], amask)
        v, a, d = out[0].tolist()
        finite_or_raise([v, a, d])
        return (
            min(1.0, max(0.0, v)),
            min(1.0, max(0.0, a)),
            min(1.0, max(0.0, d)),
        )


__all__ = ["MultimodalVADService", "VADOutputInvalid"]
