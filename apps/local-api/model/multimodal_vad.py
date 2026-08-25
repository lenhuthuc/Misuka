"""Multimodal V/A/D regressor: WavLM + PhoBERT, concatenated, -> regressor.

Architecture matches `model/best_multimodal_vad.pt`'s training run exactly
(`audio_encoder.{wavlm,projection}` + `text_encoder.{phobert,projection}` +
`regressor`, verified via a strict `load_state_dict()`). Trained target range
is [0, 1] — the raw head has no final activation, so callers clamp the output
themselves.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from model.encoders import PhoBertEncoder, WavLmEncoder, build_regressor

_TOKENIZER_DIR = Path(__file__).resolve().parent / "tokenizer"


class MultimodalVAD(nn.Module):
    def __init__(self, proj_dim: int = 256, dropout: float = 0.25) -> None:
        super().__init__()
        self.audio_encoder = WavLmEncoder(proj_dim)
        self.text_encoder = PhoBertEncoder(proj_dim)
        self.regressor = build_regressor(proj_dim * 2, dropout)

    def forward(
        self,
        input_values: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        audio_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        audio = self.audio_encoder(input_values, audio_attention_mask)
        text = self.text_encoder(text_input_ids, text_attention_mask)
        return self.regressor(torch.cat([audio, text], dim=-1))


def _load_state_dict(checkpoint_path: str) -> dict:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"MULTIMODAL_VAD checkpoint not found: {path}")
    try:
        ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception:
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    if isinstance(ckpt, dict):
        return ckpt
    raise RuntimeError(f"MULTIMODAL_VAD checkpoint has an unexpected format: {path}")


def load_multimodal_vad(checkpoint_path: str) -> tuple[MultimodalVAD, PreTrainedTokenizerBase]:
    """Build the model, load its weights, and load the matching (local,
    vendored) PhoBERT tokenizer — the same tokenizer the text-only checkpoint
    uses (only the fine-tuned encoder weights differ, per checkpoint). Raises
    `RuntimeError` if the checkpoint's tensors don't match this architecture
    shape-for-shape."""
    state_dict = _load_state_dict(checkpoint_path)
    model = MultimodalVAD()
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"MULTIMODAL_VAD checkpoint at {checkpoint_path!r} does not match the "
            f"MultimodalVAD architecture (model/multimodal_vad.py): {exc}"
        ) from exc
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(str(_TOKENIZER_DIR), use_fast=False)
    return model, tokenizer
