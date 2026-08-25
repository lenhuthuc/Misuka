"""Text-only V/A/D regressor: PhoBERT -> projection -> regressor.

Architecture matches `model/best_text_vad.pt`'s training run exactly
(`text_encoder.{phobert,projection}` + `regressor`, verified via a strict
`load_state_dict()`). Trained target range is [0, 1] — the raw head has no
final activation, so callers clamp the output themselves.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from model.encoders import PhoBertEncoder, build_regressor

_TOKENIZER_DIR = Path(__file__).resolve().parent / "tokenizer"


class TextVAD(nn.Module):
    def __init__(self, proj_dim: int = 256, dropout: float = 0.25) -> None:
        super().__init__()
        self.text_encoder = PhoBertEncoder(proj_dim)
        self.regressor = build_regressor(proj_dim, dropout)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.regressor(self.text_encoder(input_ids, attention_mask))


def _load_state_dict(checkpoint_path: str) -> dict:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"TEXT_VAD checkpoint not found: {path}")
    try:
        ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception:
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    if isinstance(ckpt, dict):
        return ckpt
    raise RuntimeError(f"TEXT_VAD checkpoint has an unexpected format: {path}")


def load_text_vad(checkpoint_path: str) -> tuple[TextVAD, PreTrainedTokenizerBase]:
    """Build the model, load its weights, and load the matching (local,
    vendored) PhoBERT tokenizer. Raises `RuntimeError` if the checkpoint's
    tensors don't match this architecture shape-for-shape."""
    state_dict = _load_state_dict(checkpoint_path)
    model = TextVAD()
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"TEXT_VAD checkpoint at {checkpoint_path!r} does not match the TextVAD "
            f"architecture (model/text_vad.py): {exc}"
        ) from exc
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(str(_TOKENIZER_DIR), use_fast=False)
    return model, tokenizer
