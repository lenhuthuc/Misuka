"""Shared PhoBERT/WavLM encoder + regressor-head building blocks for the trained
VAD checkpoints (`model/best_text_vad.pt`, `model/best_multimodal_vad.pt`).

Every shape and submodule name here was reverse-engineered from the checkpoints'
`model_state_dict` and verified with a strict `load_state_dict()` — do not change
a shape or a submodule name without re-verifying against the checkpoint, or
loading will fail (or silently load the wrong tensor into the wrong slot).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import RobertaConfig, RobertaModel, WavLMConfig, WavLMModel

PHOBERT_VOCAB_SIZE = 64001
PHOBERT_MAX_POSITION_EMBEDDINGS = 258


def phobert_config() -> RobertaConfig:
    """vinai/phobert-base-v2's architecture (hidden=768, 12 layers, 12 heads)."""
    return RobertaConfig(
        vocab_size=PHOBERT_VOCAB_SIZE,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=PHOBERT_MAX_POSITION_EMBEDDINGS,
        type_vocab_size=1,
        pad_token_id=1,
    )


def wavlm_config() -> WavLMConfig:
    """microsoft/wavlm-base-plus's architecture — `WavLMConfig()` defaults already
    match it exactly (verified against the checkpoint's state_dict shapes)."""
    return WavLMConfig()


class PhoBertEncoder(nn.Module):
    """PhoBERT + a projection head (Linear -> LayerNorm), mean-pooled over
    non-padding tokens.

    Construct one instance per checkpoint — never share an instance between the
    text-only and multimodal models. Both fine-tune their own copy of PhoBERT's
    weights, so the checkpoints disagree past the shared architecture.
    """

    def __init__(self, proj_dim: int = 256) -> None:
        super().__init__()
        self.phobert = RobertaModel(phobert_config())
        self.projection = nn.Sequential(
            nn.Linear(self.phobert.config.hidden_size, proj_dim),
            nn.LayerNorm(proj_dim),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return self.projection(pooled)


class WavLmEncoder(nn.Module):
    """WavLM + a projection head (Linear -> LayerNorm), mean-pooled over the
    feature-extractor's own attention mask (padding excluded)."""

    def __init__(self, proj_dim: int = 256) -> None:
        super().__init__()
        self.wavlm = WavLMModel(wavlm_config())
        self.projection = nn.Sequential(
            nn.Linear(self.wavlm.config.hidden_size, proj_dim),
            nn.LayerNorm(proj_dim),
        )

    def forward(self, input_values: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        out = self.wavlm(input_values=input_values, attention_mask=attention_mask)
        hidden = out.last_hidden_state
        if attention_mask is None:
            pooled = hidden.mean(dim=1)
        else:
            # Waveform-length attention_mask -> feature-frame-length mask, since
            # the conv feature extractor downsamples time by ~320x.
            feat_mask = self.wavlm._get_feature_vector_attention_mask(hidden.shape[1], attention_mask)
            mask = feat_mask.unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return self.projection(pooled)


def build_regressor(in_dim: int, dropout: float = 0.25) -> nn.Sequential:
    """Fused-embedding -> V/A/D head shared by both checkpoints (only `in_dim`
    differs: 256 for text-only, 512 for the audio+text concatenation)."""
    return nn.Sequential(
        nn.Linear(in_dim, 256),
        nn.LayerNorm(256),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(128, 3),
    )
