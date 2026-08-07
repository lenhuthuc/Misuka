"""
Audio emotion analysis using audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim.

Output indices: 0 = arousal, 1 = dominance, 2 = valence  (all in [0, 1]).
We convert to [-1, 1] via  x * 2 - 1  before returning.
"""

import numpy as np
import torch
import torch.nn as nn
from transformers import Wav2Vec2FeatureExtractor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)

_MODEL_ID = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
_LABEL_ORDER = ["arousal", "dominance", "valence"]


class _RegressionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense    = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout  = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        return self.out_proj(x)


class _EmotionModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2   = Wav2Vec2Model(config)
        self.classifier = _RegressionHead(config)
        self.post_init()

    def forward(self, input_values: torch.Tensor):
        hidden = self.wav2vec2(input_values).last_hidden_state
        pooled = hidden.mean(dim=1)
        return self.classifier(pooled)


class AudioEmotionService:
    """Load once at startup, call predict() per request."""

    def __init__(self) -> None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device_str)
        print(f"[AudioEmotion] Loading {_MODEL_ID} on {device_str} …")
        self._extractor = Wav2Vec2FeatureExtractor.from_pretrained(_MODEL_ID)
        self._model = _EmotionModel.from_pretrained(_MODEL_ID).to(self._device)
        self._model.eval()
        print("[AudioEmotion] Ready.")

    def predict(self, audio: np.ndarray, sample_rate: int = 16000) -> tuple[float, float, float]:
        """
        Return (valence, arousal, dominance) in [-1, 1].

        Parameters
        ----------
        audio:
            Mono float32 PCM at `sample_rate` Hz.
        """
        inputs = self._extractor(
            audio,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs.input_values.to(self._device)

        with torch.no_grad():
            logits = self._model(input_values)

        probs: np.ndarray = torch.sigmoid(logits).cpu().numpy()[0]
        scores = {label: float(probs[i]) * 2.0 - 1.0 for i, label in enumerate(_LABEL_ORDER)}
        return scores["valence"], scores["arousal"], scores["dominance"]
