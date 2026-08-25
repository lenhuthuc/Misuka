import numpy as np
import torch
import pytest

from service.multimodal_vad_service import MultimodalVADService, VADOutputInvalid


class _FakeTokenizer:
    def __call__(self, text, **kwargs):
        return {"input_ids": torch.zeros((1, 4), dtype=torch.long), "attention_mask": torch.ones((1, 4), dtype=torch.long)}


class _FakeModel:
    def __init__(self, output: list[float]) -> None:
        self._output = output
        self.calls: list[tuple] = []

    def __call__(self, input_values, text_input_ids, text_attention_mask, audio_attention_mask=None):
        self.calls.append((input_values.shape, text_input_ids.shape, audio_attention_mask.shape if audio_attention_mask is not None else None))
        return torch.tensor([self._output])


def test_predict_clamps_and_normalizes_audio_before_the_model_sees_it():
    model = _FakeModel([0.2, 0.6, 0.9])
    service = MultimodalVADService(model, _FakeTokenizer())
    audio = np.random.RandomState(1).randn(64000).astype(np.float32) * 3 + 2
    mask = np.ones(64000, dtype=np.int64)

    v, a, d = service.predict(audio, mask, "xin chao")

    assert (v, a, d) == pytest.approx((0.2, 0.6, 0.9))
    input_values_shape, text_ids_shape, audio_mask_shape = model.calls[-1]
    assert input_values_shape == (1, 64000)
    assert audio_mask_shape == (1, 64000)


def test_predict_handles_empty_transcript_without_erroring():
    model = _FakeModel([0.5, 0.5, 0.5])
    service = MultimodalVADService(model, _FakeTokenizer())
    audio = np.zeros(64000, dtype=np.float32)
    mask = np.ones(64000, dtype=np.int64)

    v, a, d = service.predict(audio, mask, "")

    assert (v, a, d) == (0.5, 0.5, 0.5)


def test_nan_output_raises():
    model = _FakeModel([float("nan"), 0.5, 0.5])
    service = MultimodalVADService(model, _FakeTokenizer())
    audio = np.zeros(64000, dtype=np.float32)
    mask = np.ones(64000, dtype=np.int64)

    with pytest.raises(VADOutputInvalid):
        service.predict(audio, mask, "xin chao")
