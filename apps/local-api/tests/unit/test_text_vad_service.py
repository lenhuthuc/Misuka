import torch
import pytest

from service.text_vad_service import TextVADService, VADOutputInvalid


class _FakeTokenizer:
    def __call__(self, text, **kwargs):
        return {"input_ids": torch.zeros((1, 4), dtype=torch.long), "attention_mask": torch.ones((1, 4), dtype=torch.long)}


class _FakeModel:
    def __init__(self, output: list[float]) -> None:
        self._output = output

    def __call__(self, input_ids, attention_mask):
        return torch.tensor([self._output])


def test_predict_raw_clamps_overshoot_into_zero_one():
    service = TextVADService(_FakeModel([-0.3, 1.4, 0.5]), _FakeTokenizer())
    v, a, d = service.predict_raw("vui qua")
    assert v == 0.0
    assert a == 1.0
    assert d == 0.5


def test_predict_signed_rescales_zero_one_to_minus_one_one():
    service = TextVADService(_FakeModel([0.0, 0.5, 1.0]), _FakeTokenizer())
    v, a, d = service.predict_signed("binh thuong")
    assert v == pytest.approx(-1.0)
    assert a == pytest.approx(0.0)
    assert d == pytest.approx(1.0)


def test_predict_is_an_alias_for_predict_signed():
    service = TextVADService(_FakeModel([0.25, 0.75, 0.5]), _FakeTokenizer())
    assert service.predict("x") == service.predict_signed("x")


def test_nan_output_raises_instead_of_being_silently_clamped():
    service = TextVADService(_FakeModel([float("nan"), 0.5, 0.5]), _FakeTokenizer())
    with pytest.raises(VADOutputInvalid):
        service.predict_raw("x")


def test_inf_output_raises_instead_of_being_silently_clamped():
    service = TextVADService(_FakeModel([0.5, float("inf"), 0.5]), _FakeTokenizer())
    with pytest.raises(VADOutputInvalid):
        service.predict_raw("x")
