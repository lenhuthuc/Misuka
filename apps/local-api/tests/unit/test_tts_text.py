import pytest

from service.tts_service import exclamation_pitch_ratio, normalize_piper_text


def test_piper_text_uses_the_requested_mitsuka_pronunciation():
    assert normalize_piper_text("Mitsuka đang ở đây.") == "Mít-su-ka đang ở đây"


def test_piper_text_skips_sentence_dots_but_keeps_decimals():
    assert normalize_piper_text("Giá là 3.14. Xong.") == "Giá là 3.14 Xong"


def test_piper_text_preserves_ellipsis_as_a_pause_marker():
    assert normalize_piper_text("Đợi một chút.... nhé!") == "Đợi một chút… nhé!"


def test_exclamation_pitch_rises_toward_the_end_without_changing_length():
    ratio = exclamation_pitch_ratio(100)

    assert len(ratio) == 100
    assert ratio[0] == pytest.approx(1.0)
    assert ratio[-1] == pytest.approx(1.055)
    assert all(left <= right for left, right in zip(ratio, ratio[1:]))
