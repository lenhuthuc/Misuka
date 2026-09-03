from service.tts_service import normalize_piper_text


def test_piper_text_uses_the_requested_mitsuka_pronunciation():
    assert normalize_piper_text("Mitsuka đang ở đây.") == "Mít-su-ka đang ở đây"


def test_piper_text_skips_sentence_dots_but_keeps_decimals():
    assert normalize_piper_text("Giá là 3.14. Xong.") == "Giá là 3.14 Xong"
