"""Question classification and image selection."""
from __future__ import annotations

import numpy as np
import pytest

from tests.vision.conftest import NgramEmbedder
from vision.buffer import ImageRecord
from vision.router import (
    COUNT_EXIST,
    DESCRIBE,
    NEED_CLARIFY,
    READ_TEXT,
    REASON,
    SELECT_DEMONSTRATIVE,
    SELECT_ONLY,
    SELECT_SIMILARITY,
    RuleClassifier,
    find_field,
    find_object_noun,
    select_image,
)


@pytest.fixture
def classifier() -> RuleClassifier:
    return RuleClassifier()


# -- classification ----------------------------------------------------------

@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Trong ảnh có mấy con chó?", COUNT_EXIST),
        ("Có mấy người trong ảnh?", COUNT_EXIST),
        ("Trong ảnh có xe máy không?", COUNT_EXIST),
        ("Đếm giúp mình số chai trong ảnh", COUNT_EXIST),
        ("Máy tôi có bao nhiêu nhân?", READ_TEXT),
        ("RAM đang dùng bao nhiêu phần trăm?", READ_TEXT),
        ("Tổng tiền là bao nhiêu?", READ_TEXT),
        ("Trên biển ghi gì vậy?", READ_TEXT),
        ("Mô tả ảnh này", DESCRIBE),
        ("Ảnh này là gì?", DESCRIBE),
        ("Trong ảnh có gì?", DESCRIBE),
        ("Tại sao CPU chỉ chạy 1.78 GHz?", REASON),
        ("Vì sao máy mình chậm thế?", REASON),
        ("Hai cái này khác nhau chỗ nào?", REASON),
        ("Mình có nên mua cái này không?", REASON),
    ],
)
def test_rule_classifier_labels(classifier: RuleClassifier, question: str, expected: str):
    label, confidence = classifier.predict(question)

    assert label == expected
    assert 0.0 < confidence <= 1.0


def test_counting_a_noun_the_detector_cannot_emit_is_not_a_count_question():
    """"nhân" carries every counting cue there is and is a word on a screenshot."""
    assert find_object_noun("Máy tôi có bao nhiêu nhân?") is None
    assert RuleClassifier().predict("Máy tôi có bao nhiêu nhân?")[0] == READ_TEXT


def test_a_reason_question_about_a_field_is_not_a_read_question():
    """Quoting "1.78 GHz" back at someone who asked why it says 1.78 GHz."""
    assert find_field("Tại sao CPU chỉ chạy 1.78 GHz?") is not None
    assert RuleClassifier().predict("Tại sao CPU chỉ chạy 1.78 GHz?")[0] == REASON


def test_object_nouns_match_on_word_boundaries_not_substrings():
    """"tô" (bowl) lives inside "tổng"; "chó" inside "chọn"."""
    assert find_object_noun("Tổng cộng bao nhiêu?") is None
    assert find_object_noun("Nên chọn cái nào?") is None
    assert find_object_noun("Có mấy cái bát?") == ("bat", "bowl")


def test_the_longest_matching_noun_wins():
    assert find_object_noun("Có mấy chiếc xe máy?") == ("xe may", "motorcycle")


def test_field_aliases_resolve_to_a_canonical_name():
    assert find_field("bộ nhớ còn bao nhiêu?") == ("bo nho", "memory")
    assert find_field("số nhân là mấy?")[1] == "cores"
    assert find_field("tổng cộng hết bao nhiêu?")[1] == "total"


def test_threads_and_logical_processors_are_different_quantities():
    """Regression: Task Manager shows both, and they are not the same number --
    3400 process threads against 16 hardware ones. "luồng" asked of a CPU means
    the hardware count; the English "Threads" tile does not."""
    from vision.answerers import _field_of_key

    assert _field_of_key("Threads") == "threads"
    assert _field_of_key("Logical processors") == "logical_processors"
    assert find_field("máy có bao nhiêu luồng?")[1] == "logical_processors"


def test_an_empty_question_is_unknown(classifier: RuleClassifier):
    assert classifier.predict("   ") == ("UNKNOWN", 0.0)


# -- image selection ---------------------------------------------------------

def _record(image_id: str, caption: str, embedder: NgramEmbedder) -> ImageRecord:
    return ImageRecord(
        image_id=image_id,
        session_id="s",
        ts=0.0,
        clip_emb=embedder.encode_text(caption),
        fast_summary=caption,
    )


@pytest.fixture
def two_records(embedder: NgramEmbedder) -> list[ImageRecord]:
    return [
        _record("img_1", "Ảnh chụp màn hình Task Manager, CPU, Memory, Cores", embedder),
        _record("img_2", "Ảnh hoá đơn bán hàng, tổng cộng tiền thanh toán", embedder),
    ]


def test_a_single_image_session_needs_no_similarity(embedder: NgramEmbedder):
    records = [_record("img_1", "bất kỳ", embedder)]

    selection = select_image("câu hỏi hoàn toàn không liên quan", records, embedder)

    assert selection.reason == SELECT_ONLY
    assert selection.record is records[0]


def test_a_demonstrative_points_at_the_newest_image(two_records, embedder):
    selection = select_image("Ảnh này là gì?", two_records, embedder)

    assert selection.reason == SELECT_DEMONSTRATIVE
    assert selection.record.image_id == "img_2"


def test_similarity_picks_the_image_the_question_is_about(two_records, embedder):
    selection = select_image("Tổng tiền là bao nhiêu?", two_records, embedder)

    assert selection.reason == SELECT_SIMILARITY
    assert selection.record.image_id == "img_2"


def test_an_unrelated_question_asks_which_image_instead_of_guessing(two_records, embedder):
    selection = select_image("Hôm nay trời thế nào?", two_records, embedder)

    assert selection.reason == NEED_CLARIFY
    assert selection.record is None


def test_the_similarity_floor_is_configurable(two_records, embedder):
    strict = select_image("Tổng tiền là bao nhiêu?", two_records, embedder, min_similarity=0.99)

    assert strict.reason == NEED_CLARIFY


def test_a_broken_embedder_degrades_to_the_newest_image(two_records):
    class Broken:
        def encode_text(self, text):
            raise RuntimeError("text tower not exported")

    selection = select_image("Tổng tiền là bao nhiêu?", two_records, Broken())

    assert selection.record.image_id == "img_2"
    assert selection.reason == SELECT_DEMONSTRATIVE


def test_an_image_with_no_embedding_ranks_last_rather_than_raising(embedder):
    records = [
        _record("img_1", "Ảnh hoá đơn bán hàng, tổng cộng tiền thanh toán", embedder),
        ImageRecord(image_id="img_2", session_id="s", ts=1.0, clip_emb=None),
    ]

    selection = select_image("Tổng tiền là bao nhiêu?", records, embedder)

    assert selection.record.image_id == "img_1"


def test_no_images_selects_nothing(embedder):
    assert select_image("bất kỳ", [], embedder).record is None


def test_embeddings_are_unit_length(embedder: NgramEmbedder):
    vector = embedder.encode_text("Trong ảnh có mấy người?")

    assert np.isclose(np.linalg.norm(vector), 1.0, atol=1e-5)
