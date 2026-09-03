"""Which image the question is about, and which answerer should handle it.

Two independent decisions live here, in this order:

  1. image selection -- demonstrative, then the single-image shortcut, then
     CLIP similarity, then giving up and asking
  2. question classification -- a rule pass over the normalised question

Classification is deliberately behind `Classifier.predict(q) -> (label, conf)`.
The rules below are stage one; a fine-tuned PhoBERT head is stage two, and it
swaps in by constructing the pipeline with a different `Classifier`. Nothing
else in this package knows how the label was produced.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Protocol

from vision.buffer import ImageRecord
from vision.config import VI_TO_COCO, OCR_FIELD_ALIASES
from vision.models.embedder import cosine
from vision.summary import fold_case, normalize_text

logger = logging.getLogger(__name__)

# -- question labels ---------------------------------------------------------
COUNT_EXIST = "COUNT_EXIST"
READ_TEXT = "READ_TEXT"
DESCRIBE = "DESCRIBE"
REASON = "REASON"
UNKNOWN = "UNKNOWN"

# -- image-selection outcomes ------------------------------------------------
SELECT_DEMONSTRATIVE = "demonstrative"
SELECT_ONLY = "only"
SELECT_SIMILARITY = "similarity"
NEED_CLARIFY = "clarify"
SELECT_EMPTY = "empty"


def _phrase(*words: str) -> re.Pattern[str]:
    """Whole-word alternation over already-normalised text.

    Substring matching is not an option in Vietnamese: "to" (bowl) sits inside
    "tong" (total) and "cho" (dog) inside "chon" (choose), so a bare `in` test
    turns a question about a receipt total into a question about crockery.
    """
    alternation = "|".join(re.escape(normalize_text(w)) for w in words)
    return re.compile(rf"(?<![0-9a-z])(?:{alternation})(?![0-9a-z])")


# Points at the most recent image regardless of what else the question says.
# Matched against `fold_case`, not `normalize_text`: strip the accents and
# "này" (this) becomes "nay", which is half of "hôm nay" (today) -- and every
# question mentioning today would then be read as pointing at an image.
_DEMONSTRATIVE = re.compile(
    r"(?<!\w)(?:"
    + "|".join(re.escape(w) for w in (
        "này", "nầy", "đó", "ấy", "kia", "vừa gửi", "vừa rồi", "vừa nãy", "phía trên",
        # Unaccented spellings people actually type, minus the ones that are
        # also ordinary words without their accents ("nay", "do", "tren").
        "vua gui", "vua roi",
    ))
    + r")(?!\w)",
    re.UNICODE,
)

# "có mấy con chó", "bao nhiêu người", "đếm giúp"
_COUNT_CUE = _phrase("có mấy", "mấy con", "mấy cái", "bao nhiêu", "đếm", "số lượng")
# "trong ảnh có chó không?"
_EXIST_CUE = re.compile(r"(?<![0-9a-z])co(?![0-9a-z]).*(?<![0-9a-z])khong(?![0-9a-z])\s*[?.!]*$")

_READ_CUE = _phrase(
    "ghi gì", "viết gì", "chữ gì", "nội dung chữ", "đọc giúp", "ghi bao nhiêu",
    "là bao nhiêu", "giá", "tổng", "số tiền", "dòng chữ",
)

_DESCRIBE_CUE = _phrase(
    "mô tả", "miêu tả", "ảnh này là gì", "ảnh gì", "trong ảnh có gì",
    "nội dung ảnh", "nội dung bức ảnh", "tấm ảnh này là gì", "cho biết về ảnh",
    "đây là ảnh gì", "hình này là gì", "chụp cái gì", "chụp gì", "thấy gì",
)

_REASON_CUE = _phrase(
    "tại sao", "vì sao", "sao lại", "so sánh", "khác nhau", "đang làm gì",
    "có vấn đề gì", "nói lên điều gì", "nên", "có phải", "ý nghĩa", "giải thích",
    "đánh giá", "nhận xét", "bị gì", "làm sao để", "nghĩa là gì",
)

# Longest first so "xe máy" wins over "xe" and "em bé" over "bé".
_OBJECT_PHRASES: list[tuple[str, str]] = sorted(
    ((normalize_text(vi), label) for vi, label in VI_TO_COCO.items()),
    key=lambda pair: -len(pair[0]),
)
_FIELD_PHRASES: list[tuple[str, str]] = sorted(
    ((normalize_text(alias), field)
     for field, aliases in OCR_FIELD_ALIASES.items()
     for alias in aliases),
    key=lambda pair: -len(pair[0]),
)


def _find_phrase(normalized: str, phrases: list[tuple[str, str]]) -> tuple[str, str] | None:
    """First (surface, canonical) whose surface occurs as whole words."""
    for surface, canonical in phrases:
        if re.search(rf"(?<![0-9a-z]){re.escape(surface)}(?![0-9a-z])", normalized):
            return surface, canonical
    return None


def find_object_noun(question: str) -> tuple[str, str] | None:
    """(Vietnamese noun, COCO label) mentioned in the question, if any."""
    return _find_phrase(normalize_text(question), _OBJECT_PHRASES)


def find_field(question: str) -> tuple[str, str] | None:
    """(alias, canonical field) the question is asking to read, if any."""
    return _find_phrase(normalize_text(question), _FIELD_PHRASES)


class Classifier(Protocol):
    """Stage-one rules today, a PhoBERT head later. Same contract either way."""

    def predict(self, question: str) -> tuple[str, float]: ...


class RuleClassifier:
    """Keyword and regex rules over the accent-folded question.

    Rule order is the whole design. Two orderings matter and neither is
    obvious:

      COUNT_EXIST before everything, but only when the counted noun is one the
      detector can actually emit. "Máy tôi có bao nhiêu nhân?" carries every
      counting cue there is and is a question about text on a screenshot; the
      dictionary lookup is what tells the two apart.

      REASON before READ_TEXT, because a "tại sao" question about a field
      ("Tại sao CPU chỉ chạy 1.78 GHz?") names that field and would otherwise
      be answered by quoting the number back -- which is what it already says.
    """

    def predict(self, question: str) -> tuple[str, float]:
        normalized = normalize_text(question)
        if not normalized:
            return UNKNOWN, 0.0

        counting = bool(_COUNT_CUE.search(normalized)) or bool(_EXIST_CUE.search(normalized))
        if counting and find_object_noun(question) is not None:
            return COUNT_EXIST, 0.9

        if _REASON_CUE.search(normalized):
            return REASON, 0.8

        if _READ_CUE.search(normalized) or find_field(question) is not None:
            return READ_TEXT, 0.8

        if _DESCRIBE_CUE.search(normalized):
            return DESCRIBE, 0.85

        # A bare counting question whose noun is not a COCO class -- the router
        # sends UNKNOWN to the VLM, which is the right home for it.
        if counting:
            return UNKNOWN, 0.4

        return UNKNOWN, 0.0


@dataclass
class ImageSelection:
    """Which record answered, and on what grounds -- both get logged."""

    record: ImageRecord | None
    reason: str
    similarity: float = 0.0


def select_image(
    question: str,
    records: list[ImageRecord],
    embedder=None,
    *,
    min_similarity: float = 0.25,
) -> ImageSelection:
    """Pick the image a question refers to.

    Falls back to the most recent image whenever similarity cannot be
    computed: an embedder that failed to load should degrade to the behaviour
    of a one-image session, not block every multi-image question behind a
    clarification.
    """
    if not records:
        return ImageSelection(None, SELECT_EMPTY)

    if len(records) == 1:
        return ImageSelection(records[0], SELECT_ONLY)

    if _DEMONSTRATIVE.search(fold_case(question)):
        return ImageSelection(records[-1], SELECT_DEMONSTRATIVE)

    if embedder is None:
        return ImageSelection(records[-1], SELECT_DEMONSTRATIVE)

    try:
        query = embedder.encode_text(question)
    except Exception as exc:
        logger.warning("vision.router text embedding failed (%s); using latest image", exc)
        return ImageSelection(records[-1], SELECT_DEMONSTRATIVE)

    scored = [(cosine(query, record.clip_emb), record) for record in records]
    best_score, best_record = max(scored, key=lambda pair: pair[0])

    if best_score >= min_similarity:
        return ImageSelection(best_record, SELECT_SIMILARITY, round(best_score, 4))

    return ImageSelection(None, NEED_CLARIFY, round(best_score, 4))
