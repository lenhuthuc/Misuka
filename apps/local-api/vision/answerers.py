"""The three ways an image question gets answered.

Each answerer returns `None` when it cannot answer, and the pipeline walks them
in order. `None` is the whole contract: it is what turns "the detector found no
dogs in a photograph" (unreliable -- ask the VLM) into a different outcome from
"the detector found no dogs in a screenshot" (reliable -- say no), without the
pipeline needing to know why.
"""
from __future__ import annotations

import logging
import re

from vision.buffer import ImageRecord
from vision.config import COCO_TO_VI, OCR_FIELD_ALIASES
from vision.models.vlm_client import DESCRIBE_QUESTION, VLMClient
from vision.router import find_field, find_object_noun
from vision.summary import TEXT_TAGS, normalize_text

logger = logging.getLogger(__name__)

_HAS_DIGIT = re.compile(r"\d")

# key alias -> canonical field, for reading an `ocr_pairs` key back.
_ALIAS_TO_FIELD: dict[str, str] = {
    normalize_text(alias): field
    for field, aliases in OCR_FIELD_ALIASES.items()
    for alias in aliases
}


def _field_of_key(key: str) -> str | None:
    """Canonical field for an OCR key, exact then prefix.

    Prefix rather than exact only: OCR keys arrive with trailing punctuation
    and stray units already stripped, but real panes still say "Base speed"
    where the alias table says "base speed" and "Memory usage" where it says
    "memory".
    """
    normalized = normalize_text(key)
    if not normalized:
        return None
    if normalized in _ALIAS_TO_FIELD:
        return _ALIAS_TO_FIELD[normalized]
    for alias, field in _ALIAS_TO_FIELD.items():
        if normalized.startswith(alias + " ") or normalized.endswith(" " + alias):
            return field
    return None


def _token_overlap(a: str, b: str) -> float:
    """Jaccard over word tokens -- the fuzzy tier of key matching."""
    ta, tb = set(normalize_text(a).split()), set(normalize_text(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def from_detections(record: ImageRecord, question: str) -> str | None:
    """Count or existence, straight from the detector.

    Returns None -- deferring to the VLM -- in the two cases where a zero is
    not evidence of absence: the question names something COCO has no class
    for, and a photograph where the detector simply may have missed it. On a
    screenshot or a document a zero is trustworthy, because COCO objects do not
    hide in a settings pane.
    """
    found = find_object_noun(question)
    if found is None:
        return None
    noun, label = found

    count = sum(1 for det in record.detections if det.get("label") == label)
    vi_name = COCO_TO_VI.get(label, noun)

    if count > 0:
        return f"Trong ảnh có {count} {vi_name}."

    if TEXT_TAGS.intersection(record.tags):
        return f"Trong ảnh không có {vi_name} nào — đây là ảnh chụp màn hình/tài liệu."

    return None


def from_ocr(record: ImageRecord, question: str) -> str | None:
    """Read a value back out of the recognised text.

    Three tiers, narrowest first: the question's field matched against a
    label/value pair, fuzzy token overlap against the same pairs, then any
    numeric line that shares wording with the question. The tiers exist because
    OCR keys are only approximately the words people use -- "Cores" for "nhân",
    "TỔNG CỘNG" for "tổng tiền".
    """
    if not record.ocr and not record.ocr_pairs:
        return None

    asked = find_field(question)

    if asked is not None:
        _, field = asked
        for pair in record.ocr_pairs:
            if _field_of_key(pair["key"]) == field:
                return f"{pair['key']}: {pair['value']}"

    best_pair, best_score = None, 0.0
    for pair in record.ocr_pairs:
        score = _token_overlap(question, pair["key"])
        if score > best_score:
            best_pair, best_score = pair, score
    if best_pair is not None and best_score >= 0.15:
        return f"{best_pair['key']}: {best_pair['value']}"

    # No pair matched: fall back to a numeric line that shares wording with the
    # question. Requiring a digit is what keeps this from returning a heading.
    best_line, best_line_score = None, 0.0
    for item in record.ocr:
        text = item.get("text", "")
        if not _HAS_DIGIT.search(text):
            continue
        score = _token_overlap(question, text)
        if score > best_line_score:
            best_line, best_line_score = text, score
    if best_line is not None and best_line_score >= 0.10:
        return f"Trong ảnh ghi: {best_line}"

    return None


async def from_vlm(
    vlm: VLMClient,
    record: ImageRecord,
    question: str,
    *,
    describe: bool = False,
) -> str | None:
    """Ask the cloud model, using the buffered thumbnail.

    Returns None when the thumbnail has expired -- the caller turns that into
    the "send it again" reply rather than sending a question with no image.
    """
    if not record.thumb_alive():
        return None

    prompt_question = DESCRIBE_QUESTION if describe else question
    return await vlm.ask(
        question=prompt_question,
        fast_summary=record.fast_summary,
        image_jpeg=record.thumb or b"",
    )
