"""Turn raw detector/OCR output into the text the rest of the system reads.

This module is the whole reason the vision layer needs no changes to PhoBERT:
every downstream consumer -- the router, the answerers, the VLM prompt -- sees
Vietnamese text, never a tensor. Three products come out of here:

  tags          coarse class of the image, decides which answerer is even
                allowed to say "no" without asking a VLM
  ocr_pairs     label/value pairs recovered from OCR *layout*, which is what
                makes "Cores: 12" answerable without a caption model
  fast_summary  the one string handed to the VLM as pre-extracted context
  chat_summary  the same content said in a sentence, for a chat model to
                answer from -- see `build_chat_summary` for why it is not
                simply `fast_summary`

Pure functions over plain dicts on purpose: no model, no I/O, fully testable.
"""
from __future__ import annotations

import re
import unicodedata

TAG_SCREENSHOT = "screenshot"
TAG_DOCUMENT = "document"
TAG_PEOPLE = "people"
TAG_SCENE = "scene"
TAG_UNKNOWN = "unknown"

TEXT_TAGS = frozenset({TAG_SCREENSHOT, TAG_DOCUMENT})

# Cap on how much OCR reaches the summary. Long enough for a settings pane or a
# receipt, short enough that the VLM prompt stays a prompt and not a document.
_MAX_SUMMARY_LINES = 40

# How far to the right of a label its value may sit, in multiples of the
# label's own line height. Roughly one indented column on a 1080p screenshot.
_MAX_RIGHT_GAP_LINES = 12

_KEY_SUFFIX = re.compile(r"[:：]\s*$")
_MOSTLY_DIGITS = re.compile(r"^[\W\d]+$", re.UNICODE)
_LEADING_DIGIT = re.compile(r"^[\W]*\d")
_WS = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Lowercase, strip accents and collapse whitespace.

    Used for every string comparison in this layer. OCR routinely drops or
    mangles diacritics ("TONG CONG" for "TỔNG CỘNG") and users type without
    them, so accent-sensitive matching would fail on exactly the inputs this
    is meant to handle.
    """
    decomposed = unicodedata.normalize("NFD", text.casefold())
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    # Vietnamese 'đ' is a distinct letter, not a combining sequence.
    stripped = stripped.replace("đ", "d")
    return _WS.sub(" ", stripped).strip()


def fold_case(text: str) -> str:
    """Lowercase and collapse whitespace, but keep the diacritics.

    For the handful of comparisons where accents carry the meaning. "này"
    (this) and "nay" (as in "hôm nay", today) differ by nothing else, and
    folding them together makes every question containing "hôm nay" look like
    it is pointing at the most recent image.
    """
    return _WS.sub(" ", text.casefold()).strip()


def _center(box: list[float]) -> tuple[float, float]:
    x, y, w, h = box
    return x + w / 2.0, y + h / 2.0


def _looks_like_key(text: str) -> bool:
    """A short label rather than a value.

    Either it ends in a colon -- unambiguous -- or it is a short phrase that
    does not lead with a number, which is what an unpunctuated form label
    ("Memory", "Base speed") looks like. The leading-digit test is what keeps
    "2.60 GHz" and "250.000d" on the value side of the line: both are short
    two-word phrases and would otherwise read as labels.
    """
    stripped = text.strip()
    if not stripped:
        return False
    if _KEY_SUFFIX.search(stripped):
        return True
    if _MOSTLY_DIGITS.match(stripped) or _LEADING_DIGIT.match(stripped):
        return False
    return len(stripped) <= 28 and len(stripped.split()) <= 3


def _clean_key(text: str) -> str:
    return _KEY_SUFFIX.sub("", text.strip()).strip()


def sort_reading_order(items: list[dict]) -> list[dict]:
    """Top-to-bottom, then left-to-right."""
    return sorted(items, key=lambda it: (it["box"][1], it["box"][0]))


def build_ocr_pairs(ocr: list[dict]) -> list[dict]:
    """Recover label/value pairs from OCR box geometry.

    Two layouts cover almost everything worth answering: a value to the right
    of its label on the same line (settings panes, invoices) and a value
    stacked directly under it (dashboard tiles, "Memory" over "17.8/39.6 GB").

    Two things about the geometry are not obvious and both come from real
    screenshots rather than from first principles:

    Stacked matching accepts left-edge alignment as well as centre alignment.
    Centre alignment alone -- the obvious rule -- fails on exactly the common
    case, because a label is narrow and its value wide, so two left-aligned
    boxes have centres far apart.

    A label-shaped box is not accepted as somebody else's value. Side-by-side
    column headings are the reason: in "Utilization | Speed" with numbers
    underneath, the nearest thing to the right of "Utilization" is "Speed", and
    pairing them produces a confident, useless answer. A colon-terminated label
    is the one exception, and only to its right, where the colon says the value
    is: "Khách hàng: Nguyễn Văn A" pairs even though the name reads like a
    label. Its *stacked* candidate gets no such exemption -- underneath
    "Sockets:" is "Cores:", and when OCR misses the small "1" beside it, the
    next label down must not be served as the answer.

    Producing no pair is a real outcome here, not a failure. OCR does drop
    isolated digits, and a key with nothing recoverable beside it is better
    left unpaired than paired with its neighbour.
    """
    pairs: list[dict] = []
    used_as_value: set[int] = set()

    for i, item in enumerate(ocr):
        text = item.get("text", "")
        # A box already claimed as somebody's value is not also a label, even
        # when it reads like one. "Virtualization: Enabled" would otherwise be
        # followed by "Enabled: 1.1 MB" from the next row down.
        if i in used_as_value or not _looks_like_key(text):
            continue

        xi, yi, wi, hi = item["box"]
        xci, yci = _center(item["box"])
        explicit = bool(_KEY_SUFFIX.search(text.strip()))

        right_plain: tuple[float, int] | None = None
        right_any: tuple[float, int] | None = None
        below: tuple[float, int] | None = None

        for j, other in enumerate(ocr):
            if j == i or j in used_as_value:
                continue
            other_text = other.get("text", "")
            other_is_key = _looks_like_key(other_text)
            xj, yj, wj, hj = other["box"]
            xcj, ycj = _center(other["box"])

            # Same line, starting to the right of this box's midpoint and
            # within reach. The distance cap is what stops a label in the left
            # column claiming a value from a table on the far side of the
            # frame: "Utilization" and "2.60 GHz" share a baseline in a Task
            # Manager grab and are 600px apart.
            if (
                abs(ycj - yci) < 0.6 * hi
                and xj >= xi + 0.5 * wi
                and xcj > xci
                and (xj - (xi + wi)) <= _MAX_RIGHT_GAP_LINES * hi
            ):
                gap = xj - (xi + wi)
                if right_any is None or gap < right_any[0]:
                    right_any = (gap, j)
                if not other_is_key and (right_plain is None or gap < right_plain[0]):
                    right_plain = (gap, j)
                continue

            # Directly underneath, less than one line of whitespace away.
            # Measured edge to edge, not centre to centre: a value is often set
            # in a larger face than its label, and a centre-distance rule
            # rejects the pair for the value being *tall* rather than far.
            # A stacked candidate is never allowed to be a label, whether or
            # not this key carries a colon.
            if other_is_key:
                continue
            aligned = abs(xcj - xci) < 0.5 * wi or abs(xj - xi) < 0.5 * max(wi, wj)
            drop = yj - (yi + hi)
            if aligned and -0.25 * hi <= drop < hi and (below is None or drop < below[0]):
                below = (drop, j)

        chosen = (right_any if explicit else right_plain) or below
        if chosen is None:
            continue

        _, j = chosen
        value = ocr[j].get("text", "").strip()
        if not value:
            continue
        used_as_value.add(j)
        pairs.append({"key": _clean_key(text), "value": value})

    return pairs


def build_tags(
    detections: list[dict],
    ocr: list[dict],
    image_size: tuple[int, int],
    ocr_pairs: list[dict] | None = None,
    *,
    area_ratio_threshold: float = 0.30,
    min_lines_text: int = 8,
    screenshot_min_pairs: int = 3,
    screenshot_pair_ratio: float = 0.30,
) -> list[str]:
    """Coarse class of the image, most specific first.

    Order matters to callers: `tags[0]` is what the summary prints and what the
    count answerer checks before it is willing to answer "khong co" from an
    empty detection list.
    """
    width, height = image_size
    frame_area = max(1.0, float(width) * float(height))
    ocr_area = sum(float(it["box"][2]) * float(it["box"][3]) for it in ocr)
    ocr_pairs = ocr_pairs if ocr_pairs is not None else []

    dense_by_area = (ocr_area / frame_area) > area_ratio_threshold
    # See VisionSettings.ocr_min_lines_text for why the area rule is not enough
    # on its own.
    dense_by_lines = bool(min_lines_text) and len(ocr) >= min_lines_text and not detections
    is_text_surface = bool(ocr) and (dense_by_area or dense_by_lines)

    tags: list[str] = []
    if is_text_surface:
        pair_ratio = len(ocr_pairs) / max(1, len(ocr))
        if len(ocr_pairs) >= screenshot_min_pairs and pair_ratio >= screenshot_pair_ratio:
            tags.append(TAG_SCREENSHOT)
        else:
            tags.append(TAG_DOCUMENT)

    if any(det.get("label") == "person" for det in detections):
        tags.append(TAG_PEOPLE)

    if not tags:
        if not detections and not ocr:
            tags.append(TAG_UNKNOWN)
        else:
            tags.append(TAG_SCENE)

    return tags


def _count_labels(detections: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for det in detections:
        label = det.get("label", "")
        if label:
            counts[label] = counts.get(label, 0) + 1
    return counts


def describe_objects(detections: list[dict], vi_names: dict[str, str] | None = None) -> str:
    """"người×3, xe máy×1" -- Vietnamese label where one is known."""
    counts = _count_labels(detections)
    if not counts:
        return ""
    vi_names = vi_names or {}
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return ", ".join(f"{vi_names.get(label, label)}×{count}" for label, count in ordered)


def _text_lines(ocr: list[dict], ocr_pairs: list[dict]) -> list[str]:
    """OCR as readable lines: pairs first, then whatever they did not consume.

    Pairs come first because they carry their own label -- "Cores 12" survives
    truncation as an answer, the bare line "12" does not.
    """
    lines: list[str] = [f"{pair['key']} {pair['value']}" for pair in ocr_pairs]
    paired_values = {pair["value"] for pair in ocr_pairs}
    paired_keys = {pair["key"] for pair in ocr_pairs}
    for item in ocr:
        text = item.get("text", "").strip()
        if not text or text in paired_values or _clean_key(text) in paired_keys:
            continue
        lines.append(text)
    return lines


# COCO classes that need a Vietnamese classifier to read as a sentence: "một
# con mèo", where "3 người" and "1 laptop" take none. Only the animals do.
_ANIMAL_LABELS = frozenset({
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
    "zebra", "giraffe",
})

_TEXT_TAG_LEAD: dict[str, str] = {
    TAG_SCREENSHOT: "Đây là ảnh chụp màn hình.",
    TAG_DOCUMENT: "Đây là ảnh một trang tài liệu.",
}

# Only ever speaks about objects: incidental text may well have been found
# and deliberately dropped, so claiming there was none would be a lie.
NOTHING_RECOGNISED = "Mình không nhận ra rõ vật thể nào trong ảnh."


def phrase_objects(detections: list[dict], vi_names: dict[str, str] | None = None) -> str:
    """"3 người và một con mèo" -- the same counts as `describe_objects`, said."""
    counts = _count_labels(detections)
    if not counts:
        return ""
    vi_names = vi_names or {}
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))

    parts = []
    for label, count in ordered:
        noun = vi_names.get(label, label)
        classifier = "con " if label in _ANIMAL_LABELS else ""
        # "một con mèo" reads; "1 con mèo" reads like a form field, and the
        # whole point of this string is that it does not sound like one.
        quantity = "một" if count == 1 else str(count)
        parts.append(f"{quantity} {classifier}{noun}")

    if len(parts) == 1:
        return parts[0]
    return f"{', '.join(parts[:-1])} và {parts[-1]}"


def build_chat_summary(
    tags: list[str],
    detections: list[dict],
    ocr: list[dict],
    ocr_pairs: list[dict],
    vi_names: dict[str, str] | None = None,
    *,
    max_lines: int = _MAX_SUMMARY_LINES,
) -> str:
    """What an image is, phrased for a chat model rather than for the router.

    `fast_summary` is written for the router and the VLM prompt: tagged, dense,
    machine-shaped. Handed to a small chat model it leaks that shape into the
    reply -- measured on this project's Qwen3-1.7B, "Ảnh scene. Vật thể:
    mèo×1." comes back as "Nội dung được nhận diện chính xác", where "Trong
    ảnh có một con mèo." comes back as a remark about the cat.

    Incidental text is dropped unless the image *is* text (a screenshot or a
    document). A watermark in the corner of a photo is not what the photo is
    about, and quoting it invites the model to talk about the watermark
    instead -- which is exactly what a stock photo's "Fago Pet" did.

    Never "": a caller reads the empty string as "captioning unavailable", so
    an image the models simply found nothing in has to say so in words.
    """
    tag = tags[0] if tags else TAG_UNKNOWN
    parts: list[str] = []

    objects = phrase_objects(detections, vi_names)
    if objects:
        parts.append(f"Trong ảnh có {objects}.")

    if tag in TEXT_TAGS:
        lead = _TEXT_TAG_LEAD.get(tag, "")
        lines = _text_lines(ocr, ocr_pairs)
        if lines:
            shown = lines[:max_lines]
            joined = " | ".join(shown)
            if len(lines) > max_lines:
                joined += f" | (+{len(lines) - max_lines} dòng nữa)"
            parts.append(f"{lead} Chữ trong ảnh: {joined}.".strip())
        elif lead:
            parts.append(lead)

    return " ".join(parts) if parts else NOTHING_RECOGNISED


def build_fast_summary(
    tags: list[str],
    detections: list[dict],
    ocr: list[dict],
    ocr_pairs: list[dict],
    vi_names: dict[str, str] | None = None,
    *,
    max_lines: int = _MAX_SUMMARY_LINES,
) -> str:
    """The single string that represents an image to everything downstream.

    Pairs come first because they carry their own label -- "Cores 12" survives
    truncation as an answer, the bare line "12" does not.
    """
    tag = tags[0] if tags else TAG_UNKNOWN

    objects = describe_objects(detections, vi_names)
    objects_part = f"Vật thể: {objects}." if objects else "Không phát hiện vật thể."

    lines = _text_lines(ocr, ocr_pairs)

    text_part = ""
    if lines:
        shown = lines[:max_lines]
        joined = " | ".join(shown)
        if len(lines) > max_lines:
            joined += f" | (+{len(lines) - max_lines} dòng nữa)"
        text_part = f" Chữ: {joined}."

    return f"Ảnh {tag}. {objects_part}{text_part}"
