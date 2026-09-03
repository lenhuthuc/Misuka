"""Tests for the two ways an image reaches `vision.VisionPipeline`.

Uploads arrive as encoded bytes and are handed to `ingest()` untouched. The
webcam/screen path has no file, only a BGR uint8 frame (the
`cv2.VideoCapture`/`cv2.imread` contract `vision_capture.py` produces), so
`caption()` has to re-encode it -- and that is where a channel-order slip
would silently swap red and blue for every captioned frame.

What comes back either way is `build_chat_summary`, not the record's own
`fast_summary`: this service exists to feed a chat model.
"""
from __future__ import annotations

import io
from types import SimpleNamespace

import numpy as np

from brain.caption_service import CaptionService


class StubPipeline:
    """Records what `ingest()` was given and answers with a scripted record."""

    def __init__(self, **overrides) -> None:
        self.ingested: list[tuple[bytes, str]] = []
        self.record = {
            "tags": ["scene"],
            "detections": [{"label": "cat", "score": 0.9}],
            "ocr": [],
            "ocr_pairs": [],
            "fast_summary": "Ảnh scene. Vật thể: mèo×1.",
            **overrides,
        }

    async def ingest(self, image_bytes: bytes, session_id: str):
        self.ingested.append((image_bytes, session_id))
        return SimpleNamespace(**self.record)


def _png_bytes(rgb: tuple[int, int, int], size: tuple[int, int] = (3, 2)) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", size, color=rgb).save(buf, format="PNG")
    return buf.getvalue()


async def test_describe_hands_the_upload_to_ingest_untouched():
    """The pipeline decodes for itself -- EXIF orientation included -- so an
    upload must reach it as the bytes that arrived, not as a re-encode."""
    pipeline = StubPipeline()
    raw = _png_bytes((10, 20, 30))

    await CaptionService(pipeline).describe(raw, session_id="s1")

    assert pipeline.ingested == [(raw, "s1")]


async def test_describe_returns_the_chat_sentence_not_the_router_shorthand():
    """@example: `fast_summary` says "Ảnh scene. Vật thể: mèo×1." -- handed to a
    1.7B model that comes back as a remark about the recognition. What the
    caller gets is the sentence instead."""
    pipeline = StubPipeline()

    caption = await CaptionService(pipeline).describe(_png_bytes((10, 20, 30)))

    assert caption == "Trong ảnh có một con mèo."


async def test_caption_re_encodes_a_bgr_frame_without_swapping_channels():
    """@example: a frame that is pure red in BGR (0, 0, 255) must arrive at the
    pipeline as a red image. Skipping the flip would make it blue, and every
    colour word in the summary of a captured frame would be wrong."""
    from PIL import Image

    pipeline = StubPipeline()
    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    frame[:, :, 2] = 255  # BGR: red

    await CaptionService(pipeline).caption(frame)

    encoded, _session = pipeline.ingested[-1]
    with Image.open(io.BytesIO(encoded)) as img:
        assert img.convert("RGB").getpixel((0, 0)) == (255, 0, 0)


async def test_caption_defaults_to_the_shared_session():
    pipeline = StubPipeline()
    await CaptionService(pipeline).caption(np.zeros((2, 2, 3), dtype=np.uint8))
    assert pipeline.ingested[-1][1] == "chat"
