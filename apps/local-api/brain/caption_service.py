"""Image -> Vietnamese text, on top of the `vision/` pipeline.

The chat model is text-only, so an attached image only ever reaches it as
words. Those words come from `vision.VisionPipeline` -- the YOLO11n + RapidOCR
+ CLIP fast path whose exported graphs already sit in `assets/models/vision/`
-- phrased by `build_chat_summary` into the one Vietnamese sentence a chat
model can answer from. Nothing here downloads a model.

Ingesting also parks the image in the pipeline's per-session buffer, so a
follow-up question about it can later be answered by `VisionPipeline.answer()`
off the extracted text, without the image being sent again. See
`vision/README.md`.
"""
from __future__ import annotations

import io
import logging
from typing import TYPE_CHECKING

import numpy as np

from vision.config import COCO_TO_VI
from vision.summary import build_chat_summary

if TYPE_CHECKING:
    from vision import VisionPipeline

logger = logging.getLogger(__name__)

# The frontend does not send a conversation id yet, so every upload lands in
# one buffer slot -- which is also what makes "ảnh vừa gửi" resolve to the
# newest image if `answer()` is ever wired up.
DEFAULT_SESSION_ID = "chat"


class CaptionService:
    """Describes a frame, or an uploaded file, through one shared pipeline."""

    def __init__(self, pipeline: VisionPipeline | None = None) -> None:
        if pipeline is None:
            from vision import VisionPipeline as _VisionPipeline

            pipeline = _VisionPipeline.build_default()
        self._pipeline = pipeline

    @property
    def pipeline(self) -> VisionPipeline:
        """The underlying pipeline, for callers that want `answer()` too."""
        return self._pipeline

    async def describe(self, image_bytes: bytes, session_id: str = DEFAULT_SESSION_ID) -> str:
        """Caption an encoded image file (PNG/JPEG/...) -- the upload path.

        `build_chat_summary`, not the record's own `fast_summary`: what comes
        back here is read by the chat model, which answers a sentence far
        better than it answers the router's tagged shorthand.
        """
        record = await self._pipeline.ingest(image_bytes, session_id=session_id)
        return build_chat_summary(
            record.tags, record.detections, record.ocr, record.ocr_pairs, COCO_TO_VI
        )

    async def caption(self, frame: np.ndarray, session_id: str = DEFAULT_SESSION_ID) -> str:
        """Caption a BGR uint8 frame -- the cv2 contract `vision_capture.py` yields.

        `ingest()` takes encoded bytes because that is what every other caller
        already has, so a webcam frame is re-encoded here (PNG: lossless, and
        the cost is milliseconds against a multi-second ingest) rather than
        opening a second entry point into the pipeline.
        """
        from PIL import Image

        buffer = io.BytesIO()
        Image.fromarray(np.ascontiguousarray(frame[:, :, ::-1])).save(buffer, format="PNG")
        return await self.describe(buffer.getvalue(), session_id=session_id)
