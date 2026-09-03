"""Image understanding for the chatbot: fast local extraction, lazy cloud VLM.

Everything an image becomes here is *text* -- object counts, OCR lines,
label/value pairs, a Vietnamese summary -- because that is the only form the
PhoBERT-based NLU downstream can read. No image vector is ever projected into
its embedding space.

    from vision import VisionPipeline
    pipeline = VisionPipeline.build_default()
    record = await pipeline.ingest(image_bytes, session_id="s1")
    result = await pipeline.answer("Trong ảnh có mấy người?", session_id="s1")
    result.source  # "detections" | "ocr" | "vlm_cache" | "vlm" | "clarify" | "expired"

Importing this package loads numpy and pydantic and nothing else; the runtimes
come in when `build_default()` actually constructs a model.
"""
from __future__ import annotations

from vision.buffer import ImageBuffer, ImageRecord
from vision.config import VisionSettings, get_vision_settings
from vision.pipeline import AnswerResult, VisionPipeline

__all__ = [
    "AnswerResult",
    "ImageBuffer",
    "ImageRecord",
    "VisionPipeline",
    "VisionSettings",
    "get_vision_settings",
]
