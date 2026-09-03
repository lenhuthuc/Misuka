"""POST /v1/vision/caption

Describes an uploaded image with the local `vision/` pipeline (YOLO11n +
RapidOCR + CLIP -- see brain/caption_service.py). The chat model is text-only,
so this is the only way an attached image reaches it at all: the frontend
captions the image here first, then folds the caption into the text it sends
to /v1/chat/stream as usual.
"""
from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from PIL import UnidentifiedImageError

from api.dependencies import get_container
from brain.caption_service import DEFAULT_SESSION_ID
from core.container import ServiceContainer
from core.logging import log_duration
from schemas.vision import CaptionResponse
from vision.pipeline import decode_image

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/vision", tags=["Vision"])


@router.post("/caption", response_model=CaptionResponse)
async def caption_image(
    image: UploadFile = File(...),
    session_id: str = Form(DEFAULT_SESSION_ID),
    container: ServiceContainer = Depends(get_container),
) -> CaptionResponse:
    # Same reasoning as /emotion-vad's mark_active(): this request is the
    # user's turn, arriving just ahead of the /v1/chat/stream call it feeds,
    # so background LLM work should already be standing down before that
    # call asks for the runner.
    container.llm_gate.mark_active()

    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Provide an 'image' file.")

    # Decoded here and thrown away: unreadable bytes are the one failure that
    # belongs to the request rather than to us, and it is worth a 422 whether
    # or not captioning is even enabled. The milliseconds it costs are noise
    # against the ingest that follows.
    try:
        decode_image(raw)
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=422, detail=f"Not a readable image: {exc}") from exc

    if container.caption is None:
        return CaptionResponse(caption="")

    try:
        with log_duration(logger, "vision.caption", component="vision"):
            caption = await asyncio.wait_for(
                container.caption.describe(raw, session_id=session_id),
                timeout=container.vision_caption_timeout_seconds,
            )
    except TimeoutError:
        logger.warning("vision.caption timed out after %.1fs", container.vision_caption_timeout_seconds)
        return CaptionResponse(caption="")

    return CaptionResponse(caption=caption)
