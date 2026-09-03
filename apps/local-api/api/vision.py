"""POST /v1/vision/caption

Describes an uploaded image with a local VLM (see brain/caption_service.py).
The chat model is text-only, so this is the only way an attached image
reaches it at all: the frontend captions the image here first, then folds
the caption into the text it sends to /v1/chat/stream as usual.
"""
from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from api.dependencies import get_container
from brain.caption_service import ImageDecodeError, decode_image_to_bgr
from core.container import ServiceContainer
from core.logging import log_duration
from schemas.vision import CaptionResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/vision", tags=["Vision"])


@router.post("/caption", response_model=CaptionResponse)
async def caption_image(
    image: UploadFile = File(...),
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

    try:
        frame = decode_image_to_bgr(raw)
    except ImageDecodeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if container.caption is None:
        return CaptionResponse(caption="")

    try:
        with log_duration(logger, "vision.caption", component="vision"):
            caption = await asyncio.wait_for(
                container.caption.caption(frame), timeout=container.vision_caption_timeout_seconds
            )
    except TimeoutError:
        logger.warning("vision.caption timed out after %.1fs", container.vision_caption_timeout_seconds)
        return CaptionResponse(caption="")

    return CaptionResponse(caption=caption)
