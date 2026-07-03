"""VLM captioning service — tries moondream2, then Florence-2, then placeholder."""
from __future__ import annotations

import asyncio
import logging

import numpy as np

logger = logging.getLogger(__name__)

_MOONDREAM_REPO = "vikhyatk/moondream2"
_MOONDREAM_REV  = "2025-01-09"
_FLORENCE_REPO  = "microsoft/Florence-2-base"
_CAPTION_PROMPT = "Describe this image briefly."


class CaptionService:
    """Load a VLM once, call caption() per frame."""

    def __init__(self) -> None:
        self._backend: str = "placeholder"
        self._model = None
        self._processor = None
        self._load_vlm()

    def _load_vlm(self) -> None:
        # ── Try moondream2 ────────────────────────────────────────────────────
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self._processor = AutoTokenizer.from_pretrained(
                _MOONDREAM_REPO, revision=_MOONDREAM_REV, trust_remote_code=True
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                _MOONDREAM_REPO, revision=_MOONDREAM_REV, trust_remote_code=True
            )
            self._model.eval()
            self._backend = "moondream2"
            logger.info("CaptionService: using moondream2")
            return
        except Exception as exc:
            logger.warning("moondream2 load failed: %s", exc)

        # ── Try Florence-2 ────────────────────────────────────────────────────
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor
            self._processor = AutoProcessor.from_pretrained(
                _FLORENCE_REPO, trust_remote_code=True
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                _FLORENCE_REPO, trust_remote_code=True
            )
            self._model.eval()
            self._backend = "florence2"
            logger.info("CaptionService: using Florence-2")
            return
        except Exception as exc:
            logger.warning("Florence-2 load failed: %s", exc)

        logger.warning("CaptionService: no VLM available, captions will be empty")

    # ── public API ────────────────────────────────────────────────────────────

    async def caption(self, frame: np.ndarray) -> str:
        """Return caption for BGR uint8 frame (async, runs in executor)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._caption_sync, frame)

    def _caption_sync(self, frame: np.ndarray) -> str:
        if self._backend == "placeholder" or self._model is None:
            return ""

        from PIL import Image
        # OpenCV BGR → RGB PIL image
        rgb = frame[:, :, ::-1]
        pil_img = Image.fromarray(rgb)

        try:
            if self._backend == "moondream2":
                enc = self._processor(pil_img, return_tensors="pt")
                result = self._model.answer_question(
                    enc["input_ids"],
                    _CAPTION_PROMPT,
                    tokenizer=self._processor,
                )
                return result.strip()

            if self._backend == "florence2":
                task = "<CAPTION>"
                inputs = self._processor(
                    text=task, images=pil_img, return_tensors="pt"
                )
                output = self._model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=128,
                )
                decoded = self._processor.batch_decode(output, skip_special_tokens=True)[0]
                return decoded.strip()
        except Exception as exc:
            logger.warning("Caption failed (%s): %s", self._backend, exc)

        return ""
