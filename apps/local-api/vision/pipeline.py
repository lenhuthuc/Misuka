"""The two entry points: ingest an image, answer a question about it.

Split by cost, not by capability. `ingest` is the fast path and runs three
small local models concurrently the moment an image arrives; it must finish
inside a second, because it happens while the user is still typing. `answer`
tries to reply out of what ingest already produced -- detections, OCR, cached
VLM answers -- and only reaches for the cloud model when none of that can.

Which of those happened is recorded in `AnswerResult.source`. That field is the
point of this design: the fallback rate to `"vlm"` is what says whether the
fast path is carrying its share, and it is measurable per turn.
"""
from __future__ import annotations

import asyncio
import io
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from vision import answerers, router
from vision.buffer import ImageBuffer, ImageRecord
from vision.config import COCO_TO_VI, VisionSettings, get_vision_settings
from vision.models.embedder import cosine
from vision.models.vlm_client import VLMClient
from vision.router import Classifier, RuleClassifier
from vision.summary import (
    build_fast_summary,
    build_ocr_pairs,
    build_tags,
    normalize_text,
    sort_reading_order,
)

logger = logging.getLogger(__name__)

SOURCE_DETECTIONS = "detections"
SOURCE_OCR = "ocr"
SOURCE_VLM_CACHE = "vlm_cache"
SOURCE_VLM = "vlm"
SOURCE_CLARIFY = "clarify"
SOURCE_EXPIRED = "expired"

EXPIRED_MESSAGE = "Ảnh đã quá lâu, bạn gửi lại giúp mình để phân tích sâu hơn."
NO_IMAGE_MESSAGE = "Mình chưa nhận được ảnh nào trong cuộc trò chuyện này."
CLARIFY_MESSAGE = "Bạn đang hỏi về ảnh nào vậy? Mình đang giữ {n} ảnh trong phiên này."


@dataclass
class AnswerResult:
    """What answered, from where, and how long it took."""

    text: str
    source: str
    image_id: str | None
    latency_ms: float
    question_label: str = router.UNKNOWN
    confidence: float = 0.0


class VisionPipeline:
    """Wires the models, the buffer and the router together.

    Every collaborator is injected. The models are the slow, machine-specific
    part and the routing is the part with the interesting bugs; keeping them
    separable is what lets the acceptance tests drive the real router and the
    real summary code with scripted detector/OCR output.
    """

    def __init__(
        self,
        *,
        detector,
        ocr,
        embedder,
        vlm: VLMClient,
        settings: VisionSettings | None = None,
        classifier: Classifier | None = None,
        buffer: ImageBuffer | None = None,
    ) -> None:
        self._settings = settings or get_vision_settings()
        self._detector = detector
        self._ocr = ocr
        self._embedder = embedder
        self._vlm = vlm
        self._classifier = classifier or RuleClassifier()
        self._buffer = buffer or ImageBuffer(max_per_session=self._settings.max_images_per_session)
        # Three workers because there are exactly three independent stages; the
        # thread cap that matters is the per-session intra-op one in backends.py.
        self._executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="vision")

    @property
    def buffer(self) -> ImageBuffer:
        return self._buffer

    @classmethod
    def build_default(cls, settings: VisionSettings | None = None) -> "VisionPipeline":
        """Construct with the real models named in config.

        Imported here rather than at module scope so importing the pipeline --
        which the tests and the router do -- never touches onnxruntime,
        RapidOCR or transformers.
        """
        settings = settings or get_vision_settings()

        from vision.models.detector import YoloDetector
        from vision.models.embedder import ClipEmbedder
        from vision.models.ocr import RapidOCREngine
        from vision.models.vlm_client import NullVLMClient, OpenAICompatibleVLMClient

        detector = YoloDetector(
            settings.resolved_yolo_onnx_path,
            openvino_path=settings.resolved_yolo_openvino_dir / "yolo11n.xml",
            runtime=settings.runtime,
            openvino_device=settings.openvino_device,
            num_threads=settings.num_threads,
            input_size=settings.yolo_input_size,
            score_min=settings.det_score_min,
            iou_threshold=settings.det_iou_threshold,
        )
        ocr = RapidOCREngine(
            num_threads=settings.num_threads,
            score_min=settings.ocr_score_min,
        )
        embedder = ClipEmbedder(
            settings.resolved_clip_image_onnx_path,
            settings.resolved_clip_text_onnx_path,
            settings.resolved_clip_tokenizer_dir,
            runtime=settings.runtime,
            openvino_device=settings.openvino_device,
            num_threads=settings.num_threads,
            image_size=settings.clip_image_size,
        )
        if settings.vlm_api_key:
            vlm: VLMClient = OpenAICompatibleVLMClient(
                base_url=settings.vlm_base_url,
                api_key=settings.vlm_api_key,
                model=settings.vlm_model,
                timeout_seconds=settings.vlm_timeout_seconds,
                max_retries=settings.vlm_max_retries,
                max_tokens=settings.vlm_max_tokens,
            )
        else:
            logger.warning("vision: no VLM API key configured; slow path disabled")
            vlm = NullVLMClient()

        return cls(detector=detector, ocr=ocr, embedder=embedder, vlm=vlm, settings=settings)

    async def warmup(self) -> None:
        """Load and run every model once, off the critical path.

        Worth calling at startup. The towers load lazily, and the CLIP text
        tower is 540MB -- measured cold, the first question that needed it took
        11.4s against 0.1s warm, and that first question is a real user's.
        Failures are logged and swallowed: warmup is an optimisation, and a
        model that cannot load here will fail the same way later, where the
        per-stage error handling already covers it.
        """
        blank = np.zeros((64, 64, 3), dtype=np.uint8)
        loop = asyncio.get_running_loop()
        for name, call in (
            ("detector", lambda: self._detector.detect(blank)),
            ("ocr", lambda: self._ocr.read(blank)),
            ("clip.image", lambda: self._embedder.encode_image(blank)),
            ("clip.text", lambda: self._embedder.encode_text("khởi động")),
        ):
            started = time.perf_counter()
            try:
                await loop.run_in_executor(self._executor, call)
                logger.info("vision.warmup %s ready in %.0fms", name, (time.perf_counter() - started) * 1000.0)
            except Exception as exc:
                logger.warning("vision.warmup %s unavailable: %s", name, exc)

    # -- ingest --------------------------------------------------------------

    async def ingest(self, image_bytes: bytes, session_id: str) -> ImageRecord:
        started = time.perf_counter()
        settings = self._settings

        image = decode_image(image_bytes)
        height, width = image.shape[:2]
        loop = asyncio.get_running_loop()

        # Four independent jobs, one round trip. The thumbnail encode is in
        # here too: it is pure CPU and would otherwise serialise behind OCR.
        det_task = loop.run_in_executor(self._executor, self._safe_detect, image)
        ocr_task = loop.run_in_executor(self._executor, self._safe_ocr, image)
        emb_task = loop.run_in_executor(self._executor, self._safe_embed, image)
        thumb_task = loop.run_in_executor(
            self._executor, make_thumbnail, image, settings.thumb_max_edge, settings.thumb_quality
        )
        detections, ocr_items, clip_emb, thumb = await asyncio.gather(
            det_task, ocr_task, emb_task, thumb_task
        )

        detections = [d for d in detections if d.get("score", 0.0) >= settings.det_score_min]
        ocr_items = sort_reading_order(
            [o for o in ocr_items if o.get("score", 0.0) >= settings.ocr_score_min]
        )

        ocr_pairs = build_ocr_pairs(ocr_items)
        tags = build_tags(
            detections,
            ocr_items,
            (width, height),
            ocr_pairs,
            area_ratio_threshold=settings.ocr_area_text_ratio,
            min_lines_text=settings.ocr_min_lines_text,
            screenshot_min_pairs=settings.screenshot_min_pairs,
            screenshot_pair_ratio=settings.screenshot_pair_ratio,
        )
        fast_summary = build_fast_summary(tags, detections, ocr_items, ocr_pairs, COCO_TO_VI)

        now = time.time()
        record = ImageRecord(
            image_id=self._buffer.next_image_id(session_id),
            session_id=session_id,
            ts=now,
            tags=tags,
            detections=detections,
            ocr=ocr_items,
            ocr_pairs=ocr_pairs,
            clip_emb=clip_emb,
            fast_summary=fast_summary,
            thumb=thumb,
            thumb_expires_at=now + settings.thumb_ttl_seconds,
            width=width,
            height=height,
        )
        self._buffer.add(record)

        logger.info(
            "vision.ingest %s tags=%s det=%d ocr=%d pairs=%d %.0fms",
            record.image_id, ",".join(tags), len(detections), len(ocr_items),
            len(ocr_pairs), (time.perf_counter() - started) * 1000.0,
        )
        return record

    def _safe_detect(self, image: np.ndarray) -> list[dict]:
        try:
            return self._detector.detect(image)
        except Exception as exc:
            logger.warning("vision.ingest detector failed: %s", exc)
            return []

    def _safe_ocr(self, image: np.ndarray) -> list[dict]:
        try:
            return self._ocr.read(image)
        except Exception as exc:
            logger.warning("vision.ingest OCR failed: %s", exc)
            return []

    def _safe_embed(self, image: np.ndarray) -> np.ndarray | None:
        try:
            return self._embedder.encode_image(image)
        except Exception as exc:
            logger.warning("vision.ingest embedder failed: %s", exc)
            return None

    # -- answer --------------------------------------------------------------

    async def answer(self, question: str, session_id: str) -> AnswerResult:
        started = time.perf_counter()

        def done(text: str, source: str, record: ImageRecord | None,
                 label: str = router.UNKNOWN, confidence: float = 0.0) -> AnswerResult:
            latency = (time.perf_counter() - started) * 1000.0
            image_id = record.image_id if record is not None else None
            logger.info(
                "vision.answer source=%s label=%s image=%s %.1fms",
                source, label, image_id, latency,
            )
            return AnswerResult(
                text=text, source=source, image_id=image_id,
                latency_ms=round(latency, 2), question_label=label, confidence=confidence,
            )

        records = self._buffer.list_session(session_id)
        if not records:
            return done(NO_IMAGE_MESSAGE, SOURCE_CLARIFY, None)

        selection = router.select_image(
            question, records, self._embedder,
            min_similarity=self._settings.image_select_min_sim,
        )
        if selection.record is None:
            return done(CLARIFY_MESSAGE.format(n=len(records)), SOURCE_CLARIFY, None)
        record = selection.record

        label, confidence = self._classifier.predict(question)

        if label == router.COUNT_EXIST:
            text = answerers.from_detections(record, question)
            if text is not None:
                return done(text, SOURCE_DETECTIONS, record, label, confidence)

        elif label == router.READ_TEXT:
            text = answerers.from_ocr(record, question)
            if text is not None:
                return done(text, SOURCE_OCR, record, label, confidence)

        elif label == router.DESCRIBE:
            if record.vlm_caption:
                return done(record.vlm_caption, SOURCE_VLM_CACHE, record, label, confidence)
            answer = await answerers.from_vlm(self._vlm, record, question, describe=True)
            if answer is None:
                return done(EXPIRED_MESSAGE, SOURCE_EXPIRED, record, label, confidence)
            record.vlm_caption = answer
            return done(answer, SOURCE_VLM, record, label, confidence)

        # REASON, UNKNOWN, and anything the fast path declined to answer.
        cached = self._lookup_qa_cache(record, question)
        if cached is not None:
            return done(cached, SOURCE_VLM_CACHE, record, label, confidence)

        answer = await answerers.from_vlm(self._vlm, record, question)
        if answer is None:
            return done(EXPIRED_MESSAGE, SOURCE_EXPIRED, record, label, confidence)

        self._store_qa_cache(record, question, answer)
        return done(answer, SOURCE_VLM, record, label, confidence)

    # -- VLM answer cache ----------------------------------------------------

    def _question_embedding(self, question: str) -> np.ndarray | None:
        try:
            return self._embedder.encode_text(question)
        except Exception as exc:
            logger.warning("vision.answer question embedding failed: %s", exc)
            return None

    def _lookup_qa_cache(self, record: ImageRecord, question: str) -> str | None:
        """Reuse a paid-for answer when the question is the same one reworded.

        Falls back to exact normalised-string equality when no embedding is
        available, so a broken text tower costs recall here, not correctness.
        """
        if not record.vlm_qa:
            return None

        query = self._question_embedding(question)
        if query is None:
            normalized = normalize_text(question)
            for entry in record.vlm_qa:
                if normalize_text(entry["q"]) == normalized:
                    return entry["a"]
            return None

        best, best_score = None, 0.0
        for entry in record.vlm_qa:
            score = cosine(query, entry.get("q_emb"))
            if score > best_score:
                best, best_score = entry, score

        if best is not None and best_score >= self._settings.vlm_cache_min_sim:
            logger.info("vision.answer VLM cache hit (sim=%.3f)", best_score)
            return best["a"]
        return None

    def _store_qa_cache(self, record: ImageRecord, question: str, answer: str) -> None:
        record.vlm_qa.append({
            "q": question,
            "a": answer,
            "q_emb": self._question_embedding(question),
        })

    # -- lifecycle -----------------------------------------------------------

    async def aclose(self) -> None:
        self._executor.shutdown(wait=False)
        await self._vlm.aclose()


def decode_image(image_bytes: bytes) -> np.ndarray:
    """Bytes -> RGB uint8 HxWx3.

    Via Pillow rather than cv2.imdecode: it handles EXIF orientation, which
    phone photos rely on, and palette/alpha modes cv2 silently mangles.
    """
    from PIL import Image, ImageOps

    with Image.open(io.BytesIO(image_bytes)) as img:
        img = ImageOps.exif_transpose(img)
        return np.asarray(img.convert("RGB"), dtype=np.uint8)


def make_thumbnail(image: np.ndarray, max_edge: int = 768, quality: int = 80) -> bytes:
    """Re-encode to a bounded JPEG. This is the only copy of the image kept.

    768px on the long edge is what the VLM tiling wants anyway, and it drops a
    12MP phone photo to roughly 100KB -- small enough that a session's worth of
    them is a rounding error against the model weights already resident.
    """
    from PIL import Image

    pil = Image.fromarray(image)
    pil.thumbnail((max_edge, max_edge), Image.LANCZOS)
    buffer = io.BytesIO()
    pil.save(buffer, format="JPEG", quality=quality, optimize=True)
    return buffer.getvalue()
