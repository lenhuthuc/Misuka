"""Text recognition via RapidOCR.

RapidOCR ships its own ONNX detection/recognition pair and does its own thread
management, so unlike the detector there is no backend choice to make here --
the only knobs are the thread cap and the score floor.

The package has been through a rename (`rapidocr_onnxruntime` -> `rapidocr`)
that also changed the return type from a tuple to a result object. Both are
handled: the installed name is whichever the machine happens to have, and
pinning would just move the problem into requirements.txt.
"""
from __future__ import annotations

import logging
from typing import Protocol

import numpy as np

from vision.models.backends import apply_thread_env
from vision.summary import sort_reading_order

logger = logging.getLogger(__name__)


class OCR(Protocol):
    """What the pipeline needs from an OCR engine, so tests can substitute one."""

    def read(self, image: np.ndarray) -> list[dict]:
        """RGB uint8 HxWx3 -> [{text, score, box:[x, y, w, h]}] in reading order."""
        ...


def _quad_to_xywh(quad) -> list[float]:
    """RapidOCR returns four corner points; the rest of this layer wants xywh."""
    points = np.asarray(quad, dtype=np.float32).reshape(-1, 2)
    x0, y0 = points.min(axis=0)
    x1, y1 = points.max(axis=0)
    return [round(float(x0), 1), round(float(y0), 1), round(float(x1 - x0), 1), round(float(y1 - y0), 1)]


class RapidOCREngine:
    """Thin adapter over RapidOCR with a confidence floor and reading order."""

    def __init__(self, *, num_threads: int = 8, score_min: float = 0.60) -> None:
        apply_thread_env(num_threads)
        self._score_min = score_min
        self._engine = self._load(num_threads)

    @staticmethod
    def _load(num_threads: int):
        try:
            from rapidocr import RapidOCR as _RapidOCR
        except ImportError:
            try:
                from rapidocr_onnxruntime import RapidOCR as _RapidOCR
            except ImportError as exc:
                raise ImportError(
                    "RapidOCR is not installed. `pip install rapidocr-onnxruntime` "
                    "(or `rapidocr`) to enable the OCR stage."
                ) from exc

        # Constructed with no overrides on purpose. Both generations accept
        # per-stage keyword arguments, and on `rapidocr_onnxruntime` 1.2 any
        # `det_*` key makes `det_model_path` mandatory as well -- so passing a
        # thread count means also hard-coding paths to the bundled weights.
        # The OMP variables set above already cap the pool, and the confidence
        # floor is applied in `read()`, which keeps it in one place: config.
        engine = _RapidOCR()
        logger.info("vision: loaded RapidOCR (thread cap %d via OMP)", num_threads)
        return engine

    def read(self, image: np.ndarray) -> list[dict]:
        raw = self._engine(image)

        # Old API: (results, elapse). New API: an object with .boxes/.txts/.scores.
        if isinstance(raw, tuple):
            rows = raw[0] or []
            triples = [(row[0], row[1], row[2]) for row in rows]
        elif raw is None or getattr(raw, "boxes", None) is None:
            triples = []
        else:
            triples = list(zip(raw.boxes, raw.txts, raw.scores))

        items: list[dict] = []
        for quad, text, score in triples:
            text = (text or "").strip()
            score = float(score)
            if not text or score < self._score_min:
                continue
            items.append({"text": text, "score": round(score, 4), "box": _quad_to_xywh(quad)})

        return sort_reading_order(items)
