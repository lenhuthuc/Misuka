"""Multilingual CLIP: one vector space for images and Vietnamese questions.

Two graphs, one space. The image tower is stock CLIP ViT-B/32; the text tower
is the multilingual distillation of its text encoder, which is what lets
"hoa don tong tien" score against a photo of a receipt without a translation
step. Both are exported to ONNX by `scripts/export_models.py`.

Embeddings are L2-normalised on the way out, so every similarity in this layer
is a plain dot product and thresholds mean the same thing everywhere.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

import numpy as np

from vision.models.backends import load_backend

logger = logging.getLogger(__name__)

# CLIP's own preprocessing constants -- not ImageNet's, which are close enough
# to look right and wrong enough to cost a few points of retrieval accuracy.
_CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

_MAX_TEXT_TOKENS = 128


class Embedder(Protocol):
    """What the pipeline needs from an embedder, so tests can substitute one."""

    def encode_image(self, image: np.ndarray) -> np.ndarray: ...

    def encode_text(self, text: str) -> np.ndarray: ...

    @property
    def dimension(self) -> int: ...


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    return (vector / (np.linalg.norm(vector) + 1e-9)).astype(np.float32)


def cosine(a: np.ndarray | None, b: np.ndarray | None) -> float:
    """Cosine similarity that tolerates a missing vector.

    Returns 0.0 rather than raising when either side is absent: an image whose
    embedding failed should rank last, not crash the turn that ranks it.
    """
    if a is None or b is None:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-9:
        return 0.0
    return float(np.dot(a, b) / denom)


def preprocess_image(image: np.ndarray, size: int = 224) -> np.ndarray:
    """RGB uint8 HxWx3 -> NCHW float32, CLIP-normalised.

    Resize-shortest-edge then centre-crop, matching CLIPImageProcessor. A plain
    square resize would distort aspect ratio, which CLIP is measurably
    sensitive to.
    """
    import cv2

    height, width = image.shape[:2]
    scale = size / min(height, width)
    new_w, new_h = max(size, int(round(width * scale))), max(size, int(round(height * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    top, left = (new_h - size) // 2, (new_w - size) // 2
    cropped = resized[top:top + size, left:left + size]

    arr = cropped.astype(np.float32) / 255.0
    arr = (arr - _CLIP_MEAN) / _CLIP_STD
    return arr.transpose(2, 0, 1)[None].astype(np.float32)


class ClipEmbedder:
    """Image and text towers behind one object, loaded lazily per tower.

    Lazily because the two are used at different times: ingest only ever
    touches the image tower, and a session where nobody asks a question that
    needs image selection never pays for the text one.
    """

    def __init__(
        self,
        image_onnx_path: Path,
        text_onnx_path: Path,
        tokenizer_dir: Path,
        *,
        runtime: str = "onnx",
        openvino_device: str = "GPU",
        num_threads: int = 8,
        image_size: int = 224,
    ) -> None:
        self._image_onnx_path = image_onnx_path
        self._text_onnx_path = text_onnx_path
        self._tokenizer_dir = tokenizer_dir
        self._runtime = runtime
        self._openvino_device = openvino_device
        self._num_threads = num_threads
        self._image_size = image_size

        self._image_backend = None
        self._text_backend = None
        self._tokenizer = None
        self._dimension = 512

    # -- towers --------------------------------------------------------------

    def _image_tower(self):
        if self._image_backend is None:
            self._image_backend = load_backend(
                runtime=self._runtime,
                onnx_path=self._image_onnx_path,
                openvino_path=self._image_onnx_path.with_suffix(".xml"),
                num_threads=self._num_threads,
                openvino_device=self._openvino_device,
            )
        return self._image_backend

    def _text_tower(self):
        if self._text_backend is None:
            self._text_backend = load_backend(
                runtime=self._runtime,
                onnx_path=self._text_onnx_path,
                openvino_path=self._text_onnx_path.with_suffix(".xml"),
                num_threads=self._num_threads,
                openvino_device=self._openvino_device,
            )
        return self._text_backend

    def _tokenize(self, text: str) -> dict[str, np.ndarray]:
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            if not self._tokenizer_dir.exists():
                raise FileNotFoundError(
                    f"CLIP tokenizer not found at {self._tokenizer_dir}. "
                    "Run `python scripts/export_models.py` first."
                )
            self._tokenizer = AutoTokenizer.from_pretrained(str(self._tokenizer_dir))

        encoded = self._tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=_MAX_TEXT_TOKENS,
            return_tensors="np",
        )
        return {
            "input_ids": encoded["input_ids"].astype(np.int64),
            "attention_mask": encoded["attention_mask"].astype(np.int64),
        }

    # -- public API ----------------------------------------------------------

    def encode_image(self, image: np.ndarray) -> np.ndarray:
        backend = self._image_tower()
        blob = preprocess_image(image, self._image_size)
        outputs = backend.run({backend.input_names[0]: blob})
        vector = np.asarray(outputs[0]).reshape(-1).astype(np.float32)
        self._dimension = vector.shape[0]
        return l2_normalize(vector)

    def encode_text(self, text: str) -> np.ndarray:
        backend = self._text_tower()
        feeds = self._tokenize(text)
        outputs = backend.run({name: feeds[name] for name in backend.input_names if name in feeds})
        vector = np.asarray(outputs[0]).reshape(-1).astype(np.float32)
        self._dimension = vector.shape[0]
        return l2_normalize(vector)

    @property
    def dimension(self) -> int:
        return self._dimension
