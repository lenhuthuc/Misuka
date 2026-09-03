"""YOLO11n object detection.

Runs the exported graph directly rather than through ultralytics: the training
package pulls in torch and its own pre/post-processing, and this process
already carries a torch install for the VAD checkpoints that must not be woken
up on the image path. Pre-processing here is ~15 lines of numpy and matches
what the exporter baked into the graph.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

import numpy as np

from vision.models.backends import load_backend

logger = logging.getLogger(__name__)

# COCO-80, in the order YOLO11 emits them. Index is the class id in the graph
# output, so this list must not be re-sorted.
COCO_CLASSES: tuple[str, ...] = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)


class Detector(Protocol):
    """What the pipeline needs from a detector, so tests can substitute one."""

    def detect(self, image: np.ndarray) -> list[dict]:
        """RGB uint8 HxWx3 -> [{label, score, box:[x, y, w, h]}] in image pixels."""
        ...


def letterbox(image: np.ndarray, size: int) -> tuple[np.ndarray, float, float, float]:
    """Resize preserving aspect ratio and pad to a square, grey fill.

    Returns the padded image plus the scale and offsets needed to map boxes
    back to the original frame.
    """
    import cv2

    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    new_w, new_h = int(round(width * scale)), int(round(height * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    dx, dy = (size - new_w) // 2, (size - new_h) // 2
    canvas[dy:dy + new_h, dx:dx + new_w] = resized
    return canvas, scale, float(dx), float(dy)


class YoloDetector:
    """YOLO11n over ONNX Runtime or OpenVINO."""

    def __init__(
        self,
        onnx_path: Path,
        *,
        openvino_path: Path | None = None,
        runtime: str = "onnx",
        openvino_device: str = "GPU",
        num_threads: int = 8,
        input_size: int = 640,
        score_min: float = 0.50,
        iou_threshold: float = 0.45,
        class_names: tuple[str, ...] = COCO_CLASSES,
    ) -> None:
        self._input_size = input_size
        self._score_min = score_min
        self._iou = iou_threshold
        self._classes = class_names
        self._backend = load_backend(
            runtime=runtime,
            onnx_path=onnx_path,
            openvino_path=openvino_path,
            num_threads=num_threads,
            openvino_device=openvino_device,
        )
        self._input_name = self._backend.input_names[0]

    def detect(self, image: np.ndarray) -> list[dict]:
        padded, scale, dx, dy = letterbox(image, self._input_size)
        # NHWC uint8 RGB -> NCHW float32 in [0, 1], which is what the export
        # expects; ultralytics bakes no normalisation into the graph.
        blob = padded.transpose(2, 0, 1)[None].astype(np.float32) / 255.0

        outputs = self._backend.run({self._input_name: blob})
        return self._postprocess(outputs[0], scale, dx, dy, image.shape[1], image.shape[0])

    def _postprocess(
        self,
        raw: np.ndarray,
        scale: float,
        dx: float,
        dy: float,
        width: int,
        height: int,
    ) -> list[dict]:
        import cv2

        # (1, 4 + nc, anchors) -> (anchors, 4 + nc). Class scores are already
        # sigmoid-activated in the exported graph.
        preds = np.squeeze(raw, axis=0).T
        if preds.shape[1] < 5:
            return []

        boxes_cxcywh = preds[:, :4]
        class_scores = preds[:, 4:]
        class_ids = class_scores.argmax(axis=1)
        scores = class_scores[np.arange(class_scores.shape[0]), class_ids]

        keep = scores >= self._score_min
        if not keep.any():
            return []
        boxes_cxcywh, scores, class_ids = boxes_cxcywh[keep], scores[keep], class_ids[keep]

        # Undo the letterbox: centre/size in padded space -> xywh in image space.
        cx, cy, bw, bh = boxes_cxcywh.T
        x = (cx - bw / 2.0 - dx) / scale
        y = (cy - bh / 2.0 - dy) / scale
        w = bw / scale
        h = bh / scale

        x = np.clip(x, 0, width)
        y = np.clip(y, 0, height)
        w = np.clip(w, 0, width - x)
        h = np.clip(h, 0, height - y)

        xywh = np.stack([x, y, w, h], axis=1)
        indices = cv2.dnn.NMSBoxes(
            xywh.tolist(), scores.astype(np.float32).tolist(), self._score_min, self._iou
        )
        if len(indices) == 0:
            return []

        results = []
        for idx in np.array(indices).reshape(-1):
            cid = int(class_ids[idx])
            label = self._classes[cid] if cid < len(self._classes) else str(cid)
            results.append({
                "label": label,
                "score": round(float(scores[idx]), 4),
                "box": [round(float(v), 1) for v in xywh[idx]],
            })
        results.sort(key=lambda d: -d["score"])
        return results
