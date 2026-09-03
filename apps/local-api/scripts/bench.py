"""Measure each vision model on this machine.

The whole architecture rests on a latency budget -- ingest under 1s, fast-path
answers under 100ms -- and those numbers are properties of *this* box, not of
the models. This script is how the README's figures get produced, and how a
runtime change (ONNX vs OpenVINO, thread count) is decided rather than guessed.

    python scripts/bench.py                     # everything, 20 runs each
    python scripts/bench.py --runs 50
    python scripts/bench.py --runtime openvino  # compare against the iGPU
    python scripts/bench.py --image path.jpg    # a real photo, not noise
    python scripts/bench.py --markdown          # table to paste into README

Anything that fails to load is reported and skipped, so a partial export still
gives partial numbers.
"""
from __future__ import annotations

import argparse
import logging
import statistics
import sys
import time
from pathlib import Path

import numpy as np

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from vision.config import get_vision_settings  # noqa: E402
from vision.pipeline import decode_image, make_thumbnail  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")


def _timeit(fn, runs: int, warmup: int = 3) -> dict[str, float]:
    """Median and p95 over `runs`, after discarding warmup.

    Median rather than mean: the first few calls after warmup still catch
    Windows scheduling noise and a cold page cache, and one 400ms outlier would
    move a mean enough to hide a real regression.
    """
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(runs):
        started = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - started) * 1000.0)
    samples.sort()
    return {
        "median_ms": statistics.median(samples),
        "p95_ms": samples[min(len(samples) - 1, int(0.95 * len(samples)))],
        "min_ms": samples[0],
        "max_ms": samples[-1],
    }


def _sample_image(path: Path | None) -> np.ndarray:
    if path is not None:
        return decode_image(path.read_bytes())
    # Structured noise, not flat grey: a uniform frame gives the detector no
    # anchors to score and OCR nothing to segment, so both finish unrealistically
    # fast and the numbers mean nothing.
    rng = np.random.default_rng(0)
    base = rng.integers(90, 170, size=(1080, 1920, 3), dtype=np.uint8)
    for y in range(80, 1000, 60):
        base[y:y + 22, 120:120 + rng.integers(200, 900)] = 20
    return base


def main() -> int:
    settings = get_vision_settings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--runtime", choices=["onnx", "openvino"], default=settings.runtime)
    parser.add_argument("--threads", type=int, default=settings.num_threads)
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--markdown", action="store_true", help="print a README-ready table")
    args = parser.parse_args()

    image = _sample_image(args.image)
    height, width = image.shape[:2]
    print(f"image      : {width}x{height}" + (f" ({args.image})" if args.image else " (synthetic)"))
    print(f"runtime    : {args.runtime}, {args.threads} threads, {args.runs} runs")
    print()

    results: list[tuple[str, dict[str, float]]] = []

    # -- detector ------------------------------------------------------------
    try:
        from vision.models.detector import YoloDetector

        detector = YoloDetector(
            settings.resolved_yolo_onnx_path,
            openvino_path=settings.resolved_yolo_openvino_dir / "yolo11n.xml",
            runtime=args.runtime,
            openvino_device=settings.openvino_device,
            num_threads=args.threads,
            input_size=settings.yolo_input_size,
            score_min=settings.det_score_min,
        )
        results.append(("YOLO11n detect", _timeit(lambda: detector.detect(image), args.runs)))
    except Exception as exc:
        print(f"YOLO11n detect     SKIPPED ({exc})")

    # -- OCR -----------------------------------------------------------------
    try:
        from vision.models.ocr import RapidOCREngine

        ocr = RapidOCREngine(num_threads=args.threads, score_min=settings.ocr_score_min)
        results.append(("RapidOCR read", _timeit(lambda: ocr.read(image), max(5, args.runs // 4))))
    except Exception as exc:
        print(f"RapidOCR read      SKIPPED ({exc})")

    # -- CLIP ----------------------------------------------------------------
    try:
        from vision.models.embedder import ClipEmbedder

        embedder = ClipEmbedder(
            settings.resolved_clip_image_onnx_path,
            settings.resolved_clip_text_onnx_path,
            settings.resolved_clip_tokenizer_dir,
            runtime=args.runtime,
            openvino_device=settings.openvino_device,
            num_threads=args.threads,
            image_size=settings.clip_image_size,
        )
        results.append(("CLIP encode_image", _timeit(lambda: embedder.encode_image(image), args.runs)))
        results.append((
            "CLIP encode_text",
            _timeit(lambda: embedder.encode_text("Trong ảnh có mấy người?"), args.runs),
        ))
    except Exception as exc:
        print(f"CLIP               SKIPPED ({exc})")

    # -- pure-CPU stages -----------------------------------------------------
    results.append((
        "thumbnail 768px",
        _timeit(lambda: make_thumbnail(image, settings.thumb_max_edge, settings.thumb_quality), args.runs),
    ))

    if args.markdown:
        print("| stage | median | p95 |")
        print("|---|---|---|")
        for name, stats in results:
            print(f"| {name} | {stats['median_ms']:.0f} ms | {stats['p95_ms']:.0f} ms |")
    else:
        print(f"{'stage':22s} {'median':>9s} {'p95':>9s} {'min':>9s} {'max':>9s}")
        for name, stats in results:
            print(
                f"{name:22s} {stats['median_ms']:>8.1f}ms {stats['p95_ms']:>8.1f}ms "
                f"{stats['min_ms']:>8.1f}ms {stats['max_ms']:>8.1f}ms"
            )

    # The three model stages run concurrently during ingest, so the budget is
    # bounded by the slowest of them, not their sum.
    model_medians = [s["median_ms"] for name, s in results if not name.startswith("thumbnail")]
    if model_medians:
        print()
        print(f"ingest lower bound (slowest concurrent stage): {max(model_medians):.0f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
