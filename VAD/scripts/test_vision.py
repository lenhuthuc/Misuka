"""
Test vision pipeline: VisualBuffer → CaptionService → VisionCapture (2 giây).

Chạy:
    cd D:\myProject\Mitsuka\VAD
    python scripts/test_vision.py                  # test buffer + caption (ảnh giả)
    python scripts/test_vision.py --webcam         # thêm webcam capture 5 giây
    python scripts/test_vision.py --screen         # thêm screen capture 5 giây
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_vision")


# ─── 1. VisualBuffer ──────────────────────────────────────────────────────────

async def test_buffer():
    print("\n[1] VisualBuffer")
    from brain.visual_buffer import VisualBuffer
    from brain.visual_memory import VisualMemory

    buf = VisualBuffer(maxlen=5)

    # Push 3 entries với embedding giả
    for i in range(3):
        emb = np.random.randn(384).astype(np.float32)
        emb /= np.linalg.norm(emb)
        await buf.push(VisualMemory(caption=f"scene {i}", embedding=emb))

    recent = await buf.get_recent_visual_context(k=2)
    assert len(recent) == 2, f"expected 2, got {len(recent)}"
    print(f"  [PASS] push/get_recent — got {[v.caption for v in recent]}")

    # Test cosine search
    query = recent[0].embedding
    results = await buf.search(query, top_k=1)
    assert results[0].caption == recent[0].caption
    print(f"  [PASS] search — top match: '{results[0].caption}'")

    # Test ring buffer maxlen
    for i in range(10):
        emb = np.random.randn(384).astype(np.float32)
        await buf.push(VisualMemory(caption=f"overflow {i}", embedding=emb))
    snap = buf.snapshot()
    assert len(snap) == 5, f"maxlen not enforced: {len(snap)}"
    print(f"  [PASS] ring buffer maxlen=5 enforced")


# ─── 2. CaptionService ────────────────────────────────────────────────────────

async def test_caption():
    print("\n[2] CaptionService")
    from brain.caption_service import CaptionService

    svc = CaptionService()
    print(f"  Backend loaded: {svc._backend}")

    # Tạo ảnh giả: gradient BGR 480x640
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:, :, 1] = np.linspace(0, 255, 640, dtype=np.uint8)   # green gradient

    caption = await svc.caption(frame)

    if svc._backend == "placeholder":
        print(f"  [SKIP] No VLM installed — caption returned: {repr(caption)}")
        print("         Để dùng VLM thật: pip install transformers accelerate")
        print("         moondream2 sẽ tự download khi CaptionService() được khởi tạo")
    else:
        print(f"  [PASS] caption: {repr(caption)}")

    # Luôn pass vì placeholder trả về "" là đúng
    print(f"  [PASS] caption() called without error")


# ─── 3. VisionCapture (ngắn) ─────────────────────────────────────────────────

async def test_capture(source: int | str, duration: float = 5.0):
    print(f"\n[3] VisionCapture source={repr(source)} for {duration}s")
    from brain.caption_service import CaptionService
    from brain.visual_buffer import VisualBuffer
    from brain.vision_capture import VisionCapture
    from brain.embeddings import EmbeddingModel

    buf     = VisualBuffer()
    caption = CaptionService()
    embedder = EmbeddingModel()
    capture = VisionCapture(
        capture_source=source,
        fps_check=2.0,
        diff_threshold=5.0,   # thấp hơn để dễ trigger khi test
    )

    task = asyncio.create_task(capture.run(buf, caption, embedder))

    await asyncio.sleep(duration)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    snap = buf.snapshot()
    print(f"  Captured {len(snap)} VisualMemory entries in {duration}s")
    for vm in snap[-3:]:
        print(f"    caption: {repr(vm.caption[:80])}")
    print(f"  [PASS] VisionCapture ran for {duration}s without crash")


# ─── Runner ───────────────────────────────────────────────────────────────────

async def main(args):
    await test_buffer()
    await test_caption()

    if args.webcam:
        await test_capture(source=0, duration=5.0)
    elif args.screen:
        await test_capture(source="screen", duration=5.0)

    print("\n=== Vision tests done ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--webcam",  action="store_true", help="Test webcam capture (5s)")
    parser.add_argument("--screen",  action="store_true", help="Test screen capture (5s)")
    args = parser.parse_args()
    asyncio.run(main(args))
