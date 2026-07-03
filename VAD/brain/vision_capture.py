"""Background async vision producer — captures frames, detects scene changes, pushes VisualMemory."""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np

from brain.visual_memory import VisualMemory

if TYPE_CHECKING:
    from brain.visual_buffer import VisualBuffer

logger = logging.getLogger(__name__)

# Type alias for any object that has aembed_one(text) — accepts EmbeddingModel
Embedder = Callable[[str], "asyncio.coroutine"]


class VisionCapture:
    """
    Background async task that captures frames from webcam or screen,
    detects significant scene changes via MAD, captions changed frames
    with a VLM, and pushes VisualMemory entries into a VisualBuffer.
    """

    def __init__(
        self,
        capture_source: int | str = 0,     # webcam index or "screen"
        fps_check: float = 2.0,            # how many times per second to sample
        diff_threshold: float = 15.0,      # MAD threshold for scene change
        diff_resize: tuple[int, int] = (160, 90),
        frame_save_dir: str | Path | None = None,
    ) -> None:
        self.capture_source = capture_source
        self.fps_check = fps_check
        self.diff_threshold = diff_threshold
        self.diff_resize = diff_resize
        self.frame_save_dir = Path(frame_save_dir) if frame_save_dir else None

    async def run(self, buffer: "VisualBuffer", caption_svc, embedder) -> None:
        """
        Main loop. Runs until cancelled.

        Parameters
        ----------
        buffer:       VisualBuffer to push results into
        caption_svc:  CaptionService instance
        embedder:     EmbeddingModel instance (uses aembed_one)
        """
        import cv2

        if self.capture_source == "screen":
            cap = None
            use_screen = True
        else:
            cap = cv2.VideoCapture(self.capture_source)
            use_screen = False

        prev_small: np.ndarray | None = None
        interval = 1.0 / self.fps_check

        if self.frame_save_dir:
            self.frame_save_dir.mkdir(parents=True, exist_ok=True)

        try:
            frame_idx = 0
            while True:
                await asyncio.sleep(interval)

                frame = await asyncio.get_event_loop().run_in_executor(
                    None, self._grab_frame, cap, use_screen
                )
                if frame is None:
                    continue

                # Resize for diff comparison
                small = cv2.resize(frame, self.diff_resize).astype(np.float32)

                if prev_small is not None:
                    mad = float(np.abs(small - prev_small).mean())
                    if mad < self.diff_threshold:
                        continue  # no significant change

                prev_small = small
                logger.debug("VisionCapture | scene change detected, captioning frame %d", frame_idx)

                # Optional: save frame to disk
                if self.frame_save_dir:
                    save_path = self.frame_save_dir / f"frame_{frame_idx:06d}.jpg"
                    await asyncio.get_event_loop().run_in_executor(
                        None, cv2.imwrite, str(save_path), frame
                    )

                caption = await caption_svc.caption(frame)
                if not caption:
                    continue

                embedding = await embedder.aembed_one(caption)
                vm = VisualMemory(caption=caption, embedding=embedding)
                await buffer.push(vm)
                logger.info("VisionCapture | pushed: %r", caption[:80])
                frame_idx += 1

        except asyncio.CancelledError:
            logger.info("VisionCapture | stopped")
        finally:
            if cap is not None:
                cap.release()

    def _grab_frame(self, cap, use_screen: bool) -> np.ndarray | None:
        if use_screen:
            try:
                import mss
                import mss.tools
                with mss.mss() as sct:
                    monitor = sct.monitors[1]
                    sct_img = sct.grab(monitor)
                    arr = np.frombuffer(sct_img.bgra, dtype=np.uint8)
                    arr = arr.reshape((sct_img.height, sct_img.width, 4))
                    return arr[:, :, :3]  # BGRA → BGR
            except Exception as exc:
                logger.warning("Screen grab failed: %s", exc)
                return None
        else:
            ret, frame = cap.read()
            return frame if ret else None
