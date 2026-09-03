"""Tests for the image-bytes -> BGR-frame decode `caption()` needs.

`caption()`'s frame contract mirrors `cv2.VideoCapture`/`cv2.imread` (BGR
channel order) because that is what the webcam/screen capture path already
hands it. A one-off image upload has no OpenCV frame of its own, so
`decode_image_to_bgr` has to land on that exact same contract via Pillow.
"""
from __future__ import annotations

import io

import numpy as np
import pytest

from brain.caption_service import ImageDecodeError, decode_image_to_bgr


def _png_bytes(rgb: tuple[int, int, int], size: tuple[int, int] = (3, 2)) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", size, color=rgb).save(buf, format="PNG")
    return buf.getvalue()


def test_decodes_to_the_bgr_uint8_hwc_shape_caption_expects():
    frame = decode_image_to_bgr(_png_bytes((10, 20, 30), size=(4, 3)))
    assert frame.shape == (3, 4, 3)  # (height, width, channels)
    assert frame.dtype == np.uint8


def test_channel_order_is_bgr_not_rgb():
    """@example: a pure-red PNG must decode with red in channel index 2, not 0 --
    `caption()` flips this array back to RGB internally, so handing it RGB
    order here would silently swap red and blue for every captioned image."""
    frame = decode_image_to_bgr(_png_bytes((255, 0, 0)))
    b, g, r = frame[0, 0]
    assert (int(b), int(g), int(r)) == (0, 0, 255)


def test_rejects_bytes_that_are_not_a_readable_image():
    with pytest.raises(ImageDecodeError):
        decode_image_to_bgr(b"this is not a real image file")
