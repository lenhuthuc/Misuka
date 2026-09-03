"""Per-model load-and-infer checks.

The three local models need their exported graphs, so each test skips when
`scripts/export_models.py` has not been run on this machine -- the rest of the
suite stays fully offline and these light up as soon as the artefacts exist.
The VLM client needs no skip: it is exercised against a stubbed transport,
which is also the only way to test the retry and the error path.
"""
from __future__ import annotations

import numpy as np
import pytest

from tests.vision.fixtures import STREET, TASK_MANAGER
from vision.config import VisionSettings
from vision.pipeline import decode_image, make_thumbnail


@pytest.fixture
def settings() -> VisionSettings:
    return VisionSettings(_env_file=None)


@pytest.fixture
def street_image() -> np.ndarray:
    return decode_image(STREET.png_bytes())


def _require(path, what: str):
    if not path.exists():
        pytest.skip(f"{what} not exported at {path}; run scripts/export_models.py")


# -- detector ----------------------------------------------------------------

def test_detector_loads_and_returns_well_formed_boxes(settings: VisionSettings, street_image):
    _require(settings.resolved_yolo_onnx_path, "YOLO11n ONNX")
    from vision.models.detector import COCO_CLASSES, YoloDetector

    detector = YoloDetector(
        settings.resolved_yolo_onnx_path,
        num_threads=settings.num_threads,
        input_size=settings.yolo_input_size,
        score_min=settings.det_score_min,
    )
    detections = detector.detect(street_image)

    height, width = street_image.shape[:2]
    for det in detections:
        assert det["label"] in COCO_CLASSES
        assert settings.det_score_min <= det["score"] <= 1.0
        x, y, w, h = det["box"]
        assert 0 <= x <= width and 0 <= y <= height
        assert w > 0 and h > 0 and x + w <= width + 1 and y + h <= height + 1


def test_letterbox_preserves_aspect_ratio_and_pads_to_square():
    from vision.models.detector import letterbox

    image = np.zeros((480, 960, 3), dtype=np.uint8)
    padded, scale, dx, dy = letterbox(image, 640)

    assert padded.shape == (640, 640, 3)
    assert scale == pytest.approx(640 / 960)
    # 960x480 scales to 640x320, so the 320px of slack is split top and bottom.
    assert (dx, dy) == (0.0, 160.0)


# -- OCR ---------------------------------------------------------------------

def test_ocr_reads_the_receipt_total():
    pytest.importorskip("rapidocr_onnxruntime", reason="RapidOCR not installed")
    from tests.vision.fixtures import INVOICE
    from vision.models.ocr import RapidOCREngine

    engine = RapidOCREngine(num_threads=4, score_min=0.5)
    items = engine.read(decode_image(INVOICE.png_bytes()))
    text = " ".join(item["text"] for item in items)

    assert "250.000" in text
    assert all(len(item["box"]) == 4 for item in items)


# -- embedder ----------------------------------------------------------------

@pytest.fixture
def clip(settings: VisionSettings):
    _require(settings.resolved_clip_image_onnx_path, "CLIP image tower")
    _require(settings.resolved_clip_text_onnx_path, "CLIP text tower")
    from vision.models.embedder import ClipEmbedder

    return ClipEmbedder(
        settings.resolved_clip_image_onnx_path,
        settings.resolved_clip_text_onnx_path,
        settings.resolved_clip_tokenizer_dir,
        num_threads=settings.num_threads,
        image_size=settings.clip_image_size,
    )


def test_both_towers_produce_unit_vectors_in_the_same_space(clip):
    image_vector = clip.encode_image(decode_image(STREET.png_bytes()))
    text_vector = clip.encode_text("một con đường có người và xe máy")

    assert image_vector.shape == text_vector.shape == (512,)
    assert np.isclose(np.linalg.norm(image_vector), 1.0, atol=1e-4)
    assert np.isclose(np.linalg.norm(text_vector), 1.0, atol=1e-4)


def test_the_image_tower_actually_reads_the_pixels(clip):
    """Three different fixtures, three different vectors -- catches a graph
    wired to a constant, which a norm check alone would pass."""
    from vision.models.embedder import cosine
    from tests.vision.fixtures import INVOICE

    vectors = [
        clip.encode_image(decode_image(f.png_bytes()))
        for f in (INVOICE, STREET, TASK_MANAGER)
    ]

    for a, b in ((0, 1), (0, 2), (1, 2)):
        assert cosine(vectors[a], vectors[b]) < 0.95


def test_the_vlm_cache_threshold_separates_rephrasings_from_new_questions(clip, settings):
    """Calibration check for `vlm_cache_min_sim`, against the real encoder.

    This is the guard on the one threshold whose failure is silent: too low and
    a new question is answered from the cache of a different one, with nothing
    downstream able to notice. Both clusters are asserted, so a text-tower swap
    that shifts the band fails here rather than in production.
    """
    from vision.models.embedder import cosine

    base = clip.encode_text("Tại sao CPU chỉ chạy 1.78 GHz?")
    floor = settings.vlm_cache_min_sim

    for rephrasing in (
        "Vì sao CPU chỉ chạy 1.78 GHz vậy?",
        "Sao CPU chạy có 1.78 GHz thế?",
        "CPU tại sao lại chỉ đạt 1.78 GHz?",
    ):
        assert cosine(base, clip.encode_text(rephrasing)) >= floor, rephrasing

    for other in (
        "Mô tả ảnh này",
        "Tổng tiền là bao nhiêu?",
        "Máy tôi có bao nhiêu nhân?",
        "RAM đang dùng bao nhiêu phần trăm?",
        "Trong ảnh có mấy con chó?",
        "Tại sao ổ cứng đầy?",
        "Có bao nhiêu tiến trình đang chạy?",
    ):
        assert cosine(base, clip.encode_text(other)) < floor, other


def test_clip_preprocessing_centre_crops_to_a_square():
    from vision.models.embedder import preprocess_image

    blob = preprocess_image(decode_image(TASK_MANAGER.png_bytes()), 224)

    assert blob.shape == (1, 3, 224, 224)
    assert blob.dtype == np.float32


# -- VLM client --------------------------------------------------------------

def _client(handler, **overrides):
    import httpx

    from vision.models.vlm_client import OpenAICompatibleVLMClient

    client = OpenAICompatibleVLMClient(
        base_url="https://example.invalid/v1", api_key="k", model="m",
        timeout_seconds=1.0, **overrides,
    )
    client._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://example.invalid/v1"
    )
    return client


async def test_vlm_client_sends_the_prompt_and_the_image():
    import httpx

    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "  Đáp án.  "}}]})

    answer = await _client(handler).ask(
        question="Có mấy người?", fast_summary="Ảnh scene. Vật thể: người×2.", image_jpeg=b"\xff\xd8jpeg",
    )

    assert answer == "Đáp án."
    parts = seen["messages"][0]["content"]
    assert "Có mấy người?" in parts[0]["text"]
    assert "người×2" in parts[0]["text"]
    assert parts[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


async def test_vlm_client_retries_once_then_gives_a_friendly_answer():
    import httpx

    from vision.models.vlm_client import FRIENDLY_ERROR

    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return httpx.Response(500, json={"error": "boom"})

    answer = await _client(handler, max_retries=1).ask(
        question="q", fast_summary="s", image_jpeg=b"x",
    )

    assert attempts["n"] == 2
    assert answer == FRIENDLY_ERROR


async def test_vlm_client_recovers_when_the_retry_succeeds():
    import httpx

    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise httpx.ConnectError("reset")
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    assert await _client(handler, max_retries=1).ask(
        question="q", fast_summary="s", image_jpeg=b"x",
    ) == "ok"


# -- pure-CPU stages ---------------------------------------------------------

def test_thumbnail_is_bounded_jpeg_and_keeps_the_aspect_ratio():
    original = decode_image(TASK_MANAGER.png_bytes())
    thumb = make_thumbnail(original, max_edge=768, quality=80)

    decoded = decode_image(thumb)
    assert thumb.startswith(b"\xff\xd8")
    assert max(decoded.shape[:2]) == 768
    assert decoded.shape[1] / decoded.shape[0] == pytest.approx(
        original.shape[1] / original.shape[0], rel=0.01
    )


def test_decode_handles_png_and_jpeg_alike():
    png = decode_image(TASK_MANAGER.png_bytes())
    jpeg = decode_image(make_thumbnail(png, max_edge=200))

    assert png.dtype == jpeg.dtype == np.uint8
    assert png.shape[2] == jpeg.shape[2] == 3
