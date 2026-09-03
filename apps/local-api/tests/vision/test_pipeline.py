"""Acceptance tests: the ten routing outcomes the vision layer exists to get right.

Each one pins a `source`. That is the assertion that matters -- the text of an
answer can be reworded, but "this question was answered from OCR and did not
cost a VLM call" is the contract. `mock_vlm.calls` is checked alongside it
wherever the point of the test is that the cloud was *not* reached.
"""
from __future__ import annotations

import time
import warnings

import pytest

from vision.pipeline import (
    SOURCE_CLARIFY,
    SOURCE_DETECTIONS,
    SOURCE_EXPIRED,
    SOURCE_OCR,
    SOURCE_VLM,
    SOURCE_VLM_CACHE,
    VisionPipeline,
)

SESSION = "s-test"


@pytest.fixture
async def with_task_manager(pipeline: VisionPipeline, task_manager_png: bytes):
    record = await pipeline.ingest(task_manager_png, SESSION)
    return pipeline, record


@pytest.fixture
async def with_task_manager_and_invoice(pipeline: VisionPipeline, task_manager_png: bytes, invoice_png: bytes):
    first = await pipeline.ingest(task_manager_png, SESSION)
    second = await pipeline.ingest(invoice_png, SESSION)
    return pipeline, first, second


# -- ingest ------------------------------------------------------------------

async def test_ingest_extracts_text_pairs_and_a_thumbnail(with_task_manager):
    _, record = with_task_manager

    assert record.image_id == "img_1"
    assert record.tags[0] in {"screenshot", "document"}
    assert {"key": "Cores", "value": "12"} in record.ocr_pairs
    assert "Cores 12" in record.fast_summary
    assert record.thumb and record.thumb.startswith(b"\xff\xd8")  # JPEG SOI
    assert record.thumb_alive()
    assert max(record.width, record.height) > 768  # the original is not shrunk


async def test_ingest_applies_the_detection_score_floor(pipeline: VisionPipeline, street_png: bytes):
    record = await pipeline.ingest(street_png, SESSION)

    # The fixture carries a 0.31-confidence person that must not be counted.
    assert len(record.detections) == 6
    assert all(det["score"] >= 0.5 for det in record.detections)


async def test_ingest_numbers_images_per_session(pipeline: VisionPipeline, task_manager_png, invoice_png):
    first = await pipeline.ingest(task_manager_png, "a")
    second = await pipeline.ingest(invoice_png, "a")
    other = await pipeline.ingest(invoice_png, "b")

    assert (first.image_id, second.image_id) == ("img_1", "img_2")
    assert other.image_id == "img_1"


# -- the ten acceptance cases ------------------------------------------------

async def test_1_core_count_is_read_from_the_screenshot(with_task_manager, mock_vlm):
    pipeline, record = with_task_manager

    result = await pipeline.answer("Máy tôi có bao nhiêu nhân?", SESSION)

    assert result.source == SOURCE_OCR
    assert "12" in result.text
    assert result.image_id == record.image_id
    assert mock_vlm.calls == []


async def test_2_memory_percentage_is_read_from_the_screenshot(with_task_manager, mock_vlm):
    pipeline, _ = with_task_manager

    result = await pipeline.answer("RAM đang dùng bao nhiêu phần trăm?", SESSION)

    assert result.source == SOURCE_OCR
    assert "45" in result.text
    assert mock_vlm.calls == []


async def test_3_no_dogs_on_a_screenshot_is_answered_without_the_vlm(with_task_manager, mock_vlm):
    pipeline, _ = with_task_manager

    result = await pipeline.answer("Trong ảnh có mấy con chó?", SESSION)

    assert result.source == SOURCE_DETECTIONS
    assert "không" in result.text.lower()
    assert mock_vlm.calls == []


async def test_4_a_why_question_reaches_the_vlm_with_the_extracted_text(with_task_manager, mock_vlm):
    pipeline, _ = with_task_manager

    result = await pipeline.answer("Tại sao CPU chỉ chạy 1.78 GHz?", SESSION)

    assert result.source == SOURCE_VLM
    assert len(mock_vlm.calls) == 1
    assert "1.78 GHz" in mock_vlm.prompts[0]
    assert mock_vlm.calls[0]["image_bytes"] > 0


async def test_5_the_same_question_reworded_is_served_from_cache(with_task_manager, mock_vlm):
    pipeline, _ = with_task_manager

    first = await pipeline.answer("Tại sao CPU chỉ chạy 1.78 GHz?", SESSION)
    second = await pipeline.answer("Vì sao CPU chỉ chạy 1.78 GHz vậy?", SESSION)

    assert first.source == SOURCE_VLM
    assert second.source == SOURCE_VLM_CACHE
    assert second.text == first.text
    assert len(mock_vlm.calls) == 1


async def test_6_a_description_is_generated_once_and_then_reused(with_task_manager, mock_vlm):
    pipeline, _ = with_task_manager

    first = await pipeline.answer("Mô tả ảnh này", SESSION)
    second = await pipeline.answer("Mô tả ảnh này", SESSION)

    assert first.source == SOURCE_VLM
    assert second.source == SOURCE_VLM_CACHE
    assert second.text == first.text
    assert len(mock_vlm.calls) == 1


async def test_7_the_receipt_is_picked_out_of_two_images_by_similarity(
    with_task_manager_and_invoice, mock_vlm
):
    pipeline, _, invoice = with_task_manager_and_invoice

    result = await pipeline.answer("Tổng tiền là bao nhiêu?", SESSION)

    assert result.image_id == invoice.image_id
    assert result.source == SOURCE_OCR
    assert "250.000" in result.text
    assert mock_vlm.calls == []


async def test_8_a_demonstrative_picks_the_newest_image(with_task_manager_and_invoice):
    pipeline, _, invoice = with_task_manager_and_invoice

    result = await pipeline.answer("Ảnh này là gì?", SESSION)

    assert result.image_id == invoice.image_id


async def test_9_people_are_counted_from_the_detector(pipeline: VisionPipeline, street_png, mock_vlm):
    await pipeline.ingest(street_png, SESSION)

    result = await pipeline.answer("Có mấy người trong ảnh?", SESSION)

    assert result.source == SOURCE_DETECTIONS
    assert "3" in result.text
    assert mock_vlm.calls == []


async def test_10_an_expired_thumbnail_asks_for_the_image_again(with_task_manager, mock_vlm):
    pipeline, record = with_task_manager
    record.thumb_expires_at = time.time() - 1.0

    result = await pipeline.answer("Tại sao CPU chậm?", SESSION)

    assert result.source == SOURCE_EXPIRED
    assert "gửi lại" in result.text
    assert mock_vlm.calls == []


# -- surrounding behaviour ---------------------------------------------------

async def test_a_question_with_no_image_in_the_session_says_so(pipeline: VisionPipeline):
    result = await pipeline.answer("Trong ảnh có gì?", "empty-session")

    assert result.source == SOURCE_CLARIFY
    assert result.image_id is None


async def test_an_ambiguous_question_over_two_images_asks_which_one(
    with_task_manager_and_invoice, mock_vlm
):
    pipeline, _, _ = with_task_manager_and_invoice

    result = await pipeline.answer("Hôm nay trời thế nào?", SESSION)

    assert result.source == SOURCE_CLARIFY
    assert mock_vlm.calls == []


async def test_a_cached_answer_survives_thumbnail_expiry(with_task_manager, mock_vlm):
    """The thumbnail is the only thing with a TTL; answers already paid for
    are text and stay for the session."""
    pipeline, record = with_task_manager
    await pipeline.answer("Tại sao CPU chỉ chạy 1.78 GHz?", SESSION)
    record.thumb_expires_at = time.time() - 1.0

    result = await pipeline.answer("Vì sao CPU chỉ chạy 1.78 GHz vậy?", SESSION)

    assert result.source == SOURCE_VLM_CACHE
    assert len(mock_vlm.calls) == 1


async def test_a_count_question_on_a_photo_defers_to_the_vlm(pipeline: VisionPipeline, street_png, mock_vlm):
    """Zero dogs in a photograph is not evidence of no dogs -- unlike zero dogs
    in a screenshot, which test 3 answers outright."""
    await pipeline.ingest(street_png, SESSION)

    result = await pipeline.answer("Trong ảnh có mấy con chó?", SESSION)

    assert result.source == SOURCE_VLM
    assert len(mock_vlm.calls) == 1


async def test_every_result_carries_a_source_and_a_latency(with_task_manager):
    pipeline, _ = with_task_manager

    for question in ("Máy tôi có bao nhiêu nhân?", "Trong ảnh có mấy con chó?", "Tại sao CPU chậm?"):
        result = await pipeline.answer(question, SESSION)
        assert result.source in {
            SOURCE_DETECTIONS, SOURCE_OCR, SOURCE_VLM, SOURCE_VLM_CACHE,
            SOURCE_CLARIFY, SOURCE_EXPIRED,
        }
        assert result.latency_ms >= 0.0


async def test_the_buffer_drops_the_oldest_image_past_its_cap(
    vision_settings, mock_vlm, embedder, task_manager_png, invoice_png, street_png
):
    from vision.buffer import ImageBuffer
    from tests.vision.conftest import ScriptedDetector, ScriptedOCR

    pipeline = VisionPipeline(
        detector=ScriptedDetector(), ocr=ScriptedOCR(), embedder=embedder, vlm=mock_vlm,
        settings=vision_settings, buffer=ImageBuffer(max_per_session=2),
    )
    for image in (task_manager_png, invoice_png, street_png):
        await pipeline.ingest(image, SESSION)

    kept = pipeline.buffer.list_session(SESSION)
    assert [r.image_id for r in kept] == ["img_2", "img_3"]


async def test_sweeping_frees_thumbnails_but_keeps_the_extracted_text(with_task_manager):
    pipeline, record = with_task_manager
    record.thumb_expires_at = time.time() - 1.0

    assert pipeline.buffer.sweep() == 1
    assert record.thumb is None
    assert record.ocr_pairs and record.fast_summary


# -- latency -----------------------------------------------------------------
# Soft budgets: the numbers below are what this machine hits when it is not
# thermally throttled, and a CI box or a laptop on battery will miss them
# without anything being wrong. Missing the target warns; the hard assertion is
# an order of magnitude looser and only catches an actual regression.

def _check_latency(name: str, elapsed_ms: float, budget_ms: float) -> None:
    if elapsed_ms > budget_ms:
        warnings.warn(
            f"{name} took {elapsed_ms:.0f}ms, over the {budget_ms:.0f}ms budget "
            "(machine may be throttling)",
            stacklevel=2,
        )
    assert elapsed_ms < budget_ms * 10


async def test_ingest_stays_inside_its_budget(pipeline: VisionPipeline, task_manager_png):
    started = time.perf_counter()
    await pipeline.ingest(task_manager_png, SESSION)
    _check_latency("ingest", (time.perf_counter() - started) * 1000.0, 1500.0)


async def test_fast_path_answers_stay_inside_their_budget(with_task_manager):
    pipeline, _ = with_task_manager

    for question in (
        "Máy tôi có bao nhiêu nhân?",
        "RAM đang dùng bao nhiêu phần trăm?",
        "Trong ảnh có mấy con chó?",
    ):
        result = await pipeline.answer(question, SESSION)
        _check_latency(f"answer({question})", result.latency_ms, 150.0)


async def test_warmup_touches_every_model_and_survives_a_broken_one(
    vision_settings, mock_vlm, embedder, task_manager_png
):
    from tests.vision.conftest import ScriptedDetector

    class BrokenOCR:
        def read(self, image):
            raise RuntimeError("weights missing")

    detector = ScriptedDetector()
    pipeline = VisionPipeline(
        detector=detector, ocr=BrokenOCR(), embedder=embedder, vlm=mock_vlm,
        settings=vision_settings,
    )
    await pipeline.warmup()

    assert detector.calls == 1
    # A model that cannot load must not stop the others, nor the ingest after it.
    record = await pipeline.ingest(task_manager_png, SESSION)
    assert record.ocr == []
    assert record.clip_emb is not None
