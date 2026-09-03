"""Tags, label/value pairs and the fast summary, over the real fixture layouts."""
from __future__ import annotations

from tests.vision.fixtures import INVOICE, STREET, TASK_MANAGER
from vision.config import COCO_TO_VI
from vision.summary import (
    TAG_DOCUMENT,
    TAG_PEOPLE,
    TAG_SCENE,
    TAG_SCREENSHOT,
    TAG_UNKNOWN,
    build_fast_summary,
    build_ocr_pairs,
    build_tags,
    normalize_text,
    sort_reading_order,
)


def _pairs(fixture) -> dict[str, str]:
    return {pair["key"]: pair["value"] for pair in build_ocr_pairs(fixture.ocr_items())}


# -- normalisation -----------------------------------------------------------

def test_normalize_folds_accents_case_and_the_vietnamese_d():
    assert normalize_text("TỔNG CỘNG") == "tong cong"
    assert normalize_text("Bộ  nhớ\n") == "bo nho"
    assert normalize_text("250.000đ") == "250.000d"


# -- ocr_pairs ---------------------------------------------------------------

def test_task_manager_pairs_recover_the_cpu_details_block():
    pairs = _pairs(TASK_MANAGER)

    assert pairs["Cores"] == "12"
    assert pairs["Logical processors"] == "16"
    assert pairs["Base speed"] == "2.60 GHz"
    assert pairs["Sockets"] == "1"


def test_a_value_stacked_under_its_label_is_paired():
    """The sidebar tile and the summary row both put the value on the next line."""
    pairs = _pairs(TASK_MANAGER)

    assert pairs["Memory"] == "17.8/39.6 GB (45%)"
    assert pairs["Utilization"] == "8%"
    assert pairs["Speed"] == "1.78 GHz"


def test_side_by_side_column_headings_are_not_paired_with_each_other():
    """"Utilization | Speed" over "8% | 1.78 GHz" is the trap this rule exists for."""
    pairs = _pairs(TASK_MANAGER)

    assert pairs["Utilization"] != "Speed"
    assert pairs["Processes"] == "250"
    assert pairs["Threads"] == "3400"
    assert pairs["Handles"] == "120000"


def test_a_label_far_across_the_frame_does_not_claim_a_value():
    """"Utilization" and "2.60 GHz" share a baseline 600px apart."""
    assert _pairs(TASK_MANAGER)["Utilization"] == "8%"


def test_a_box_already_used_as_a_value_is_not_reused_as_a_label():
    pairs = _pairs(TASK_MANAGER)

    assert "Enabled" not in pairs          # it is Virtualization's value
    assert pairs["Virtualization"] == "Enabled"
    assert pairs["L1 cache"] == "1.1 MB"


def test_invoice_total_is_paired_and_kept_apart_from_the_subtotal():
    pairs = _pairs(INVOICE)

    assert pairs["TỔNG CỘNG"] == "250.000đ"
    assert pairs["TẠM TÍNH"] == "250.000đ"
    assert pairs["Khách hàng"] == "Nguyễn Văn A"


def test_a_table_header_does_not_pair_across_the_rule_line_beneath_it():
    """A column heading sits a full line clear of its first row; a stacked
    label sits right on top of its value. One line of whitespace is what
    separates the two."""
    header_and_row = [
        {"text": "Thành tiền", "score": 1.0, "box": [540.0, 260.0, 100.0, 20.0]},
        {"text": "90.000", "score": 1.0, "box": [540.0, 300.0, 70.0, 20.0]},
    ]
    assert build_ocr_pairs(header_and_row) == []

    tile = [
        {"text": "Memory", "score": 1.0, "box": [24.0, 200.0, 82.0, 22.0]},
        {"text": "17.8 GB", "score": 1.0, "box": [24.0, 226.0, 90.0, 20.0]},
    ]
    assert build_ocr_pairs(tile) == [{"key": "Memory", "value": "17.8 GB"}]


def test_a_colon_key_whose_value_ocr_missed_stays_unpaired():
    """Regression: RapidOCR does not reliably detect isolated small digits, so
    the "1" and "12" beside "Sockets:" and "Cores:" go missing on a real grab.
    The next label down must not be served as the answer."""
    stack_with_values_missing = [
        {"text": "Base speed:", "score": 1.0, "box": [758.0, 563.0, 92.0, 20.0]},
        {"text": "Sockets:", "score": 1.0, "box": [758.0, 593.0, 66.0, 21.0]},
        {"text": "Cores:", "score": 1.0, "box": [758.0, 623.0, 52.0, 21.0]},
        {"text": "Logical processors:", "score": 1.0, "box": [757.0, 652.0, 148.0, 25.0]},
    ]

    assert build_ocr_pairs(stack_with_values_missing) == []


def test_a_colon_key_still_takes_a_label_shaped_value_to_its_right():
    """The colon says where the value is, so "Nguyễn Văn A" pairs even though
    it is short and word-shaped."""
    row = [
        {"text": "Khách hàng:", "score": 1.0, "box": [60.0, 210.0, 124.0, 20.0]},
        {"text": "Nguyễn Văn A", "score": 1.0, "box": [200.0, 210.0, 150.0, 20.0]},
    ]

    assert build_ocr_pairs(row) == [{"key": "Khách hàng", "value": "Nguyễn Văn A"}]


def test_pairs_are_empty_when_there_is_nothing_to_pair():
    assert build_ocr_pairs(STREET.ocr_items()) == []


# -- tags --------------------------------------------------------------------

def test_a_dense_ui_grab_is_tagged_as_a_text_surface():
    ocr = TASK_MANAGER.ocr_items()
    tags = build_tags([], ocr, TASK_MANAGER.size, build_ocr_pairs(ocr))

    assert tags[0] in {TAG_SCREENSHOT, TAG_DOCUMENT}


def test_a_screenshot_needs_key_value_density_not_just_text():
    ocr = TASK_MANAGER.ocr_items()
    pairs = build_ocr_pairs(ocr)

    assert build_tags([], ocr, TASK_MANAGER.size, pairs)[0] == TAG_SCREENSHOT
    # Same lines, no pairs recovered -> prose, not a settings pane.
    assert build_tags([], ocr, TASK_MANAGER.size, [])[0] == TAG_DOCUMENT


def test_a_person_detection_adds_the_people_tag():
    detections = [d for d in STREET.detections if d["score"] >= 0.5]
    tags = build_tags(detections, STREET.ocr_items(), STREET.size)

    assert TAG_PEOPLE in tags


def test_an_empty_image_is_unknown_and_a_photo_is_a_scene():
    assert build_tags([], [], (640, 480)) == [TAG_UNKNOWN]

    dogs = [{"label": "dog", "score": 0.9, "box": [10.0, 10.0, 50.0, 50.0]}]
    assert build_tags(dogs, [], (640, 480)) == [TAG_SCENE]


def test_a_few_ocr_lines_on_a_photo_do_not_make_it_a_document():
    """The line-count rule needs the frame to have nothing detected in it."""
    detections = [d for d in STREET.detections if d["score"] >= 0.5]
    tags = build_tags(detections, STREET.ocr_items(), STREET.size)

    assert TAG_SCREENSHOT not in tags
    assert TAG_DOCUMENT not in tags


def test_the_line_count_rule_can_be_switched_off():
    ocr = TASK_MANAGER.ocr_items()
    tags = build_tags([], ocr, TASK_MANAGER.size, build_ocr_pairs(ocr), min_lines_text=0)

    # OCR covers well under 30% of a 1280x800 grab, so only the area rule is
    # left and it does not fire.
    assert tags == [TAG_SCENE]


# -- fast_summary ------------------------------------------------------------

def test_summary_leads_with_pairs_then_the_remaining_lines():
    ocr = TASK_MANAGER.ocr_items()
    pairs = build_ocr_pairs(ocr)
    summary = build_fast_summary([TAG_SCREENSHOT], [], ocr, pairs)

    assert summary.startswith("Ảnh screenshot.")
    assert "Không phát hiện vật thể." in summary
    assert "Cores 12" in summary
    assert summary.index("Cores 12") < summary.index("100%")


def test_summary_counts_objects_in_vietnamese():
    detections = [d for d in STREET.detections if d["score"] >= 0.5]
    summary = build_fast_summary([TAG_PEOPLE], detections, [], [], COCO_TO_VI)

    assert "người×3" in summary
    assert "xe máy×2" in summary


def test_summary_truncates_and_says_how_much_it_dropped():
    ocr = [
        {"text": f"dòng số {i}", "score": 0.9, "box": [0.0, float(i * 30), 100.0, 20.0]}
        for i in range(60)
    ]
    summary = build_fast_summary([TAG_DOCUMENT], [], ocr, [], max_lines=40)

    assert "dòng số 39" in summary
    assert "dòng số 40" not in summary
    assert "+20 dòng nữa" in summary


def test_reading_order_is_top_down_then_left_right():
    items = [
        {"text": "b", "score": 1.0, "box": [200.0, 10.0, 10.0, 10.0]},
        {"text": "c", "score": 1.0, "box": [0.0, 90.0, 10.0, 10.0]},
        {"text": "a", "score": 1.0, "box": [10.0, 10.0, 10.0, 10.0]},
    ]

    assert [it["text"] for it in sort_reading_order(items)] == ["a", "b", "c"]
