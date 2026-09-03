"""Stand-in models for the vision layer, so the tests stay offline.

The pipeline takes its detector, OCR, embedder and VLM as constructor
arguments, and this module supplies all four. That is not only for speed: with
scripted models the acceptance tests exercise the parts with the interesting
behaviour -- pair extraction, question routing, the VLM cache, TTL expiry --
against a fixed, inspectable input, instead of against whatever YOLO happened
to score today.

Two of the stand-ins deserve a note on what they do and do not prove:

`ScriptedDetector`/`ScriptedOCR` key off the image's pixel dimensions. Each
fixture has a distinct size, so decoding really happens and the record really
carries that fixture's ground truth, but nothing here tests YOLO or RapidOCR.
`tests/vision/test_models.py` does that, and skips when the exports are absent.

`NgramEmbedder` is character-trigram cosine, not CLIP. It is a real similarity
function with the same shape -- rephrasings score high, unrelated text scores
low -- which is what the cache threshold and the image picker need in order to
be tested at all. The numbers it produces are not CLIP's numbers, so the tests
assert on *which branch was taken*, never on a specific similarity.
"""
from __future__ import annotations

import sys
import zlib
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

APP_DIR = Path(__file__).resolve().parents[2]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from tests.vision.fixtures import ALL_FIXTURES, INVOICE, STREET, TASK_MANAGER, VisionFixture  # noqa: E402
from vision.buffer import ImageBuffer  # noqa: E402
from vision.config import VisionSettings  # noqa: E402
from vision.models.vlm_client import VLMClient  # noqa: E402
from vision.pipeline import VisionPipeline  # noqa: E402
from vision.summary import normalize_text  # noqa: E402

_BY_SIZE: dict[tuple[int, int], VisionFixture] = {f.size: f for f in ALL_FIXTURES}


def _fixture_for(image: np.ndarray) -> VisionFixture | None:
    height, width = image.shape[:2]
    return _BY_SIZE.get((width, height))


class ScriptedDetector:
    """Returns the fixture's ground-truth detections, unfiltered."""

    def __init__(self) -> None:
        self.calls = 0

    def detect(self, image: np.ndarray) -> list[dict]:
        self.calls += 1
        fixture = _fixture_for(image)
        return [dict(det) for det in fixture.detections] if fixture else []


class ScriptedOCR:
    """Returns the fixture's declared boxes, in declaration order."""

    def __init__(self, score: float = 0.95) -> None:
        self.calls = 0
        self._score = score

    def read(self, image: np.ndarray) -> list[dict]:
        self.calls += 1
        fixture = _fixture_for(image)
        return fixture.ocr_items(self._score) if fixture else []


class NgramEmbedder:
    """Character-trigram bag, hashed into a fixed-width unit vector.

    Images are embedded by encoding the fixture's caption, which is how a
    scripted image gets a vector comparable with a question. crc32 rather than
    `hash()` because Python salts string hashing per process and the tests
    would then pass or fail depending on the interpreter's startup.
    """

    def __init__(self, dim: int = 512) -> None:
        self._dim = dim
        self.text_calls: list[str] = []

    def encode_text(self, text: str) -> np.ndarray:
        self.text_calls.append(text)
        return self._vector(text)

    def encode_image(self, image: np.ndarray) -> np.ndarray:
        fixture = _fixture_for(image)
        return self._vector(fixture.caption if fixture else "")

    def _vector(self, text: str) -> np.ndarray:
        padded = f" {normalize_text(text)} "
        grams = Counter(padded[i:i + 3] for i in range(max(0, len(padded) - 2)))
        vector = np.zeros(self._dim, dtype=np.float32)
        for gram, count in grams.items():
            vector[zlib.crc32(gram.encode("utf-8")) % self._dim] += count
        return (vector / (np.linalg.norm(vector) + 1e-9)).astype(np.float32)

    @property
    def dimension(self) -> int:
        return self._dim


class MockVLM(VLMClient):
    """Records every call and answers from a canned reply.

    `calls` is what the acceptance tests assert on: the fast path is only worth
    having if it keeps this list empty for the questions it claims to cover.
    """

    def __init__(self, reply: str = "Đây là câu trả lời từ VLM.") -> None:
        self.reply = reply
        self.calls: list[dict] = []

    async def ask(self, *, question: str, fast_summary: str, image_jpeg: bytes) -> str:
        self.calls.append({
            "question": question,
            "fast_summary": fast_summary,
            "image_bytes": len(image_jpeg),
        })
        return f"{self.reply} (#{len(self.calls)})"

    @property
    def prompts(self) -> list[str]:
        from vision.models.vlm_client import build_prompt

        return [build_prompt(c["question"], c["fast_summary"]) for c in self.calls]


@pytest.fixture
def vision_settings() -> VisionSettings:
    """Production defaults, with two deliberate departures.

    The env file is ignored, so a developer's `.env` -- an API key, a tuned
    threshold -- cannot change what the tests assert.

    `vlm_cache_min_sim` is re-scaled for `NgramEmbedder`. The production 0.97
    is calibrated for the CLIP text tower, whose sentence cosines sit in a
    compressed high band; trigram cosines are spread much wider. Measured on
    the same probe set, the stand-in puts rephrasings at 0.69-0.81 and clearly
    different questions at 0.03-0.39, so 0.55 is the equivalent point -- the
    empty gap between the two clusters. The production value is verified
    against the real encoder in `test_models.py`.
    """
    return VisionSettings(_env_file=None, vlm_cache_min_sim=0.55)


@pytest.fixture
def mock_vlm() -> MockVLM:
    return MockVLM()


@pytest.fixture
def embedder() -> NgramEmbedder:
    return NgramEmbedder()


@pytest.fixture
def pipeline(vision_settings: VisionSettings, mock_vlm: MockVLM, embedder: NgramEmbedder) -> VisionPipeline:
    return VisionPipeline(
        detector=ScriptedDetector(),
        ocr=ScriptedOCR(),
        embedder=embedder,
        vlm=mock_vlm,
        settings=vision_settings,
        buffer=ImageBuffer(max_per_session=vision_settings.max_images_per_session),
    )


@pytest.fixture
def task_manager_png() -> bytes:
    return TASK_MANAGER.png_bytes()


@pytest.fixture
def invoice_png() -> bytes:
    return INVOICE.png_bytes()


@pytest.fixture
def street_png() -> bytes:
    return STREET.png_bytes()
