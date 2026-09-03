"""Three reference images, plus the detector/OCR output they should produce.

Each fixture is declared once as a list of `(text, x, y, w, h)` boxes and then
used twice: to *draw* the PNG, and to script what the stand-in OCR returns.
Keeping those in one place is what makes the fixtures honest -- the scripted
boxes are the geometry the pixels actually have, so `build_ocr_pairs` is being
tested against a real layout rather than against numbers chosen to make it
pass. The same PNGs feed the real models when they are installed
(`tests/vision/test_models.py`, `scripts/bench.py --image`).

Written for eyeballing with:

    python -m tests.vision.fixtures --write
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

# Boxes are (text, x, y, w, h) in pixels.
Box = tuple[str, int, int, int, int]


@dataclass
class VisionFixture:
    """One image plus the ground truth two of the three models should return."""

    name: str
    size: tuple[int, int]  # (width, height)
    boxes: list[Box] = field(default_factory=list)
    detections: list[dict] = field(default_factory=list)
    # Stands in for a CLIP image embedding: the stand-in embedder encodes this
    # sentence instead of the pixels. See conftest.NgramEmbedder.
    caption: str = ""
    background: tuple[int, int, int] = (245, 245, 245)
    panels: list[tuple[int, int, int, int, tuple[int, int, int]]] = field(default_factory=list)

    def ocr_items(self, score: float = 0.95) -> list[dict]:
        return [
            {"text": text, "score": score, "box": [float(x), float(y), float(w), float(h)]}
            for text, x, y, w, h in self.boxes
        ]

    def png_bytes(self) -> bytes:
        return _render(self)


# -- fixture 1: Windows Task Manager, Performance tab ------------------------
# Layout traced from a real grab: a sidebar of resource tiles on the left, the
# big CPU chart in the middle, the summary row underneath it, and the static
# CPU details block on the right. Every value the acceptance tests ask for
# (Cores 12, Memory 45%, Speed 1.78 GHz) sits where Windows puts it.
TASK_MANAGER = VisionFixture(
    name="task_manager",
    size=(1280, 800),
    background=(32, 32, 32),
    panels=[
        (0, 0, 1280, 56, (24, 24, 24)),      # title bar
        (0, 56, 260, 744, (40, 40, 40)),     # sidebar
        (300, 140, 940, 380, (18, 30, 46)),  # CPU chart
    ],
    caption=(
        "Ảnh chụp màn hình Task Manager Windows, tab Performance, "
        "biểu đồ CPU, Memory, Cores, tốc độ xung nhịp"
    ),
    boxes=[
        ("Task Manager", 24, 16, 130, 24),
        ("Performance", 200, 16, 120, 24),
        # sidebar tiles
        ("CPU", 24, 90, 45, 22),
        ("8% 1.78 GHz", 24, 116, 120, 20),
        ("Memory", 24, 200, 82, 22),
        ("17.8/39.6 GB (45%)", 24, 226, 190, 20),
        ("Disk 0 (C:)", 24, 300, 105, 22),
        ("2%", 24, 326, 35, 20),
        ("Wi-Fi", 24, 380, 55, 22),
        ("S: 0 R: 1.2 Mbps", 24, 406, 165, 20),
        ("GPU 0", 24, 460, 65, 22),
        ("Intel(R) Iris(R) Xe", 24, 486, 175, 20),
        # chart header
        ("CPU", 300, 96, 50, 26),
        ("Intel(R) Core(TM) i5-13500H", 880, 98, 360, 22),
        ("% Utilization", 300, 150, 115, 18),
        ("100%", 1180, 150, 55, 18),
        ("60 seconds", 300, 500, 100, 18),
        # summary row
        ("Utilization", 300, 560, 105, 20),
        ("8%", 300, 588, 40, 26),
        ("Speed", 460, 560, 70, 20),
        ("1.78 GHz", 460, 588, 115, 26),
        ("Processes", 300, 660, 100, 20),
        ("250", 300, 688, 45, 26),
        ("Threads", 430, 660, 80, 20),
        ("3400", 430, 688, 60, 26),
        ("Handles", 570, 660, 80, 20),
        ("120000", 570, 688, 90, 26),
        ("Up time", 300, 740, 80, 20),
        ("0:05:12", 300, 766, 90, 22),
        # CPU details block
        ("Base speed:", 760, 560, 115, 20),
        ("2.60 GHz", 1000, 560, 100, 20),
        ("Sockets:", 760, 590, 85, 20),
        ("1", 1000, 590, 15, 20),
        ("Cores:", 760, 620, 68, 20),
        ("12", 1000, 620, 30, 20),
        ("Logical processors:", 760, 650, 180, 20),
        ("16", 1000, 650, 30, 20),
        ("Virtualization:", 760, 680, 130, 20),
        ("Enabled", 1000, 680, 85, 20),
        ("L1 cache:", 760, 710, 95, 20),
        ("1.1 MB", 1000, 710, 70, 20),
    ],
)

# -- fixture 2: a Vietnamese sales receipt -----------------------------------
# The 40px gap between the table header and the first item row is not padding
# for looks: it is what stops "Thành tiền" pairing with the first line amount,
# and a real printed receipt has the same rule line there.
INVOICE = VisionFixture(
    name="invoice",
    size=(700, 900),
    background=(252, 250, 245),
    caption="Ảnh hoá đơn bán hàng, tổng cộng tiền thanh toán, số tiền phải trả",
    boxes=[
        ("CỬA HÀNG ABC", 210, 40, 280, 30),
        ("HOÁ ĐƠN BÁN HÀNG", 230, 84, 240, 24),
        ("Ngày:", 60, 150, 58, 20),
        ("02/09/2026", 200, 150, 125, 20),
        ("Số hoá đơn:", 60, 180, 118, 20),
        ("HD-00123", 200, 180, 110, 20),
        ("Khách hàng:", 60, 210, 124, 20),
        ("Nguyễn Văn A", 200, 210, 150, 20),
        ("Mặt hàng", 60, 260, 95, 20),
        ("SL", 330, 260, 28, 20),
        ("Đơn giá", 420, 260, 80, 20),
        ("Thành tiền", 540, 260, 100, 20),
        ("Cà phê sữa", 60, 300, 110, 20),
        ("2", 330, 300, 14, 20),
        ("45.000", 420, 300, 70, 20),
        ("90.000", 540, 300, 70, 20),
        ("Bánh mì", 60, 330, 78, 20),
        ("2", 330, 330, 14, 20),
        ("35.000", 420, 330, 70, 20),
        ("70.000", 540, 330, 70, 20),
        ("Nước ép", 60, 360, 80, 20),
        ("3", 330, 360, 14, 20),
        ("30.000", 420, 360, 70, 20),
        ("90.000", 540, 360, 70, 20),
        ("TẠM TÍNH", 380, 440, 110, 22),
        ("250.000đ", 540, 440, 105, 22),
        ("TỔNG CỘNG", 380, 490, 132, 24),
        ("250.000đ", 540, 490, 110, 24),
        ("Cảm ơn quý khách!", 240, 580, 210, 20),
    ],
)

# -- fixture 3: a street scene -----------------------------------------------
# Detections stand in for a COCO photo: three people, two motorcycles, a car.
# Drawn as coloured blocks at exactly those boxes so the PNG and the ground
# truth agree; the acceptance test only ever reads the counts.
STREET = VisionFixture(
    name="street",
    size=(960, 640),
    background=(168, 178, 186),
    panels=[
        (0, 420, 960, 220, (96, 96, 100)),   # road
        (0, 0, 960, 180, (176, 196, 214)),   # sky
    ],
    caption="Ảnh đường phố có người đi bộ và xe máy chạy trên đường",
    boxes=[
        ("PHỞ HÀ NỘI", 620, 200, 180, 28),
    ],
    detections=[
        {"label": "person", "score": 0.93, "box": [120.0, 300.0, 70.0, 180.0]},
        {"label": "person", "score": 0.88, "box": [240.0, 310.0, 66.0, 172.0]},
        {"label": "person", "score": 0.74, "box": [530.0, 296.0, 62.0, 168.0]},
        {"label": "motorcycle", "score": 0.86, "box": [360.0, 380.0, 140.0, 110.0]},
        {"label": "motorcycle", "score": 0.67, "box": [640.0, 392.0, 132.0, 104.0]},
        {"label": "car", "score": 0.81, "box": [790.0, 340.0, 160.0, 120.0]},
        # Below det_score_min -- ingest must drop it, so a test that counts
        # people is also testing that the floor is applied.
        {"label": "person", "score": 0.31, "box": [880.0, 300.0, 40.0, 120.0]},
    ],
)

ALL_FIXTURES: tuple[VisionFixture, ...] = (TASK_MANAGER, INVOICE, STREET)


def _font(size: int):
    from PIL import ImageFont

    # Segoe UI covers Vietnamese and is present on every Windows install; the
    # bundled bitmap default does not, and would render the receipt as boxes.
    for candidate in ("C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf"):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _render(fixture: VisionFixture) -> bytes:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", fixture.size, fixture.background)
    draw = ImageDraw.Draw(image)

    for x, y, w, h, colour in fixture.panels:
        draw.rectangle([x, y, x + w, y + h], fill=colour)

    for det in fixture.detections:
        x, y, w, h = det["box"]
        shade = {"person": (60, 70, 90), "motorcycle": (120, 60, 50), "car": (60, 90, 120)}
        draw.rectangle([x, y, x + w, y + h], fill=shade.get(det["label"], (100, 100, 100)))

    dark_background = sum(fixture.background) < 330
    for text, x, y, w, h in fixture.boxes:
        colour = (235, 235, 235) if dark_background else (20, 20, 20)
        draw.text((x, y), text, fill=colour, font=_font(max(11, int(h * 0.85))))

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def write_all(directory: Path = FIXTURES_DIR) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for fixture in ALL_FIXTURES:
        path = directory / f"{fixture.name}.png"
        path.write_bytes(fixture.png_bytes())
        written.append(path)
    return written


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Render the vision test fixtures to PNG.")
    parser.add_argument("--write", action="store_true", help="write them to tests/vision/fixtures/")
    args = parser.parse_args()
    if args.write:
        for path in write_all():
            print(path)
    else:
        for fixture in ALL_FIXTURES:
            print(f"{fixture.name:14s} {fixture.size} {len(fixture.png_bytes()) / 1024:.0f} KB")
