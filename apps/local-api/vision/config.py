"""Configuration for the image-understanding layer.

Everything tunable lives here so the pipeline can be re-pointed at different
model files, thresholds or a different VLM provider without touching logic.
Env vars use the `VISION_` prefix (`VISION_VLM_MODEL=...`), read from the same
`.env` the rest of the app uses.

The two lookup tables at the bottom -- Vietnamese noun to COCO label, and OCR
field aliases -- are module constants rather than settings fields: they are
data the router *reasons over*, too large for env vars, and every entry is a
deliberate mapping decision rather than a knob.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# parents[1] from this file (vision/config.py) is apps/local-api; parents[3] is
# the repo root, which is where assets/models already lives.
_APP_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = Path(__file__).resolve().parents[3]
_MODELS_DIR = _REPO_ROOT / "assets" / "models" / "vision"


class VisionSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="VISION_",
        case_sensitive=False,
        extra="ignore",
    )

    # -- Runtime -------------------------------------------------------------
    # No CUDA on this machine (i5-13500H + Iris Xe). "openvino" tries the iGPU
    # first and falls back to its own CPU plugin; "onnx" is plain ONNX Runtime
    # CPU. OpenVINO is preferred for the detector -- the iGPU is otherwise idle
    # while the inference threads are busy -- but ONNX Runtime is the only
    # backend guaranteed to be installed, so it is the default.
    runtime: str = Field(default="onnx")  # "onnx" | "openvino"
    openvino_device: str = Field(default="GPU")  # falls back to CPU if absent
    # Ceiling for every session's intra-op pool. The box has 16 threads; the
    # chat/TTS path needs the rest, and past 8 the detector stops scaling.
    num_threads: int = Field(default=8, ge=1, le=16)

    # -- Model artefacts (produced by scripts/export_models.py) --------------
    models_dir: Path = Field(default=_MODELS_DIR)
    yolo_onnx_path: Path = Field(default=_MODELS_DIR / "yolo11n.onnx")
    yolo_openvino_dir: Path = Field(default=_MODELS_DIR / "yolo11n_openvino_int8")
    yolo_input_size: int = Field(default=640)
    clip_image_onnx_path: Path = Field(default=_MODELS_DIR / "clip_image.onnx")
    clip_text_onnx_path: Path = Field(default=_MODELS_DIR / "clip_text.onnx")
    clip_tokenizer_dir: Path = Field(default=_MODELS_DIR / "clip_tokenizer")
    clip_image_size: int = Field(default=224)
    # Multilingual so a Vietnamese question and an English-captioned image land
    # in the same space; the image tower is the matching CLIP ViT-B/32.
    clip_text_model_id: str = Field(default="sentence-transformers/clip-ViT-B-32-multilingual-v1")
    clip_image_model_id: str = Field(default="sentence-transformers/clip-ViT-B-32")
    yolo_source_weights: str = Field(default="yolo11n.pt")

    # -- Thresholds ----------------------------------------------------------
    # A detection below this is noise the answerer would otherwise count as an
    # object.
    det_score_min: float = Field(default=0.50, ge=0.0, le=1.0)
    det_iou_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    # OCR is quoted back to the user verbatim, so its floor is stricter than
    # the detector's: a wrong digit in a price is worse than a missed line.
    ocr_score_min: float = Field(default=0.60, ge=0.0, le=1.0)
    # Share of the frame covered by OCR boxes above which the image reads as a
    # text surface (screenshot/document) rather than a photo.
    ocr_area_text_ratio: float = Field(default=0.30, ge=0.0, le=1.0)
    # Second, deliberately non-spec signal for the same question. The area rule
    # alone misses the most common real case: a UI screenshot whose text is
    # small against large charts and empty chrome (a Task Manager grab measures
    # around 0.10). Many recognised lines with nothing detected among them is a
    # text surface however little of the frame the glyphs cover. Set to 0 to
    # disable and leave only the area rule.
    ocr_min_lines_text: int = Field(default=8, ge=0)
    # Key/value density that separates a settings-style screenshot from prose.
    screenshot_min_pairs: int = Field(default=3, ge=0)
    screenshot_pair_ratio: float = Field(default=0.30, ge=0.0, le=1.0)
    # Cosine floor for picking which image a question is about. Below it the
    # router asks rather than guesses -- answering about the wrong image is a
    # worse failure than one extra turn.
    #
    # The margin here is thin and worth knowing about. Measured against a real
    # photo, question-shaped text (not caption-shaped) lands at 0.26-0.27 when
    # on topic and 0.20-0.21 when not, so 0.25 will occasionally ask "which
    # image?" about an image the question really was about. Left at 0.25
    # anyway: ranking picks the image, this only decides whether to ask, and
    # one measured photo is not enough evidence to move it.
    image_select_min_sim: float = Field(default=0.25, ge=0.0, le=1.0)
    # Cosine floor for reusing a cached VLM answer, against the CLIP text tower.
    #
    # 0.97, not the 0.80 that looks reasonable on paper. Text-to-text cosines
    # from `clip-ViT-B-32-multilingual-v1` sit in a compressed high band -- it
    # is a distilled mean-pooled DistilBERT, not a model trained to spread
    # sentences out. Measured on this machine against "Tại sao CPU chỉ chạy
    # 1.78 GHz?": three rephrasings scored 0.9979-0.9989, while seven plainly
    # different questions about the same screenshot scored 0.8095-0.9301. At
    # 0.80 every one of those seven would have been served the cached answer
    # to a question nobody asked, and nothing downstream can catch that. 0.97
    # sits in the empty band between the two clusters.
    #
    # Recalibrate this if the text tower changes -- the right value is a
    # property of the encoder, not of the task.
    vlm_cache_min_sim: float = Field(default=0.97, ge=0.0, le=1.0)

    # -- Thumbnail buffer ----------------------------------------------------
    # The original image is never stored. What survives ingest is a re-encoded
    # JPEG kept only long enough for a lazy VLM call in the same conversation.
    thumb_max_edge: int = Field(default=768, ge=64)
    thumb_quality: int = Field(default=80, ge=1, le=100)
    thumb_ttl_seconds: float = Field(default=900.0, ge=0.0)
    # Per-session cap on retained records. Bounds RAM even if a session never
    # ends; the oldest record is dropped first.
    max_images_per_session: int = Field(default=8, ge=1)

    # -- VLM (cloud, lazy) ---------------------------------------------------
    # Any OpenAI-compatible chat/completions endpoint that accepts image parts.
    vlm_base_url: str = Field(default="https://api.openai.com/v1")
    vlm_api_key: str = Field(default="")
    vlm_model: str = Field(default="gpt-4o-mini")
    vlm_timeout_seconds: float = Field(default=15.0, gt=0.0)
    # One retry, not more: the caller is a person waiting mid-conversation, and
    # a second failure inside 30s is a provider problem no third try will fix.
    vlm_max_retries: int = Field(default=1, ge=0)
    vlm_max_tokens: int = Field(default=300, ge=1)

    @property
    def resolved_yolo_onnx_path(self) -> Path:
        return self._resolve(self.yolo_onnx_path)

    @property
    def resolved_yolo_openvino_dir(self) -> Path:
        return self._resolve(self.yolo_openvino_dir)

    @property
    def resolved_clip_image_onnx_path(self) -> Path:
        return self._resolve(self.clip_image_onnx_path)

    @property
    def resolved_clip_text_onnx_path(self) -> Path:
        return self._resolve(self.clip_text_onnx_path)

    @property
    def resolved_clip_tokenizer_dir(self) -> Path:
        return self._resolve(self.clip_tokenizer_dir)

    @property
    def resolved_models_dir(self) -> Path:
        return self._resolve(self.models_dir)

    @staticmethod
    def _resolve(path: Path) -> Path:
        return path if path.is_absolute() else _APP_DIR / path


@lru_cache(maxsize=1)
def get_vision_settings() -> VisionSettings:
    return VisionSettings()


# -- Vietnamese noun -> COCO label -------------------------------------------
# Only classes YOLO11n actually predicts (COCO-80) may appear here: an entry
# for a class the detector cannot emit would turn "the model has never heard of
# this" into a confident "khong co". A noun *absent* from this table routes to
# the VLM, which is the right answer for anything COCO does not cover.
VI_TO_COCO: dict[str, str] = {
    "người": "person", "nguoi": "person", "em bé": "person", "trẻ em": "person",
    "đứa trẻ": "person", "khách": "person",
    "xe đạp": "bicycle", "xe dap": "bicycle",
    "ô tô": "car", "oto": "car", "xe hơi": "car", "xe ô tô": "car", "xe con": "car",
    "xe máy": "motorcycle", "xe may": "motorcycle", "mô tô": "motorcycle",
    "xe gắn máy": "motorcycle",
    "máy bay": "airplane", "phi cơ": "airplane",
    "xe buýt": "bus", "xe bus": "bus", "xe khách": "bus",
    "tàu hoả": "train", "tàu hỏa": "train", "xe lửa": "train",
    "xe tải": "truck", "xe tai": "truck",
    "thuyền": "boat", "tàu thuyền": "boat", "ghe": "boat",
    "đèn giao thông": "traffic light", "đèn đỏ": "traffic light",
    "biển báo dừng": "stop sign", "biển stop": "stop sign",
    "ghế dài": "bench", "băng ghế": "bench",
    "chim": "bird", "con chim": "bird",
    "mèo": "cat", "con mèo": "cat", "meo": "cat",
    "chó": "dog", "con chó": "dog", "cho": "dog", "cún": "dog",
    "ngựa": "horse", "con ngựa": "horse",
    "cừu": "sheep", "con cừu": "sheep",
    "bò": "cow", "con bò": "cow", "trâu": "cow",
    "voi": "elephant", "con voi": "elephant",
    "gấu": "bear", "con gấu": "bear",
    "ngựa vằn": "zebra", "hươu cao cổ": "giraffe",
    "ba lô": "backpack", "balo": "backpack", "cặp sách": "backpack",
    "ô": "umbrella", "dù": "umbrella", "cái ô": "umbrella",
    "túi xách": "handbag", "túi": "handbag",
    "cà vạt": "tie",
    "vali": "suitcase", "va li": "suitcase",
    "chai": "bottle", "chai nước": "bottle",
    "ly rượu": "wine glass",
    "cốc": "cup", "ly": "cup", "tách": "cup",
    "nĩa": "fork", "dĩa": "fork", "dao": "knife", "thìa": "spoon", "muỗng": "spoon",
    "bát": "bowl", "tô": "bowl", "chén": "bowl",
    "chuối": "banana", "táo": "apple", "cam": "orange", "quả cam": "orange",
    "bánh mì kẹp": "sandwich", "bông cải": "broccoli", "cà rốt": "carrot",
    "xúc xích": "hot dog", "pizza": "pizza", "bánh donut": "donut",
    "bánh ngọt": "cake",
    "ghế": "chair", "cái ghế": "chair",
    "sofa": "couch", "ghế sofa": "couch",
    "cây cảnh": "potted plant",
    "giường": "bed", "bàn ăn": "dining table", "bàn": "dining table",
    "bồn cầu": "toilet",
    "tivi": "tv", "ti vi": "tv", "màn hình": "tv", "tv": "tv",
    "laptop": "laptop", "máy tính xách tay": "laptop",
    "chuột": "mouse", "chuột máy tính": "mouse",
    "điều khiển": "remote", "remote": "remote",
    "bàn phím": "keyboard",
    "điện thoại": "cell phone", "đt": "cell phone",
    "lò vi sóng": "microwave", "lò nướng": "oven", "máy nướng bánh": "toaster",
    "bồn rửa": "sink", "tủ lạnh": "refrigerator",
    "sách": "book", "quyển sách": "book", "cuốn sách": "book",
    "đồng hồ": "clock", "bình hoa": "vase", "lọ hoa": "vase",
    "kéo": "scissors", "gấu bông": "teddy bear", "thú nhồi bông": "teddy bear",
    "máy sấy tóc": "hair drier", "bàn chải đánh răng": "toothbrush",
}

# label -> the Vietnamese word used when reading a count back to the user.
# First key wins, so the most natural term must come first in VI_TO_COCO.
COCO_TO_VI: dict[str, str] = {}
for _vi, _label in VI_TO_COCO.items():
    COCO_TO_VI.setdefault(_label, _vi)

# -- OCR field aliases -------------------------------------------------------
# canonical field -> surface forms that may appear either in the question or as
# the key half of an `ocr_pairs` entry. Matching is accent- and case-folded
# (see summary.normalize_text), so unaccented spellings need no entry of their
# own unless the accented form differs by more than diacritics.
OCR_FIELD_ALIASES: dict[str, list[str]] = {
    "memory": ["memory", "ram", "bộ nhớ", "dung lượng ram"],
    "cores": ["cores", "core", "nhân", "lõi", "số nhân", "nhân cpu"],
    # "luồng" belongs here, not under `threads`: asked about a CPU it means
    # hardware threads. Task Manager's own "Threads" tile is a different
    # quantity entirely -- how many threads the running processes have between
    # them -- so the English word gets its own field rather than aliasing this
    # one and answering "16" to a question about 3400.
    "logical_processors": [
        "logical processors", "logical processor", "luồng", "số luồng",
        "bộ xử lý logic",
    ],
    "threads": ["threads", "thread"],
    "speed": ["speed", "tốc độ", "xung nhịp", "ghz", "clock"],
    "base_speed": ["base speed", "tốc độ cơ bản", "xung cơ bản", "xung nhịp cơ bản"],
    "cpu": ["cpu", "bộ xử lý", "vi xử lý", "processor"],
    "utilization": ["utilization", "mức sử dụng", "sử dụng", "tải", "usage"],
    "uptime": ["up time", "uptime", "thời gian hoạt động", "thời gian bật"],
    "processes": ["processes", "tiến trình", "số tiến trình"],
    "handles": ["handles"],
    "sockets": ["sockets", "socket"],
    "gpu": ["gpu", "card đồ hoạ", "card đồ họa", "vga"],
    "disk": ["disk", "ổ đĩa", "ổ cứng", "dung lượng đĩa"],
    # "thành tiền" is deliberately absent: on a Vietnamese invoice it labels the
    # per-line amount column, not the grand total, and treating it as one turns
    # the price of the first item into the bill.
    "total": [
        "tổng cộng", "tổng tiền", "tổng", "total", "grand total",
        "phải trả", "thanh toán", "tổng thanh toán",
    ],
    "line_amount": ["thành tiền", "amount", "line total"],
    "subtotal": ["tạm tính", "subtotal", "cộng tiền hàng"],
    "vat": ["vat", "thuế", "thuế gtgt", "tax"],
    "discount": ["giảm giá", "chiết khấu", "discount", "khuyến mãi"],
    "price": ["giá", "đơn giá", "price", "unit price"],
    "quantity": ["số lượng", "quantity", "qty"],
    "date": ["ngày", "date", "ngày lập"],
    "invoice_no": ["số hoá đơn", "số hóa đơn", "invoice", "mã hoá đơn", "số phiếu"],
    "customer": ["khách hàng", "customer", "tên khách", "người mua"],
    "phone": ["điện thoại", "sđt", "phone", "số điện thoại"],
    "address": ["địa chỉ", "address"],
}
