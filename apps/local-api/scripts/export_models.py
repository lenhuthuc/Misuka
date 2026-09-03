"""Download and export the vision models this machine runs.

Produces, under `assets/models/vision/`:

    yolo11n.onnx                    detector, FP32 ONNX
    yolo11n_openvino_int8/          detector, INT8 OpenVINO IR (optional)
    clip_image.onnx                 CLIP ViT-B/32 vision tower + projection
    clip_text.onnx                  multilingual CLIP text tower + projection
    clip_tokenizer/                 tokenizer files for the text tower

Run once, offline afterwards:

    python scripts/export_models.py                 # everything, ONNX only
    python scripts/export_models.py --runtime openvino   # + INT8 IR for YOLO
    python scripts/export_models.py --only clip     # just the CLIP towers

`ultralytics` and `torch` are needed here and nowhere else in the vision layer
-- the runtime path loads the exported graphs directly. They are deliberately
not in requirements.txt: this is a one-time developer step, and pinning a
training stack into the service image to produce a file that ships as an
artefact is the wrong trade.
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from vision.config import get_vision_settings  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("export_models")


def export_yolo(models_dir: Path, weights: str, imgsz: int, runtime: str) -> None:
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics is not installed. `pip install ultralytics` and re-run.")
        return

    models_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Loading %s (downloads on first run)", weights)
    model = YOLO(weights)

    onnx_out = models_dir / "yolo11n.onnx"
    logger.info("Exporting ONNX -> %s", onnx_out)
    # opset 17 is what onnxruntime 1.23 handles without a shape-inference
    # fallback; dynamic=False because every call letterboxes to imgsz anyway
    # and a static graph is measurably faster on CPU.
    produced = Path(model.export(format="onnx", imgsz=imgsz, opset=17, dynamic=False, simplify=True))
    if produced.resolve() != onnx_out.resolve():
        shutil.move(str(produced), str(onnx_out))
    logger.info("ONNX detector: %s (%.1f MB)", onnx_out, onnx_out.stat().st_size / 1e6)

    if runtime != "openvino":
        return

    ir_out = models_dir / "yolo11n_openvino_int8"
    logger.info("Exporting INT8 OpenVINO IR -> %s", ir_out)
    try:
        # int8=True runs post-training quantisation against a small calibration
        # set ultralytics downloads itself; it is the only reason to prefer the
        # iGPU path at all, since FP32 on the Iris Xe is slower than the CPU.
        produced_dir = Path(model.export(format="openvino", imgsz=imgsz, int8=True))
        if produced_dir.resolve() != ir_out.resolve():
            if ir_out.exists():
                shutil.rmtree(ir_out)
            shutil.move(str(produced_dir), str(ir_out))
        logger.info("OpenVINO IR: %s", ir_out)
    except Exception as exc:
        logger.warning("OpenVINO export failed (%s); the ONNX detector still works", exc)


def export_clip(models_dir: Path, image_model_id: str, text_model_id: str, image_size: int) -> None:
    try:
        import torch
        from transformers import AutoTokenizer, CLIPVisionModelWithProjection
    except ImportError:
        logger.error("torch/transformers are not installed. `pip install torch transformers` and re-run.")
        return

    models_dir.mkdir(parents=True, exist_ok=True)

    # -- image tower ---------------------------------------------------------
    # sentence-transformers/clip-ViT-B-32 wraps the OpenAI checkpoint; going to
    # the HF model directly is the same weights with an ONNX-exportable
    # forward().
    image_out = models_dir / "clip_image.onnx"
    logger.info("Exporting CLIP image tower -> %s", image_out)
    vision = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").eval()

    class _ImageTower(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, pixel_values):
            return self.inner(pixel_values=pixel_values).image_embeds

    torch.onnx.export(
        _ImageTower(vision),
        (torch.randn(1, 3, image_size, image_size),),
        str(image_out),
        input_names=["pixel_values"],
        output_names=["image_embeds"],
        dynamic_axes={"pixel_values": {0: "batch"}, "image_embeds": {0: "batch"}},
        opset_version=17,
    )
    logger.info("CLIP image tower: %s (%.1f MB)", image_out, image_out.stat().st_size / 1e6)

    # -- text tower ----------------------------------------------------------
    # The multilingual model is a sentence-transformers stack: DistilBERT,
    # mean pooling, then a Dense layer projecting 768 -> 512 into CLIP's space.
    # All three have to be in the graph, so it is rebuilt here rather than
    # exported through the ST wrapper (which would bake in tokenisation too).
    text_out = models_dir / "clip_text.onnx"
    logger.info("Exporting CLIP text tower (%s) -> %s", text_model_id, text_out)

    from sentence_transformers import SentenceTransformer

    st_model = SentenceTransformer(text_model_id)
    transformer = st_model[0].auto_model.eval()
    dense = st_model[2] if len(st_model) > 2 else None

    class _TextTower(torch.nn.Module):
        def __init__(self, backbone, projection):
            super().__init__()
            self.backbone = backbone
            self.projection = projection

        def forward(self, input_ids, attention_mask):
            hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            if self.projection is not None:
                pooled = self.projection({"sentence_embedding": pooled})["sentence_embedding"]
            return pooled

    dummy_ids = torch.ones(1, 32, dtype=torch.long)
    dummy_mask = torch.ones(1, 32, dtype=torch.long)
    torch.onnx.export(
        _TextTower(transformer, dense),
        (dummy_ids, dummy_mask),
        str(text_out),
        input_names=["input_ids", "attention_mask"],
        output_names=["text_embeds"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "text_embeds": {0: "batch"},
        },
        opset_version=17,
    )
    logger.info("CLIP text tower: %s (%.1f MB)", text_out, text_out.stat().st_size / 1e6)

    tokenizer_dir = models_dir / "clip_tokenizer"
    AutoTokenizer.from_pretrained(text_model_id).save_pretrained(str(tokenizer_dir))
    logger.info("Tokenizer: %s", tokenizer_dir)


def main() -> int:
    settings = get_vision_settings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=["yolo", "clip"], help="export just one model family")
    parser.add_argument(
        "--runtime", choices=["onnx", "openvino"], default=settings.runtime,
        help="openvino also produces the INT8 IR for the detector",
    )
    parser.add_argument("--models-dir", type=Path, default=settings.resolved_models_dir)
    args = parser.parse_args()

    models_dir: Path = args.models_dir
    logger.info("Output directory: %s", models_dir)

    if args.only in (None, "yolo"):
        export_yolo(models_dir, settings.yolo_source_weights, settings.yolo_input_size, args.runtime)
    if args.only in (None, "clip"):
        export_clip(
            models_dir,
            settings.clip_image_model_id,
            settings.clip_text_model_id,
            settings.clip_image_size,
        )

    logger.info("Done. Run `python scripts/bench.py` to measure this machine.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
