"""Inspect vad_phobert_final.pt checkpoint keys."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

ckpt_path = Path(__file__).parent.parent / "vad_phobert_final.pt"
ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
print("head_state keys:", list(ckpt["head_state"].keys()))
for k, v in ckpt["head_state"].items():
    print(f"  {k}: {tuple(v.shape)}")
