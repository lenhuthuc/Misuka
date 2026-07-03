"""Inspect vad_phobert_final.pt checkpoint config."""
import os
import sys
from pathlib import Path

# Add VAD root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoModel, AutoTokenizer

ckpt_path = os.path.abspath(str(Path(__file__).parent.parent / "vad_phobert_final.pt"))
ckpt = torch.load(ckpt_path, map_location="cpu")
config = ckpt.get("config", {})
bert_name = config.get("bert_name")
print("bert_name:", bert_name)

tokenizer = AutoTokenizer.from_pretrained(bert_name)
print("Tokenizer vocab size:", len(tokenizer))

bert = AutoModel.from_pretrained(bert_name)
print("Model embedding size before resize:", bert.get_input_embeddings().weight.shape[0])
bert.resize_token_embeddings(len(tokenizer))
print("Model embedding size after resize:", bert.get_input_embeddings().weight.shape[0])
