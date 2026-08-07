import torch
from transformers import AutoTokenizer

from model.vad_model import MyModel


class VADService:
    def __init__(self, model: MyModel, tokenizer: AutoTokenizer, max_length: int = 100):
        self.model     = model
        self.tokenizer = tokenizer
        self.max_length = max_length

    def predict(self, text: str) -> tuple[float, float, float]:
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        with torch.no_grad():
            out = self.model(enc["input_ids"], enc["attention_mask"])
        v, a, d = out[0].tolist()
        return v, a, d
