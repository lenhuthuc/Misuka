import torch
import torch.nn as nn
from pathlib import Path
from transformers import AutoTokenizer


class MyModel(nn.Module):
    def __init__(self, bert, hidden=384, dropout=0.25):
        super(MyModel, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(768, hidden)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden, 3)

    def forward(self, input_ids, attention_mask):
        
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # mean pooling thay vì CLS
        mask = attention_mask.unsqueeze(-1).float()
        summed = (out.last_hidden_state * mask).sum(1)
        pooled = summed / mask.sum(1).clamp(min=1e-9)
        x = self.dropout(pooled)
        x = self.relu(self.linear1(x))
        x = self.dropout(x)
        return torch.tanh(self.linear2(x))


def _load_checkpoint(checkpoint_path: str):
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


_TOKENIZER_DIR = Path(__file__).resolve().parent / "tokenizer"


def load_model(checkpoint_path: str) -> tuple[MyModel, AutoTokenizer]:
    from transformers import RobertaConfig, RobertaModel

    
    state_dict = _load_checkpoint(checkpoint_path)
    print(state_dict["bert.embeddings.word_embeddings.weight"].shape)
    if isinstance(state_dict, dict) and "bert.embeddings.word_embeddings.weight" not in state_dict:
        state_dict = state_dict.get("model_state_dict") or state_dict.get("head_state") or state_dict

    hidden                   = state_dict["linear1.weight"].shape[0]
    vocab_size               = state_dict["bert.embeddings.word_embeddings.weight"].shape[0]
    hidden_size              = state_dict["bert.embeddings.word_embeddings.weight"].shape[1]
    max_position_embeddings  = state_dict["bert.embeddings.position_embeddings.weight"].shape[0]
    intermediate_size        = state_dict["bert.encoder.layer.0.intermediate.dense.weight"].shape[0]
    num_hidden_layers        = sum(
        1 for k in state_dict
        if k.startswith("bert.encoder.layer.") and k.endswith(".attention.self.query.weight")
    )

    config = RobertaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=hidden_size // 64,
        intermediate_size=intermediate_size,
        max_position_embeddings=max_position_embeddings,
        type_vocab_size=1,
    )

    bert      = RobertaModel(config)
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilroberta-base")

    model = MyModel(bert, hidden=hidden)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    return model, tokenizer
