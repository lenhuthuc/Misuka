"""Persistent BM25 sentence index used to suppress repeated assistant phrasing."""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

_VERSION = 1
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
_SENTENCE_RE = re.compile(r"\s*[^.!?\n]+(?:[.!?]+|(?=\n)|$)", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.casefold())


def split_sentences(text: str) -> list[str]:
    return [match.group(0) for match in _SENTENCE_RE.finditer(text) if match.group(0).strip()]


def take_complete_sentences(text: str) -> tuple[list[str], str]:
    """Split completed punctuation/newline-terminated sentences from a stream buffer."""
    completed: list[str] = []
    end = 0
    for match in _SENTENCE_RE.finditer(text):
        sentence = match.group(0)
        stripped = sentence.rstrip()
        if not stripped or stripped[-1] not in ".!?\n":
            break
        completed.append(sentence)
        end = match.end()
    return completed, text[end:]


class BM25RepetitionIndex:
    """A small persistent sparse index; documents are individual AI sentences.

    Similarity is the best BM25 score divided by the query sentence's self-score.
    This keeps the configured threshold stable as the corpus grows while retaining
    BM25's term-frequency, inverse-document-frequency and length normalization.
    """

    def __init__(
        self,
        path: Path,
        *,
        threshold: float = 0.78,
        min_tokens: int = 5,
        max_sentences: int = 2000,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.path = path
        self.threshold = threshold
        self.min_tokens = min_tokens
        self.max_sentences = max_sentences
        self.k1 = k1
        self.b = b
        self._docs: list[dict[str, object]] = []

    @property
    def size(self) -> int:
        return len(self._docs)

    def load(self) -> bool:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if payload.get("version") != _VERSION or not isinstance(payload.get("documents"), list):
                return False
            self._docs = [
                doc for doc in payload["documents"]
                if isinstance(doc, dict)
                and isinstance(doc.get("id"), str)
                and isinstance(doc.get("text"), str)
                and isinstance(doc.get("vector"), dict)
                and isinstance(doc.get("length"), int)
            ][-self.max_sentences:]
            return True
        except (OSError, ValueError, TypeError):
            self._docs = []
            return False

    def rebuild(self, messages: Iterable[tuple[int, str]]) -> None:
        self._docs = []
        for message_id, text in messages:
            self.add_message(message_id, text, persist=False)
        self.persist()

    def persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(
                {"version": _VERSION, "documents": self._docs},
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        temporary.replace(self.path)

    def add_message(self, message_id: int, text: str, *, persist: bool = True) -> None:
        prefix = f"message:{message_id}:"
        self._docs = [doc for doc in self._docs if not str(doc["id"]).startswith(prefix)]
        for sentence_number, sentence in enumerate(split_sentences(text)):
            tokens = tokenize(sentence)
            if tokens:
                vector = dict(Counter(tokens))
                self._docs.append({
                    "id": f"{prefix}{sentence_number}",
                    "text": sentence.strip(),
                    "vector": vector,
                    "length": len(tokens),
                })
        self._docs = self._docs[-self.max_sentences:]
        if persist:
            self.persist()

    def similarity(self, sentence: str, *, exclude_message_id: int | None = None) -> float:
        query = tokenize(sentence)
        if len(query) < self.min_tokens:
            return 0.0
        prefix = f"message:{exclude_message_id}:" if exclude_message_id is not None else None
        docs = [
            (Counter(doc["vector"]), int(doc["length"]))
            for doc in self._docs
            if prefix is None or not str(doc["id"]).startswith(prefix)
        ]
        if not docs:
            return 0.0

        document_frequency = Counter()
        for vector, _ in docs:
            document_frequency.update(vector.keys())
        average_length = sum(length for _, length in docs) / len(docs)
        query_counts = Counter(query)

        def score(frequencies: Counter[str], length: int) -> float:
            length_norm = 1.0 - self.b + self.b * length / max(average_length, 1.0)
            total = 0.0
            for term, query_frequency in query_counts.items():
                frequency = frequencies[term]
                if not frequency:
                    continue
                df = document_frequency[term]
                idf = math.log(1.0 + (len(docs) - df + 0.5) / (df + 0.5))
                total += idf * (frequency * (self.k1 + 1.0)) / (
                    frequency + self.k1 * length_norm
                ) * query_frequency
            return total

        best_match = max(score(vector, length) for vector, length in docs)
        self_score = score(query_counts, len(query))
        if self_score <= 0.0:
            return 0.0
        return min(1.0, best_match / self_score)

    def filter_text(
        self,
        text: str,
        *,
        exclude_message_id: int | None = None,
        fallback: str = "",
    ) -> tuple[str, int]:
        kept: list[str] = []
        removed = 0
        for sentence in split_sentences(text):
            if self.similarity(sentence, exclude_message_id=exclude_message_id) >= self.threshold:
                removed += 1
            else:
                kept.append(sentence)
        filtered = "".join(kept).strip()
        return (filtered or fallback), removed
