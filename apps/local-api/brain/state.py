from __future__ import annotations

from typing import Any
from typing_extensions import TypedDict


class RetrievedDoc(TypedDict):
    doc_id: str
    content: str
    score: float
    metadata: dict[str, Any]
