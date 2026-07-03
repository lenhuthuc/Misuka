from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class VisualMemory:
    """One visual snapshot: VLM caption + its embedding vector."""
    caption: str
    embedding: np.ndarray
