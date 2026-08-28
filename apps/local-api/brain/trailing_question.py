"""Remove a reply's trailing question on turns where a question is forbidden.

This exists because prompting could not close the gap on its own. On a
delegation turn -- the user has just answered the assistant's question by
handing the choice back -- the reply must not end by asking again, and against
mitsuka-ft the best-measured instruction still ended with a question one time
in three (see `_DELIVER_NOW_INSTRUCTION` in nodes/generate.py for the numbers).
A 1.7B has no setting at which a behavioural rule holds every time, so the last
third is taken deterministically, after generation, where it is exact.

Only the *trailing* question goes. A question in the middle of a reply is part
of how the reply reads and is left alone; the rule being enforced is "do not
hand the turn back", not "never use a question mark".
"""
from __future__ import annotations

import re

_SENTENCE_SPLIT = re.compile(r"[^.!?\n]*(?:[.!?\n]+|$)")

# A question the punctuation does not mark. Vietnamese ends yes/no questions
# with a bare final particle, and the model does sometimes close on one with a
# full stop rather than a question mark ("Bạn thấy thú vị không."). Deliberately
# a short list of sentence-final particles, checked only at the very end: "không"
# mid-sentence is a negation, not a question, and is far more common.
_INTERROGATIVE_TAIL = re.compile(
    r"(?:không|chưa|chứ|nhỉ|hả|hử|đúng không|phải không|được không|thế nào|ra sao)"
    r"\s*[.!]*\s*$",
    re.IGNORECASE,
)


def is_question(sentence: str) -> bool:
    """Return True if this sentence hands the turn back to the user."""
    stripped = sentence.strip()
    if not stripped:
        return False
    return "?" in stripped or bool(_INTERROGATIVE_TAIL.search(stripped))


def split_sentences(text: str) -> list[str]:
    """Split into sentences, preserving each one's trailing punctuation and spacing."""
    return [m.group(0) for m in _SENTENCE_SPLIT.finditer(text) if m.group(0).strip()]


def strip_trailing_question(text: str) -> str:
    """Drop trailing question sentences, unless that would empty the reply.

    Returning the model's own text when every sentence is a question follows
    the same rule the repetition filter uses: a reply that breaks the guard is
    a better failure than no reply at all. It is also the case worth logging --
    it means the turn produced nothing but a deflection.
    """
    sentences = split_sentences(text)
    if not sentences:
        return text

    kept = list(sentences)
    while kept and is_question(kept[-1]):
        kept.pop()
    if not kept:
        return text
    return "".join(kept).rstrip()


class TrailingQuestionSuppressor:
    """Streaming counterpart: holds a question back until something follows it.

    The buffered path can look at the whole reply, but a stream has to decide
    on each sentence as it completes, before knowing whether it is the last
    one. So a question is held rather than emitted; if more content arrives it
    is released (mid-reply questions are fine), and if the stream ends while it
    is still held, it is dropped -- which is exactly the case being suppressed.

    Cost is one sentence of added latency, and only on a turn that forbids
    questions and whose reply contains one.
    """

    def __init__(self) -> None:
        self._held: str | None = None
        self._emitted = False

    def feed(self, sentence: str) -> list[str]:
        """Take one completed sentence; return the sentences now safe to emit."""
        out: list[str] = []
        if self._held is not None:
            out.append(self._held)
            self._held = None
        if is_question(sentence):
            self._held = sentence
        else:
            out.append(sentence)
        if out:
            self._emitted = True
        return out

    def flush(self) -> list[str]:
        """End of stream: drop a still-held question, unless it is the whole reply."""
        held, self._held = self._held, None
        if held is None:
            return []
        return [] if self._emitted else [held]
