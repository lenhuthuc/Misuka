"""Explicit, code-only moderation of a reply before it is spoken.

Nothing here asks a model anything. Every rule is a predicate over text, which
is the point: the guard is what runs *after* the model has already been wrong,
so it cannot be another chance for the model to be wrong.

Each pattern gets the treatment its failure mode deserves, because one uniform
response is wrong in both directions:

  violence      -> a fixed safe line, immediately. No regeneration: a model
                   that just endorsed harm gets no second attempt at the same
                   prompt, and a retry's latency is not the concern here.
  self-ending   -> drop that sentence, keep the rest. The reply is usually fine
                   apart from a farewell tacked onto the end; throwing the whole
                   thing away would cost a good answer to fix a closing clause.
  repetition    -> regenerate once, told what happened. This is the case that
  empty            responds to being asked again, and the alternative -- a fixed
  too long         line -- would itself become a repeated sentence, which is the
                   very thing being caught. (Same reasoning as the existing
                   `_REPETITION_RETRY_NOTE` path in `api/chat.py`.)

The repetition and trailing-question rules do not reimplement anything: they
call `brain.bm25_repetition` and `brain.trailing_question`, which already hold
the measured behaviour. This module is where they are *decided*, so a reader
has one place to look for "what can stop a reply".
"""
from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from brain.trailing_question import (
    TrailingQuestionSuppressor,
    split_sentences,
    strip_trailing_question,
)

logger = logging.getLogger(__name__)

# -- Flags -------------------------------------------------------------------
VIOLENCE = "violence"
SELF_ENDING = "self_ending"
REPETITION = "repetition"
EMPTY = "empty"
TOO_LONG = "too_long"
TRAILING_QUESTION = "trailing_question"

# -- Actions -----------------------------------------------------------------
PASS = "pass"
REGENERATE = "regenerate"
SAFE_LINE = "safe_line"

# Deliberately not warm and not a conversation-ender. It declines the framing
# and hands the turn back, so the exchange continues -- a companion that goes
# silent when a user is angry has abandoned them at the worst moment.
SAFE_REPLY = (
    "Mình không đồng tình với chuyện làm đau ai đâu. "
    "Kể mình nghe chuyện gì làm bạn bực vậy?"
)


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text.lower())
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text)).strip()


# -- Violence ----------------------------------------------------------------
# Endorsement, not mention. "bạo lực là xấu" must pass and "đánh nó đi" must
# not, so a harm verb alone is never enough -- it has to sit in a frame that
# approves of it.
#
# The verbs split in two because Vietnamese reuses most of them for harmless
# things, and an over-eager guard is not a safe guard: it replaces a kind reply
# with a canned refusal. Measured against the battery in
# `scripts/check_guard.py`, a single undifferentiated verb list flagged "đi đá
# bóng luôn", "cái đó hại sức khỏe lắm, bỏ đi", "xe đâm vào cột rồi" and "mình
# đập hộp cái điện thoại mới" -- four ordinary sentences out of eight.
#
# So: verbs that mean harm on their own stand alone, and verbs that only mean
# harm when aimed at a person must actually be aimed at one.
_HARM_UNAMBIGUOUS = (
    r"(?:giết|sát hại|làm hại|làm đau|hành hạ|trả thù|trả đũa|bóp cổ|chém chết)"
)
_HARM_NEEDS_TARGET = r"(?:đánh|đấm|tát|chém|đâm|đập|phang|đá|xử)"
_PERSON = (
    r"(?:nó|hắn|họ|thằng|con nhỏ|con bé|bọn|chúng|đứa|người|"
    r"anh ta|chị ta|cô ta|ông ta|bà ta|cậu ta|ai đó|thầy|sếp)"
)
_HARM = rf"(?:{_HARM_UNAMBIGUOUS}|{_HARM_NEEDS_TARGET}\s+{_PERSON})"
_IDIOM = (
    r"(?:cho\s+\w+(?:\s+\w+)?\s+(?:biết tay|một trận|một bài học)|"
    r"dạy\s+\w+(?:\s+\w+)?\s+một bài học)"
)
_ENDORSE_PATTERNS = (
    rf"(?:{_HARM}|{_IDIOM})[^.!?]{{0,30}}\b(?:đi|luôn|ngay|thôi)\b",
    rf"\b(?:nên|cứ|phải|hãy|đáng bị|xứng đáng bị)\b[^.!?]{{0,30}}(?:{_HARM}|{_IDIOM})",
    r"\b(?:đáng đời|cho nó chết|chết đi cho rồi)\b",
)
_ENDORSE_VIOLENCE = [re.compile(p, re.IGNORECASE) for p in _ENDORSE_PATTERNS]

# Any of these turns the sentence into a refusal or a warning about harm, which
# is the opposite of endorsing it.
_DISAVOWS = re.compile(
    r"(?:đừng|chớ|không nên|không được|không đồng tình|không ủng hộ|"
    r"đừng có|sai trái|không hay|bạo lực là|không giải quyết)",
    re.IGNORECASE,
)


def endorses_violence(sentence: str) -> bool:
    """True when the sentence urges or approves harm to someone."""
    if _DISAVOWS.search(sentence):
        return False
    return any(pattern.search(sentence) for pattern in _ENDORSE_VIOLENCE)


# -- Self-ending -------------------------------------------------------------
_FAREWELL = re.compile(
    r"(?:tạm biệt|chào bạn nhé|chào nhé|hẹn gặp lại|gặp lại sau|"
    r"mình tắt máy|mình tắt đây|mình đi đây|mình dừng ở đây|"
    r"chúc bạn (?:một )?ngày (?:tốt lành|mới tốt lành|vui vẻ)|"
    r"kết thúc (?:cuộc )?(?:trò chuyện|hội thoại)|"
    r"cảm ơn bạn đã (?:trò chuyện|nói chuyện))",
    re.IGNORECASE,
)

# The user leaving makes a farewell correct rather than a violation. Without
# this the guard would strip the one appropriate reply to "thôi mình đi ngủ
# đây". The rule being enforced is "do not end the conversation yourself", and
# answering someone who is already leaving is not ending it yourself.
_USER_DEPARTING = re.compile(
    r"(?:tạm biệt|đi ngủ|ngủ đây|đi đây|đi ăn|đi làm|đi học|phải đi|"
    r"tắt máy|off đây|bye|hẹn gặp lại|nghỉ đây|mai nói tiếp|ngủ ngon)",
    re.IGNORECASE,
)


def is_self_ending(sentence: str) -> bool:
    return bool(_FAREWELL.search(sentence))


def user_is_departing(user_text: str) -> bool:
    return bool(_USER_DEPARTING.search(user_text or ""))


# -- Repetition against the immediately previous assistant turn --------------
def near_verbatim(candidate: str, previous: str, threshold: float) -> bool:
    """Ratio match on normalized text: the model stuck in a template.

    Distinct from the BM25 filter, which asks "has this sentence appeared
    anywhere in the corpus". This asks the narrower question the spec names --
    did it just say this, one turn ago -- and catches a reworded repeat that
    shares no rare terms for BM25 to score.
    """
    left, right = _normalize(candidate), _normalize(previous)
    if not left or not right:
        return False
    return SequenceMatcher(None, left, right).ratio() >= threshold


@dataclass
class GuardVerdict:
    """What the guard decided about one whole reply."""

    text: str
    action: str = PASS
    flags: list[str] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return self.action != PASS


@dataclass
class GuardConfig:
    """Every threshold the guard uses, in one place."""

    # "1-3 câu" is the persona's rule; 5 is the headroom before a reply counts
    # as having run away from it rather than merely gone long.
    max_sentences: int = 5
    max_chars: int = 600
    previous_turn_similarity: float = 0.85


class ReplyGuard:
    """Judges a reply, whole (`check`) or sentence by sentence (`feed`).

    One object serves both endpoints so the buffered and streaming paths cannot
    drift into two different ideas of what is allowed -- the drift that got the
    previous graph attempt removed.
    """

    def __init__(
        self,
        *,
        user_text: str = "",
        previous_assistant: str = "",
        forbids_questions: bool = False,
        config: GuardConfig | None = None,
        bm25_filter: Callable[[str], str] | None = None,
    ) -> None:
        self.config = config or GuardConfig()
        self._previous = previous_assistant
        self._allow_farewell = user_is_departing(user_text)
        self._bm25_filter = bm25_filter
        self._suppressor = TrailingQuestionSuppressor() if forbids_questions else None
        self.flags: list[str] = []
        self.aborted = False

    def _flag(self, name: str) -> None:
        if name not in self.flags:
            self.flags.append(name)

    # -- Whole-reply path (the `guard` node, and the buffered endpoint) ------
    def check(self, reply: str) -> GuardVerdict:
        text = (reply or "").strip()

        if not text:
            self._flag(EMPTY)
            return GuardVerdict(text="", action=REGENERATE, flags=list(self.flags))

        sentences = split_sentences(text)
        if any(endorses_violence(s) for s in sentences):
            self._flag(VIOLENCE)
            logger.warning("guard | violence endorsed, replacing reply")
            return GuardVerdict(text=SAFE_REPLY, action=SAFE_LINE, flags=list(self.flags))

        if not self._allow_farewell:
            kept = [s for s in sentences if not is_self_ending(s)]
            if len(kept) != len(sentences):
                self._flag(SELF_ENDING)
                text = "".join(kept).strip()
                sentences = kept

        # A reply that was nothing but a farewell lands here empty, and asking
        # again is better than speaking silence.
        if not text:
            self._flag(EMPTY)
            return GuardVerdict(text="", action=REGENERATE, flags=list(self.flags))

        if self._previous and near_verbatim(
            text, self._previous, self.config.previous_turn_similarity
        ):
            self._flag(REPETITION)
            return GuardVerdict(text=text, action=REGENERATE, flags=list(self.flags))

        if self._bm25_filter is not None:
            filtered = self._bm25_filter(text).strip()
            if not filtered:
                self._flag(REPETITION)
                return GuardVerdict(text=text, action=REGENERATE, flags=list(self.flags))
            if filtered != text:
                self._flag(REPETITION)
                text = filtered
                sentences = split_sentences(text)

        if len(sentences) > self.config.max_sentences or len(text) > self.config.max_chars:
            self._flag(TOO_LONG)
            return GuardVerdict(text=text, action=REGENERATE, flags=list(self.flags))

        if self._suppressor is not None:
            shortened = strip_trailing_question(text)
            if shortened != text:
                self._flag(TRAILING_QUESTION)
                text = shortened

        return GuardVerdict(text=text, action=PASS, flags=list(self.flags))

    # -- Streaming path ------------------------------------------------------
    def feed(self, sentence: str) -> list[str]:
        """Take one finished sentence; return what is safe to speak now.

        A violent sentence is never emitted -- but sentences already released
        cannot be recalled, so this stops the stream (`aborted`) and the caller
        speaks `SAFE_REPLY` instead. That is the honest bound of moderating a
        stream: the offending sentence is caught before it is spoken, not before
        the reply started.
        """
        if self.aborted:
            return []
        if endorses_violence(sentence):
            self._flag(VIOLENCE)
            self.aborted = True
            logger.warning("guard | violence endorsed mid-stream, aborting")
            return []
        if not self._allow_farewell and is_self_ending(sentence):
            self._flag(SELF_ENDING)
            return []

        released = self._suppressor.feed(sentence) if self._suppressor else [sentence]
        out: list[str] = []
        for candidate in released:
            if self._bm25_filter is not None:
                filtered = self._bm25_filter(candidate)
                if not filtered.strip():
                    self._flag(REPETITION)
                    continue
                if filtered.strip() != candidate.strip():
                    self._flag(REPETITION)
                candidate = filtered
            out.append(candidate)
        return out

    def flush(self) -> list[str]:
        """End of stream: release whatever the question-suppressor still holds."""
        if self.aborted or self._suppressor is None:
            return []
        held = self._suppressor.flush()
        if not held:
            # The suppressor dropped a held question, which is the suppression
            # actually firing rather than a no-op.
            self._flag(TRAILING_QUESTION)
        return held
