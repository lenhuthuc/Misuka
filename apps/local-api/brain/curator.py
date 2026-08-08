"""Background worker that learns durable facts about the user from finished
conversations.

Why this runs on a queue instead of per-turn:

Facts are the only memory that survives both the history window and the vector
store's per-session churn, so they are worth extracting — but extraction is an
LLM call, and on this CPU the runner is serialised and memory-bandwidth-bound.
Issuing it per-turn as a fire-and-forget task queued it in front of the user's
next turn: measured, that pushed time-to-first-token from 3.3s to 16-19s.
Serving both concurrently is not an alternative either; two concurrent
generations measured 0.46x the aggregate throughput of one. A durable queue
lets the work wait for a quiet runner, survive being abandoned when the user
comes back, and amortise prefill across several exchanges per call.

Why there is no summarisation step:

An earlier design also asked a small model to distil each exchange into a
compact note, so retrieval could carry more memories per character. Measured on
real conversations it did not work at any setting tried: qwen2.5:0.5b
misattributed notes across a batch, batching down to one exchange per call
merely traded that for generic filler ("Got it, Thuc!"), and qwen2.5:1.5b
copied the user's sentence back verbatim instead of compressing it. The cause
was the question, not the model — "summarise this exchange" against an
advice-giving assistant yields advice, which is worthless as memory, whereas
"what about the user is still true next week" is the question worth asking, and
that is exactly what fact extraction already asks. Prompt growth, the original
motivation, is handled by the retrieval and history character budgets instead.

Call stack:

  run_memory_tasks (brain.background)
    -> MemoryService.enqueue_curation
    -> {@link MemoryCurator.notify}
         -> {@link MemoryCurator._run}          (long-lived worker task)
              -> LLMPriorityGate.run_when_idle
                   -> {@link MemoryCurator._prompt}
              -> MemoryService.upsert_fact
"""
from __future__ import annotations

import asyncio
import logging
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from brain.llm_service import LLMService
    from brain.memory_service import MemoryService
    from core.llm_priority import LLMPriorityGate

logger = logging.getLogger(__name__)

# At most this many facts per batch, stated here and in the prompt so the model
# and the parser agree on the ceiling.
_MAX_FACTS_PER_BATCH = 6

_PROMPT = f"""\
Extract durable facts about the user from the conversation below.
A durable fact is something still true next week: their name, language, level, \
goal, schedule, preferences, or persistent difficulties.
Output at most {_MAX_FACTS_PER_BATCH} facts, or NONE if there are none.

Write one fact per line as lower_snake_case_key: short value
Do not number the lines. Do not use asterisks, bold, or any other markdown.

{{exchanges}}
"""

_FACT_LINE = re.compile(r"^\s*(.+?)\s*:\s*(.+?)\s*$")
# Decoration the model adds despite being told not to: "1. ", "2) ", "**bold**".
# Stripping it before validation rather than rejecting the line outright,
# because a live run returned all six facts correct and every one of them was
# discarded purely for arriving as "1. **Name: Thuc**".
# Bullets and list numbering both appear in practice -- the model has returned
# "1. **Name: Thuc**" and "- **Name: Thuc**" on different runs of the same
# prompt. The whitespace after the marker is required, not optional: without it
# this strips the "6." out of a band score of "6.0" and stores the level as "0".
_DECORATION = re.compile(r"^\s*(?:[-+•]\s+|\d+\s*[.)]\s+)?[*_`\s]*|[*_`\s]*$")
_CONTENT_WORD = re.compile(r"[a-z0-9]{3,}")

# Words that appear in every prompt as scaffolding rather than as something the
# user said. Without this the speaker label teaches the model that the user is
# called "User" -- observed live, and it overwrote a correctly extracted name.
_SCAFFOLD_WORDS = {"user", "assistant", "exchange", "conversation"}

# A model that ignores the format keeps writing prose, and every prose line with
# a colon looks like a fact. Without these guards the table filled with markdown
# headings, numbered list items, and the prompt's own "key: value" placeholder --
# all observed in a live run, 16 junk rows and not one usable fact.
_PLACEHOLDER_KEYS = {"key", "value", "fact", "facts", "none", "note", "notes"}
_MAX_FACT_KEY_CHARS = 40
# Every key the model has actually produced here is one or two words ("name",
# "schedule", "IELTS band"). Three is where headings start: "Main Topic
# Discussion" survives the length and value checks and is caught only by this.
_MAX_FACT_KEY_WORDS = 2
# A stored fact is a value, not a paragraph. Real ones measured here run to
# about 60 characters ("After 9pm, due to work at a hospital"); the prose that
# once polluted this table ran to 150 and beyond.
_MAX_FACT_VALUE_CHARS = 100


_KEYLIKE = re.compile(r"^[A-Za-z][A-Za-z_ -]*$")


def _unwrap_nested(key: str, value: str) -> tuple[str, str]:
    """Drop a speaker prefix the model sometimes puts in front of a fact.

    Before:
    - ("Thuc", "persistent difficulties: Pronunciation of /th/ sounds")

    After:
    - ("persistent difficulties", "Pronunciation of /th/ sounds")

    Only unwraps when the inner key looks like a key, so a value that merely
    contains a colon -- "practice_time: 9:00 PM" -- is left alone.
    """
    m = _FACT_LINE.match(value)
    if not m:
        return key, value

    inner_key, inner_value = m.group(1).strip(), m.group(2).strip()
    if not _KEYLIKE.match(inner_key) or len(inner_key.split()) > _MAX_FACT_KEY_WORDS:
        return key, value
    return inner_key, inner_value


def _undecorate(text: str) -> str:
    """Strip list numbering and markdown emphasis from a fact line.

    Before:
    - "1. **Name"
    - "**Session Start "

    After:
    - "Name"
    - "Session Start"
    """
    return _DECORATION.sub("", text).strip()


def _is_usable_fact(key: str, value: str) -> bool:
    """Reject prose that merely happens to contain a colon.

    Before:
    - ("Main Topic Discussion", "Focus on the chosen topic and discuss it in
       detail using an introduction, main points, and summary structure.")
    - ("key", "value")

    After:
    - both rejected; only short identifier-like keys with short values survive,
      e.g. ("practice_time", "evening")
    """
    if not key or not value:
        return False
    if len(key) > _MAX_FACT_KEY_CHARS or len(value) > _MAX_FACT_VALUE_CHARS:
        return False
    if len(key.split()) > _MAX_FACT_KEY_WORDS:
        return False
    return key.lower() not in _PLACEHOLDER_KEYS


def _grounding_corpus(rows: list[dict[str, Any]]) -> set[str]:
    """Content words the batch actually contains, excluding prompt scaffolding.

    Built from the exchange text alone, never from the "User:"/"Assistant:"
    labels wrapped around it, so a value lifted from the scaffolding cannot pass
    as something the user said.
    """
    blob = " ".join(f"{r['query']} {r['response']}" for r in rows).lower()
    return set(_CONTENT_WORD.findall(blob)) - _SCAFFOLD_WORDS


def _is_grounded(value: str, corpus: set[str]) -> bool:
    """Require a fact's value to echo something the batch actually said.

    Asked for facts the conversation does not contain, the model invents
    plausible ones -- "age: 25" and "gender: Female" for a user who stated
    neither, and "name: User" lifted straight from the speaker label. Stored
    facts go into every later system prompt, and `upsert_fact` overwrites by
    key, so one invented value silently replaces a correct one and then keeps
    being asserted to the user. A missing fact costs recall once; a wrong one
    misinforms every turn that follows.
    """
    words = set(_CONTENT_WORD.findall(value.lower())) - _SCAFFOLD_WORDS
    if not words:
        return False
    return bool(words & corpus)


def _parse_facts(raw: str, corpus: set[str] | None = None) -> list[tuple[str, str]]:
    """Read durable facts out of the model's reply, keeping only grounded ones.

    Before:
    - "name: Thuc\\nage: 25\\n3. Main Topic Discussion: focus on the topic"

    After (with a corpus in which the user said "Thuc" but never an age):
    - [("name", "Thuc")]

    Passing no corpus skips the grounding check, which suits parsing tests but
    not production.
    """
    facts: list[tuple[str, str]] = []
    for line in raw.splitlines():
        if line.strip().upper() in {"", "NONE"}:
            continue
        m = _FACT_LINE.match(line)
        if not m or len(facts) >= _MAX_FACTS_PER_BATCH:
            continue

        key, value = _unwrap_nested(_undecorate(m.group(1)), _undecorate(m.group(2)))
        # Facts are upserted on an exact key match, so "Name" and "name" from
        # two different runs would become two facts asserting the same thing.
        # The model ignores the prompt's snake_case instruction often enough
        # that normalising here is the only thing that actually holds.
        key = key.lower()
        if not _is_usable_fact(key, value):
            continue
        if corpus is not None and not _is_grounded(value, corpus):
            logger.info("MemoryCurator | dropped ungrounded fact %r: %r", key, value)
            continue
        facts.append((key, value))
    return facts


class MemoryCurator:
    """Drains the curation queue whenever the LLM runner is quiet.

    Use when: the app is running; start it once from the lifespan and stop it
    on shutdown.

    Expects: a single instance per process. The queue is read without claiming
    rows, which is safe precisely because there is one consumer, and is what
    makes an abandoned batch retryable.

    Returns: nothing directly — its effect is durable facts in SQLite, which
    `build_messages` puts into every later system prompt.
    """

    def __init__(
        self,
        memory: "MemoryService",
        llm: "LLMService",
        gate: "LLMPriorityGate",
        batch_size: int = 5,
        idle_timeout: float = 30.0,
        retry_seconds: float = 20.0,
        max_attempts: int = 3,
    ) -> None:
        self._memory = memory
        self._llm = llm
        self._gate = gate
        self._batch_size = batch_size
        self._idle_timeout = idle_timeout
        self._retry_seconds = retry_seconds
        self._max_attempts = max_attempts
        self._work_ready = asyncio.Event()
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="memory_curator")
            logger.info("MemoryCurator started")

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        await asyncio.gather(self._task, return_exceptions=True)
        self._task = None
        logger.info("MemoryCurator stopped")

    def notify(self) -> None:
        """Wake the worker because something was just enqueued."""
        self._work_ready.set()

    async def _run(self) -> None:
        while True:
            try:
                if await self._memory.pending_curation_count(self._max_attempts) == 0:
                    self._work_ready.clear()
                    await self._work_ready.wait()
                    continue

                if not await self._process_batch():
                    # No quiet moment arrived. Back off rather than spin: the
                    # queue is durable, so waiting costs nothing but recall lag.
                    await asyncio.sleep(self._retry_seconds)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("MemoryCurator | batch failed, backing off")
                await asyncio.sleep(self._retry_seconds)

    async def _process_batch(self) -> bool:
        """Returns True if a batch was processed and cleared, False if the
        runner never went quiet (the batch stays queued for a later attempt)."""
        rows = await self._memory.next_curation_batch(self._batch_size, self._max_attempts)
        if not rows:
            return True

        ids = [row["id"] for row in rows]
        facts: list[tuple[str, str]] = []

        async def extract() -> None:
            nonlocal facts
            facts = _parse_facts(
                await self._llm.generate(self._prompt(rows)), _grounding_corpus(rows),
            )

        try:
            if not await self._gate.run_when_idle(extract, timeout=self._idle_timeout):
                return False
        except Exception:
            await self._memory.record_curation_failure(ids)
            raise

        # Writing touches SQLite only -- no LLM runner, so it is safe to finish
        # outside the gate while the user may already be typing again.
        for key, value in facts:
            await self._memory.upsert_fact(key, value)

        await self._memory.complete_curation(ids)
        logger.info(
            "MemoryCurator | %d exchange(s) curated, %d fact(s) learned", len(rows), len(facts),
        )
        return True

    def _prompt(self, rows: list[dict[str, Any]]) -> str:
        exchanges = "\n\n".join(
            f"Exchange {i}:\nUser: {row['query']}\nAssistant: {row['response']}"
            for i, row in enumerate(rows, 1)
        )
        return _PROMPT.format(exchanges=exchanges)
