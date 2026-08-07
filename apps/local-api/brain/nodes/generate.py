from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from brain.memory_service import MemoryService

SYSTEM_PROMPT = """\
You are BrainMaster, a helpful and knowledgeable assistant.
Use the provided context to answer the user's question accurately and concisely.
If the context does not contain enough information, say so honestly.
{emotion_line}
Context:
{context}
"""


def _last_assistant_emotion(recent: list[dict]) -> str | None:
    """Most recent stored emotion label — the agent's current mood carried over."""
    return next(
        (row["emotion"] for row in reversed(recent) if row["role"] == "assistant" and row.get("emotion")),
        None,
    )


async def build_messages(
    query: str,
    context: str,
    memory: "MemoryService",
    recent_limit: int = 10,
) -> list[dict[str, str]]:
    """Build the full message list for a chat completion call."""
    recent = await memory.get_recent(recent_limit)
    history: list[dict[str, str]] = [
        {"role": r["role"], "content": r["content"]} for r in recent
    ]
    emotion = _last_assistant_emotion(recent)
    emotion_line = f"Your current emotional state is: {emotion}.\n" if emotion else ""
    system = SYSTEM_PROMPT.format(context=context, emotion_line=emotion_line)
    return [{"role": "system", "content": system}] + history + [{"role": "user", "content": query}]
