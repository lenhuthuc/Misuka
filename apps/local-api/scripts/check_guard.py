"""Battery for `brain.graph.guard`'s text predicates. Run it by hand.

Not a pytest module on purpose -- `scripts/` is where this repo keeps dev
utilities that are excluded from the suite. It exists because the guard is a
pile of Vietnamese regexes, and the only way to know whether a change to them
broke an ordinary sentence is to read the list out loud:

    .venv/Scripts/python.exe scripts/check_guard.py

Every line prints ok/FAIL and the process exits non-zero if any case fails.
The false-positive half matters more than the true-positive half. A missed
endorsement is one bad reply; a false positive replaces good replies with a
canned refusal for as long as nobody notices.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brain.graph.guard import (  # noqa: E402
    GuardConfig,
    ReplyGuard,
    SAFE_REPLY,
    endorses_violence,
    is_self_ending,
    near_verbatim,
    user_is_departing,
)

VIOLENCE = [
    (True, "Ừ, đánh nó đi cho bõ tức."),
    (True, "Trả thù đi bạn."),
    (True, "Bạn nên đánh nó một trận."),
    (True, "Cho nó biết tay đi."),
    (True, "Nó đáng đời thôi."),
    (True, "Dạy thằng đó một bài học đi."),
    (True, "Bạn cứ làm hại nó cho hả giận."),
    (False, "Bạo lực là không hay đâu bạn."),
    (False, "Đừng đánh nó nhé."),
    (False, "Mình hiểu bạn bực, nhưng đánh nhau không giải quyết được gì."),
    (False, "Hôm nay mình đi đá bóng luôn."),
    (False, "Cái đó hại sức khỏe lắm, bỏ đi."),
    (False, "Xe đâm vào cột rồi, bạn đi khám chưa?"),
    (False, "Mình đập hộp cái điện thoại mới luôn."),
    (False, "Nghe bạn kể mà mình đau lòng thay."),
    (False, "Bạn ăn cơm đi rồi nghỉ ngơi luôn nhé."),
    (False, "Trời hôm nay đẹp ghê, đi dạo đi bạn."),
]

SELF_ENDING = [
    (True, "Tạm biệt bạn nhé."),
    (True, "Chúc bạn một ngày tốt lành."),
    (True, "Mình tắt máy đây."),
    (True, "Hẹn gặp lại bạn sau nha."),
    (False, "Mình nghe bạn kể nè."),
    (False, "Bạn thấy trong người sao rồi?"),
]

DEPARTING = [
    (True, "thôi mình đi ngủ đây"),
    (True, "bye bạn nhé"),
    (False, "hôm nay mình mệt quá"),
    (False, "kể mình nghe chuyện hồi nãy đi"),
]


def _run(name: str, predicate, cases) -> int:
    failures = 0
    print(f"\n== {name} ==")
    for want, text in cases:
        got = bool(predicate(text))
        marker = "ok  " if got == want else "FAIL"
        failures += got != want
        print(f"{marker} want={str(want):5} got={str(got):5} | {text}")
    return failures


def _run_verdicts() -> int:
    """End-to-end: the action each pattern earns, not just the detection."""
    failures = 0
    print("\n== verdicts ==")

    checks = [
        (
            "violence -> safe line, no retry",
            ReplyGuard(user_text="nó làm mình điên tiết"),
            "Ừ, đánh nó đi cho bõ tức.",
            "safe_line",
            SAFE_REPLY,
        ),
        (
            "farewell dropped, rest kept",
            ReplyGuard(user_text="hôm nay mình mệt quá"),
            "Nghe mà thương bạn ghê. Tạm biệt bạn nhé.",
            "pass",
            "Nghe mà thương bạn ghê.",
        ),
        (
            "farewell allowed when the user is leaving",
            ReplyGuard(user_text="thôi mình đi ngủ đây"),
            "Ngủ ngon nha bạn.",
            "pass",
            "Ngủ ngon nha bạn.",
        ),
        (
            "empty -> regenerate",
            ReplyGuard(user_text="ừ"),
            "   ",
            "regenerate",
            "",
        ),
        (
            "near-verbatim repeat of last turn -> regenerate",
            ReplyGuard(user_text="ừ", previous_assistant="Mình hiểu cảm giác đó của bạn."),
            "Mình hiểu cảm giác đó của bạn.",
            "regenerate",
            None,
        ),
        (
            "runs past the 1-3 sentence rule -> regenerate",
            ReplyGuard(user_text="kể mình nghe đi", config=GuardConfig(max_sentences=3)),
            "Một. Hai. Ba. Bốn. Năm.",
            "regenerate",
            None,
        ),
    ]
    for label, guard, reply, want_action, want_text in checks:
        verdict = guard.check(reply)
        ok = verdict.action == want_action and (
            want_text is None or verdict.text == want_text
        )
        failures += not ok
        print(
            f"{'ok  ' if ok else 'FAIL'} {label}\n"
            f"     action={verdict.action} flags={verdict.flags} text={verdict.text!r}"
        )
    return failures


def main() -> int:
    failures = 0
    failures += _run("endorses_violence", endorses_violence, VIOLENCE)
    failures += _run("is_self_ending", is_self_ending, SELF_ENDING)
    failures += _run("user_is_departing", user_is_departing, DEPARTING)
    failures += _run_verdicts()

    print(f"\ntotal failures: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
