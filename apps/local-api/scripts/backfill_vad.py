"""Backfill V/A/D + emotion for existing assistant messages in brain.db.

Chạy:
    cd D:\myProject\Mitsuka\VAD
    python scripts/backfill_vad.py              # backfill assistant rows thiếu V/A/D
    python scripts/backfill_vad.py --dry-run    # chỉ in ra, không ghi DB
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

VAD_ROOT = Path(__file__).resolve().parents[1]

_MIGRATION_COLUMNS = (("valence", "REAL"), ("arousal", "REAL"), ("dominance", "REAL"), ("emotion", "TEXT"))


def _ensure_columns(conn: sqlite3.Connection) -> None:
    existing = {row[1] for row in conn.execute("PRAGMA table_info(conversations)")}
    for name, sql_type in _MIGRATION_COLUMNS:
        if name not in existing:
            conn.execute(f"ALTER TABLE conversations ADD COLUMN {name} {sql_type}")
            print(f"  added column conversations.{name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill V/A/D + emotion for stored responses")
    parser.add_argument("--db", default=str(VAD_ROOT / "data" / "brain.db"), help="path to brain.db")
    parser.add_argument("--dry-run", action="store_true", help="print predictions without writing")
    args = parser.parse_args()

    from model.vad_model import load_model
    from service.vad_service import VADService
    from brain.emotion_mapper import map_vad_to_emotion

    model_path = os.getenv("VAD_MODEL_PATH", str(VAD_ROOT / "model" / "vad_bert_final.pt"))
    model, tokenizer = load_model(model_path)
    svc = VADService(model, tokenizer)

    conn = sqlite3.connect(args.db)
    _ensure_columns(conn)

    rows = conn.execute(
        "SELECT id, content FROM conversations WHERE role='assistant' AND valence IS NULL ORDER BY id"
    ).fetchall()
    print(f"{len(rows)} assistant messages to backfill in {args.db}")

    for row_id, content in rows:
        v, a, d = svc.predict(content)
        emotion = map_vad_to_emotion(v, a, d)
        print(f"  #{row_id}: v={v:+.3f} a={a:+.3f} d={d:+.3f} → {emotion}")
        if not args.dry_run:
            conn.execute(
                "UPDATE conversations SET valence=?, arousal=?, dominance=?, emotion=? WHERE id=?",
                (v, a, d, emotion, row_id),
            )

    if not args.dry_run:
        conn.commit()
        print("Backfill committed.")
    conn.close()


if __name__ == "__main__":
    main()
