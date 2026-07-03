"""
Test suite cho VAD APIs.
Chạy server trước: uvicorn main:app --host 0.0.0.0 --port 8000
Sau đó chạy: python tests/test_apis.py
"""

import io
import json
import struct
import sys
import wave
from pathlib import Path

import requests

BASE = "http://localhost:8000"


def _ok(name: str, resp: requests.Response, expected_keys: list[str] | None = None):
    if resp.status_code == 200:
        if expected_keys:
            body = resp.json()
            missing = [k for k in expected_keys if k not in body]
            if missing:
                print(f"  [FAIL] {name} — missing keys: {missing}")
                print(f"         response: {body}")
                return False
        print(f"  [PASS] {name}")
        return True
    else:
        print(f"  [FAIL] {name} — HTTP {resp.status_code}: {resp.text[:200]}")
        return False


def _make_wav_bytes(duration_sec: float = 1.0, sample_rate: int = 16000) -> bytes:
    """Tạo WAV im lặng để test audio endpoints."""
    n_samples = int(duration_sec * sample_rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_samples)
    return buf.getvalue()


# ─── /health ──────────────────────────────────────────────────────────────────

def test_health():
    print("\n[health]")
    r = requests.get(f"{BASE}/health")
    return _ok("GET /health", r, ["status"])


# ─── /vad ─────────────────────────────────────────────────────────────────────

def test_vad():
    print("\n[VAD]")
    results = []

    # Happy path
    r = requests.post(f"{BASE}/vad", json={"text": "I am very happy today!"})
    ok = _ok("POST /vad - happy text", r, ["v", "a", "d"])
    if ok:
        body = r.json()
        v, a, d = body["v"], body["a"], body["d"]
        in_range = all(-1.0 <= x <= 1.0 for x in [v, a, d])
        if in_range:
            print(f"         v={v:.3f} a={a:.3f} d={d:.3f} (in [-1,1])")
        else:
            print(f"  [WARN] values out of [-1,1]: v={v} a={a} d={d}")
    results.append(ok)

    # Vietnamese text
    r = requests.post(f"{BASE}/vad", json={"text": "Tôi rất vui hôm nay"})
    results.append(_ok("POST /vad - vietnamese text", r, ["v", "a", "d"]))

    # Empty text
    r = requests.post(f"{BASE}/vad", json={"text": ""})
    results.append(_ok("POST /vad - empty text", r, ["v", "a", "d"]))

    # Missing field → 422
    r = requests.post(f"{BASE}/vad", json={})
    ok = r.status_code == 422
    print(f"  {'[PASS]' if ok else '[FAIL]'} POST /vad - missing text → 422")
    results.append(ok)

    return all(results)


# ─── /v1/audio/voices ─────────────────────────────────────────────────────────

def test_tts_voices():
    print("\n[TTS voices]")
    r = requests.get(f"{BASE}/v1/audio/voices")
    return _ok("GET /v1/audio/voices", r, ["voices"])


# ─── /v1/audio/speech ─────────────────────────────────────────────────────────

def test_tts_speech():
    print("\n[TTS speech]")
    results = []

    r = requests.post(f"{BASE}/v1/audio/speech", json={
        "model": "piper",
        "input": "Hello, this is a test.",
        "voice": "default",
    })
    ok = r.status_code == 200 and r.headers.get("content-type", "").startswith("audio/wav")
    print(f"  {'[PASS]' if ok else '[FAIL]'} POST /v1/audio/speech - returns WAV")
    if ok:
        print(f"         WAV size: {len(r.content)} bytes")
    else:
        print(f"         HTTP {r.status_code}: {r.text[:200]}")
    results.append(ok)

    # Empty input → 422
    r = requests.post(f"{BASE}/v1/audio/speech", json={"model": "piper", "input": "  ", "voice": "default"})
    ok = r.status_code == 422
    print(f"  {'[PASS]' if ok else '[FAIL]'} POST /v1/audio/speech - empty input → 422")
    results.append(ok)

    return all(results)


# ─── /v1/audio/transcriptions ─────────────────────────────────────────────────

def test_whisper():
    print("\n[Whisper STT]")
    results = []

    wav_bytes = _make_wav_bytes()
    r = requests.post(
        f"{BASE}/v1/audio/transcriptions",
        files={"file": ("test.wav", wav_bytes, "audio/wav")},
        data={"model": "whisper-1", "language": "en", "response_format": "json"},
    )
    ok = _ok("POST /v1/audio/transcriptions - silent WAV", r, ["text"])
    if ok:
        print(f"         transcript: {repr(r.json()['text'])}")
    results.append(ok)

    return all(results)


# ─── /v1/models ───────────────────────────────────────────────────────────────

def test_models():
    print("\n[models]")
    r = requests.get(f"{BASE}/v1/models")
    return _ok("GET /v1/models", r, ["object", "data"])


# ─── Runner ───────────────────────────────────────────────────────────────────

def main():
    print(f"Testing {BASE} ...")
    results = {
        "health":      test_health(),
        "vad":         test_vad(),
        "tts_voices":  test_tts_voices(),
        "tts_speech":  test_tts_speech(),
        "whisper":     test_whisper(),
        "models":      test_models(),
    }

    print("\n" + "=" * 50)
    passed = sum(results.values())
    total  = len(results)
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
    print(f"\n{passed}/{total} test groups passed")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
