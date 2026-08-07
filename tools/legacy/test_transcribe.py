# Manual smoke-test client for apps/local-api's /v1/audio/transcriptions
# (port 8000) — start that server first (`cd apps/local-api && python main.py`),
# then: python tools/legacy/test_transcribe.py path/to/audio.wav
import sys
import requests

audio_file = sys.argv[1] if len(sys.argv) > 1 else "test.wav"

with open(audio_file, "rb") as f:
    r = requests.post(
        "http://localhost:8000/v1/audio/transcriptions",
        files={"file": (audio_file, f, "audio/wav")},
        data={"model": "whisper-1", "language": "vi"},
    )

print(r.status_code)
print(r.json())
