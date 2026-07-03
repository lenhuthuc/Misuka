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
