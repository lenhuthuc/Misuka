# Mitsuka Project

## 🎯 Overview
Mitsuka is a local speech‑to‑text and Voice Activity Detection (VAD) solution built with **FastAPI** and **faster‑whisper**. It provides an OpenAI‑compatible transcription endpoint and a lightweight VAD service that returns valence‑arousal‑dominance (VAD) scores for a given text.

---

## 🙏 Credits
The `airi/` directory is based on [moeru-ai/airi](https://github.com/moeru-ai/airi), an open‑source AI companion / Tamagotchi framework — all credit for that codebase goes to its original authors.

**Everything from the Architecture section onward in this README — the `apps/local-api` service, the Whisper transcription server, the model wiring under `ModelPreVoice/`, and the integration work tying them into `airi/` — is my own work**, built on top of that base.

---

## 🏗️ Architecture
- **`apps/local-api`**: the single FastAPI app (port 8000, entry point `main.py`) — VAD/emotion
  analysis, Whisper transcription, Piper TTS, and the RAG/LLM "brain" all live here. See
  [CONTEXT.md](CONTEXT.md) for the full architecture (data flow, SSE envelope, logging).
- **`tools/legacy/whisper_server.py`**: a standalone, legacy Whisper-only server (port 9000,
  OpenAI-compatible `/v1/audio/transcriptions`) kept separate from `apps/local-api` for ad-hoc testing.
- **Model files**: Piper TTS voices live under `assets/models/voices/` (`*.onnx` + `*.onnx.json`,
  tracked in git); the VAD `.pt` checkpoint and downloaded Whisper models live inside
  `apps/local-api/` and are gitignored — see `.gitignore`.
- `airi/` (the Vue front-end monorepo) discovers `apps/local-api` at `http://localhost:8000` by default.

---

## 📁 Directory Layout
```
Mitsuka/
├─ .github/                # CI/CD workflows
├─ ModelPreVoice/          # Whisper model weights & tokenizer
├─ assets/
│  └─ models/
│     └─ voices/           # Piper TTS voices (*.onnx + *.onnx.json), tracked in git
├─ apps/
│  └─ local-api/           # The FastAPI service — see CONTEXT.md for the full breakdown
│     ├─ main.py           # App entry: create_app() factory
│     ├─ api/               # HTTP routes
│     ├─ brain/, application/, core/, service/, model/, schemas/
│     ├─ tests/             # pytest — offline unit/integration + opt-in smoke
│     └─ requirements.txt / requirements-dev.txt
├─ airi/                   # Upstream airi monorepo (front-end apps, Tamagotchi UI)
├─ tools/
│  └─ legacy/              # whisper_server.py, test_transcribe.py — not part of the production entry point
└─ README.md               # **This file**
```
*(files truncated for brevity — see [CONTEXT.md](CONTEXT.md) for the full `apps/local-api` tree)*

---

## 📡 API Reference
### `tools/legacy/whisper_server.py` (standalone, port 9000)
- **GET** `/health`
  ```json
  {"status": "ok", "model": "small"}
  ```
- **GET** `/v1/models`
  Returns a list with a single entry `whisper-1` (compatible with OpenAI client libraries).
- **POST** `/v1/audio/transcriptions`
  ```http
  POST /v1/audio/transcriptions
  Content‑Type: multipart/form-data
  
  file: <audio file>
  model: "whisper-1" (default)
  language: "vi" | "en" (optional)
  prompt: <string> (optional)
  response_format: "json" | "text"
  ```
  **Response (JSON)**
  ```json
  {
    "text": "Transcribed text…",
    "language": "vi",
    "duration": 12.34
  }
  ```

### `apps/local-api` (main FastAPI service, port 8000)
- **GET** `/health/live`, `/health/ready`, `/health` (alias of `/health/live`)
- **POST** `/vad`
  ```http
  POST /vad
  Content‑Type: application/json

  {"text": "I feel happy"}
  ```
  **Response**
  ```json
  {"v": 0.73, "a": 0.55, "d": 0.61}
  ```
  *V, A, D* each range **-1 to 1**.
- **POST** `/emotion-vad` — multipart audio + `language` field; fuses audio (wav2vec2) and text
  (Whisper → PhoBERT) V/A/D.
- **POST** `/v1/chat`, **POST** `/v1/chat/stream` (SSE), **POST** `/v1/chat/seed`
- **POST** `/v1/audio/speech`, **POST** `/v1/audio/speech/stream`, **GET** `/v1/audio/voices`
- **POST** `/v1/audio/transcriptions` (OpenAI-compatible)

See [CONTEXT.md](CONTEXT.md) for request/response shapes, the SSE event envelope, and error codes.

---

## 🛠️ Setup & Installation
1. **Python 3.10+**.
2. Create a virtual environment inside `apps/local-api` and install dependencies:
   ```bash
   cd apps/local-api
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt        # runtime deps
   pip install -r requirements-dev.txt    # + pytest, for running the test suite
   ```
3. **Model files** – `apps/local-api/model/vad_bert_final.pt` (VAD checkpoint) and Whisper models
   under `apps/local-api/models/` are gitignored; place them there (or point
   `VAD_MODEL_PATH`/`WHISPER_MODEL` env vars elsewhere) before running the service. Piper `.onnx`
   voices ship under `assets/models/voices/`. Override the defaults via environment variables — see
   `apps/local-api/brain/config.py` for the full list (`WHISPER_MODEL`, `WHISPER_DEVICE`,
   `WHISPER_COMPUTE`, `PIPER_MODELS_DIR`, `OLLAMA_BASE_URL`, `QDRANT_URL`, `LOG_LEVEL`, `LOG_JSON`, ...).

---

## ▶️ Running the Services
### Whisper Server (standalone, legacy)
```bash
python tools/legacy/whisper_server.py
# defaults to http://0.0.0.0:9000
```
### Main local API (VAD, emotion, chat/RAG, TTS, Whisper)
```bash
cd apps/local-api
python main.py
# defaults to http://0.0.0.0:8000
```
Both servers support hot‑reload when run with `uvicorn --reload`.

### Running the test suite
```bash
cd apps/local-api
python -m pytest -q
# Fully offline — no model, Ollama, or Qdrant needed; see tests/conftest.py.
```

---

## 📌 Quick‑Start Test
Transcribe a local WAV file using the bundled test script (`apps/local-api` must be running):
```bash
python tools/legacy/test_transcribe.py path/to/audio.wav
```
You should see a `200` status and the JSON transcription result.

---

## 🤝 Contributing
1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature/awesome‑thing
   ```
3. Ensure linting passes (`pnpm run lint` or the project's configured linter).
4. Open a Pull Request describing the change.

---

## 📄 License
This project is licensed under the **MIT License** – see `LICENSE` for details.

---

## 🏷️ Badges (optional)
Add badges such as build status, license, and version if you use CI pipelines.

---

*Generated on 2026‑06‑20.*
