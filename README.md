# Mitsuka Project

![Architecture Diagram](file:///C:/Users/admin/.gemini/antigravity/brain/88b20f11-d775-4522-b2ed-e472b30d6bbf/architecture_diagram_1781968605207.png)

---

## 🎯 Overview
Mitsuka is a local speech‑to‑text and Voice Activity Detection (VAD) solution built with **FastAPI** and **faster‑whisper**. It provides an OpenAI‑compatible transcription endpoint and a lightweight VAD service that returns valence‑arousal‑dominance (VAD) scores for a given text.

---

## 🏗️ Architecture
The diagram above illustrates the main components:
- **Whisper Server** (`whisper_server.py`): FastAPI app exposing `/v1/audio/transcriptions` and `/health`. Uses `faster-whisper` models for Vietnamese/English transcription.
- **VAD Service** (`VAD/VAD_service.py`): FastAPI app exposing `/vad` that loads a PHOBERT‑based model to predict VAD scores.
- **Model files** (`*.onnx`, `*.safetensors`, `vad_phobert_final.pt`): Stored in the repository and loaded at runtime.
- Both services can be run independently and are discovered automatically by the front‑end applications (e.g., Tamagotchi UI).

---

## 📁 Directory Layout
```
Mitsuka/
├─ .github/                # CI/CD workflows
├─ ModelPreVoice/          # Whisper model weights & tokenizer
├─ VAD/                    # VAD service source & model
│   ├─ VAD_service.py      # FastAPI VAD API
│   ├─ model/…             # Model checkpoint & helpers
│   └─ requirements.txt    # VAD dependencies
├─ airi/                   # Main monorepo root (this README lives here)
│   ├─ apps/               # Front‑end applications
│   ├─ services/…          # Additional services
│   ├─ scripts/…           # Utility scripts
│   ├─ whisper_server.py   # FastAPI Whisper API
│   └─ test_transcribe.py  # Simple client test script
├─ en_US‑hfc_female-medium.onnx*  # Example Whisper model (English)
├─ vi_VN‑25hours_single-low.onnx* # Example Whisper model (Vietnamese)
└─ README.md               # **This file**
```
*(files truncated for brevity)*

---

## 📡 API Reference
### Whisper Transcription Server
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

### VAD Service
- **GET** `/health`
  Returns `{ "status": "ok" }`.
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
  *V, A, D* range from 0‑1.

---

## 🛠️ Setup & Installation
1. **Python ≥ 3.9** (recommended: 3.11).
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   ```
3. Install core dependencies:
   ```bash
   pip install -r requirements.txt   # root requirements (if present)
   # VAD specific deps
   pip install -r VAD/requirements.txt
   ```
4. Install Whisper dependencies (already in `requirements.txt` of the repo):
   ```bash
   pip install faster-whisper fastapi uvicorn python-multipart
   ```
5. **Model files** – the repository ships with pre‑downloaded ONNX models. Ensure the environment variables point to the correct model size if you wish to change it:
   ```bash
   export WHISPER_MODEL=small   # tiny / base / small / medium / large‑v3
   export WHISPER_DEVICE=cpu    # or cuda
   export WHISPER_COMPUTE=int8  # int8 / float16 / float32
   ```

---

## ▶️ Running the Services
### Whisper Server
```bash
python whisper_server.py
# defaults to http://0.0.0.0:9000
```
### VAD Service
```bash
python VAD/VAD_service.py
# defaults to http://0.0.0.0:8000
```
Both servers support hot‑reload when run with `uvicorn --reload`.

---

## 📌 Quick‑Start Test
Transcribe a local WAV file using the bundled test script:
```bash
python test_transcribe.py path/to/audio.wav
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
