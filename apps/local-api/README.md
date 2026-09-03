# Mitsuka local API

The chat endpoints run one LangGraph conversation pipeline with per-session
history, cloud-to-local fallback, guarded sentence streaming, rolling summaries,
and JSONL turn metrics. Both `POST /v1/chat` and `POST /v1/chat/stream` accept:

```json
{"query": "xin chao", "session_id": "demo"}
```

`session_id` defaults to `default`. Use a stable ID per conversation so its
transcript and rolling summary remain isolated from other chats.

## Run

From this directory:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8010
```

Ollama must be running with the local model named by `OLLAMA_MODEL` (default
`mitsuka-ft`). To use Gemini first and fall back to Ollama, set
`GEMINI_API_KEY` (the default is `gemini-3.5-flash-lite`); set
`CLOUD_ENABLED=false` to force local generation. The full
set of model, graph, guard, and metrics switches is documented in `.env.example`.

For a direct graph smoke test without starting FastAPI:

```powershell
.venv\Scripts\python.exe scripts\demo_graph.py --session-id demo
```

Type `/quit` to exit. Each response prints the selected route and guard flags.

## Metrics

Completed turns are appended to `logs/turns.jsonl` by default. Each line is an
independent JSON object containing the session and turn IDs, route/model, token
counts, latency, fallback reason, guard flags, and whether regeneration ran.
Change `TURN_METRICS_PATH` or disable it with `TURN_METRICS_ENABLED=false`.
