# Bàn giao: tầng sinh hội thoại LangGraph cho Mitsuka

Trạng thái: **làm dở**. Tầng graph đã dựng xong và import chạy được, nhưng
**chưa cắm vào API và chưa từng chạy end-to-end lần nào.**

Mọi đường dẫn dưới đây tương đối từ `apps/local-api/`.

---

## 1. Bối cảnh: spec gốc va chạm với code đang chạy

Spec ban đầu được viết như thể repo chỉ có "một lời gọi model đơn lẻ". Thực tế
không phải vậy, và 5 điểm sau đã được phát hiện rồi giải quyết bằng cách hỏi
người dùng — **đừng tự ý đảo lại các quyết định này**:

1. **LangGraph đã từng có và bị gỡ.** `application/conversation_turn.py:1-10`
   ghi rõ `brain.graph.build_fast_graph` bị bỏ vì "LangGraph's node model
   doesn't stream token-by-token cleanly", dẫn tới hai implementation lệch nhau.
2. **"Cùng một SYSTEM cho cả hai backend" không thể đúng theo nghĩa đen.**
   Persona nằm trong fine-tune + `SYSTEM` của `assets/models/LLM/Modelfile`.
   `brain/nodes/generate.py:16-31` đã **đo được**: gửi system message ở index 0
   thì Ollama im lặng thay thế persona (model đổi tên thành "Bố", xưng "tôi").
3. **Spec cấm reasoning, repo cố tình có reasoning** (`brain/reasoning_policy.py`
   + `LLMService.reason()`, `REASONING_ENABLED=true`).
4. **`session_id` không tồn tại** — bảng `conversations` là transcript toàn cục.
5. **Node `guard` trùng** với `bm25_repetition.py`, `trailing_question.py`,
   `response_policy.py` đã có.

## 2. Quyết định đã chốt với người dùng

| Câu hỏi | Quyết định |
|---|---|
| Phạm vi | Thay **cả** `/v1/chat` và `/v1/chat/stream` |
| Persona | Giữ trong Modelfile; `prompts/system_prompt.txt` là bản sao **đúng từng byte**, có hàm kiểm tra drift |
| Reasoning | **Giữ cho local, tắt cho cloud** |
| Bộ nhớ | Thêm `session_id` vào `brain.db` (không dùng checkpointer riêng) |
| Guard bắt được mẫu | **Tùy mẫu**: bạo lực → câu an toàn ngay, không sinh lại; tự kết thúc → cắt câu đó giữ phần còn lại; lặp/rỗng/quá dài → sinh lại 1 lần |
| Guard vs bộ lọc cũ | **Gộp hết vào node guard** (gọi lại module cũ, không viết lại logic) |
| Few-shot | **Chỉ cho cloud** (`prompts/fewshot.json`) |
| Rolling summary | **Gọi model local tóm tắt**, chạy nền qua gate |

### Mặc định tôi tự chọn (người dùng chưa phản đối)
- `graph_history_turns = 12`; `memory_recent_limit` giữ **6** (knob khác, là ngân sách prefill).
- Log đo lường → `logs/turns.jsonl` (JSONL riêng, không trộn `mitsuka.log`).
- Tên model local = **`mitsuka-ft`**. Spec ghi `mitsuka-test` — **model đó không tồn tại**
  (`ollama list` chỉ có `mitsuka-ft:latest`, `qwen3-1.7b-rp`, `qwen3:1.7b`).
- Sẽ **sửa test có sẵn cho khớp**, **không viết test mới** (theo yêu cầu người dùng).

## 3. Môi trường

- Python **3.13.7**, venv tại `apps/local-api/.venv`.
- Đã cài: `langgraph 1.2.11`, `langchain-core 1.6.1`, `langchain-google-genai 4.4.0`,
  `langchain-ollama 1.1.0`, `langgraph-checkpoint-sqlite 3.1.1`.
  → **`langgraph-checkpoint-sqlite` chưa dùng đến** (state nằm trong `brain.db`), cân nhắc gỡ.
- **Chưa có `GEMINI_API_KEY`** → graph luôn rơi về local. Đây là điều kiện (c) trong spec, chạy đúng.

## 4. Đã tạo mới

```
prompts/system_prompt.txt          sinh từ Modelfile, drift check trả None
prompts/fewshot.json               5 cặp, chỉ gửi cho cloud
brain/graph/__init__.py            export MitsukaGraph, GraphDeps, GraphConfig
brain/graph/prompts.py             load + parse Modelfile SYSTEM + check_persona_drift
brain/graph/state.py               MitsukaState (TypedDict) + new_state()
brain/graph/deps.py                GraphConfig (mọi knob) + GraphDeps (services)
brain/graph/guard.py               ReplyGuard + các predicate thuần
brain/graph/metrics.py             TurnMetrics (JSONL) + build_record()
brain/graph/graph.py               build_graph() + MitsukaGraph(.ainvoke/.astream)
brain/graph/backends/base.py       ChatBackend ABC, Usage, Generation, GenerationOptions,
                                   BackendUnavailable / DailyQuotaExceeded
brain/graph/backends/local.py      LocalBackend (ChatOllama)
brain/graph/backends/cloud.py      CloudBackend (ChatGoogleGenerativeAI)
brain/graph/backends/__init__.py   CloudAvailability (circuit breaker) + BackendRegistry
brain/graph/nodes/load_context.py
brain/graph/nodes/route_backend.py select_backend(), after_cloud()
brain/graph/nodes/generate.py      node lớn nhất: prompt + reasoning + stream + fallback
brain/graph/nodes/guard.py         after_guard()
brain/graph/nodes/finalize.py
brain/graph/nodes/update_memory.py metrics + lên lịch tóm tắt
scripts/check_guard.py             battery cho guard, chạy tay, KHÔNG phải pytest
```

## 5. Đã sửa file cũ

- **`brain/memory_service.py`** — thêm cột `session_id` (mặc định `'default'`),
  index `idx_conversations_session`, bảng `session_state`
  (`rolling_summary`, `summarized_through_id` = watermark). Thêm
  `get_session_state`, `set_rolling_summary`, `get_messages_between`,
  `count_messages`; `get_recent`/`save_message` nhận `session_id`.
  ⚠️ `CREATE INDEX` **phải** nằm trong `_migrate` chứ không phải `_DDL` — trên DB cũ
  chưa có cột thì `_DDL` sẽ ném "no such column".
- **`application/conversation_turn.py`** — `prepare_turn` nhận thêm
  `session_id` và `rolling_summary`; thêm `_SUMMARY_HEADER`, nối summary vào
  cuối khối `context` (coi như untrusted, cùng header cảnh báo với RAG).
- **`brain/config.py`** — thêm nhóm Cloud / Conversation graph / Reply guard /
  Turn metrics. Sửa comment cũ đã sai ("store is currently global").
- **`core/container.py`** — thêm `build_conversation_graph()` (hàm module-level
  để script demo dùng lại được), field `graph: MitsukaGraph`, hoisted `tasks`.

## 6. Kiến trúc chốt lại — đọc kỹ phần này

### Vì sao lần này streaming không drift
`langgraph.config.get_stream_writer()` **no-op dưới `ainvoke`** và phát dưới
`astream(stream_mode=["custom","values"])`. Đã kiểm chứng bằng script riêng.
Nhờ đó **một** node `generate` phục vụ cả hai endpoint: nó luôn stream từ
backend, luôn đẩy từng câu qua writer; buffered chỉ đơn giản là không đọc kênh
đó. Đây chính là thứ lần trước không có.

### Hình graph
```
START -> load_context -> route_backend
route_backend  -[select_backend]-> generate_cloud | generate_local
generate_cloud -[after_cloud]->    generate_local (fallback) | guard
generate_local ->                  guard
guard          -[after_guard]->    generate_* (sinh lại, budget 1) | finalize
finalize -> update_memory -> END
```
Mọi quyết định backend là **conditional edge**, không phải `if` trong node.

### Guard
`ReplyGuard` dùng chung cho cả hai đường: `check()` cho nguyên câu trả lời,
`feed()/flush()` cho từng câu khi stream. Cờ: `violence`, `self_ending`,
`repetition`, `empty`, `too_long`, `trailing_question`.

**Giới hạn thành thật:** stream đã phát ra thì không thu lại được. Câu bạo lực
không bao giờ được phát (bắt trước khi release), nhưng các câu lành trước đó thì
đã ra rồi — khi đó `guard` node phát `SAFE_REPLY` và dừng.

**Bẫy đã sửa:** khi sinh lại phải tạo `ReplyGuard` **mới**; tái dùng cái cũ sẽ
mang theo câu mà `TrailingQuestionSuppressor` đang giữ từ lần sinh trước. Cờ cũ
được chuyển sang riêng qua `prior_flags`.

### Routing
`CloudAvailability` là circuit breaker: lỗi kết nối park 60s, hết quota ngày
park tới nửa đêm giờ Pacific (quota free-tier của Google reset theo mốc đó, không
phải 24h sau). Phân biệt 429/ngày với 429/phút bằng cách đọc `QuotaFailure`
trong `details` (`PerDay`), không đoán theo status code.

## 7. Đã kiểm chứng (và chưa)

✅ Đã chạy và xác nhận:
- `check_persona_drift()` trả `None` — `system_prompt.txt` khớp Modelfile tuyệt đối.
- `ChatOllama(reasoning=False, keep_alive="30m")` sinh payload
  `{"think": false, "keep_alive": "30m", "options": {...}}` — giữ nguyên hành vi
  `LLMService` cũ (mất `keep_alive` là mất 4.2s/lượt mà repo đã đo).
- `get_stream_writer()` no-op dưới `ainvoke`, phát dưới `astream`.
- `scripts/check_guard.py` → **0 failures** (17 ca bạo lực, 6 tự-kết-thúc,
  4 user-rời-đi, 6 verdict end-to-end).
- Migration chạy trên **bản sao** `brain.db` thật (1052 dòng): toàn bộ dòng cũ
  về `session_id='default'`, session mới cô lập đúng, rolling summary theo session.
- Toàn bộ package `brain.graph` import sạch.

❌ **Chưa hề chạy:**
- Graph chưa chạy end-to-end lần nào (chưa gọi Ollama thật qua graph).
- Chưa chạy pytest sau khi sửa.
- Chưa test cloud backend (không có API key).

## 8. Việc còn lại

1. **`api/chat.py`** — viết lại cả hai endpoint qua `container.graph`.
   Đang định làm thì dừng. Cần:
   - `ChatRequest` thêm `session_id: str = "default"`.
   - Buffered: `state = await container.graph.ainvoke(...)` trong
     `container.llm_gate.foreground()`, rồi `_emotion_state` / `_agent_vad` /
     `container.tasks.spawn(run_memory_tasks(...))` như cũ.
   - Stream: lặp `container.graph.astream(...)`, `("delta", text)` →
     `ChatStreamDeltaEvent`, cuối cùng `("state", state)` để lấy `flags`/`route`.
   - **Xóa** các helper đã chuyển vào graph: `_filter_repeated_response`,
     `_regenerate_without_repetition`, `_with_repetition_note`,
     `_with_optional_reasoning`, `_REPETITION_RETRY_NOTE`.
2. **`requirements.txt`** — thêm 4 gói langgraph/langchain (chưa thêm).
3. **`.env.example`** — tài liệu hóa `GEMINI_API_KEY`, `GEMINI_MODEL`,
   `CLOUD_ENABLED`, `GRAPH_HISTORY_TURNS`, `GUARD_*`, `TURN_METRICS_*`.
4. **`scripts/demo_graph.py`** — CLI gõ tay vài lượt (yêu cầu trong spec).
5. **README ngắn** — cách chạy, cách đổi model trong config, cách đọc
   `logs/turns.jsonl`.
6. **Sửa test đang hỏng** (do đổi interface):
   `tests/integration/test_chat.py`, `test_chat_stream.py`,
   `test_adaptive_reasoning.py`, `test_correlation_ids.py`, và
   `tests/conftest.py` (fixture container giờ cần field `graph`).
7. Chạy thật: `ollama` + `uvicorn`, gõ vài lượt, xem `logs/turns.jsonl`.

## 9. Rủi ro cần nói với người dùng

1. **Summariser nền gọi LLM.** `core/llm_priority.py:21-24` nói rõ
   `run_when_idle` và kênh interrupt đã bị gỡ, và *"nothing on the server issues
   background LLM calls any more; if something ever does, that gate has to come
   back with it"*. Summariser chính là "something" đó. Hiện dùng
   `wait_until_spoken()` + `tasks.spawn()` (giống hệt `brain/background.py`),
   nhưng **không có gì hủy nó nếu người dùng nói tiếp giữa chừng**. Giảm thiểu:
   chỉ chạy khi cửa sổ history tràn (~1 lần mỗi 12 lượt), cap `summary_max_tokens=160`.
   Nếu nghe thấy độ trễ → phải dựng lại gate abandon-on-activity, **không** đưa inline.
2. **Guard là heuristic regex tiếng Việt.** Đã tránh 4 false positive thật
   ("đi đá bóng luôn", "hại sức khỏe... bỏ đi", "xe đâm vào cột", "đập hộp")
   bằng cách tách động từ rõ nghĩa / động từ cần tân ngữ chỉ người. Sửa regex thì
   **chạy lại `scripts/check_guard.py`**.
3. **Prompt hai backend khác nhau về *cách giao*, giống nhau về *chữ*.** Local
   không nhận system message (Modelfile lo); cloud nhận persona + few-shot ở đầu
   và turn-note gộp vào message user. Lý do đã ghi trong docstring
   `CloudBackend.prepare`.
