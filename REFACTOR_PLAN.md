# Kế hoạch ổn định và refactor Mitsuka

## 1. Mục tiêu

Đợt refactor này tập trung vào phần Mitsuka tự tích hợp, bao gồm cả cấu trúc thư mục, không cố dọn toàn bộ mã nguồn upstream AIRI trong một lần.

Kết quả cần đạt:

- Luồng chính `microphone -> emotion/STT -> chat stream -> TTS -> avatar emotion` có thể kiểm thử độc lập.
- Lỗi có thể truy từ frontend sang FastAPI bằng `request_id` và `turn_id`.
- Startup, readiness, shutdown và background task có trạng thái rõ ràng.
- Typecheck, lint và test tối thiểu chạy tự động trước mỗi lần merge.
- Tách ranh giới giữa mã Mitsuka và upstream AIRI để việc cập nhật upstream ít xung đột hơn.
- Cấu trúc thư mục thể hiện đúng boundary: source code, test, model artifact, runtime data, tool và upstream integration không nằm lẫn nhau.

## 2. Phạm vi

### Ưu tiên refactor

- `VAD/main.py`
- `VAD/api/`
- `VAD/service/`
- `VAD/brain/`
- `VAD/schemas/`
- `VAD/tests/`
- `airi/packages/stage-ui/src/composables/local-conversation.ts`
- Phần local mode trong `airi/apps/stage-web/src/pages/index.vue`
- Cấu hình provider/voice Mitsuka đang sửa trong `airi/packages/stage-ui/src/stores/providers.ts`

### Chưa làm trong đợt này

- Refactor toàn bộ 4.000+ file upstream AIRI.
- Thay pnpm/Turborepo bằng Nx.
- Đổi model ML hoặc tinh chỉnh chất lượng model.
- Di chuyển/xóa model binary khỏi Git ngay trong PR sửa runtime. Việc quản lý artifact model là một phase riêng.
- Sửa hàng loạt `console.*` không thuộc luồng local conversation.

## 3. Baseline đã xác nhận ngày 2026-08-04

### Typecheck

`pnpm -F @proj-airi/stage-web typecheck` đang fail:

- `apps/stage-web/src/pages/index.vue`: dùng `.value` sai trên emotion store.
- `packages/stage-ui/src/composables/local-conversation.ts`: biến `language` không được dùng.
- `packages/stage-ui/src/components/scenarios/dialogs/audio-input/hearing-config.vue`: import `toRef` thừa.

`pnpm -F @proj-airi/stage-ui typecheck` cũng fail ở hai lỗi cuối.

### Lint trên các file thuộc luồng local

Hiện có 15 error và 1 warning, gồm:

- `console.log` không được phép theo ESLint config.
- `emotionStore` được dùng trước khi khai báo.
- SSE loop dùng label/break label bị lint cấm.
- Nhiều statement trên một dòng.
- Import thừa/sai thứ tự.
- Hai string concatenation trong `providers.ts`.

### Python và test

- 49 file Python parse cú pháp thành công.
- Có ba `SyntaxWarning` về invalid escape sequence trong script; cần xác định và sửa riêng.
- `VAD/tests/test_apis.py` hiện là smoke script gọi server thật tại port 8000, không cô lập dependency và không phù hợp làm regression suite.
- Root repo chưa có CI riêng cho phần Mitsuka.

### Khả năng quan sát lỗi

- Python có `logging`, nhưng log chủ yếu là chuỗi tự do và không có correlation ID.
- Một số exception chỉ log message, không có stack trace (`logger.warning(..., exc)` thay vì `logger.exception(...)`).
- Một số HTTP 500 trả thẳng `str(exception)` cho client.
- Frontend có nhiều `console.*`; riêng local pipeline có nhánh `.catch(() => null)` và `catch { return }`, làm mất nguyên nhân lỗi.
- `/health` luôn trả `ok`, chưa phản ánh model/LLM/vector store đã sẵn sàng hay chưa.

### Rủi ro kiến trúc thấy ngay

- Model VAD và audio emotion được tạo ở import time trong `VAD/main.py`; import app có thể chậm hoặc crash trước khi logging/lifespan sẵn sàng.
- `active_tts_id` là mutable state toàn app, tăng ở cả middleware và route TTS; semantics interrupt chưa nằm trong một abstraction rõ ràng.
- `ThreadPoolExecutor` trong emotion route là global và chưa có shutdown lifecycle.
- SSE gửi nhiều payload shape (`content`, `emotion/state`, `error`, `[DONE]`) nhưng client ép mọi JSON event thành `{ content: string }`. Event emotion có thể làm nối `undefined` vào response.
- Client bỏ qua body lỗi từ `/emotion-vad`, chat và TTS nên chỉ còn status code hoặc im lặng.
- TTS fetch chạy song song nhưng lỗi bị đổi thành `null`; UI không phân biệt synth fail, playback fail, abort hay network fail.
- Background memory task được fire-and-forget ở hai nơi với callback lặp lại; chưa quản lý task lúc shutdown.
- Cấu hình và tài liệu có dấu hiệu lệch nhau về ngôn ngữ, range V/A/D, model/voice ID và persistence của Qdrant.

## 4. Nguyên tắc thực hiện

1. Stabilize trước, refactor sau: khóa lỗi hiện tại bằng test rồi mới chuyển module.
2. Mỗi PR chỉ thay đổi một boundary và phải giữ luồng chạy được.
3. Entry point chỉ làm composition; policy nằm trong service/domain module.
4. Không log audio, prompt đầy đủ, response đầy đủ, token, secret hoặc thông tin nhạy cảm.
5. Lỗi trả cho client dùng mã ổn định; chi tiết và stack trace chỉ nằm trong server log.
6. Mọi background task phải được theo dõi, log khi fail và được drain/cancel khi shutdown.
7. Không sửa mã upstream không liên quan chỉ để làm đẹp.

## 5. Kế hoạch theo phase

### Phase 0 — Chụp baseline và dựng safety net

Mục tiêu: biết chính xác lỗi nào có trước refactor.

Công việc:

- Ghi lại lệnh chuẩn cho Python và AIRI trong README phát triển.
- Thêm test runner Python thực sự (`pytest`) và test FastAPI bằng `httpx`/ASGI transport.
- Override/mock model, Ollama, Qdrant, Piper và Whisper; unit test không được tải model thật.
- Thêm test cho:
  - app startup/shutdown;
  - `/health` và readiness;
  - `/emotion-vad` success/decode fail/branch fail;
  - chat thường và SSE success/RAG fail/LLM fail/client disconnect;
  - TTS success/unknown voice/empty input/interrupt;
  - background task success/fail.
- Thêm Vitest cho parser SSE, sentence buffer, abort/barge-in và TTS queue.
- Tạo CI root chạy Python test + typecheck/lint/test theo filter cho `stage-ui` và `stage-web`.

Exit criteria:

- Baseline test tái hiện được các lỗi SSE/error swallowing hiện tại.
- Test không cần server, Ollama, Qdrant hoặc model thật để chạy.
- CI có kết quả pass/fail trong một lệnh rõ ràng.

### Phase 1 — Sửa blocker hiện tại, không đổi kiến trúc lớn

Mục tiêu: đưa nhánh hiện tại về trạng thái build được.

Công việc:

- Sửa ba lỗi typecheck đã xác nhận.
- Sửa lint trong các file local integration.
- Sửa parser SSE để phân biệt event `delta`, `emotion`, `error`, `done`.
- Không dùng non-null assertion cho `response.body`; xử lý response không có stream.
- Parse error response thành một error type thống nhất thay vì nuốt lỗi.
- Gắn giá trị vào `error` ref và hiển thị trạng thái lỗi/retry trong UI.
- Đảm bảo abort không được báo như lỗi hệ thống.
- Thêm test regression trước mỗi bug fix.

Exit criteria:

- `stage-ui` và `stage-web` typecheck pass.
- Lint targeted pass.
- Không còn trường hợp emotion SSE làm hỏng text response.
- Người dùng nhìn thấy lỗi có thể hành động được khi local backend không chạy.

### Phase 2 — Chuẩn hóa protocol giữa frontend và FastAPI

Mục tiêu: bỏ contract ngầm và parsing tùy tiện.

Công việc:

- Định nghĩa schema cho request/response của emotion, chat, TTS và health.
- Định nghĩa SSE envelope có version và discriminated event type, ví dụ:

```json
{"type":"delta","turn_id":"...","content":"xin chào"}
{"type":"emotion","turn_id":"...","emotion":"happy","state":{"valence":0.4,"arousal":0.2,"dominance":0.1}}
{"type":"error","turn_id":"...","error":{"code":"LLM_UNAVAILABLE","message":"...","retryable":true}}
{"type":"done","turn_id":"..."}
```

- Tách SSE decoder/parser thành module thuần, không nằm trong Vue composable.
- Thống nhất tên field V/A/D; chỉ convert tại boundary avatar nếu avatar cần `{v,a,d}`.
- Thống nhất ngôn ngữ từ UI xuống Whisper; bỏ option không dùng.
- Xác thực voice/model ID tại server và trả error code ổn định.
- Viết contract tests hai phía cho các payload fixture.

Exit criteria:

- Không còn cast JSON mù ở client.
- Mọi event có `type`, `turn_id` và schema test.
- Backend thay đổi payload sai sẽ làm test fail ngay.

### Phase 3 — Refactor FastAPI composition và lifecycle

Mục tiêu: app import nhanh, dependency rõ và shutdown sạch.

Công việc:

- Đưa `create_app(settings, dependencies)` thành app factory.
- Di chuyển model initialization khỏi import time vào lifespan/service container.
- Tách cấu hình runtime, model path, thread count, CORS, URL backend và timeout vào settings typed.
- Tạo một application container/dataclass chứa VAD, audio emotion, Whisper, TTS, LLM, memory, vector và RAG.
- Thay dependency override toàn cục bằng dependency getter đọc container từ `app.state` hoặc typed lifespan state.
- Bọc executor bằng lifecycle owner và shutdown rõ ràng.
- Dùng task registry cho fire-and-forget task; drain/cancel có timeout khi shutdown.
- Tách TTS interruption thành coordinator có API `begin_turn`, `is_cancelled`, `cancel_active`.
- Đổi health thành:
  - `/health/live`: process còn sống;
  - `/health/ready`: dependency bắt buộc đã sẵn sàng;
  - optional dependency có status degraded, không giả `ok`.

Exit criteria:

- Import app không tải model.
- Startup fail chỉ rõ dependency nào fail.
- Shutdown không để executor/task/DB client treo.
- Test có thể inject fake service mà không patch global.

### Phase 4 — Tách orchestration của một conversation turn

Mục tiêu: route và Vue page không còn ôm toàn bộ workflow.

Công việc backend:

- Tạo application service cho một chat turn, sở hữu RAG, generation, emotion và schedule memory.
- Dùng cùng một policy cho chat thường và chat stream; tránh hai implementation lệch nhau.
- Chuẩn hóa fallback: RAG optional có thể degraded, LLM fail là terminal, emotion fail có fallback được mô tả rõ.
- Tách lưu conversation, vector indexing và fact extraction thành các job có tên/kết quả riêng.

Công việc frontend:

- Tách `useLocalConversation` thành các boundary có trách nhiệm rõ:
  - API client typed;
  - SSE parser;
  - turn state machine;
  - TTS playback queue.
- Page `index.vue` chỉ wiring VAD, store và hiển thị trạng thái.
- Biểu diễn state bằng transition hợp lệ thay vì gán string tự do.
- Chống race giữa turn cũ và turn mới bằng `turn_id`/generation token.
- Cleanup audio URL và abort listener đúng một lần.

Exit criteria:

- Route/page mỏng và test tập trung vào public behavior.
- Hai lượt nói liên tiếp không trộn transcript, SSE, emotion hoặc audio.
- Barge-in có test deterministic.

### Phase 5 — Logging và observability

Mục tiêu: từ một lỗi UI tìm được đúng request, stage và dependency bị lỗi.

Công việc backend:

- Cấu hình logging trước startup; hỗ trợ console dễ đọc ở dev và JSON ở production.
- Middleware nhận hoặc sinh `request_id`; conversation sinh `turn_id`; trả lại qua header/SSE.
- Dùng field chuẩn:
  - `timestamp`, `level`, `service`, `environment`;
  - `request_id`, `turn_id`, `route`, `method`, `status_code`;
  - `operation`, `component`, `duration_ms`, `outcome`;
  - `error_code`, `error_type`, `retryable`.
- Log start/end cho operation quan trọng, nhưng chỉ log INFO ở mốc có giá trị; chi tiết branch dùng DEBUG.
- Dùng `logger.exception` ở unexpected failure để giữ traceback.
- Redact/không ghi raw audio, full prompt/response, Authorization, token và model secret.
- Thêm timing cho decode, resample, Whisper, text VAD, audio VAD, RAG, first token, total LLM, TTS và background jobs.
- Thêm exception handler chung trả `{error: {code, message, request_id}}`.
- Có log rotation/retention khi ghi file; tránh file log tăng vô hạn.

Công việc frontend:

- Tạo logger adapter nhỏ cho local pipeline thay vì gọi `console.*` rải rác.
- Mỗi log có `component`, `operation`, `turn_id`, `duration_ms`, `outcome`.
- Dev cho phép debug log; production mặc định warn/error.
- Hiển thị `request_id` trong thông báo lỗi để tra log server.

Ví dụ log backend:

```json
{"level":"INFO","service":"mitsuka-api","operation":"emotion_vad","request_id":"req_...","turn_id":"turn_...","duration_ms":842,"outcome":"success"}
{"level":"ERROR","service":"mitsuka-api","operation":"whisper.transcribe","request_id":"req_...","turn_id":"turn_...","error_code":"STT_FAILED","error_type":"RuntimeError","retryable":true}
```

Exit criteria:

- Một turn có thể grep bằng `turn_id` từ browser tới background memory job.
- Unexpected exception có traceback ở server nhưng không lộ nội bộ cho client.
- Có test cho correlation ID, redaction và error mapping.

### Phase 6 — Refactor cấu trúc thư mục

Mục tiêu: nhìn vào đường dẫn là biết file thuộc application, domain, infrastructure, test, artifact hay runtime data; không còn tình trạng model, virtual environment, database và source code nằm lẫn nhau.

Các vấn đề hiện tại cần xử lý:

- Root chứa lẫn entry point Python, test script và nhiều model/voice ONNX.
- `VAD/` vừa là tên feature, tên service, vừa chứa toàn bộ backend chat/RAG/TTS/STT nên không phản ánh đúng trách nhiệm.
- Có đồng thời `VAD/.venv/` và `VAD/venv/` trong working directory.
- Có checkpoint VAD ở cả `VAD/vad_bert_final.pt` và `VAD/model/vad_bert_final.pt`.
- `model/`, `models/` và `ModelPreVoice/` có tên gần giống nhưng chứa các loại artifact khác nhau.
- `VAD/data/brain.db` là runtime state nằm cạnh source code.
- `service/` số ít chứa adapter ML/IO, còn `brain/*_service.py` chứa cả domain orchestration và infrastructure.
- `scripts/test_*.py` và `tests/test_apis.py` không có boundary rõ giữa dev tool, unit test, integration test và smoke test.
- Custom integration đang cắm trực tiếp vào các file lõi của AIRI, làm khó theo dõi diff và cập nhật upstream.

#### Cấu trúc đích đề xuất

```text
Mitsuka/
├─ apps/
│  └─ local-api/
│     ├─ pyproject.toml
│     ├─ src/
│     │  └─ mitsuka/
│     │     ├─ main.py                 # ASGI entry point rất mỏng
│     │     ├─ api/
│     │     │  ├─ routers/             # chat, emotion, speech, transcription, health
│     │     │  ├─ dependencies.py
│     │     │  ├─ middleware.py
│     │     │  └─ errors.py
│     │     ├─ application/
│     │     │  ├─ conversation.py      # orchestration của một turn
│     │     │  ├─ speech_pipeline.py
│     │     │  └─ background_tasks.py
│     │     ├─ domain/
│     │     │  ├─ conversation/
│     │     │  ├─ emotion/
│     │     │  └─ memory/
│     │     ├─ infrastructure/
│     │     │  ├─ llm/
│     │     │  ├─ persistence/
│     │     │  ├─ vector_store/
│     │     │  ├─ stt/
│     │     │  ├─ tts/
│     │     │  └─ ml_models/
│     │     └─ core/
│     │        ├─ config.py
│     │        ├─ logging.py
│     │        └─ lifecycle.py
│     └─ tests/
│        ├─ unit/
│        ├─ integration/
│        ├─ contract/
│        └─ smoke/
├─ airi/                                # upstream monorepo, giữ nguyên root của AIRI
│  └─ packages/
│     └─ mitsuka-local-runtime/         # phần custom có thể cô lập khỏi stage-ui
│        └─ src/
│           ├─ api-client/
│           ├─ protocol/
│           ├─ conversation/
│           ├─ playback/
│           └─ logging/
├─ assets/
│  └─ models/
│     ├─ vad/
│     ├─ speech-to-text/
│     ├─ audio-emotion/
│     └─ voices/
├─ var/                                 # luôn gitignore
│  ├─ data/
│  ├─ logs/
│  ├─ cache/
│  └─ tmp/
├─ tools/
│  ├─ model-inspection/
│  ├─ data-migrations/
│  └─ smoke/
├─ docs/
│  ├─ architecture/
│  ├─ operations/
│  └─ development/
├─ .env.example
├─ README.md
└─ CONTEXT.md
```

Tên package/thư mục cuối cùng có thể rút gọn, nhưng phải giữ các boundary trong cây trên. Không tạo thư mục chỉ để chứa một file nếu nó không đại diện cho một trách nhiệm ổn định.

#### Mapping từ cấu trúc hiện tại

| Hiện tại | Đích | Ghi chú |
|---|---|---|
| `VAD/main.py` | `apps/local-api/src/mitsuka/main.py` | Chỉ còn app factory/composition |
| `VAD/api/*` | `.../mitsuka/api/routers/*` | Route không chứa business workflow lớn |
| `VAD/schemas/*` | `domain/*` hoặc `api/*` | Domain model và transport schema tách riêng |
| `VAD/brain/nodes/*` | `application/` hoặc `domain/` | Xếp theo policy sở hữu, không theo thứ tự chạy |
| `VAD/brain/llm_service.py` | `infrastructure/llm/` | Adapter external runtime |
| `VAD/brain/memory_service.py` | `infrastructure/persistence/` | SQLite implementation |
| `VAD/brain/vector_service.py` | `infrastructure/vector_store/` | Qdrant implementation |
| `VAD/service/whisper_service.py` | `infrastructure/stt/` | STT adapter |
| `VAD/service/tts_service.py` | `infrastructure/tts/` | Piper adapter |
| `VAD/model/`, `VAD/models/`, `ModelPreVoice/` | `assets/models/*` | Chỉ artifact; code loader ở infrastructure |
| `VAD/data/brain.db` | `var/data/brain.db` | Runtime data, không commit |
| `VAD/scripts/backfill_vad.py` | `tools/data-migrations/` | Migration có version/hướng dẫn |
| `VAD/scripts/inspect_*.py` | `tools/model-inspection/` | Dev tool, không import từ production |
| `VAD/scripts/test_*.py` | `tests/smoke/` hoặc `tools/smoke/` | Phân loại lại theo assertion/runner |
| `test_transcribe.py` | `tools/smoke/` | Không để test script ở root |
| `whisper_server.py` | hợp nhất vào local API hoặc `legacy/` tạm thời | Chỉ giữ một production entry point |
| `local-conversation.ts` | `airi/packages/mitsuka-local-runtime/src/conversation/` | Cô lập custom runtime khỏi `stage-ui` |

#### Thứ tự migration thư mục

1. Thêm test và khóa import/public behavior trước khi move.
2. Tạo `pyproject.toml` và `src` layout, xác nhận package import được ở dev/test/production.
3. Di chuyển transport schema và pure domain code trước.
4. Di chuyển infrastructure adapter từng nhóm: persistence, vector, LLM, STT, TTS, ML.
5. Di chuyển application orchestration và cuối cùng mới làm mỏng `main.py`/routers.
6. Tạo package `mitsuka-local-runtime`, chuyển parser/state machine/API client vào đó; page AIRI chỉ wiring.
7. Di chuyển tool/test, sau đó mới di chuyển artifact model và runtime data.
8. Khi không còn import cũ, xóa thư mục rỗng và cập nhật README, CI, `.gitignore`, model path.

Quy tắc migration:

- Mỗi PR move một boundary bằng `git mv`, hạn chế trộn rename hàng loạt với thay đổi hành vi.
- Không duy trì đồng thời module cũ và mới bằng compatibility shim dài hạn.
- Sau mỗi nhóm move phải chạy unit test, import test, typecheck và smoke command tương ứng.
- Mọi path model/data đi qua typed settings; không hardcode đường dẫn tương đối theo current working directory.
- `var/`, virtual environment, cache, `__pycache__` và log phải nằm trong `.gitignore`.
- Model artifact có manifest gồm logical ID, version, checksum và expected runtime; code không phụ thuộc trực tiếp vào tên file ngẫu nhiên.
- Chỉ tạo package frontend custom sau khi xác nhận public API tối thiểu cần export; không biến nó thành nơi gom mọi helper.

Exit criteria:

- Root chỉ còn entry/config/docs cấp repo, không còn test script, database hoặc model rải rác.
- Chỉ có một Python environment bên ngoài source tree hoặc một `.venv` chuẩn đã gitignore.
- Không còn `model`/`models` mơ hồ; artifact và loader code tách nhau.
- Production code không import từ `tools/` hoặc `tests/`.
- Custom Mitsuka frontend có boundary riêng và AIRI page/store chỉ làm composition.
- Tất cả command dev/test/run hoạt động độc lập với current working directory.

### Phase 7 — Tài liệu và dependency hygiene

Mục tiêu: giảm nhầm lẫn vận hành và giảm xung đột với upstream.

Công việc:

- Sửa README/layout đang lệch thực tế và chuẩn hóa UTF-8.
- Ghi rõ một entry point chính; đánh dấu `whisper_server.py` standalone là legacy/dev hoặc loại khỏi đường chạy chính sau khi xác nhận.
- Tách requirements runtime/dev; khóa phiên bản theo chiến lược có thể tái tạo.
- Thêm `.env.example`, không chứa secret.
- Quy định nơi chứa model và cách tải/check checksum.
- Đánh giá Git LFS hoặc release artifact cho ONNX/checkpoint lớn trong một migration riêng.
- Ghi upstream commit/version và cô lập custom code vào package/module Mitsuka khi khả thi.
- Loại script test/ad-hoc khỏi production module hoặc chuyển vào `tools/` với hướng dẫn rõ.

Exit criteria:

- Người mới có thể setup, chạy, test và tìm log theo README.
- Runtime dependency không lẫn hoàn toàn với dev/tooling dependency.
- Ranh giới custom/upstream được tài liệu hóa.

## 6. Test matrix tối thiểu

| Boundary | Unit | Integration giả lập | Smoke thật |
|---|---:|---:|---:|
| VAD text | Có | FastAPI dependency override | Model thật, opt-in |
| Emotion audio | Fuse/resample/error policy | Fake audio + fake models | Model thật, opt-in |
| Whisper | Adapter/error mapping | Fake transcriber | Model thật, opt-in |
| Chat/RAG | Policy/state | Fake LLM/Qdrant/SQLite temp | Ollama/Qdrant, opt-in |
| SSE | Encoder + parser fixtures | ASGI stream + frontend parser | Browser local stack |
| TTS | Voice resolution/interrupt | Fake synthesizer | Piper voice thật, opt-in |
| Frontend turn | State machine/abort | Mock fetch + fake Audio | Browser + microphone |
| Logging | Context/redaction/error code | Request middleware | Tra log end-to-end |

## 7. Thứ tự PR đề xuất

1. `test: establish Mitsuka regression baseline`
2. `fix(stage): restore local conversation typecheck and lint`
3. `fix(stage): parse typed chat stream events`
4. `refactor(api): add app factory and managed lifecycle`
5. `refactor(chat): unify buffered and streaming turn orchestration`
6. `feat(observability): add correlated structured logging`
7. `refactor(layout): introduce local-api src layout and move backend boundaries`
8. `refactor(stage): extract Mitsuka local runtime package`
9. `refactor(assets): separate model artifacts and runtime data from source`
10. `ci: verify Mitsuka backend and AIRI integration`
11. `docs: align setup, runtime architecture, logging and model management`

## 8. Definition of done toàn đợt

- Không còn typecheck/lint error trong phạm vi Mitsuka.
- Test nhanh chạy offline; smoke test model thật là opt-in.
- Luồng local conversation có regression coverage cho success, abort và từng dependency fail.
- Frontend không còn nuốt lỗi trong luồng chính.
- Log có correlation ID, duration, outcome và error code; không lộ dữ liệu nhạy cảm.
- App có liveness/readiness thực chất và shutdown sạch.
- Source, model artifact, runtime data, test và tool nằm đúng boundary; root không còn file chạy rải rác.
- README và kiến trúc phản ánh đúng code đang chạy.
- Mỗi PR nhỏ, review được và có rollback độc lập.
