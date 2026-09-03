# Vision layer

Answers questions about images the user sends, in Vietnamese, on a machine with
no CUDA.

Everything an image becomes here is **text**. Object counts, OCR lines,
label/value pairs and a summary sentence — that is the whole interface to the
rest of the app, because the NLU downstream is PhoBERT and no image vector is
ever projected into its embedding space.

## How it decides

Two paths, split by cost.

**Fast path — runs the moment an image arrives.** YOLO11n, RapidOCR and CLIP
run concurrently in a thread pool. The target was under a second; measured, it
is 1.4–2.9s and OCR is the reason — see [Measured on this machine](#measured-on-this-machine). What comes out is
a record: detections above 0.5, OCR lines above 0.6 in reading order,
label/value pairs recovered from the OCR *layout*, a CLIP embedding, and a
Vietnamese `fast_summary`. No captioning, no background job, nothing queued.

**Slow path — a cloud VLM, called only when a question needs it.** It never
runs at ingest. It runs when the router has established that neither the
detections nor the OCR text can answer, and its answer is cached as text so the
same question reworded costs nothing.

A question is routed in three steps:

1. **Which image.** A demonstrative ("ảnh *này*", "vừa gửi") means the newest.
   One image in the session means that one. Otherwise the question is encoded
   with the CLIP text tower and matched against each image's embedding; below
   the similarity floor the layer asks which image rather than guessing.
2. **What kind of question.** `COUNT_EXIST`, `READ_TEXT`, `DESCRIBE`, `REASON`
   or `UNKNOWN`, from rules over the accent-folded question.
3. **Who answers.** Detections, then OCR, then the VLM cache, then the VLM.

Every answer reports where it came from:

```python
result = await pipeline.answer("Máy tôi có bao nhiêu nhân?", session_id="s1")
result.source      # "ocr"
result.text        # "Cores: 12"
result.latency_ms  # 15.0
```

`source` is one of `detections`, `ocr`, `vlm_cache`, `vlm`, `clarify`,
`expired`. It is the number to watch: the fast path is only worth having if the
share of turns falling through to `vlm` stays low.

## What is kept, and for how long

The uploaded image is **never written to disk and never retained**. It is
decoded, read by the three models, and dropped. What survives is:

- the extracted text — for the life of the session, it is what the fast path
  answers from and it is the size of a chat message;
- one JPEG thumbnail, long edge 768px, quality 80, **in RAM for 15 minutes**,
  which exists only so a lazy VLM call has something to send.

Past the TTL the thumbnail is freed and a question that needs the VLM answers
`source="expired"` — "Ảnh đã quá lâu, bạn gửi lại giúp mình…". Answers already
paid for survive expiry, because they are text.

The buffer is a plain dict keyed by session, capped at 8 images each,
oldest evicted first. `ImageBuffer.persist()` is an empty hook where a durable
store would attach.

## Setup

```bash
cd apps/local-api
pip install -r requirements.txt          # includes rapidocr-onnxruntime, opencv-python
python scripts/export_models.py          # downloads + exports, ~1.5 GB, run once
python scripts/bench.py --markdown       # measure this machine
```

`export_models.py` additionally needs `ultralytics`, `torch` and `onnxscript`,
which are **not** in `requirements.txt` — it is a one-time developer step, and
the service itself loads the exported graphs directly. It writes to
`assets/models/vision/` (gitignored: reproducible artefacts, not source):

| file | what | size |
|---|---|---|
| `yolo11n.onnx` | detector, static 640×640, opset 17 | 10 MB |
| `clip_image.onnx` (+ `.data`) | CLIP ViT-B/32 vision tower + projection | 351 MB |
| `clip_text.onnx` (+ `.data`) | multilingual CLIP text tower + projection | 540 MB |
| `clip_tokenizer/` | tokenizer for the text tower | 5 MB |
| `yolo11n_openvino_int8/` | optional INT8 IR, `--runtime openvino` | 6 MB |

Then:

```python
from vision import VisionPipeline

pipeline = VisionPipeline.build_default()
await pipeline.warmup()          # at startup: the towers load lazily and the
                                 # text one is 540MB, so the first question
                                 # that needs it otherwise waits 12s
await pipeline.ingest(image_bytes, session_id="s1")
result = await pipeline.answer("Trong ảnh có mấy người?", session_id="s1")
```

`build_default()` reads everything from config. To swap a model — or to drive
the router with scripted output, as the tests do — construct `VisionPipeline`
directly with your own `detector`, `ocr`, `embedder` and `vlm`.

## Configuration

Every field below is a `VisionSettings` attribute, set from the environment
with a `VISION_` prefix (`VISION_VLM_MODEL=gpt-4o`), read from the same `.env`
as the rest of the app. See [`config.py`](config.py) for the reasoning behind
each default.

### VLM provider

Any endpoint speaking OpenAI `chat/completions` with image parts. Leave the key
empty and the slow path is disabled — the fast path keeps working and
unanswerable questions get a polite "mình chỉ đọc được những gì hiện rõ".

```dotenv
VISION_VLM_BASE_URL=https://api.openai.com/v1
VISION_VLM_API_KEY=sk-...
VISION_VLM_MODEL=gpt-4o-mini
VISION_VLM_TIMEOUT_SECONDS=15
VISION_VLM_MAX_RETRIES=1
```

The prompt sent is fixed (`vlm_client.VLM_PROMPT_TEMPLATE`): the pre-extracted
`fast_summary` first, then the question, then an instruction to answer in
Vietnamese from the image alone and to say so when the image is not enough.

### Thresholds worth tuning

| setting | default | what moving it does |
|---|---|---|
| `det_score_min` | `0.50` | Lower to count more objects and more noise. |
| `ocr_score_min` | `0.60` | Stricter than the detector's on purpose — OCR is quoted back verbatim, and a wrong digit in a price is worse than a missed line. |
| `image_select_min_sim` | `0.25` | Raise to ask "which image?" more often; lower to guess more often. |
| `vlm_cache_min_sim` | `0.97` | **Not 0.80** — see below. |
| `thumb_ttl_seconds` | `900` | How long a lazy VLM call stays possible. |
| `max_images_per_session` | `8` | Per-session RAM ceiling. |
| `ocr_area_text_ratio` | `0.30` | Share of the frame under OCR boxes above which the image reads as a text surface. |
| `ocr_min_lines_text` | `8` | Second signal for the same thing; set `0` to disable — see below. |
| `num_threads` | `8` | Intra-op cap. The box has 16; chat and TTS need the rest. |
| `runtime` | `onnx` | `openvino` puts the detector on the Iris Xe, CPU fallback automatic. |

### Two thresholds that are not what they look like

**`vlm_cache_min_sim` is 0.97, not the 0.80 that sounds right.** Text-to-text
cosines from `clip-ViT-B-32-multilingual-v1` sit in a compressed high band — it
is a distilled, mean-pooled DistilBERT, not a model trained to spread sentences
apart. Measured here against *"Tại sao CPU chỉ chạy 1.78 GHz?"*:

| | cosine |
|---|---|
| three rephrasings of the same question | 0.9979 – 0.9989 |
| seven different questions about the same screenshot | 0.8095 – 0.9301 |

At 0.80, every one of those seven would have been served the cached answer to a
question nobody asked — and a false cache hit is silent, because no later step
re-checks it. 0.97 sits in the empty band between the clusters.
`tests/vision/test_models.py` asserts both clusters against the real encoder,
so swapping the text tower fails there rather than in production. **Recalibrate
this if you change the encoder — the right value is a property of the model.**

**`image_select_min_sim` at 0.25 has a thin margin.** Against a real photo,
question-shaped text (not caption-shaped) scored 0.26–0.27 on topic and
0.20–0.21 off it. It is left at 0.25 because ranking is what picks the image
and this only decides whether to ask instead — but expect the occasional
"which image?" about an image the question really was about, and note that one
measured photo is not enough evidence to move it either way.

`ocr_min_lines_text` is a **deliberate addition to the spec.** The area rule
alone misses the commonest real case: a UI screenshot whose text is small
against large charts and empty chrome. A Task Manager grab measures about
`0.10`, so the area test never fires and the image is classified as a photo —
which then makes "có mấy con chó?" fall through to a VLM call instead of being
answered "no" from an empty detection list. Many recognised lines with nothing
detected among them is a text surface regardless of how much of the frame the
glyphs cover.

## Measured on this machine

Intel i5-13500H (12C/16T, no CUDA), 40 GB RAM, Windows 11, ONNX Runtime CPU,
8 threads, 25 runs, against `tests/vision/fixtures/task_manager.png` (1280×800).
Reproduce with `python scripts/bench.py --runs 25 --image <path> --markdown`.

| stage | median | p95 |
|---|---|---|
| YOLO11n detect | 83 ms | 101 ms |
| RapidOCR read | 2173 ms | 3243 ms |
| CLIP encode_image | 70 ms | 81 ms |
| CLIP encode_text | 77 ms | 81 ms |
| thumbnail 768px | 25 ms | 30 ms |

End to end, through the real pipeline:

| | | |
|---|---|---|
| `warmup()`, all four models cold | 12.1 s | one-off, at startup |
| `ingest` — street photo, 1 OCR line | 1.4 s | |
| `ingest` — receipt, 25 OCR lines | 2.4 s | |
| `ingest` — Task Manager, 37 OCR lines | 2.9 s | |
| `answer` — detections, single-image session | 0.3 ms | |
| `answer` — OCR, single-image session | 15 ms | |
| `answer` — OCR, multi-image session | 115 ms | includes one CLIP text encode |

### Two budgets are missed, and neither is a bug in the routing

**Ingest is 1.4–2.9 s, not under 1 s.** The three model stages do run
concurrently, so the floor is the slowest of them and not their sum — but the
slowest is RapidOCR by a factor of 25. Its cost is roughly a 700 ms detection
floor plus per-box recognition, so it tracks how much text is in the image, not
how large the image is. Tuning was tried and did not help: forcing detection
down to a 736px long edge (`det_limit_type=max`) moved a dense screenshot from
1.9 s to 1.9 s and made the sparse receipt slower, because the cost is in the
recognition batches. Getting under a second means a faster recogniser or a
smaller one, not a config change. In practice ingest overlaps with the user
still typing, so this is latency the conversation usually absorbs.

**A fast-path answer is under 100 ms only in a single-image session.** With one
image the router short-circuits image selection and answers in well under
20 ms. With two or more it encodes the question with the CLIP text tower, and
that one call is ~77 ms of the ~115 ms total.

**Cold start is 12 s** and lands on the first user question unless
`await pipeline.warmup()` runs at startup — the CLIP text tower alone is
540 MB. Warm, the same call is ~0.1 s.

### What the models actually read

Worth knowing before trusting a number the layer quotes back:

- **RapidOCR's bundled recogniser is Chinese/English and drops Vietnamese
  diacritics** — it reads "TỔNG CỘNG 250.000đ" as "TONG CONG 250.000d". Field
  matching is accent-folded throughout (`summary.normalize_text`), so lookups
  still work; the *quoted text* is unaccented.
- **It does not reliably detect isolated small digits.** On the Task Manager
  fixture it reads every label in the CPU details block and misses the "1",
  "12" and "16" beside them, so `Cores → 12` is recovered from the scripted
  fixture but not from that grab. This is why a colon-terminated key with
  nothing to its right stays unpaired rather than pairing with the label below.
- **The exported detector matches ultralytics exactly** — verified against
  `ultralytics` running the same ONNX file: identical labels, scores to 0.001,
  boxes to 0.1 px.
- **The exported text tower matches sentence-transformers exactly** — cosine
  1.00000 against `SentenceTransformer(...).encode()`.

## Tests

```bash
python -m pytest tests/vision -q
```

Fully offline. The pipeline takes its four collaborators as constructor
arguments and the suite supplies scripted stand-ins, so the parts with the
interesting behaviour — pair extraction, question routing, the VLM cache, TTL
expiry — are tested against fixed, inspectable input rather than against
whatever YOLO scored today.

Two things follow from that and are worth knowing:

- The fixtures in [`tests/vision/fixtures.py`](../tests/vision/fixtures.py)
  declare each image once as a list of `(text, x, y, w, h)` boxes, then use it
  twice: to draw the PNG and to script the OCR. The scripted geometry *is* the
  geometry the pixels have, so `build_ocr_pairs` is tested against a real
  layout. `python -m tests.vision.fixtures --write` dumps them for eyeballing.
- The stand-in embedder is character-trigram cosine, not CLIP. It has the right
  shape — rephrasings score high, unrelated text scores low — which is what
  makes the cache threshold testable at all, but the numbers are not CLIP's.
  The tests therefore assert on *which branch was taken*, never on a similarity
  value.

`tests/vision/test_models.py` covers the real models and skips per model when
its export is missing.

## Not in scope here

No HTTP route, and nothing wired into `ServiceContainer` — this package is the
capability, not its exposure. No background job after ingest, no captioning
ahead of time, no image written to disk, no model beyond the four named above.
