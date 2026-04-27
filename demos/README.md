# demos/

End-to-end runnable scripts demonstrating each layer of the codebase,
organised by **scenario** (what you want to do) rather than by source
file. Most live LLM demos talk to a local Qwen3-32B vLLM at
`http://localhost:26592/v1`; override via `DEMO_LLM_BASE_URL` /
`DEMO_LLM_MODEL`.

```
demos/
├── _print.py         # unified Console + section/step/banner helpers
├── _shared.py        # higher-level helpers (engine factory, render_*, …)
├── basics/           # standalone primitives (no I/O)
├── batch/            # offline / batch translation flows
├── runtime/          # the App + builders + pipeline runtime
├── streaming/        # WebSocket / live / push-queue scenarios
├── service/          # FastAPI HTTP service + browser scenarios
└── internals/        # walk-through of the lower-level building blocks
```

Every subdir carries a self-contained `_bootstrap.py` so each script is
runnable as `python demos/<scenario>/<file>.py`.

## Scenarios

### `basics/` — primitives without I/O

| File | What it shows |
|---|---|
| `lang_ops.py` | `LangOps` factory, tokenization, sentence/clause splitting across languages. |
| `subtitle.py` | SRT / WhisperX parsing, word alignment, segment rebuild. |
| `checker.py`  | Translation quality checker rules + profiles. |

### `batch/` — offline batch translation

| File | What it shows | Needs LLM? |
|---|---|---|
| `translate.py` | Main translate path: preprocess → streaming translate → bilingual table → cache hit on re-run. **Start here.** | ✅ |
| `preprocess.py` | Preprocess in isolation: NER/LLM punc + composite chunk + per-pipeline state. | ✅ |
| `transcribe.py` | WhisperX + remote transcriber adapters. | — |
| `advanced.py` | Four advanced topics behind `--only`: dynamic terms, prompt degradation, chunked translate, summary. | ✅ |
| `multilingual.py` | Cross-language smokes (en / zh / ja / ko …). `--only translate,processing,course`. | ✅ for `translate,course` |
| `comparisons/` | 30-segment hand-built fixture × 5 preprocess pipelines + isolated backend deep-dives (`demo_standalone.py`, `demo_sentence.py`). | ✅ |

### `runtime/` — App + builders + pipeline

| File | What it shows |
|---|---|
| `app.py` | `App` / `VideoBuilder` / `CourseBuilder` / `AppConfig` end-to-end. |
| `pipeline.py` | Stage-based pipeline runtime: builder + YAML + tracing middleware. |
| `admin.py` | Admin scenarios: tenant namespacing, hot reload, registry-bound DSL validation. |

### `streaming/` — live / WebSocket / push-queue

| File | What it shows |
|---|---|
| `memory.py` | In-memory streaming orchestrator end-to-end. |
| `redis_bus.py` | Same flow over Redis pub/sub. |
| `tenant.py` | Tenant-scoped scheduler & quota enforcement. |
| `ws_client.py` | Walk-through of `/api/ws/streams` frame protocol via `TestClient`. |
| `preprocess_server/` | FastAPI + WebSocket preprocess service + reference client. |

### `service/` — FastAPI HTTP

| File / Dir | What it shows |
|---|---|
| `translate_api.py` | FastAPI + SSE service entry-point (Stage 7). |
| `browser_upload/`  | **Drag-drop SRT in a browser** → real `/api/ws/streams` → bilingual table streamed back. Vanilla HTML+JS, no bundler. |

### `internals/` — building blocks

| Path | What it shows |
|---|---|
| `media.py` | yt-dlp source + ffmpeg probe / extract_audio. |
| `llm_ops/` | Six-chapter walk-through of LLM ops:<br>1 checker · 2 bypasses · 3-4 translate · 5 degrade · 6 streaming/OneShotTerms.<br>Run `python demos/internals/llm_ops/__main__.py`. |

## Conventions

- **No private-attribute access.** Demos must go through public API:
  `Subtitle.pipeline_chunks()`, `App.set_engine()`, etc.
- **Print style.** Every demo routes through `demos/_print.py`
  (`section` / `step` / `banner` / `kv` / `info` / `ok` / `warn` / `err`).
  `_shared.step` and `internals/llm_ops/_common.{header,sub}` already
  delegate there — keep new code on the same path.
- **Live LLM demos must guard with `llm_alive()`** (or `llm_up()`) so
  the script remains importable in CI without a real LLM.
- **Backend config schema is locked by `tests/demos/test_shared_configs.py`.**
  Run that test if you change `_shared.make_punc_config` / `make_chunk_config`.

## Smoke

```bash
# Format
/home/ysl/workspace/.venv/bin/ruff format demos/

# Demo regression suite
/home/ysl/workspace/.venv/bin/pytest tests/demos -q

# Whole suite (baseline 2309 passed / 3 skipped)
/home/ysl/workspace/.venv/bin/pytest tests/ -q
```
