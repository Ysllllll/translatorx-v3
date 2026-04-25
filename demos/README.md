# demos/

End-to-end runnable scripts demonstrating each layer of the codebase.
Most live LLM demos talk to a local Qwen3-32B vLLM at
`http://localhost:26592/v1` — adjust via `LLM_ENGINE_URL` / `LLM_MODEL` /
`LLM_API_KEY` env vars (or the `DEMO_LLM_*` prefix for `course_batch/`).

All demos are runnable two ways:

```bash
python demos/<file>.py [args...]
python -m demos.<package>            # for package-style demos
```

## Top-level demos

| File | Lines | What it shows | Needs LLM? |
|---|---:|---|---|
| **`demo_batch_translate.py`** | 332 | Main translate path: preprocess → streaming translate → bilingual table → STEP 4 cache hit (re-runs, expects ~thousands× speedup). **Start here.** | ✅ |
| **`demo_batch_preprocess.py`** | 485 | Preprocess path in isolation: NER/LLM punc + composite chunk + per-pipeline state rendering. STEP 7 record table is the canonical record renderer. | ✅ |
| `demo_advanced_features.py` | 524 | Four advanced topics behind `--only`: dynamic terms, prompt degradation (FlakyEngine), chunked sliding-window translate, summary integration. | ✅ |
| `demo_app.py` | 286 | `App` / `VideoBuilder` / `CourseBuilder` / `AppConfig` end-to-end. | ✅ |
| `demo_llm_ops.py` | 895 | Six-chapter llm_ops walk-through (engine → context → translate → checker → degradation → summary). | ✅ |
| `demo_checker.py` | 89 | Translation quality checker rules + profiles. | — |
| `demo_lang_ops.py` | 85 | LangOps factory, tokenization, sentence/clause splitting across languages. | — |
| `demo_subtitle.py` | 105 | Subtitle parsing, word alignment, segment rebuild. | — |
| `demo_media.py` | 136 | yt-dlp source + ffmpeg probe / extract_audio. | — |
| `demo_service.py` | 258 | FastAPI + SSE service entry-point (Stage 7). | ✅ |
| `_demo_shared.py` | 316 | **Library** for the top-level demos (engine factory, preprocess config factories, `render_records` / `render_translations`, `preprocess()` + `translate_records()` primitives). Imported, never run. | — |

## Sub-packages

### `demos/course_batch/`

Backend deep-dives that don't fit a single video translate. After the
December 2025 cleanup only the two unique demos remain (the redundant
`demo_translate` / `demo_preprocess` were removed; their workflows are
covered by `demo_batch_translate.py` / `demo_batch_preprocess.py`).

| File | Lines | What it shows |
|---|---:|---|
| `demo_standalone.py` | 287 | Six isolated backend demos: NER punc, LLM punc, Remote punc (doc only), spaCy splitter, LLM chunker, full hand-stepped pipeline. |
| `demo_sentence.py` | 375 | 30-segment hand-built fixture compared across 5 pipelines (Baseline / A=punc_global→sentences / B=sent→punc→sent / C=punc→sent→punc→sent / D=A+chunk). Best for understanding which preprocess order to use. |
| `_shared.py` | 101 | Local helpers (constants, `header`/`sub`/`ts`, `print_*_comparison`). |
| `__main__.py` | — | `python -m demos.course_batch` runs both. Set `DEMO_RUN=standalone` or `DEMO_RUN=sentence` to pick. |

### `demos/multilingual/`

Cross-language smoke demos (en / zh / ja / ko / etc.).

| File | Purpose |
|---|---|
| `demo_translate.py` | Translate fixture sentences across language pairs. |
| `demo_processing.py` | Apply per-language processing chains. |
| `demo_course.py` | Course-level multilingual run. |
| `__main__.py` | `python -m demos.multilingual` runs all three. |

### `demos/demo_stream_preprocess/`

WebSocket streaming preprocess demo (server + client).

| File | Purpose |
|---|---|
| `server.py` / `ws_app.py` | FastAPI + WebSocket service emitting preprocess events. |
| `client.py` | Reference client. |
| `backends.py` / `processors.py` | Service-side wiring. |

## Convention reminders

- **No private-attribute access.** Demos must go through public API:
  `Subtitle.pipeline_chunks() / pipeline_words() / pipeline_count()`,
  `App.set_engine() / wrap_engine()`. The `_pipelines` / `_engines["…"]`
  patterns are deprecated and will fail review.
- **Backend config schema is locked by `tests/demos/test_shared_configs.py`.**
  The `_demo_shared.make_punc_config` / `make_chunk_config` factories are
  asserted to actually load via `PuncRestorer.from_config` and
  `Chunker.from_config`. If you change those factories, run that test.
- **Rich rendering for new demos.** New top-level demos should use
  `_demo_shared.render_records` / `render_translations` (rich `Console` +
  `Table` + `Panel`) rather than ad-hoc `print()`. The `course_batch/`
  demos still use plain `print()` for now.
- **Live LLM demos must guard with `llm_up()`** (or skip with a clear
  message) so the script remains importable in CI.

## Smoke test

```bash
# Format
/home/ysl/workspace/.venv/bin/ruff format demos/

# Schema lock for _demo_shared
/home/ysl/workspace/.venv/bin/pytest tests/demos/test_shared_configs.py -v

# Whole suite
/home/ysl/workspace/.venv/bin/pytest tests/ -q
```
