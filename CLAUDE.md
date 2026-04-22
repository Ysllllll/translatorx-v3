# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Code Style

After writing or modifying any Python file, always run:

```bash
/home/ysl/workspace/.venv/bin/ruff format <file>
```

Or for multiple files / full directory:

```bash
/home/ysl/workspace/.venv/bin/ruff format src/ tests/
```

This ensures consistent formatting. Never skip this step.

## Commands

```bash
# Run all tests
pytest tests/ -v

# Run a single test file
pytest tests/lang_ops_tests/test_chinese.py -v
pytest tests/lang_ops_tests/chunk/test_chinese.py -v
pytest tests/subtitle/build_tests/test_english.py -v

# Run via the venv explicitly (if pytest not on PATH)
/home/ysl/workspace/.venv/bin/pytest tests/ -v

# Run with coverage
/home/ysl/workspace/.venv/bin/pytest tests/ -v --cov=src --cov-report=term-missing
```

`pyproject.toml` sets `pythonpath = ["src"]` so imports like `from domain.lang import LangOps` resolve directly.

## Architecture

A subtitle translation platform organized as **Hexagonal (Ports & Adapters)** with 5 layers under `src/`.

### Layout overview

```
src/
├── domain/                          【L0 · pure domain, no I/O】
│   ├── model/                       # Word, Segment, SentenceRecord, Usage, CompletionResult
│   ├── lang/                        # LangOps + TextPipeline + per-language tokenizers (was lang_ops/)
│   └── subtitle/                    # Subtitle class + word-timing alignment (was subtitle/core.py + align.py)
│
├── ports/                           【L1 · abstract protocols + generic utilities】
│   ├── source.py                    # Source, Processor, VideoKey, Priority
│   ├── processor.py                 # ProcessorBase
│   ├── engine.py                    # LLMEngine, Message
│   ├── media.py                     # MediaSource, MediaProbe, MediaInfo, PlaylistInfo, DownloadResult
│   ├── errors.py                    # ErrorCategory, ErrorInfo, EngineError, ErrorReporter
│   ├── apply_fn.py                  # ApplyFn protocol (list[str] -> list[list[str]])
│   └── retries.py                   # retry_until_valid, OnFailure, AttemptOutcome (generic)
│
├── adapters/                        【L2 · concrete external implementations】
│   ├── sources/                     # SrtSource, WhisperXSource, PushQueueSource
│   ├── storage/                     # JsonFileStore, Workspace, Store protocol
│   ├── engines/openai_compat.py     # OpenAICompatEngine + EngineConfig
│   ├── parsers/                     # parse_srt, read_srt, sanitize_srt, parse_whisperx, read_whisperx
│   ├── media/                       # YtdlpSource, ffmpeg.probe, ffmpeg.extract_audio
│   ├── preprocess/                  # NerPuncRestorer, LlmPuncRestorer, RemotePuncRestorer,
│   │                                # Chunker + backends (rule/spacy/llm/composite) + _availability
│   └── reporters/reporters.py       # LoggerReporter, JsonlErrorReporter, ChainReporter
│
├── application/                     【L3 · use cases / orchestration】
│   ├── orchestrator/                # VideoOrchestrator, StreamingOrchestrator (video.py), CourseOrchestrator (course.py)
│   ├── translate/                   # TranslationContext, translate_with_verify, providers, agents, prompts
│   ├── checker/                     # Checker engine, rules, per-language profiles, default_checker
│   ├── processors/                  # TranslateProcessor, SummaryProcessor, prefix.PrefixHandler
│   ├── observability/               # ProgressEvent, ProgressReporter (errors live in ports/errors.py)
│   ├── resources.py                 # InMemoryResourceManager + UserTier + DEFAULT_TIERS
│   └── config.py                    # AppConfig (Pydantic v2)
│
└── api/                             【L4 · user entrypoints】
    ├── app/                         # App + VideoBuilder + CourseBuilder + StreamBuilder
    └── trx/                         # create_engine, create_context, translate_srt + minimal type re-exports
```

### Dependency rule

Layers can only depend inward — each layer may import from itself or layers to its left:

```
api  →  application  →  adapters  →  ports  →  domain
```

Violations are caught by `tests/test_architecture.py` which parses every `src/` file's imports and flags any upward reference (imports inside `if TYPE_CHECKING:` blocks are exempt because they don't execute).

### Key design decisions

**Factory pattern:** `LangOps.for_language(code)` returns a cached `_BaseOps` subclass via `functools.lru_cache` — thread-safe.

**Two language families:**
- **EnType** (`domain/lang/en_type.py`): Space-delimited languages (en, ru, es, fr, de, pt, vi).
- **CJK** (`domain/lang/_core/_cjk_common.py`): Character-based, external tokenizers (jieba / MeCab / Kiwi). Korean preserves eojeol boundaries.

**Immutability:** `TextPipeline`, `Subtitle`, all domain dataclasses are frozen. `align.py` uses `dataclasses.replace()` rather than mutation.

**Per-sentence pipelines:** After `Subtitle.sentences()`, each sentence owns its own `TextPipeline` + word subset. Operations never cross sentence boundaries.

**Runtime orchestration invariants:**
- Processors are stateless — state flows through `Store` (one JSON file per video under `<root>/<course>/zzz_translation/<video>.json`).
- Builders (`VideoBuilder`, `CourseBuilder`, `StreamBuilder`) are immutable — each stage returns a fresh instance via `dataclasses.replace()`.
- Cancel discipline: `finally` blocks use `asyncio.shield()` for Store flushes.

### Test structure

```
tests/
├── lang_ops_tests/              # Token + chunk tests
│   ├── _base.py                 # TextOpsTestCase — shared assertion helpers
│   ├── conftest.py              # Font path resolution, pixel length fixture
│   ├── test_{language}.py       # Per-language token-level tests (10 files, English full names)
│   ├── chunk/
│   │   ├── _base.py             # SplitterTestBase — reconstruction assertions
│   │   └── test_{language}.py   # Per-language chunk tests (English full names)
│   └── _core/
│       ├── test_mechanism.py    # Factory tests
│       ├── test_normalize.py    # Language code normalization
│       └── test_punctuation.py  # Punctuation constants tests
├── subtitle/
│   ├── align_tests/             # Word timing tests
│   │   ├── test_align.py        # align_segments
│   │   ├── test_attach_punct.py # attach_punct_words
│   │   ├── test_distribute.py   # distribute_words
│   │   ├── test_fill.py         # fill_words
│   │   ├── test_find.py         # find_words
│   │   ├── test_normalize.py    # normalize_words
│   │   └── test_pipeline.py     # Pipeline integration
│   ├── test_model.py            # Data type display/pretty tests
│   ├── build_tests/             # Subtitle tests (English full names)
│   │   ├── _base.py             # BuilderTestBase
│   │   ├── test_english.py
│   │   ├── test_korean.py
│   │   └── test_chinese.py
│   └── io_tests/
│       ├── test_sanitize_srt.py # SRT sanitization tests
│       ├── test_srt.py          # SRT parser tests
│       └── test_whisperx.py     # WhisperX parser tests
├── llm_ops_tests/               # LLM engine + context + translate tests
│   ├── test_checker.py          # Checker integration via translate_with_verify
│   ├── test_context.py          # ContextWindow, StaticTerms, TermsProvider
│   ├── test_protocol.py         # LLMEngine Protocol conformance
│   ├── test_translate.py        # translate_with_verify micro-loop
│   └── engines_tests/
│       └── test_openai_compat.py # OpenAI-compatible engine
├── media_tests/                 # Media download + extraction tests
│   ├── test_ffmpeg.py           # ffprobe + extract_audio
│   ├── test_protocol.py         # MediaSource Protocol conformance
│   └── test_ytdlp.py            # yt-dlp source tests
├── preprocess_tests/            # Preprocessing tests
│   ├── test_ner_punc.py         # NerPuncRestorer + dotted-word + trailing-punc protection
│   ├── chunk/                   # Chunker orchestrator + registry + per-backend tests
│   │   ├── test_registry.py
│   │   ├── test_chunker.py
│   │   ├── test_llm_backend.py
│   │   ├── test_spacy_backend.py
│   │   └── test_composite_backend.py
└── runtime_tests/               # Runtime orchestration tests
    ├── test_app.py              # App + VideoBuilder + CourseBuilder
    ├── test_config.py           # AppConfig (YAML + dict + env overrides)
    ├── test_store.py            # JsonFileStore
    ├── test_workspace.py        # Workspace layout
    ├── test_resource_manager.py # InMemoryResourceManager
    ├── test_reporters.py        # Logger / Jsonl reporters
    ├── test_usage.py            # Usage, CompletionResult
    ├── test_base.py             # Protocol/errors/progress shape tests
    ├── orchestrator_tests/      # VideoOrchestrator / Streaming / Course
    ├── processors_tests/        # TranslateProcessor + prefix
    └── sources_tests/           # SrtSource, WhisperXSource, PushQueueSource
```

Test directory is `lang_ops_tests` / `runtime_tests` etc. (not matching src package names exactly) to prevent Python from shadowing the package during collection. Inside tests we import from the new layout (`from domain.lang import LangOps`, `from adapters.sources.srt import SrtSource`, etc.).

**Architecture guard**: `tests/test_architecture.py` parametrizes over every file under `src/` and asserts that runtime imports only point to allowed (equal-or-lower) layers. Adding a new file automatically enrolls it.

## Dependencies

- **Python 3.10+** (`list[list]`, `str | None`, `slots=True`, `frozen=True`)
- **Pillow** — pixel length via `plength()`
- **jieba** / **MeCab** / **kiwipiepy** — CJK tokenizers (conditional, tests skip if missing)

- **deepmultilingualpunctuation** — NER punctuation restoration (conditional, tests skip if missing)
- **spacy** — sentence splitting via `spacy_backend` (conditional, tests skip if missing)

Check availability at runtime: `jieba_is_available()`, `mecab_is_available()`, `kiwi_is_available()`, `langdetect_is_available()` (exported from `domain.lang`); `punc_model_is_available()`, `spacy_is_available()` (exported from `adapters.preprocess`).

## API quick reference

### trx — unified slim facade (recommended entry point)

```
from api import trx

# Factory functions
engine = trx.create_engine(model="Qwen/Qwen3-32B", base_url="http://localhost:26592/v1")
ctx = trx.create_context("en", "zh", terms={"AI": "人工智能"})

# One-line SRT translation
records = await trx.translate_srt(srt_content, engine, src="en", tgt="zh")

# Config-driven App + chainable Builders (recommended for real apps)
from api.app import App

app = App.from_config("app.yaml")       # or App.from_yaml(text) / App.from_dict({...})

# Single-video builder
result = await (
    app.video(course="course-1", video="lec01")
        .source("lec01.srt", language="en")
        .translate(src="en", tgt="zh")
        .run()
)

# Course builder — batches many videos with bounded concurrency
course_result = await (
    app.course(course="course-1")
        .add_video("lec01", "lec01.srt", language="en")
        .add_video("lec02", "lec02.srt", language="en")
        .translate(src="en", tgt="zh")
        .run()
)
```

`trx` only exports factories + a handful of common types. For everything else, import directly from the canonical location:

| Need | Import from |
|------|-------------|
| `Word`, `Segment`, `SentenceRecord`, `Usage` | `domain.model` |
| `LangOps`, `TextPipeline` | `domain.lang` |
| `Subtitle`, `fill_words`, `align_segments`, ... | `domain.subtitle` |
| `parse_srt`, `read_srt`, `parse_whisperx` | `adapters.parsers` |
| `extract_audio`, `YtdlpSource`, `MediaSource` | `adapters.media` / `ports.media` |
| `OpenAICompatEngine`, `EngineConfig` | `adapters.engines.openai_compat` |
| `PuncRestorer`, `Chunker` (registry-based) | `adapters.preprocess` |
| `TranslationContext`, `translate_with_verify`, `StaticTerms` | `application.translate` |
| `Checker`, `default_checker`, `CheckReport` | `application.checker` |
| `TranslateProcessor`, `SummaryProcessor` | `application.processors` |
| `VideoOrchestrator`, `CourseOrchestrator` | `application.orchestrator` |
| `JsonFileStore`, `Workspace` | `adapters.storage` |
| `AppConfig` | `application.config` |
| `App`, `VideoBuilder`, `CourseBuilder`, `StreamBuilder` | `api.app` |

### Language operations

```
from domain.lang import LangOps

ops = LangOps.for_language("en")     # Factory — cached, returns _BaseOps subclass

# Token-level
ops.split(text, mode="word")         # "word" | "character" ("w" | "c")
ops.join(tokens)
ops.length(text, cjk_width=1)
ops.normalize(text)
ops.transfer_punc(text_a, text_b)

# Segment-level shortcuts
ops.split_sentences(text) → list[str]
ops.split_clauses(text)   → list[str]   # sentence-aware (splits at sentence boundaries too)
ops.split_by_length(text, max_len) → list[str]
ops.merge_by_length(chunks, max_len) → list[str]  # greedy merge (inverse of split)
ops.chunk(text) → TextPipeline
```

### TextPipeline (chainable, immutable — pure text structuring)

```
ops.chunk(text)
  .sentences()
  .clauses(merge_under=60)  # sentence-aware; merge_under merges back short clauses
  .split(max_len=50)        # split by length
  .merge(max_len=80)        # greedy merge all adjacent chunks
  .result()                 → list[str]

# Alternative construction
TextPipeline.from_chunks(chunks, ops)  # from pre-split chunk list
```

### Subtitle (chainable, immutable — per-sentence pipelines)

```
from domain.subtitle import Subtitle

sub = Subtitle(segments, language="zh")           # or ops=ops; split_by_speaker=True groups by speaker
sub = Subtitle.from_words(words, language="zh")   # from flat word list
sub.sentences()                        → Subtitle  # splits into per-sentence pipelines (early word alignment)
sub.clauses(merge_under=60)            → Subtitle  # per-sentence clause splitting
sub.split(max_len=40)                  → Subtitle  # per-sentence length splitting
sub.merge(max_len=60)                  → Subtitle  # per-sentence greedy merge
sub.transform(fn, *, cache=None, scope="chunk", batch_size=1, workers=1, skip_if=None)  → Subtitle  # unified transform
sub.build()                            → list[Segment]
sub.records()                          → list[SentenceRecord]

# Streaming mode (split_by_speaker=True groups by speaker)
stream = Subtitle.stream(language="zh")
done = stream.feed(segment)            → list[Segment]  # completed sentences
remaining = stream.flush()             → list[Segment]
```

After `sentences()`, each operation is implicitly per-sentence — it never crosses sentence boundaries.

`transform()` scope parameter:
- **`scope="chunk"`** (default): applies `fn` to each chunk individually.
- **`scope="joined"`**: joins all chunks within a pipeline before sending to `fn`, then rebuilds the pipeline from the result. Use for punc restoration where the fn needs full context.

### Subtitle word timing

```
normalize_words(text, words, split_fn=None, start=0.0, end=0.0) → tuple[str, list[Word]]  # reconcile text + words
attach_punct_words(words) → list[Word]           # merge standalone punct into adjacent words
fill_words(segment, split_fn=None) → Segment     # populate segment.words (auto-attaches punct)
find_words(words, sub_text, start=0) → (start_idx, end_idx)
distribute_words(words, texts) → list[list[Word]]
align_segments(chunks, words) → list[Segment]    # text chunks + timed words → Segments
```

### Data types (all frozen)

- `Word(word, start, end, speaker=None, extra={})` — `content` property returns word stripped of punctuation
- `Segment(start, end, text, speaker=None, words=[], extra={})`
- `SentenceRecord(src_text, start, end, segments=[], ...)` — also has `translations`, `alignment`

### SRT reader

```
from adapters.parsers import sanitize_srt, parse_srt, read_srt

sanitize_srt(content) → str            # text-level cleaning (BOM, CRLF, HTML, invisible chars, etc.)
parse_srt(content) → list[Segment]     # parse SRT string
read_srt(path)     → list[Segment]     # parse SRT file
```

### WhisperX reader

```
from adapters.parsers import sanitize_whisperx, parse_whisperx, read_whisperx

sanitize_whisperx(word_segments) → list[Word]  # sanitize raw word dicts (dedup, interpolate, attach punct, collapse repeats)
parse_whisperx(data)             → list[Word]  # parse JSON dict (expects 'word_segments' key)
read_whisperx(path)              → list[Word]  # read JSON file
```

### Preprocessing (ApplyFn-based — all conform to `list[str] → list[list[str]]`)

Punctuation restoration and chunking both use the same registry + orchestrator pattern:

```
from adapters.preprocess import PuncRestorer, Chunker

# Unified punctuation restorer with per-language backend dispatch
restorer = PuncRestorer(backends={
    "en": {"library": "deepmultilingualpunctuation"},
    "zh": {"library": "llm", "engine": engine},
})
fn = restorer.for_language("en")
fn(["hello world this is a test"])  # → [["Hello world, this is a test."]]

# Unified chunker — "rule" | "spacy" | "llm" | "composite" backends
chunker = Chunker(backends={
    "en": {"library": "composite", "language": "en", "chunk_len": 90,
           "inner": {"library": "spacy"},
           "refine": {"library": "llm", "engine": engine}},
})
fn = chunker.for_language("en")
fn(["long text..."])  # → [["chunk1", "chunk2"]]
```

### Orchestration (async, immutable)

```
from application.orchestrator import VideoOrchestrator, CourseOrchestrator, StreamingOrchestrator
from application.processors import TranslateProcessor, TranslateNodeConfig
from application.config import AppConfig
from adapters.sources.srt import SrtSource
from adapters.sources.whisperx import WhisperXSource
from adapters.sources.push import PushQueueSource
from adapters.storage import JsonFileStore, Workspace
from ports.source import VideoKey
from api.app import App

# Lower-level: assemble orchestrator manually
orch = VideoOrchestrator(
    source=SrtSource(path, language="en"),
    processors=[TranslateProcessor(engine, checker)],
    ctx=ctx,
    store=JsonFileStore(Workspace(root=..., course="c1")),
    video_key=VideoKey(course="c1", video="lec01"),
)
result = await orch.run()
records = result.records

# Configuration (Pydantic v2 with `extra="forbid"`):
cfg = AppConfig.load("app.yaml")             # file
cfg = AppConfig.from_yaml("engines:\n ...")   # string
cfg = AppConfig.from_dict({...})             # dict
# Env overrides: TRX_<SECTION>__<KEY> (double underscore between levels)
```

Key invariants:
- **Processor stateless** — all state flows through `Store` (JSON-per-video under `<root>/<course>/zzz_translation/<video>.json`)
- **Immutable Builders** — each stage returns a fresh instance via `dataclasses.replace()`
- **Auto resolution** — Builders resolve engine/ctx/checker from the App by language pair; users never pass them by hand
- **Cancel discipline** — `finally` blocks use `asyncio.shield()` for Store flushes so in-flight work is persisted
- **No print** — all logging goes through `logger` + `ProgressReporter`; demos are the only place stdout is used directly

## Fonts

Pixel-length tests require system fonts. `conftest.py` tries in order:
1. `/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc`
2. `/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf`
3. `/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf`

Raises `SkipTest` if none found.
