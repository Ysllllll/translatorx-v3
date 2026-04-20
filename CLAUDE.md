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

`pyproject.toml` sets `pythonpath = ["src"]` so tests resolve `lang_ops`, `subtitle`, `model`, `checker`, `llm_ops`, `media`, `runtime`, `trx`, and `preprocess` from `src/`.

## Architecture

A subtitle translation platform with nine top-level packages under `src/`.

### Package overview

```
src/
├── model/                           # Shared data types (L0 — no cross-package deps except lang_ops)
│   └── __init__.py                  # Word, Segment, SentenceRecord (frozen dataclasses)
├── lang_ops/                        # Language-adapted text operations (L1)
│   ├── __init__.py                  # Public API: LangOps, TextPipeline, normalize_language
│   ├── en_type.py                   # EnTypeOps (shared by 7 space-delimited languages)
│   ├── chinese.py / japanese.py / korean.py  # CJK language ops
│   ├── _core/
│   │   ├── _base_ops.py             # _BaseOps ABC — abstract interface + shared concrete methods
│   │   ├── _mechanism.py            # LangOps factory (cached via lru_cache)
│   │   ├── _cjk_common.py           # _BaseCjkOps + token parsing/attachment/join helpers
│   │   ├── _chars.py                # Unicode classification (CJK/Hangul/Kana detection)
│   │   ├── _punctuation.py          # Single source of truth for all punctuation constants
│   │   ├── _normalize.py            # Language code normalization
│   │   └── _availability.py         # Optional dependency guards (jieba/mecab/kiwi)
│   └── chunk/
│       ├── _pipeline.py             # TextPipeline (immutable, chainable — pure text structuring)
│       ├── _boundary.py             # Token-based boundary detection (sentences + clauses)
│       ├── _length.py               # Length-based splitting (uses Protocol for decoupling)
│       └── _merge.py                # Length-based merging (inverse of splitting)
├── media/                           # Media download + audio extraction (L1)
│   ├── protocol.py                  # MediaSource Protocol
│   ├── ytdlp.py                     # yt-dlp implementation
│   └── ffmpeg.py                    # ffprobe + extract_audio
├── subtitle/                        # Subtitle timing alignment + segment building (L2)
│   ├── __init__.py                  # Re-exports model types + Subtitle/Stream + alignment utilities
│   ├── model.py                     # Backward-compat shim → re-exports from model package
│   ├── align.py                     # Word timing: fill_words, find_words, distribute_words, align_segments
│   ├── core.py                      # Subtitle — chainable segment restructuring + transform dispatch
│   └── io/
│       ├── srt.py                   # SRT file parser + sanitize_srt
│       └── whisperx.py              # WhisperX JSON parser + word-level sanitizer
├── llm_ops/                         # LLM engine + translation context (L2)
│   ├── protocol.py                  # LLMEngine Protocol (complete + stream)
│   ├── context.py                   # TermsProvider, StaticTerms, ContextWindow, TranslationContext
│   ├── translate.py                 # translate_with_verify micro-loop (prompt degradation)
│   └── engines/
│       └── openai_compat.py         # OpenAI-compatible engine
├── checker/                         # Translation quality checker (L2, top-level)
│   ├── types.py                     # Severity, Issue, CheckReport
│   ├── rules.py                     # Rule Protocol + 5 rule classes
│   ├── config.py                    # ProfileOverrides, PROFILES
│   ├── checkers.py                  # Checker (rule engine, ERROR short-circuit)
│   ├── factory.py                   # default_checker(src, tgt)
│   └── lang/                        # 10 per-language LangProfile files
├── runtime/                         # Orchestration + Processors + App (L3)
│   ├── protocol.py                  # Processor[In,Out] / Source / VideoKey
│   ├── errors.py                    # ErrorCategory, ErrorInfo, ErrorReporter
│   ├── progress.py                  # ProgressEvent, ProgressReporter
│   ├── usage.py                     # Usage, CompletionResult (D-048)
│   ├── store.py                     # Store Protocol + JsonFileStore (D-041..044)
│   ├── workspace.py                 # Workspace layout (course/video paths)
│   ├── resource_manager.py          # InMemoryResourceManager (D-033, D-051)
│   ├── reporters.py                 # LoggerReporter / JsonlErrorReporter
│   ├── orchestrator.py              # VideoOrchestrator + StreamingOrchestrator (D-060)
│   ├── course.py                    # CourseOrchestrator (D-055)
│   ├── app.py                       # App + VideoBuilder + CourseBuilder (D-059)
│   ├── config.py                    # AppConfig YAML/dict (Pydantic v2) (D-057)
│   ├── sources/                     # Source impls: srt, whisperx, push
│   └── processors/                  # TranslateProcessor + prefix + more
├── trx/                             # Unified API facade (L3)
│   └── __init__.py                  # create_engine, create_context, translate_srt + App/Builders re-exports
├── preprocess/                      # Preprocessing — punc restoration, sentence splitting, chunking (L2)
│   ├── __init__.py                  # Conditional re-exports (NerPuncRestorer, SpacySplitter, etc.)
│   ├── _protocol.py                 # ApplyFn Protocol: list[str] → list[list[str]]
│   ├── _availability.py             # Optional dep guards (deepmultilingualpunctuation, spacy, langdetect)
│   ├── _ner_punc.py                 # NerPuncRestorer — NER-based punc restoration (singleton, thread-safe)
│   ├── _llm_punc.py                 # LlmPuncRestorer — LLM-based punc restoration
│   ├── _remote_punc.py              # RemotePuncRestorer — remote API punc restoration
│   ├── _spacy.py                    # SpacySplitter — spaCy sentence splitting (singleton, dotted-word protection)
│   ├── _chunk.py                    # LlmChunker — LLM-based recursive binary splitting
│   └── _spacy_llm_chunk.py          # SpacyLlmChunker — two-stage: spaCy coarse + LLM fine split
```

### Key design decisions

**Factory pattern:** `LangOps.for_language(code)` returns a cached `_BaseOps` subclass. Uses `functools.lru_cache` — thread-safe, no manual cache management.

**Two language families:**
- **EnType** (`en_type.py`): Space-delimited languages (en, ru, es, fr, de, pt, vi). `split()` uses `str.split()`. Per-language abbreviation sets. French `normalize()` has special spacing rules.
- **CJK** (`_cjk_common.py` base): Character-based. `split()` uses external tokenizers (jieba/MeCab/Kiwi). Korean overrides `split()`/`join()` to preserve eojeol boundaries. CJK terminators include both full-width and half-width punctuation (e.g. `"。", "！", "？", "!", "?"`).

**strip_spaces property:** `_BaseOps.strip_spaces` controls whether `split_sentences`/`split_clauses` strip leading spaces from chunks. Defaults to `self.is_cjk` (True for Chinese/Japanese, since CJK doesn't use inter-sentence spaces). Korean overrides to `False` because it uses spaces between eojeols.

**Immutability:** `TextPipeline` and `Subtitle` return new instances per step. All `subtitle` dataclasses use `frozen=True`. `align.py` uses `dataclasses.replace()` instead of mutation.

**Protocol decoupling:** `_length.py` and `_merge.py` define Protocol types instead of importing `_BaseOps`, keeping the chunk package independent from the ops layer.

**Token-based boundary detection:** `_boundary.py` unifies sentence and clause splitting via `find_boundaries()` / `split_tokens_by_boundaries()`. Sentence splitting uses token-level boundary markers (terminators, abbreviations, ellipsis guards). Clause splitting (`split_clauses`) is sentence-aware — it splits at clause separators and sentence boundaries in one pass.

**Per-sentence pipelines:** After `Subtitle.sentences()`, the instance holds one `TextPipeline` per sentence with its corresponding words. Subsequent operations (`clauses`, `split`, `merge`, `transform`) are applied per-sentence — they never cross sentence boundaries. This structural isolation replaces the previous `_parent_ids` mechanism.

### Layer relationship

```
L0: model (Word, Segment, SentenceRecord)
L1: lang_ops, media
L2: subtitle, llm_ops, checker, preprocess
L3: runtime, trx (facade)
```

Dependencies flow downward only. `model` depends on `lang_ops._core._punctuation` for `strip_punct`.
`subtitle` re-exports model types for backward compatibility.
`llm_ops` re-exports checker types (`Checker`, `CheckReport`, `Severity`, `default_checker`) for convenience.
`trx` is a pure facade — re-exports from all lower packages + factory functions. No new logic.
`runtime` owns all orchestration state (Store, Workspace, Orchestrator, App/Builder).
The legacy `pipeline/` package was removed in Stage 5 — use `runtime.TranslateProcessor` + `VideoOrchestrator` (or the higher-level `App`/`Builder`) instead.

`subtitle` depends on `lang_ops` via `TextPipeline` for text structuring and `Subtitle` which takes an `ops` or `language` parameter. Transform dispatch (`_call_apply_fn`) lives in `subtitle/core.py` (L2), not in `lang_ops` (L1).

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
│   ├── test_spacy.py            # SpacySplitter + dotted-word sentence splitting
│   └── ...                      # LlmChunker, SpacyLlmChunker, etc.
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

Test directory is `lang_ops_tests` (not `lang_ops`) to prevent Python from importing it instead of `src/lang_ops`.

## Dependencies

- **Python 3.10+** (`list[list]`, `str | None`, `slots=True`, `frozen=True`)
- **Pillow** — pixel length via `plength()`
- **jieba** / **MeCab** / **kiwipiepy** — CJK tokenizers (conditional, tests skip if missing)

- **deepmultilingualpunctuation** — NER punctuation restoration (conditional, tests skip if missing)
- **spacy** — sentence splitting via `SpacySplitter` (conditional, tests skip if missing)

Check availability at runtime: `jieba_is_available()`, `mecab_is_available()`, `kiwi_is_available()` (exported from `lang_ops`); `punc_model_is_available()`, `spacy_is_available()` (exported from `preprocess`).

## API quick reference

### trx — unified facade (recommended entry point)

```
import trx

# Factory functions
engine = trx.create_engine(model="Qwen/Qwen3-32B", base_url="http://localhost:26592/v1")
ctx = trx.create_context("en", "zh", terms={"AI": "人工智能"})

# One-line SRT translation
records = await trx.translate_srt(srt_content, engine, src="en", tgt="zh")

# Config-driven App + chainable Builders (recommended for real apps)
from runtime import App

app = App.from_config("app.yaml")       # or App.from_yaml(text) / App.from_dict({...})

# Single-video builder
result = await (
    app.video(course="course-1", video="lec01")
        .source("lec01.srt", language="en")   # kind auto-detected from .srt / .json
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

# All common types available: trx.Subtitle, trx.Word, trx.Segment, trx.App, trx.VideoBuilder, trx.AppConfig, ...
```

### Language operations

```
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
from subtitle import Subtitle

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
from subtitle.io import sanitize_srt, parse_srt, read_srt

sanitize_srt(content) → str            # text-level cleaning (BOM, CRLF, HTML, invisible chars, etc.)
parse_srt(content) → list[Segment]     # parse SRT string
read_srt(path)     → list[Segment]     # parse SRT file
```

### WhisperX reader

```
from subtitle.io import sanitize_whisperx, parse_whisperx, read_whisperx

sanitize_whisperx(word_segments) → list[Word]  # sanitize raw word dicts (dedup, interpolate, attach punct, collapse repeats)
parse_whisperx(data)             → list[Word]  # parse JSON dict (expects 'word_segments' key)
read_whisperx(path)              → list[Word]  # read JSON file
```

### Preprocessing (ApplyFn-based — all conform to `list[str] → list[list[str]]`)

```
from preprocess import NerPuncRestorer, SpacySplitter, SpacyLlmChunker, LlmChunker

# NER punctuation restoration (singleton, thread-safe)
restorer = NerPuncRestorer.get_instance()
results = restorer(["hello world this is a test"])  # → [["Hello world, this is a test."]]
# Protects dotted words (Node.js) from corruption, preserves trailing punc (...)

# spaCy sentence splitting (singleton per model)
splitter = SpacySplitter.get_instance()             # default model from DEFAULT_SPACY_MODEL
results = splitter(["Hello world. This is Node.js."])  # → [["Hello world.", "This is Node.js."]]
# Protects dotted compound words (Node.js, Vue.js) from false sentence splits

# Two-stage chunker: spaCy coarse split → LLM fine split for oversized chunks
chunker = SpacyLlmChunker(splitter, llm_chunker, chunk_len=90)
results = chunker(["long text..."])                 # → [["chunk1", "chunk2"]]
```

### Runtime orchestration (async, immutable)

```
from runtime import (
    App, AppConfig,
    VideoOrchestrator, VideoKey, VideoResult,
    CourseOrchestrator, VideoSpec, CourseResult,
    StreamingOrchestrator,
    TranslateProcessor, TranslateNodeConfig,
    SrtSource, WhisperXSource, PushQueueSource,
    Workspace, JsonFileStore,
)

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

# StreamingOrchestrator — feed segments incrementally (browser-plugin scenario)
# CourseOrchestrator — batch-translate many videos with bounded concurrency

# Configuration (Pydantic v2 with `extra="forbid"`):
cfg = AppConfig.load("app.yaml")            # file
cfg = AppConfig.from_yaml("engines:\n ...")  # string
cfg = AppConfig.from_dict({...})            # dict
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
