# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run all tests
pytest tests/ -v

# Run a single test file
pytest tests/lang_ops_tests/test_chinese.py -v
pytest tests/lang_ops_tests/splitter/test_en.py -v
pytest tests/subtitle/builder_tests/test_en.py -v

# Run via the venv explicitly (if pytest not on PATH)
/home/ysl/workspace/.venv/bin/pytest tests/ -v

# Run with coverage
/home/ysl/workspace/.venv/bin/pytest tests/ -v --cov=src --cov-report=term-missing
```

`pyproject.toml` sets `pythonpath = ["src"]` so tests resolve `lang_ops` and `subtitle` from `src/`.

## Architecture

A subtitle processing toolkit with two top-level packages under `src/`.

### Package overview

```
src/
├── lang_ops/                        # Language-adapted text operations
│   ├── __init__.py                  # Public API: LangOps, ChunkPipeline, normalize_language
│   ├── en_type.py                   # EnTypeOps (shared by 7 space-delimited languages)
│   ├── chinese.py / japanese.py / korean.py  # CJK language ops
│   ├── _core/
│   │   ├── _base_ops.py             # _BaseOps ABC — abstract interface + shared concrete methods
│   │   ├── _mechanism.py            # LangOps factory (cached via lru_cache)
│   │   ├── _cjk_common.py           # _BaseCjkOps + token parsing/attachment/join helpers
│   │   ├── _chars.py                # Unicode classification + punctuation frozensets
│   │   ├── _normalize.py            # Language code normalization
│   │   └── _availability.py         # Optional dependency guards (jieba/mecab/kiwi)
│   └── splitter/
│       ├── _pipeline.py             # ChunkPipeline (immutable, chainable)
│       ├── _boundary.py             # Token-based boundary detection (sentences + clauses)
│       └── _length.py               # Length-based splitter (uses Protocol for decoupling)
└── subtitle/                        # Subtitle data structures + word timing + segment building
    ├── __init__.py                  # Exports Word, Segment, SentenceRecord, SegmentBuilder, etc.
    ├── _types.py                    # Frozen dataclasses (Word, Segment, SentenceRecord)
    ├── words.py                     # Word timing: fill_words, find_words, distribute_words, align_segments
    ├── builder.py                   # SegmentBuilder — chainable segment restructuring + streaming
    └── readers/
        └── srt.py                   # SRT file parser
```

### Key design decisions

**Factory pattern:** `LangOps.for_language(code)` returns a cached `_BaseOps` subclass. Uses `functools.lru_cache` — thread-safe, no manual cache management.

**Two language families:**
- **EnType** (`en_type.py`): Space-delimited languages (en, ru, es, fr, de, pt, vi). `split()` uses `str.split()`. Per-language abbreviation sets. French `normalize()` has special spacing rules.
- **CJK** (`_cjk_common.py` base): Character-based. `split()` uses external tokenizers (jieba/MeCab/Kiwi). Korean overrides `split()`/`join()` to preserve eojeol boundaries. CJK terminators include both full-width and half-width punctuation (e.g. `"。", "！", "？", "!", "?"`).

**strip_spaces property:** `_BaseOps.strip_spaces` controls whether `split_sentences`/`split_clauses` strip leading spaces from chunks. Defaults to `self.is_cjk` (True for Chinese/Japanese, since CJK doesn't use inter-sentence spaces). Korean overrides to `False` because it uses spaces between eojeols.

**Immutability:** `ChunkPipeline` and `SegmentBuilder` return new instances per step. All `subtitle` dataclasses use `frozen=True`. `words.py` and `builder.py` use `dataclasses.replace()` instead of mutation.

**Protocol decoupling:** `_length.py` defines `_HasSplitJoin` Protocol instead of importing `_BaseOps`, keeping the splitter independent from the ops layer.

**Token-based boundary detection:** `_boundary.py` unifies sentence and clause splitting via `find_boundaries()` / `split_tokens_by_boundaries()`. Sentence splitting uses token-level boundary markers (terminators, abbreviations, ellipsis guards). Clause splitting (`split_clauses`) is sentence-aware — it splits at clause separators and sentence boundaries in one pass.

### Layer relationship

```
lang_ops                              ←  subtitle
  token: split/join/length/normalize       _types (frozen dataclasses)
  segment: sentences/clauses               words (fill/find/distribute/align)
  pipeline: ChunkPipeline                  builder (SegmentBuilder, _StreamBuilder)
  shortcuts: ops.split_sentences() etc.    readers (SRT)
```

`subtitle` is independent of `lang_ops` except `ChunkPipeline.segments()` (deferred import of `subtitle.words.align_segments`) and `SegmentBuilder` which takes an `ops` parameter.

### Test structure

```
tests/
├── lang_ops_tests/              # Token + splitter tests
│   ├── _base.py                 # TextOpsTestCase — shared assertion helpers
│   ├── conftest.py              # Font path resolution, pixel length fixture
│   ├── test_{language}.py       # Per-language token-level tests (10 files)
│   ├── splitter/
│   │   ├── _base.py             # SplitterTestBase — reconstruction assertions
│   │   └── test_{lang}.py       # Per-language splitter tests
│   └── _core/
│       ├── test_mechanism.py    # Factory tests
│       └── test_normalize.py    # Language code normalization
└── subtitle/
    ├── test_words.py            # fill_words, find_words, distribute_words, align_segments
    ├── test_types.py            # Data type display/pretty tests
    ├── builder_tests/           # SegmentBuilder tests
    │   ├── _base.py             # BuilderTestBase
    │   ├── test_en.py
    │   └── test_zh.py
    └── readers/
        └── test_srt.py          # SRT parser tests
```

Test directory is `lang_ops_tests` (not `lang_ops`) to prevent Python from importing it instead of `src/lang_ops`.

## Dependencies

- **Python 3.10+** (`list[list]`, `str | None`, `slots=True`, `frozen=True`)
- **Pillow** — pixel length via `plength()`
- **jieba** / **MeCab** / **kiwipiepy** — CJK tokenizers (conditional, tests skip if missing)

Check availability at runtime: `jieba_is_available()`, `mecab_is_available()`, `kiwi_is_available()` (exported from `lang_ops`).

## API quick reference

### Language operations

```
ops = LangOps.for_language("en")     # Factory — cached, returns _BaseOps subclass

# Token-level
ops.split(text, mode="word")         # "word" | "character" ("w" | "c")
ops.join(tokens)
ops.length(text, cjk_width=1)
ops.normalize(text)
ops.restore_punc(text_a, text_b)

# Segment-level shortcuts
ops.split_sentences(text) → list[str]
ops.split_clauses(text)   → list[str]   # sentence-aware (splits at sentence boundaries too)
ops.split_by_length(text, max_length) → list[str]
ops.chunk(text) → ChunkPipeline
```

### Pipeline (chainable, immutable)

```
ops.chunk(text)
  .sentences()
  .clauses()            # sentence-aware
  .by_length(50)        # token-boundary aware, uses ops.length()
  .result()             → list[str]
  .segments(words)      → list[Segment]   # deferred import from subtitle.words
```

### SegmentBuilder (chainable, immutable)

```
from subtitle import SegmentBuilder

builder = SegmentBuilder(segments, ops, split_by_speaker=False)
builder.sentences()                    → SegmentBuilder
builder.clauses()                      → SegmentBuilder  # sentence-aware
builder.by_length(40)                  → SegmentBuilder
builder.merge(60)                      → SegmentBuilder  # greedy merge within groups
builder.build()                        → list[Segment]
builder.records(max_length=40)         → list[SentenceRecord]

# Streaming mode
stream = SegmentBuilder.stream(ops)
done = stream.feed(segment)            → list[Segment]  # completed sentences
remaining = stream.flush()             → list[Segment]
```

Group boundaries (set by `sentences()`) are respected by `merge()` — sentences never merge across boundaries.

### Subtitle word timing

```
fill_words(segment, split_fn=None) → Segment    # populate segment.words from text
find_words(words, sub_text, start=0) → (start_idx, end_idx)
distribute_words(words, texts) → list[list[Word]]
align_segments(chunks, words) → list[Segment]    # text chunks + timed words → Segments
```

### Data types (all frozen)

- `Word(word, start, end, speaker=None, extra={})`
- `Segment(start, end, text, speaker=None, words=[], extra={})`
- `SentenceRecord(src_text, start, end, segments=[], ...)` — also has `chunk_cache`, `translations`, `alignment`

## Fonts

Pixel-length tests require system fonts. `conftest.py` tries in order:
1. `/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc`
2. `/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf`
3. `/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf`

Raises `SkipTest` if none found.
