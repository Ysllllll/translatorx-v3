# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run all tests
pytest tests/ -v

# Run a single test file
pytest tests/lang_ops_tests/test_chinese.py -v
pytest tests/lang_ops_tests/chunk/test_en.py -v
pytest tests/subtitle/build_tests/test_en.py -v

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
│   │   ├── _chars.py                # Unicode classification (CJK/Hangul/Kana detection)
│   │   ├── _punctuation.py          # Single source of truth for all punctuation constants
│   │   ├── _normalize.py            # Language code normalization
│   │   └── _availability.py         # Optional dependency guards (jieba/mecab/kiwi)
│   └── chunk/
│       ├── _pipeline.py             # ChunkPipeline (immutable, chainable)
│       ├── _boundary.py             # Token-based boundary detection (sentences + clauses)
│       ├── _length.py               # Length-based splitting (uses Protocol for decoupling)
│       └── _merge.py                # Length-based merging (inverse of splitting)
└── subtitle/                        # Subtitle data structures + timing alignment + segment building
    ├── __init__.py                  # Exports data types + Subtitle/Stream + alignment utilities (fill/find/distribute/align)
    ├── model.py                     # Frozen dataclasses (Word, Segment, SentenceRecord)
    ├── align.py                     # Word timing: fill_words, find_words, distribute_words, align_segments
    ├── core.py                      # Subtitle — chainable segment restructuring
    └── io/
        └── srt.py                   # SRT file parser
```

### Key design decisions

**Factory pattern:** `LangOps.for_language(code)` returns a cached `_BaseOps` subclass. Uses `functools.lru_cache` — thread-safe, no manual cache management.

**Two language families:**
- **EnType** (`en_type.py`): Space-delimited languages (en, ru, es, fr, de, pt, vi). `split()` uses `str.split()`. Per-language abbreviation sets. French `normalize()` has special spacing rules.
- **CJK** (`_cjk_common.py` base): Character-based. `split()` uses external tokenizers (jieba/MeCab/Kiwi). Korean overrides `split()`/`join()` to preserve eojeol boundaries. CJK terminators include both full-width and half-width punctuation (e.g. `"。", "！", "？", "!", "?"`).

**strip_spaces property:** `_BaseOps.strip_spaces` controls whether `split_sentences`/`split_clauses` strip leading spaces from chunks. Defaults to `self.is_cjk` (True for Chinese/Japanese, since CJK doesn't use inter-sentence spaces). Korean overrides to `False` because it uses spaces between eojeols.

**Immutability:** `ChunkPipeline` and `Subtitle` return new instances per step. All `subtitle` dataclasses use `frozen=True`. `align.py` uses `dataclasses.replace()` instead of mutation.

**Protocol decoupling:** `_length.py` and `_merge.py` define Protocol types instead of importing `_BaseOps`, keeping the chunk package independent from the ops layer.

**Token-based boundary detection:** `_boundary.py` unifies sentence and clause splitting via `find_boundaries()` / `split_tokens_by_boundaries()`. Sentence splitting uses token-level boundary markers (terminators, abbreviations, ellipsis guards). Clause splitting (`split_clauses`) is sentence-aware — it splits at clause separators and sentence boundaries in one pass.

### Layer relationship

```
lang_ops                              ←  subtitle
  token: split/join/length/normalize       _types (frozen dataclasses)
  segment: sentences/clauses               words (fill/find/distribute/align)
  pipeline: ChunkPipeline                  core (Subtitle, SubtitleStream)
  shortcuts: ops.split_sentences() etc.    readers (SRT)
```

`subtitle` is independent of `lang_ops` except `ChunkPipeline.segments()` (deferred import of `subtitle.align.align_segments`) and `Subtitle` which takes an `ops` or `language` parameter.

### Test structure

```
tests/
├── lang_ops_tests/              # Token + chunk tests
│   ├── _base.py                 # TextOpsTestCase — shared assertion helpers
│   ├── conftest.py              # Font path resolution, pixel length fixture
│   ├── test_{language}.py       # Per-language token-level tests (10 files)
│   ├── chunk/
│   │   ├── _base.py             # SplitterTestBase — reconstruction assertions
│   │   └── test_{lang}.py       # Per-language chunk tests
│   └── _core/
│       ├── test_mechanism.py    # Factory tests
│       ├── test_normalize.py    # Language code normalization
│       └── test_punctuation.py  # Punctuation constants tests
└── subtitle/
    ├── align_tests/             # Word timing tests
    │   ├── test_align.py        # align_segments
    │   ├── test_attach_punct.py # attach_punct_words
    │   ├── test_distribute.py   # distribute_words
    │   ├── test_fill.py         # fill_words
    │   ├── test_find.py         # find_words
    │   ├── test_normalize.py    # normalize_words
    │   └── test_pipeline.py     # Pipeline integration
    ├── test_model.py            # Data type display/pretty tests
    ├── build_tests/             # Subtitle tests
    │   ├── _base.py             # BuilderTestBase
    │   ├── test_en.py
    │   ├── test_ko.py
    │   └── test_zh.py
    └── io_tests/
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
ops.split_by_length(text, max_len) → list[str]
ops.merge_by_length(chunks, max_len) → list[str]  # greedy merge (inverse of split)
ops.chunk(text) → ChunkPipeline
```

### Pipeline (chainable, immutable)

```
ops.chunk(text)
  .sentences()
  .clauses(merge_under=60)  # sentence-aware; merge_under merges back short clauses
  .split(max_len=50)        # split by length
  .merge(max_len=80)        # greedy merge all adjacent chunks
  .apply(fn, skip_if=None)  # external fn: list[str] → list[list[str]]; skip_if skips chunks
  .result()                 → list[str]
  .segments(words)          → list[Segment]   # deferred import from subtitle.align

# Alternative construction
ChunkPipeline.from_chunks(chunks, ops)  # from pre-split chunk list
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
sub.apply(fn, cache, batch_size, workers, skip_if)  → Subtitle  # batched across all pipelines
sub.build()                            → list[Segment]
sub.records()                          → list[SentenceRecord]

# Streaming mode (split_by_speaker=True groups by speaker)
stream = Subtitle.stream(language="zh")
done = stream.feed(segment)            → list[Segment]  # completed sentences
remaining = stream.flush()             → list[Segment]
```

After `sentences()`, each operation is implicitly per-sentence — it never crosses sentence boundaries. This replaces the old `_parent_ids` mechanism with structural isolation (separate pipeline instances per sentence).

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
- `SentenceRecord(src_text, start, end, segments=[], ...)` — also has `chunk_cache`, `translations`, `alignment`

### SRT reader

```
from subtitle.io import parse_srt, read_srt

parse_srt(content) → list[Segment]   # parse SRT string
read_srt(path)     → list[Segment]   # parse SRT file
```

## Fonts

Pixel-length tests require system fonts. `conftest.py` tries in order:
1. `/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc`
2. `/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf`
3. `/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf`

Raises `SkipTest` if none found.
