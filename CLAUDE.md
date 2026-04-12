# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run all tests
pytest tests/ -v

# Run a single test file
pytest tests/lang_ops_tests/test_chinese.py -v
pytest tests/lang_ops_tests/splitter/test_en.py -v

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
│   ├── __init__.py                  # Public API: TextOps, ChunkPipeline, Span, normalize_language
│   ├── en_type.py                   # EnTypeOps (shared by 7 space-delimited languages)
│   ├── chinese.py / japanese.py / korean.py  # CJK language ops
│   ├── _core/
│   │   ├── _base_ops.py             # _BaseOps ABC — abstract interface + shared concrete methods
│   │   ├── _mechanism.py            # TextOps factory (cached via lru_cache)
│   │   ├── _cjk_common.py           # _BaseCjkOps + token parsing/attachment/join helpers
│   │   ├── _chars.py                # Unicode classification + punctuation frozensets
│   │   ├── _normalize.py            # Language code normalization
│   │   ├── _availability.py         # Optional dependency guards
│   │   └── _types.py                # Span dataclass
│   └── splitter/
│       ├── _pipeline.py             # ChunkPipeline (immutable, chainable)
│       ├── _sentence.py             # Sentence splitter (abbreviation/ellipsis guards)
│       ├── _clause.py               # split_clauses + split_clauses_full (sentence-aware)
│       ├── _paragraph.py            # Paragraph splitter
│       └── _length.py               # Length-based splitter (uses Protocol for decoupling)
└── subtitle/                        # Subtitle data structures + word timing
    ├── __init__.py                  # Exports Word, Segment, SentenceRecord, fill_words, find_words, distribute_words, align_segments
    ├── _types.py                    # Frozen dataclasses (Word, Segment, SentenceRecord)
    ├── words.py                     # Word timing: fill_words, find_words, distribute_words, align_segments
    └── readers/
        └── srt.py                   # SRT file parser
```

### Key design decisions

**Factory pattern:** `TextOps.for_language(code)` returns a cached `_BaseOps` subclass. Uses `functools.lru_cache` — thread-safe, no manual cache management.

**Two language families:**
- **EnType** (`en_type.py`): Space-delimited languages (en, ru, es, fr, de, pt, vi). `split()` uses `str.split()`. Per-language abbreviation sets. French `normalize()` has special spacing rules.
- **CJK** (`_cjk_common.py` base): Character-based. `split()` uses external tokenizers (jieba/MeCab/Kiwi). Korean overrides `split()`/`join()` to preserve eojeol boundaries. CJK terminators include both full-width and half-width punctuation (e.g. `"。", "！", "？", "!", "?"`).

**strip_spaces property:** `_BaseOps.strip_spaces` controls whether `split_sentences`/`split_clauses_full` strip leading spaces from chunks. Defaults to `self.is_cjk` (True for Chinese/Japanese, since CJK doesn't use inter-sentence spaces). Korean overrides to `False` because it uses spaces between eojeols.

**Immutability:** `ChunkPipeline` returns new instances per step. All `subtitle` dataclasses use `frozen=True`. `words.py` uses `dataclasses.replace()` instead of mutation.

**Protocol decoupling:** `_length.py` defines `_HasSplitJoin` Protocol instead of importing `_BaseOps`, keeping the splitter independent from the ops layer.

**Sentence-aware clause splitting:** `split_clauses_full()` splits at both clause separators and sentence terminators in one pass. `split_clauses()` (lower-level) only uses clause separators. The pipeline and shortcuts use the full version.

### Layer relationship

```
lang_ops                              ←  subtitle
  token: split/join/length/normalize       _types (frozen dataclasses)
  segment: sentences/clauses/paragraphs    words (fill/find/distribute/align)
  pipeline: ChunkPipeline                  readers (SRT)
  shortcuts: ops.split_sentences() etc.
```

`subtitle` is independent of `lang_ops` except `ChunkPipeline.segments()` which does a deferred import of `subtitle.words.align_segments`.

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
    └── readers/
        └── test_srt.py          # SRT parser tests
```

Test directory is `lang_ops_tests` (not `lang_ops`) to prevent Python from importing it instead of `src/lang_ops`.

## Dependencies

- **Python 3.10+** (`list[list]`, `str | None`, `slots=True`, `frozen=True`)
- **Pillow** — pixel length via `plength()`
- **jieba** / **MeCab** / **kiwipiepy** — CJK tokenizers (conditional, tests skip if missing)

## API quick reference

### Language operations

```
ops = TextOps.for_language("en")     # Factory — cached, returns _BaseOps subclass

# Token-level
ops.split(text, mode="word")         # "word" | "character" ("w" | "c")
ops.join(tokens)
ops.length(text, cjk_width=1)
ops.normalize(text)
ops.restore_punc(text_a, text_b)

# Segment-level shortcuts
ops.split_sentences(text) → list[str]
ops.split_clauses(text)   → list[str]   # sentence-aware (splits at sentence boundaries too)
ops.split_paragraphs(text) → list[str]
ops.split_by_length(text, max_length) → list[str]
ops.chunk(text) → ChunkPipeline
```

### Pipeline (chainable, immutable)

```
ops.chunk(text)
  .paragraphs()
  .sentences()
  .clauses()            # sentence-aware
  .by_length(50)        # token-boundary aware, uses ops.length()
  .result()             → list[str]
  .spans()              → list[Span]
  .segments(words)      → list[Segment]   # deferred import from subtitle.words
```

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
- `SentenceRecord(src_text, start, end, segments=[], ...)`
- `Span(text, start, end)` — positional text fragment

## Fonts

Pixel-length tests require system fonts. `conftest.py` tries in order:
1. `/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc`
2. `/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf`
3. `/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf`

Raises `SkipTest` if none found.
