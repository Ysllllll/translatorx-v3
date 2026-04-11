# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run all tests
pytest tests/ -v

# Run lang_ops tests for a specific language
pytest tests/lang_ops_tests/test_chinese.py -v
pytest tests/lang_ops_tests/test_english.py -v

# Run splitter tests
pytest tests/lang_ops_tests/test_sentence.py -v
pytest tests/lang_ops_tests/test_clause.py -v
pytest tests/lang_ops_tests/test_pipeline.py -v
pytest tests/lang_ops_tests/test_chunk_shortcuts.py -v

# Run per-language long-text tests
pytest tests/lang_ops_tests/test_split_en.py -v
pytest tests/lang_ops_tests/test_split_zh.py -v

# Run reader tests
pytest tests/subtitle/readers/test_srt.py -v

# Run via the venv explicitly
/home/ysl/workspace/.venv/bin/pytest tests/ -v
```

## Architecture

A subtitle processing toolkit with two top-level packages under `src/`.

### Module layout

```
src/
├── lang_ops/                        # Language-adapted text operations (token + segment)
│   ├── __init__.py                  # Public API: TextOps, MultilingualText, ChunkPipeline, normalize_language
│   ├── en_type.py                   # EnTypeOps (shared by 7 space-delimited languages)
│   ├── chinese.py                   # ChineseOps (jieba)
│   ├── japanese.py                  # JapaneseOps (MeCab)
│   ├── korean.py                    # KoreanOps (Kiwi)
│   ├── _core/
│   │   ├── _mechanism.py            # TextOps factory, MultilingualText
│   │   ├── _cjk_common.py           # _BaseCjkOps, token parsing, attachment, join logic
│   │   ├── _chars.py                # Unicode character classification (CJK, hangul, kana, punctuation)
│   │   ├── _mode.py                 # Mode normalization ("c"→"character", "w"→"word")
│   │   ├── _normalize.py            # Language code normalization (aliases → ISO codes)
│   │   ├── _availability.py         # Optional dependency checks (jieba, mecab, kiwi)
│   │   └── _types.py                # AnalysisUnit dataclass
│   └── splitter/                    # Text splitting pipeline
│       ├── __init__.py              # Exports ChunkPipeline
│       ├── _pipeline.py             # ChunkPipeline class (immutable, chainable)
│       ├── _paragraph.py            # Paragraph splitter
│       ├── _sentence.py             # Sentence splitter (abbreviation/ellipsis guards)
│       ├── _clause.py               # Clause splitter
│       └── _length.py               # Length-based splitter
└── subtitle/                        # Subtitle data structures and file I/O
    ├── __init__.py                  # Exports Word, Segment, SentenceRecord
    ├── _types.py                    # Core dataclasses
    └── readers/
        ├── __init__.py              # Exports parse_srt, read_srt
        └── srt.py                   # SRT file parser → list[Segment]
```

### Test structure

```
tests/
├── __init__.py
├── lang_ops_tests/              # All lang_ops tests (token + segment)
│   ├── __init__.py
│   ├── _base.py                 # TextOpsTestCase base class with shared assertion helpers
│   ├── conftest.py              # Font path resolution, pixel length calculation
│   ├── test_english.py          # Token-level tests per language (10 total)
│   ├── test_chinese.py
│   ├── ...
│   ├── test_paragraph.py        # Paragraph splitting
│   ├── test_sentence.py         # Sentence splitting (abbreviations, ellipsis, quotes)
│   ├── test_clause.py           # Clause splitting
│   ├── test_length.py           # Length-based splitting
│   ├── test_pipeline.py         # ChunkPipeline chaining, immutability
│   ├── test_chunk_shortcuts.py  # Shortcut methods (split_sentences, split_clauses, chunk)
│   ├── test_split_en.py         # Per-language long-text tests (10 total)
│   ├── test_split_zh.py
│   ├── ...
│   └── _core/
│       ├── test_mechanism.py    # Factory-level tests (unsupported language)
│       └── test_normalize.py    # Language code normalization tests
└── subtitle/
    └── readers/
        └── test_srt.py          # SRT reader integration tests
```

**Import note:** All tests import `from lang_ops import ...`. The test directory is named `lang_ops_tests` to prevent Python from finding it instead of the `src/` package.

### Layer relationship

```
lang_ops                              ←  subtitle
  token: split/join/length/normalize       readers
  segment: sentences/clauses/paragraphs
  pipeline: ChunkPipeline
  shortcuts: ops.split_sentences() etc.
```

`subtitle` is independent of `lang_ops`.

### Language families

Two fundamentally different token-level processing strategies:

- **EnType** (`en_type.py`): Space-delimited languages (en, ru, es, fr, de, pt, vi). `split()` uses `str.split()`. `normalize()` fixes punctuation spacing (French has special rules: adds space before `!?;:`). Per-language abbreviation sets for sentence splitting.
- **CJK** (`_core/_cjk_common.py` base + `chinese.py`/`japanese.py`/`korean.py`): Character-based languages. `split()` uses external tokenizers (jieba/MeCab/Kiwi). `normalize()` is currently identity (no-op). Korean overrides `split()` and `join()` to preserve eojeol (space-separated word group) boundaries. Each language has its own `sentence_terminators` and `clause_separators`.

### Factory pattern

```
TextOps.for_language(code)  → returns EnTypeOps | ChineseOps | JapaneseOps | KoreanOps
```

`_core/_mechanism.py` defines `TextOps` (factory) and `MultilingualText` (convenience wrapper). Results are cached by normalized language code.

## Dependencies

- **Python 3.10+** (uses `list[list]`, `str | None`, `slots=True`)
- **Pillow** — pixel length via `plength()`
- **jieba** — Chinese segmentation (conditional)
- **MeCab** — Japanese morphological analysis (conditional)
- **kiwipiepy** — Korean morphological analysis (conditional)

CJK tests guard on availability and skip gracefully if the tokenizer is not installed.

## API

### Data types (`subtitle._types`)

- `Word(word, start, end, speaker=None, extra={})` — single word with timing
- `Segment(start, end, text, speaker=None, words=[], extra={})` — subtitle segment
- `SentenceRecord(src_text, start, end, segments=[], chunk_cache={}, translations={}, alignment={}, extra={})` — sentence with translations

### Language operations (`lang_ops`)

- `TextOps.for_language(code)` — factory, returns language-specific mechanism

**Token-level:**
- `mechanism.split(text, mode, attach_punctuation)` — tokenize; modes: `"word"`, `"character"` (shorthands: `"w"`, `"c"`)
- `mechanism.join(tokens)` — rejoin tokens to string
- `mechanism.length(text, cjk_width)` — character/token count; `cjk_width=2` normalizes CJK width
- `mechanism.plength(text, font_path, font_size)` — pixel width via Pillow
- `mechanism.normalize(text)` — fix punctuation spacing drift
- `mechanism.strip/lstrip/rstrip(text, chars)` — thin wrappers around Python `str` methods
- `mechanism.strip_punc/lstrip_punc/rstrip_punc(text)` — strip punctuation characters
- `mechanism.restore_punc(text_a, text_b)` — apply punctuation from text_b onto text_a's content by token alignment

**Segment-level shortcuts:**
- `mechanism.split_sentences(text)` → `list[str]` — split by terminal punctuation
- `mechanism.split_clauses(text)` → `list[str]` — split by comma/pause punctuation
- `mechanism.split_paragraphs(text)` → `list[str]` — split by blank lines
- `mechanism.chunk(text)` → `ChunkPipeline` — create a chainable pipeline

**Pipeline (chainable):**
- `ChunkPipeline(text, language="en")` or `ops.chunk(text)` — immutable, chainable pipeline
- `.paragraphs()` — split by blank lines
- `.sentences()` — split by terminal punctuation (abbreviation/ellipsis aware)
- `.clauses()` — split by comma/pause punctuation
- `.by_length(max_length, unit="character")` — split at token boundaries by length
- `.result()` → `list[str]`

Each pipeline method returns a **new** `ChunkPipeline` instance (immutable).

**Other:**
- `MultilingualText(text, language)` — convenience wrapper
- `normalize_language(value)` — normalize aliases ("中文" → "zh", "english" → "en")

### Readers (`subtitle.readers`)

- `parse_srt(content: str)` — parse SRT content string → `list[Segment]`
- `read_srt(path)` — read SRT file → `list[Segment]`

## Fonts

Pixel-length tests require system fonts. `conftest.py` tries in order:
1. `/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc`
2. `/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf`
3. `/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf`

Raises `SkipTest` if none found.
