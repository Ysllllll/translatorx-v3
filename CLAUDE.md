# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run all tests
pytest tests/ -v

# Run text_ops tests for a specific language
pytest tests/text_ops_tests/test_chinese.py -v
pytest tests/text_ops_tests/test_english.py -v

# Run reader tests
pytest tests/subtitle/readers/test_srt.py -v

# Run via the venv explicitly
/home/ysl/workspace/.venv/bin/pytest tests/ -v
```

## Architecture

A subtitle processing toolkit with two top-level packages under `src/`, each with its own test directory.

### Module layout

```
src/
├── subtitle/                    # Subtitle data structures and file I/O
│   ├── __init__.py              # Exports Word, Segment, SentenceRecord
│   ├── _types.py                # Core dataclasses
│   └── readers/
│       ├── __init__.py          # Exports parse_srt, read_srt
│       └── srt.py               # SRT file parser → list[Segment]
└── text_ops/                    # Multilingual text processing
    ├── __init__.py              # Public API: TextOps, MultilingualText, normalize_language
    ├── en_type.py               # EnTypeMechanism (shared by 7 space-delimited languages)
    ├── chinese.py               # ChineseMechanism (jieba)
    ├── japanese.py              # JapaneseMechanism (MeCab)
    ├── korean.py                # KoreanMechanism (Kiwi)
    └── _core/
        ├── _mechanism.py         # TextOps factory, MultilingualText
        ├── _cjk_common.py        # _BaseCjkMechanism, token parsing, attachment, join logic
        ├── _chars.py             # Unicode character classification (CJK, hangul, kana, punctuation)
        ├── _mode.py              # Mode normalization ("c"→"character", "w"→"word")
        ├── _normalize.py         # Language code normalization (aliases → ISO codes)
        ├── _availability.py      # Optional dependency checks (jieba, mecab, kiwi)
        └── _types.py             # AnalysisUnit dataclass
```

### Test structure

```
tests/
├── __init__.py
├── text_ops_tests/              # Named to avoid collision with src/text_ops/
│   ├── __init__.py
│   ├── _base.py                 # TextOpsTestCase base class with shared assertion helpers
│   ├── conftest.py              # Font path resolution, pixel length calculation
│   ├── test_english.py          # One test file per language (10 total)
│   ├── test_chinese.py
│   ├── ...
│   └── _core/
│       ├── test_mechanism.py    # Factory-level tests (unsupported language)
│       └── test_normalize.py    # Language code normalization tests
└── subtitle/
    └── readers/
        └── test_srt.py          # SRT reader integration tests
```

**Import note:** Tests import `from text_ops import ...` (not `from subtitle.text_ops`). The test directory is named `text_ops_tests` to prevent Python from finding it instead of `src/text_ops/`.

### Language families

Two fundamentally different text processing strategies:

- **EnType** (`en_type.py`): Space-delimited languages (en, ru, es, fr, de, pt, vi). `split()` uses `str.split()`. `normalize()` fixes punctuation spacing (French has special rules: adds space before `!?;:`).
- **CJK** (`_core/_cjk_common.py` base + `chinese.py`/`japanese.py`/`korean.py`): Character-based languages. `split()` uses external tokenizers (jieba/MeCab/Kiwi). `normalize()` is currently identity (no-op). Korean overrides `split()` and `join()` to preserve eojeol (space-separated word group) boundaries.

### Factory pattern

```
TextOps.for_language(code)  → returns EnTypeMechanism | ChineseMechanism | JapaneseMechanism | KoreanMechanism
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

### Text operations (`text_ops`)

- `TextOps.for_language(code)` — factory, returns language-specific mechanism
- `mechanism.split(text, mode, attach_punctuation)` — tokenize; modes: `"word"`, `"character"` (shorthands: `"w"`, `"c"`)
- `mechanism.join(tokens)` — rejoin tokens to string
- `mechanism.length(text, cjk_width)` — character/token count; `cjk_width=2` normalizes CJK width
- `mechanism.plength(text, font_path, font_size)` — pixel width via Pillow
- `mechanism.normalize(text)` — fix punctuation spacing drift
- `mechanism.strip/lstrip/rstrip(text, chars)` — thin wrappers around Python `str` methods
- `mechanism.strip_punc/lstrip_punc/rstrip_punc(text)` — strip punctuation characters
- `mechanism.restore_punc(text_a, text_b)` — apply punctuation from text_b onto text_a's content by token alignment
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
