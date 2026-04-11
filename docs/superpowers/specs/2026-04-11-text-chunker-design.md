# Text Chunker Design

A pipeline-based text chunking module that provides multi-granularity text splitting, built on top of `text_ops`.

## Overview

`text_chunker` is an independent package under `src/` that depends on `text_ops` for language-specific text operations. It supports four splitting granularities — paragraphs, sentences, clauses, and length-based — with automatic language adaptation.

## API

```python
from text_chunker import ChunkPipeline

# Single step
sentences = ChunkPipeline(text, language="en").sentences().result()

# Chained: paragraphs -> sentences -> clauses
result = (ChunkPipeline(text, language="zh")
    .paragraphs()
    .sentences()
    .clauses()
    .result())

# Length-based splitting
chunks = (ChunkPipeline(text, language="fr")
    .sentences()
    .by_length(max_length=50, unit="word")
    .result())
```

### Methods

| Method | Description |
|--------|-------------|
| `ChunkPipeline(text, language)` | Constructor. Binds text and language. Internally calls `TextOps.for_language()`. |
| `.paragraphs()` | Split by blank lines. Language-independent. |
| `.sentences()` | Split by sentence-terminal punctuation. Language-adapted. |
| `.clauses()` | Split by comma/pause punctuation. Language-adapted. |
| `.by_length(max_length, unit="word")` | Split at nearest punctuation/space boundary when length exceeds limit. `unit`: `"word"` or `"character"`. |
| `.result()` | Return `list[str]`. |

### Immutability

Each method returns a **new** `ChunkPipeline` instance. The original is never modified.

## File Structure

```
src/text_chunker/
├── __init__.py              # Exports ChunkPipeline
├── _pipeline.py             # ChunkPipeline class, orchestrates call chain
└── _splitters/
    ├── __init__.py
    ├── _paragraph.py        # split_paragraphs(text) -> list[str]
    ├── _sentence.py         # split_sentences(text, ops, config) -> list[str]
    ├── _clause.py           # split_clauses(text, ops, config) -> list[str]
    └── _length.py           # split_by_length(text, ops, max_length, unit) -> list[str]

src/text_chunker/_lang_config.py  # Per-language punctuation sets
```

## Data Flow

```
ChunkPipeline("一篇文章...", "zh")
  │
  ├─ Internal state: pieces = ["一篇文章..."]   (single-element list)
  │
  ├─ .sentences()
  │   For each piece, call split_sentences(piece, ops, config)
  │   Flatten results → ["第一句。", "第二句。", "第三句。"]
  │
  ├─ .clauses()
  │   For each piece, call split_clauses(piece, ops, config)
  │   Flatten results → ["第一句前半，", "第一句后半。", ...]
  │
  └─ .result() → list[str]
```

## Language Configuration (`_lang_config.py`)

### Sentence Terminal Punctuation

```python
SENTENCE_TERMINALS = {
    "default": {".", "!", "?"},
    "zh": {"。", "！", "？"},
    "ja": {"。", "！", "？"},
    "ko": {"。", "!", "?"},
}
```

Languages not listed fall back to `"default"`: `en`, `ru`, `es`, `fr`, `de`, `pt`, `vi`.

### Clause Separator Punctuation

```python
CLAUSE_SEPARATORS = {
    "default": {",", ";", ":", "—"},
    "zh": {"，", "、", "；", "："},
    "ja": {"、", "；"},
    "ko": {",", "；"},
}
```

### English Abbreviation Whitelist

```python
ABBREVIATIONS = frozenset({
    "Mr", "Mrs", "Ms", "Dr", "Prof", "Sr", "Jr", "St",
    "Inc", "Ltd", "Co", "Corp", "vs", "etc", "eg", "ie",
    "Jan", "Feb", "Mar", "Apr", "Jun", "Jul", "Aug", "Sep",
    "Oct", "Nov", "Dec",
})
```

## Splitter Logic

### Paragraph Splitter (`_paragraph.py`)

- Split on consecutive blank lines (`\n\n` or `\r\n\r\n`)
- Trim leading/trailing whitespace per paragraph
- Discard empty paragraphs
- Language-independent

### Sentence Splitter (`_sentence.py`)

**English/default rules:**

1. Scan for sentence-terminal punctuation (`.`, `!`, `?`)
2. Before splitting at `.`, check the preceding token:
   - If it's in `ABBREVIATIONS` → don't split
   - If it's a number (`3.14`) → don't split
   - If it's a date (`2024.01.15`) → don't split
3. `...` (ellipsis) is kept as a unit, never split
4. Closing quote after terminal: `."` or `?"` → split after the quote

**CJK rules:**

- Split directly after `。！？`
- `……` (CJK ellipsis) is kept as a unit, never split
- No abbreviation handling needed

**General:**

- Terminal punctuation stays with the current sentence (not discarded)
- Empty / pure-whitespace results are discarded

### Clause Splitter (`_clause.py`)

- Split at `CLAUSE_SEPARATORS` for the given language
- Separator stays with the preceding clause
- Simpler than sentence splitting — no abbreviation or ellipsis edge cases

### Length Splitter (`_length.py`)

1. Tokenize with `ops.split(text)`
2. Accumulate tokens, checking `ops.length()` after each addition
3. When length exceeds `max_length`, find the nearest separator (punctuation or space) before the overflow point
4. Split there, start a new chunk
5. `unit="word"` counts tokens; `unit="character"` counts characters
6. Never split mid-word — fall back to the previous boundary if no good break point exists

## Error Handling

| Scenario | Behavior |
|----------|----------|
| Unsupported language | `ValueError` propagated from `TextOps.for_language()` |
| Empty string input | `.result()` returns `[]` |
| Pure punctuation text | Returns `["!!!"]`, not discarded |
| Consecutive blank lines | `paragraphs()` discards empty paragraphs |
| Single sentence (no terminators) | `sentences()` returns `["the text"]` as-is |
| No good break point for length | Splits at `max_length` boundary (hard break) |

## Testing

```
tests/text_chunker_tests/
├── __init__.py
├── _base.py              # Shared test utilities
├── test_pipeline.py      # Pipeline chaining, immutability
├── test_paragraph.py     # Paragraph splitting
├── test_sentence.py      # Sentence splitting (abbreviations, ellipsis, quotes)
├── test_clause.py        # Clause splitting
└── test_length.py        # Length-based splitting
```

### Key Test Cases

- **Sentence**: English abbreviations (`Dr. Smith went home.` → not split at `Dr.`), ellipsis (`Wait... go on.`), CJK terminals, mixed script text, quotes
- **Clause**: consecutive commas, CJK dun-hao (`、`), semicolons
- **Length**: exactly `max_length`, over `max_length` with no break point, CJK full-width counting
- **Pipeline**: chaining order, immutability (original pipeline unchanged after calls), empty input passthrough, multiple chain combinations
- **Multi-language**: smoke test for each supported language

## Dependencies

- `text_ops` (same repository, `src/text_ops/`) — for `TextOps.for_language()`, `split()`, `join()`, `length()`, punctuation sets
- No external dependencies beyond what `text_ops` already requires
