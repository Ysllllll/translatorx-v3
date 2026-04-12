# Merge text_ops into lang_ops — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Merge `text_ops/splitter` into `lang_ops` so that one package handles all language-aware text operations (token + segment).

**Architecture:** Move splitter modules into `lang_ops/splitter/`. Add shortcut methods (`split_sentences`, `split_clauses`, `split_paragraphs`, `chunk`) to the ops base classes. Delete `src/text_ops/`. Update all imports. Merge test directories.

**Tech Stack:** Python 3.10+, pytest

---

## Target API

```python
from lang_ops import TextOps, ChunkPipeline

ops = TextOps.for_language("en")

# token-level (unchanged)
tokens = ops.split("Hello, world")
text = ops.join(tokens)

# segment-level shortcuts
sentences = ops.split_sentences(text)
clauses = ops.split_clauses(text)
paragraphs = ops.split_paragraphs(text)

# pipeline (chainable)
chunks = ops.chunk(text).paragraphs().sentences().clauses().by_length(50).result()
```

## Target directory structure

```
src/lang_ops/
├── __init__.py                  # Export TextOps, normalize_language, ChunkPipeline
├── en_type.py                   # EnTypeOps (unchanged, already has properties)
├── chinese.py                   # ChineseOps (unchanged)
├── japanese.py                  # JapaneseOps (unchanged)
├── korean.py                    # KoreanOps (unchanged)
├── _core/
│   ├── __init__.py
│   ├── _mechanism.py            # TextOps factory
│   ├── _cjk_common.py           # _BaseCjkOps
│   ├── _chars.py
│   ├── _mode.py
│   ├── _normalize.py
│   ├── _availability.py
│   └── _types.py
└── splitter/                    # ← MOVED from text_ops/splitter
    ├── __init__.py              # Export ChunkPipeline
    ├── _pipeline.py             # ChunkPipeline (updated imports)
    ├── _paragraph.py            # split_paragraphs (no changes)
    ├── _sentence.py             # split_sentences (no changes)
    ├── _clause.py               # split_clauses (no changes)
    └── _length.py               # split_by_length (no changes)

tests/
├── lang_ops_tests/              # ALL tests merged here
│   ├── __init__.py
│   ├── _base.py
│   ├── conftest.py
│   ├── test_english.py ... test_korean.py   # token-level (unchanged)
│   ├── _core/
│   │   ├── test_normalize.py
│   │   └── test_mechanism.py
│   ├── test_paragraph.py        # ← MOVED
│   ├── test_sentence.py         # ← MOVED
│   ├── test_clause.py           # ← MOVED
│   ├── test_length.py           # ← MOVED
│   ├── test_pipeline.py         # ← MOVED
│   ├── test_split_en.py ... test_split_ko.py  # ← MOVED (10 files)
│   └── test_chunk_shortcuts.py  # NEW — tests for ops.split_sentences() etc.
```

### Deleted

- `src/text_ops/` — entire package
- `tests/text_ops_tests/` — entire directory

---

### Task 1: Move splitter source into lang_ops

**Files:**
- Move: `src/text_ops/splitter/_pipeline.py` → `src/lang_ops/splitter/_pipeline.py`
- Move: `src/text_ops/splitter/_paragraph.py` → `src/lang_ops/splitter/_paragraph.py`
- Move: `src/text_ops/splitter/_sentence.py` → `src/lang_ops/splitter/_sentence.py`
- Move: `src/text_ops/splitter/_clause.py` → `src/lang_ops/splitter/_clause.py`
- Move: `src/text_ops/splitter/_length.py` → `src/lang_ops/splitter/_length.py`
- Create: `src/lang_ops/splitter/__init__.py`

- [ ] **Step 1: Create lang_ops/splitter/ directory and move files**

```bash
mkdir -p src/lang_ops/splitter
cp src/text_ops/splitter/_pipeline.py src/lang_ops/splitter/_pipeline.py
cp src/text_ops/splitter/_paragraph.py src/lang_ops/splitter/_paragraph.py
cp src/text_ops/splitter/_sentence.py src/lang_ops/splitter/_sentence.py
cp src/text_ops/splitter/_clause.py src/lang_ops/splitter/_clause.py
cp src/text_ops/splitter/_length.py src/lang_ops/splitter/_length.py
```

- [ ] **Step 2: Create `src/lang_ops/splitter/__init__.py`**

```python
"""Text splitting pipeline."""

from lang_ops.splitter._pipeline import ChunkPipeline

__all__ = ["ChunkPipeline"]
```

- [ ] **Step 3: Update imports in `_pipeline.py`**

Change all `text_ops.splitter.*` imports to `lang_ops.splitter.*`:

```python
# Old:
from text_ops.splitter._paragraph import split_paragraphs
from text_ops.splitter._sentence import split_sentences
from text_ops.splitter._clause import split_clauses
from text_ops.splitter._length import split_by_length

# New:
from lang_ops.splitter._paragraph import split_paragraphs
from lang_ops.splitter._sentence import split_sentences
from lang_ops.splitter._clause import split_clauses
from lang_ops.splitter._length import split_by_length
```

The `from lang_ops import TextOps` import in `_pipeline.py` stays the same (now intra-package).

- [ ] **Step 4: Update `src/lang_ops/__init__.py`**

Add `ChunkPipeline` to exports:

```python
"""Multilingual text operations library."""

from ._core._mechanism import TextOps
from ._core._normalize import normalize_language
from ._core._availability import jieba_is_available, mecab_is_available, kiwi_is_available
from .splitter import ChunkPipeline

__all__ = [
    "TextOps",
    "normalize_language",
    "ChunkPipeline",
    "jieba_is_available",
    "mecab_is_available",
    "kiwi_is_available",
]
```

- [ ] **Step 5: Verify imports resolve**

Run: `python -c "from lang_ops import ChunkPipeline; print('OK')"`
Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add src/lang_ops/splitter/ src/lang_ops/__init__.py
git commit -m "refactor: move splitter from text_ops into lang_ops"
```

---

### Task 2: Add shortcut methods to ops classes

**Files:**
- Modify: `src/lang_ops/en_type.py` — add `split_sentences`, `split_clauses`, `split_paragraphs`, `chunk`
- Modify: `src/lang_ops/_core/_cjk_common.py` — same methods on `_BaseCjkOps`

- [ ] **Step 1: Add methods to `EnTypeOps`**

Add these methods to `EnTypeOps` in `src/lang_ops/en_type.py`:

```python
def split_sentences(self, text: str) -> list[str]:
    """Split text into sentences."""
    from lang_ops.splitter._sentence import split_sentences as _split
    return _split(text, self.sentence_terminators, self.abbreviations, is_cjk=self.is_cjk)

def split_clauses(self, text: str) -> list[str]:
    """Split text into clauses."""
    from lang_ops.splitter._clause import split_clauses as _split
    return _split(text, self.clause_separators)

def split_paragraphs(self, text: str) -> list[str]:
    """Split text into paragraphs."""
    from lang_ops.splitter._paragraph import split_paragraphs as _split
    return _split(text)

def chunk(self, text: str) -> "ChunkPipeline":
    """Create a ChunkPipeline for chainable splitting."""
    from lang_ops.splitter._pipeline import ChunkPipeline
    return ChunkPipeline(text, ops=self)
```

- [ ] **Step 2: Add same methods to `_BaseCjkOps`**

Add identical methods to `_BaseCjkOps` in `src/lang_ops/_core/_cjk_common.py`.

- [ ] **Step 3: Update `ChunkPipeline.__init__` to accept ops object directly**

Modify `src/lang_ops/splitter/_pipeline.py`:

```python
def __init__(self, text: str, *, language: str | None = None, ops: _LangOps | None = None) -> None:
    if ops is not None:
        self._ops = ops
    elif language is not None:
        from lang_ops import TextOps
        self._ops = TextOps.for_language(language)
    else:
        raise TypeError("ChunkPipeline requires either language or ops")
    self._language = getattr(self._ops, '_language', '')
    self._pieces: list[str] = [text] if text else []
```

The `from lang_ops import TextOps` import at the top of `_pipeline.py` can stay; the `TextOps` class is already imported for type resolution. But we should also import the `_LangOps` type for the annotation.

- [ ] **Step 4: Run existing tests to verify nothing broke**

Run: `python -m pytest tests/ -v --tb=short`
Expected: All tests pass (still using old test imports — they haven't moved yet).

- [ ] **Step 5: Commit**

```bash
git add src/lang_ops/en_type.py src/lang_ops/_core/_cjk_common.py src/lang_ops/splitter/_pipeline.py
git commit -m "feat: add split_sentences/split_clauses/split_paragraphs/chunk shortcuts to ops"
```

---

### Task 3: Move tests into lang_ops_tests and update imports

**Files:**
- Move 15 test files from `tests/text_ops_tests/` to `tests/lang_ops_tests/`
- Update all `from text_ops.*` imports to `from lang_ops.*`
- Delete `tests/text_ops_tests/`

- [ ] **Step 1: Move test files**

```bash
cp tests/text_ops_tests/test_paragraph.py tests/lang_ops_tests/test_paragraph.py
cp tests/text_ops_tests/test_sentence.py tests/lang_ops_tests/test_sentence.py
cp tests/text_ops_tests/test_clause.py tests/lang_ops_tests/test_clause.py
cp tests/text_ops_tests/test_length.py tests/lang_ops_tests/test_length.py
cp tests/text_ops_tests/test_pipeline.py tests/lang_ops_tests/test_pipeline.py
cp tests/text_ops_tests/test_split_en.py tests/lang_ops_tests/test_split_en.py
cp tests/text_ops_tests/test_split_es.py tests/lang_ops_tests/test_split_es.py
cp tests/text_ops_tests/test_split_pt.py tests/lang_ops_tests/test_split_pt.py
cp tests/text_ops_tests/test_split_fr.py tests/lang_ops_tests/test_split_fr.py
cp tests/text_ops_tests/test_split_de.py tests/lang_ops_tests/test_split_de.py
cp tests/text_ops_tests/test_split_ru.py tests/lang_ops_tests/test_split_ru.py
cp tests/text_ops_tests/test_split_vi.py tests/lang_ops_tests/test_split_vi.py
cp tests/text_ops_tests/test_split_zh.py tests/lang_ops_tests/test_split_zh.py
cp tests/text_ops_tests/test_split_ja.py tests/lang_ops_tests/test_split_ja.py
cp tests/text_ops_tests/test_split_ko.py tests/lang_ops_tests/test_split_ko.py
```

- [ ] **Step 2: Update imports in all moved test files**

In all moved files, replace:
- `from text_ops.splitter import ChunkPipeline` → `from lang_ops import ChunkPipeline`
- `from text_ops.splitter._sentence import split_sentences` → `from lang_ops.splitter._sentence import split_sentences`
- `from text_ops.splitter._clause import split_clauses` → `from lang_ops.splitter._clause import split_clauses`
- `from text_ops.splitter._paragraph import split_paragraphs` → `from lang_ops.splitter._paragraph import split_paragraphs`
- `from text_ops.splitter._length import split_by_length` → `from lang_ops.splitter._length import split_by_length`

- [ ] **Step 3: Run all tests**

Run: `python -m pytest tests/lang_ops_tests/ -v --tb=short`
Expected: All tests pass.

- [ ] **Step 4: Delete old text_ops test directory**

```bash
rm -rf tests/text_ops_tests/
```

- [ ] **Step 5: Run full test suite**

Run: `python -m pytest tests/ -v --tb=short`
Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add tests/lang_ops_tests/ tests/text_ops_tests/
git commit -m "refactor: move text_ops tests into lang_ops_tests"
```

---

### Task 4: Delete src/text_ops/

**Files:**
- Delete: `src/text_ops/` — entire package

- [ ] **Step 1: Verify no remaining imports reference text_ops**

Run: `grep -r "from text_ops" src/ tests/ --include="*.py"`
Expected: No matches.

- [ ] **Step 2: Delete the package**

```bash
rm -rf src/text_ops/
```

- [ ] **Step 3: Run full test suite**

Run: `python -m pytest tests/ -v --tb=short`
Expected: All tests pass.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor: delete text_ops package (merged into lang_ops)"
```

---

### Task 5: Add tests for shortcut methods

**Files:**
- Create: `tests/lang_ops_tests/test_chunk_shortcuts.py`

- [ ] **Step 1: Write tests for ops.split_sentences/split_clauses/split_paragraphs/chunk**

```python
"""Tests for shortcut splitting methods on ops classes."""

import pytest

from lang_ops import TextOps


class TestSplitSentences:
    def test_en(self) -> None:
        ops = TextOps.for_language("en")
        result = ops.split_sentences("Hello world. How are you?")
        assert result == ["Hello world.", " How are you?"]

    def test_zh(self) -> None:
        ops = TextOps.for_language("zh")
        result = ops.split_sentences("你好。世界！")
        assert result == ["你好。", "世界！"]

    def test_ja(self) -> None:
        ops = TextOps.for_language("ja")
        result = ops.split_sentences("今日は。いい天気！")
        assert result == ["今日は。", "いい天気！"]

    def test_ko(self) -> None:
        ops = TextOps.for_language("ko")
        result = ops.split_sentences("안녕하세요. 반갑습니다!")
        assert result == ["안녕하세요.", " 반갑습니다!"]


class TestSplitClauses:
    def test_en_comma(self) -> None:
        ops = TextOps.for_language("en")
        result = ops.split_clauses("Hello, world, how are you?")
        assert result == ["Hello,", " world,", " how are you?"]

    def test_zh_dunhao(self) -> None:
        ops = TextOps.for_language("zh")
        result = ops.split_clauses("苹果、香蕉、橘子")
        assert result == ["苹果、", "香蕉、", "橘子"]

    def test_ja_touten(self) -> None:
        ops = TextOps.for_language("ja")
        result = ops.split_clauses("今日は、いい天気ですね")
        assert result == ["今日は、", "いい天気ですね"]


class TestSplitParagraphs:
    def test_basic(self) -> None:
        ops = TextOps.for_language("en")
        result = ops.split_paragraphs("Para 1\n\nPara 2\n\nPara 3")
        assert result == ["Para 1", "Para 2", "Para 3"]

    def test_single(self) -> None:
        ops = TextOps.for_language("en")
        result = ops.split_paragraphs("No paragraph break")
        assert result == ["No paragraph break"]

    def test_empty(self) -> None:
        ops = TextOps.for_language("en")
        assert ops.split_paragraphs("") == []


class TestChunkPipeline:
    def test_en_sentences(self) -> None:
        ops = TextOps.for_language("en")
        result = ops.chunk("Hello. World.").sentences().result()
        assert result == ["Hello.", " World."]

    def test_en_paragraphs_then_sentences(self) -> None:
        ops = TextOps.for_language("en")
        text = "First sentence. Second.\n\nThird sentence."
        result = ops.chunk(text).paragraphs().sentences().result()
        assert result == ["First sentence. Second.", " Third sentence."]

    def test_zh_sentences(self) -> None:
        ops = TextOps.for_language("zh")
        result = ops.chunk("你好。世界！").sentences().result()
        assert result == ["你好。", "世界！"]

    def test_immutability(self) -> None:
        ops = TextOps.for_language("en")
        p1 = ops.chunk("Hello. World.")
        p2 = p1.sentences()
        assert p1 is not p2
        assert p1.result() == ["Hello. World."]
        assert p2.result() == ["Hello.", " World."]
```

- [ ] **Step 2: Run the new tests**

Run: `python -m pytest tests/lang_ops_tests/test_chunk_shortcuts.py -v`
Expected: All tests pass.

- [ ] **Step 3: Commit**

```bash
git add tests/lang_ops_tests/test_chunk_shortcuts.py
git commit -m "test: add shortcut method tests for split_sentences/split_clauses/chunk"
```

---

### Task 6: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update architecture section**

Remove `text_ops` from the module layout. Update `lang_ops` section to include splitter. Update the layer relationship diagram. Update the API section. Update the test structure. Update import notes.

Key changes:
- Module layout: remove `text_ops/` entry, add splitter under `lang_ops/`
- Layer relationship: single layer now
- API: add `split_sentences`, `split_clauses`, `split_paragraphs`, `chunk` methods
- Test structure: remove `text_ops_tests/`, add tests under `lang_ops_tests/`
- Import note: all imports from `lang_ops` now

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for merged lang_ops architecture"
```
