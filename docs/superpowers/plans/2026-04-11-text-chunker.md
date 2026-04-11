# Text Chunker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a pipeline-based text chunking module (`src/text_chunker/`) that splits text by paragraphs, sentences, clauses, and length, with automatic language adaptation via `text_ops`.

**Architecture:** Independent package under `src/` that depends on `text_ops`. `ChunkPipeline` is the single entry point — immutable, each method returns a new instance. Splitters are stateless functions in `_splitters/`, each taking text + language config. Language-specific punctuation sets live in `_lang_config.py`.

**Tech Stack:** Python 3.10+, pytest, `text_ops` (same repo)

---

## Files to Create

```
src/text_chunker/
├── __init__.py              # Exports ChunkPipeline
├── _pipeline.py             # ChunkPipeline class
├── _lang_config.py          # Per-language punctuation sets
└── _splitters/
    ├── __init__.py
    ├── _paragraph.py        # split_paragraphs(text) -> list[str]
    ├── _sentence.py         # split_sentences(text, ops, config) -> list[str]
    ├── _clause.py           # split_clauses(text, ops, config) -> list[str]
    └── _length.py           # split_by_length(text, ops, max_length, unit) -> list[str]

tests/text_chunker_tests/
├── __init__.py
├── test_pipeline.py
├── test_paragraph.py
├── test_sentence.py
├── test_clause.py
└── test_length.py
```

No existing files are modified.

---

### Task 1: Language Config

**Files:**
- Create: `src/text_chunker/_lang_config.py`

- [ ] **Step 1: Write `_lang_config.py`**

```python
"""Per-language punctuation configuration for text chunking."""

from __future__ import annotations

from typing import Mapping


# Sentence-terminal punctuation per language.
# Languages not listed fall back to "default".
SENTENCE_TERMINALS: Mapping[str, frozenset[str]] = {
    "default": frozenset({".", "!", "?"}),
    "zh": frozenset({"。", "！", "？"}),
    "ja": frozenset({"。", "！", "？"}),
    "ko": frozenset({"。", "!", "?"}),
}

# Clause-separator punctuation per language.
CLAUSE_SEPARATORS: Mapping[str, frozenset[str]] = {
    "default": frozenset({",", ";", ":", "\u2014"}),  # \u2014 = em dash
    "zh": frozenset({"，", "、", "；", "："}),
    "ja": frozenset({"、", "；"}),
    "ko": frozenset({",", "；"}),
}

# English abbreviation whitelist — period after these is NOT a sentence boundary.
ABBREVIATIONS: frozenset[str] = frozenset({
    "Mr", "Mrs", "Ms", "Dr", "Prof", "Sr", "Jr", "St",
    "Inc", "Ltd", "Co", "Corp", "vs", "etc", "eg", "ie",
    "Jan", "Feb", "Mar", "Apr", "Jun", "Jul", "Aug", "Sep",
    "Oct", "Nov", "Dec",
})


def get_sentence_terminators(language: str) -> frozenset[str]:
    return SENTENCE_TERMINALS.get(language, SENTENCE_TERMINALS["default"])


def get_clause_separators(language: str) -> frozenset[str]:
    return CLAUSE_SEPARATORS.get(language, CLAUSE_SEPARATORS["default"])
```

- [ ] **Step 2: Verify import works**

Run: `cd /home/ysl/workspace/develop/translatorx-v3 && /home/ysl/workspace/.venv/bin/python -c "from text_chunker._lang_config import get_sentence_terminators, get_clause_separators, ABBREVIATIONS; print(get_sentence_terminators('en')); print(get_clause_separators('zh')); print(len(ABBREVIATIONS))"`

Expected: prints default set, CJK set, and abbreviation count (24).

Note: this step will fail until `__init__.py` exists. Create a minimal one first:

```python
# src/text_chunker/__init__.py  (temporary, will be replaced in Task 6)
```

Create the directory and minimal init:

```bash
mkdir -p /home/ysl/workspace/develop/translatorx-v3/src/text_chunker/_splitters
touch /home/ysl/workspace/develop/translatorx-v3/src/text_chunker/__init__.py
touch /home/ysl/workspace/develop/translatorx-v3/src/text_chunker/_splitters/__init__.py
```

Then re-run the verification.

---

### Task 2: Paragraph Splitter + Tests

**Files:**
- Create: `src/text_chunker/_splitters/_paragraph.py`
- Create: `tests/text_chunker_tests/__init__.py`
- Create: `tests/text_chunker_tests/test_paragraph.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/text_chunker_tests/test_paragraph.py
"""Tests for paragraph splitting."""

import pytest

from text_chunker._splitters._paragraph import split_paragraphs


class TestSplitParagraphs:

    def test_single_paragraph(self) -> None:
        text = "Hello world."
        assert split_paragraphs(text) == ["Hello world."]

    def test_two_paragraphs(self) -> None:
        text = "First paragraph.\n\nSecond paragraph."
        assert split_paragraphs(text) == ["First paragraph.", "Second paragraph."]

    def test_trims_whitespace(self) -> None:
        text = "  Hello.  \n\n  World.  "
        assert split_paragraphs(text) == ["Hello.", "World."]

    def test_discards_empty_paragraphs(self) -> None:
        text = "First.\n\n\n\nSecond."
        assert split_paragraphs(text) == ["First.", "Second."]

    def test_crlf_line_endings(self) -> None:
        text = "First.\r\n\r\nSecond."
        assert split_paragraphs(text) == ["First.", "Second."]

    def test_empty_input(self) -> None:
        assert split_paragraphs("") == []

    def test_whitespace_only_input(self) -> None:
        assert split_paragraphs("   \n\n   ") == []

    def test_single_newline_does_not_split(self) -> None:
        text = "Line one.\nLine two."
        assert split_paragraphs(text) == ["Line one.\nLine two."]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/home/ysl/workspace/.venv/bin/pytest tests/text_chunker_tests/test_paragraph.py -v`

Expected: FAIL (module not found).

- [ ] **Step 3: Write implementation**

```python
# src/text_chunker/_splitters/_paragraph.py
"""Paragraph splitter — language-independent."""

from __future__ import annotations

import re

_SPLIT_RE = re.compile(r"\n\s*\n")


def split_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs on blank lines.

    - Consecutive blank lines are treated as one separator.
    - Leading/trailing whitespace is trimmed per paragraph.
    - Empty paragraphs are discarded.
    """
    if not text or not text.strip():
        return []

    raw_parts = _SPLIT_RE.split(text)
    result = [part.strip() for part in raw_parts]
    return [p for p in result if p]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/home/ysl/workspace/.venv/bin/pytest tests/text_chunker_tests/test_paragraph.py -v`

Expected: all 8 tests PASS.

---

### Task 3: Clause Splitter + Tests

**Files:**
- Create: `src/text_chunker/_splitters/_clause.py`
- Create: `tests/text_chunker_tests/test_clause.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/text_chunker_tests/test_clause.py
"""Tests for clause splitting."""

import pytest

from text_chunker._lang_config import get_clause_separators
from text_chunker._splitters._clause import split_clauses


class TestSplitClauses:

    def test_english_comma(self) -> None:
        seps = get_clause_separators("en")
        result = split_clauses("Hello, world, how are you?", seps)
        assert result == ["Hello,", " world,", " how are you?"]

    def test_single_clause(self) -> None:
        seps = get_clause_separators("en")
        result = split_clauses("No commas here", seps)
        assert result == ["No commas here"]

    def test_chinese_dunhao(self) -> None:
        seps = get_clause_separators("zh")
        result = split_clauses("苹果、香蕉、橘子", seps)
        assert result == ["苹果、", "香蕉、", "橘子"]

    def test_japanese_touten(self) -> None:
        seps = get_clause_separators("ja")
        result = split_clauses("今日は、いい天気ですね", seps)
        assert result == ["今日は、", "いい天気ですね"]

    def test_semicolon(self) -> None:
        seps = get_clause_separators("en")
        result = split_clauses("First; second; third", seps)
        assert result == ["First;", " second;", " third"]

    def test_empty_input(self) -> None:
        seps = get_clause_separators("en")
        assert split_clauses("", seps) == []

    def test_trailing_separator(self) -> None:
        seps = get_clause_separators("en")
        result = split_clauses("Hello,", seps)
        assert result == ["Hello,"]

    def test_leading_separator(self) -> None:
        seps = get_clause_separators("en")
        result = split_clauses(",Hello", seps)
        assert result == [",Hello"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/home/ysl/workspace/.venv/bin/pytest tests/text_chunker_tests/test_clause.py -v`

Expected: FAIL (module not found).

- [ ] **Step 3: Write implementation**

```python
# src/text_chunker/_splitters/_clause.py
"""Clause splitter — splits at comma/pause punctuation."""

from __future__ import annotations


def split_clauses(text: str, separators: frozenset[str]) -> list[str]:
    """Split text into clauses at separator characters.

    Separators stay with the preceding clause.
    """
    if not text:
        return []

    result: list[str] = []
    current_start = 0

    for i, ch in enumerate(text):
        if ch in separators:
            # Include this character and everything up to it in the current clause
            clause = text[current_start : i + 1]
            current_start = i + 1
            result.append(clause)

    # Remaining text after the last separator
    remainder = text[current_start:]
    if remainder:
        result.append(remainder)

    return result if result else []
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/home/ysl/workspace/.venv/bin/pytest tests/text_chunker_tests/test_clause.py -v`

Expected: all 8 tests PASS.

---

### Task 4: Sentence Splitter + Tests

**Files:**
- Create: `src/text_chunker/_splitters/_sentence.py`
- Create: `tests/text_chunker_tests/test_sentence.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/text_chunker_tests/test_sentence.py
"""Tests for sentence splitting."""

import pytest

from text_chunker._lang_config import get_sentence_terminators, ABBREVIATIONS
from text_chunker._splitters._sentence import split_sentences


class TestSplitSentencesEnglish:

    def test_two_simple_sentences(self) -> None:
        result = split_sentences("Hello world. How are you?", "en")
        assert result == ["Hello world.", " How are you?"]

    def test_exclamation_and_question(self) -> None:
        result = split_sentences("Wow! Really? Yes.", "en")
        assert result == ["Wow!", " Really?", " Yes."]

    def test_abbreviation_dr(self) -> None:
        result = split_sentences("Dr. Smith went home.", "en")
        assert result == ["Dr. Smith went home."]

    def test_abbreviation_mid_sentence(self) -> None:
        result = split_sentences("He met Dr. Smith. Then he left.", "en")
        assert result == ["He met Dr. Smith.", " Then he left."]

    def test_ellipsis_preserved(self) -> None:
        result = split_sentences("Wait... Go on.", "en")
        assert result == ["Wait... Go on."]

    def test_number_dot(self) -> None:
        result = split_sentences("The value is 3.14 approx.", "en")
        assert result == ["The value is 3.14 approx."]

    def test_closing_quote(self) -> None:
        result = split_sentences('He said "hello." Then he left.', "en")
        assert result == ['He said "hello."', " Then he left."]

    def test_single_sentence(self) -> None:
        result = split_sentences("No terminators here", "en")
        assert result == ["No terminators here"]

    def test_empty_input(self) -> None:
        assert split_sentences("", "en") == []


class TestSplitSentencesCJK:

    def test_chinese(self) -> None:
        result = split_sentences("你好。世界！", "zh")
        assert result == ["你好。", "世界！"]

    def test_chinese_question(self) -> None:
        result = split_sentences("你吃了吗？我吃了。", "zh")
        assert result == ["你吃了吗？", "我吃了。"]

    def test_japanese(self) -> None:
        result = split_sentences("今日は。いい天気！", "ja")
        assert result == ["今日は。", "いい天気！"]

    def test_korean(self) -> None:
        result = split_sentences("안녕하세요. 반갑습니다!", "ko")
        assert result == ["안녕하세요.", " 반갑습니다!"]

    def test_cjk_ellipsis(self) -> None:
        result = split_sentences("他……走了。", "zh")
        assert result == ["他……走了。"]

    def test_no_terminators(self) -> None:
        result = split_sentences("这是一段文字", "zh")
        assert result == ["这是一段文字"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/home/ysl/workspace/.venv/bin/pytest tests/text_chunker_tests/test_sentence.py -v`

Expected: FAIL (module not found).

- [ ] **Step 3: Write implementation**

```python
# src/text_chunker/_splitters/_sentence.py
"""Sentence splitter with language-adapted rules."""

from __future__ import annotations

import re

from text_chunker._lang_config import (
    ABBREVIATIONS,
    get_sentence_terminators,
)


def _is_abbreviation(text: str, dot_pos: int) -> bool:
    """Check if the period at dot_pos follows an abbreviation."""
    # Look backwards for the word before the dot
    i = dot_pos - 1
    while i >= 0 and text[i].isalnum():
        i -= 1
    word = text[i + 1 : dot_pos]
    # Single-letter abbreviations (U.S., initials like "A.")
    if len(word) <= 1:
        return True
    return word in ABBREVIATIONS


def _is_number_dot(text: str, dot_pos: int) -> bool:
    """Check if the period at dot_pos is part of a number (e.g. 3.14)."""
    # Check character after dot
    after = dot_pos + 1
    if after < len(text) and text[after].isdigit():
        return True
    # Check character before dot
    if dot_pos > 0 and text[dot_pos - 1].isdigit():
        return True
    return False


def _is_ellipsis(text: str, pos: int) -> bool:
    """Check if the character at pos is part of an ellipsis (...)."""
    if text[pos] != ".":
        return False
    # Is there a dot right before or after?
    before = pos > 0 and text[pos - 1] == "."
    after = pos + 1 < len(text) and text[pos + 1] == "."
    return before or after


def _is_cjk_ellipsis(text: str, pos: int) -> bool:
    """Check if char at pos is … (U+2026) or part of CJK double-dot …… """
    if text[pos] == "\u2026":
        return True
    return False


def split_sentences(text: str, language: str) -> list[str]:
    """Split text into sentences at terminal punctuation.

    Rules:
    - CJK (zh, ja, ko): split at 。！？. Preserve ……
    - Default (en, etc.): split at .!? with abbreviation/number/ellipsis guards.
    - Terminal punctuation stays with the current sentence.
    """
    if not text:
        return []

    terminators = get_sentence_terminators(language)
    is_cjk = language in ("zh", "ja", "ko")

    result: list[str] = []
    current_start = 0

    i = 0
    while i < len(text):
        ch = text[i]

        if ch in terminators:
            # Check for ellipsis — skip if part of one
            if is_cjk:
                if _is_cjk_ellipsis(text, i):
                    i += 1
                    continue
            else:
                if _is_ellipsis(text, i):
                    i += 1
                    continue

            # For period in default languages, check guards
            if not is_cjk and ch == ".":
                if _is_abbreviation(text, i):
                    i += 1
                    continue
                if _is_number_dot(text, i):
                    i += 1
                    continue

            # This is a sentence break.
            # Include any closing quote after the terminator.
            end = i + 1
            while end < len(text) and text[end] in ('"', "\u201d", "\u2019", "'", "\u300d", "\u300f"):
                end += 1

            sentence = text[current_start:end]
            result.append(sentence)
            current_start = end
            i = end
        else:
            i += 1

    # Remaining text after the last terminator
    remainder = text[current_start:]
    if remainder:
        result.append(remainder)

    return result if result else []
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/home/ysl/workspace/.venv/bin/pytest tests/text_chunker_tests/test_sentence.py -v`

Expected: all 15 tests PASS.

- [ ] **Step 5: Fix any edge cases**

If tests reveal issues with abbreviation detection or quote handling, fix in `_sentence.py` and re-run.

---

### Task 5: Length Splitter + Tests

**Files:**
- Create: `src/text_chunker/_splitters/_length.py`
- Create: `tests/text_chunker_tests/test_length.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/text_chunker_tests/test_length.py
"""Tests for length-based splitting."""

import pytest

from text_ops import TextOps

from text_chunker._splitters._length import split_by_length


class TestSplitByLength:

    def test_short_text_unchanged(self) -> None:
        ops = TextOps.for_language("en")
        result = split_by_length("Hello world", ops, max_length=20, unit="character")
        assert result == ["Hello world"]

    def test_splits_at_word_boundary(self) -> None:
        ops = TextOps.for_language("en")
        result = split_by_length("one two three four", ops, max_length=3, unit="word")
        assert result == ["one two", "three four"]

    def test_exact_length(self) -> None:
        ops = TextOps.for_language("en")
        result = split_by_length("Hi there", ops, max_length=8, unit="character")
        assert result == ["Hi there"]

    def test_character_unit(self) -> None:
        ops = TextOps.for_language("en")
        result = split_by_length("abcdefghij", ops, max_length=5, unit="character")
        assert len(result) >= 2

    def test_empty_input(self) -> None:
        ops = TextOps.for_language("en")
        assert split_by_length("", ops, max_length=10, unit="character") == []

    def test_cjk(self) -> None:
        ops = TextOps.for_language("zh")
        result = split_by_length("这是一段比较长的中文文本需要切分", ops, max_length=8, unit="character")
        assert len(result) >= 2

    def test_single_long_word(self) -> None:
        ops = TextOps.for_language("en")
        result = split_by_length("supercalifragilisticexpialidocious", ops, max_length=5, unit="character")
        # Hard break when no word boundary exists
        assert len(result) >= 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/home/ysl/workspace/.venv/bin/pytest tests/text_chunker_tests/test_length.py -v`

Expected: FAIL (module not found).

- [ ] **Step 3: Write implementation**

```python
# src/text_chunker/_splitters/_length.py
"""Length-based text splitter."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from text_ops._core._mechanism import _LangOps


def split_by_length(
    text: str,
    ops: _LangOps,
    max_length: int,
    unit: str = "character",
) -> list[str]:
    """Split text into chunks that don't exceed max_length.

    Args:
        text: Input text to split.
        ops: Language ops from TextOps.for_language().
        max_length: Maximum length per chunk.
        unit: "character" counts characters, "word" counts word tokens.

    Returns:
        List of text chunks, each <= max_length in the given unit.
    """
    if not text:
        return []

    if unit == "word":
        return _split_by_word_count(text, ops, max_length)
    return _split_by_char_count(text, ops, max_length)


def _split_by_char_count(
    text: str,
    ops: _LangOps,
    max_length: int,
) -> list[str]:
    """Split by character count, breaking at word/token boundaries."""
    tokens = ops.split(text)
    if not tokens:
        return []

    result: list[str] = []
    chunk_tokens: list[str] = []
    chunk_len = 0

    for token in tokens:
        token_len = len(token)

        # If adding this token would exceed max_length, flush the current chunk
        if chunk_tokens and chunk_len + token_len > max_length:
            result.append(ops.join(chunk_tokens))
            chunk_tokens = []
            chunk_len = 0

        # If a single token exceeds max_length, hard-break it
        if token_len > max_length:
            # Flush any accumulated chunk first
            if chunk_tokens:
                result.append(ops.join(chunk_tokens))
                chunk_tokens = []
                chunk_len = 0
            # Hard break the oversized token
            i = 0
            while i < len(token):
                result.append(token[i : i + max_length])
                i += max_length
        else:
            chunk_tokens.append(token)
            chunk_len += token_len

    if chunk_tokens:
        result.append(ops.join(chunk_tokens))

    return result


def _split_by_word_count(
    text: str,
    ops: _LangOps,
    max_length: int,
) -> list[str]:
    """Split by word/token count."""
    tokens = ops.split(text)
    if not tokens:
        return []

    result: list[str] = []
    i = 0
    while i < len(tokens):
        chunk = tokens[i : i + max_length]
        result.append(ops.join(chunk))
        i += max_length

    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/home/ysl/workspace/.venv/bin/pytest tests/text_chunker_tests/test_length.py -v`

Expected: all 7 tests PASS.

---

### Task 6: ChunkPipeline + Tests

**Files:**
- Create: `src/text_chunker/_pipeline.py`
- Modify: `src/text_chunker/__init__.py` (replace temporary with real exports)
- Create: `tests/text_chunker_tests/test_pipeline.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/text_chunker_tests/test_pipeline.py
"""Tests for ChunkPipeline."""

import pytest

from text_chunker import ChunkPipeline


class TestPipelineBasic:

    def test_sentences_only(self) -> None:
        result = ChunkPipeline("Hello. World.", language="en").sentences().result()
        assert result == ["Hello.", " World."]

    def test_clauses_only(self) -> None:
        result = ChunkPipeline("Hello, world, how are you?", language="en").clauses().result()
        assert result == ["Hello,", " world,", " how are you?"]

    def test_paragraphs_only(self) -> None:
        result = ChunkPipeline("First.\n\nSecond.", language="en").paragraphs().result()
        assert result == ["First.", "Second."]

    def test_by_length(self) -> None:
        result = (ChunkPipeline("one two three four five", language="en")
            .by_length(max_length=3, unit="word")
            .result())
        assert len(result) >= 2

    def test_result_returns_list(self) -> None:
        result = ChunkPipeline("Hello world", language="en").result()
        assert isinstance(result, list)
        assert result == ["Hello world"]


class TestPipelineChaining:

    def test_paragraphs_then_sentences(self) -> None:
        text = "First sentence. Second sentence.\n\nThird sentence."
        result = (ChunkPipeline(text, language="en")
            .paragraphs()
            .sentences()
            .result())
        assert result == ["First sentence.", " Second sentence.", "Third sentence."]

    def test_sentences_then_clauses(self) -> None:
        result = (ChunkPipeline("Hello, world. Goodbye, world.", language="en")
            .sentences()
            .clauses()
            .result())
        assert result == ["Hello,", " world.", " Goodbye,", " world."]


class TestPipelineImmutability:

    def test_original_unchanged_after_sentences(self) -> None:
        original = ChunkPipeline("Hello. World.", language="en")
        new_pipeline = original.sentences()
        # Original should still return the full text
        assert original.result() == ["Hello. World."]
        # New pipeline has the split
        assert new_pipeline.result() == ["Hello.", " World."]

    def test_each_step_creates_new_instance(self) -> None:
        p1 = ChunkPipeline("A. B.", language="en")
        p2 = p1.sentences()
        p3 = p2.clauses()
        assert p1 is not p2
        assert p2 is not p3


class TestPipelineEdgeCases:

    def test_empty_input(self) -> None:
        assert ChunkPipeline("", language="en").sentences().result() == []

    def test_unsupported_language(self) -> None:
        with pytest.raises(ValueError):
            ChunkPipeline("Hello", language="xx").result()

    def test_single_step_no_terminators(self) -> None:
        result = ChunkPipeline("No terminators", language="en").sentences().result()
        assert result == ["No terminators"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/home/ysl/workspace/.venv/bin/pytest tests/text_chunker_tests/test_pipeline.py -v`

Expected: FAIL (import error or assertion failure).

- [ ] **Step 3: Write ChunkPipeline implementation**

```python
# src/text_chunker/_pipeline.py
"""ChunkPipeline — immutable, chainable text chunking."""

from __future__ import annotations

from text_ops import TextOps

from text_chunker._lang_config import get_clause_separators
from text_chunker._splitters._paragraph import split_paragraphs
from text_chunker._splitters._sentence import split_sentences
from text_chunker._splitters._clause import split_clauses
from text_chunker._splitters._length import split_by_length


class ChunkPipeline:
    """Immutable pipeline for multi-granularity text splitting.

    Each method returns a new ChunkPipeline instance.
    Call .result() to get the final list[str].
    """

    __slots__ = ("_pieces", "_ops", "_language")

    def __init__(self, text: str, *, language: str) -> None:
        self._ops = TextOps.for_language(language)
        self._language = language
        self._pieces: list[str] = [text] if text else []

    def _with_pieces(self, pieces: list[str]) -> ChunkPipeline:
        """Create a new pipeline with updated pieces. Shares ops and language."""
        new = object.__new__(ChunkPipeline)
        new._ops = self._ops
        new._language = self._language
        new._pieces = pieces
        return new

    def paragraphs(self) -> ChunkPipeline:
        """Split each piece into paragraphs (on blank lines)."""
        result: list[str] = []
        for piece in self._pieces:
            result.extend(split_paragraphs(piece))
        return self._with_pieces(result)

    def sentences(self) -> ChunkPipeline:
        """Split each piece into sentences."""
        result: list[str] = []
        for piece in self._pieces:
            result.extend(split_sentences(piece, self._language))
        return self._with_pieces(result)

    def clauses(self) -> ChunkPipeline:
        """Split each piece into clauses (at comma/pause punctuation)."""
        seps = get_clause_separators(self._language)
        result: list[str] = []
        for piece in self._pieces:
            result.extend(split_clauses(piece, seps))
        return self._with_pieces(result)

    def by_length(self, max_length: int, unit: str = "character") -> ChunkPipeline:
        """Split each piece by length, breaking at token boundaries."""
        result: list[str] = []
        for piece in self._pieces:
            result.extend(split_by_length(piece, self._ops, max_length, unit))
        return self._with_pieces(result)

    def result(self) -> list[str]:
        """Return the current list of text pieces."""
        return list(self._pieces)
```

- [ ] **Step 4: Update `__init__.py` with real exports**

```python
# src/text_chunker/__init__.py
"""Text chunking library — pipeline-based multi-granularity text splitting."""

from ._pipeline import ChunkPipeline

__all__ = ["ChunkPipeline"]
```

- [ ] **Step 5: Run all tests**

Run: `/home/ysl/workspace/.venv/bin/pytest tests/text_chunker_tests/ -v`

Expected: all tests PASS.

---

### Task 7: Run Full Test Suite

- [ ] **Step 1: Run all existing + new tests**

Run: `/home/ysl/workspace/.venv/bin/pytest tests/ -v`

Expected: all existing text_ops + subtitle tests still pass, all new text_chunker tests pass.

- [ ] **Step 2: Fix any regressions**

If any pre-existing tests break, investigate. The new code should not modify any existing files, so regressions are unlikely.

---

### Task 8: Final Commit

- [ ] **Step 1: Review all new files**

```bash
git status
```

Verify only `src/text_chunker/` and `tests/text_chunker_tests/` are new.

- [ ] **Step 2: Commit**

```bash
git add src/text_chunker/ tests/text_chunker_tests/
git commit -m "feat: add text_chunker module with pipeline-based multi-granularity splitting"
```
