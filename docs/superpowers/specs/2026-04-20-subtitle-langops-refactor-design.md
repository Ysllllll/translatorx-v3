# Subtitle + lang_ops Module Refactor Design

**Date:** 2026-04-20
**Branch:** feature/runtime-refactor
**Status:** Draft

---

## Problem Statement

The `subtitle` and `lang_ops` packages have unclear module boundaries and overlapping responsibilities, making the codebase hard to reason about — especially around punctuation restoration and text transformation flows.

### Specific Issues

1. **`ChunkPipeline` conflates three concerns**: text structuring (sentences/clauses/split/merge), content transform dispatch (`apply()`), and word alignment (`segments()`).
2. **`Subtitle` reimplements `ChunkPipeline.apply()`**: imports `_call_apply_fn` from the pipeline module and rebuilds token groups itself, duplicating tokenization logic.
3. **Three apply methods with confusing semantics**: `apply()`, `apply_global()`, `apply_per_sentence()` have different preconditions and scopes but similar names.
4. **Reverse dependency L1→L2**: `ChunkPipeline.segments()` lazy-imports `subtitle.align`, breaking the clean layer relationship.
5. **Naming collision**: `_BaseOps.restore_punc()` (token-level punctuation transfer) vs `LlmPuncRestorer` (LLM-based punctuation generation) use similar names for unrelated operations.
6. **Dual `ApplyFn` definition**: type alias in `lang_ops.chunk._pipeline` and Protocol in `preprocess._protocol`.

---

## Design Principles

1. **API intuitiveness**: Users should be able to chain operations without reading source code.
2. **Standalone text pipeline**: Pure text scenarios (no word timing) must work independently via `lang_ops`, without importing `subtitle`.
3. **Internal testability**: Each module testable in isolation without mocking other modules.
4. **Extensibility**: Adding a new language or transform requires changes in one place only.

---

## Architecture Overview

### Layer Diagram

```
L0: model (Word, Segment, SentenceRecord)          ← no changes

L1: lang_ops
    ├── ops/       TextOps (_BaseOps + subclasses)
    │              split, join, normalize, length, transfer_punc
    └── chunk/     TextPipeline (was ChunkPipeline)
                   sentences, clauses, split, merge, result
                   ✗ NO apply/transform
                   ✗ NO segments/alignment

L2: subtitle
    ├── align.py   Word timing alignment
    │              fill_words, find_words, distribute_words, align_segments
    ├── core.py    Subtitle (sole orchestrator)
    │              transform, sentences, clauses, split, merge, build, records
    └── io/        SRT/WhisperX parsers

L2: preprocess     ApplyFn protocol + implementations
    ├── _protocol  ApplyFn (single canonical definition)
    └── impls      LlmPuncRestorer, NerPuncRestorer, LlmChunker, ...
```

### Dependency Flow (all downward)

```
L0: model ←── lang_ops._core._punctuation (strip_punct)

L1: lang_ops
    TextPipeline ←── _boundary, _length, _merge, _base_ops
    (NO imports from subtitle or preprocess)

L2: subtitle.core
    Subtitle ←── TextPipeline (structural ops)
             ←── subtitle.align (word alignment)
             ←── preprocess._protocol (ApplyFn type, for annotation)

L2: preprocess
    ApplyFn defined here (sole definition)
    Implementations ←── llm_ops (TYPE_CHECKING only)
```

**No circular dependencies. No reverse (L1→L2) dependencies.**

---

## Detailed Changes

### 1. TextPipeline (was ChunkPipeline)

**Rename:** `ChunkPipeline` → `TextPipeline`

**Remove:**
- `apply()` method and the `ApplyFn` type alias — transform dispatch is not a structuring concern.
- `segments()` method — word alignment belongs to L2 (`subtitle`).
- The `_call_apply_fn()` helper function — moves to `subtitle/core.py`.

**Retain (unchanged):**
- `__init__(text, *, language=None, ops=None)` — tokenizes once via `ops.split()`.
- `from_chunks(chunks, ops)` — constructs from pre-split chunks.
- `_from_groups(groups, ops)` — internal constructor from pre-tokenized groups.
- `sentences()` → `TextPipeline` — split by sentence boundaries.
- `clauses(merge_under=None)` → `TextPipeline` — split by clause boundaries (sentence-aware).
- `split(max_len)` → `TextPipeline` — split by length.
- `merge(max_len)` → `TextPipeline` — greedy merge adjacent groups.
- `result()` → `list[str]` — join each group into text.
- `_groups` attribute (for internal use by Subtitle).

**`_BaseOps` convenience methods remain unchanged:**
- `split_sentences(text)`, `split_clauses(text)`, `split_by_length(text, max_len)`, `merge_by_length(chunks, max_len)`, `chunk(text)` — all return `TextPipeline` or `list[str]`. These provide standalone text-only usage without `subtitle`.

### 2. Subtitle — Unified `transform()` API

**Remove:**
- `apply(fn, cache, batch_size, workers, skip_if)` — replaced by `transform()`.
- `apply_global(name, fn, cache, batch_size, workers, skip_if)` — replaced by `transform()`.
- `apply_per_sentence(name, fn, batch_size, workers, skip_if)` — replaced by `transform()`.

**Add:**

```python
def transform(
    self,
    fn: ApplyFn,
    *,
    cache: dict[str, list[str]] | None = None,
    name: str | None = None,
    batch_size: int = 1,
    workers: int = 1,
    skip_if: Callable[[str], bool] | None = None,
) -> Subtitle:
```

**Behavior:**
- **Pre-sentence** (when `_sentence_split is False`): Collects all pipeline texts, dispatches to `fn` as a single batch, re-tokenizes results, returns new `Subtitle`. The `name` parameter is ignored in this mode.
- **Post-sentence** (when `_sentence_split is True`): Collects texts from all sentence pipelines, dispatches to `fn` in one batch (for efficiency), re-tokenizes results, distributes back to per-pipeline groups. If `name` is provided, stamps results into `chunk_cache[name]` for each sentence.

**The `_call_apply_fn` helper** (batched dispatch with `ThreadPoolExecutor`) moves from `lang_ops/chunk/_pipeline.py` to `subtitle/core.py` as a module-level function.

**Structural operations remain unchanged** — they delegate to `TextPipeline`:
- `sentences()`, `clauses(merge_under)`, `split(max_len)`, `merge(max_len)`

**`sentences()` remains re-callable** — this supports patterns like `sentences() → transform(punc) → sentences()` (Pipeline B).

### 3. Naming Changes

| Current | New | Reason |
|---------|-----|--------|
| `ChunkPipeline` | `TextPipeline` | Reflects "pure text structuring" role |
| `_BaseOps.restore_punc(text_a, text_b)` | `transfer_punc(text_a, text_b)` | Distinguishes token-level "punctuation transfer" from LLM-based "punctuation restoration/generation" |
| `apply_global()` | removed, merged into `transform()` | Single method |
| `apply_per_sentence()` | removed, merged into `transform()` | Single method |
| `apply()` on Subtitle | removed, merged into `transform()` | Single method |

### 4. ApplyFn Protocol Unification

**Single canonical definition** in `preprocess/_protocol.py`:

```python
class ApplyFn(Protocol):
    def __call__(self, texts: list[str]) -> list[list[str]]: ...
```

- `lang_ops/chunk/_pipeline.py` no longer defines `ApplyFn` type alias.
- `subtitle/core.py` imports `ApplyFn` from `preprocess._protocol` for type annotation.

### 5. Timestamp Alignment (No Changes)

The existing deferred alignment strategy is clean and remains unchanged:

1. **`Subtitle.__init__`**: Extracts `(words, full_text)` via `_extract()`. Missing words are filled via `fill_words()`.
2. **`transform()`**: Modifies text content. Words are **not** redistributed — punctuation/chunking changes don't affect word timestamps.
3. **`sentences()`**: Coarse alignment — `distribute_words(words, sent_texts)` maps words to sentences.
4. **`build()` / `records()`**: Fine alignment — `align_segments(chunks, words)` maps words to final chunks.

This two-phase alignment (coarse at `sentences()`, fine at output) works correctly because:
- Text transforms (adding punctuation, splitting chunks) don't change the underlying word content or timing.
- `align_segments` matches words to chunks by text content at output time, after all transforms are complete.

---

## Usage Examples

### Pipeline A: Punctuation Restoration → Sentences

```python
from subtitle import Subtitle
from preprocess import LlmPuncRestorer

restorer = LlmPuncRestorer(engine)
punc_cache: dict[str, list[str]] = {}

records = (
    Subtitle(segments, language="en")
    .transform(restorer, cache=punc_cache)   # pre-sentence: global
    .sentences()
    .records()
)
```

### Pipeline D: Punc → Sentences → Smart Chunk

```python
from preprocess import LlmPuncRestorer, LlmChunker

restorer = LlmPuncRestorer(engine)
chunker = LlmChunker(engine, chunk_len=90)

records = (
    Subtitle(segments, language="en")
    .transform(restorer, cache=punc_cache)
    .sentences()
    .transform(chunker, name="chunk", skip_if=lambda t: len(t) <= 90)  # post-sentence: per-sentence
    .records()
)
```

### Pipeline B: Sentences → Per-Sentence Punc → Re-split

```python
records = (
    Subtitle(segments, language="en")
    .sentences()
    .transform(restorer, name="punc")  # post-sentence: per-sentence
    .sentences()                       # re-split with new punctuation
    .records()
)
```

### Standalone Text Chunking (No Subtitle)

```python
from lang_ops import LangOps

ops = LangOps.for_language("en")

# Via convenience methods
sentences = ops.split_sentences(text)
chunks = ops.split_by_length(text, max_len=50)

# Via TextPipeline for chaining
result = ops.chunk(text).sentences().split(50).merge(80).result()
```

---

## Migration Path

### Phase 1: Rename + Internal Restructure (No API Break)

1. Rename `ChunkPipeline` → `TextPipeline` with backward-compat alias.
2. Rename `_BaseOps.restore_punc()` → `transfer_punc()` with backward-compat alias.
3. Move `_call_apply_fn` from `lang_ops/chunk/_pipeline.py` to `subtitle/core.py`.
4. Remove `ChunkPipeline.segments()` (move logic to `Subtitle.build()`).

### Phase 2: API Simplification

5. Add `Subtitle.transform()` method with auto-scope detection.
6. Deprecate `apply()`, `apply_global()`, `apply_per_sentence()` (keep as thin wrappers that delegate to `transform()`).

### Phase 3: Cleanup

7. Remove deprecated methods.
8. Remove `ApplyFn` type alias from `TextPipeline`.
9. Remove `apply()` method from `TextPipeline`.
10. Update all demos and tests.

---

## Files Affected

### Modified

| File | Changes |
|------|---------|
| `src/lang_ops/chunk/_pipeline.py` | Rename class, remove `apply()`, `segments()`, `ApplyFn`, `_call_apply_fn` |
| `src/lang_ops/chunk/__init__.py` | Update re-exports |
| `src/lang_ops/__init__.py` | Update re-exports |
| `src/lang_ops/_core/_base_ops.py` | Rename `restore_punc` → `transfer_punc`, update `chunk()` return type name |
| `src/subtitle/core.py` | Add `transform()`, absorb `_call_apply_fn`, remove old apply methods |
| `src/subtitle/__init__.py` | Update re-exports |
| `demos/course_batch/demo_sentence.py` | Update to use `transform()` |
| `tests/preprocess_tests/test_llm_punc.py` | No change (tests the restorer, not pipeline integration) |
| `tests/lang_ops_tests/chunk/` | Remove tests for `apply()` and `segments()` if any |
| `tests/subtitle/build_tests/` | Update to use `transform()` |

### Test Coverage

- Existing `lang_ops_tests/` — structural operations tested as-is.
- Existing `subtitle/build_tests/` — update for new API.
- New test: `Subtitle.transform()` pre/post-sentence scope detection.
- New test: `transform()` with `name` parameter stamps `chunk_cache`.
- Regression: all existing pipelines (A/B/C/D) produce equivalent output.

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Breaking `trx` facade re-exports | Phase 1 uses backward-compat aliases |
| Breaking `runtime` processors that use `apply()` | Search for all `apply(` / `apply_global(` / `apply_per_sentence(` call sites before removing |
| `TextPipeline` standalone users (via `_BaseOps.chunk()`) lose `apply()` | These users should use `transform()` on `Subtitle` instead; document migration |
| `sentences()` re-call after transform may produce different word alignment | Already works correctly today; add regression test |

---

## Out of Scope

- **`SubtitleStream`**: The streaming interface (`feed`/`flush`) does not use `apply()` and is unaffected by this refactor.
- Refactoring `_BaseOps` subclass hierarchy (Chinese/Japanese/Korean/EnType).
- Changing the `preprocess` package structure.
- Modifying the `runtime` or `trx` packages (beyond updating call sites).
- Changing word alignment algorithms in `subtitle/align.py`.
- Adding new preprocessing capabilities.
