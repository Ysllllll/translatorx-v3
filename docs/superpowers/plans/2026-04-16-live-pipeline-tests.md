# Live Pipeline Tests Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add live-model pipeline coverage for stable pipeline behavior and weak semantic coherence without introducing brittle exact-match assertions.

**Architecture:** Extend the existing live integration test module with a small set of end-to-end tests that reuse the current local Qwen engine fixture and skip behavior. Keep assertions focused on pipeline invariants such as skip paths, ordering, metadata, and repeated-term continuity rather than exact translation wording.

**Tech Stack:** `pytest`, `pytest-asyncio`, local OpenAI-compatible Qwen server, existing `Pipeline` and `TranslateNodeConfig`

---

### Task 1: Add a stable mixed-behavior live pipeline test

**Files:**
- Modify: `tests/pipeline_tests/test_live.py`
- Test: `tests/pipeline_tests/test_live.py`

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_live_mixed_processing_paths(...):
    ...
    assert results[0].skipped
    assert results[1].skipped
    assert not results[2].skipped
    assert results[3].skipped
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/pipeline_tests/test_live.py -k "mixed_processing_paths" -v -s`
Expected: FAIL if assertions or helper wiring are wrong, or SKIP when the local live model is unavailable.

- [ ] **Step 3: Write minimal implementation**

```python
# Add a live test that mixes existing-translation, direct-translate,
# prefix-handled, normal LLM, and skip-long records in one pipeline run.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/pipeline_tests/test_live.py -k "mixed_processing_paths" -v -s`
Expected: PASS with a reachable local model, otherwise SKIP.

### Task 2: Add concurrency alignment live coverage

**Files:**
- Modify: `tests/pipeline_tests/test_live.py`
- Test: `tests/pipeline_tests/test_live.py`

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_live_concurrency_preserves_record_alignment(...):
    ...
    assert [r.src_text for r in built] == source_texts
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/pipeline_tests/test_live.py -k "concurrency_preserves_record_alignment" -v -s`
Expected: FAIL if output order or assertions are wrong, or SKIP when the local live model is unavailable.

- [ ] **Step 3: Write minimal implementation**

```python
# Add a live test that runs Pipeline.translate(concurrency=3) and asserts
# output order, per-record translation presence, and result metadata shape.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/pipeline_tests/test_live.py -k "concurrency_preserves_record_alignment" -v -s`
Expected: PASS with a reachable local model, otherwise SKIP.

### Task 3: Add a weak semantic consistency live test

**Files:**
- Modify: `tests/pipeline_tests/test_live.py`
- Test: `tests/pipeline_tests/test_live.py`

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_live_repeated_term_stays_visible_across_context(...):
    ...
    assert all("X" in text for text in translated_texts)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/pipeline_tests/test_live.py -k "repeated_term_stays_visible_across_context" -v -s`
Expected: FAIL if the assertions are too strict or helpers are wrong, or SKIP when the local live model is unavailable.

- [ ] **Step 3: Write minimal implementation**

```python
# Add a live context-window test that checks repeated symbol/term continuity
# with weak assertions instead of exact translation snapshots.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/pipeline_tests/test_live.py -k "repeated_term_stays_visible_across_context" -v -s`
Expected: PASS with a reachable local model, otherwise SKIP.
