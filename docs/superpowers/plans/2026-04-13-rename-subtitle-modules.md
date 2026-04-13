# Rename Subtitle Modules Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename `subtitle` modules to clearer names (`model`, `align`, `build`, `io`) and remove the old paths entirely without changing runtime behavior.

**Architecture:** Keep the public top-level `subtitle` exports stable while moving implementation files and package structure to a more direct layout. Update every internal import, test import, and documentation reference so the new structure is the only valid module surface.

**Tech Stack:** Python 3.10, pytest

---

### Task 1: Lock Down The New Import Surface

**Files:**
- Create: `tests/subtitle/test_module_layout.py`

- [ ] **Step 1: Write the failing test**

```python
import importlib

import pytest


def test_new_subtitle_modules_exist_and_old_paths_are_removed() -> None:
    assert importlib.import_module("subtitle.model")
    assert importlib.import_module("subtitle.align")
    assert importlib.import_module("subtitle.build")
    assert importlib.import_module("subtitle.io.srt")

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("subtitle._types")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("subtitle.words")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("subtitle.builder")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("subtitle.readers.srt")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/subtitle/test_module_layout.py -v`
Expected: FAIL because the new modules do not exist yet and old ones still import.

- [ ] **Step 3: Write minimal implementation**

Move source files to:

```text
src/subtitle/model.py
src/subtitle/align.py
src/subtitle/build.py
src/subtitle/io/srt.py
```

Update `src/subtitle/__init__.py` and all internal imports to use the new module names. Delete the old source paths.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/subtitle/test_module_layout.py -v`
Expected: PASS

### Task 2: Update Tests And Docs To The New Layout

**Files:**
- Modify: `tests/subtitle/test_types.py`
- Modify: `tests/subtitle/test_words.py`
- Modify: `tests/subtitle/readers/test_srt.py`
- Modify: `src/lang_ops/chunk/_pipeline.py`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Write the failing test**

Use the import-layout test from Task 1 plus any test files that still import the old module paths.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/subtitle/test_module_layout.py tests/subtitle -q`
Expected: FAIL until all imports are updated.

- [ ] **Step 3: Write minimal implementation**

Rename tests to match the new module names where appropriate and switch all imports/doc references from:

```text
subtitle._types -> subtitle.model
subtitle.words -> subtitle.align
subtitle.builder -> subtitle.build
subtitle.readers.srt -> subtitle.io.srt
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/subtitle tests/lang_ops_tests/chunk -q`
Expected: PASS
