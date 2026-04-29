"""Smoke-test the rewritten checker demos.

Both ``demos/basics/checker.py`` and ``demos/internals/llm_ops/checker.py``
expose row-builder functions that emit dicts conforming to the unified
10-column schema::

    case | scene | source | target | sanitized | result | E | W | I | issues
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEMOS = ROOT / "demos"

MAIN_KEYS = {"case", "scene", "source", "target", "sanitized", "result", "E", "W", "I", "issues"}


def _load_basics():
    sys.path.insert(0, str(DEMOS / "basics"))
    sys.path.insert(0, str(DEMOS))
    try:
        spec = importlib.util.spec_from_file_location("basics_checker_demo", DEMOS / "basics" / "checker.py")
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for p in (str(DEMOS), str(DEMOS / "basics")):
            while p in sys.path:
                sys.path.remove(p)


def _load_internals():
    sys.path.insert(0, str(DEMOS))
    try:
        return importlib.import_module("internals.llm_ops.checker")
    finally:
        while str(DEMOS) in sys.path:
            sys.path.remove(str(DEMOS))


# ---------------------------------------------------------------------------
# basics demo
# ---------------------------------------------------------------------------


def test_basics_checker_module_imports_and_main_runs(capsys):
    """The whole demo must execute end-to-end without raising."""
    demo = _load_basics()
    demo.main()
    out = capsys.readouterr().out
    assert "translate.en.zh" in out
    assert "Registry" in out
    # YAML examples are part of the demo (Part 8)
    assert "yaml" in demo.__dict__
    assert "YAML_EXAMPLES" in demo.__dict__


def test_basics_checker_row_helper_uses_unified_schema():
    demo = _load_basics()
    chk = demo.default_checker("en", "zh")

    row_pass = demo._check(chk, "clean", "Hello.", "你好。")
    row_fail = demo._check(chk, "empty", "Hello.", "")

    assert set(row_pass) == MAIN_KEYS
    assert row_pass["result"] == "PASS"
    assert row_pass["E"] == "0"

    assert row_fail["result"] == "FAIL"
    assert int(row_fail["E"]) >= 1
    assert "non_empty" in row_fail["issues"]
    assert row_fail["target"] == "(empty)"


# ---------------------------------------------------------------------------
# internals/llm_ops demo
# ---------------------------------------------------------------------------


def test_llm_ops_rule_matrix_covers_pass_warn_fail():
    demo = _load_internals()

    rows = demo.build_rule_matrix_rows()
    assert len(rows) >= 12
    assert all(set(row) == MAIN_KEYS for row in rows)

    results = {row["result"] for row in rows}
    assert "PASS" in results
    assert "WARN" in results
    assert "FAIL" in results

    by_case = {row["case"]: row for row in rows}
    assert by_case["1.1 clean pass"]["result"] == "PASS"
    assert by_case["1.12 WARNING-only"]["result"] == "WARN"
    assert by_case["1.12 WARNING-only"]["W"] == "1"


def test_llm_ops_sanitize_rows_show_before_after():
    demo = _load_internals()

    rows = demo.build_sanitize_rows()
    by_case = {row["case"]: row for row in rows}

    assert by_case["code fence"]["sanitized"] == "你好"
    assert by_case["leading punctuation"]["sanitized"] == "你好"
    assert by_case["quoted"]["sanitized"] == "你好"
    assert by_case["code fence"]["target"] == "`你好`"


def test_llm_ops_language_profile_rows_use_main_schema():
    demo = _load_internals()

    rows = demo.build_language_profile_rows()
    assert all(set(row) == MAIN_KEYS for row in rows)

    cases = [row["case"] for row in rows]
    assert any(c.startswith("en->zh") for c in cases)
    assert any(c.startswith("zh->en") for c in cases)
    assert any("latin->cjk" in c for c in cases)
    assert any("cjk->latin" in c for c in cases)


def test_llm_ops_resolved_scene_rows():
    demo = _load_internals()

    rows = demo.build_resolved_scene_rows()
    assert {"kind", "name", "severity", "params"} <= set(rows[0])
    kinds = {row["kind"] for row in rows}
    assert kinds == {"sanitize", "check"}
