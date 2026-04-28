"""Smoke-test the checker demos' table data and YAML examples."""

from __future__ import annotations

import importlib.util
import importlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEMOS = ROOT / "demos"


def _load_module(name: str, path: Path):
    sys.path.insert(0, str(path.parent))
    sys.path.insert(0, str(DEMOS))
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for p in (str(DEMOS), str(path.parent)):
            while p in sys.path:
                sys.path.remove(p)


def test_basics_checker_yaml_examples_parse_and_run():
    demo = _load_module("basics_checker_demo", DEMOS / "basics" / "checker.py")

    rows = demo.build_yaml_example_rows()

    assert [row["name"] for row in rows] == ["translate_lenient_yaml", "llm_response_yaml"]
    assert rows[0]["default_scene"] == "demo.translate.lenient"
    assert rows[0]["passed"] == "PASS"
    assert rows[0]["warnings"] == "1"
    assert rows[1]["default_scene"] == "demo.llm.response"
    assert rows[1]["passed"] == "PASS"


def test_basics_checker_case_rows_aggregate_issue_counts():
    demo = _load_module("basics_checker_demo_rows", DEMOS / "basics" / "checker.py")

    rows = demo.build_quickstart_rows()

    by_label = {row["case"]: row for row in rows}
    assert by_label["pass"]["passed"] == "PASS"
    assert by_label["empty"]["passed"] == "FAIL"
    assert by_label["empty"]["errors"] == "1"
    assert "non_empty" in by_label["empty"]["issues"]


def test_llm_ops_checker_matrix_rows_cover_pass_fail_and_warning():
    sys.path.insert(0, str(DEMOS))
    try:
        demo = importlib.import_module("internals.llm_ops.checker")
    finally:
        while str(DEMOS) in sys.path:
            sys.path.remove(str(DEMOS))

    rows = demo.build_rule_matrix_rows()

    assert len(rows) >= 12
    assert rows[0]["passed"] == "PASS"
    assert any(row["passed"] == "FAIL" and int(row["errors"]) > 0 for row in rows)
    assert any(row["passed"] == "PASS" and int(row["warnings"]) > 0 for row in rows)
