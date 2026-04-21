"""Tests for :mod:`runtime.reporters`."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from domain.model import SentenceRecord
from adapters.reporters.reporters import (
    ChainReporter,
    JsonlErrorReporter,
    LoggerReporter,
)
from application.observability.errors import ErrorInfo


def _make_err(**kw) -> ErrorInfo:
    defaults: dict = dict(
        processor="T",
        category="permanent",
        code="oops",
        message="boom",
        retryable=False,
        attempts=1,
        at=1700000000.0,
        cause="ValueError: x",
    )
    defaults.update(kw)
    return ErrorInfo(**defaults)


def _make_rec() -> SentenceRecord:
    return SentenceRecord(
        src_text="hello",
        start=0.0,
        end=1.0,
        extra={"stream_id": 42},
    )


# ---------------------------------------------------------------------------
# LoggerReporter
# ---------------------------------------------------------------------------


def test_logger_reporter_uses_level_map(caplog) -> None:
    lg = logging.getLogger("test.runtime.reporter")
    lg.setLevel(logging.DEBUG)
    reporter = LoggerReporter(logger=lg)

    with caplog.at_level(logging.DEBUG, logger="test.runtime.reporter"):
        reporter.report(_make_err(category="permanent"), _make_rec(), {})
        reporter.report(_make_err(category="degraded", code="missing"), _make_rec(), {})
        reporter.report(_make_err(category="transient", code="retry"), _make_rec(), {})

    levels = [r.levelno for r in caplog.records if r.name == "test.runtime.reporter"]
    assert logging.ERROR in levels
    assert logging.WARNING in levels
    assert logging.INFO in levels


def test_logger_reporter_custom_level_map(caplog) -> None:
    lg = logging.getLogger("test.runtime.reporter.custom")
    lg.setLevel(logging.DEBUG)
    reporter = LoggerReporter(logger=lg, level_map={"permanent": logging.CRITICAL})
    with caplog.at_level(logging.DEBUG, logger="test.runtime.reporter.custom"):
        reporter.report(_make_err(category="permanent"), _make_rec(), {})
    levels = [r.levelno for r in caplog.records if r.name == "test.runtime.reporter.custom"]
    assert logging.CRITICAL in levels


# ---------------------------------------------------------------------------
# JsonlErrorReporter
# ---------------------------------------------------------------------------


def test_jsonl_reporter_writes_line(tmp_path: Path) -> None:
    log_file = tmp_path / "errors.jsonl"
    reporter = JsonlErrorReporter(log_file)
    try:
        reporter.report(
            _make_err(),
            _make_rec(),
            {"video": "lec1", "course": "cs101", "fingerprint": "abc123"},
        )
    finally:
        reporter.close()

    content = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(content) == 1
    row = json.loads(content[0])
    assert row["video"] == "lec1"
    assert row["course"] == "cs101"
    assert row["record_id"] == 42
    assert row["processor"] == "T"
    assert row["category"] == "permanent"
    assert row["code"] == "oops"
    assert row["fingerprint"] == "abc123"
    assert row["cause"] == "ValueError: x"


def test_jsonl_reporter_creates_parent_dir(tmp_path: Path) -> None:
    log_file = tmp_path / "sub" / "deeper" / "errors.jsonl"
    reporter = JsonlErrorReporter(log_file)
    try:
        reporter.report(_make_err(), _make_rec(), {})
    finally:
        reporter.close()
    assert log_file.exists()


def test_jsonl_reporter_category_filter(tmp_path: Path) -> None:
    log_file = tmp_path / "errors.jsonl"
    reporter = JsonlErrorReporter(log_file, categories={"permanent", "fatal"})
    try:
        reporter.report(_make_err(category="transient"), _make_rec(), {})
        reporter.report(_make_err(category="permanent"), _make_rec(), {})
        reporter.report(_make_err(category="degraded"), _make_rec(), {})
    finally:
        reporter.close()

    rows = [json.loads(line) for line in log_file.read_text(encoding="utf-8").strip().splitlines()]
    assert len(rows) == 1
    assert rows[0]["category"] == "permanent"


def test_jsonl_reporter_handles_unicode(tmp_path: Path) -> None:
    log_file = tmp_path / "errors.jsonl"
    reporter = JsonlErrorReporter(log_file)
    try:
        reporter.report(_make_err(message="中文 错误 メッセージ"), _make_rec(), {"video": "视频"})
    finally:
        reporter.close()
    row = json.loads(log_file.read_text(encoding="utf-8").strip())
    assert row["message"] == "中文 错误 メッセージ"
    assert row["video"] == "视频"


# ---------------------------------------------------------------------------
# ChainReporter
# ---------------------------------------------------------------------------


class _CapturingReporter:
    def __init__(self) -> None:
        self.calls: list[ErrorInfo] = []

    def report(self, err, record, context) -> None:
        self.calls.append(err)


class _FailingReporter:
    def report(self, err, record, context) -> None:
        raise RuntimeError("I am broken")


def test_chain_fans_out_to_all() -> None:
    a, b = _CapturingReporter(), _CapturingReporter()
    chain = ChainReporter([a, b])
    chain.report(_make_err(), _make_rec(), {})
    assert len(a.calls) == 1
    assert len(b.calls) == 1


def test_chain_swallows_individual_failures() -> None:
    good = _CapturingReporter()
    chain = ChainReporter([_FailingReporter(), good, _FailingReporter()])
    # Must not raise.
    chain.report(_make_err(), _make_rec(), {})
    assert len(good.calls) == 1
