"""Tests for :class:`runtime.base.ProcessorBase`."""

from __future__ import annotations

from typing import AsyncIterator

import pytest

from domain.model import SentenceRecord
from ports.processor import ProcessorBase


class _Concrete(ProcessorBase[SentenceRecord, SentenceRecord]):
    name = "concrete"

    def fingerprint(self) -> str:
        return "deadbeef"

    async def process(self, upstream, *, ctx, store, video_key):  # type: ignore[override]
        async for rec in upstream:
            yield rec


# ---------------------------------------------------------------------------
# Abstract enforcement
# ---------------------------------------------------------------------------


def test_cannot_instantiate_base_directly() -> None:
    with pytest.raises(TypeError):
        ProcessorBase()  # type: ignore[abstract]


def test_subclass_missing_abstract_fails() -> None:
    class _MissingProcess(ProcessorBase[SentenceRecord, SentenceRecord]):
        name = "bad"

        def fingerprint(self) -> str:
            return "x"

    with pytest.raises(TypeError):
        _MissingProcess()  # type: ignore[abstract]


def test_concrete_instantiates() -> None:
    p = _Concrete()
    assert p.name == "concrete"
    assert p.fingerprint() == "deadbeef"


# ---------------------------------------------------------------------------
# Default hooks
# ---------------------------------------------------------------------------


def test_output_is_stale_default_false() -> None:
    p = _Concrete()
    rec = SentenceRecord(src_text="hi", start=0.0, end=1.0)
    assert p.output_is_stale(rec) is False


@pytest.mark.asyncio
async def test_aclose_default_noop() -> None:
    p = _Concrete()
    assert await p.aclose() is None


# ---------------------------------------------------------------------------
# _missing_inputs
# ---------------------------------------------------------------------------


def test_missing_inputs_all_present() -> None:
    p = _Concrete()
    rec = SentenceRecord(
        src_text="hi",
        start=0.0,
        end=1.0,
        translations={"zh": "你好", "ja": "こんにちは"},
        extra={"foo": 1},
    )
    result = p._missing_inputs(
        rec,
        required_translations=("zh", "ja"),
        required_extra=("foo",),
    )
    assert result == []


def test_missing_inputs_reports_missing_and_empty() -> None:
    p = _Concrete()
    rec = SentenceRecord(
        src_text="hi",
        start=0.0,
        end=1.0,
        translations={"zh": "", "ja": "こんにちは"},  # zh present but empty
        extra={},
    )
    result = p._missing_inputs(
        rec,
        required_translations=("zh", "ko"),
        required_extra=("foo",),
    )
    # empty translation counts as missing, ko not in dict, foo not in extra
    assert set(result) == {"translations[zh]", "translations[ko]", "extra[foo]"}


# ---------------------------------------------------------------------------
# _record_with_error
# ---------------------------------------------------------------------------


def test_record_with_error_appends_error_info() -> None:
    p = _Concrete()
    rec = SentenceRecord(src_text="hi", start=0.0, end=1.0)
    new_rec = p._record_with_error(
        rec,
        category="permanent",
        code="oops",
        message="broken",
        attempts=2,
    )
    assert new_rec is not rec
    errors = new_rec.extra["errors"]
    assert len(errors) == 1
    err = errors[0]
    assert err.processor == "concrete"
    assert err.category == "permanent"
    assert err.code == "oops"
    assert err.attempts == 2
    # source record unchanged (frozen)
    assert "errors" not in rec.extra


def test_record_with_error_accumulates() -> None:
    p = _Concrete()
    rec = SentenceRecord(src_text="hi", start=0.0, end=1.0)
    rec = p._record_with_error(rec, category="transient", code="e1", message="first")
    rec = p._record_with_error(rec, category="permanent", code="e2", message="second")
    errors = rec.extra["errors"]
    assert [e.code for e in errors] == ["e1", "e2"]


def test_record_with_error_preserves_other_extras() -> None:
    p = _Concrete()
    rec = SentenceRecord(src_text="hi", start=0.0, end=1.0, extra={"terms_ready_at_translate": True})
    new_rec = p._record_with_error(rec, category="degraded", code="missing_input", message="need translations[zh]")
    assert new_rec.extra["terms_ready_at_translate"] is True
    assert len(new_rec.extra["errors"]) == 1


def test_record_with_error_cause_exception_formatted() -> None:
    p = _Concrete()
    rec = SentenceRecord(src_text="hi", start=0.0, end=1.0)
    cause = ValueError("bad input")
    new_rec = p._record_with_error(rec, category="permanent", code="e", message="m", cause=cause)
    assert new_rec.extra["errors"][0].cause == "ValueError: bad input"


def test_record_with_error_cause_string_passthrough() -> None:
    p = _Concrete()
    rec = SentenceRecord(src_text="hi", start=0.0, end=1.0)
    new_rec = p._record_with_error(rec, category="permanent", code="e", message="m", cause="raw cause")
    assert new_rec.extra["errors"][0].cause == "raw cause"
