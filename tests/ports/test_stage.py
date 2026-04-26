"""Tests for :mod:`ports.stage` — Stage Protocol surface."""

from __future__ import annotations

from typing import AsyncIterator

import pytest

from domain.model import SentenceRecord
from ports.stage import RecordStage, SourceStage, StageStatus, SubtitleStage


def _rec(idx: int) -> SentenceRecord:
    return SentenceRecord(src_text=f"r{idx}", start=0.0, end=1.0)


# ---------------------------------------------------------------------------
# StageStatus enum
# ---------------------------------------------------------------------------


def test_stage_status_has_six_states() -> None:
    expected = {"pending", "running", "completed", "failed", "cancelled", "skipped"}
    assert {s.value for s in StageStatus} == expected


def test_stage_status_is_str_enum() -> None:
    assert StageStatus.RUNNING == "running"


# ---------------------------------------------------------------------------
# SourceStage runtime_checkable conformance
# ---------------------------------------------------------------------------


class _ConcreteSource:
    name = "fake_source"

    async def open(self, ctx) -> None:
        return None

    def stream(self, ctx) -> AsyncIterator[SentenceRecord]:
        async def _gen():
            yield _rec(0)

        return _gen()

    async def close(self) -> None:
        return None


def test_source_stage_protocol_accepts_conformant() -> None:
    assert isinstance(_ConcreteSource(), SourceStage)


def test_source_stage_protocol_rejects_missing_methods() -> None:
    class _Bad:
        name = "bad"

        async def open(self, ctx) -> None: ...

    assert not isinstance(_Bad(), SourceStage)


# ---------------------------------------------------------------------------
# SubtitleStage runtime_checkable conformance
# ---------------------------------------------------------------------------


class _ConcreteSubtitleStage:
    name = "fake_subtitle"

    async def apply(self, records, ctx):
        return records


def test_subtitle_stage_protocol_accepts_conformant() -> None:
    assert isinstance(_ConcreteSubtitleStage(), SubtitleStage)


# ---------------------------------------------------------------------------
# RecordStage runtime_checkable conformance
# ---------------------------------------------------------------------------


class _ConcreteRecordStage:
    name = "fake_record"

    def transform(self, upstream, ctx):
        async def _gen():
            async for rec in upstream:
                yield rec

        return _gen()


def test_record_stage_protocol_accepts_conformant() -> None:
    assert isinstance(_ConcreteRecordStage(), RecordStage)


@pytest.mark.asyncio
async def test_record_stage_passthrough() -> None:
    stage = _ConcreteRecordStage()

    async def _src():
        for i in range(3):
            yield _rec(i)

    out = [r async for r in stage.transform(_src(), ctx=None)]
    assert [r.src_text for r in out] == ["r0", "r1", "r2"]
