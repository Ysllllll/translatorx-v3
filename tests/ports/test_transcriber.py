"""Tests for Transcriber Protocol + TranscriptionResult value types."""

from __future__ import annotations

from pathlib import Path

from domain.model import Segment
from ports.transcriber import TranscribeOptions, Transcriber, TranscriptionResult


class _StubTranscriber:
    async def transcribe(self, audio, opts=None):
        return TranscriptionResult(segments=[], language="en", duration=0.0)

    async def aclose(self) -> None:
        return None


def test_options_defaults():
    opts = TranscribeOptions()
    assert opts.language is None
    assert opts.word_timestamps is True
    assert opts.temperature == 0.0


def test_result_defaults():
    r = TranscriptionResult(segments=[Segment(start=0.0, end=1.0, text="hi")])
    assert r.language == ""
    assert r.duration == 0.0


def test_protocol_match():
    t = _StubTranscriber()
    assert isinstance(t, Transcriber)


def test_protocol_reject_non_transcriber():
    class NotIt:
        pass

    assert not isinstance(NotIt(), Transcriber)
