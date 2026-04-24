"""Tests for WhisperX sanitization reporting (W-rule tracking)."""

from __future__ import annotations

from adapters.parsers.whisperx import WhisperXReport, WordReport, format_report, report_to_jsonl, sanitize_whisperx, sanitize_whisperx_with_report


def _w(word: str, start: float | None = None, end: float | None = None, score: float = 0.5) -> dict:
    d: dict = {"word": word}
    if start is not None:
        d["start"] = start
        d["end"] = end
        d["score"] = score
    return d


def test_tracking_collapse_and_attach_punct():
    # W3 collapses A,B repeated ≥4 times; W5 attaches the trailing ','.
    raw = [_w("A", 0.0, 0.1), _w("B", 0.1, 0.2), _w("A", 0.2, 0.3), _w("B", 0.3, 0.4), _w("A", 0.4, 0.5), _w("B", 0.5, 0.6), _w("A", 0.6, 0.7), _w("B", 0.7, 0.8), _w("hello", 0.8, 0.9), _w(",", 0.9, 1.0)]

    fast = sanitize_whisperx(raw)
    words, report = sanitize_whisperx_with_report(raw)

    # Tracked path must produce identical final Word list.
    assert [(w.word, w.start, w.end) for w in words] == [(w.word, w.start, w.end) for w in fast]

    assert isinstance(report, WhisperXReport)
    assert report.words_in == len(raw)
    assert report.words_out < report.words_in  # collapse + attach drop words
    assert len(report.rule_counts) > 0
    # Both W3 (collapse) and W5 (attach-punct) fired.
    assert "W3" in report.rule_counts
    assert "W5" in report.rule_counts

    # At least one dropped / merged word has index_out=None.
    assert any(wr.index_out is None for wr in report.words)
    # Every input word has a WordReport.
    assert len(report.words) == len(raw)
    assert all(isinstance(wr, WordReport) for wr in report.words)


def test_format_report_and_jsonl_nonempty():
    raw = [_w("A", 0.0, 0.1), _w("B", 0.1, 0.2), _w("A", 0.2, 0.3), _w("B", 0.3, 0.4), _w("A", 0.4, 0.5), _w("B", 0.5, 0.6), _w("A", 0.6, 0.7), _w("B", 0.7, 0.8), _w("word", 0.8, 0.9), _w(".", 0.9, 1.0)]
    _, report = sanitize_whisperx_with_report(raw)

    text = format_report(report, path="x.json", level="full")
    assert isinstance(text, str) and text
    assert "WHISPERX SUMMARY" in text

    lines = report_to_jsonl(report, path="x.json")
    assert len(lines) >= 1
    assert lines[-1].startswith('{"type": "summary"')


def test_empty_input_returns_empty_report():
    words, report = sanitize_whisperx_with_report([])
    assert words == []
    assert report.words_in == 0
    assert report.words_out == 0
    assert report.rule_counts == {}


def test_fast_vs_tracked_parity_on_interpolation():
    # W2 interpolation + W1 dedup of untimed duplicates.
    raw = [
        _w("hello", 0.0, 0.5),
        _w("um"),  # untimed
        _w("um"),  # untimed duplicate — dropped by W1
        _w("world", 1.0, 1.5),
    ]
    fast = sanitize_whisperx(raw)
    tracked, report = sanitize_whisperx_with_report(raw)
    assert [(w.word, round(w.start, 6), round(w.end, 6)) for w in tracked] == [(w.word, round(w.start, 6), round(w.end, 6)) for w in fast]
    # Expect W1 drop recorded and W2 interpolation recorded.
    assert "W1" in report.rule_counts
    assert "W2" in report.rule_counts
