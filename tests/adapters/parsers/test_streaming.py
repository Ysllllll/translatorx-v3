"""Streaming-mode tests for SRT and WhisperX pipelines.

Verifies:
  * ``feed_many + flush == batch`` (bit-identical output).
  * Partial emission during streaming (not everything emitted at flush).
  * RecordingTracker hit counts match between batch and streaming.
"""

from __future__ import annotations

from adapters.parsers import clean_srt, clean_stream, sanitize_stream, sanitize_whisperx
from adapters.parsers.engine import RecordingTracker
from adapters.parsers.srt import Cue, default_pipeline as srt_default_pipeline
from adapters.parsers.whisperx.pipeline import _dict_to_word, default_pipeline as whisperx_default_pipeline


# ── SRT streaming ─────────────────────────────────────────────────────


def _make_cues() -> list[Cue]:
    """10 cues with a zero-duration pair at the same timepoint."""
    return [
        Cue(1_000, 2_000, "hello  world"),
        Cue(2_000, 3_000, "<b>bold</b> text"),
        Cue(3_000, 4_000, "smart \u201cquotes\u201d here"),
        Cue(4_000, 5_000, "ellipsis\u2026"),
        # Zero-duration pair at same timepoint, followed by a real cue.
        Cue(5_000, 5_000, "zero A"),
        Cue(5_000, 5_000, "zero B"),
        Cue(5_000, 6_000, "real cue"),
        Cue(6_000, 7_000, "entities &amp; Tom"),
        Cue(7_000, 8_000, "double  space"),
        Cue(8_000, 9_000, "final cue"),
    ]


def test_srt_stream_matches_batch():
    cues = _make_cues()
    pipe = srt_default_pipeline()

    # Batch.
    batch_out, _ = pipe.run([Cue(c.start_ms, c.end_ms, c.text, c.note) for c in cues])
    batch_tuples = [(c.start_ms, c.end_ms, c.text) for c in batch_out]

    # Stream.
    session = clean_stream()
    feed_copies = [Cue(c.start_ms, c.end_ms, c.text, c.note) for c in cues]
    partial: list[Cue] = []
    for cue in feed_copies:
        partial.extend(session.feed(cue))
    final = session.flush()
    stream_out = partial + final
    stream_tuples = [(c.start_ms, c.end_ms, c.text) for c in stream_out]

    assert stream_tuples == batch_tuples


def test_srt_stream_emits_progressively():
    """feed() should emit some items before flush()."""
    cues = _make_cues()
    session = clean_stream()
    partial: list[Cue] = []
    for cue in cues:
        partial.extend(session.feed(Cue(cue.start_ms, cue.end_ms, cue.text, cue.note)))
    # At least one item emitted before the final flush.
    assert len(partial) > 0
    _ = session.flush()


def test_srt_stream_tracker_matches_batch():
    cues = _make_cues()
    pipe = srt_default_pipeline()

    # Batch tracker.
    batch_tracker = RecordingTracker()
    pipe.run([Cue(c.start_ms, c.end_ms, c.text, c.note) for c in cues], tracker=batch_tracker)

    # Streaming tracker.
    stream_tracker = RecordingTracker()
    session = pipe.stream(tracker=stream_tracker)
    for cue in cues:
        session.feed(Cue(cue.start_ms, cue.end_ms, cue.text, cue.note))
    session.flush()

    assert stream_tracker.rule_counts == batch_tracker.rule_counts


# ── WhisperX streaming ────────────────────────────────────────────────


def _w(word, start=None, end=None, score=0.5):
    d = {"word": word}
    if start is not None:
        d["start"] = start
        d["end"] = end
        d["score"] = score
    return d


def _make_words() -> list[dict]:
    """15 word dicts with repeats and untimed gaps."""
    return [
        _w("hello", 0.0, 0.5),
        _w("world", 0.5, 1.0),
        _w("um"),
        _w("um"),  # untimed duplicate -> W1 drop
        _w("okay", 1.0, 1.2),
        _w("A", 1.2, 1.3),
        _w("B", 1.3, 1.4),
        _w("A", 1.4, 1.5),
        _w("B", 1.5, 1.6),
        _w("A", 1.6, 1.7),
        _w("B", 1.7, 1.8),
        _w("A", 1.8, 1.9),
        _w("B", 1.9, 2.0),
        _w("end", 2.0, 2.2),
        _w(".", 2.2, 2.3),
    ]


def test_whisperx_stream_matches_batch():
    raws = _make_words()

    batch = sanitize_whisperx([dict(w) for w in raws])

    # Stream uses dicts; convert to Word for comparison.
    session = sanitize_stream()
    partial_dicts: list[dict] = []
    for w in raws:
        partial_dicts.extend(session.feed(dict(w)))
    partial_dicts.extend(session.flush())

    stream_words = []
    for d in partial_dicts:
        wo = _dict_to_word(d)
        if wo is not None:
            stream_words.append(wo)

    assert [(w.word, w.start, w.end) for w in stream_words] == [(w.word, w.start, w.end) for w in batch]


def test_whisperx_stream_emits_progressively():
    raws = _make_words()
    session = sanitize_stream()
    partial: list[dict] = []
    for w in raws:
        partial.extend(session.feed(dict(w)))
    assert len(partial) > 0
    _ = session.flush()


def test_whisperx_stream_tracker_matches_batch():
    raws = _make_words()
    pipe = whisperx_default_pipeline()

    batch_tracker = RecordingTracker()
    pipe.run([dict(w) for w in raws], tracker=batch_tracker)

    stream_tracker = RecordingTracker()
    session = pipe.stream(tracker=stream_tracker)
    for w in raws:
        session.feed(dict(w))
    session.flush()

    assert stream_tracker.rule_counts == batch_tracker.rule_counts


# ── sanity: clean_srt text-parsing path unaffected ────────────────────


def test_clean_srt_still_works():
    src = "1\n00:00:01,000 --> 00:00:02,000\nHello &amp; world\n"
    result = clean_srt(src)
    assert result.ok
    assert result.cues[0].text == "Hello & world"
