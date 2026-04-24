"""Unified WhisperX sanitization pipeline — fast and tracked paths share one impl."""

from __future__ import annotations

from domain.model import Word

from .._reporting import RuleHit
from .rules import (
    w1_dedup_untimed,
    w2_interpolate_timestamps,
    w3_collapse_repeats,
    w4_replace_long_words,
    w5_attach_punctuation,
)
from .types import WhisperXReport, WordReport


def _run_pipeline(
    word_segments: list[dict],
    *,
    track: dict[int, list[RuleHit]] | None = None,
) -> tuple[list[dict], list[int]]:
    """Run the canonical WhisperX sanitization pipeline.

    Pipeline order matches the pre-refactor ``sanitize_whisperx`` exactly:
        W1 dedup → W2 interpolate → W5 attach-punct → W3 (2-gram) → W3 (3-gram) → W4 long-words.
    """
    origins = list(range(len(word_segments)))
    ws: list[dict] = list(word_segments)
    ws, origins = w1_dedup_untimed(ws, origins, track=track)
    ws, origins = w2_interpolate_timestamps(ws, origins, track=track)
    ws, origins = w5_attach_punctuation(ws, origins, track=track)
    ws, origins = w3_collapse_repeats(ws, origins, pattern_len=2, min_repeats=4, track=track)
    ws, origins = w3_collapse_repeats(ws, origins, pattern_len=3, min_repeats=4, track=track)
    ws, origins = w4_replace_long_words(ws, origins, track=track)
    return ws, origins


def sanitize(
    word_segments: list[dict],
    *,
    track: dict[int, list[RuleHit]] | None = None,
) -> list[Word]:
    """Sanitize raw WhisperX word dicts and return ``Word`` objects.

    ``track=None`` is the fast path. Pass a dict to collect ``RuleHit``
    per origin-index for ``sanitize_with_report``.
    """
    if not word_segments:
        return []

    ws, _origins = _run_pipeline(word_segments, track=track)
    return [
        Word(
            word=w["word"].strip(),
            start=w["start"],
            end=w["end"],
            speaker=w.get("speaker"),
        )
        for w in ws
        if w["word"].strip()
    ]


def sanitize_whisperx(word_segments: list[dict]) -> list[Word]:
    """Fast-path sanitizer — identical semantics to the pre-refactor function."""
    return sanitize(word_segments, track=None)


def sanitize_with_report(
    word_segments: list[dict],
) -> tuple[list[Word], WhisperXReport]:
    """Sanitize and build a per-word report.

    Each input word dict becomes a :class:`WordReport`, with ``index_out=None``
    when the word was dropped / merged out by any rule. The final ``list[Word]``
    is identical to :func:`sanitize_whisperx` on the same input.
    """
    if not word_segments:
        return [], WhisperXReport(words=[], words_in=0, words_out=0, rule_counts={})

    track: dict[int, list[RuleHit]] = {}
    ws, origins = _run_pipeline(word_segments, track=track)

    # Origin → index in the pre-final-filter output list.
    origin_to_out_pos: dict[int, int] = {}
    final_words: list[Word] = []
    for pos, (w, origin) in enumerate(zip(ws, origins)):
        stripped = w["word"].strip()
        if not stripped:
            continue
        origin_to_out_pos[origin] = len(final_words)
        final_words.append(
            Word(
                word=stripped,
                start=w["start"],
                end=w["end"],
                speaker=w.get("speaker"),
            )
        )

    reports: list[WordReport] = []
    rule_counts: dict[str, int] = {}
    for idx, raw in enumerate(word_segments):
        out_pos = origin_to_out_pos.get(idx)
        if out_pos is not None:
            final = final_words[out_pos]
            word_out, start_out, end_out = final.word, final.start, final.end
            index_out: int | None = out_pos
        else:
            word_out = ""
            start_out = None
            end_out = None
            index_out = None
        steps = track.get(idx, [])
        for h in steps:
            rule_counts[h.rule_id] = rule_counts.get(h.rule_id, 0) + 1
        reports.append(
            WordReport(
                index_in=idx,
                index_out=index_out,
                word_in=raw.get("word", ""),
                word_out=word_out,
                start_in=raw.get("start"),
                end_in=raw.get("end"),
                start_out=start_out,
                end_out=end_out,
                steps=steps,
            )
        )

    return final_words, WhisperXReport(
        words=reports,
        words_in=len(word_segments),
        words_out=len(final_words),
        rule_counts=rule_counts,
    )


__all__ = ["sanitize", "sanitize_whisperx", "sanitize_with_report"]
