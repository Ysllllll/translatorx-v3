"""Extract raw word-dict list from a WhisperX JSON payload.

WhisperX outputs two top-level keys:

- ``segments``      — list of segment dicts, each with ``text``,
  ``start``, ``end`` and an optional inner ``words`` list (the result
  of forced alignment).
- ``word_segments`` — flat list of word dicts (the inner ``words``
  concatenated in order). Segments whose forced-alignment failed are
  *missing* from ``word_segments``, which silently drops the audio
  region they cover.

To preserve those regions we walk ``segments`` and:

1. when a segment has inner ``words`` → reuse them as-is;
2. when it doesn't → synthesize one dict per whitespace-delimited
   token of ``segment.text`` with timestamps distributed evenly
   across ``[segment.start, segment.end]``.

The result is a homogeneous ``list[dict]`` matching the historic
``word_segments`` shape, so the existing W1–W5 sanitization pipeline
applies unchanged.
"""

from __future__ import annotations

from typing import Any


def _tokenize(text: str) -> list[str]:
    """Whitespace-tokenize ``text``; fall back to single-token for CJK without spaces."""
    tokens = text.split()
    return tokens if tokens else ([text] if text.strip() else [])


def _synthesize_words(seg: dict[str, Any]) -> list[dict[str, Any]]:
    """Create per-token dicts for a segment that lacks word-level alignment.

    Time is distributed evenly across ``[start, end]``. Returns an
    empty list when the segment has no usable text or timing.
    """
    text = (seg.get("text") or "").strip()
    if not text:
        return []
    start = seg.get("start")
    end = seg.get("end")
    if start is None or end is None or end < start:
        return []

    tokens = _tokenize(text)
    if not tokens:
        return []

    n = len(tokens)
    duration = float(end) - float(start)
    speaker = seg.get("speaker")

    out: list[dict[str, Any]] = []
    if duration <= 0 or n == 1:
        item: dict[str, Any] = {"word": tokens[0] if tokens else text, "start": start, "end": end}
        if speaker is not None:
            item["speaker"] = speaker
        out.append(item)
        # If duration is zero but we have multiple tokens, collapse
        # them into one to avoid degenerate intervals.
        if n > 1:
            out[-1]["word"] = " ".join(tokens)
        return out

    step = duration / n
    for i, tok in enumerate(tokens):
        s = float(start) + i * step
        e = float(start) + (i + 1) * step if i < n - 1 else float(end)
        item = {"word": tok, "start": s, "end": e}
        if speaker is not None:
            item["speaker"] = speaker
        out.append(item)
    return out


def _segment_words(seg: dict[str, Any]) -> list[dict[str, Any]]:
    """Return word dicts for a single segment.

    Prefers ``segment.words`` when present and non-empty; otherwise
    synthesizes from ``segment.text``. Word dicts inherit the segment's
    ``speaker`` when their own is missing.
    """
    inner = seg.get("words")
    if isinstance(inner, list) and inner:
        speaker = seg.get("speaker")
        out: list[dict[str, Any]] = []
        for w in inner:
            if not isinstance(w, dict):
                continue
            if speaker is not None and "speaker" not in w:
                w = {**w, "speaker": speaker}
            out.append(w)
        if out:
            return out
        # Fall through to synthesis if every entry was unusable.
    return _synthesize_words(seg)


def extract_word_dicts(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract a homogeneous ``list[dict]`` of words from WhisperX JSON.

    The order follows ``segments`` so that newly-synthesized words are
    interleaved at the correct point in the timeline. When ``segments``
    is absent we fall back to the legacy top-level ``word_segments``
    list for backward compatibility.
    """
    segments = data.get("segments")
    if isinstance(segments, list) and segments:
        out: list[dict[str, Any]] = []
        for seg in segments:
            if isinstance(seg, dict):
                out.extend(_segment_words(seg))
        if out:
            return out

    ws = data.get("word_segments")
    if isinstance(ws, list):
        return list(ws)
    return []


__all__ = ["extract_word_dicts"]
