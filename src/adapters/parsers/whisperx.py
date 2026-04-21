"""WhisperX JSON parser and word-level sanitizer.

WhisperX produces word-level timestamps via forced alignment, but the output
contains systematic quality issues (see docs/srt-issues.md, Part B).  This
module sanitizes raw ``word_segments`` before they enter the subtitle pipeline.

Pipeline order::

    raw dicts → dedup_untimed → interpolate_timestamps → attach_punctuation
              → collapse_repeats → replace_long_words → list[Word]
"""

from __future__ import annotations

import json
import string
from pathlib import Path

from domain.model import Word


# ── Repeating-pattern detection ───────────────────────────────────────


def _collapse_repeats(
    words: list[dict],
    pattern_len: int = 2,
    min_repeats: int = 4,
) -> list[dict]:
    """Collapse N-gram patterns that repeat ≥ *min_repeats* times.

    For example, with *pattern_len=2* and *min_repeats=4*::

        [A, B, A, B, A, B, A, B, A, B]  →  [A, B]
    """
    if not words:
        return words

    result: list[dict] = []
    i = 0
    n = len(words)

    while i < n:
        # Count how many times the pattern at [i : i+pattern_len] repeats
        repeat_count = 1
        j = i + pattern_len
        while j + pattern_len <= n:
            match = True
            for k in range(pattern_len):
                if words[j + k]["word"] != words[i + k]["word"]:
                    match = False
                    break
            if not match:
                break
            repeat_count += 1
            j += pattern_len

        if repeat_count >= min_repeats:
            # Keep only one occurrence
            result.extend(words[i : i + pattern_len])
            i = j
        else:
            # No repeat — advance by 1 to avoid skipping a valid pattern start
            result.append(words[i])
            i += 1

    return result


# ── Core sanitization steps ───────────────────────────────────────────


def _dedup_untimed(words: list[dict]) -> list[dict]:
    """Remove consecutive duplicate words that have no timestamps.

    WhisperX sometimes emits the same untimed word multiple times in a
    row (e.g. ``♪ ♪ ♪ ♪``).  We keep the first occurrence and drop
    the rest.  Timed duplicates are always kept.
    """
    if not words:
        return words

    result = [words[0]]
    for w in words[1:]:
        prev = result[-1]
        # Both untimed and same word → skip
        if prev.get("start") is None and w.get("start") is None and prev["word"] == w["word"]:
            continue
        result.append(w)
    return result


def _interpolate_timestamps(words: list[dict]) -> list[dict]:
    """Fill in missing ``start``/``end`` for untimed words.

    Uses a running char-rate estimate (total duration / total chars so
    far) to assign durations proportional to word length.  Untimed words
    are placed between the previous word's ``end`` and the next timed
    word's ``start``.
    """
    if not words:
        return words

    result: list[dict] = []
    total_duration = 0.0
    total_chars = 1e-7  # avoid division by zero

    for idx, word in enumerate(words):
        if word.get("start") is not None:
            # Already timed — just update running stats
            result.append(word)
            total_duration += word["end"] - word["start"]
            total_chars += len(word["word"])
            continue

        # Find the previous end time
        prev_end = result[-1]["end"] if result else 0.0

        # Find the next timed word's start
        next_start = None
        for j in range(idx + 1, len(words)):
            if words[j].get("start") is not None:
                next_start = words[j]["start"]
                break

        # If prev_end == next_start, steal a small gap from the previous word
        if next_start is not None and abs(prev_end - next_start) < 1e-6:
            if result:
                steal = min(1.0, result[-1]["end"] - result[-1]["start"]) * 0.01
                prev_end = prev_end - steal
                result[-1] = {**result[-1], "end": prev_end}

        # Assign proportional duration
        char_rate = total_duration / total_chars
        estimated = char_rate * len(word["word"])
        upper = next_start if next_start is not None else prev_end + estimated
        end = min(upper, prev_end + estimated)

        patched = {**word, "start": prev_end, "end": end, "score": 0.0}
        result.append(patched)
        total_duration += end - prev_end
        total_chars += len(word["word"])

    return result


def _attach_punctuation(words: list[dict]) -> list[dict]:
    """Merge standalone punctuation tokens into the preceding word.

    WhisperX sometimes emits punctuation as separate tokens
    (e.g. ``["word", "."]``).  We attach them to the previous word,
    extending its ``end`` time.
    """
    if not words:
        return words

    result: list[dict] = []
    for w in words:
        text = w["word"].strip()
        if all(c in string.punctuation for c in text) and result:
            # Merge into previous word
            prev = result[-1]
            result[-1] = {
                **prev,
                "word": prev["word"] + w["word"],
                "end": w["end"],
            }
        else:
            result.append(w)
    return result


def _replace_long_words(
    words: list[dict],
    max_len: int = 30,
) -> list[dict]:
    """Replace abnormally long word text with ``...``.

    WhisperX occasionally concatenates multiple words into one token
    (e.g. ``ENVIRONMENTALISTENVIRONMENTALIST``).  Words longer than
    *max_len* that are all-uppercase (or longer than 50 chars) are
    replaced.
    """
    import re

    result: list[dict] = []
    for w in words:
        text = w["word"].strip()
        if len(text) <= max_len:
            result.append(w)
            continue

        alpha_words = re.findall(r"[A-Za-z]+", text)
        all_upper = all(aw == aw.upper() for aw in alpha_words) if alpha_words else False

        if all_upper or len(text) > 50:
            result.append({**w, "word": "..."})
        else:
            result.append(w)
    return result


# ── Public API ────────────────────────────────────────────────────────


def sanitize_whisperx(word_segments: list[dict]) -> list[Word]:
    """Sanitize raw WhisperX word dicts and return ``Word`` objects.

    Applies the full sanitization pipeline:
    1. Deduplicate consecutive untimed words
    2. Interpolate missing timestamps (char-rate based)
    3. Attach standalone punctuation to adjacent words
    4. Collapse repeating 2/3-gram patterns (≥4 repeats)
    5. Replace abnormally long word text

    Args:
        word_segments: Raw dicts from WhisperX JSON ``word_segments``.
            Each dict has ``word`` (str), optional ``start``/``end``
            (float), optional ``score`` (float).

    Returns:
        Sanitized list of ``Word`` objects with valid timestamps.
    """
    if not word_segments:
        return []

    ws = _dedup_untimed(word_segments)
    ws = _interpolate_timestamps(ws)
    ws = _attach_punctuation(ws)
    ws = _collapse_repeats(ws, pattern_len=2, min_repeats=4)
    ws = _collapse_repeats(ws, pattern_len=3, min_repeats=4)
    ws = _replace_long_words(ws)

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


def parse_whisperx(data: dict) -> list[Word]:
    """Parse a WhisperX JSON dict into sanitized ``Word`` objects.

    Args:
        data: Parsed JSON dict with a ``word_segments`` key.

    Returns:
        Sanitized list of ``Word`` objects.

    Raises:
        KeyError: If ``word_segments`` is missing.
        ValueError: If ``word_segments`` is empty.
    """
    ws = data.get("word_segments")
    if ws is None:
        raise KeyError("Missing 'word_segments' in WhisperX JSON")
    if not ws:
        raise ValueError("Empty 'word_segments' in WhisperX JSON")
    return sanitize_whisperx(ws)


def read_whisperx(path: str | Path) -> list[Word]:
    """Read a WhisperX JSON file and return sanitized ``Word`` objects.

    Args:
        path: Path to the JSON file.

    Returns:
        Sanitized list of ``Word`` objects.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return parse_whisperx(data)
