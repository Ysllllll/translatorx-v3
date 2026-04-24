"""WhisperX sanitization rules — each rule supports an optional tracker.

A tracker is a ``dict[int, list[RuleHit]]`` keyed by the **origin index** of
each input word (its position in the raw ``word_segments`` list). Rules take
both ``words`` and a parallel ``origins`` list so hits can be attributed back
to the original input word even after dicts are replaced / merged / dropped.

Each rule returns ``(new_words, new_origins)``.
"""

from __future__ import annotations

import re
import string

from .._reporting import RuleHit


_WORD_REASONS: dict[str, str] = {
    "W1": "去除无时间戳的连续重复词",
    "W2": "为缺失时间戳的词线性插值",
    "W3": "折叠高频重复的 N-gram（≥4 次）",
    "W4": "异常超长词（全大写 / >50 字符）替换为 '...'",
    "W5": "独立标点附着到前一个词",
}


def _reason(rule_id: str) -> str:
    return _WORD_REASONS.get(rule_id, "")


def _fmt(word: dict) -> str:
    t = word.get("word", "")
    s, e = word.get("start"), word.get("end")
    if s is None or e is None:
        return repr(t)
    return f"{t!r} [{s:.3f}..{e:.3f}]"


# ── W1: dedup untimed ──────────────────────────────────────────────────


def w1_dedup_untimed(
    words: list[dict],
    origins: list[int],
    track: dict[int, list[RuleHit]] | None = None,
) -> tuple[list[dict], list[int]]:
    if not words:
        return words, origins
    result: list[dict] = [words[0]]
    new_origins: list[int] = [origins[0]]
    for w, o in zip(words[1:], origins[1:]):
        prev = result[-1]
        if prev.get("start") is None and w.get("start") is None and prev["word"] == w["word"]:
            if track is not None:
                track.setdefault(o, []).append(RuleHit("W1", _reason("W1"), _fmt(w), "<dropped>"))
            continue
        result.append(w)
        new_origins.append(o)
    return result, new_origins


# ── W2: interpolate timestamps ────────────────────────────────────────


def w2_interpolate_timestamps(
    words: list[dict],
    origins: list[int],
    track: dict[int, list[RuleHit]] | None = None,
) -> tuple[list[dict], list[int]]:
    if not words:
        return words, origins

    result: list[dict] = []
    new_origins: list[int] = []
    total_duration = 0.0
    total_chars = 1e-7

    for idx, (word, origin) in enumerate(zip(words, origins)):
        if word.get("start") is not None:
            result.append(word)
            new_origins.append(origin)
            total_duration += word["end"] - word["start"]
            total_chars += len(word["word"])
            continue

        prev_end = result[-1]["end"] if result else 0.0
        next_start = None
        for j in range(idx + 1, len(words)):
            if words[j].get("start") is not None:
                next_start = words[j]["start"]
                break

        if next_start is not None and abs(prev_end - next_start) < 1e-6:
            if result:
                steal = min(1.0, result[-1]["end"] - result[-1]["start"]) * 0.01
                prev_end = prev_end - steal
                result[-1] = {**result[-1], "end": prev_end}

        char_rate = total_duration / total_chars
        estimated = char_rate * len(word["word"])
        upper = next_start if next_start is not None else prev_end + estimated
        end = min(upper, prev_end + estimated)

        before = _fmt(word)
        patched = {**word, "start": prev_end, "end": end, "score": 0.0}
        result.append(patched)
        new_origins.append(origin)
        if track is not None:
            track.setdefault(origin, []).append(RuleHit("W2", _reason("W2"), before, _fmt(patched)))
        total_duration += end - prev_end
        total_chars += len(word["word"])

    return result, new_origins


# ── W3: collapse repeats ──────────────────────────────────────────────


def w3_collapse_repeats(
    words: list[dict],
    origins: list[int],
    pattern_len: int = 2,
    min_repeats: int = 4,
    track: dict[int, list[RuleHit]] | None = None,
) -> tuple[list[dict], list[int]]:
    if not words:
        return words, origins

    result: list[dict] = []
    new_origins: list[int] = []
    i = 0
    n = len(words)

    while i < n:
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
            result.extend(words[i : i + pattern_len])
            new_origins.extend(origins[i : i + pattern_len])
            if track is not None:
                for k in range(i + pattern_len, j):
                    track.setdefault(origins[k], []).append(RuleHit("W3", _reason("W3"), _fmt(words[k]), "<collapsed repeat>"))
            i = j
        else:
            result.append(words[i])
            new_origins.append(origins[i])
            i += 1

    return result, new_origins


# ── W4: replace long words ────────────────────────────────────────────


def w4_replace_long_words(
    words: list[dict],
    origins: list[int],
    max_len: int = 30,
    track: dict[int, list[RuleHit]] | None = None,
) -> tuple[list[dict], list[int]]:
    result: list[dict] = []
    new_origins: list[int] = []
    for w, o in zip(words, origins):
        text = w["word"].strip()
        if len(text) <= max_len:
            result.append(w)
            new_origins.append(o)
            continue

        alpha_words = re.findall(r"[A-Za-z]+", text)
        all_upper = all(aw == aw.upper() for aw in alpha_words) if alpha_words else False

        if all_upper or len(text) > 50:
            before = _fmt(w)
            patched = {**w, "word": "..."}
            result.append(patched)
            new_origins.append(o)
            if track is not None:
                track.setdefault(o, []).append(RuleHit("W4", _reason("W4"), before, _fmt(patched)))
        else:
            result.append(w)
            new_origins.append(o)
    return result, new_origins


# ── W5: attach punctuation ────────────────────────────────────────────


def w5_attach_punctuation(
    words: list[dict],
    origins: list[int],
    track: dict[int, list[RuleHit]] | None = None,
) -> tuple[list[dict], list[int]]:
    if not words:
        return words, origins

    result: list[dict] = []
    new_origins: list[int] = []
    for w, o in zip(words, origins):
        text = w["word"].strip()
        if text and all(c in string.punctuation for c in text) and result:
            prev = result[-1]
            before_prev = _fmt(prev)
            merged = {
                **prev,
                "word": prev["word"] + w["word"],
                "end": w["end"],
            }
            result[-1] = merged
            if track is not None:
                track.setdefault(o, []).append(RuleHit("W5", _reason("W5"), _fmt(w), f"<merged into {merged['word']!r}>"))
                track.setdefault(new_origins[-1], []).append(RuleHit("W5", _reason("W5"), before_prev, _fmt(merged)))
        else:
            result.append(w)
            new_origins.append(o)
    return result, new_origins


# ── Legacy private names — kept as thin wrappers for backward compat ──


def _dedup_untimed(words: list[dict]) -> list[dict]:
    out, _ = w1_dedup_untimed(list(words), list(range(len(words))))
    return out


def _interpolate_timestamps(words: list[dict]) -> list[dict]:
    out, _ = w2_interpolate_timestamps(list(words), list(range(len(words))))
    return out


def _collapse_repeats(words: list[dict], pattern_len: int = 2, min_repeats: int = 4) -> list[dict]:
    out, _ = w3_collapse_repeats(list(words), list(range(len(words))), pattern_len, min_repeats)
    return out


def _replace_long_words(words: list[dict], max_len: int = 30) -> list[dict]:
    out, _ = w4_replace_long_words(list(words), list(range(len(words))), max_len)
    return out


def _attach_punctuation(words: list[dict]) -> list[dict]:
    out, _ = w5_attach_punctuation(list(words), list(range(len(words))))
    return out


__all__ = [
    "w1_dedup_untimed",
    "w2_interpolate_timestamps",
    "w3_collapse_repeats",
    "w4_replace_long_words",
    "w5_attach_punctuation",
    "_dedup_untimed",
    "_interpolate_timestamps",
    "_collapse_repeats",
    "_replace_long_words",
    "_attach_punctuation",
    "_WORD_REASONS",
    "_reason",
]
