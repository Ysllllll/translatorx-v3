"""WhisperX sanitization rules as :class:`engine.Rule` ``[dict]`` instances.

Each rule operates on a raw ``word_segments`` list (list of dicts). A
rule's ``origin`` id is the word's index in the original input, stable
across every transformation (drops, merges, splits, replacements). This
is what lets :class:`~engine.RecordingTracker` attribute rule hits back
to the correct input word for report generation.

Legacy helpers ``_dedup_untimed`` / ``_interpolate_timestamps`` /
``_attach_punctuation`` / ``_collapse_repeats`` / ``_replace_long_words``
are retained as thin wrappers for tests that import them directly.
"""

from __future__ import annotations

import re
import string

from ..engine import Rule
from ..engine.rule import RuleHit


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


class W1DedupUntimed(Rule[dict]):
    id = "W1"
    reason = _reason("W1")
    lookahead = 1

    def apply(self, items, origins, *, tracker):
        if not items:
            return items, origins
        result: list[dict] = [items[0]]
        new_origins: list[int] = [origins[0]]
        for w, o in zip(items[1:], origins[1:]):
            prev = result[-1]
            if prev.get("start") is None and w.get("start") is None and prev["word"] == w["word"]:
                tracker.fire(self.id, self.reason, before=_fmt(w), after="<dropped>", origin=o)
                continue
            result.append(w)
            new_origins.append(o)
        return result, new_origins


# ── W2: interpolate timestamps ────────────────────────────────────────


class W2InterpolateTimestamps(Rule[dict]):
    id = "W2"
    reason = _reason("W2")
    # Conservative upper bound; W2 scans forward to the next timed word.
    lookahead = 10

    def apply(self, items, origins, *, tracker):
        if not items:
            return items, origins

        result: list[dict] = []
        new_origins: list[int] = []
        total_duration = 0.0
        total_chars = 1e-7

        for idx, (word, origin) in enumerate(zip(items, origins)):
            if word.get("start") is not None:
                result.append(word)
                new_origins.append(origin)
                total_duration += word["end"] - word["start"]
                total_chars += len(word["word"])
                continue

            prev_end = result[-1]["end"] if result else 0.0
            next_start = None
            for j in range(idx + 1, len(items)):
                if items[j].get("start") is not None:
                    next_start = items[j]["start"]
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
            tracker.fire(self.id, self.reason, before=before, after=_fmt(patched), origin=origin)
            total_duration += end - prev_end
            total_chars += len(word["word"])

        return result, new_origins


# ── W3: collapse repeats ──────────────────────────────────────────────


class W3CollapseRepeats(Rule[dict]):
    id = "W3"
    reason = _reason("W3")

    def __init__(self, *, pattern_len: int = 2, min_repeats: int = 4) -> None:
        self._pattern_len = pattern_len
        self._min_repeats = min_repeats
        self.lookahead = pattern_len * min_repeats

    def apply(self, items, origins, *, tracker):
        if not items:
            return items, origins
        pattern_len = self._pattern_len
        min_repeats = self._min_repeats
        result: list[dict] = []
        new_origins: list[int] = []
        i = 0
        n = len(items)

        while i < n:
            repeat_count = 1
            j = i + pattern_len
            while j + pattern_len <= n:
                match = True
                for k in range(pattern_len):
                    if items[j + k]["word"] != items[i + k]["word"]:
                        match = False
                        break
                if not match:
                    break
                repeat_count += 1
                j += pattern_len

            if repeat_count >= min_repeats:
                result.extend(items[i : i + pattern_len])
                new_origins.extend(origins[i : i + pattern_len])
                for k in range(i + pattern_len, j):
                    tracker.fire(
                        self.id,
                        self.reason,
                        before=_fmt(items[k]),
                        after="<collapsed repeat>",
                        origin=origins[k],
                    )
                i = j
            else:
                result.append(items[i])
                new_origins.append(origins[i])
                i += 1

        return result, new_origins


# ── W4: replace long words ────────────────────────────────────────────


class W4ReplaceLongWords(Rule[dict]):
    id = "W4"
    reason = _reason("W4")
    lookahead = 0

    def __init__(self, *, max_len: int = 30) -> None:
        self._max_len = max_len

    def apply(self, items, origins, *, tracker):
        max_len = self._max_len
        result: list[dict] = []
        new_origins: list[int] = []
        for w, o in zip(items, origins):
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
                tracker.fire(self.id, self.reason, before=before, after=_fmt(patched), origin=o)
            else:
                result.append(w)
                new_origins.append(o)
        return result, new_origins


# ── W5: attach punctuation ────────────────────────────────────────────


class W5AttachPunctuation(Rule[dict]):
    id = "W5"
    reason = _reason("W5")
    lookahead = 1

    def apply(self, items, origins, *, tracker):
        if not items:
            return items, origins

        result: list[dict] = []
        new_origins: list[int] = []
        for w, o in zip(items, origins):
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
                tracker.fire(
                    self.id,
                    self.reason,
                    before=_fmt(w),
                    after=f"<merged into {merged['word']!r}>",
                    origin=o,
                )
                tracker.fire(
                    self.id,
                    self.reason,
                    before=before_prev,
                    after=_fmt(merged),
                    origin=new_origins[-1],
                )
            else:
                result.append(w)
                new_origins.append(o)
        return result, new_origins


# ── Legacy callable wrappers (imported by existing tests) ─────────────


class _NullTracker:
    __slots__ = ()

    def fire(self, *args, **kwargs) -> None:
        return None


class _TrackShim:
    """Adapter that writes engine hits into a legacy ``dict[int, list[RuleHit]]``."""

    __slots__ = ("_track",)

    def __init__(self, track: dict[int, list[RuleHit]]) -> None:
        self._track = track

    def fire(self, rule_id, reason, *, before, after, origin):
        self._track.setdefault(origin, []).append(RuleHit(rule_id, reason, before, after))


def _apply_rule(rule: Rule[dict], words: list[dict]) -> list[dict]:
    out, _ = rule.apply(list(words), list(range(len(words))), tracker=_NullTracker())
    return out


def _dedup_untimed(words: list[dict]) -> list[dict]:
    return _apply_rule(W1DedupUntimed(), words)


def _interpolate_timestamps(words: list[dict]) -> list[dict]:
    return _apply_rule(W2InterpolateTimestamps(), words)


def _collapse_repeats(words: list[dict], pattern_len: int = 2, min_repeats: int = 4) -> list[dict]:
    return _apply_rule(W3CollapseRepeats(pattern_len=pattern_len, min_repeats=min_repeats), words)


def _replace_long_words(words: list[dict], max_len: int = 30) -> list[dict]:
    return _apply_rule(W4ReplaceLongWords(max_len=max_len), words)


def _attach_punctuation(words: list[dict]) -> list[dict]:
    return _apply_rule(W5AttachPunctuation(), words)


__all__ = [
    "W1DedupUntimed",
    "W2InterpolateTimestamps",
    "W3CollapseRepeats",
    "W4ReplaceLongWords",
    "W5AttachPunctuation",
    "_dedup_untimed",
    "_interpolate_timestamps",
    "_collapse_repeats",
    "_replace_long_words",
    "_attach_punctuation",
    "_WORD_REASONS",
    "_reason",
]
