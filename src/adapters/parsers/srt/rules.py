"""SRT rules — text cleaning + timestamp cleanup as :class:`engine.Rule` s.

This module consolidates what used to live in ``patterns.py``,
``rules.py``, ``clean_text.py``, and ``clean_timestamps.py``. The text
sub-pipeline is exposed both as a simple list of :class:`TextRule` (for
callers that want to run text cleaning on bare strings) and wrapped as a
single :class:`TextSweepRule` (a :class:`engine.Rule` ``[Cue]``) so it
composes with the timestamp rules.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Callable

from ..engine import Rule, Tracker
from ..engine.rule import RuleHit
from .model import CleanOptions, Cue, Issue


# ── patterns / constants ───────────────────────────────────────────────

_INVISIBLE_RE = re.compile(
    "["
    "\u00ad"
    "\u034f"
    "\u061c"
    "\u115f\u1160"
    "\u17b4\u17b5"
    "\u180e"
    "\u200b\u200c\u200d\u200e\u200f"
    "\u2028\u2029"
    "\u202a-\u202e"
    "\u2060-\u2064"
    "\u2066-\u206f"
    "\u3164"
    "\ufe00-\ufe0f"
    "\ufeff"
    "\uffa0"
    "\U000e0100-\U000e01ef"
    "\u007f"
    "]"
)

_WHITESPACE_MAP = str.maketrans(
    {
        "\u00a0": " ",
        "\u1680": " ",
        "\u2000": " ",
        "\u2001": " ",
        "\u2002": " ",
        "\u2003": " ",
        "\u2004": " ",
        "\u2005": " ",
        "\u2006": " ",
        "\u2007": " ",
        "\u2008": " ",
        "\u2009": " ",
        "\u200a": " ",
        "\u202f": " ",
        "\u205f": " ",
        "\u3000": " ",
    }
)

_SMART_QUOTE_MAP = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u201f": '"',
        "\u2032": "'",
        "\u2033": '"',
    }
)

_HTML_TAG_RE = re.compile(
    r"</?(?:i|b|u|s|br|em|strong|font|p|span|div)(?:\s[^>]*)?/?>",
    re.IGNORECASE,
)

_HTML_ENTITY_RE = re.compile(r"&(?:#x[0-9a-fA-F]+|#[0-9]+|[a-zA-Z]{2,8});")
_NAMED_ENTITIES: dict[str, str] = {
    "amp": "&",
    "lt": "<",
    "gt": ">",
    "quot": '"',
    "apos": "'",
    "nbsp": "\u00a0",
    "hellip": "\u2026",
    "ndash": "-",
    "mdash": "-",
    "lsquo": "'",
    "rsquo": "'",
    "ldquo": '"',
    "rdquo": '"',
    "prime": "'",
    "Prime": '"',
    "copy": "\u00a9",
    "reg": "\u00ae",
    "trade": "\u2122",
    "deg": "\u00b0",
}


def _entity_sub(match: re.Match[str]) -> str:
    s = match.group(0)
    body = s[1:-1]
    try:
        if body.startswith(("#x", "#X")):
            return chr(int(body[2:], 16))
        if body.startswith("#"):
            return chr(int(body[1:]))
        if body in _NAMED_ENTITIES:
            return _NAMED_ENTITIES[body]
    except (ValueError, OverflowError):
        pass
    return s


_MULTI_SPACE_RE = re.compile(r" {2,}")
_ELLIPSIS_RE = re.compile(r"\u2026")
_DOT_RUN_RE = re.compile(r"\.{2,}")
_TIMESTAMP_RE = re.compile(
    r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->\s*"
    r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})"
)

_ATTACH_PUNCTS = ",.!?:;"
_CJK_PUNCTS = "，。！？：；、"
_SPACE_BEFORE_PUNCT_RE = re.compile(rf" +([{re.escape(_ATTACH_PUNCTS + _CJK_PUNCTS)}])")
_COMMA_LIKE_RE = re.compile(r"(\w)([,:;!?])(?=[A-Za-z])")
_PERIOD_RE = re.compile(r"([A-Za-z]{2,})\.(?=[A-Z][a-z])")

_MIN_DURATION_MS = 50
_MAX_OVERLAP_FIX_MS = 100


# ── rule catalog ──────────────────────────────────────────────────────

_RULE_REASONS: dict[str, str] = {
    "E2": "多行文本用空格拼成单行",
    "E3": "首尾空白修剪",
    "E4": "清洗后文本为空，丢弃此条目",
    "C1": "剥离零宽/控制/双向标记等不可见字符",
    "C2": "各类 NBSP/全角空白规整为 ASCII 空格",
    "C3": "智能引号规整为 ASCII 引号",
    "C4": "单字符省略号 '…' 规整为 '...'",
    "C5": "连续点号（2 个或 ≥4 个）规整为 '...'",
    "C6": "剥离 HTML 标签 (i/b/u/s/br/em/strong/font/p/span/div)（格式标记，非内容）",
    "C7": "标点附着：移除标点前空白 / 标点后字母前补空格",
    "C8": "连续空格压缩为单个空格",
    "C9": "Tab 转空格，剩余不可打印控制字符移除",
    "C10": "HTML 实体解码（&amp;/&nbsp;/&lt; 等）",
    "T1": "零时长 cue，从邻近空档借用时长",
    "T1M": "零时长 cue 合并到同时间点的有效 cue",
    "T1M!": "零时长 cue 合并会超过显示上限，标记为不可修复",
    "T2": "轻微重叠，下调前一条 end 到后一条 start",
    "T3": "时间戳非法（负值或越界），丢弃此条目",
    "T4": "时间戳截断到最大允许值",
    "N1": "按顺序重编号 1..N",
}


def _rule(rule_id: str) -> str:
    return _RULE_REASONS.get(rule_id, "")


# ── text-level rules ──────────────────────────────────────────────────


@dataclass(frozen=True)
class TextRule:
    """One text cleaning rule applied within :func:`run_text_pipeline`."""

    id: str
    reason: str
    apply: Callable[[str], str]


def _apply_c9(text: str) -> str:
    text = text.replace("\t", " ")
    return "".join(ch for ch in text if ch == " " or unicodedata.category(ch)[0] != "C")


def _apply_c7(text: str) -> str:
    text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    return _COMMA_LIKE_RE.sub(r"\1\2 ", text)


# Order: C10 → C1 → C2 → C3 → C4 → C6 → C9 → C8 → C7 → C5 → C8 → E3
TEXT_RULES: tuple[TextRule, ...] = (
    TextRule("C10", _rule("C10"), lambda t: _HTML_ENTITY_RE.sub(_entity_sub, t)),
    TextRule("C1", _rule("C1"), lambda t: _INVISIBLE_RE.sub("", t)),
    TextRule("C2", _rule("C2"), lambda t: t.translate(_WHITESPACE_MAP)),
    TextRule("C3", _rule("C3"), lambda t: t.translate(_SMART_QUOTE_MAP)),
    TextRule("C4", _rule("C4"), lambda t: _ELLIPSIS_RE.sub("...", t)),
    TextRule("C6", _rule("C6"), lambda t: _HTML_TAG_RE.sub("", t)),
    TextRule("C9", _rule("C9"), _apply_c9),
    TextRule("C8", _rule("C8"), lambda t: _MULTI_SPACE_RE.sub(" ", t)),
    TextRule("C7", _rule("C7"), _apply_c7),
    TextRule("C5", _rule("C5"), lambda t: _DOT_RUN_RE.sub("...", t)),
    TextRule("C8", _rule("C8"), lambda t: _MULTI_SPACE_RE.sub(" ", t)),
    TextRule("E3", _rule("E3"), lambda t: t.strip()),
)


def run_text_pipeline(
    text: str,
    rules=TEXT_RULES,
    *,
    track: list[RuleHit] | None = None,
) -> str:
    """Apply a sequence of :class:`TextRule` to ``text``. ``track=None`` → fast path."""
    for rule in rules:
        new_text = rule.apply(text)
        if new_text != text:
            if track is not None:
                track.append(RuleHit(rule.id, rule.reason, text, new_text))
            text = new_text
    return text


# ── helpers used by engine Rules ──────────────────────────────────────


def _ms_to_ts(ms: int) -> str:
    ms = max(0, int(ms))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1_000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _ts_str(c: Cue) -> str:
    return f"{_ms_to_ts(c.start_ms)} --> {_ms_to_ts(c.end_ms)}"


# ── engine Rule[Cue] wrappers ─────────────────────────────────────────


class TextSweepRule(Rule[Cue]):
    """Apply :data:`TEXT_RULES` to ``cue.text`` in place."""

    id = "C*"
    reason = "text cleaning sweep"
    lookahead = 0

    def apply(self, items, origins, *, tracker):
        for cue, origin in zip(items, origins):
            hits: list[RuleHit] = []
            new_text = run_text_pipeline(cue.text, track=hits)
            if new_text != cue.text:
                cue.text = new_text
            for h in hits:
                tracker.fire(
                    h.rule_id,
                    h.reason,
                    before=h.before,
                    after=h.after,
                    origin=origin,
                )
        return items, origins


class DropEmptyRule(Rule[Cue]):
    """E4 — drop cues whose text is empty."""

    id = "E4"
    reason = _rule("E4")
    lookahead = 0

    def apply(self, items, origins, *, tracker):
        out_items: list[Cue] = []
        out_origins: list[int] = []
        for cue, origin in zip(items, origins):
            if cue.text:
                out_items.append(cue)
                out_origins.append(origin)
            else:
                tracker.fire(
                    self.id,
                    self.reason,
                    before="<empty>",
                    after="<dropped>",
                    origin=origin,
                )
        return out_items, out_origins


# ── timestamp helpers ─────────────────────────────────────────────────


def _merge_texts(parts: list[str]) -> str:
    return _MULTI_SPACE_RE.sub(" ", " ".join(p.strip() for p in parts if p.strip())).strip()


def _estimated_display_lines(text: str, line_chars: int) -> int:
    line_chars = max(1, line_chars)
    return max(1, (len(text) + line_chars - 1) // line_chars)


def _merge_would_exceed_limits(text: str, cue_count: int, zero_count: int, options: CleanOptions) -> bool:
    return (
        cue_count > options.max_merged_cues
        or zero_count > options.max_zero_run
        or len(text) > options.max_text_chars
        or _estimated_display_lines(text, options.display_line_chars) > options.max_display_lines
    )


class TimestampRule(Rule[Cue]):
    """Composite rule applying T3 → T1M → T1 → T2 + final-1ms guarantee.

    Timestamp rules are tightly coupled (T1M may drop cues, affecting T1's
    indexing; T1 mutates durations that T2 then reads), so we keep them
    as one :class:`Rule` operating on the whole list.

    ``lookahead = options.max_zero_run + 2`` covers:
      * T1M needs to see up to ``max_zero_run`` zero-duration fragments
        plus the target cue that follows.
      * T1 needs the next real cue.
      * T2 needs the next cue.
    """

    id = "T*"
    reason = "timestamp cleanup"

    def __init__(
        self,
        options: CleanOptions | None = None,
        *,
        issues: list[Issue] | None = None,
    ) -> None:
        self._options = options or CleanOptions()
        self._issues = issues
        self.lookahead = self._options.max_zero_run + 2

    def apply(self, items, origins, *, tracker):
        options = self._options
        issues = self._issues

        # T3 — drop negatives / impossibles.
        cues: list[Cue] = []
        new_origins: list[int] = []
        for c, o in zip(items, origins):
            if 0 <= c.start_ms <= c.end_ms and c.end_ms < 360_000_000:
                cues.append(c)
                new_origins.append(o)
            else:
                tracker.fire(
                    "T3",
                    _rule("T3"),
                    before=_ts_str(c),
                    after="<dropped>",
                    origin=o,
                )

        cues, new_origins = self._merge_zero_duration_clusters(cues, new_origins, options=options, tracker=tracker, issues=issues)

        # T1 — fix zero-duration runs.
        i = 0
        while i < len(cues):
            if cues[i].end_ms <= cues[i].start_ms:
                i = self._fix_zero_duration_run(cues, new_origins, i, tracker=tracker)
            else:
                i += 1

        # T2 — fix small overlaps.
        for i in range(len(cues) - 1):
            a, b = cues[i], cues[i + 1]
            if a.end_ms <= b.start_ms:
                continue
            overlap = a.end_ms - b.start_ms
            if overlap <= _MAX_OVERLAP_FIX_MS and b.start_ms > a.start_ms:
                before = _ts_str(a)
                a.end_ms = b.start_ms
                tracker.fire(
                    "T2",
                    _rule("T2"),
                    before=before,
                    after=_ts_str(a),
                    origin=new_origins[i],
                )
            else:
                a.note = (a.note + " overlap").strip()

        # Final guarantee: every cue has ≥1ms of duration.
        for c in cues:
            if c.end_ms <= c.start_ms:
                c.end_ms = c.start_ms + 1

        return cues, new_origins

    def _merge_zero_duration_clusters(
        self,
        cues: list[Cue],
        origins: list[int],
        *,
        options: CleanOptions,
        tracker: Tracker,
        issues: list[Issue] | None,
    ) -> tuple[list[Cue], list[int]]:
        if not cues:
            return cues, origins

        drop_positions: set[int] = set()
        i = 0
        while i < len(cues):
            c = cues[i]
            if i in drop_positions or c.end_ms > c.start_ms:
                i += 1
                continue

            start = c.start_ms
            zero_positions: list[int] = []
            j = i
            while j < len(cues) and cues[j].start_ms == start and cues[j].end_ms <= cues[j].start_ms:
                zero_positions.append(j)
                j += 1

            target_pos: int | None = None
            if j < len(cues) and cues[j].start_ms == start and cues[j].end_ms > cues[j].start_ms:
                target_pos = j
            else:
                k = i - 1
                while k >= 0 and cues[k].start_ms == start:
                    if k not in drop_positions and cues[k].end_ms > cues[k].start_ms:
                        target_pos = k
                        break
                    k -= 1

            if target_pos is None:
                i = j
                continue

            target = cues[target_pos]
            ordered = sorted([target_pos, *zero_positions])
            merged = _merge_texts([cues[p].text for p in ordered])
            if _merge_would_exceed_limits(merged, len(ordered), len(zero_positions), options):
                if issues is not None:
                    issues.append(
                        Issue(
                            code="T1_MERGE_LIMIT_EXCEEDED",
                            severity="error",
                            message="zero-duration same-time cluster would exceed subtitle display limits",
                            cue_indices=tuple(p + 1 for p in ordered),
                        )
                    )
                for p in zero_positions:
                    z = cues[p]
                    tracker.fire(
                        "T1M!",
                        _rule("T1M!"),
                        before=f"{_ts_str(z)} {z.text}".strip(),
                        after="<unrepairable>",
                        origin=origins[p],
                    )
                i = j
                continue

            before_target = target.text
            target.text = merged
            target.note = (target.note + " merged-zero-duration").strip()
            if before_target != target.text:
                tracker.fire(
                    "T1M",
                    _rule("T1M"),
                    before=before_target,
                    after=target.text,
                    origin=origins[target_pos],
                )
            for p in zero_positions:
                z = cues[p]
                drop_positions.add(p)
                tracker.fire(
                    "T1M",
                    _rule("T1M"),
                    before=f"{_ts_str(z)} {z.text}".strip(),
                    after=f"<merged into {target.text}>",
                    origin=origins[p],
                )
            i = j

        if not drop_positions:
            return cues, origins
        kept_cues = [c for idx, c in enumerate(cues) if idx not in drop_positions]
        kept_origins = [o for idx, o in enumerate(origins) if idx not in drop_positions]
        return kept_cues, kept_origins

    def _fix_zero_duration_run(
        self,
        cues: list[Cue],
        origins: list[int],
        start_idx: int,
        *,
        tracker: Tracker,
    ) -> int:
        i = start_idx
        run_start = cues[i].start_ms
        j = i
        while j < len(cues) and cues[j].start_ms == run_start and cues[j].end_ms <= cues[j].start_ms:
            j += 1
        run_len = j - i
        if run_len == 0:
            return i + 1

        next_real_start: int | None = None
        for k in range(j, len(cues)):
            if cues[k].start_ms > run_start:
                next_real_start = cues[k].start_ms
                break

        window = (next_real_start - run_start) if next_real_start is not None else run_len * _MIN_DURATION_MS
        if j < len(cues) and cues[j].start_ms == run_start:
            window = 0

        if window >= run_len:
            per = min(_MIN_DURATION_MS, window // run_len)
            per = max(1, per)
            for k in range(run_len):
                c = cues[i + k]
                before = _ts_str(c)
                c.start_ms = run_start + k * per
                c.end_ms = c.start_ms + per
                c.note = "interpolated"
                tracker.fire(
                    "T1",
                    _rule("T1"),
                    before=before,
                    after=_ts_str(c),
                    origin=origins[i + k],
                )
        else:
            for k in range(run_len):
                c = cues[i + k]
                before = _ts_str(c)
                c.start_ms = run_start + k
                c.end_ms = c.start_ms + 1
                c.note = "interpolated"
                tracker.fire(
                    "T1",
                    _rule("T1"),
                    before=before,
                    after=_ts_str(c),
                    origin=origins[i + k],
                )
            needed_start = cues[i + run_len - 1].end_ms
            for k in range(j, len(cues)):
                if cues[k].start_ms >= needed_start:
                    break
                before = _ts_str(cues[k])
                old_dur = cues[k].end_ms - cues[k].start_ms
                cues[k].start_ms = needed_start
                cues[k].end_ms = max(cues[k].end_ms, needed_start + max(1, old_dur))
                tracker.fire(
                    "T1",
                    _rule("T1"),
                    before=before,
                    after=_ts_str(cues[k]),
                    origin=origins[k],
                )
                needed_start = cues[k].end_ms
        return j


__all__ = [
    "TextRule",
    "TEXT_RULES",
    "run_text_pipeline",
    "TextSweepRule",
    "DropEmptyRule",
    "TimestampRule",
    "_RULE_REASONS",
    "_rule",
    "_ms_to_ts",
    "_ts_str",
    "_INVISIBLE_RE",
    "_HTML_TAG_RE",
    "_HTML_ENTITY_RE",
    "_entity_sub",
    "_WHITESPACE_MAP",
    "_SMART_QUOTE_MAP",
    "_ELLIPSIS_RE",
    "_MULTI_SPACE_RE",
    "_TIMESTAMP_RE",
]
