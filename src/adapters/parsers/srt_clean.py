"""SRT 清洗 v2 —— 基于真实 12k 语料统计构建的"标准 SRT"规范实现.

本模块独立于 ``srt.py``，不影响原有管道；仅用于：
  1. 定义「标准 SRT」应该长什么样（见下方 RULES）；
  2. 把任意真实 SRT 规范化到该形态（``clean`` + ``dump``）；
  3. 通过 load → dump → load → dump 的两次幂等校验 + 内容不变式，
     验证清洗过程不会篡改文本内容。

核心原则：**清洗不修改文本本身**。只整理空白、标点附着、时间戳和
条目结构。**任何可见字符（字母 / 数字 / CJK / 括号内的 SDH 文字 /
音符 / HTML 标签文字 / 省略号之外的标点）都必须保留**。

RULES（"标准 SRT"定义）
========================================

格式层（文件级）
----------------
F1. 编码 UTF-8，无 BOM。
F2. 行尾统一为 ``\\n``（LF）。
F3. 文件末尾以单个 ``\\n`` 结尾。
F4. 条目之间使用单个空行分隔。
F5. 文件中不允许残留零宽 / 控制字符。

条目层
------
E1. 每个 cue 由三部分组成：
    - 行 1：递增序号（从 1 起，连续，无跳号）；
    - 行 2：``HH:MM:SS,mmm --> HH:MM:SS,mmm``（2 位小时，逗号毫秒）；
    - 行 3..：text。
E2. **text 必须是单行** —— 原文多行时用单个 ASCII 空格 join。
E3. text 首尾无空白；内部所有空白序列压缩为单个 ASCII 空格。
E4. 空 text 的条目丢弃。
E5. 条目序号重排为 1..N（无跳号、无重复）。

字符层（text 内部）—— 非破坏性
-------------------------------
C1. 去除零宽 / 控制字符：
    ZWSP/ZWNJ/ZWJ/LRM/RLM/WJ/SHY/CGJ/ALM/MVS/FA/IT/IS/IP/
    INHIBIT-SS..NOMINAL-DS/BOM-mid/DEL。
    （这些字符不可见、无语义，移除不改变内容）
C2. 各类 NBSP 规整为 ASCII 空格：
    NBSP / NNBSP / IDEOGRAPHIC_SPACE / MMSP /
    EN / EM / THIN / HAIR / FIGURE / PUNCTUATION / ZERO-WIDTH-NBSP。
C3. smart quotes → ASCII：``" " ' '`` → ``" ' " '``。
C4. 单字符省略号 ``…`` → ``...``。
C5. 连续 ``..`` 或 ≥4 个 ``.`` → 规整为 ``...``。
C6. HTML 标签剥壳（``<i>x</i>`` → ``x``），保留标签内文字。
    SDH 方括号 ``[Music]`` / ``(laughter)`` / ``♪``——**保留**。
C7. 标点附着：
    - 标点前的空白移除（``word ,`` → ``word,``、``word !`` → ``word!``）；
    - 标点后若紧跟字母 / 数字，补一个空格（``hello,world`` → ``hello, world``），
      但小数点 ``1.5``、连续标点 ``...``、数字千分位 ``1,000`` 不处理。
C8. 多个连续 ASCII 空格压缩为 1 个。
C9. 残余的不可打印控制字符（除 ``\\t`` → 空格）一律移除。

时间戳层
--------
T1. 时间戳必须满足 ``start < end``；若 ``start == end``（零时长），
    向相邻条目借 5~50ms 做插值修正。
T2. 相邻条目要求 ``end_i <= start_{i+1}``；若 ``end_i > start_{i+1}``：
    - 若重叠 ≤ 100ms：将前一条 ``end_i`` 下移到 ``start_{i+1}``；
    - 若重叠 > 100ms：保留原时间，**仅标记**（不合并，以免破坏信息）。
T3. 负时间戳、格式错误的时间戳直接丢弃所在条目。
T4. 时间戳最小粒度 1ms；超过 99:59:59,999 视为异常。

编号层
------
N1. 忽略原有序号，按最终条目顺序重排为 1..N。
N2. 跳过任何无时间戳的块。

清洗不变式
==========

对任意输入 ``raw``：

    text_content(raw) ⊆ text_content(clean(raw))  -- 内容不丢失

其中 ``text_content`` 定义为：所有 cue 的 text 拼接后，去除全部
空白 + 标点 + HTML 标签后的剩余字符序列（保留 CJK、字母、数字、
SDH 括号内的文字字符、音符字符）。

同时 ``clean`` 必须幂等：

    dump(parse(dump(parse(dump(clean(raw)))))) == dump(clean(raw))
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Character tables
# ---------------------------------------------------------------------------

# C1: zero-width / invisible / bidi control chars (NON-CONTENT) → remove
_INVISIBLE_RE = re.compile(
    "["
    "\u00ad"  # SOFT HYPHEN
    "\u034f"  # COMBINING GRAPHEME JOINER
    "\u061c"  # ARABIC LETTER MARK
    "\u115f\u1160"  # HANGUL CHOSEONG/JUNGSEONG FILLER
    "\u17b4\u17b5"  # KHMER VOWEL INHERENT AQ/AA
    "\u180e"  # MONGOLIAN VOWEL SEPARATOR
    "\u200b\u200c\u200d\u200e\u200f"  # ZWSP/ZWNJ/ZWJ/LRM/RLM
    "\u2028\u2029"  # LINE/PARA SEP (kept-out; will be turned into \n first)
    "\u202a-\u202e"  # bidi embedding/override
    "\u2060-\u2064"  # WJ/FA/IT/IS/IP
    "\u2066-\u206f"  # bidi isolates + inhibit-ss..nominal-ds
    "\u3164"  # HANGUL FILLER
    "\ufe00-\ufe0f"  # variation selectors
    "\ufeff"  # BOM / zero-width no-break space
    "\uffa0"  # HALFWIDTH HANGUL FILLER
    "\U000e0100-\U000e01ef"  # variation selectors supplement
    "\u007f"  # DEL
    "]"
)

# C2: NBSP-like whitespace → ASCII space
_WHITESPACE_MAP = str.maketrans(
    {
        "\u00a0": " ",  # NBSP
        "\u1680": " ",  # OGHAM SPACE MARK
        "\u2000": " ",  # EN QUAD
        "\u2001": " ",  # EM QUAD
        "\u2002": " ",  # EN SPACE
        "\u2003": " ",  # EM SPACE
        "\u2004": " ",  # THREE-PER-EM
        "\u2005": " ",  # FOUR-PER-EM
        "\u2006": " ",  # SIX-PER-EM
        "\u2007": " ",  # FIGURE SPACE
        "\u2008": " ",  # PUNCTUATION SPACE
        "\u2009": " ",  # THIN SPACE
        "\u200a": " ",  # HAIR SPACE
        "\u202f": " ",  # NARROW NBSP
        "\u205f": " ",  # MEDIUM MATHEMATICAL SPACE
        "\u3000": " ",  # IDEOGRAPHIC SPACE
    }
)

# C3: smart quotes → ASCII
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
    # Only strip real HTML-ish tags commonly found in SRT: ``<i> <b> <u> <s>
    # <br> <font …> <p> <span> <div>`` (and their closing forms).
    # Do NOT strip math-like ``<L u m u n>`` or ``<g u m>``.
    r"</?(?:i|b|u|s|br|em|strong|font|p|span|div)(?:\s[^>]*)?/?>",
    re.IGNORECASE,
)
_MULTI_SPACE_RE = re.compile(r" {2,}")
_ELLIPSIS_RE = re.compile(r"\u2026")  # …
_DOT_RUN_RE = re.compile(r"\.{2,}")  # .. or ....+
_TIMESTAMP_RE = re.compile(
    r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->\s*"
    r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})"
)

# C7: punctuation-attachment rules.
# ASCII sentence punctuation that should attach to the preceding word.
_ATTACH_PUNCTS = ",.!?:;"
# Full-width / CJK sentence punctuation treated similarly.
_CJK_PUNCTS = "，。！？：；、"

# ``word <sp>+ ,`` → ``word,``   (ASCII + full-width)
_SPACE_BEFORE_PUNCT_RE = re.compile(rf" +([{re.escape(_ATTACH_PUNCTS + _CJK_PUNCTS)}])")

# ``word,letter`` → ``word, letter``  (ASCII only; avoid 1,000 / 1.5 / ... / Mr.X style)
_COMMA_LIKE_RE = re.compile(r"(\w)([,:;!?])(?=[A-Za-z])")
_PERIOD_RE = re.compile(r"([A-Za-z]{2,})\.(?=[A-Z][a-z])")


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class Cue:
    """One SRT cue. Immutable from the outside after parse; we mutate during clean."""

    start_ms: int
    end_ms: int
    text: str
    note: str = ""  # non-fatal diagnostics: "overlap", "interpolated", ...


@dataclass
class RuleHit:
    """A single rule firing on a cue."""

    rule_id: str  # e.g. "E2", "C5", "C6", "T1", ...
    reason: str  # Chinese one-liner explaining why the rule fired
    before: str  # serialized snapshot before the rule (text, or "HH:MM,mmm --> ...")
    after: str  # serialized snapshot after the rule


@dataclass
class CueReport:
    """Report for a single cue: input, output, and every rule that touched it."""

    index_in: int  # 1-based index in the raw file (None if block_without_timestamp)
    index_out: int | None  # 1-based index in the cleaned file; None if dropped
    start_ms_in: int
    end_ms_in: int
    start_ms_out: int
    end_ms_out: int
    text_in: str  # raw text AS IT APPEARED in the source file (may contain \n)
    text_out: str  # text after full cleaning
    steps: list[RuleHit]

    @property
    def modified(self) -> bool:
        return bool(self.steps)


@dataclass
class Report:
    """Full cleaning report for one SRT content."""

    cues: list[CueReport]
    cues_in: int
    cues_out: int
    rule_counts: dict[str, int]


# Rule catalog: rule_id → Chinese reason. Used for both text + timestamp + entry
# rules. Keeping them in one place keeps format/jsonl output consistent.
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
    "T1": "零时长 cue，从邻近空档借用时长",
    "T2": "轻微重叠，下调前一条 end 到后一条 start",
    "T3": "时间戳非法（负值或越界），丢弃此条目",
    "T4": "时间戳截断到最大允许值",
    "N1": "按顺序重编号 1..N",
}


def _rule(rule_id: str) -> str:
    return _RULE_REASONS.get(rule_id, "")


# ---------------------------------------------------------------------------
# Parse
# ---------------------------------------------------------------------------


def _ts_to_ms(h: str, m: str, s: str, ms: str) -> int:
    return int(h) * 3_600_000 + int(m) * 60_000 + int(s) * 1_000 + int(ms.ljust(3, "0"))


def _ms_to_ts(ms: int) -> str:
    ms = max(0, int(ms))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1_000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def parse(content: str) -> list[Cue]:
    """Parse an SRT string into cues. Tolerant of malformed input (skips bad blocks)."""
    # Normalize line-endings up front so block-split is stable.
    content = content.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")
    cues: list[Cue] = []
    for block in re.split(r"\n\s*\n", content):
        lines = [ln for ln in block.split("\n") if ln.strip() != ""]
        if len(lines) < 2:
            continue
        # Find timestamp line (may be line 0 or line 1 depending on index presence)
        ts_line_idx = None
        ts_match = None
        for i in (1, 0):
            if i < len(lines):
                m = _TIMESTAMP_RE.search(lines[i])
                if m:
                    ts_line_idx = i
                    ts_match = m
                    break
        if ts_match is None:
            continue
        try:
            start = _ts_to_ms(*ts_match.group(1, 2, 3, 4))
            end = _ts_to_ms(*ts_match.group(5, 6, 7, 8))
        except ValueError:
            continue
        text_lines = lines[ts_line_idx + 1 :]
        if not text_lines:
            continue
        # E2: join multi-line text with a single space.
        text = " ".join(text_lines)
        cues.append(Cue(start_ms=start, end_ms=end, text=text))
    return cues


# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------


def _clean_text(text: str) -> str:
    # F5/C1 — remove invisible / control chars
    text = _INVISIBLE_RE.sub("", text)
    # C2 — NBSP-family → ASCII space
    text = text.translate(_WHITESPACE_MAP)
    # C3 — smart quotes
    text = text.translate(_SMART_QUOTE_MAP)
    # C4 — ellipsis
    text = _ELLIPSIS_RE.sub("...", text)
    # C6 — strip HTML tags but keep inner text
    text = _HTML_TAG_RE.sub("", text)
    # C9 — replace tab with space; drop remaining non-printable ASCII controls
    text = text.replace("\t", " ")
    text = "".join(ch for ch in text if ch == " " or unicodedata.category(ch)[0] != "C")
    # E3 + C8 — collapse runs of spaces
    text = _MULTI_SPACE_RE.sub(" ", text)
    # C7 — punctuation attachment (removes spaces before punct — may create
    # new dot runs like ``. .`` → ``..``, so run dot-run rule AFTER this).
    text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    text = _COMMA_LIKE_RE.sub(r"\1\2 ", text)
    # C5 — dot runs (``..`` or 4+ dots) → ``...``. Run AFTER C7 so any
    # ``. .`` → ``..`` gaps collapsed in C7 also get normalised.
    text = _DOT_RUN_RE.sub("...", text)
    # Second pass for any spaces just introduced
    text = _MULTI_SPACE_RE.sub(" ", text)
    return text.strip()


# T1/T2 parameters
_MIN_DURATION_MS = 50
_MAX_OVERLAP_FIX_MS = 100


def _fix_timestamps(cues: list[Cue]) -> list[Cue]:
    # T3 — drop negatives / impossibles.
    cues = [c for c in cues if 0 <= c.start_ms <= c.end_ms and c.end_ms < 360_000_000]

    # T1 — fix zero-duration: borrow from neighbors (up to 50ms).
    for i, c in enumerate(cues):
        if c.end_ms > c.start_ms:
            continue
        nxt = cues[i + 1] if i + 1 < len(cues) else None
        gap_next = (nxt.start_ms - c.end_ms) if nxt else 500
        if gap_next > 5:
            c.end_ms = c.start_ms + min(_MIN_DURATION_MS, gap_next - 1)
        else:
            # No room after — try to borrow from the previous cue if it has
            # extra room, otherwise accept a 1ms overlap with next.
            prev = cues[i - 1] if i > 0 else None
            gap_prev = (c.start_ms - prev.end_ms) if prev else 500
            if gap_prev > 5:
                c.start_ms = max(0, c.start_ms - min(_MIN_DURATION_MS, gap_prev - 1))
            c.end_ms = max(c.end_ms, c.start_ms + 1)
        c.note = "interpolated"

    # T2 — fix small overlaps. Never collapse a cue to zero duration.
    for i in range(len(cues) - 1):
        a, b = cues[i], cues[i + 1]
        if a.end_ms <= b.start_ms:
            continue
        overlap = a.end_ms - b.start_ms
        if overlap <= _MAX_OVERLAP_FIX_MS and b.start_ms > a.start_ms:
            a.end_ms = b.start_ms
        else:
            a.note = (a.note + " overlap").strip()

    # Final guarantee: every cue has at least 1ms of duration.
    for c in cues:
        if c.end_ms <= c.start_ms:
            c.end_ms = c.start_ms + 1

    return cues


def _clean_text_tracked(text: str, steps: list[RuleHit] | None = None) -> str:
    """Same as ``_clean_text`` but optionally records a RuleHit per fired rule.

    Passing ``steps=None`` runs the fast path (no tracking, no allocation).
    """
    if steps is None:
        return _clean_text(text)

    def _apply(rule_id: str, new_text: str, cur: str) -> str:
        if new_text != cur:
            steps.append(RuleHit(rule_id, _rule(rule_id), cur, new_text))
        return new_text

    cur = text
    cur = _apply("C1", _INVISIBLE_RE.sub("", cur), cur)
    cur = _apply("C2", cur.translate(_WHITESPACE_MAP), cur)
    cur = _apply("C3", cur.translate(_SMART_QUOTE_MAP), cur)
    cur = _apply("C4", _ELLIPSIS_RE.sub("...", cur), cur)
    cur = _apply("C6", _HTML_TAG_RE.sub("", cur), cur)
    # C9: tab → space, then drop remaining non-printable controls
    c9_mid = cur.replace("\t", " ")
    c9_mid = "".join(ch for ch in c9_mid if ch == " " or unicodedata.category(ch)[0] != "C")
    cur = _apply("C9", c9_mid, cur)
    cur = _apply("C8", _MULTI_SPACE_RE.sub(" ", cur), cur)
    # C7 combined (space-before-punct + comma-like spacing)
    c7_mid = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", cur)
    c7_mid = _COMMA_LIKE_RE.sub(r"\1\2 ", c7_mid)
    cur = _apply("C7", c7_mid, cur)
    cur = _apply("C5", _DOT_RUN_RE.sub("...", cur), cur)
    cur = _apply("C8", _MULTI_SPACE_RE.sub(" ", cur), cur)
    cur = _apply("E3", cur.strip(), cur)
    return cur


def _fix_timestamps_tracked(
    cues: list[Cue],
    cue_steps: dict[int, list[RuleHit]] | None,
) -> list[Cue]:
    """Same logic as ``_fix_timestamps`` but also records T1/T2/T3 hits.

    Uses id() of each Cue as the key into ``cue_steps`` so that reports can
    be matched back to CueReport records after renumbering.
    """
    track = cue_steps is not None

    def _ts(c: Cue) -> str:
        return f"{_ms_to_ts(c.start_ms)} --> {_ms_to_ts(c.end_ms)}"

    # T3 — drop negatives / impossibles.
    out: list[Cue] = []
    for c in cues:
        if 0 <= c.start_ms <= c.end_ms and c.end_ms < 360_000_000:
            out.append(c)
        elif track:
            cue_steps.setdefault(id(c), []).append(RuleHit("T3", _rule("T3"), _ts(c), "<dropped>"))
    cues = out

    # T1 — fix zero-duration.
    for i, c in enumerate(cues):
        if c.end_ms > c.start_ms:
            continue
        before = _ts(c)
        nxt = cues[i + 1] if i + 1 < len(cues) else None
        gap_next = (nxt.start_ms - c.end_ms) if nxt else 500
        if gap_next > 5:
            c.end_ms = c.start_ms + min(_MIN_DURATION_MS, gap_next - 1)
        else:
            prev = cues[i - 1] if i > 0 else None
            gap_prev = (c.start_ms - prev.end_ms) if prev else 500
            if gap_prev > 5:
                c.start_ms = max(0, c.start_ms - min(_MIN_DURATION_MS, gap_prev - 1))
            c.end_ms = max(c.end_ms, c.start_ms + 1)
        c.note = "interpolated"
        if track:
            cue_steps.setdefault(id(c), []).append(RuleHit("T1", _rule("T1"), before, _ts(c)))

    # T2 — fix small overlaps.
    for i in range(len(cues) - 1):
        a, b = cues[i], cues[i + 1]
        if a.end_ms <= b.start_ms:
            continue
        overlap = a.end_ms - b.start_ms
        if overlap <= _MAX_OVERLAP_FIX_MS and b.start_ms > a.start_ms:
            before = _ts(a)
            a.end_ms = b.start_ms
            if track:
                cue_steps.setdefault(id(a), []).append(RuleHit("T2", _rule("T2"), before, _ts(a)))
        else:
            a.note = (a.note + " overlap").strip()

    # Final guarantee: every cue has at least 1ms of duration.
    for c in cues:
        if c.end_ms <= c.start_ms:
            c.end_ms = c.start_ms + 1

    return cues


def _parse_with_raw(content: str) -> tuple[list[Cue], list[str]]:
    """Like ``parse`` but also returns the raw pre-join text of each cue.

    raw_texts[i] is the multi-line text exactly as it appeared in the source
    (newlines intact), suitable for the E2 report step.
    """
    content = content.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")
    cues: list[Cue] = []
    raws: list[str] = []
    for block in re.split(r"\n\s*\n", content):
        lines = [ln for ln in block.split("\n") if ln.strip() != ""]
        if len(lines) < 2:
            continue
        ts_line_idx = None
        ts_match = None
        for i in (1, 0):
            if i < len(lines):
                m = _TIMESTAMP_RE.search(lines[i])
                if m:
                    ts_line_idx = i
                    ts_match = m
                    break
        if ts_match is None:
            continue
        try:
            start = _ts_to_ms(*ts_match.group(1, 2, 3, 4))
            end = _ts_to_ms(*ts_match.group(5, 6, 7, 8))
        except ValueError:
            continue
        text_lines = lines[ts_line_idx + 1 :]
        if not text_lines:
            continue
        raws.append("\n".join(text_lines))
        cues.append(Cue(start_ms=start, end_ms=end, text=" ".join(text_lines)))
    return cues, raws


def clean_with_report(content: str) -> tuple[list[Cue], Report]:
    """Run ``clean`` while also building a per-cue Report.

    Always collects the full step trace. The caller picks a verbosity level
    at render time via ``format_report(level=...)``.
    """
    cues, raws = _parse_with_raw(content)
    reports: list[CueReport] = []
    cue_to_report: dict[int, CueReport] = {}

    for i, (c, raw_text) in enumerate(zip(cues, raws), start=1):
        steps: list[RuleHit] = []
        # Synthesise E2 step if multi-line join actually changed the text.
        if raw_text != c.text:
            steps.append(RuleHit("E2", _rule("E2"), raw_text, c.text))
        start_in, end_in, text_in = c.start_ms, c.end_ms, raw_text
        # Text cleaning
        c.text = _clean_text_tracked(c.text, steps)
        rep = CueReport(
            index_in=i,
            index_out=None,  # filled in after filtering + renumber
            start_ms_in=start_in,
            end_ms_in=end_in,
            start_ms_out=c.start_ms,
            end_ms_out=c.end_ms,
            text_in=text_in,
            text_out=c.text,
            steps=steps,
        )
        reports.append(rep)
        cue_to_report[id(c)] = rep

    # E4 — drop empties. Record before drop.
    kept: list[Cue] = []
    for c in cues:
        rep = cue_to_report[id(c)]
        if c.text:
            kept.append(c)
        else:
            rep.steps.append(RuleHit("E4", _rule("E4"), rep.text_out or "<empty>", "<dropped>"))
    cues = kept

    # Timestamp fixes (record into the same reports via id()).
    ts_steps: dict[int, list[RuleHit]] = {}
    cues = _fix_timestamps_tracked(cues, ts_steps)
    for cid, hits in ts_steps.items():
        if cid in cue_to_report:
            cue_to_report[cid].steps.extend(hits)

    # Final drop of any residual zero-duration cues.
    kept2: list[Cue] = []
    for c in cues:
        if c.text and c.end_ms > c.start_ms:
            kept2.append(c)
        else:
            rep = cue_to_report.get(id(c))
            if rep is not None:
                rep.steps.append(RuleHit("E4", _rule("E4"), rep.text_out or "<empty>", "<dropped>"))
    cues = kept2

    # Finalize: N1 renumber + fill index_out/start_ms_out/end_ms_out
    for new_idx, c in enumerate(cues, start=1):
        rep = cue_to_report[id(c)]
        rep.index_out = new_idx
        rep.start_ms_out = c.start_ms
        rep.end_ms_out = c.end_ms
        rep.text_out = c.text

    # rule_counts
    rule_counts: dict[str, int] = {}
    for rep in reports:
        for h in rep.steps:
            rule_counts[h.rule_id] = rule_counts.get(h.rule_id, 0) + 1

    report = Report(
        cues=reports,
        cues_in=len(reports),
        cues_out=len(cues),
        rule_counts=rule_counts,
    )
    return cues, report


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


def _format_summary(report: Report, path: str | None = None) -> str:
    lines = ["─── FILE SUMMARY " + "─" * 56]
    if path:
        lines.append(f"path:            {path}")
    dropped = report.cues_in - report.cues_out
    lines.append(
        f"cues in / out:   {report.cues_in} / {report.cues_out}    (-{dropped} dropped)"
        if dropped
        else f"cues in / out:   {report.cues_in} / {report.cues_out}"
    )
    n_mod = sum(1 for r in report.cues if r.modified)
    pct = n_mod * 100.0 / max(1, report.cues_in)
    lines.append(f"cues modified:   {n_mod}   ({pct:.1f}%)")
    if report.rule_counts:
        items = sorted(report.rule_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        lines.append("rules triggered: " + ", ".join(f"{k}×{v}" for k, v in items))
    return "\n".join(lines)


def format_report(
    report: Report,
    *,
    path: str | None = None,
    level: str = "full",
    only_modified: bool = True,
) -> str:
    """Format a report as human-readable text.

    ``level`` in {"minimal", "result", "full"}.
    """
    if level not in ("minimal", "result", "full"):
        raise ValueError(f"unknown level: {level!r}")

    def _esc(s: str) -> str:
        return s.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")

    parts: list[str] = []
    for rep in report.cues:
        if only_modified and not rep.modified:
            continue
        ts_out = f"{_ms_to_ts(rep.start_ms_out)} --> {_ms_to_ts(rep.end_ms_out)}" if rep.index_out is not None else "<dropped>"
        idx_label = f"#{rep.index_in}" + (f"→{rep.index_out}" if rep.index_out else " <dropped>")
        header = f"{idx_label}  {ts_out}"
        block = [header]
        if level == "minimal":
            block.append(f"  - {_esc(rep.text_in)}")
            block.append(f"  + {_esc(rep.text_out)}")
        elif level == "result":
            block.append(f"  in:   {_esc(rep.text_in)}")
            for h in rep.steps:
                block.append(f"  after {h.rule_id}: {_esc(h.after)}")
            block.append(f"  out:  {_esc(rep.text_out)}")
        else:  # full
            block.append(f"  in:   {_esc(rep.text_in)}")
            for h in rep.steps:
                block.append(f"  step {h.rule_id}  [{h.reason}]")
                block.append(f"           → {_esc(h.after)}")
            block.append(f"  out:  {_esc(rep.text_out)}")
        parts.append("\n".join(block))

    if parts:
        return "\n\n".join(parts) + "\n\n" + _format_summary(report, path) + "\n"
    return _format_summary(report, path) + "\n"


def report_to_jsonl(report: Report, *, path: str | None = None) -> list[str]:
    """Serialize a report to a list of JSONL lines (one per modified cue + summary).

    Each cue line has shape::

        {"type": "cue", "path": ..., "index_in": 37, "index_out": 37,
         "start_in": 195207, "end_in": 198320,
         "start_out": 195207, "end_out": 198320,
         "text_in": "...", "text_out": "...",
         "steps": [{"rule": "C6", "reason": "剥离 HTML ...", "before": "...", "after": "..."}, ...]}

    Final line is the summary::

        {"type": "summary", "path": ..., "cues_in": 424, "cues_out": 423,
         "cues_modified": 41, "rule_counts": {"C6": 17, "C7": 23, ...}}
    """
    import json

    lines: list[str] = []
    for rep in report.cues:
        if not rep.modified:
            continue
        lines.append(
            json.dumps(
                {
                    "type": "cue",
                    "path": path,
                    "index_in": rep.index_in,
                    "index_out": rep.index_out,
                    "start_in": rep.start_ms_in,
                    "end_in": rep.end_ms_in,
                    "start_out": rep.start_ms_out,
                    "end_out": rep.end_ms_out,
                    "text_in": rep.text_in,
                    "text_out": rep.text_out,
                    "steps": [
                        {
                            "rule": h.rule_id,
                            "reason": h.reason,
                            "before": h.before,
                            "after": h.after,
                        }
                        for h in rep.steps
                    ],
                },
                ensure_ascii=False,
            )
        )
    n_mod = sum(1 for r in report.cues if r.modified)
    lines.append(
        json.dumps(
            {
                "type": "summary",
                "path": path,
                "cues_in": report.cues_in,
                "cues_out": report.cues_out,
                "cues_modified": n_mod,
                "rule_counts": report.rule_counts,
            },
            ensure_ascii=False,
        )
    )
    return lines


def clean(content: str) -> list[Cue]:
    """Parse → normalize text per cue → fix timestamps → drop empties → renumber."""
    cues = parse(content)
    for c in cues:
        c.text = _clean_text(c.text)
    cues = [c for c in cues if c.text]  # E4
    cues = _fix_timestamps(cues)
    cues = [c for c in cues if c.text and c.end_ms > c.start_ms]
    return cues


# ---------------------------------------------------------------------------
# Dump
# ---------------------------------------------------------------------------


def dump(cues: list[Cue]) -> str:
    """Serialize cues to a standard-shaped SRT string (F1-F5, E1-E5, N1)."""
    parts: list[str] = []
    for idx, c in enumerate(cues, start=1):
        parts.append(str(idx))
        parts.append(f"{_ms_to_ts(c.start_ms)} --> {_ms_to_ts(c.end_ms)}")
        parts.append(c.text)
        parts.append("")
    return "\n".join(parts).rstrip("\n") + "\n"


# ---------------------------------------------------------------------------
# Invariant helpers (used by the verification harness)
# ---------------------------------------------------------------------------

_STRIP_PUNCT_RE = re.compile(
    # Keep letters (Latin + CJK + Hiragana/Katakana/Hangul + marks), digits,
    # and "content" symbols (♪, brackets keep inner? → brackets are punct).
    # We drop: spaces, ASCII punctuation, CJK punctuation, Unicode punctuation,
    # control chars, line terminators.
    r"[\s\u2000-\u206f\u3000-\u303f\uff00-\uffef" + re.escape("""!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~""") + r"]+"
)


def text_content(cues_or_text: list[Cue] | str) -> str:
    """Extract content invariant: all text joined, spaces + punctuation removed.

    Used to assert ``clean`` does not drop real content characters (letters,
    digits, CJK, musical symbols, etc).
    """
    if isinstance(cues_or_text, str):
        joined = cues_or_text
    else:
        joined = "".join(c.text for c in cues_or_text)
    # HTML tags are formatting, not content — strip before comparing.
    joined = _HTML_TAG_RE.sub("", joined)
    # Zero-width / bidi chars are non-content; ignore them in the invariant too.
    joined = _INVISIBLE_RE.sub("", joined)
    # Normalize NFKC so e.g. ``１`` == ``1`` across cleaning rounds.
    joined = unicodedata.normalize("NFKC", joined)
    # Drop the punct/space set.
    joined = _STRIP_PUNCT_RE.sub("", joined)
    # Also drop remaining connector/dash punctuation categories.
    joined = "".join(ch for ch in joined if unicodedata.category(ch)[0] != "P")
    return joined
