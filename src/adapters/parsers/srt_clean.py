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
