"""SRT serialization + text-level facade.

Combines what used to be ``parse.py``, ``dump.py`` and ``facade.py``.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from domain.model import Segment

from .model import Cue
from .rules import (
    _ELLIPSIS_RE,
    _HTML_ENTITY_RE,
    _HTML_TAG_RE,
    _INVISIBLE_RE,
    _MULTI_SPACE_RE,
    _SMART_QUOTE_MAP,
    _TIMESTAMP_RE,
    _WHITESPACE_MAP,
    _entity_sub,
    _ms_to_ts,
)
from domain.lang import DEFAULT_FENCES, mask_fences, unmask_fences


def _ts_to_ms(h: str, m: str, s: str, ms: str) -> int:
    return int(h) * 3_600_000 + int(m) * 60_000 + int(s) * 1_000 + int(ms.ljust(3, "0"))


def parse(content: str, *, keep_raw: bool = False) -> list[Cue] | tuple[list[Cue], list[str]]:
    """Parse an SRT string into cues. Tolerant of malformed blocks.

    ``keep_raw=True`` additionally returns the pre-join multi-line text per
    cue for use in the E2 report step.
    """
    content = content.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")  # 去除字节顺序标记 BOM
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
        if keep_raw:
            raws.append("\n".join(text_lines))
        cues.append(Cue(start_ms=start, end_ms=end, text=" ".join(text_lines)))
    if keep_raw:
        return cues, raws
    return cues


def dump(cues: list[Cue]) -> str:
    """Serialize cues to a standard-shaped SRT string."""
    parts: list[str] = []
    for idx, c in enumerate(cues, start=1):
        parts.append(str(idx))
        parts.append(f"{_ms_to_ts(c.start_ms)} --> {_ms_to_ts(c.end_ms)}")
        parts.append(c.text)
        parts.append("")
    return "\n".join(parts).rstrip("\n") + "\n"


# 用于「内容不变量」比较：剥掉所有空白与标点。
# Unicode 区段：
#   U+2000..U+206F  通用标点区（半/全宽空格、引号、连字符、省略号等）
#   U+3000..U+303F  中日韩符号与标点（、。「」全角空格 等）
#   U+FF00..U+FFEF  半宽/全宽变体表（！？．， 等全角符号）
_STRIP_PUNCT_RE = re.compile(r"[\s\u2000-\u206f\u3000-\u303f\uff00-\uffef" + re.escape("""!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~""") + r"]+")


def text_content(cues_or_text: list[Cue] | str) -> str:
    """Extract content invariant: all text joined, spaces + punctuation removed."""
    if isinstance(cues_or_text, str):
        joined = cues_or_text
    else:
        joined = "".join(c.text for c in cues_or_text)
    joined = _HTML_ENTITY_RE.sub(_entity_sub, joined)
    joined = _HTML_TAG_RE.sub("", joined)
    joined = _INVISIBLE_RE.sub("", joined)
    joined = unicodedata.normalize("NFKC", joined)
    joined = _STRIP_PUNCT_RE.sub("", joined)
    joined = "".join(ch for ch in joined if unicodedata.category(ch)[0] != "P")
    return joined


def sanitize_srt(content: str) -> str:
    """Text-level SRT sanitizer. Normalizes text artifacts in-place; no timestamp repair.

    C8 — fence markers (``[? ... ?]`` / ``[! ... !]`` and friends from
    :data:`DEFAULT_FENCES`) are masked before transformations such as
    HTML-tag stripping or whitespace collapsing can damage their inner
    payload. The original text is restored after sanitisation.
    """
    content = content.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")  # 去除字节顺序标记 BOM
    content, fence_map = mask_fences(content, DEFAULT_FENCES)
    content = _HTML_ENTITY_RE.sub(_entity_sub, content)
    content = _INVISIBLE_RE.sub("", content)
    content = content.translate(_WHITESPACE_MAP)
    content = content.translate(_SMART_QUOTE_MAP)
    content = _ELLIPSIS_RE.sub("...", content)
    content = _HTML_TAG_RE.sub("", content)
    content = content.replace("\t", " ")
    content = "".join(ch for ch in content if ch in "\n " or unicodedata.category(ch)[0] != "C")
    content = _MULTI_SPACE_RE.sub(" ", content)
    content = re.sub(r"(?<!\.)\.\.(?!\.)", ".", content)
    if fence_map:
        content = unmask_fences(content, fence_map)
    return content


def parse_srt(content: str) -> list[Segment]:
    """Parse and clean SRT content into domain :class:`Segment` objects."""
    from .pipeline import clean_srt

    result = clean_srt(content)
    if not result.ok:
        codes = ", ".join(issue.code for issue in result.issues) or "unknown"
        raise ValueError(f"SRT is not safely repairable: {codes}")
    return [Segment(start=c.start_ms / 1000, end=c.end_ms / 1000, text=c.text) for c in result.cues]


def read_srt(path: str | Path) -> list[Segment]:
    """Read an SRT file and return cleaned domain :class:`Segment` objects."""
    return parse_srt(Path(path).read_text(encoding="utf-8"))


__all__ = [
    "parse",
    "dump",
    "text_content",
    "sanitize_srt",
    "parse_srt",
    "read_srt",
    "_ts_to_ms",
    "_STRIP_PUNCT_RE",
]
