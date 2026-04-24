"""Regex and character-table constants for SRT cleaning."""

from __future__ import annotations

import re

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
    r"</?(?:i|b|u|s|br|em|strong|font|p|span|div)(?:\s[^>]*)?/?>",
    re.IGNORECASE,
)

# C10: HTML entity decode — conservative, explicit whitelist.
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

# C7: punctuation-attachment rules.
_ATTACH_PUNCTS = ",.!?:;"
_CJK_PUNCTS = "，。！？：；、"

_SPACE_BEFORE_PUNCT_RE = re.compile(rf" +([{re.escape(_ATTACH_PUNCTS + _CJK_PUNCTS)}])")
_COMMA_LIKE_RE = re.compile(r"(\w)([,:;!?])(?=[A-Za-z])")
_PERIOD_RE = re.compile(r"([A-Za-z]{2,})\.(?=[A-Z][a-z])")

# T1/T2 parameters
_MIN_DURATION_MS = 50
_MAX_OVERLAP_FIX_MS = 100
