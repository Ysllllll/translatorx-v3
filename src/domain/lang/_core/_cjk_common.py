"""Shared CJK mechanism base class and helpers."""

from __future__ import annotations

import re
from typing import Iterator

from ._chars import (
    is_east_asian,
    is_cjk_ideograph,
    is_hangul,
    is_hiragana,
    is_katakana,
    is_opening_punct_char,
    is_attach_to_prev_char,
    CONTENT_LIKE_CHARS,
)
from ._base_ops import _BaseOps, normalize_mode, _VALID_MODES


# 智能引号（Smart Quotes）：作为 CJK 侧字符处理，让它们紧贴中文文本（符合 CJK 排版习惯）。
# \u201c=“  \u201d=”  \u2018=‘  \u2019=’
_SMART_QUOTE_CHARS = frozenset("\u201c\u201d\u2018\u2019")


# Multi-character Latin fragments that must tokenize as a single unit:
# URLs, dotted identifiers (``deeplearning.ai``), contractions
# (``I'm``, ``rock'n'roll``). These are opaque to the script
# segmenter — they're replaced with an ASCII alnum placeholder before
# segmentation and restored afterwards.
_PROTECTED_LATIN_FRAGMENT_RE = re.compile(
    r"""
    https?://[A-Za-z0-9._~:/?\#\[\]@!$&'()*+,;=%-]+
    | [A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+
    | [A-Za-z]+(?:'[A-Za-z]+)+
    """,
    re.VERBOSE,
)


def _encode_placeholder_index(index: int) -> str:
    chars: list[str] = []
    current = index
    while True:
        current, remainder = divmod(current, 26)
        chars.append(chr(ord("A") + remainder))
        if current == 0:
            break
        current -= 1
    return "".join(reversed(chars))


def _make_protected_placeholder(index: int) -> str:
    return f"PROTECTEDTOKEN{_encode_placeholder_index(index)}"


def _protect_latin_fragments(text: str) -> tuple[str, dict[str, str]]:
    parts: list[str] = []
    mapping: dict[str, str] = {}
    last = 0
    for index, match in enumerate(_PROTECTED_LATIN_FRAGMENT_RE.finditer(text)):
        parts.append(text[last : match.start()])
        placeholder = _make_protected_placeholder(index)
        mapping[placeholder] = match.group(0)
        parts.append(placeholder)
        last = match.end()
    if not mapping:
        return text, {}
    parts.append(text[last:])
    return "".join(parts), mapping


def _restore_protected_tokens(tokens: list[str], mapping: dict[str, str]) -> list[str]:
    if not mapping:
        return tokens
    return [mapping.get(token, token) for token in tokens]


def _is_cjk_or_kana(ch: str) -> bool:
    return is_cjk_ideograph(ch) or is_hiragana(ch) or is_katakana(ch) or is_hangul(ch)


def _is_full_width_char(ch: str) -> bool:
    if is_east_asian(ch):
        return True
    cp = ord(ch)
    if 0x3000 <= cp <= 0x303F:
        return True
    if 0xFF01 <= cp <= 0xFF5E:
        return True
    if ch in CONTENT_LIKE_CHARS:
        return True
    return False


def _is_cjk_side(ch: str) -> bool:
    """Whether a char is "CJK-side" for script segmentation.

    Treats smart quotes (U+201C..201D, U+2018..2019) as CJK-side so
    they pair with adjacent CJK text rather than with ASCII/Latin
    characters. ASCII ``"`` and ``'`` are **not** CJK-side — they
    stay in Latin runs with adjacent Latin alnum characters.
    """
    return _is_full_width_char(ch) or ch in _SMART_QUOTE_CHARS


def _is_latin_run_char(ch: str) -> bool:
    """Whether ``ch`` belongs to a contiguous Latin "word" run.

    Latin run chars are ASCII alphanumerics plus ASCII quotes (``"``,
    ``'``). Quotes glue to adjacent alnum text so ``"AI"`` tokenizes
    as one unit. Other ASCII punctuation (``,.!?`` etc.) ends the run
    and becomes a standalone single-char token.
    """
    return ch.isalnum() and not _is_cjk_side(ch) or ch in ('"', "'")


def _iter_script_segments(text: str) -> Iterator[tuple[str, str]]:
    """Yield ``(kind, segment)`` pairs for each maximal CJK / Latin run.

    Whitespace acts as a segment separator and is dropped. Segments are:
    - ``("cjk", run)`` for maximal CJK-side chars (including CJK punct).
    - ``("latin", run)`` for maximal Latin-run chars (alnum + ASCII quotes).
    - ``("latin", ch)`` for each other single non-space char (ASCII punct,
      symbols), emitted as its own one-char "latin" segment. ``_attach_tokens``
      downstream merges these into neighboring words as appropriate.
    """
    buf: list[str] = []
    kind: str | None = None

    def flush() -> Iterator[tuple[str, str]]:
        nonlocal buf, kind
        if buf:
            yield kind, "".join(buf)  # type: ignore[misc]
            buf = []
            kind = None

    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch.isspace():
            yield from flush()
            i += 1
            continue
        if _is_cjk_side(ch):
            cur_kind = "cjk"
        elif _is_latin_run_char(ch):
            cur_kind = "latin"
        else:
            yield from flush()
            if ch == ".":
                j = i
                while j < n and text[j] == ".":
                    j += 1
                yield "latin", text[i:j]
                i = j
                continue
            yield "latin", ch
            i += 1
            continue
        if kind is None or cur_kind != kind:
            yield from flush()
            kind = cur_kind
        buf.append(ch)
        i += 1
    yield from flush()


def _cjk_length(text: str, cjk_width: int = 1) -> int:
    total = 0
    latin_count = 0

    def flush() -> None:
        nonlocal total, latin_count
        if latin_count > 0:
            total += (latin_count + cjk_width - 1) // cjk_width
            latin_count = 0

    for ch in text:
        if _is_full_width_char(ch):
            flush()
            total += 1
        elif ch.isspace():
            flush()
        else:
            latin_count += 1

    flush()
    return total


def _encode_placeholder_index(index: int) -> str:
    chars: list[str] = []
    current = index
    while True:
        current, remainder = divmod(current, 26)
        chars.append(chr(ord("A") + remainder))
        if current == 0:
            break
        current -= 1
    return "".join(reversed(chars))


def _is_opening_token(token: str) -> bool:
    return len(token) == 1 and is_opening_punct_char(token)


def _is_trailing_or_closing_token(token: str, multi_dot_attaches: bool = True) -> bool:
    if len(token) == 1 and is_attach_to_prev_char(token):
        return True
    if multi_dot_attaches and len(token) > 1 and all(c == "." for c in token):
        return True
    return False


def _attach_tokens(raw_tokens: list[str], multi_dot_attaches: bool = True) -> list[str]:
    result: list[str] = []
    current = ""
    pending_open = ""

    for token in raw_tokens:
        if _is_opening_token(token):
            if current:
                result.append(current)
                current = ""
            pending_open += token
        elif _is_trailing_or_closing_token(token, multi_dot_attaches):
            if current:
                current += token
            elif pending_open:
                pending_open += token
            else:
                if result:
                    result[-1] += token
                else:
                    current = token
        else:
            if current:
                result.append(current)
            current = pending_open + token
            pending_open = ""

    if current:
        result.append(current)
    if pending_open:
        result.append(pending_open)

    return result


def _cjk_join_tokens(tokens: list[str]) -> str:
    """Join tokens using script-aware spacing.

    Rules (boundary chars = ``prev[-1]`` and ``curr[0]``):
    - If either boundary char is non-alphanumeric (punctuation, including
      smart quotes, full-width punct, or ASCII ``.``/``"``/etc.), no
      space.
    - Else if both boundary chars are CJK-side, no space.
    - Else insert a single space.
    """
    if not tokens:
        return ""
    parts = [tokens[0]]
    for i in range(1, len(tokens)):
        prev = tokens[i - 1]
        curr = tokens[i]
        if not prev or not curr:
            parts.append(curr)
            continue
        prev_last = prev[-1]
        curr_first = curr[0]
        if not prev_last.isalnum() or not curr_first.isalnum():
            parts.append(curr)
            continue
        if _is_cjk_side(prev_last) and _is_cjk_side(curr_first):
            parts.append(curr)
            continue
        parts.append(" ")
        parts.append(curr)
    return "".join(parts)


class _BaseCjkOps(_BaseOps):
    """Base class for CJK text operations.

    Subclasses must implement ``_word_tokenize``.
    Override ``split`` and ``join`` if the language requires
    special handling (e.g. Korean eojeol tracking).
    """

    @property
    def sentence_terminators(self) -> frozenset[str]:
        return frozenset({"。", "！", "？", "!", "?"})

    @property
    def clause_separators(self) -> frozenset[str]:
        return frozenset({"，", "、", "；", "：", ",", ";", ":"})

    @property
    def abbreviations(self) -> frozenset[str]:
        return frozenset()

    @property
    def is_cjk(self) -> bool:
        return True

    def _word_tokenize(self, text: str) -> list[str]:
        raise NotImplementedError

    def split(self, text: str, mode: str = "word", attach_punctuation: bool = True) -> list[str]:
        mode = normalize_mode(mode)
        if mode not in _VALID_MODES:
            raise ValueError(f"Invalid mode: {mode!r}")

        protected, mapping = _protect_latin_fragments(text)

        raw: list[str] = []
        for kind, seg in _iter_script_segments(protected):
            if kind == "cjk":
                if mode == "character":
                    raw.extend(list(seg))
                else:
                    raw.extend(self._word_tokenize(seg))
            else:  # latin segment: stay as single token (ASCII quotes ride along)
                raw.append(seg)

        raw = _restore_protected_tokens(raw, mapping)

        if attach_punctuation:
            return _attach_tokens(raw, multi_dot_attaches=(mode == "character"))
        return raw

    def join(self, tokens: list[str]) -> str:
        return _cjk_join_tokens(tokens)

    def length(self, text: str, **kwargs: int) -> int:
        cjk_width = kwargs.get("cjk_width", 1)
        return _cjk_length(text, cjk_width)

    def normalize(self, text: str) -> str:
        """Beautify CJK↔Latin transitions by inserting spaces.

        Inserts a space between a CJK-side char and an adjacent Latin-side
        char when the Latin side belongs to a run that contains at least
        one alphanumeric character. Pure-punct Latin runs (e.g. trailing
        ``!``) are left adjacent to CJK text.
        """
        n = len(text)
        if n == 0:
            return text

        # Precompute for each position i whether text[i] is inside a
        # Latin run (contiguous non-whitespace non-CJK-side chars) that
        # contains at least one alphanumeric character.
        run_has_alnum = [False] * n
        i = 0
        while i < n:
            ch = text[i]
            if ch.isspace() or _is_cjk_side(ch):
                i += 1
                continue
            j = i
            while j < n and not text[j].isspace() and not _is_cjk_side(text[j]):
                j += 1
            if any(text[k].isalnum() for k in range(i, j)):
                for k in range(i, j):
                    run_has_alnum[k] = True
            i = j

        out: list[str] = []
        for i, ch in enumerate(text):
            if i > 0:
                prev = text[i - 1]
                if not prev.isspace() and not ch.isspace():
                    p_cjk = _is_cjk_side(prev)
                    c_cjk = _is_cjk_side(ch)
                    if p_cjk != c_cjk:
                        # Boundary. Only insert space if the Latin side
                        # run contains alnum.
                        if p_cjk and run_has_alnum[i]:
                            out.append(" ")
                        elif not p_cjk and run_has_alnum[i - 1]:
                            out.append(" ")
            out.append(ch)
        return "".join(out)
