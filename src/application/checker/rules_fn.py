"""Function-based rules and sanitizers (P2).

This module re-implements every check / sanitize step as a registered
factory function on top of :mod:`application.checker.registry`.

Each factory takes configuration keyword arguments (severity,
thresholds, ...) and returns a callable with the registry's standard
shape:

- check:    ``(ctx: CheckContext, spec: RuleSpec) -> Iterable[Issue]``
- sanitize: ``(ctx: CheckContext, spec: RuleSpec) -> str``

Importing this module triggers all `@register` decorators — call
:func:`ensure_loaded` from any code path that needs the registry to be
populated (the package ``__init__`` does this implicitly).

Rule names (kind="check"):

  - ``non_empty``           reject empty translations
  - ``length_bounds``       absolute char cap + short-target inverse-ratio
  - ``length_ratio``        tgt/src ratio, segmented by source word count
  - ``format_artifacts``    newline / markdown / hallucination / bracket
  - ``question_mark``       source ends ``?`` but target lacks it
  - ``keywords``            forbidden terms + cross-language keyword pairs
  - ``output_tokens``       Usage-aware token-explosion guard
  - ``cjk_content``         CJK target must contain CJK characters
  - ``trailing_annotation`` strip-style trailing ``（注…）`` notes
  - ``pixel_width``         optional Pillow rendered-width ratio guard

Sanitize names (kind="sanitize"):

  - ``strip_backticks``
  - ``trailing_annotation_strip``
  - ``colon_to_punctuation``
  - ``quote_strip``
  - ``leading_punct_strip``
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Iterable

from .registry import register
from .types import CheckContext, Issue, RuleSpec, Severity

if TYPE_CHECKING:
    from domain.model.usage import Usage  # noqa: F401  (only for typing)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


def _cjk_char_count(text: str) -> int:
    count = 0
    for ch in text:
        cp = ord(ch)
        if (
            0x4E00 <= cp <= 0x9FFF
            or 0x3400 <= cp <= 0x4DBF
            or 0x20000 <= cp <= 0x2A6DF
            or 0xF900 <= cp <= 0xFAFF
            or 0x2F800 <= cp <= 0x2FA1F
        ):
            count += 1
    return count


def _hangul_char_count(text: str) -> int:
    return sum(1 for ch in text if 0xAC00 <= ord(ch) <= 0xD7A3)


def _kana_char_count(text: str) -> int:
    return sum(1 for ch in text if 0x3040 <= ord(ch) <= 0x309F or 0x30A0 <= ord(ch) <= 0x30FF)


def _estimate_words(text: str) -> int:
    stripped = text.strip()
    cjk = _cjk_char_count(stripped) + _hangul_char_count(stripped) + _kana_char_count(stripped)
    if cjk > len(stripped) * 0.3:
        return cjk
    return len(stripped.split())


_CJK_LANGS = frozenset({"zh", "ja", "ko"})


# -------------------------------------------------------------------
# Check rules
# -------------------------------------------------------------------


@register("non_empty", kind="check")
def _non_empty_factory():
    """Reject empty/whitespace-only translations when source is non-empty."""

    def _fn(ctx: CheckContext, spec: RuleSpec) -> Iterable[Issue]:
        if ctx.source.strip() and not ctx.target.strip():
            yield Issue("non_empty", spec.severity, "translation is empty")

    return _fn


@register("length_bounds", kind="check")
def _length_bounds_factory(
    *,
    abs_max: int = 200,
    short_target_max: int = 3,
    short_target_inverse_ratio: float = 4.0,
):
    """Absolute char cap + short-target hallucination guard."""

    def _fn(ctx: CheckContext, spec: RuleSpec) -> Iterable[Issue]:
        src = ctx.source.strip()
        tgt = ctx.target.strip()
        if not src or not tgt:
            return
        if len(tgt) > abs_max:
            yield Issue(
                "length_abs_max",
                spec.severity,
                f"translation length {len(tgt)} exceeds absolute cap {abs_max}",
                details={"tgt_len": len(tgt), "cap": abs_max},
            )
            return
        if len(tgt) <= short_target_max and len(src) > short_target_inverse_ratio * len(tgt):
            yield Issue(
                "length_short_target",
                spec.severity,
                f"translation too short ({len(tgt)} chars) for source ({len(src)} chars)",
                details={"tgt_len": len(tgt), "src_len": len(src)},
            )

    return _fn


@register("length_ratio", kind="check")
def _length_ratio_factory(
    *,
    short: float = 5.0,
    medium: float = 3.0,
    long: float = 2.0,
    very_long: float = 1.6,
):
    """Char-length ratio guard, segmented by source word count."""

    def _fn(ctx: CheckContext, spec: RuleSpec) -> Iterable[Issue]:
        src = ctx.source.strip()
        tgt = ctx.target.strip()
        if not src or not tgt:
            return
        src_len = len(src)
        tgt_len = len(tgt)
        ratio = tgt_len / src_len
        words = _estimate_words(ctx.source)
        threshold = short if words < 3 else medium if words < 8 else long if words < 20 else very_long
        if ratio > threshold:
            yield Issue(
                "length_ratio",
                spec.severity,
                (f"length_ratio={ratio:.2f} exceeds threshold={threshold:.1f} (src_len={src_len}, tgt_len={tgt_len}, ~{words} words)"),
                details={"ratio": ratio, "threshold": threshold, "words": words},
            )

    return _fn


@register("format_artifacts", kind="check")
def _format_artifacts_factory(
    *,
    allow_newlines: bool = False,
    hallucination_starts: list[tuple[str, str | None]] | None = None,
):
    """Newlines, markdown bold, hallucination prefixes, bracket asymmetry."""
    # Pre-compile every pattern once at factory time so the hot path
    # (one match per call) is just a regex execution, not a recompile.
    compiled_starts: tuple[re.Pattern[str], ...] = tuple(
        re.compile(f"{pattern}(?!{exclude})" if exclude else pattern) for pattern, exclude in (hallucination_starts or ())
    )

    def _fn(ctx: CheckContext, spec: RuleSpec) -> Iterable[Issue]:
        tgt = ctx.target.strip()
        src = ctx.source.strip()
        tgt_lower = tgt.lower()

        # Allow newlines inside LaTeX math blocks (``$$ ... $$``): two or
        # more ``$$`` markers means the target is a multi-line equation,
        # not a hallucinated paragraph.
        if not allow_newlines and "\n" in tgt and tgt.count("$$") < 2:
            yield Issue("format_newline", spec.severity, "unexpected newline in translation")

        if "**" in tgt:
            yield Issue("format_markdown", spec.severity, "markdown bold artifact '**' in translation")

        for pat in compiled_starts:
            if pat.match(tgt_lower):
                yield Issue(
                    "format_hallucination",
                    spec.severity,
                    f"hallucination pattern: translation starts with '{tgt[:10]}...'",
                )
                break

        zh_openers = ("（", "【", "[", "(")
        en_openers = ("[", "(")
        if tgt.startswith(zh_openers) and not src.startswith(en_openers):
            yield Issue(
                "format_bracket",
                spec.severity,
                "translation starts with bracket but source does not",
            )

    return _fn


@register("question_mark", kind="check")
def _question_mark_factory(
    *,
    source_marks: list[str] | None = None,
    expected_marks: list[str] | None = None,
    whitelist_suffixes: list[str] | None = None,
    whitelist_severity: Severity = Severity.INFO,
):
    """Source ends with a question mark but translation has none.

    Both the source-side detector (``source_marks``) and the target-side
    expectation (``expected_marks``) are configurable so non-Latin source
    languages (Arabic ``؟``, Greek ``;`` …) can plug in their own marks
    via :class:`LangProfile`.
    """
    src_marks = tuple(source_marks or ["?", "？"])
    tgt_marks = tuple(expected_marks or ["?"])
    whitelist = tuple(
        suf.lower() for suf in (whitelist_suffixes if whitelist_suffixes is not None else ["right?", "ok?", "okay?", "okey?", "why?"])
    )

    def _fn(ctx: CheckContext, spec: RuleSpec) -> Iterable[Issue]:
        src = ctx.source.rstrip()
        if not any(src.endswith(m) for m in src_marks):
            return
        if any(m in ctx.target for m in tgt_marks):
            return
        is_whitelisted = any(src.lower().endswith(suf) for suf in whitelist)
        sev = whitelist_severity if is_whitelisted else spec.severity
        yield Issue(
            "question_mark",
            sev,
            ("source ends with '?' but translation has no question mark" + (" (whitelisted casual suffix)" if is_whitelisted else "")),
            details={"whitelisted": is_whitelisted},
        )

    return _fn


@register("keywords", kind="check")
def _keywords_factory(
    *,
    forbidden_terms: list[str] | None = None,
    keyword_pairs: list[tuple[list[str], list[str]]] | None = None,
):
    """Forbidden terms + cross-language keyword consistency."""
    forbidden = tuple(forbidden_terms or ())
    pairs = tuple((tuple(s), tuple(t)) for s, t in (keyword_pairs or ()))

    def _fn(ctx: CheckContext, spec: RuleSpec) -> Iterable[Issue]:
        tgt_lower = ctx.target.lower()
        src_lower = ctx.source.lower()
        for term in forbidden:
            if term.lower() in tgt_lower:
                yield Issue("keyword_forbidden", spec.severity, f"forbidden term found: '{term}'")
                break
        for src_kws, tgt_kws in pairs:
            tgt_match = any(kw.lower() in tgt_lower for kw in tgt_kws)
            if tgt_match:
                src_match = any(kw.lower() in src_lower for kw in src_kws)
                if not src_match:
                    yield Issue(
                        "keyword_inconsistency",
                        spec.severity,
                        f"target contains {list(tgt_kws)} but source lacks any of {list(src_kws)}",
                    )
                    break

    return _fn


@register("output_tokens", kind="check")
def _output_tokens_factory(
    *,
    max_output: int = 800,
    short_input_threshold: int = 50,
    short_input_max_output: int = 80,
    output_input_ratio_max: float = 10.0,
):
    """Token-explosion guard. Silently no-op when ``ctx.usage is None``."""

    def _fn(ctx: CheckContext, spec: RuleSpec) -> Iterable[Issue]:
        usage = ctx.usage
        if usage is None:
            return
        out_tok = usage.completion_tokens
        in_tok = usage.prompt_tokens

        if out_tok > max_output:
            yield Issue(
                "output_tokens_max",
                spec.severity,
                f"completion_tokens={out_tok} exceeds max={max_output}",
                details={"completion_tokens": out_tok, "max": max_output},
            )
            return
        if in_tok and in_tok < short_input_threshold and out_tok > short_input_max_output:
            yield Issue(
                "output_tokens_short_input",
                spec.severity,
                f"short input ({in_tok} tok) produced large output ({out_tok} tok)",
                details={"prompt_tokens": in_tok, "completion_tokens": out_tok},
            )
            return
        if in_tok and out_tok / in_tok > output_input_ratio_max:
            yield Issue(
                "output_tokens_ratio",
                spec.severity,
                f"output/input ratio {out_tok / in_tok:.2f} exceeds {output_input_ratio_max}",
                details={"ratio": out_tok / in_tok, "max": output_input_ratio_max},
            )

    return _fn


@register("cjk_content", kind="check")
def _cjk_content_factory(
    *,
    target_lang: str = "",
    short_passthrough_max: int = 10,
):
    """For zh/ja/ko targets, require at least one CJK character."""

    def _fn(ctx: CheckContext, spec: RuleSpec) -> Iterable[Issue]:
        lang = target_lang or ctx.target_lang
        if lang not in _CJK_LANGS:
            return
        tgt = ctx.target.strip()
        if not tgt:
            return
        if _cjk_char_count(tgt) + _hangul_char_count(tgt) + _kana_char_count(tgt) > 0:
            return
        if ctx.source.strip() == tgt and len(tgt) <= short_passthrough_max:
            return
        yield Issue(
            "cjk_content",
            spec.severity,
            f"target language is '{lang}' but translation has no CJK characters",
            details={"target_lang": lang},
        )

    return _fn


@register("trailing_annotation", kind="check")
def _trailing_annotation_factory(*, min_non_ascii: int = 12):
    """Detect trailing parenthesised annotation with ≥N non-ASCII chars inside."""
    pattern = re.compile(r"（([^（）]*?)）[,.?;!，。？；！]*$")

    def _fn(ctx: CheckContext, spec: RuleSpec) -> Iterable[Issue]:
        results = pattern.findall(ctx.target)
        if not results:
            return
        last = results[-1]
        non_ascii = sum(1 for ch in last if not ch.isascii())
        if non_ascii > min_non_ascii:
            yield Issue(
                "trailing_annotation",
                spec.severity,
                f"trailing parenthesized annotation ({non_ascii} non-ASCII chars): ...（{last[:20]}...）",
            )

    return _fn


@register("pixel_width", kind="check")
def _pixel_width_factory(
    *,
    font_path: str = "",
    font_size: int = 16,
    max_ratio: float = 4.0,
):
    """Optional Pillow-based pixel-width hallucination guard.

    The font is loaded **once** at factory time. No-ops when Pillow or
    the font path is unavailable. Because the Checker compiles each
    scene exactly once (see :class:`Checker._compiled`), this factory
    runs at most once per scene per process.
    """
    font = None
    if font_path:
        try:
            from PIL import ImageFont

            font = ImageFont.truetype(font_path, font_size)
        except Exception:
            font = None

    def _measure(text: str) -> int:
        try:
            bbox = font.getbbox(text)
            return int(bbox[2] - bbox[0])
        except Exception:
            try:
                return int(font.getlength(text))
            except Exception:
                return 0

    def _fn(ctx: CheckContext, spec: RuleSpec) -> Iterable[Issue]:
        if font is None:
            return
        src = ctx.source.strip()
        tgt = ctx.target.strip()
        if not src or not tgt:
            return
        src_w = _measure(src)
        if src_w <= 0:
            return
        tgt_w = _measure(tgt)
        ratio = tgt_w / src_w
        if ratio > max_ratio:
            yield Issue(
                "pixel_width",
                spec.severity,
                f"pixel-width ratio {ratio:.2f} exceeds {max_ratio}",
                details={"src_px": src_w, "tgt_px": tgt_w, "ratio": ratio},
            )

    return _fn


# -------------------------------------------------------------------
# Sanitize transforms
# -------------------------------------------------------------------


@register("strip_backticks", kind="sanitize")
def _strip_backticks_factory():
    """Strip surrounding backticks/newlines and any in-string backticks."""

    def _fn(ctx: CheckContext, spec: RuleSpec) -> str:
        t = re.sub(r"^`+|`+$|^\n+|\n+$", "", ctx.target)
        return t.replace("`", "")

    return _fn


@register("trailing_annotation_strip", kind="sanitize")
def _trailing_annotation_strip_factory(*, prefixes: tuple[str, ...] | list[str] | None = None):
    """Strip trailing ``（注…）`` / ``（说明…）`` LLM annotations."""
    prefs = tuple(prefixes or ("注", "说明", "注释"))
    patterns = [re.compile(rf"（{p}[^（）]*?）\s*[,.?;!，。？；！]*$") for p in prefs]

    def _fn(ctx: CheckContext, spec: RuleSpec) -> str:
        out = ctx.target
        for pat in patterns:
            out = pat.sub("", out)
        return out.strip()

    return _fn


_COLON_MAP = {".": "。", ",": "，", "!": "！", "?": "？"}


@register("colon_to_punctuation", kind="sanitize")
def _colon_to_punctuation_factory():
    """Trailing ``：`` → mirror source's ``. , ! ?`` to CJK punctuation."""

    def _fn(ctx: CheckContext, spec: RuleSpec) -> str:
        src = ctx.source.rstrip()
        tgt = ctx.target.rstrip()
        if not tgt.endswith("：") or not src:
            return ctx.target
        last = src[-1]
        if last in _COLON_MAP and last != ":":
            return tgt[:-1] + _COLON_MAP[last]
        return ctx.target

    return _fn


_QUOTE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("\u201c", "\u201d"),
    ("\u2018", "\u2019"),
    ('"', '"'),
    ("'", "'"),
)


@register("quote_strip", kind="sanitize")
def _quote_strip_factory():
    """Strip up to two layers of surrounding quote marks."""

    def _fn(ctx: CheckContext, spec: RuleSpec) -> str:
        cur = ctx.target
        for _ in range(2):
            stripped = cur
            for opener, closer in _QUOTE_PATTERNS:
                if len(stripped) >= 2 and stripped.startswith(opener) and stripped.endswith(closer):
                    stripped = stripped[len(opener) : -len(closer)]
                    break
            if stripped == cur:
                break
            cur = stripped
        return cur

    return _fn


@register("leading_punct_strip", kind="sanitize")
def _leading_punct_strip_factory():
    """Strip leading ``，`` / ``、`` / ``。`` / whitespace artifacts."""
    pat = re.compile(r"^[，、。\s]+")

    def _fn(ctx: CheckContext, spec: RuleSpec) -> str:
        return pat.sub("", ctx.target)

    return _fn


def ensure_loaded() -> None:
    """No-op marker that confirms this module has been imported.

    Importing :mod:`application.checker.rules_fn` is sufficient to
    populate the registry; this helper exists so downstream code can
    document the dependency explicitly.
    """
    return None
