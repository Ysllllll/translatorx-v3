"""Built-in checker rules — classes that inspect translations.

Each rule is a class conforming to the :class:`Rule` Protocol::

    class MyRule:
        name: str
        severity: Severity

        def check(self, source: str, translation: str, **_) -> list[Issue]:
            ...

Rules own their configuration (thresholds, patterns, etc.) and are
instantiated by the factory with language-appropriate parameters.

Rules may optionally read ``usage: Usage | None`` from kwargs to make
token-aware decisions; rules that don't care just keep accepting
``**_`` and ignore it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from domain.model.usage import Usage

from .types import Issue, Severity


# -------------------------------------------------------------------
# Rule Protocol
# -------------------------------------------------------------------


@runtime_checkable
class Rule(Protocol):
    """Interface that every checker rule must satisfy.

    The ``check`` method accepts arbitrary keyword args; in addition to
    ``source`` and ``translation``, the orchestrator may pass
    ``usage: Usage | None``. Rules that don't need it should simply
    swallow ``**_``.
    """

    @property
    def name(self) -> str: ...

    @property
    def severity(self) -> Severity: ...

    def check(self, source: str, translation: str, **kwargs) -> list[Issue]: ...


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


def _cjk_char_count(text: str) -> int:
    """Count CJK unified ideograph characters."""
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
    """Estimate word count — CJK chars each count as one word."""
    cjk = _cjk_char_count(text) + _hangul_char_count(text) + _kana_char_count(text)
    if cjk > len(text.strip()) * 0.3:
        return cjk
    return len(text.split())


# -------------------------------------------------------------------
# Rule: LengthRatioRule
# -------------------------------------------------------------------


@dataclass(frozen=True)
class RatioThresholds:
    """Character-length-ratio thresholds segmented by source word count."""

    short: float = 5.0  # < 3 words
    medium: float = 3.0  # < 8 words
    long: float = 2.0  # < 20 words
    very_long: float = 1.6  # >= 20 words


class LengthRatioRule:
    """Reject translations whose char-length ratio is abnormally high."""

    __slots__ = ("_severity", "_thresholds")

    def __init__(
        self,
        severity: Severity = Severity.ERROR,
        thresholds: RatioThresholds | None = None,
    ) -> None:
        self._severity = severity
        self._thresholds = thresholds or RatioThresholds()

    @property
    def name(self) -> str:
        return "length_ratio"

    @property
    def severity(self) -> Severity:
        return self._severity

    @property
    def thresholds(self) -> RatioThresholds:
        return self._thresholds

    def check(self, source: str, translation: str, **_) -> list[Issue]:
        if not source.strip() or not translation.strip():
            return []

        src_len = len(source.strip())
        tgt_len = len(translation.strip())
        ratio = tgt_len / src_len
        word_count = _estimate_words(source)
        t = self._thresholds

        threshold = t.short if word_count < 3 else t.medium if word_count < 8 else t.long if word_count < 20 else t.very_long

        if ratio > threshold:
            return [
                Issue(
                    rule="length_ratio",
                    severity=self._severity,
                    message=(
                        f"length_ratio={ratio:.2f} exceeds threshold={threshold:.1f} "
                        f"(src_len={src_len}, tgt_len={tgt_len}, ~{word_count} words)"
                    ),
                    details={"ratio": ratio, "threshold": threshold, "words": word_count},
                )
            ]
        return []


# -------------------------------------------------------------------
# Rule: FormatRule
# -------------------------------------------------------------------


class FormatRule:
    """Check structural issues: newlines, markdown, hallucination starts, brackets."""

    __slots__ = ("_severity", "_allow_newlines", "_hallucination_starts")

    def __init__(
        self,
        severity: Severity = Severity.ERROR,
        allow_newlines: bool = False,
        hallucination_starts: list[tuple[str, str | None]] | None = None,
    ) -> None:
        self._severity = severity
        self._allow_newlines = allow_newlines
        self._hallucination_starts = hallucination_starts or []

    @property
    def name(self) -> str:
        return "format"

    @property
    def severity(self) -> Severity:
        return self._severity

    @property
    def allow_newlines(self) -> bool:
        return self._allow_newlines

    @property
    def hallucination_starts(self) -> list[tuple[str, str | None]]:
        return list(self._hallucination_starts)

    def check(self, source: str, translation: str, **_) -> list[Issue]:
        issues: list[Issue] = []
        tgt = translation.strip()
        src = source.strip()
        tgt_lower = tgt.lower()

        # Unexpected newlines
        if not self._allow_newlines and "\n" in tgt:
            if tgt.count("$$") < 2:
                issues.append(
                    Issue(
                        "format_newline",
                        self._severity,
                        "unexpected newline in translation",
                    )
                )

        # Markdown bold artifacts
        if "**" in tgt:
            issues.append(
                Issue(
                    "format_markdown",
                    self._severity,
                    "markdown bold artifact '**' in translation",
                )
            )

        # Hallucination opening patterns
        for pattern, exclude in self._hallucination_starts:
            full = f"{pattern}(?!{exclude})" if exclude else pattern
            if re.match(full, tgt_lower):
                issues.append(
                    Issue(
                        "format_hallucination",
                        self._severity,
                        f"hallucination pattern: translation starts with '{tgt[:10]}...'",
                    )
                )
                break

        # Bracket inconsistency
        zh_openers = ("（", "【", "[", "(")
        en_openers = ("[", "(")
        if tgt.startswith(zh_openers) and not src.startswith(en_openers):
            issues.append(
                Issue(
                    "format_bracket",
                    self._severity,
                    "translation starts with bracket but source does not",
                )
            )

        return issues


# -------------------------------------------------------------------
# Rule: QuestionMarkRule
# -------------------------------------------------------------------


class QuestionMarkRule:
    """Source ends with ``?``/``？`` but translation has no question mark.

    The *whitelist_suffixes* parameter mirrors the legacy ``CNENMatcher``
    behaviour: when the source ends with one of these casual tail tokens
    (``right?``, ``ok?``, ``why?``…) the rule degrades to ``INFO``
    severity, since these rarely require a literal "？" in CJK output.
    """

    __slots__ = ("_severity", "_expected_marks", "_whitelist_suffixes", "_whitelist_severity")

    def __init__(
        self,
        severity: Severity = Severity.WARNING,
        expected_marks: list[str] | None = None,
        whitelist_suffixes: list[str] | None = None,
        whitelist_severity: Severity = Severity.INFO,
    ) -> None:
        self._severity = severity
        self._expected_marks = expected_marks or ["?"]
        self._whitelist_suffixes = whitelist_suffixes or [
            "right?",
            "ok?",
            "okay?",
            "okey?",
            "why?",
        ]
        self._whitelist_severity = whitelist_severity

    @property
    def name(self) -> str:
        return "question_mark"

    @property
    def severity(self) -> Severity:
        return self._severity

    @property
    def expected_marks(self) -> list[str]:
        return list(self._expected_marks)

    @property
    def whitelist_suffixes(self) -> list[str]:
        return list(self._whitelist_suffixes)

    def check(self, source: str, translation: str, **_) -> list[Issue]:
        src = source.rstrip()
        if not (src.endswith("?") or src.endswith("？")):
            return []
        if any(m in translation for m in self._expected_marks):
            return []

        src_lower = src.lower()
        whitelisted = any(src_lower.endswith(suf.lower()) for suf in self._whitelist_suffixes)
        sev = self._whitelist_severity if whitelisted else self._severity

        return [
            Issue(
                "question_mark",
                sev,
                "source ends with '?' but translation has no question mark" + (" (whitelisted casual suffix)" if whitelisted else ""),
                details={"whitelisted": whitelisted},
            )
        ]


# -------------------------------------------------------------------
# Rule: KeywordRule
# -------------------------------------------------------------------


class KeywordRule:
    """Check forbidden terms and cross-language keyword consistency."""

    __slots__ = ("_severity", "_forbidden_terms", "_keyword_pairs")

    def __init__(
        self,
        severity: Severity = Severity.ERROR,
        forbidden_terms: list[str] | None = None,
        keyword_pairs: list[tuple[list[str], list[str]]] | None = None,
    ) -> None:
        self._severity = severity
        self._forbidden_terms = forbidden_terms or []
        self._keyword_pairs = keyword_pairs or []

    @property
    def name(self) -> str:
        return "keywords"

    @property
    def severity(self) -> Severity:
        return self._severity

    @property
    def forbidden_terms(self) -> list[str]:
        return list(self._forbidden_terms)

    @property
    def keyword_pairs(self) -> list[tuple[list[str], list[str]]]:
        return list(self._keyword_pairs)

    def check(self, source: str, translation: str, **_) -> list[Issue]:
        issues: list[Issue] = []
        tgt_lower = translation.lower()
        src_lower = source.lower()

        for term in self._forbidden_terms:
            if term.lower() in tgt_lower:
                issues.append(
                    Issue(
                        "keyword_forbidden",
                        self._severity,
                        f"forbidden term found: '{term}'",
                    )
                )
                break

        for src_keywords, tgt_keywords in self._keyword_pairs:
            tgt_match = any(kw.lower() in tgt_lower for kw in tgt_keywords)
            if tgt_match:
                src_match = any(kw.lower() in src_lower for kw in src_keywords)
                if not src_match:
                    issues.append(
                        Issue(
                            "keyword_inconsistency",
                            self._severity,
                            f"target contains {tgt_keywords} but source lacks any of {src_keywords}",
                        )
                    )
                    break

        return issues


# -------------------------------------------------------------------
# Rule: EmptyTranslationRule
# -------------------------------------------------------------------


class EmptyTranslationRule:
    """Reject empty / whitespace-only translations when source is non-empty."""

    __slots__ = ("_severity",)

    def __init__(self, severity: Severity = Severity.ERROR) -> None:
        self._severity = severity

    @property
    def name(self) -> str:
        return "empty_translation"

    @property
    def severity(self) -> Severity:
        return self._severity

    def check(self, source: str, translation: str, **_) -> list[Issue]:
        if source.strip() and not translation.strip():
            return [Issue("empty_translation", self._severity, "translation is empty")]
        return []


# -------------------------------------------------------------------
# Rule: LengthBoundsRule
# -------------------------------------------------------------------


@dataclass(frozen=True)
class LengthBounds:
    """Absolute character bounds + inverse-ratio short-target rule."""

    abs_max: int = 200
    short_target_max: int = 3
    short_target_inverse_ratio: float = 4.0


class LengthBoundsRule:
    """Catch absurd absolute lengths and short-target hallucinations.

    Complements :class:`LengthRatioRule` (which guards tgt/src upper
    bound).  Three guards:

    1. ``abs_max`` — translations longer than this are almost always
       hallucinations.
    2. Short-target rule — when the target is very short
       (``<= short_target_max`` chars) but the source is more than
       ``short_target_inverse_ratio × tgt_len`` long, the translation
       likely dropped content.
    """

    __slots__ = ("_severity", "_bounds")

    def __init__(
        self,
        severity: Severity = Severity.ERROR,
        bounds: LengthBounds | None = None,
    ) -> None:
        self._severity = severity
        self._bounds = bounds or LengthBounds()

    @property
    def name(self) -> str:
        return "length_bounds"

    @property
    def severity(self) -> Severity:
        return self._severity

    @property
    def bounds(self) -> LengthBounds:
        return self._bounds

    def check(self, source: str, translation: str, **_) -> list[Issue]:
        src = source.strip()
        tgt = translation.strip()
        if not src or not tgt:
            return []

        b = self._bounds
        issues: list[Issue] = []

        if len(tgt) > b.abs_max:
            issues.append(
                Issue(
                    "length_abs_max",
                    self._severity,
                    f"translation length {len(tgt)} exceeds absolute cap {b.abs_max}",
                    details={"tgt_len": len(tgt), "cap": b.abs_max},
                )
            )
            return issues

        if len(tgt) <= b.short_target_max and len(src) > b.short_target_inverse_ratio * len(tgt):
            issues.append(
                Issue(
                    "length_short_target",
                    self._severity,
                    f"translation too short ({len(tgt)} chars) for source ({len(src)} chars)",
                    details={"tgt_len": len(tgt), "src_len": len(src)},
                )
            )
        return issues


# -------------------------------------------------------------------
# Rule: CJKContentRule
# -------------------------------------------------------------------


_CJK_LANGS = frozenset({"zh", "ja", "ko"})


class CJKContentRule:
    """For CJK target languages, require at least one CJK character.

    A common LLM failure is returning the source untranslated.  When
    the target language is zh/ja/ko but the output has zero CJK
    characters, that's almost certainly a regression.

    The rule is a no-op for non-CJK targets, and degrades to no-op
    when source ≈ target and is short (<10 chars) — proper nouns,
    code identifiers, etc.
    """

    __slots__ = ("_severity", "_target_lang", "_short_passthrough_max")

    def __init__(
        self,
        severity: Severity = Severity.ERROR,
        target_lang: str = "",
        short_passthrough_max: int = 10,
    ) -> None:
        self._severity = severity
        self._target_lang = target_lang
        self._short_passthrough_max = short_passthrough_max

    @property
    def name(self) -> str:
        return "cjk_content"

    @property
    def severity(self) -> Severity:
        return self._severity

    @property
    def target_lang(self) -> str:
        return self._target_lang

    def check(self, source: str, translation: str, **_) -> list[Issue]:
        if self._target_lang not in _CJK_LANGS:
            return []
        tgt = translation.strip()
        if not tgt:
            return []
        if _cjk_char_count(tgt) + _hangul_char_count(tgt) + _kana_char_count(tgt) > 0:
            return []
        if source.strip() == tgt and len(tgt) <= self._short_passthrough_max:
            return []
        return [
            Issue(
                "cjk_content",
                self._severity,
                f"target language is '{self._target_lang}' but translation has no CJK characters",
                details={"target_lang": self._target_lang},
            )
        ]


# -------------------------------------------------------------------
# Rule: OutputTokenRule
# -------------------------------------------------------------------


@dataclass(frozen=True)
class OutputTokenLimits:
    """Token-level guards — backfill of legacy ``check_response`` token checks."""

    max_output: int = 800
    short_input_threshold: int = 50
    short_input_max_output: int = 80
    output_input_ratio_max: float = 10.0


class OutputTokenRule:
    """Catch token-explosion failures using ``Usage`` from the LLM call.

    Three guards (mirror old ``Agent.check_response``):

    1. ``output_tokens > max_output``  → hard cap.
    2. ``input < short_input_threshold`` *and*
       ``output > short_input_max_output`` → small input → big output.
    3. ``output / input > output_input_ratio_max`` → ratio explosion.

    Silently no-op if ``usage`` is not provided.
    """

    __slots__ = ("_severity", "_limits")

    def __init__(
        self,
        severity: Severity = Severity.WARNING,
        limits: OutputTokenLimits | None = None,
    ) -> None:
        self._severity = severity
        self._limits = limits or OutputTokenLimits()

    @property
    def name(self) -> str:
        return "output_tokens"

    @property
    def severity(self) -> Severity:
        return self._severity

    @property
    def limits(self) -> OutputTokenLimits:
        return self._limits

    def check(self, source: str, translation: str, **kwargs) -> list[Issue]:
        usage: Usage | None = kwargs.get("usage")
        if usage is None:
            return []

        lim = self._limits
        out_tok = usage.completion_tokens
        in_tok = usage.prompt_tokens
        issues: list[Issue] = []

        if out_tok > lim.max_output:
            issues.append(
                Issue(
                    "output_tokens_max",
                    self._severity,
                    f"completion_tokens={out_tok} exceeds max={lim.max_output}",
                    details={"completion_tokens": out_tok, "max": lim.max_output},
                )
            )
            return issues

        if in_tok and in_tok < lim.short_input_threshold and out_tok > lim.short_input_max_output:
            issues.append(
                Issue(
                    "output_tokens_short_input",
                    self._severity,
                    f"short input ({in_tok} tok) produced large output ({out_tok} tok)",
                    details={"prompt_tokens": in_tok, "completion_tokens": out_tok},
                )
            )

        if in_tok and out_tok / in_tok > lim.output_input_ratio_max:
            issues.append(
                Issue(
                    "output_tokens_ratio",
                    self._severity,
                    f"output/input ratio {out_tok / in_tok:.2f} exceeds {lim.output_input_ratio_max}",
                    details={"ratio": out_tok / in_tok, "max": lim.output_input_ratio_max},
                )
            )

        return issues


# -------------------------------------------------------------------
# Rule: PixelWidthRule
# -------------------------------------------------------------------


@dataclass(frozen=True)
class PixelWidthLimits:
    """Pixel-width hallucination thresholds (legacy CNENMatcher behaviour)."""

    font_path: str = ""
    font_size: int = 16
    max_ratio: float = 4.0


class PixelWidthRule:
    """Detect translations whose rendered pixel-width is suspiciously long.

    Mirrors the legacy ``CNENMatcher`` pixel-width check that uses the
    same font as the player to spot lines that *look* much longer than
    the source — a common hallucination signature.

    Pillow is imported lazily; if the library or font is unavailable
    the rule silently no-ops.
    """

    __slots__ = ("_severity", "_limits", "_font", "_load_attempted")

    def __init__(
        self,
        severity: Severity = Severity.WARNING,
        limits: PixelWidthLimits | None = None,
    ) -> None:
        self._severity = severity
        self._limits = limits or PixelWidthLimits()
        self._font = None
        self._load_attempted = False

    @property
    def name(self) -> str:
        return "pixel_width"

    @property
    def severity(self) -> Severity:
        return self._severity

    @property
    def limits(self) -> PixelWidthLimits:
        return self._limits

    def _ensure_font(self):
        if self._load_attempted:
            return self._font
        self._load_attempted = True
        if not self._limits.font_path:
            return None
        try:
            from PIL import ImageFont

            self._font = ImageFont.truetype(self._limits.font_path, self._limits.font_size)
        except Exception:
            self._font = None
        return self._font

    def _pixel_width(self, text: str, font) -> int:
        try:
            bbox = font.getbbox(text)
            return int(bbox[2] - bbox[0])
        except Exception:
            try:
                return int(font.getlength(text))
            except Exception:
                return 0

    def check(self, source: str, translation: str, **_) -> list[Issue]:
        font = self._ensure_font()
        if font is None:
            return []
        src = source.strip()
        tgt = translation.strip()
        if not src or not tgt:
            return []
        src_w = self._pixel_width(src, font)
        tgt_w = self._pixel_width(tgt, font)
        if src_w <= 0:
            return []
        ratio = tgt_w / src_w
        if ratio > self._limits.max_ratio:
            return [
                Issue(
                    "pixel_width",
                    self._severity,
                    f"pixel-width ratio {ratio:.2f} exceeds {self._limits.max_ratio}",
                    details={"src_px": src_w, "tgt_px": tgt_w, "ratio": ratio},
                )
            ]
        return []


# -------------------------------------------------------------------
# Rule: TrailingAnnotationRule
# -------------------------------------------------------------------


class TrailingAnnotationRule:
    """Detect LLM-added trailing annotations in parentheses.

    LLMs often append explanatory notes like ``（注：这里指...）`` at the end.
    If the non-ASCII content inside trailing parentheses exceeds a threshold,
    it's likely a hallucinated annotation.
    """

    __slots__ = ("_severity", "_min_non_ascii")

    def __init__(
        self,
        severity: Severity = Severity.ERROR,
        min_non_ascii: int = 12,
    ) -> None:
        self._severity = severity
        self._min_non_ascii = min_non_ascii

    @property
    def name(self) -> str:
        return "trailing_annotation"

    @property
    def severity(self) -> Severity:
        return self._severity

    def check(self, source: str, translation: str, **_) -> list[Issue]:
        pattern = r"（([^（）]*?)）[,.?;!，。？；！]*$"
        results = re.findall(pattern, translation)
        if results and len(results) > 0:
            non_ascii = sum(1 for ch in results[-1] if not ch.isascii())
            if non_ascii > self._min_non_ascii:
                return [
                    Issue(
                        "trailing_annotation",
                        self._severity,
                        f"trailing parenthesized annotation ({non_ascii} non-ASCII chars): ...（{results[-1][:20]}...）",
                    )
                ]
        return []


# -------------------------------------------------------------------
# Default rule list builder
# -------------------------------------------------------------------


def build_default_rules(
    *,
    ratio_severity: Severity = Severity.ERROR,
    ratio_thresholds: RatioThresholds | None = None,
    format_severity: Severity = Severity.ERROR,
    allow_newlines: bool = False,
    hallucination_starts: list[tuple[str, str | None]] | None = None,
    question_mark_severity: Severity = Severity.WARNING,
    expected_question_marks: list[str] | None = None,
    question_mark_whitelist: list[str] | None = None,
    keyword_severity: Severity = Severity.ERROR,
    forbidden_terms: list[str] | None = None,
    keyword_pairs: list[tuple[list[str], list[str]]] | None = None,
    annotation_severity: Severity = Severity.ERROR,
    annotation_min_non_ascii: int = 12,
    empty_severity: Severity = Severity.ERROR,
    bounds_severity: Severity = Severity.ERROR,
    length_bounds: LengthBounds | None = None,
    cjk_content_severity: Severity = Severity.ERROR,
    target_lang: str = "",
    output_token_severity: Severity = Severity.WARNING,
    output_token_limits: OutputTokenLimits | None = None,
    pixel_width_severity: Severity = Severity.WARNING,
    pixel_width_limits: PixelWidthLimits | None = None,
) -> list[Rule]:
    """Build the default ordered rule list with the given parameters."""
    return [
        EmptyTranslationRule(severity=empty_severity),
        LengthBoundsRule(severity=bounds_severity, bounds=length_bounds),
        LengthRatioRule(severity=ratio_severity, thresholds=ratio_thresholds),
        FormatRule(
            severity=format_severity,
            allow_newlines=allow_newlines,
            hallucination_starts=hallucination_starts,
        ),
        QuestionMarkRule(
            severity=question_mark_severity,
            expected_marks=expected_question_marks,
            whitelist_suffixes=question_mark_whitelist,
        ),
        KeywordRule(
            severity=keyword_severity,
            forbidden_terms=forbidden_terms,
            keyword_pairs=keyword_pairs,
        ),
        OutputTokenRule(
            severity=output_token_severity,
            limits=output_token_limits,
        ),
        PixelWidthRule(
            severity=pixel_width_severity,
            limits=pixel_width_limits,
        ),
        TrailingAnnotationRule(
            severity=annotation_severity,
            min_non_ascii=annotation_min_non_ascii,
        ),
        CJKContentRule(
            severity=cjk_content_severity,
            target_lang=target_lang,
        ),
    ]
