"""Built-in checker rules — classes that inspect translations.

Each rule is a class conforming to the :class:`Rule` Protocol::

    class MyRule:
        name: str
        severity: Severity

        def check(self, source: str, translation: str) -> list[Issue]:
            ...

Rules own their configuration (thresholds, patterns, etc.) and are
instantiated by the factory with language-appropriate parameters.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from .types import Issue, Severity


# -------------------------------------------------------------------
# Rule Protocol
# -------------------------------------------------------------------


@runtime_checkable
class Rule(Protocol):
    """Interface that every checker rule must satisfy."""

    @property
    def name(self) -> str: ...

    @property
    def severity(self) -> Severity: ...

    def check(self, source: str, translation: str) -> list[Issue]: ...


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

    def check(self, source: str, translation: str) -> list[Issue]:
        if not source.strip() or not translation.strip():
            return []

        src_len = len(source.strip())
        tgt_len = len(translation.strip())
        ratio = tgt_len / src_len
        word_count = _estimate_words(source)
        t = self._thresholds

        threshold = (
            t.short if word_count < 3 else t.medium if word_count < 8 else t.long if word_count < 20 else t.very_long
        )

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

    def check(self, source: str, translation: str) -> list[Issue]:
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
    """Source ends with ``?``/``？`` but translation has no question mark."""

    __slots__ = ("_severity", "_expected_marks")

    def __init__(
        self,
        severity: Severity = Severity.WARNING,
        expected_marks: list[str] | None = None,
    ) -> None:
        self._severity = severity
        self._expected_marks = expected_marks or ["?"]

    @property
    def name(self) -> str:
        return "question_mark"

    @property
    def severity(self) -> Severity:
        return self._severity

    @property
    def expected_marks(self) -> list[str]:
        return list(self._expected_marks)

    def check(self, source: str, translation: str) -> list[Issue]:
        src = source.rstrip()
        if src.endswith("?") or src.endswith("？"):
            if not any(m in translation for m in self._expected_marks):
                return [
                    Issue(
                        "question_mark",
                        self._severity,
                        "source ends with '?' but translation has no question mark",
                    )
                ]
        return []


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

    def check(self, source: str, translation: str) -> list[Issue]:
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

    def check(self, source: str, translation: str) -> list[Issue]:
        pattern = r"（([^（）]*?)）[,.?;!，。？；！]*$"
        results = re.findall(pattern, translation)
        if results and len(results) > 0:
            non_ascii = sum(1 for ch in results[-1] if not ch.isascii())
            if non_ascii > self._min_non_ascii:
                return [
                    Issue(
                        "trailing_annotation",
                        self._severity,
                        f"trailing parenthesized annotation ({non_ascii} non-ASCII chars): "
                        f"...（{results[-1][:20]}...）",
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
    keyword_severity: Severity = Severity.ERROR,
    forbidden_terms: list[str] | None = None,
    keyword_pairs: list[tuple[list[str], list[str]]] | None = None,
    annotation_severity: Severity = Severity.ERROR,
    annotation_min_non_ascii: int = 12,
) -> list[Rule]:
    """Build the default ordered rule list with the given parameters."""
    return [
        LengthRatioRule(severity=ratio_severity, thresholds=ratio_thresholds),
        FormatRule(
            severity=format_severity,
            allow_newlines=allow_newlines,
            hallucination_starts=hallucination_starts,
        ),
        QuestionMarkRule(
            severity=question_mark_severity,
            expected_marks=expected_question_marks,
        ),
        KeywordRule(
            severity=keyword_severity,
            forbidden_terms=forbidden_terms,
            keyword_pairs=keyword_pairs,
        ),
        TrailingAnnotationRule(
            severity=annotation_severity,
            min_non_ascii=annotation_min_non_ascii,
        ),
    ]
