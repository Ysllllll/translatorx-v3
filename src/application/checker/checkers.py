"""Checker — rule engine for translation quality checking.

The :class:`Checker` holds an ordered list of :class:`Rule` instances
and optional profile-based rule sets.  Calling :meth:`Checker.check`
runs every rule in order, collecting :class:`Issue` objects.  On the
first ``ERROR``-level issue the engine short-circuits and returns
immediately.
"""

from __future__ import annotations

from domain.model.usage import Usage

from .config import PROFILES, ProfileOverrides
from .rules import Rule, build_default_rules, RatioThresholds
from .types import CheckReport, Issue, Severity


class Checker:
    """Rule-based translation quality checker.

    Parameters
    ----------
    rules :
        Ordered rule list (the "base" rules).
    source_lang / target_lang :
        Bound at construction time so :meth:`check` doesn't need them.
    profile_rules :
        Optional mapping from profile name to alternative rule list.
        When ``check(..., profile="lenient")`` is called, the rules
        from this mapping are used instead of the base rules.
    """

    __slots__ = ("_rules", "_source_lang", "_target_lang", "_profile_rules")

    def __init__(
        self,
        rules: list[Rule],
        *,
        source_lang: str = "",
        target_lang: str = "",
        profile_rules: dict[str, list[Rule]] | None = None,
    ) -> None:
        self._rules = rules
        self._source_lang = source_lang
        self._target_lang = target_lang
        self._profile_rules = profile_rules or {}

    # ---- public API ------------------------------------------------

    def check(
        self,
        source: str,
        translation: str,
        *,
        profile: str | None = None,
        usage: Usage | None = None,
    ) -> CheckReport:
        """Run all rules and return a :class:`CheckReport`.

        If *profile* is given (e.g. ``"lenient"``), uses the
        profile-specific rule set if available.

        ``usage`` (optional) is forwarded to rules that support
        token-aware checks (e.g. :class:`OutputTokenRule`); rules that
        don't care swallow it via ``**_``.

        Short-circuit: on the first ``ERROR``-level issue, stop
        and return immediately (collecting everything up to that point).
        """
        rules = self._profile_rules.get(profile, self._rules) if profile else self._rules

        issues: list[Issue] = []
        for rule in rules:
            new_issues = rule.check(source, translation, usage=usage)
            issues.extend(new_issues)
            if any(i.severity is Severity.ERROR for i in new_issues):
                return CheckReport(issues=tuple(issues))

        return CheckReport(issues=tuple(issues))

    def regression(
        self,
        source: str,
        prior: str,
        candidate: str,
        *,
        profile: str | None = None,
        usage: Usage | None = None,
    ) -> bool:
        """Return True iff *candidate* is at least as good as *prior*.

        Mirrors the legacy CNENMatcher behaviour where re-translation
        attempts were rejected if they introduced new quality issues.
        Comparison is by error-count-then-warning-count: candidate is
        accepted when its (errors, warnings) tuple is ≤ prior's.
        """
        if prior == candidate:
            return True
        prior_report = self.check(source, prior, profile=profile, usage=usage)
        cand_report = self.check(source, candidate, profile=profile, usage=usage)

        def _score(report: CheckReport) -> tuple[int, int]:
            errors = sum(1 for i in report.issues if i.severity is Severity.ERROR)
            warnings = sum(1 for i in report.issues if i.severity is Severity.WARNING)
            return (errors, warnings)

        return _score(cand_report) <= _score(prior_report)

    # ---- introspection ---------------------------------------------

    @property
    def source_lang(self) -> str:
        return self._source_lang

    @property
    def target_lang(self) -> str:
        return self._target_lang

    @property
    def rules(self) -> list[Rule]:
        return list(self._rules)
