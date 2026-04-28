"""Core types for the checker subsystem.

Defines severity levels, individual issues, and aggregate check reports.
All types are frozen (immutable).

Also exports the scene-redesign primitives:

- :class:`CheckContext` — generic payload (source/target/langs/usage/metadata)
- :class:`RuleSpec` — a rule reference (name + severity + params)
- :class:`ResolvedScene` — extends/disable/overrides 解析完毕后的不可变 scene
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from domain.model.usage import Usage


class Severity(Enum):
    """Check-result severity level."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass(frozen=True)
class Issue:
    """A single problem found by a rule."""

    rule: str
    severity: Severity
    message: str
    details: dict = field(default_factory=dict)


@dataclass(frozen=True)
class CheckReport:
    """Aggregate result of running all rules on one translation pair."""

    issues: tuple[Issue, ...] = ()

    @property
    def passed(self) -> bool:
        """True when no ERROR-level issues are present."""
        return not any(i.severity is Severity.ERROR for i in self.issues)

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity is Severity.ERROR]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity is Severity.WARNING]

    @property
    def infos(self) -> list[Issue]:
        return [i for i in self.issues if i.severity is Severity.INFO]

    @staticmethod
    def ok() -> CheckReport:
        return CheckReport()


# ---------------------------------------------------------------------------
# Scene-redesign primitives (additive in P1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CheckContext:
    """Generic payload passed to every check / sanitize step.

    Translation scenes populate ``source`` / ``target`` / language fields.
    Other scenes (subtitle, llm-response, terminology) may only use
    ``target`` and ``metadata``.

    The dataclass is **frozen**; callers wanting to advance ``target``
    after a sanitize step should use :func:`dataclasses.replace`.
    """

    source: str = ""
    target: str = ""
    source_lang: str = ""
    target_lang: str = ""
    usage: Usage | None = None
    prior: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RuleSpec:
    """Reference to a registered rule with optional override parameters.

    Produced by scene resolution (:mod:`application.checker._resolve`).
    Carried into rule functions so each rule can read its own parameters
    and severity uniformly.
    """

    name: str
    severity: Severity = Severity.ERROR
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ResolvedScene:
    """Frozen scene after extends / disable / overrides 全部展开。

    Built once at config-load time by :func:`resolve_scene`; pure data.
    """

    name: str
    sanitize: tuple[RuleSpec, ...] = ()
    rules: tuple[RuleSpec, ...] = ()
