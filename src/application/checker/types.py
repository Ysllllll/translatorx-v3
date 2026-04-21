"""Core types for the checker subsystem.

Defines severity levels, individual issues, and aggregate check reports.
All types are frozen (immutable).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


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
