"""Checker configuration — profile presets for rule construction.

Profiles define severity overrides and threshold adjustments.
They are applied by the factory when constructing rule instances.
"""

from __future__ import annotations

from pydantic import BaseModel

from .types import Severity


class ProfileOverrides(BaseModel, frozen=True):
    """Partial overrides applied when a profile is selected at check time.

    Only non-None fields take effect; everything else inherits from
    the base configuration.
    """

    ratio_severity: Severity | None = None
    ratio_thresholds_short: float | None = None
    ratio_thresholds_medium: float | None = None
    ratio_thresholds_long: float | None = None
    ratio_thresholds_very_long: float | None = None
    format_severity: Severity | None = None
    keyword_severity: Severity | None = None
    question_mark_severity: Severity | None = None
    annotation_severity: Severity | None = None


# Built-in profiles
PROFILES: dict[str, ProfileOverrides] = {
    "strict": ProfileOverrides(),
    "lenient": ProfileOverrides(
        ratio_severity=Severity.WARNING,
        ratio_thresholds_short=8.0,
        ratio_thresholds_medium=5.0,
        ratio_thresholds_long=3.5,
        ratio_thresholds_very_long=2.5,
        question_mark_severity=Severity.INFO,
    ),
    "minimal": ProfileOverrides(
        ratio_severity=Severity.WARNING,
        ratio_thresholds_short=10.0,
        ratio_thresholds_medium=8.0,
        ratio_thresholds_long=5.0,
        ratio_thresholds_very_long=4.0,
        format_severity=Severity.WARNING,
        keyword_severity=Severity.WARNING,
        question_mark_severity=Severity.INFO,
    ),
}

