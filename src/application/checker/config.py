"""Checker configuration — profile presets for rule construction.

Profiles define severity overrides and threshold adjustments.
They are applied by the factory when constructing rule instances.

The :class:`CheckerConfig` model provides YAML wiring for the full
checker stack: thresholds, sanitizer toggles, optional pixel-width,
and profile selection.  It is consumed by :func:`from_config` in
:mod:`application.checker.factory`.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

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


# -------------------------------------------------------------------
# YAML-facing CheckerConfig
# -------------------------------------------------------------------


class RatioThresholdsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    short: float | None = None
    medium: float | None = None
    long: float | None = None
    very_long: float | None = None


class LengthBoundsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    abs_max: int = 200
    short_target_max: int = 3
    short_target_inverse_ratio: float = 4.0


class QuestionMarksConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    whitelist_suffixes: list[str] = Field(default_factory=lambda: ["right?", "ok?", "okay?", "okey?", "why?"])
    whitelist_severity: Literal["error", "warning", "info"] = "info"


class OutputTokensConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_output: int = 800
    short_input_threshold: int = 50
    short_input_max_output: int = 80
    output_input_ratio_max: float = 10.0


class PixelWidthConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    font_path: str = ""
    font_size: int = 16
    max_ratio: float = 4.0


class SanitizeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    backticks: bool = True
    trailing_annotation: bool = True
    trailing_colon_to_punct: bool = True
    quote_strip: bool = True
    leading_punct_strip: bool = True


class CheckerConfig(BaseModel):
    """YAML-facing configuration for the checker stack."""

    model_config = ConfigDict(extra="forbid")

    default_profile: Literal["strict", "lenient", "minimal"] = "strict"
    ratio_thresholds: RatioThresholdsConfig = Field(default_factory=RatioThresholdsConfig)
    length_bounds: LengthBoundsConfig = Field(default_factory=LengthBoundsConfig)
    question_marks: QuestionMarksConfig = Field(default_factory=QuestionMarksConfig)
    output_tokens: OutputTokensConfig = Field(default_factory=OutputTokensConfig)
    pixel_width: PixelWidthConfig = Field(default_factory=PixelWidthConfig)
    sanitize: SanitizeConfig = Field(default_factory=SanitizeConfig)
