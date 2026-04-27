"""PipelineDef validator — registry-aware structural + param checks.

Phase 2 (Step B1). Pure function: given a :class:`PipelineDef` and a
:class:`StageRegistry`, verify that every referenced stage is registered
and that its params satisfy the stage's Pydantic ``Params`` model.

Two modes:

* :func:`validate_pipeline` — raises :class:`PipelineValidationError`
  on the first issue (default) or, with ``collect=True``, returns a
  :class:`ValidationReport` with every issue found.

The validator does **not** mutate either argument and stays free of I/O
so it can run on user-submitted YAML before a pipeline is dispatched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable

from ports.pipeline import ErrorPolicy, PipelineDef, StageDef

if TYPE_CHECKING:
    from .registry import StageRegistry

__all__ = [
    "PipelineValidationError",
    "ValidationIssue",
    "ValidationReport",
    "validate_pipeline",
]


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One problem found during validation."""

    path: str
    message: str

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.path}: {self.message}"


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Result of a collect-mode validation pass."""

    issues: tuple[ValidationIssue, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return not self.issues

    def raise_if_failed(self) -> None:
        if self.issues:
            raise PipelineValidationError(self.issues)


class PipelineValidationError(ValueError):
    """Raised when a pipeline fails validation."""

    def __init__(self, issues: Iterable[ValidationIssue]) -> None:
        self.issues: tuple[ValidationIssue, ...] = tuple(issues)
        msg = "; ".join(str(i) for i in self.issues) or "pipeline validation failed"
        super().__init__(msg)


def validate_pipeline(
    defn: PipelineDef,
    registry: "StageRegistry",
    *,
    collect: bool = False,
) -> ValidationReport:
    """Validate ``defn`` against ``registry``.

    Args:
        defn: The pipeline to check.
        registry: Registry to resolve stage names against.
        collect: If ``True`` accumulate every issue and return a report.
            If ``False`` (default) raise :class:`PipelineValidationError`
            on the first failure.
    """
    issues: list[ValidationIssue] = []

    def _emit(path: str, message: str) -> None:
        issue = ValidationIssue(path=path, message=message)
        if not collect:
            raise PipelineValidationError([issue])
        issues.append(issue)

    if not defn.name:
        _emit("name", "pipeline name must be a non-empty string")

    if not isinstance(defn.on_error, ErrorPolicy):
        _emit("on_error", f"unexpected on_error type {type(defn.on_error).__name__}")

    seen_ids: dict[str, str] = {}

    def _check_stage(stage: StageDef, where: str) -> None:
        if not registry.is_registered(stage.name):
            _emit(f"{where}.stage", f"stage {stage.name!r} is not registered")
            return
        schema = registry.schema_of(stage.name)
        if schema is not None:
            try:
                schema(**dict(stage.params))
            except Exception as exc:  # Pydantic ValidationError or TypeError
                _emit(
                    f"{where}.params",
                    f"invalid params for stage {stage.name!r}: {exc}",
                )
        sid = stage.id or stage.name
        if sid in seen_ids:
            _emit(
                f"{where}.id",
                f"duplicate stage id {sid!r} (also used at {seen_ids[sid]})",
            )
        else:
            seen_ids[sid] = where

    _check_stage(defn.build, "build")
    for i, s in enumerate(defn.structure):
        _check_stage(s, f"structure[{i}]")
    for i, s in enumerate(defn.enrich):
        _check_stage(s, f"enrich[{i}]")

    return ValidationReport(issues=tuple(issues))
