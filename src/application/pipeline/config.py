"""YAML pipeline config loader — parse declarative pipeline definitions.

Phase 1 / Step 7. Supports::

    name: my_pipeline
    build:
      stage: from_srt
      params:
        path: lec01.srt
        language: en
    structure:
      - stage: punc
        params:
          language: en
      - stage: chunk
        params:
          language: en
    enrich:
      - stage: translate
    on_error: abort
    version: 1
    metadata:
      owner: data-team

The schema is intentionally minimal — all stage-specific validation
lives in the stage's own ``Params`` model, applied by
:class:`StageRegistry.build`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from ports.pipeline import ErrorPolicy, PipelineDef, StageDef

__all__ = [
    "load_pipeline_yaml",
    "load_pipeline_dict",
    "parse_pipeline_yaml",
]


def parse_pipeline_yaml(text: str) -> PipelineDef:
    """Parse a YAML string into a :class:`PipelineDef`."""
    data = yaml.safe_load(text) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"pipeline YAML must be a mapping at top level, got {type(data).__name__}")
    return load_pipeline_dict(data)


def load_pipeline_yaml(path: str | Path) -> PipelineDef:
    """Load and parse a YAML pipeline config file."""
    text = Path(path).read_text(encoding="utf-8")
    return parse_pipeline_yaml(text)


def load_pipeline_dict(data: Mapping[str, Any]) -> PipelineDef:
    """Convert a parsed YAML/JSON mapping into a :class:`PipelineDef`."""
    name = data.get("name", "pipeline")
    if not isinstance(name, str):
        raise ValueError(f"pipeline.name must be a string, got {type(name).__name__}")

    build_raw = data.get("build")
    if build_raw is None:
        raise ValueError("pipeline config requires a 'build' stage")
    build = _parse_stage(build_raw, where="build")

    structure_raw = data.get("structure", []) or []
    enrich_raw = data.get("enrich", []) or []
    if not isinstance(structure_raw, list):
        raise ValueError("pipeline.structure must be a list")
    if not isinstance(enrich_raw, list):
        raise ValueError("pipeline.enrich must be a list")

    structure = tuple(_parse_stage(s, where=f"structure[{i}]") for i, s in enumerate(structure_raw))
    enrich = tuple(_parse_stage(s, where=f"enrich[{i}]") for i, s in enumerate(enrich_raw))

    on_error_raw = data.get("on_error", "abort")
    if isinstance(on_error_raw, ErrorPolicy):
        on_error = on_error_raw
    elif isinstance(on_error_raw, str):
        try:
            on_error = ErrorPolicy(on_error_raw)
        except ValueError as exc:
            valid = ", ".join(p.value for p in ErrorPolicy)
            raise ValueError(f"pipeline.on_error={on_error_raw!r} not in [{valid}]") from exc
    else:
        raise ValueError(f"pipeline.on_error must be a string, got {type(on_error_raw).__name__}")

    version = int(data.get("version", 1))
    metadata = data.get("metadata") or {}
    if not isinstance(metadata, Mapping):
        raise ValueError("pipeline.metadata must be a mapping")

    return PipelineDef(
        name=name,
        build=build,
        structure=structure,
        enrich=enrich,
        on_error=on_error,
        version=version,
        metadata=dict(metadata),
    )


def _parse_stage(raw: Any, *, where: str) -> StageDef:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{where}: expected a mapping, got {type(raw).__name__}")
    # Allow either ``stage: name`` or ``name: name`` for the stage identifier.
    stage_name = raw.get("stage") or raw.get("name")
    if not stage_name or not isinstance(stage_name, str):
        raise ValueError(f"{where}: missing 'stage' (or 'name') key")

    params_raw = raw.get("params") or {}
    if not isinstance(params_raw, Mapping):
        raise ValueError(f"{where}.params: expected a mapping, got {type(params_raw).__name__}")

    when = raw.get("when")
    if when is not None and not isinstance(when, str):
        raise ValueError(f"{where}.when: expected a string, got {type(when).__name__}")

    stage_id = raw.get("id")
    if stage_id is not None and not isinstance(stage_id, str):
        raise ValueError(f"{where}.id: expected a string, got {type(stage_id).__name__}")

    return StageDef(
        name=stage_name,
        params=dict(params_raw),
        when=when,
        id=stage_id,
    )
