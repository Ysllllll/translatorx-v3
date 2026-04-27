"""YAML / dict pipeline loader — declarative ``PipelineDef`` source.

Phase 2 (Step B1). Supersedes the original :mod:`application.pipeline.config`
module which is now a thin re-export shim.

Adds, on top of the original loader:

* **Defaults block + vars override** — pipelines can declare ``defaults:``
  for substitution variables; callers can override via ``vars=`` to
  ``parse_pipeline_yaml`` / ``load_pipeline_yaml`` / ``load_pipeline_dict``.

* **Jinja-lite placeholders** — ``{{ name }}`` and
  ``{{ name | default(value) }}`` are interpolated everywhere in the
  config (string values, including nested ones inside ``params``).
  Single-placeholder strings are *typed* — if the value substitutes to a
  number / bool / null / list / dict, the result is that native value
  rather than its string repr.

* **Structured ``on_error``** — accepts either a string policy
  (``abort`` / ``continue`` / ``retry``) or a mapping
  ``{policy: ..., max_retries: 3}``. ``max_retries`` is stashed in
  ``PipelineDef.metadata['on_error_max_retries']``.

* **``on_cancel``** — optional mapping; its ``flush_store`` flag is
  stashed in ``PipelineDef.metadata['on_cancel_flush_store']``.

Stage-level *param* validation against the registry stays in
:mod:`application.pipeline.validator` to keep the loader pure (no
registry dependency).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

import yaml

from ports.pipeline import ErrorPolicy, PipelineDef, StageDef

__all__ = [
    "load_pipeline_yaml",
    "load_pipeline_dict",
    "parse_pipeline_yaml",
]


_PLACEHOLDER_RE = re.compile(
    r"""\{\{\s*
        (?P<name>[A-Za-z_][\w]*)
        (?:\s*\|\s*default\(\s*(?P<default>.*?)\s*\)\s*)?
        \s*\}\}""",
    re.VERBOSE,
)
_FULL_PLACEHOLDER_RE = re.compile(
    r"""^\s*\{\{\s*
        (?P<name>[A-Za-z_][\w]*)
        (?:\s*\|\s*default\(\s*(?P<default>.*?)\s*\)\s*)?
        \s*\}\}\s*$""",
    re.VERBOSE,
)


def parse_pipeline_yaml(text: str, *, vars: Mapping[str, Any] | None = None) -> PipelineDef:
    """Parse a YAML string into a :class:`PipelineDef`.

    ``vars`` overrides the YAML's ``defaults:`` block (caller wins).
    """
    data = yaml.safe_load(text) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"pipeline YAML must be a mapping at top level, got {type(data).__name__}")
    return load_pipeline_dict(data, vars=vars)


def load_pipeline_yaml(path: str | Path, *, vars: Mapping[str, Any] | None = None) -> PipelineDef:
    """Load and parse a YAML pipeline config file."""
    text = Path(path).read_text(encoding="utf-8")
    return parse_pipeline_yaml(text, vars=vars)


def load_pipeline_dict(data: Mapping[str, Any], *, vars: Mapping[str, Any] | None = None) -> PipelineDef:
    """Convert a parsed YAML/JSON mapping into a :class:`PipelineDef`."""
    defaults_raw = data.get("defaults") or {}
    if not isinstance(defaults_raw, Mapping):
        raise ValueError("pipeline.defaults must be a mapping")
    context: dict[str, Any] = dict(defaults_raw)
    if vars is not None:
        context.update(vars)

    name = data.get("name", "pipeline")
    if not isinstance(name, str):
        raise ValueError(f"pipeline.name must be a string, got {type(name).__name__}")

    build_raw = data.get("build")
    if build_raw is None:
        raise ValueError("pipeline config requires a 'build' stage")
    build = _parse_stage(build_raw, where="build", context=context)

    structure_raw = data.get("structure", []) or []
    enrich_raw = data.get("enrich", []) or []
    if not isinstance(structure_raw, list):
        raise ValueError("pipeline.structure must be a list")
    if not isinstance(enrich_raw, list):
        raise ValueError("pipeline.enrich must be a list")

    structure = tuple(_parse_stage(s, where=f"structure[{i}]", context=context) for i, s in enumerate(structure_raw))
    enrich = tuple(_parse_stage(s, where=f"enrich[{i}]", context=context) for i, s in enumerate(enrich_raw))

    on_error, on_error_extras = _parse_on_error(data.get("on_error", "abort"))

    on_cancel_raw = data.get("on_cancel")
    on_cancel_extras = _parse_on_cancel(on_cancel_raw)

    version = int(data.get("version", 1))
    metadata_raw = data.get("metadata") or {}
    if not isinstance(metadata_raw, Mapping):
        raise ValueError("pipeline.metadata must be a mapping")
    metadata: dict[str, Any] = dict(metadata_raw)
    metadata.update(on_error_extras)
    metadata.update(on_cancel_extras)

    return PipelineDef(
        name=name,
        build=build,
        structure=structure,
        enrich=enrich,
        on_error=on_error,
        version=version,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# stage parsing
# ---------------------------------------------------------------------------


def _parse_stage(raw: Any, *, where: str, context: Mapping[str, Any]) -> StageDef:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{where}: expected a mapping, got {type(raw).__name__}")

    stage_name = raw.get("stage") or raw.get("name")
    if not stage_name or not isinstance(stage_name, str):
        raise ValueError(f"{where}: missing 'stage' (or 'name') key")

    params_raw = raw.get("params") or {}
    if not isinstance(params_raw, Mapping):
        raise ValueError(f"{where}.params: expected a mapping, got {type(params_raw).__name__}")
    params = _interpolate(dict(params_raw), context=context, where=f"{where}.params")

    when = raw.get("when")
    if when is not None and not isinstance(when, str):
        raise ValueError(f"{where}.when: expected a string, got {type(when).__name__}")
    if isinstance(when, str):
        when = _interpolate_str(when, context=context, where=f"{where}.when")

    stage_id = raw.get("id")
    if stage_id is not None and not isinstance(stage_id, str):
        raise ValueError(f"{where}.id: expected a string, got {type(stage_id).__name__}")

    return StageDef(name=stage_name, params=params, when=when, id=stage_id)


# ---------------------------------------------------------------------------
# error / cancel sub-blocks
# ---------------------------------------------------------------------------


def _parse_on_error(raw: Any) -> tuple[ErrorPolicy, dict[str, Any]]:
    extras: dict[str, Any] = {}
    if isinstance(raw, ErrorPolicy):
        return raw, extras
    if isinstance(raw, str):
        try:
            return ErrorPolicy(raw), extras
        except ValueError as exc:
            valid = ", ".join(p.value for p in ErrorPolicy)
            raise ValueError(f"pipeline.on_error={raw!r} not in [{valid}]") from exc
    if isinstance(raw, Mapping):
        policy_raw = raw.get("policy", "abort")
        if not isinstance(policy_raw, str):
            raise ValueError(f"pipeline.on_error.policy must be a string, got {type(policy_raw).__name__}")
        try:
            policy = ErrorPolicy(policy_raw)
        except ValueError as exc:
            valid = ", ".join(p.value for p in ErrorPolicy)
            raise ValueError(f"pipeline.on_error={policy_raw!r} not in [{valid}]") from exc
        if "max_retries" in raw:
            mr = raw["max_retries"]
            if not isinstance(mr, int) or isinstance(mr, bool) or mr < 0:
                raise ValueError(f"pipeline.on_error.max_retries must be a non-negative int, got {mr!r}")
            extras["on_error_max_retries"] = mr
        return policy, extras
    raise ValueError(f"pipeline.on_error must be a string or mapping, got {type(raw).__name__}")


def _parse_on_cancel(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"pipeline.on_cancel must be a mapping, got {type(raw).__name__}")
    extras: dict[str, Any] = {}
    if "flush_store" in raw:
        flush = raw["flush_store"]
        if not isinstance(flush, bool):
            raise ValueError(f"pipeline.on_cancel.flush_store must be a bool, got {type(flush).__name__}")
        extras["on_cancel_flush_store"] = flush
    return extras


# ---------------------------------------------------------------------------
# placeholder interpolation
# ---------------------------------------------------------------------------


def _interpolate(value: Any, *, context: Mapping[str, Any], where: str) -> Any:
    if isinstance(value, str):
        return _interpolate_str(value, context=context, where=where)
    if isinstance(value, Mapping):
        return {k: _interpolate(v, context=context, where=f"{where}.{k}") for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate(v, context=context, where=f"{where}[{i}]") for i, v in enumerate(value)]
    return value


def _interpolate_str(value: str, *, context: Mapping[str, Any], where: str) -> Any:
    full = _FULL_PLACEHOLDER_RE.match(value)
    if full is not None:
        return _resolve_placeholder(full.group("name"), full.group("default"), context=context, where=where)

    def _sub(match: re.Match[str]) -> str:
        resolved = _resolve_placeholder(match.group("name"), match.group("default"), context=context, where=where)
        return str(resolved) if resolved is not None else ""

    return _PLACEHOLDER_RE.sub(_sub, value)


def _resolve_placeholder(name: str, default_expr: str | None, *, context: Mapping[str, Any], where: str) -> Any:
    if name in context:
        return context[name]
    if default_expr is not None:
        try:
            return yaml.safe_load(default_expr)
        except yaml.YAMLError as exc:
            raise ValueError(f"{where}: invalid default expression for {{{{ {name} | default({default_expr}) }}}}") from exc
    raise ValueError(f"{where}: undefined placeholder {{{{ {name} }}}}")
