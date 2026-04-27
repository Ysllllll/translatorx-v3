"""JSON Schema export for pipeline configs and stage params.

Phase 2 (Step B2). Three entry points:

* :func:`pipeline_json_schema` — schema for the YAML/dict shape consumed
  by :mod:`application.pipeline.loader`. Stage-name / params validation
  is *not* baked in here (it happens at runtime via the registry); the
  schema describes the structural envelope only.

* :func:`stage_params_schema` — JSON Schema for one stage's Pydantic
  ``Params`` model, derived via Pydantic's ``model_json_schema()``.

* :func:`registry_json_schema` — full pipeline schema, augmented with
  per-stage ``oneOf`` discriminator on ``stage`` so a frontend editor
  (e.g. react-jsonschema-form) can dispatch the right ``params``
  sub-schema after a stage name is picked.

The schemas target JSON Schema draft 2020-12 (Pydantic v2 default).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ports.pipeline import ErrorPolicy

if TYPE_CHECKING:
    from .registry import StageRegistry

__all__ = [
    "pipeline_json_schema",
    "stage_params_schema",
    "registry_json_schema",
]


_STAGE_REF_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [],
    "properties": {
        "stage": {
            "type": "string",
            "description": "Registry name of the stage to instantiate.",
        },
        "name": {
            "type": "string",
            "description": "Alias for 'stage'. Either key is accepted.",
        },
        "id": {
            "type": "string",
            "description": "Optional unique id within the pipeline.",
        },
        "when": {
            "type": ["string", "null"],
            "description": "Optional Jinja-style condition (Phase 2 runtime).",
        },
        "params": {
            "type": "object",
            "description": "Stage-specific parameters. Validated against the stage's Params model.",
            "additionalProperties": True,
        },
    },
    "anyOf": [
        {"required": ["stage"]},
        {"required": ["name"]},
    ],
    "additionalProperties": False,
}


def _on_error_schema() -> dict[str, Any]:
    policies = [p.value for p in ErrorPolicy]
    return {
        "oneOf": [
            {"type": "string", "enum": policies},
            {
                "type": "object",
                "required": ["policy"],
                "properties": {
                    "policy": {"type": "string", "enum": policies},
                    "max_retries": {"type": "integer", "minimum": 0},
                },
                "additionalProperties": False,
            },
        ],
        "default": "abort",
    }


def _on_cancel_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "flush_store": {"type": "boolean"},
        },
        "additionalProperties": False,
    }


def pipeline_json_schema() -> dict[str, Any]:
    """JSON Schema for the YAML/dict pipeline envelope.

    The schema validates the *shape* loaded by
    :func:`application.pipeline.loader.load_pipeline_dict`. It does not
    constrain stage names or params — combine with
    :func:`registry_json_schema` for that.
    """
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "PipelineDef",
        "type": "object",
        "required": ["build"],
        "properties": {
            "name": {"type": "string", "default": "pipeline"},
            "version": {"type": "integer", "minimum": 1, "default": 1},
            "tenant": {
                "type": ["string", "null"],
                "description": "Optional tenant scope. Pipelines without a tenant are global.",
                "default": None,
            },
            "defaults": {
                "type": "object",
                "description": "Variable defaults for {{ placeholder }} substitution.",
                "additionalProperties": True,
            },
            "build": _STAGE_REF_SCHEMA,
            "structure": {
                "type": "array",
                "items": _STAGE_REF_SCHEMA,
                "default": [],
            },
            "enrich": {
                "type": "array",
                "items": _STAGE_REF_SCHEMA,
                "default": [],
            },
            "on_error": _on_error_schema(),
            "on_cancel": _on_cancel_schema(),
            "metadata": {
                "type": "object",
                "additionalProperties": True,
                "default": {},
            },
        },
        "additionalProperties": False,
    }


def stage_params_schema(registry: "StageRegistry", stage_name: str) -> dict[str, Any]:
    """Return the JSON Schema for one stage's ``Params`` model.

    Stages with no ``params_schema`` get a permissive
    ``{"type": "object", "additionalProperties": true}`` schema.
    """
    if not registry.is_registered(stage_name):
        raise KeyError(f"Stage {stage_name!r} is not registered")
    schema_cls = registry.schema_of(stage_name)
    if schema_cls is None:
        return {
            "type": "object",
            "title": f"{stage_name} params",
            "description": "No typed schema declared; arbitrary params accepted.",
            "additionalProperties": True,
        }
    if hasattr(schema_cls, "model_json_schema"):
        return schema_cls.model_json_schema()
    raise TypeError(f"Stage {stage_name!r} params_schema {schema_cls!r} is not a Pydantic v2 model")


def registry_json_schema(registry: "StageRegistry") -> dict[str, Any]:
    """Pipeline JSON Schema augmented with per-stage params dispatch.

    The returned schema mirrors :func:`pipeline_json_schema` but each
    stage reference uses a ``oneOf`` discriminator on ``stage`` so a
    frontend can resolve the matching ``params`` sub-schema after a
    name is chosen.
    """
    base = pipeline_json_schema()

    stage_variants = []
    for name in registry.names():
        stage_variants.append(
            {
                "type": "object",
                "title": name,
                "required": [],
                "properties": {
                    "stage": {"const": name},
                    "name": {"const": name},
                    "id": {"type": "string"},
                    "when": {"type": ["string", "null"]},
                    "params": stage_params_schema(registry, name),
                },
                "anyOf": [
                    {"required": ["stage"]},
                    {"required": ["name"]},
                ],
                "additionalProperties": False,
            }
        )

    if stage_variants:
        stage_ref: dict[str, Any] = {"oneOf": stage_variants}
        base["properties"]["build"] = stage_ref
        base["properties"]["structure"]["items"] = stage_ref
        base["properties"]["enrich"]["items"] = stage_ref

    base["title"] = "PipelineDef (registry-bound)"
    return base
