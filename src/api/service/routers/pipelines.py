"""Pipelines router — read + validate the pipeline catalog (Phase 2 / B5).

Endpoints
    * ``GET  /api/pipelines`` — list named pipelines available on the App
    * ``GET  /api/pipelines/{name}`` — return the raw pipeline dict
    * ``POST /api/pipelines/validate`` — validate a YAML body against the
      registry and return either ``{"ok": true}`` or a list of issues

Triggering a pipeline run is **not** included in this MVP — the existing
``/api/courses/{course}/videos`` endpoint already covers that. A future
``/api/pipelines/{name}/run`` endpoint will land alongside the tenant
namespace work.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, HTTPException, Request, status
import yaml as _yaml

from api.service.auth import Principal, RequirePrincipal
from application.pipeline.loader import parse_pipeline_yaml
from application.pipeline.validator import (
    PipelineValidationError,
    validate_pipeline,
)

router = APIRouter(prefix="/api/pipelines", tags=["pipelines"])


def _app(request: Request):
    app = getattr(request.app.state, "app", None)
    if app is None:
        raise RuntimeError("App not initialized on app.state.app")
    return app


@router.get("")
async def list_pipelines(request: Request, principal: Principal = RequirePrincipal) -> dict[str, Any]:
    """List named pipelines configured on the App."""
    catalog = _app(request).pipelines()
    return {"pipelines": sorted(catalog.keys())}


@router.get("/{name}")
async def get_pipeline(name: str, request: Request, principal: Principal = RequirePrincipal) -> dict[str, Any]:
    """Return the raw pipeline dict for ``name``."""
    catalog = _app(request).pipelines()
    if name not in catalog:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"pipeline {name!r} not found",
        )
    return {"name": name, "definition": catalog[name]}


@router.post("/validate")
async def validate_pipeline_body(
    request: Request,
    payload: dict[str, Any] = Body(..., embed=False),
    principal: Principal = RequirePrincipal,
) -> dict[str, Any]:
    """Validate a pipeline body against the App's stage registry.

    The request body is either ``{"yaml": "..."}`` or a literal pipeline
    dict (same shape accepted by the loader). Returns a structured report
    instead of raising — easier for editor surfaces to consume.
    """
    if "yaml" in payload and len(payload) == 1:
        try:
            defn = parse_pipeline_yaml(str(payload["yaml"]))
        except (ValueError, _yaml.YAMLError) as exc:
            return {"ok": False, "issues": [{"path": "yaml", "message": str(exc)}]}
    else:
        from application.pipeline.loader import load_pipeline_dict

        try:
            defn = load_pipeline_dict(payload)
        except ValueError as exc:
            return {"ok": False, "issues": [{"path": "body", "message": str(exc)}]}

    registry = _app(request).registry()
    report = validate_pipeline(defn, registry, collect=True)
    if report.ok:
        return {"ok": True, "issues": []}
    return {
        "ok": False,
        "issues": [{"path": i.path, "message": i.message} for i in report.issues],
    }


__all__ = ["router", "PipelineValidationError"]
