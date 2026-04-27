"""Stages router — discover registered stages + JSON Schema (Phase 2 / B5).

Endpoints
    * ``GET /api/stages`` — list registered stage names
    * ``GET /api/stages/schema`` — full registry-bound pipeline JSON Schema
    * ``GET /api/stages/{name}/schema`` — params JSON Schema for one stage

The schemas follow JSON Schema draft 2020-12 (Pydantic v2 default) and
are designed to plug straight into editor frontends like
``react-jsonschema-form``.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request, status

from api.service.auth import Principal, RequirePrincipal
from application.pipeline.schema import (
    registry_json_schema,
    stage_params_schema,
)

router = APIRouter(prefix="/api/stages", tags=["stages"])


def _registry(request: Request):
    app = getattr(request.app.state, "app", None)
    if app is None:
        raise RuntimeError("App not initialized on app.state.app")
    return app.registry()


@router.get("")
async def list_stages(request: Request, principal: Principal = RequirePrincipal) -> dict[str, Any]:
    """List registered stage names."""
    reg = _registry(request)
    return {"stages": list(reg.names())}


@router.get("/schema")
async def pipeline_schema(request: Request, principal: Principal = RequirePrincipal) -> dict[str, Any]:
    """Return the full pipeline JSON Schema bound to this registry."""
    return registry_json_schema(_registry(request))


@router.get("/{name}/schema")
async def stage_schema(name: str, request: Request, principal: Principal = RequirePrincipal) -> dict[str, Any]:
    """Return the JSON Schema for one stage's ``Params`` model."""
    reg = _registry(request)
    if not reg.is_registered(name):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"stage {name!r} not registered",
        )
    return stage_params_schema(reg, name)


__all__ = ["router"]
