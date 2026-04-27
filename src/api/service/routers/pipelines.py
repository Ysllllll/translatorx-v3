"""Pipelines router — read + validate the pipeline catalog (Phase 2 / B5).

Endpoints
    * ``GET  /api/pipelines`` — list named pipelines available on the App
    * ``GET  /api/pipelines/{name}`` — return the raw pipeline dict
    * ``POST /api/pipelines/validate`` — validate a YAML body against the
      registry and return either ``{"ok": true}`` or a list of issues

Tenant scoping (Phase 2 / Step B4):
    Each principal carries an optional ``tenant``. By default a caller
    sees their own tenant (plus globally-scoped pipelines). Admin
    principals (whose tier name contains ``"admin"``) may override the
    tenant via ``?tenant=`` and also use ``?tenant=*`` to view every
    tenant's pipelines.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, HTTPException, Query, Request, status
from pydantic import BaseModel, ConfigDict, Field
import yaml as _yaml

from api.service.auth import Principal, RequirePrincipal
from application.pipeline.loader import parse_pipeline_yaml
from application.pipeline.validator import (
    PipelineValidationError,
    validate_pipeline,
)

router = APIRouter(prefix="/api/pipelines", tags=["pipelines"])


# -- response models ----------------------------------------------------


class PipelineListResponse(BaseModel):
    """Response for ``GET /api/pipelines``."""

    model_config = ConfigDict(json_schema_extra={"example": {"pipelines": ["standard_translate"], "tenant": None}})

    pipelines: list[str] = Field(description="Sorted catalog keys visible to the caller.")
    tenant: str | None = Field(description='Effective tenant scope for this listing. ``"*"`` when admin lists everything.')


class PipelineGetResponse(BaseModel):
    """Response for ``GET /api/pipelines/{name}``."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "standard_translate",
                "definition": {
                    "name": "standard_translate",
                    "build": {"stage": "from_srt", "params": {"path": "in.srt", "language": "en"}},
                    "structure": [{"stage": "merge", "params": {"max_len": 80}}],
                    "enrich": [{"stage": "translate", "params": {"src": "en", "tgt": "zh"}}],
                },
            }
        }
    )

    name: str
    definition: dict[str, Any]


class ValidationIssue(BaseModel):
    path: str = Field(description="Dotted JSON path of the offending field.")
    message: str = Field(description="Human-readable explanation.")


class ValidateResponse(BaseModel):
    """Response for ``POST /api/pipelines/validate``."""

    model_config = ConfigDict(
        json_schema_extra={"example": {"ok": False, "issues": [{"path": "build.params", "message": "missing key 'path'"}]}}
    )

    ok: bool
    issues: list[ValidationIssue]


class _ValidateBody(BaseModel):
    """Either ``{"yaml": "..."}`` or a literal pipeline dict.

    Documented for the OpenAPI surface; the actual handler accepts a
    raw ``dict`` for flexibility.
    """

    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={"example": {"yaml": "name: p\nbuild:\n  stage: from_srt\n  params: {path: x.srt, language: en}\n"}},
    )

    yaml: str | None = Field(default=None, description="Optional raw YAML body. When set, all other keys are ignored.")


# -- helpers ------------------------------------------------------------


def _app(request: Request):
    app = getattr(request.app.state, "app", None)
    if app is None:
        raise RuntimeError("App not initialized on app.state.app")
    return app


def _is_admin(principal: Principal) -> bool:
    return "admin" in principal.tier.name.lower()


def _resolve_scope(principal: Principal, query_tenant: str | None) -> tuple[str | None, bool]:
    """Resolve which catalog slice the caller should see.

    Returns ``(tenant, include_all)`` ready to pass to
    :meth:`api.app.App.pipelines`. Non-admin callers may not override
    their own tenant — any explicit ``?tenant=`` is rejected.
    """
    if query_tenant is None:
        return principal.tenant, False
    if not _is_admin(principal):
        if query_tenant != principal.tenant:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="not allowed to inspect another tenant",
            )
        return principal.tenant, False
    if query_tenant == "*":
        return None, True
    return query_tenant, False


# -- routes -------------------------------------------------------------


@router.get("", response_model=PipelineListResponse)
async def list_pipelines(
    request: Request,
    tenant: str | None = Query(None, description="Override tenant scope (admin only). Use '*' for all."),
    principal: Principal = RequirePrincipal,
) -> dict[str, Any]:
    """List named pipelines configured on the App, scoped by tenant."""
    scope_tenant, include_all = _resolve_scope(principal, tenant)
    catalog = _app(request).pipelines(scope_tenant, include_all=include_all)
    return {"pipelines": sorted(catalog.keys()), "tenant": "*" if include_all else scope_tenant}


@router.get("/{name}", response_model=PipelineGetResponse)
async def get_pipeline(
    name: str,
    request: Request,
    tenant: str | None = Query(None, description="Override tenant scope (admin only)."),
    principal: Principal = RequirePrincipal,
) -> dict[str, Any]:
    """Return the raw pipeline dict for ``name`` within the caller's scope."""
    scope_tenant, include_all = _resolve_scope(principal, tenant)
    catalog = _app(request).pipelines(scope_tenant, include_all=include_all)
    if name not in catalog:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"pipeline {name!r} not found",
        )
    return {"name": name, "definition": catalog[name]}


@router.post("/validate", response_model=ValidateResponse)
async def validate_pipeline_body(
    request: Request,
    payload: dict[str, Any] = Body(..., embed=False),
    principal: Principal = RequirePrincipal,
) -> dict[str, Any]:
    """Validate a pipeline body against the App's stage registry."""
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


__all__ = ["router", "PipelineValidationError", "PipelineListResponse", "PipelineGetResponse", "ValidateResponse"]
