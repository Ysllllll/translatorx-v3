"""Health router — /health + /ready."""

from __future__ import annotations

from fastapi import APIRouter, Request

from api.service.schemas import Health


router = APIRouter(tags=["health"])


@router.get("/health", response_model=Health)
async def health() -> Health:
    return Health()


@router.get("/ready", response_model=Health)
async def ready(request: Request) -> Health:
    # Minimal readiness: app + task manager must be attached.
    assert getattr(request.app.state, "app", None) is not None
    assert getattr(request.app.state, "tasks", None) is not None
    return Health()


__all__ = ["router"]
