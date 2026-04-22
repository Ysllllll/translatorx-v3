"""Health router — /health + /ready."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from api.service.schemas import Health


router = APIRouter(tags=["health"])


@router.get("/health", response_model=Health)
async def health() -> Health:
    return Health()


@router.get("/ready")
async def ready(request: Request):
    """Deep readiness check.

    Verifies:

    * ``app.state.app`` + ``app.state.tasks`` are attached
    * If configured, a Redis ping succeeds
    """
    checks: dict[str, Any] = {}
    ok = True

    if getattr(request.app.state, "app", None) is None:
        ok = False
        checks["app"] = "missing"
    else:
        checks["app"] = "ok"

    if getattr(request.app.state, "tasks", None) is None:
        ok = False
        checks["tasks"] = "missing"
    else:
        checks["tasks"] = "ok"

    svc_cfg = getattr(request.app.state.app.config, "service", None) if checks["app"] == "ok" else None
    if svc_cfg and (svc_cfg.resource_backend == "redis" or svc_cfg.task_backend == "arq"):
        url = svc_cfg.redis_url
        if url:
            try:
                import redis.asyncio as redis_async

                client = redis_async.from_url(url, decode_responses=True)
                try:
                    await client.ping()
                    checks["redis"] = "ok"
                finally:
                    try:
                        await client.aclose()
                    except Exception:
                        pass
            except Exception as exc:
                ok = False
                checks["redis"] = f"fail: {type(exc).__name__}"
        else:
            ok = False
            checks["redis"] = "missing url"

    status_code = 200 if ok else 503
    return JSONResponse(status_code=status_code, content={"status": "ok" if ok else "degraded", "checks": checks})


__all__ = ["router"]
