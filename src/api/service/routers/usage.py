"""Usage router — /api/usage/{user_id}, /api/usage/summary, /api/usage/top."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from api.service.auth import RequirePrincipal, Principal

router = APIRouter(prefix="/api/usage", tags=["usage"])


def _snapshot_dict(snap) -> dict:
    return {
        "user_id": snap.user_id,
        "period_start": snap.period_start.isoformat(),
        "cost_usd": snap.cost_usd,
        "prompt_tokens": snap.prompt_tokens,
        "completion_tokens": snap.completion_tokens,
        "requests": snap.requests,
        "by_model": snap.by_model,
    }


@router.get("/summary")
async def get_usage_summary(request: Request, principal: Principal = RequirePrincipal) -> dict:
    """Return aggregated cost / token totals across all users today (admin only)."""
    if "admin" not in principal.tier.name.lower():
        raise HTTPException(status_code=403, detail="admin required")
    rm = request.app.state.rm
    if not hasattr(rm, "list_daily_ledgers"):
        raise HTTPException(status_code=501, detail="resource_manager does not support summary")
    snaps = await rm.list_daily_ledgers(limit=10_000)
    total_cost = sum(s.cost_usd for s in snaps)
    total_prompt = sum(s.prompt_tokens for s in snaps)
    total_completion = sum(s.completion_tokens for s in snaps)
    total_requests = sum(s.requests for s in snaps)
    by_model: dict[str, float] = {}
    for s in snaps:
        for m, c in s.by_model.items():
            by_model[m] = by_model.get(m, 0.0) + c
    return {
        "users": len(snaps),
        "cost_usd": total_cost,
        "prompt_tokens": total_prompt,
        "completion_tokens": total_completion,
        "requests": total_requests,
        "by_model": by_model,
    }


@router.get("/top")
async def get_usage_top(
    request: Request,
    limit: int = 20,
    principal: Principal = RequirePrincipal,
) -> list[dict]:
    """Return the top-``limit`` users by cost today (admin only)."""
    if "admin" not in principal.tier.name.lower():
        raise HTTPException(status_code=403, detail="admin required")
    rm = request.app.state.rm
    if not hasattr(rm, "list_daily_ledgers"):
        raise HTTPException(status_code=501, detail="resource_manager does not support top")
    snaps = await rm.list_daily_ledgers(limit=limit)
    return [_snapshot_dict(s) for s in snaps]


@router.get("/{user_id}")
async def get_user_usage(user_id: str, request: Request, principal: Principal = RequirePrincipal) -> dict:
    """Return today's ledger for ``user_id``.

    Users can query their own ledger; any other user_id requires a principal
    whose tier name contains ``admin``.
    """
    if user_id != principal.user_id and "admin" not in principal.tier.name.lower():
        raise HTTPException(status_code=403, detail="forbidden")
    rm = request.app.state.rm
    snap = await rm.get_daily_ledger(user_id)
    return _snapshot_dict(snap)


__all__ = ["router"]
