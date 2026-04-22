"""Admin router — privileged endpoints for operations dashboards.

All routes require a principal whose tier ``name`` contains ``"admin"``
(case-insensitive). Mounted at ``/api/admin/*``.

Endpoints (stable subset):

Tasks
    * ``GET  /api/admin/tasks`` — list tasks across all courses
    * ``GET  /api/admin/tasks/{task_id}``
    * ``POST /api/admin/tasks/{task_id}/cancel``

Users
    * ``GET  /api/admin/users`` — list API keys → principals
    * ``POST /api/admin/users`` — add / upsert an API key (dev memory only)
    * ``DELETE /api/admin/users/{api_key}``

Engines / Workers
    * ``GET /api/admin/engines`` — configured LLM engines
    * ``GET /api/admin/workers`` — task backend + worker queue metadata

Workspace
    * ``GET /api/admin/workspace/{course}`` — list videos / stored artefacts
    * ``GET /api/admin/workspace/{course}/{video}`` — per-video state snapshot

Errors & audit
    * ``GET /api/admin/errors`` — last N structured errors emitted by the
      default ``ErrorReporter`` chain (when JSONL reporter is wired)

Terms (static context)
    * ``GET /api/admin/terms/{src}/{tgt}``
    * ``PUT /api/admin/terms/{src}/{tgt}``

Config
    * ``GET /api/admin/config`` — dumps redacted :class:`AppConfig`
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from api.service.auth import Principal, RequirePrincipal

if TYPE_CHECKING:
    from api.app.app import App
    from api.service.tasks import Task, TaskManager


router = APIRouter(prefix="/api/admin", tags=["admin"])


def _require_admin(principal: Principal) -> None:
    if "admin" not in principal.tier.name.lower():
        raise HTTPException(status_code=403, detail="admin required")


def _app(request: Request) -> "App":
    return request.app.state.app


def _tm(request: Request) -> "TaskManager":
    return request.app.state.tasks


def _task_dict(task: "Task") -> dict:
    return {
        "task_id": task.task_id,
        "course": task.course,
        "video": task.video,
        "src": task.src,
        "tgt": task.tgt,
        "stages": task.stages,
        "status": task.status,
        "done": task.done,
        "total": task.total,
        "error": task.error,
        "elapsed_s": task.elapsed_s,
    }


# -- Tasks -------------------------------------------------------------


@router.get("/tasks")
async def admin_list_tasks(request: Request, principal: Principal = RequirePrincipal) -> dict:
    _require_admin(principal)
    tm = _tm(request)
    tasks = list(getattr(tm, "_tasks", {}).values()) + list(getattr(tm, "_local_tasks", {}).values())
    seen: set[str] = set()
    out: list[dict] = []
    for t in tasks:
        if t.task_id in seen:
            continue
        seen.add(t.task_id)
        out.append(_task_dict(t))
    return {"tasks": out, "count": len(out)}


@router.get("/tasks/{task_id}")
async def admin_get_task(task_id: str, request: Request, principal: Principal = RequirePrincipal) -> dict:
    _require_admin(principal)
    tm = _tm(request)
    task = tm.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="task not found")
    return _task_dict(task)


@router.post("/tasks/{task_id}/cancel")
async def admin_cancel_task(task_id: str, request: Request, principal: Principal = RequirePrincipal) -> dict:
    _require_admin(principal)
    tm = _tm(request)
    ok = await tm.cancel(task_id)
    if not ok:
        raise HTTPException(status_code=404, detail="task not found")
    return {"ok": True, "task_id": task_id}


# -- Users -------------------------------------------------------------


class UserUpsert(BaseModel):
    api_key: str
    user_id: str
    tier: str = "free"


@router.get("/users")
async def admin_list_users(request: Request, principal: Principal = RequirePrincipal) -> dict:
    _require_admin(principal)
    auth_map: dict[str, Principal] = getattr(request.app.state, "auth_map", {}) or {}
    return {
        "users": [{"api_key_prefix": k[:6] + "...", "user_id": p.user_id, "tier": p.tier.name} for k, p in auth_map.items()],
        "count": len(auth_map),
    }


@router.post("/users", status_code=201)
async def admin_upsert_user(body: UserUpsert, request: Request, principal: Principal = RequirePrincipal) -> dict:
    _require_admin(principal)
    from application.resources import DEFAULT_TIERS

    tier = DEFAULT_TIERS.get(body.tier)
    if tier is None:
        raise HTTPException(status_code=400, detail=f"unknown tier {body.tier!r}")
    auth_map: dict[str, Principal] = getattr(request.app.state, "auth_map", None)
    if auth_map is None:
        request.app.state.auth_map = auth_map = {}
    auth_map[body.api_key] = Principal(user_id=body.user_id, tier=tier)
    return {"ok": True, "user_id": body.user_id, "tier": tier.name}


@router.delete("/users/{api_key}")
async def admin_delete_user(api_key: str, request: Request, principal: Principal = RequirePrincipal) -> dict:
    _require_admin(principal)
    auth_map: dict[str, Principal] = getattr(request.app.state, "auth_map", {}) or {}
    removed = auth_map.pop(api_key, None)
    return {"ok": removed is not None}


# -- Engines / Workers -------------------------------------------------


@router.get("/engines")
async def admin_list_engines(request: Request, principal: Principal = RequirePrincipal) -> dict:
    _require_admin(principal)
    app = _app(request)
    out: list[dict] = []
    for name, entry in app.config.engines.items():
        out.append(
            {
                "name": name,
                "model": entry.model,
                "base_url": entry.base_url,
                "api_key_set": bool(entry.api_key),
                "max_retries": getattr(entry, "max_retries", None),
                "timeout_s": getattr(entry, "timeout_s", None),
            }
        )
    return {"engines": out, "count": len(out)}


@router.get("/workers")
async def admin_list_workers(request: Request, principal: Principal = RequirePrincipal) -> dict:
    _require_admin(principal)
    app = _app(request)
    svc = app.config.service
    tm = _tm(request)
    backend = "arq" if type(tm).__name__ == "ArqTaskManager" else "inproc"
    info: dict[str, Any] = {"backend": backend}
    if backend == "arq":
        info.update(
            {
                "queue_name": svc.arq_queue_name,
                "task_prefix": svc.arq_task_prefix,
                "events_prefix": svc.arq_events_prefix,
                "redis_url": svc.redis_url,
            }
        )
    return info


# -- Workspace --------------------------------------------------------


@router.get("/workspace/{course}")
async def admin_workspace(course: str, request: Request, principal: Principal = RequirePrincipal) -> dict:
    _require_admin(principal)
    app = _app(request)
    ws = app.workspace(course)
    translation_dir = ws.translation.path
    videos: list[dict] = []
    if translation_dir.exists():
        for f in sorted(translation_dir.glob("*.json")):
            st = f.stat()
            videos.append({"video": f.stem, "size": st.st_size, "mtime": st.st_mtime})
    return {"course": course, "translation_dir": str(translation_dir), "videos": videos, "count": len(videos)}


@router.get("/workspace/{course}/{video}")
async def admin_workspace_video(course: str, video: str, request: Request, principal: Principal = RequirePrincipal) -> dict:
    _require_admin(principal)
    app = _app(request)
    ws = app.workspace(course)
    out: dict[str, Any] = {"course": course, "video": video, "artefacts": {}}
    translation = ws.translation.path_for(video, suffix=".json")
    subtitle = ws.subtitle.path_for(video, suffix=".json") if hasattr(ws, "subtitle") else None
    audio = ws.audio.path_for(video, suffix=".wav") if hasattr(ws, "audio") else None
    for key, path in [("translation", translation), ("subtitle", subtitle), ("audio", audio)]:
        if path is None:
            continue
        if path.exists():
            st = path.stat()
            out["artefacts"][key] = {"path": str(path), "size": st.st_size, "mtime": st.st_mtime}
    return out


# -- Errors ----------------------------------------------------------


@router.get("/errors")
async def admin_errors(request: Request, limit: int = 100, principal: Principal = RequirePrincipal) -> dict:
    _require_admin(principal)
    buf = getattr(request.app.state, "error_buffer", None)
    if buf is None:
        return {"errors": [], "count": 0}
    if hasattr(buf, "snapshot"):
        items = buf.snapshot(limit)
    else:
        items = list(buf)[-limit:]
    return {"errors": items, "count": len(items)}


# -- Terms -----------------------------------------------------------


class TermsUpdate(BaseModel):
    terms: dict[str, str] = Field(default_factory=dict)


@router.get("/terms/{src}/{tgt}")
async def admin_get_terms(src: str, tgt: str, request: Request, principal: Principal = RequirePrincipal) -> dict:
    _require_admin(principal)
    app = _app(request)
    entry = app.config.contexts.get(f"{src}-{tgt}") or app.config.contexts.get(f"{src}_{tgt}")
    terms = getattr(entry, "terms", {}) if entry else {}
    return {"src": src, "tgt": tgt, "terms": terms}


@router.put("/terms/{src}/{tgt}")
async def admin_put_terms(src: str, tgt: str, body: TermsUpdate, request: Request, principal: Principal = RequirePrincipal) -> dict:
    _require_admin(principal)
    app = _app(request)
    key = f"{src}-{tgt}"
    entry = app.config.contexts.get(key)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"no context for {key}")
    # Mutate in memory. Not persisted — callers should also edit their YAML.
    try:
        object.__setattr__(entry, "terms", dict(body.terms))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"ok": True, "terms": body.terms}


# -- Config ----------------------------------------------------------


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: ("***" if "key" in k.lower() or "secret" in k.lower() or "token" in k.lower() else _redact(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(v) for v in value]
    return value


@router.get("/config")
async def admin_get_config(request: Request, principal: Principal = RequirePrincipal) -> dict:
    _require_admin(principal)
    app = _app(request)
    data = app.config.model_dump()
    return _redact(data)


@router.post("/reload")
async def admin_reload(request: Request, principal: Principal = RequirePrincipal) -> dict:
    """Hot-reload the subset of config safe to swap at runtime.

    Re-parses the YAML referenced by ``service.reload_config_path`` and
    updates:

    * ``service.api_keys`` → ``app.state.auth_map``
    * ``service.cors_origins`` → recorded on ``app.state.app.config``
      (note: CORS middleware is installed at startup, so this only
      affects future re-initialisations)

    Anything else requires a full restart.
    """
    _require_admin(principal)
    app = _app(request)
    svc = app.config.service
    if not svc.reload_enabled:
        raise HTTPException(status_code=409, detail="reload disabled (service.reload_enabled=false)")
    path = svc.reload_config_path
    if not path:
        raise HTTPException(status_code=409, detail="service.reload_config_path not set")

    from application.config import AppConfig
    from application.resources import DEFAULT_TIERS

    try:
        new_cfg = AppConfig.load(path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"reload failed: {exc}")

    new_svc = new_cfg.service
    tier_map = getattr(request.app.state, "tier_map", DEFAULT_TIERS)

    new_auth: dict = {}
    for key, entry in new_svc.api_keys.items():
        tier = tier_map.get(entry.tier)
        if tier is None:
            raise HTTPException(status_code=400, detail=f"unknown tier {entry.tier!r}")
        new_auth[key] = Principal(user_id=entry.user_id, tier=tier)

    request.app.state.auth_map = new_auth
    # Overwrite the in-memory ServiceConfig so future introspection sees new values.
    try:
        object.__setattr__(app.config, "service", new_svc)
    except Exception:
        pass
    return {"ok": True, "api_keys": len(new_auth), "cors_origins": new_svc.cors_origins}


__all__ = ["router"]
