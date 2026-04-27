"""Videos router — task CRUD + SSE progress + result download."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException, Request, Response, status
from sse_starlette.sse import EventSourceResponse

from api.service.auth import Principal, RequirePrincipal
from api.service.schemas import CreateVideoRequest, VideoList, VideoState
from api.service.sse import task_event_stream
from api.service.runtime.tasks import Task, TaskManager, load_result_bytes

if TYPE_CHECKING:
    from api.app.app import App


router = APIRouter(prefix="/api/courses/{course}/videos", tags=["videos"])


def _manager(request: Request) -> TaskManager:
    manager: TaskManager | None = getattr(request.app.state, "tasks", None)
    if manager is None:
        raise RuntimeError("TaskManager not initialized on app.state.tasks")
    return manager


def _as_state(task: Task) -> VideoState:
    return VideoState(
        task_id=task.task_id,
        course=task.course,
        video=task.video,
        status=task.status,
        stages=task.stages,
        src=task.src,
        tgt=task.tgt,
        done=task.done,
        total=task.total,
        error=task.error,
        elapsed_s=task.elapsed_s,
    )


def _resolve_source(app: "App", course: str, req: CreateVideoRequest, *, auth_enabled: bool) -> tuple[Path, str | None]:
    """Resolve the request body to a source path on disk + kind hint.

    R2 — path-traversal hardening: when auth is enabled (production
    mode, ``app.state.auth_map`` non-empty) the resolved path **must**
    sit under ``app.config.store.root``. In anonymous dev mode the
    legacy "any path the OS user can read" behaviour is preserved so
    local CLIs continue to work, but ``..`` segments and traversal
    attempts are still rejected.
    """
    if req.source_path:
        candidate = Path(req.source_path)
        store_root = Path(app.config.store.root).expanduser().resolve()
        if candidate.is_absolute():
            try:
                resolved = candidate.resolve()
            except OSError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="source_path could not be resolved",
                )
            if auth_enabled:
                try:
                    resolved.relative_to(store_root)
                except ValueError:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="source_path must be under the store root",
                    )
        else:
            try:
                resolved = (store_root / candidate).resolve()
                resolved.relative_to(store_root)
            except (ValueError, OSError):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="source_path escapes the store root",
                )
        if not resolved.exists():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="source_path not found",
            )
        return resolved, req.source_kind

    if req.source_content:
        ws = app.store(course).workspace
        sub = ws.get_subdir("subtitle")
        target = sub.path_for(req.video, suffix=".srt")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(req.source_content, encoding="utf-8")
        return target, "srt"

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="one of source_path / source_content must be provided",
    )


def _assert_can_access(task: "Task", principal: Principal) -> None:
    """R1 — per-resource authz: only the submitting principal may
    inspect / cancel / stream a task. Anonymous (dev) mode keeps the
    legacy "anyone can see anything" behaviour.
    """
    if task.principal_user_id is None:
        return
    if task.principal_user_id == principal.user_id:
        return
    # Surface as 404 (not 403) so probe-callers can't enumerate the
    # task ID space across tenants.
    raise HTTPException(status_code=404, detail="task not found")


@router.post("", status_code=status.HTTP_202_ACCEPTED, response_model=VideoState)
async def create_video_task(
    course: str,
    body: CreateVideoRequest,
    request: Request,
    principal: Principal = RequirePrincipal,
) -> VideoState:
    manager = _manager(request)
    auth_enabled = bool(getattr(request.app.state, "auth_map", {}) or {})
    source_path, source_kind = _resolve_source(manager.app, course, body, auth_enabled=auth_enabled)
    task = manager.submit(
        course=course,
        video=body.video,
        src=body.src,
        tgt=body.tgt,
        stages=body.stages,
        source_path=source_path,
        source_kind=source_kind,
        engine_name=body.engine,
        principal=principal,
    )
    return _as_state(task)


@router.get("", response_model=VideoList)
async def list_video_tasks(
    course: str,
    request: Request,
    principal: Principal = RequirePrincipal,
) -> VideoList:
    """List tasks for the course.

    R1 — when authentication is enabled the listing is filtered to the
    calling principal's own tasks. Anonymous dev mode (empty
    ``auth_map``) preserves the legacy "everyone sees everything"
    behaviour so existing tooling keeps working.
    """
    manager = _manager(request)
    auth_enabled = bool(getattr(request.app.state, "auth_map", {}) or {})
    tasks = list(manager.list_for_course(course))
    if auth_enabled:
        tasks = [t for t in tasks if t.principal_user_id is None or t.principal_user_id == principal.user_id]
    items = [_as_state(t) for t in tasks]
    return VideoList(items=items)


@router.get("/{task_id}", response_model=VideoState)
async def get_video_task(
    course: str,
    task_id: str,
    request: Request,
    principal: Principal = RequirePrincipal,
) -> VideoState:
    task = _manager(request).get(task_id)
    if task is None or task.course != course:
        raise HTTPException(status_code=404, detail="task not found")
    _assert_can_access(task, principal)
    return _as_state(task)


@router.post("/{task_id}/cancel", response_model=VideoState)
async def cancel_video_task(
    course: str,
    task_id: str,
    request: Request,
    principal: Principal = RequirePrincipal,
) -> VideoState:
    manager = _manager(request)
    task = manager.get(task_id)
    if task is None or task.course != course:
        raise HTTPException(status_code=404, detail="task not found")
    _assert_can_access(task, principal)
    await manager.cancel(task_id)
    # Give the runner a tick to observe the cancel.
    await asyncio.sleep(0)
    return _as_state(task)


@router.get("/{task_id}/events")
async def stream_video_events(
    course: str,
    task_id: str,
    request: Request,
    principal: Principal = RequirePrincipal,
):
    task = _manager(request).get(task_id)
    if task is None or task.course != course:
        raise HTTPException(status_code=404, detail="task not found")
    _assert_can_access(task, principal)
    return EventSourceResponse(task_event_stream(task))


@router.get("/{video}/result")
async def get_video_result(
    course: str,
    video: str,
    request: Request,
    format: str = "json",
    principal: Principal = RequirePrincipal,
):
    manager = _manager(request)
    # Per-video result has no task_id, so authorize by checking that
    # any in-memory task for this (course, video) belongs to the caller.
    # If no task is found we fall through to the on-disk lookup; this is
    # intentional because the file may have been persisted by a previous
    # process the principal authored — strict cross-process ownership
    # tracking is Phase 5 work.
    for t in manager.list_for_course(course):
        if t.video == video:
            _assert_can_access(t, principal)
            break
    try:
        content, media_type = load_result_bytes(manager.app, course, video, format)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="video result not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return Response(content=content, media_type=media_type)


__all__ = ["router"]
