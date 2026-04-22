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


def _resolve_source(app: "App", course: str, req: CreateVideoRequest) -> tuple[Path, str | None]:
    """Resolve the request body to a source path on disk + kind hint."""
    if req.source_path:
        p = Path(req.source_path)
        if not p.is_absolute():
            p = Path(app.config.store.root).expanduser() / p
        if not p.exists():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"source_path not found: {p}",
            )
        return p, req.source_kind

    if req.source_content:
        # Persist inline SRT into the course's subtitle subdir so later
        # result lookups by video stem still work.
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


@router.post("", status_code=status.HTTP_202_ACCEPTED, response_model=VideoState)
async def create_video_task(
    course: str,
    body: CreateVideoRequest,
    request: Request,
    principal: Principal = RequirePrincipal,
) -> VideoState:
    manager = _manager(request)
    source_path, source_kind = _resolve_source(manager.app, course, body)
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
    _p: Principal = RequirePrincipal,
) -> VideoList:
    manager = _manager(request)
    items = [_as_state(t) for t in manager.list_for_course(course)]
    return VideoList(items=items)


@router.get("/{task_id}", response_model=VideoState)
async def get_video_task(
    course: str,
    task_id: str,
    request: Request,
    _p: Principal = RequirePrincipal,
) -> VideoState:
    task = _manager(request).get(task_id)
    if task is None or task.course != course:
        raise HTTPException(status_code=404, detail="task not found")
    return _as_state(task)


@router.post("/{task_id}/cancel", response_model=VideoState)
async def cancel_video_task(
    course: str,
    task_id: str,
    request: Request,
    _p: Principal = RequirePrincipal,
) -> VideoState:
    manager = _manager(request)
    task = manager.get(task_id)
    if task is None or task.course != course:
        raise HTTPException(status_code=404, detail="task not found")
    await manager.cancel(task_id)
    # Give the runner a tick to observe the cancel.
    await asyncio.sleep(0)
    return _as_state(task)


@router.get("/{task_id}/events")
async def stream_video_events(
    course: str,
    task_id: str,
    request: Request,
    _p: Principal = RequirePrincipal,
):
    task = _manager(request).get(task_id)
    if task is None or task.course != course:
        raise HTTPException(status_code=404, detail="task not found")
    return EventSourceResponse(task_event_stream(task))


@router.get("/{video}/result")
async def get_video_result(
    course: str,
    video: str,
    request: Request,
    format: str = "json",
    _p: Principal = RequirePrincipal,
):
    manager = _manager(request)
    try:
        content, media_type = load_result_bytes(manager.app, course, video, format)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="video result not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return Response(content=content, media_type=media_type)


__all__ = ["router"]
