"""In-process task registry for video translation tasks.

Not a production queue — this is a dict of ``asyncio.Task`` objects
intended for single-worker deployments or development. Multi-worker
deployments should back this onto Redis / a job queue (future work;
not in Stage 7 scope).

Each :class:`Task` owns:

* an ``asyncio.Queue`` of SSE events emitted from the orchestrator's
  progress callback;
* a cancel flag mapped to ``asyncio.Task.cancel()``;
* a terminal ``result`` (``VideoResult`` on success, exception string
  on failure).

The registry **does not** persist across restarts — durable state lives
inside the per-video ``zzz_translation/<video>.json`` file managed by
:class:`JsonFileStore`, so a restarted worker can read partial progress
from disk.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from application.observability.progress import ProgressEvent

if TYPE_CHECKING:
    from api.app.app import App
    from api.service.auth import Principal
    from application.orchestrator.video import VideoResult
    from application.resources import ResourceManager


logger = logging.getLogger(__name__)


TaskStatus = str  # "queued" | "running" | "done" | "failed" | "cancelled"


@dataclass
class Task:
    """One video translation task managed by :class:`TaskManager`."""

    task_id: str
    course: str
    video: str
    src: str | None
    tgt: list[str]
    stages: list[str]
    status: TaskStatus = "queued"
    done: int = 0
    total: int | None = None
    error: str | None = None
    elapsed_s: float | None = None
    result: Any = None  # VideoResult
    _runner: asyncio.Task | None = None
    _queues: list[asyncio.Queue] = field(default_factory=list)
    _terminal: asyncio.Event = field(default_factory=asyncio.Event)

    # -- event distribution ---------------------------------------------

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=1024)
        self._queues.append(q)
        # On subscribe, send a snapshot so clients see current state.
        q.put_nowait(_snapshot_event(self))
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        try:
            self._queues.remove(q)
        except ValueError:
            pass

    def emit(self, event: dict[str, Any]) -> None:
        for q in list(self._queues):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning("task %s: subscriber queue full; dropping event", self.task_id)

    def mark_terminal(self) -> None:
        self.emit(_snapshot_event(self))
        self._terminal.set()
        # Sentinel so SSE readers can stop cleanly.
        for q in list(self._queues):
            try:
                q.put_nowait(None)
            except asyncio.QueueFull:
                pass


def _snapshot_event(task: Task) -> dict[str, Any]:
    return {
        "event": "status",
        "data": {
            "task_id": task.task_id,
            "status": task.status,
            "done": task.done,
            "total": task.total,
            "error": task.error,
            "elapsed_s": task.elapsed_s,
        },
    }


class TaskManager:
    """Registry + async runner for video tasks."""

    def __init__(self, app: "App", resource_mgr: "ResourceManager") -> None:
        self._app = app
        self._rm = resource_mgr
        self._tasks: dict[str, Task] = {}

    # -- accessors ------------------------------------------------------

    @property
    def app(self) -> "App":
        return self._app

    def get(self, task_id: str) -> Task | None:
        return self._tasks.get(task_id)

    def list_for_course(self, course: str) -> list[Task]:
        return [t for t in self._tasks.values() if t.course == course]

    async def cancel(self, task_id: str) -> bool:
        task = self._tasks.get(task_id)
        if task is None or task._runner is None:
            return False
        if task.status in ("done", "failed", "cancelled"):
            return True
        task._runner.cancel()
        return True

    async def shutdown(self) -> None:
        pending = [t._runner for t in self._tasks.values() if t._runner and not t._runner.done()]
        for r in pending:
            r.cancel()
        for r in pending:
            try:
                await r
            except BaseException:
                pass

    # -- submission -----------------------------------------------------

    def submit(
        self,
        *,
        course: str,
        video: str,
        src: str | None,
        tgt: list[str],
        stages: list[str],
        source_path: Path,
        source_kind: str | None,
        engine_name: str,
        principal: "Principal",
    ) -> Task:
        task_id = uuid.uuid4().hex
        task = Task(
            task_id=task_id,
            course=course,
            video=video,
            src=src,
            tgt=list(tgt),
            stages=list(stages),
            status="queued",
        )
        self._tasks[task_id] = task
        loop = asyncio.get_running_loop()
        task._runner = loop.create_task(self._run_task(task, source_path, source_kind, engine_name, principal))
        return task

    # -- runner ---------------------------------------------------------

    async def _run_task(
        self,
        task: Task,
        source_path: Path,
        source_kind: str | None,
        engine_name: str,
        principal: "Principal",
    ) -> None:
        from application.orchestrator.video import VideoResult  # local import

        app = self._app
        task.status = "running"
        task.emit(_snapshot_event(task))

        def on_progress(event: ProgressEvent) -> None:
            if event.kind == "record":
                task.done = event.done
            if event.total is not None:
                task.total = event.total
            task.emit(
                {
                    "event": event.kind,
                    "data": {
                        "processor": event.processor,
                        "done": event.done,
                        "total": event.total,
                        "record_id": event.record_id,
                        "cache_hit": event.cache_hit,
                    },
                }
            )

        try:
            async with self._rm.acquire_video_slot(principal.user_id, principal.tier):
                last: VideoResult | None = None
                for tgt_lang in task.tgt:
                    builder = app.video(course=task.course, video=task.video)
                    if "transcribe" in task.stages:
                        builder = builder.transcribe(audio=source_path, language=task.src)
                    else:
                        builder = builder.source(source_path, language=task.src, kind=source_kind)
                    if "summary" in task.stages:
                        builder = builder.summary(engine=engine_name)
                    if "translate" in task.stages:
                        builder = builder.translate(src=task.src, tgt=tgt_lang, engine=engine_name)
                    else:
                        # translate is mandatory for a video task — auto-add.
                        builder = builder.translate(src=task.src, tgt=tgt_lang, engine=engine_name)
                    if "align" in task.stages:
                        builder = builder.align(engine=engine_name)
                    if "tts" in task.stages:
                        builder = builder.tts()
                    if hasattr(builder, "with_progress"):
                        builder = builder.with_progress(on_progress)
                    last = await builder.run()
                task.result = last
                task.elapsed_s = getattr(last, "elapsed_s", None)
            task.status = "done"
        except asyncio.CancelledError:
            task.status = "cancelled"
            raise
        except Exception as exc:
            logger.exception("task %s failed", task.task_id)
            task.status = "failed"
            task.error = f"{type(exc).__name__}: {exc}"
        finally:
            task.mark_terminal()


def load_result_bytes(app: "App", course: str, video: str, fmt: str) -> tuple[bytes, str]:
    """Return ``(bytes, media_type)`` for the stored video translation.

    ``fmt`` is ``"json"`` (raw record list) or ``"srt"``.
    """
    store = app.store(course)
    data_path = store.workspace.translation.path_for(video, suffix=".json")
    raw = Path(data_path).read_bytes()
    if fmt == "json":
        return raw, "application/json"
    if fmt == "srt":
        data = json.loads(raw.decode())
        return _records_to_srt(data).encode("utf-8"), "application/x-subrip"
    raise ValueError(f"unsupported format: {fmt!r}")


def _records_to_srt(data: dict) -> str:
    records = data.get("records") or []
    out: list[str] = []
    idx = 0
    for r in records:
        segments = r.get("segments") or []
        translations = r.get("translations") or {}
        alignment = r.get("alignment") or {}
        tgt = next(iter(translations.keys()), None)
        if tgt is None:
            continue
        pieces = alignment.get(tgt)
        if not (isinstance(pieces, list) and len(pieces) == len(segments)):
            pieces = [translations.get(tgt, "")] + [""] * max(0, len(segments) - 1)
        for seg, piece in zip(segments, pieces):
            if not piece:
                continue
            idx += 1
            out.append(str(idx))
            out.append(f"{_srt_ts(seg.get('start', 0))} --> {_srt_ts(seg.get('end', 0))}")
            out.append(piece)
            out.append("")
    return "\n".join(out)


def _srt_ts(seconds: float) -> str:
    ms_total = int(round(float(seconds) * 1000))
    ms = ms_total % 1000
    s = (ms_total // 1000) % 60
    m = (ms_total // 60_000) % 60
    h = ms_total // 3_600_000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


__all__ = ["Task", "TaskManager", "load_result_bytes"]
