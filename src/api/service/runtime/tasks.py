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

from application.events.progress import ProgressEvent

if TYPE_CHECKING:
    from api.app.app import App
    from api.service.auth import Principal
    from application.orchestrator.video import VideoResult
    from application.resources import ResourceManager


logger = logging.getLogger(__name__)


TaskStatus = str  # "queued" | "running" | "done" | "failed" | "cancelled"


class TaskStore:
    """Atomic JSON persistence for :class:`Task` metadata.

    One file per task under ``<root>/.trx-tasks/<task_id>.json``. Writes
    are atomic (tmp + rename). Only the serialisable public fields are
    persisted — asyncio runners/queues are intentionally omitted.
    """

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def path(self, task_id: str) -> Path:
        return self._root / f"{task_id}.json"

    def save(self, task: "Task") -> None:
        payload = {
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
        p = self.path(task.task_id)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(p)

    def load_all(self) -> list[dict]:
        out: list[dict] = []
        for p in self._root.glob("*.json"):
            try:
                out.append(json.loads(p.read_text(encoding="utf-8")))
            except Exception:
                logger.warning("task-store: failed to parse %s", p)
        return out

    def delete(self, task_id: str) -> None:
        try:
            self.path(task_id).unlink()
        except FileNotFoundError:
            pass


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

    def __init__(
        self,
        app: "App",
        resource_mgr: "ResourceManager",
        *,
        store: "TaskStore | None" = None,
    ) -> None:
        self._app = app
        self._rm = resource_mgr
        self._tasks: dict[str, Task] = {}
        self._store = store

    # -- persistence recovery ------------------------------------------

    def recover(self) -> int:
        """Load persisted tasks; mark previously-running ones as failed.

        Returns the number of ghost tasks that were recovered. Called
        once at startup from the FastAPI lifespan.
        """
        if self._store is None:
            return 0
        n = 0
        for row in self._store.load_all():
            task = Task(
                task_id=row["task_id"],
                course=row["course"],
                video=row["video"],
                src=row.get("src"),
                tgt=list(row.get("tgt") or []),
                stages=list(row.get("stages") or []),
                status=row.get("status") or "queued",
                done=row.get("done") or 0,
                total=row.get("total"),
                error=row.get("error"),
                elapsed_s=row.get("elapsed_s"),
            )
            # Anything mid-flight at restart time is forever lost —
            # mark failed so clients can resubmit.
            if task.status in ("queued", "running"):
                task.status = "failed"
                task.error = "interrupted by server restart"
                self._store.save(task)
            # Ensure the terminal flag is set so new /events subscribers
            # get snapshot + sentinel immediately.
            task._terminal.set()
            self._tasks[task.task_id] = task
            n += 1
        return n

    def _persist(self, task: Task) -> None:
        if self._store is not None:
            try:
                self._store.save(task)
            except Exception:
                logger.exception("task %s: persistence write failed", task.task_id)

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
        # Cancel runners first so _run_task's finally runs mark_terminal
        # and flushes subscriber queues with a sentinel.
        pending = [t._runner for t in self._tasks.values() if t._runner and not t._runner.done()]
        for r in pending:
            r.cancel()
        for r in pending:
            try:
                await r
            except BaseException:
                pass
        # Belt-and-braces: any task that never ran (still queued) still
        # needs its subscribers released so /events handlers can exit.
        for t in self._tasks.values():
            if not t._terminal.is_set():
                if t.status == "queued":
                    t.status = "cancelled"
                t.mark_terminal()

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
        self._persist(task)
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
        self._persist(task)
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
                prom_metrics = getattr(self, "_prom_metrics", None)

                async def _usage_sink(usage, _uid=principal.user_id) -> None:
                    await self._rm.record_usage(_uid, usage)
                    if prom_metrics is not None:
                        model = getattr(usage, "model", "") or "unknown"
                        prom_metrics["engine_requests_total"].labels(model).inc()
                        if getattr(usage, "prompt_tokens", None):
                            prom_metrics["engine_tokens_total"].labels(model, "prompt").inc(usage.prompt_tokens)
                        if getattr(usage, "completion_tokens", None):
                            prom_metrics["engine_tokens_total"].labels(model, "completion").inc(usage.completion_tokens)
                        if getattr(usage, "cost_usd", None):
                            prom_metrics["engine_cost_usd_total"].labels(model).inc(usage.cost_usd)

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
                    if hasattr(builder, "with_usage_sink"):
                        builder = builder.with_usage_sink(_usage_sink)
                    reporter = getattr(self, "_error_reporter", None)
                    if reporter is not None and hasattr(builder, "with_error_reporter"):
                        builder = builder.with_error_reporter(reporter)
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
            self._persist(task)
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
        selected = r.get("selected") or {}
        alignment = r.get("alignment") or {}
        tgt = next(iter(translations.keys()), None)
        if tgt is None:
            continue
        bucket = translations.get(tgt) or {}
        # Pick the variant text: per-record selected → first available.
        if isinstance(bucket, dict):
            chosen = selected.get(tgt) if isinstance(selected, dict) else None
            text = ""
            if chosen and chosen in bucket:
                text = bucket[chosen]
            elif bucket:
                text = next(iter(bucket.values()))
        else:  # legacy bare-string fallthrough
            text = str(bucket or "")
        pieces = alignment.get(tgt)
        if not (isinstance(pieces, list) and len(pieces) == len(segments)):
            pieces = [text] + [""] * max(0, len(segments) - 1)
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


__all__ = ["Task", "TaskManager", "TaskStore", "load_result_bytes"]
