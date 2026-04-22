"""Arq-backed distributed task manager.

Stores task metadata in Redis hashes and fans out SSE events via Redis
Pub/Sub, allowing the API tier to scale to many replicas while workers
run out-of-process on the ``trx:tasks`` arq queue.

The web process (:class:`ArqTaskManager`) only enqueues jobs and
subscribes to per-task event channels — the heavy lifting runs inside
:func:`_worker_run_task` invoked by the :mod:`arq` worker
(``translatorx-worker``).

This module is **optional**: importing it triggers an ``ImportError``
only when :mod:`arq` is not installed. Configure via
``service.task_backend = "arq"`` in :class:`AppConfig`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from api.service.tasks import Task

if TYPE_CHECKING:
    from api.app.app import App
    from api.service.auth import Principal
    from application.resources import ResourceManager


logger = logging.getLogger(__name__)


def _task_key(prefix: str, task_id: str) -> str:
    return f"{prefix}{task_id}"


def _events_channel(prefix: str, task_id: str) -> str:
    return f"{prefix}{task_id}"


class ArqTaskManager:
    """Redis-backed counterpart of :class:`TaskManager` for multi-worker deployments.

    Implements the same surface (``submit``, ``get``, ``cancel``, ``list_for_course``,
    ``shutdown``, plus a :meth:`subscribe_async` hook for SSE) so routers work
    unchanged.
    """

    def __init__(
        self,
        app: "App",
        resource_mgr: "ResourceManager",
        redis_client: Any,
        *,
        queue_name: str,
        task_prefix: str,
        events_prefix: str,
    ) -> None:
        try:
            from arq.connections import ArqRedis
        except ImportError as exc:  # pragma: no cover - optional dep
            raise ImportError("arq is required for ArqTaskManager; install with `pip install arq`") from exc
        if not isinstance(redis_client, ArqRedis):
            raise TypeError("redis_client must be an ArqRedis instance")
        self._app = app
        self._rm = resource_mgr
        self._redis = redis_client
        self._queue = queue_name
        self._task_prefix = task_prefix
        self._events_prefix = events_prefix
        self._local_tasks: dict[str, Task] = {}
        self._pubsub_tasks: dict[str, asyncio.Task] = {}

    @property
    def app(self) -> "App":
        return self._app

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
        self._local_tasks[task_id] = task
        payload = {
            "task_id": task_id,
            "course": course,
            "video": video,
            "src": src,
            "tgt": list(tgt),
            "stages": list(stages),
            "source_path": str(source_path),
            "source_kind": source_kind,
            "engine_name": engine_name,
            "user_id": principal.user_id,
            "tier": principal.tier.name,
            "status": "queued",
            "submitted_at": time.time(),
        }

        async def _enqueue() -> None:
            await self._redis.hset(_task_key(self._task_prefix, task_id), mapping={"meta": json.dumps(payload)})
            await self._redis.enqueue_job("translate_video", payload, _queue_name=self._queue)
            # Start forwarding events from Redis to the local Task queue.
            self._pubsub_tasks[task_id] = asyncio.create_task(self._forward_events(task))

        loop = asyncio.get_running_loop()
        task._runner = loop.create_task(_enqueue())
        return task

    # -- accessors ------------------------------------------------------

    def get(self, task_id: str) -> Task | None:
        return self._local_tasks.get(task_id)

    def list_for_course(self, course: str) -> list[Task]:
        return [t for t in self._local_tasks.values() if t.course == course]

    async def cancel(self, task_id: str) -> bool:
        task = self._local_tasks.get(task_id)
        if task is None:
            return False
        if task.status in ("done", "failed", "cancelled"):
            return True
        # Best-effort: set a Redis cancel flag the worker checks between stages.
        await self._redis.hset(_task_key(self._task_prefix, task_id), mapping={"cancel": "1"})
        return True

    async def shutdown(self) -> None:
        for t in list(self._pubsub_tasks.values()):
            t.cancel()
        for t in list(self._pubsub_tasks.values()):
            try:
                await t
            except BaseException:
                pass

    # -- event forwarding ----------------------------------------------

    async def _forward_events(self, task: Task) -> None:
        """Subscribe to the worker's event stream for this task and fan into ``task._queues``."""
        channel = _events_channel(self._events_prefix, task.task_id)
        try:
            pubsub = self._redis.pubsub()
            await pubsub.subscribe(channel)
            try:
                async for msg in pubsub.listen():
                    if msg is None or msg.get("type") != "message":
                        continue
                    data = msg.get("data")
                    if isinstance(data, bytes):
                        data = data.decode()
                    try:
                        event = json.loads(data)
                    except Exception:
                        continue
                    evkind = event.get("event")
                    evdata = event.get("data") or {}
                    if evkind == "status":
                        task.status = evdata.get("status", task.status)
                        task.done = evdata.get("done", task.done)
                        task.total = evdata.get("total", task.total)
                        task.error = evdata.get("error", task.error)
                        task.elapsed_s = evdata.get("elapsed_s", task.elapsed_s)
                    elif evkind == "record":
                        task.done = evdata.get("done", task.done)
                        task.total = evdata.get("total", task.total)
                    task.emit(event)
                    if evkind == "terminal":
                        task.mark_terminal()
                        return
            finally:
                await pubsub.unsubscribe(channel)
                await pubsub.close()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("pubsub forward failed for task %s", task.task_id)


# -- worker-side ------------------------------------------------------


async def _publish(redis: Any, events_prefix: str, task_id: str, event: dict) -> None:
    await redis.publish(_events_channel(events_prefix, task_id), json.dumps(event))


async def _worker_run_task(ctx: dict, payload: dict) -> None:
    """Arq job: run a video translation task end-to-end.

    ``ctx`` is the arq context (contains ``redis``, ``app``, ``rm``,
    ``events_prefix``). ``payload`` is what :meth:`ArqTaskManager.submit`
    wrote to the queue.
    """
    from application.observability.progress import ProgressEvent

    redis = ctx["redis"]
    app = ctx["app"]
    rm = ctx["rm"]
    events_prefix = ctx["events_prefix"]
    task_prefix = ctx["task_prefix"]
    tier_map = ctx["tier_map"]

    task_id = payload["task_id"]
    course = payload["course"]
    video = payload["video"]
    src = payload.get("src")
    tgt_list = payload.get("tgt") or []
    stages = payload.get("stages") or ["translate"]
    source_path = Path(payload["source_path"])
    source_kind = payload.get("source_kind")
    engine_name = payload["engine_name"]
    user_id = payload["user_id"]
    tier_name = payload.get("tier", "free")
    tier = tier_map.get(tier_name)
    if tier is None:  # pragma: no cover - config mismatch
        await _publish(
            redis,
            events_prefix,
            task_id,
            {"event": "terminal", "data": {"status": "failed", "error": f"unknown tier {tier_name}"}},
        )
        return

    started = time.time()

    async def _check_cancel() -> bool:
        raw = await redis.hget(_task_key(task_prefix, task_id), "cancel")
        return bool(raw) and raw != b"0" and raw != "0"

    def on_progress(event: ProgressEvent) -> None:
        ev = {
            "event": event.kind,
            "data": {
                "processor": event.processor,
                "done": event.done,
                "total": event.total,
                "record_id": event.record_id,
                "cache_hit": event.cache_hit,
            },
        }
        # Fire-and-forget publish from sync callback.
        asyncio.create_task(_publish(redis, events_prefix, task_id, ev))

    async def _usage_sink(usage, _uid=user_id) -> None:
        await rm.record_usage(_uid, usage)

    await _publish(redis, events_prefix, task_id, {"event": "status", "data": {"status": "running"}})
    try:
        async with rm.acquire_video_slot(user_id, tier):
            last = None
            for tgt_lang in tgt_list:
                if await _check_cancel():
                    raise asyncio.CancelledError()
                builder = app.video(course=course, video=video)
                if "transcribe" in stages:
                    builder = builder.transcribe(audio=source_path, language=src)
                else:
                    builder = builder.source(source_path, language=src, kind=source_kind)
                if "summary" in stages:
                    builder = builder.summary(engine=engine_name)
                builder = builder.translate(src=src, tgt=tgt_lang, engine=engine_name)
                if "align" in stages:
                    builder = builder.align(engine=engine_name)
                if "tts" in stages:
                    builder = builder.tts()
                if hasattr(builder, "with_progress"):
                    builder = builder.with_progress(on_progress)
                if hasattr(builder, "with_usage_sink"):
                    builder = builder.with_usage_sink(_usage_sink)
                last = await builder.run()
        elapsed = time.time() - started
        await _publish(
            redis,
            events_prefix,
            task_id,
            {"event": "terminal", "data": {"status": "done", "elapsed_s": elapsed}},
        )
    except asyncio.CancelledError:
        await _publish(redis, events_prefix, task_id, {"event": "terminal", "data": {"status": "cancelled"}})
        raise
    except Exception as exc:
        logger.exception("worker task %s failed", task_id)
        await _publish(
            redis,
            events_prefix,
            task_id,
            {"event": "terminal", "data": {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}},
        )


def build_worker_settings(app: "App") -> dict:
    """Return an arq ``WorkerSettings`` dict configured from ``app.config``.

    Used by the ``translatorx-worker`` console script.
    """
    try:
        from arq.connections import RedisSettings
    except ImportError as exc:  # pragma: no cover
        raise ImportError("arq is required; install with `pip install arq`") from exc

    svc = app.config.service
    if not svc.redis_url:
        raise ValueError("service.redis_url is required for arq worker")

    from application.resources import DEFAULT_TIERS

    async def on_startup(ctx: dict) -> None:
        import redis.asyncio as redis_async

        from application.resources import RedisResourceConfig, RedisResourceManager

        ctx["app"] = app
        plain = redis_async.from_url(svc.redis_url, decode_responses=True)
        ctx["rm"] = RedisResourceManager(plain, RedisResourceConfig(key_prefix=svc.redis_key_prefix))
        ctx["events_prefix"] = svc.arq_events_prefix
        ctx["task_prefix"] = svc.arq_task_prefix
        ctx["tier_map"] = DEFAULT_TIERS

    return {
        "functions": [_worker_run_task],
        "redis_settings": RedisSettings.from_dsn(svc.redis_url),
        "queue_name": svc.arq_queue_name,
        "on_startup": on_startup,
    }


__all__ = ["ArqTaskManager", "build_worker_settings"]
