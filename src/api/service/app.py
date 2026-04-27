"""FastAPI application factory.

Call :func:`create_app` with an :class:`App` and a
:class:`ResourceManager`. The returned FastAPI instance exposes:

* ``POST /api/courses/{course}/videos`` — submit translation task
* ``GET  /api/courses/{course}/videos`` — list tasks
* ``GET  /api/courses/{course}/videos/{task_id}`` — task state
* ``GET  /api/courses/{course}/videos/{task_id}/events`` — SSE progress
* ``GET  /api/courses/{course}/videos/{video}/result?format=srt|json``
* ``POST /api/courses/{course}/videos/{task_id}/cancel``
* ``POST /api/streams`` + ``/segments`` + ``/events`` + ``/close``
* ``GET  /health`` / ``/ready``

Authentication via ``X-API-Key`` when ``api_keys`` is non-empty; when
empty the service is open (dev mode) and every request is treated as
an anonymous ``free``-tier principal.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI

from api.service.auth import Principal
from api.service.observability import install_opentelemetry, install_prometheus
from api.service.routers import admin, events, health, pipelines, stages, streams, usage, videos
from api.service.runtime.stream_registry import InMemoryStreamRegistry
from api.service.runtime.tasks import TaskManager, TaskStore
from application.resources import DEFAULT_TIERS, InMemoryResourceManager, UserTier

if TYPE_CHECKING:
    from api.app.app import App
    from application.resources import ResourceManager


def from_app_config(app: "App") -> FastAPI:
    """Build a FastAPI service from ``app.config.service`` alone.

    Resolves ``api_keys``, ``resource_backend`` (memory/redis),
    ``redis_url``, and ``task_backend`` (inproc/arq) so callers don't
    have to duplicate wiring.
    """
    svc = app.config.service
    api_keys: dict[str, tuple[str, str]] = {key: (entry.user_id, entry.tier) for key, entry in svc.api_keys.items()}
    rm: "ResourceManager"
    if svc.resource_backend == "redis":
        if not svc.redis_url:
            raise ValueError("service.resource_backend='redis' requires service.redis_url")
        import redis.asyncio as redis_async

        from application.resources import RedisResourceConfig, RedisResourceManager

        client = redis_async.from_url(svc.redis_url, decode_responses=True)
        rm = RedisResourceManager(client, RedisResourceConfig(key_prefix=svc.redis_key_prefix))
    else:
        rm = InMemoryResourceManager()
    task_manager = None
    if svc.task_backend == "arq":
        if not svc.redis_url:
            raise ValueError("service.task_backend='arq' requires service.redis_url")
        try:
            from arq.connections import RedisSettings, create_pool
        except ImportError as exc:
            raise ImportError("service.task_backend='arq' requires `pip install arq`") from exc

        from api.service.runtime.tasks_arq import ArqTaskManager

        async def _mk_arq(_app: "App" = app, _rm: "ResourceManager" = rm):
            pool = await create_pool(RedisSettings.from_dsn(svc.redis_url), default_queue_name=svc.arq_queue_name)
            return ArqTaskManager(
                _app,
                _rm,
                pool,
                queue_name=svc.arq_queue_name,
                task_prefix=svc.arq_task_prefix,
                events_prefix=svc.arq_events_prefix,
            )

        task_manager = _mk_arq
    return create_app(app, resource_manager=rm, api_keys=api_keys, task_manager_factory=task_manager)


def create_app(
    app: "App",
    *,
    resource_manager: "ResourceManager | None" = None,
    api_keys: dict[str, tuple[str, str]] | None = None,
    tier_map: dict[str, UserTier] | None = None,
    task_manager_factory: Any | None = None,
) -> FastAPI:
    """Build a FastAPI app wired to the given :class:`App` facade.

    Args:
        app: Top-level :class:`App` (config + Builder factories).
        resource_manager: :class:`ResourceManager`; defaults to an
            :class:`InMemoryResourceManager` for dev.
        api_keys: Mapping ``{api_key: (user_id, tier_name)}``. Empty →
            dev mode (no auth).
        tier_map: Override for tier name → :class:`UserTier`. Defaults
            to :data:`DEFAULT_TIERS`.
        task_manager_factory: Optional async factory returning a task
            manager (e.g. :class:`ArqTaskManager`). When ``None`` the
            in-process :class:`TaskManager` is used.
    """
    rm = resource_manager or InMemoryResourceManager()
    tier_resolver = dict(tier_map or DEFAULT_TIERS)
    auth_map: dict[str, Principal] = {}
    for key, (user_id, tier_name) in (api_keys or {}).items():
        tier = tier_resolver.get(tier_name)
        if tier is None:
            raise ValueError(f"unknown tier {tier_name!r} for api_key mapping")
        auth_map[key] = Principal(user_id=user_id, tier=tier)

    @asynccontextmanager
    async def lifespan(api: FastAPI):
        api.state.app = app
        api.state.rm = rm

        # Build the TaskStore if configured.
        task_store: TaskStore | None = None
        persist_path = getattr(app.config.service, "task_persist_path", "") or ""
        if persist_path:
            from pathlib import Path as _P

            p = _P(persist_path).expanduser()
            if not p.is_absolute():
                p = _P(app.config.store.root).expanduser() / p
            task_store = TaskStore(p)

        if task_manager_factory is not None:
            api.state.tasks = await task_manager_factory()
        else:
            api.state.tasks = TaskManager(app, rm, store=task_store)
            api.state.tasks.recover()

        # Error buffer + optional JSONL reporter chain.
        from api.service.runtime.error_buffer import ErrorBuffer

        delegate = None
        errlog = getattr(app.config.service, "error_log_path", "") or ""
        if errlog:
            from pathlib import Path as _P

            from adapters.reporters import JsonlErrorReporter

            ep = _P(errlog).expanduser()
            if not ep.is_absolute():
                ep = _P(app.config.store.root).expanduser() / ep
            delegate = JsonlErrorReporter(ep)
        api.state.error_buffer = ErrorBuffer(delegate=delegate)
        api.state.tasks._error_reporter = api.state.error_buffer
        # Wire Prometheus metrics (if installed) into the task manager sink.
        metrics = getattr(api.state, "prom_metrics", None)
        if metrics is not None:
            api.state.tasks._prom_metrics = metrics
        api.state.streams = InMemoryStreamRegistry()
        api.state.auth_map = auth_map
        api.state.tier_map = tier_resolver
        try:
            yield
        finally:
            # Drain live streams first so their pump tasks stop emitting
            # into SSE queues we're about to tear down.
            streams_reg = getattr(api.state, "streams", None)
            if streams_reg is None:
                all_streams = []
            elif hasattr(streams_reg, "values"):
                try:
                    all_streams = list(streams_reg.values())
                except TypeError:
                    # dict.values() returns a view, handled above; here
                    # a custom registry returned a callable/iterable.
                    all_streams = list(streams_reg.values())
            else:
                all_streams = []
            for s in all_streams:
                try:
                    await s.handle.close()
                except Exception:
                    pass
                if s.pump_task is not None and not s.pump_task.done():
                    s.pump_task.cancel()
                    try:
                        await s.pump_task
                    except BaseException:
                        pass
                s.status = "closed"
                try:
                    s.queue.put_nowait(None)
                except Exception:
                    pass
            if streams_reg is not None and hasattr(streams_reg, "close"):
                try:
                    await streams_reg.close()
                except Exception:
                    pass
            await api.state.tasks.shutdown()

    api = FastAPI(title="translatorx API", lifespan=lifespan)
    from api.service.middleware.errors import install_error_handlers

    install_error_handlers(api)
    api.include_router(health.router)
    api.include_router(videos.router)
    api.include_router(streams.router)
    api.include_router(events.router)
    api.include_router(usage.router)
    api.include_router(admin.router)
    api.include_router(pipelines.router)
    api.include_router(stages.router)

    svc_cfg = app.config.service
    if svc_cfg.cors_origins:
        from fastapi.middleware.cors import CORSMiddleware

        api.add_middleware(
            CORSMiddleware,
            allow_origins=svc_cfg.cors_origins,
            allow_credentials=svc_cfg.cors_allow_credentials,
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=["X-API-Key"],
        )
    if svc_cfg.rps_limit > 0:
        from api.service.middleware.rate_limit import RateLimitMiddleware

        api.add_middleware(RateLimitMiddleware, rps=svc_cfg.rps_limit, burst=svc_cfg.rps_burst or None)
    if svc_cfg.request_log_enabled:
        from api.service.middleware.logging import RequestLogMiddleware

        api.add_middleware(RequestLogMiddleware)
    install_prometheus(api, enabled=svc_cfg.prometheus_enabled, path=svc_cfg.prometheus_path)
    install_opentelemetry(
        api,
        enabled=svc_cfg.otel_enabled,
        service_name=svc_cfg.otel_service_name,
        exporter=svc_cfg.otel_exporter,
        endpoint=svc_cfg.otel_endpoint or None,
    )
    return api


__all__ = ["create_app", "from_app_config"]
