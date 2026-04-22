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
from typing import TYPE_CHECKING

from fastapi import FastAPI

from api.service.auth import Principal
from api.service.routers import health, streams, videos
from api.service.tasks import TaskManager
from application.resources import DEFAULT_TIERS, InMemoryResourceManager, UserTier

if TYPE_CHECKING:
    from api.app.app import App
    from application.resources import ResourceManager


def create_app(
    app: "App",
    *,
    resource_manager: "ResourceManager | None" = None,
    api_keys: dict[str, tuple[str, str]] | None = None,
    tier_map: dict[str, UserTier] | None = None,
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
        api.state.tasks = TaskManager(app, rm)
        api.state.streams = {}
        api.state.auth_map = auth_map
        try:
            yield
        finally:
            await api.state.tasks.shutdown()

    api = FastAPI(title="translatorx API", lifespan=lifespan)
    api.include_router(health.router)
    api.include_router(videos.router)
    api.include_router(streams.router)
    return api


__all__ = ["create_app"]
