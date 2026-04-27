"""Pipeline hot-reload — watch :attr:`AppConfig.pipelines_dir` for changes.

Phase 2 (D) — opt-in, default OFF. When enabled, the watcher fires
:meth:`App._invalidate_pipelines` whenever a YAML file in the directory
is added, modified, or removed. The next call to
:meth:`App.pipelines` then re-reads the directory.

Two backends share a small :class:`Watcher` Protocol:

* :class:`PollWatcher` — zero-dependency mtime polling (default).
* :class:`WatchdogWatcher` — uses ``watchdog`` if installed, otherwise
  falls back to ``PollWatcher`` with a warning logged once.

The watcher exposes two coroutine entry points so it integrates with
FastAPI lifespan hooks and tests alike::

    watcher = make_watcher(directory, on_change, cfg)
    await watcher.start()
    ...
    await watcher.stop()

``on_change`` is a sync callable invoked from the watcher's task; it
must be cheap (typically just clears a cache).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from application.config import HotReloadConfig

logger = logging.getLogger(__name__)


_OnChange = Callable[[], None]


class Watcher(Protocol):
    async def start(self) -> None: ...
    async def stop(self) -> None: ...


class PollWatcher:
    """mtime-based polling watcher — zero dependencies."""

    def __init__(self, directory: Path, on_change: _OnChange, *, interval_s: float = 2.0) -> None:
        self._dir = Path(directory).expanduser()
        self._cb = on_change
        self._interval = interval_s
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()
        self._snapshot: dict[str, float] = {}

    def _scan(self) -> dict[str, float]:
        if not self._dir.is_dir():
            return {}
        out: dict[str, float] = {}
        for f in self._dir.glob("**/*.yaml"):
            try:
                out[str(f)] = f.stat().st_mtime
            except OSError:
                continue
        return out

    def poll_once(self) -> bool:
        """Public for tests — compare current scan to snapshot, fire cb if changed."""
        cur = self._scan()
        if cur != self._snapshot:
            self._snapshot = cur
            self._cb()
            return True
        return False

    async def _loop(self) -> None:
        self._snapshot = self._scan()
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self._interval)
            except asyncio.TimeoutError:
                pass
            if self._stop.is_set():
                break
            try:
                self.poll_once()
            except Exception:  # noqa: BLE001
                logger.exception("PollWatcher poll failed")

    async def start(self) -> None:
        if self._task is not None:
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._loop(), name="pipeline-hot-reload")

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None


class WatchdogWatcher:
    """watchdog-based watcher — falls back to :class:`PollWatcher` if missing."""

    def __init__(self, directory: Path, on_change: _OnChange, *, interval_s: float = 2.0) -> None:
        self._dir = Path(directory).expanduser()
        self._cb = on_change
        self._interval = interval_s
        self._fallback: PollWatcher | None = None
        self._observer = None
        self._loop: asyncio.AbstractEventLoop | None = None

    async def start(self) -> None:
        try:
            from watchdog.events import FileSystemEventHandler
            from watchdog.observers import Observer
        except ImportError:
            logger.warning("watchdog not installed; hot_reload falling back to PollWatcher")
            self._fallback = PollWatcher(self._dir, self._cb, interval_s=self._interval)
            await self._fallback.start()
            return

        self._loop = asyncio.get_running_loop()
        cb = self._cb
        loop = self._loop

        class _Handler(FileSystemEventHandler):  # type: ignore[misc]
            def on_any_event(self, event) -> None:  # noqa: ANN001
                src = getattr(event, "src_path", "")
                if not str(src).endswith(".yaml"):
                    return
                loop.call_soon_threadsafe(cb)

        self._observer = Observer()
        self._observer.schedule(_Handler(), str(self._dir), recursive=True)
        self._observer.start()

    async def stop(self) -> None:
        if self._fallback is not None:
            await self._fallback.stop()
            self._fallback = None
            return
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=2.0)
            self._observer = None


def make_watcher(directory: str | Path, on_change: _OnChange, cfg: HotReloadConfig) -> Watcher:
    """Construct a watcher per :class:`HotReloadConfig`."""
    p = Path(directory)
    if cfg.backend == "watchdog":
        return WatchdogWatcher(p, on_change, interval_s=cfg.interval_s)
    return PollWatcher(p, on_change, interval_s=cfg.interval_s)


__all__ = ["Watcher", "PollWatcher", "WatchdogWatcher", "make_watcher"]
