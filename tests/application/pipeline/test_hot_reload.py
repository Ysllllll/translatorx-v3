"""Tests for application/pipeline/hot_reload.py (Phase 2 / B3)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from application.config import HotReloadConfig
from application.pipeline.hot_reload import PollWatcher, WatchdogWatcher, make_watcher


class TestPollWatcherSync:
    """poll_once() is synchronous → easy to test deterministically."""

    def test_initial_no_change(self, tmp_path: Path):
        calls = []
        w = PollWatcher(tmp_path, lambda: calls.append(1))
        # First scan populates baseline; poll_once compares against snapshot
        # → empty == empty, no change.
        w._snapshot = w._scan()
        assert w.poll_once() is False

    def test_added_file_triggers(self, tmp_path: Path):
        calls = []
        w = PollWatcher(tmp_path, lambda: calls.append(1))
        w._snapshot = w._scan()
        (tmp_path / "a.yaml").write_text("name: a", encoding="utf-8")
        assert w.poll_once() is True
        assert calls == [1]

    def test_removed_file_triggers(self, tmp_path: Path):
        (tmp_path / "a.yaml").write_text("name: a", encoding="utf-8")
        calls = []
        w = PollWatcher(tmp_path, lambda: calls.append(1))
        w._snapshot = w._scan()
        (tmp_path / "a.yaml").unlink()
        assert w.poll_once() is True
        assert calls == [1]

    def test_modified_mtime_triggers(self, tmp_path: Path):
        f = tmp_path / "a.yaml"
        f.write_text("name: a", encoding="utf-8")
        calls = []
        w = PollWatcher(tmp_path, lambda: calls.append(1))
        w._snapshot = w._scan()
        # Force mtime bump
        import os

        st = f.stat()
        os.utime(f, (st.st_atime, st.st_mtime + 5))
        assert w.poll_once() is True

    def test_non_yaml_ignored(self, tmp_path: Path):
        calls = []
        w = PollWatcher(tmp_path, lambda: calls.append(1))
        w._snapshot = w._scan()
        (tmp_path / "readme.txt").write_text("hi", encoding="utf-8")
        assert w.poll_once() is False
        assert calls == []

    def test_missing_dir_returns_empty(self, tmp_path: Path):
        ghost = tmp_path / "does-not-exist"
        w = PollWatcher(ghost, lambda: None)
        assert w._scan() == {}


class TestPollWatcherAsync:
    @pytest.mark.asyncio
    async def test_start_stop_idempotent(self, tmp_path: Path):
        w = PollWatcher(tmp_path, lambda: None, interval_s=0.05)
        await w.start()
        await w.start()  # second start is no-op
        await w.stop()
        await w.stop()  # second stop is no-op


class TestMakeWatcher:
    def test_poll_backend_default(self, tmp_path: Path):
        cfg = HotReloadConfig()
        w = make_watcher(tmp_path, lambda: None, cfg)
        assert isinstance(w, PollWatcher)

    def test_watchdog_backend(self, tmp_path: Path):
        cfg = HotReloadConfig(backend="watchdog")
        w = make_watcher(tmp_path, lambda: None, cfg)
        assert isinstance(w, WatchdogWatcher)


class TestAppIntegration:
    @pytest.mark.asyncio
    async def test_app_start_hot_reload_no_op_when_disabled(self, tmp_path: Path):
        from api.app import App
        from application.config import AppConfig, EngineEntry, StoreConfig

        cfg = AppConfig(
            store=StoreConfig(root=str(tmp_path)),
            engines={"default": EngineEntry(model="m", base_url="http://localhost", api_key="k")},
            pipelines_dir=str(tmp_path),
            # hot_reload defaults to enabled=False
        )
        app = App(cfg)
        await app.start_hot_reload()
        assert app._hot_reload_watcher is None
        await app.stop_hot_reload()  # also no-op

    @pytest.mark.asyncio
    async def test_app_hot_reload_invalidates_cache(self, tmp_path: Path):
        from api.app import App
        from application.config import AppConfig, EngineEntry, HotReloadConfig, StoreConfig

        pdir = tmp_path / "pipes"
        pdir.mkdir()
        cfg = AppConfig(store=StoreConfig(root=str(tmp_path)), engines={"default": EngineEntry(model="m", base_url="http://localhost", api_key="k")}, pipelines_dir=str(pdir), hot_reload=HotReloadConfig(enabled=True, interval_s=0.05))
        app = App(cfg)
        # warm cache
        assert app.pipelines() == {}
        assert app._pipelines is not None

        await app.start_hot_reload()
        try:
            # Drive watcher deterministically — reset snapshot to empty,
            # write file, call poll_once. Avoids race vs background task.
            w = app._hot_reload_watcher
            assert w is not None
            w._snapshot = {}
            (pdir / "p1.yaml").write_text("name: p1\nbuild: {stage: from_srt, params: {path: x.srt, language: en}}\n", encoding="utf-8")
            assert w.poll_once() is True
            assert app._pipelines is None
            assert "p1" in app.pipelines()
        finally:
            await app.stop_hot_reload()
