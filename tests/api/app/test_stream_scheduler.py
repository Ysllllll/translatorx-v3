"""Phase 5 L3 — StreamBuilder + LiveStreamHandle ↔ scheduler integration."""

from __future__ import annotations

from pathlib import Path

import pytest

from api.app import App
from application.checker import CheckReport, Checker
from application.scheduler import DEFAULT_QUOTAS, FairScheduler, QuotaExceeded, TenantQuota
from domain.model import Segment
from domain.model.usage import CompletionResult


YAML = """
engines:
  default:
    kind: openai_compat
    model: test-model
    base_url: http://localhost:0/v1
    api_key: EMPTY
contexts:
  en_zh:
    src: en
    tgt: zh
store:
  kind: json
  root: "{root}"
runtime:
  default_checker_profile: strict
  max_concurrent_videos: 2
  flush_every: 1
tenants:
  acme:
    max_concurrent_streams: 1
    qos_tier: standard
"""


@pytest.fixture
def app(tmp_path: Path) -> App:
    ws_root = tmp_path / "ws"
    cfg_path = tmp_path / "app.yaml"
    cfg_path.write_text(YAML.format(root=ws_root.as_posix()), encoding="utf-8")
    return App.from_config(cfg_path)


class _FakeEngine:
    model = "test-model"

    async def complete(self, messages, **_):
        return CompletionResult(text=f"[{messages[-1]['content']}]")

    async def stream(self, messages, **_):
        yield (await self.complete(messages)).text


class _PassChecker(Checker):
    def __init__(self) -> None:
        super().__init__(rules=[])

    def check(self, source, translation, profile=None, **_) -> CheckReport:
        return CheckReport.ok()


def _wire(app: App, monkeypatch):
    fake = _FakeEngine()
    monkeypatch.setattr(app, "engine", lambda name="default": fake)
    monkeypatch.setattr(app, "checker", lambda s, t: _PassChecker())


class TestStreamBuilderScheduler:
    @pytest.mark.asyncio
    async def test_start_async_no_tenant_skips_scheduler(self, app: App, monkeypatch):
        """Without `.tenant(...)`, start_async behaves like start()."""
        _wire(app, monkeypatch)
        b = app.stream(course="c1", video="v1", language="en").translate(src="en", tgt="zh")
        handle = await b.start_async()
        try:
            await handle.feed(Segment(start=0.0, end=1.0, text="Hello."))
            await handle.close()
            recs = [r async for r in handle.records()]
            assert len(recs) == 1
            assert handle._ticket is None
        finally:
            await handle.close()

    @pytest.mark.asyncio
    async def test_start_async_acquires_ticket_for_tenant(self, app: App, monkeypatch):
        _wire(app, monkeypatch)
        b = app.stream(course="c1", video="v1", language="en").translate(src="en", tgt="zh").tenant("acme")
        handle = await b.start_async()
        try:
            assert handle._ticket is not None
            assert handle._ticket.tenant_id == "acme"
            stats = app.scheduler.stats()
            assert stats.active_by_tenant.get("acme") == 1
        finally:
            await handle.close()

        # Ticket released on close.
        stats = app.scheduler.stats()
        assert stats.active_by_tenant.get("acme", 0) == 0

    @pytest.mark.asyncio
    async def test_quota_exceeded_when_no_wait(self, app: App, monkeypatch):
        """`.tenant(id, wait=False)` raises QuotaExceeded when saturated."""
        _wire(app, monkeypatch)

        first = await app.stream(course="c1", video="v1", language="en").translate(src="en", tgt="zh").tenant("acme").start_async()
        try:
            with pytest.raises(QuotaExceeded):
                await app.stream(course="c1", video="v2", language="en").translate(src="en", tgt="zh").tenant("acme", wait=False).start_async()
        finally:
            await first.close()

    @pytest.mark.asyncio
    async def test_set_scheduler_override(self, app: App, monkeypatch):
        """App.set_scheduler swaps in a custom scheduler."""
        _wire(app, monkeypatch)
        custom = FairScheduler(quotas={"acme": TenantQuota(max_concurrent_streams=2)}, default_quota=DEFAULT_QUOTAS["free"])
        app.set_scheduler(custom)
        assert app.scheduler is custom

        h1 = await app.stream(course="c1", video="v1", language="en").translate(src="en", tgt="zh").tenant("acme").start_async()
        h2 = await app.stream(course="c1", video="v2", language="en").translate(src="en", tgt="zh").tenant("acme").start_async()
        try:
            assert custom.stats().active_by_tenant["acme"] == 2
        finally:
            await h1.close()
            await h2.close()

    @pytest.mark.asyncio
    async def test_ticket_released_even_if_start_fails(self, app: App, monkeypatch):
        """If StreamBuilder.start() throws after slot acquisition, ticket released."""
        _wire(app, monkeypatch)
        # No translate() set → start() raises ValueError.
        b = app.stream(course="c1", video="v1", language="en").tenant("acme")
        with pytest.raises(ValueError):
            await b.start_async()
        # Slot must have been released.
        assert app.scheduler.stats().active_by_tenant.get("acme", 0) == 0

    @pytest.mark.asyncio
    async def test_is_closed_property(self, app: App, monkeypatch):
        """LiveStreamHandle.is_closed exposes _closed publicly (T1 cleanup)."""
        _wire(app, monkeypatch)
        h = await app.stream(course="c1", video="v1", language="en").translate(src="en", tgt="zh").tenant("acme").start_async()
        assert h.is_closed is False
        await h.close()
        assert h.is_closed is True
        # idempotent
        await h.close()
        assert h.is_closed is True
