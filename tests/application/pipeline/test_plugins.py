"""Tests for application/pipeline/plugins.py — entry-point discovery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from application.pipeline import StageRegistry
from application.pipeline.plugins import PluginLoadError, discover_stages, load_plugin


@dataclass
class _FakeEP:
    """Stand-in for importlib.metadata.EntryPoint."""

    name: str
    value: str
    target: Any

    def load(self) -> Any:
        return self.target


def _make_register_fn(name: str):
    def _register(reg: StageRegistry) -> None:
        reg.register(name, lambda params: _DummyStage(name))

    return _register


class _DummyStage:
    def __init__(self, name: str) -> None:
        self.name = name


# ---------------------------------------------------------------------------
# load_plugin
# ---------------------------------------------------------------------------


def test_load_plugin_returns_callable() -> None:
    target = _make_register_fn("foo")
    ep = _FakeEP(name="foo", value="x:y", target=target)
    assert load_plugin(ep) is target


def test_load_plugin_rejects_non_callable() -> None:
    ep = _FakeEP(name="x", value="m:obj", target=42)
    with pytest.raises(PluginLoadError, match="expected a callable"):
        load_plugin(ep)


# ---------------------------------------------------------------------------
# discover_stages
# ---------------------------------------------------------------------------


def _patch_eps(monkeypatch: pytest.MonkeyPatch, eps: list[_FakeEP]) -> None:
    """Patch importlib.metadata.entry_points to return our fake EPs."""
    from application.pipeline import plugins as plugins_mod

    class _Adapter:
        def __init__(self, items: list[_FakeEP]) -> None:
            self._items = items

        def __iter__(self):
            return iter(self._items)

    def _entry_points(*, group: str):
        return _Adapter(eps)

    monkeypatch.setattr(plugins_mod.importlib_metadata, "entry_points", _entry_points)


def test_discover_loads_each_plugin(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_eps(monkeypatch, [_FakeEP(name="alpha", value="a:r", target=_make_register_fn("alpha_stage")), _FakeEP(name="beta", value="b:r", target=_make_register_fn("beta_stage"))])

    reg = StageRegistry()
    loaded = discover_stages(reg)

    assert sorted(loaded) == ["alpha", "beta"]
    assert reg.is_registered("alpha_stage")
    assert reg.is_registered("beta_stage")


def test_discover_filters_by_names(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_eps(monkeypatch, [_FakeEP(name="alpha", value="a:r", target=_make_register_fn("alpha_stage")), _FakeEP(name="beta", value="b:r", target=_make_register_fn("beta_stage"))])
    reg = StageRegistry()
    loaded = discover_stages(reg, names=["beta"])

    assert loaded == ["beta"]
    assert not reg.is_registered("alpha_stage")
    assert reg.is_registered("beta_stage")


def test_discover_warn_skips_broken_plugin(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    def boom(_reg):
        raise RuntimeError("boom")

    _patch_eps(monkeypatch, [_FakeEP(name="bad", value="b:r", target=boom), _FakeEP(name="ok", value="o:r", target=_make_register_fn("ok_stage"))])

    reg = StageRegistry()
    with caplog.at_level("WARNING", logger="application.pipeline.plugins"):
        loaded = discover_stages(reg)

    assert loaded == ["ok"]
    assert any("bad" in rec.message for rec in caplog.records)
    assert reg.is_registered("ok_stage")


def test_discover_raise_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(_reg):
        raise RuntimeError("boom")

    _patch_eps(monkeypatch, [_FakeEP(name="bad", value="b:r", target=boom)])

    reg = StageRegistry()
    with pytest.raises(PluginLoadError, match="bad"):
        discover_stages(reg, on_error="raise")


def test_discover_ignore_swallows_errors(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    def boom(_reg):
        raise RuntimeError("boom")

    _patch_eps(monkeypatch, [_FakeEP(name="bad", value="b:r", target=boom)])

    reg = StageRegistry()
    with caplog.at_level("WARNING", logger="application.pipeline.plugins"):
        loaded = discover_stages(reg, on_error="ignore")

    assert loaded == []
    assert not caplog.records  # ignore mode is silent


def test_discover_invalid_on_error_value() -> None:
    with pytest.raises(ValueError, match="on_error must be"):
        discover_stages(StageRegistry(), on_error="explode")


def test_discover_no_eps_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_eps(monkeypatch, [])
    reg = StageRegistry()
    assert discover_stages(reg) == []
