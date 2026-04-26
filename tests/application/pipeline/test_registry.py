"""StageRegistry tests."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from application.pipeline import StageRegistry
from ports.pipeline import StageDef


class _DummyStage:
    name = "dummy"

    def __init__(self, params: Any) -> None:
        self.params = params


class _DummyParams(BaseModel):
    foo: str
    n: int = 1


def test_register_and_build() -> None:
    reg = StageRegistry()
    reg.register("dummy", lambda p: _DummyStage(p))
    assert reg.is_registered("dummy")
    assert "dummy" in reg.names()

    stage = reg.build(StageDef(name="dummy", params={"x": 1}))
    assert isinstance(stage, _DummyStage)
    assert dict(stage.params) == {"x": 1}


def test_register_with_pydantic_schema() -> None:
    reg = StageRegistry()
    reg.register("dummy", lambda p: _DummyStage(p), params_schema=_DummyParams)

    stage = reg.build(StageDef(name="dummy", params={"foo": "bar", "n": 3}))
    assert isinstance(stage, _DummyStage)
    assert stage.params.foo == "bar"
    assert stage.params.n == 3


def test_schema_lookup() -> None:
    reg = StageRegistry()
    reg.register("a", lambda p: _DummyStage(p))
    reg.register("b", lambda p: _DummyStage(p), params_schema=_DummyParams)

    assert reg.schema_of("a") is None
    assert reg.schema_of("b") is _DummyParams
    assert reg.schema_of("missing") is None


def test_duplicate_registration_raises() -> None:
    reg = StageRegistry()
    reg.register("dummy", lambda p: _DummyStage(p))
    with pytest.raises(ValueError):
        reg.register("dummy", lambda p: _DummyStage(p))


def test_unknown_stage_build_raises() -> None:
    reg = StageRegistry()
    with pytest.raises(KeyError):
        reg.build(StageDef(name="nope", params={}))


def test_unregister_idempotent() -> None:
    reg = StageRegistry()
    reg.register("dummy", lambda p: _DummyStage(p))
    reg.unregister("dummy")
    reg.unregister("dummy")
    assert not reg.is_registered("dummy")
