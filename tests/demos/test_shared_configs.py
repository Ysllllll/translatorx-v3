"""Lock the schema of demos/_shared.py config factories.

These tests prevent silent regressions where the demo factories drift away
from what ``PuncRestorer.from_config`` / ``Chunker.from_config`` actually
accept (we have already shipped two such bugs: ``model_name`` vs ``model``
and the legacy ``inner``/``refine``/``chunk_len`` shape vs the current
``stages: [...]`` shape). Each assertion below mirrors a real load path so
that future schema changes either flow through here or break loudly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


_DEMOS_DIR = Path(__file__).resolve().parents[2] / "demos"
sys.path.insert(0, str(_DEMOS_DIR))


@pytest.fixture(scope="module")
def demo_shared():
    import _shared

    return _shared


def test_make_punc_config_loads_in_restorer(demo_shared):
    from adapters.preprocess import PuncRestorer

    cfg = demo_shared.make_punc_config("en")
    assert "backends" in cfg, "must wrap per-language map under top-level 'backends'"
    assert "en" in cfg["backends"]
    assert cfg["backends"]["en"]["library"] == "deepmultilingualpunctuation"
    # The deepmultilingualpunctuation factory takes ``model`` (NOT ``model_name``)
    assert "model" in cfg["backends"]["en"]
    assert "model_name" not in cfg["backends"]["en"]

    PuncRestorer.from_config(cfg)


def test_make_chunk_config_loads_in_chunker(demo_shared):
    from adapters.preprocess import Chunker

    cfg = demo_shared.make_chunk_config("en", engine=None)
    assert "backends" in cfg
    en = cfg["backends"]["en"]
    assert en["library"] == "composite"
    # Composite takes ``stages: [...]`` (ordered list); legacy
    # ``inner``/``refine``/``chunk_len`` keys must NOT be emitted.
    assert isinstance(en["stages"], list) and en["stages"], "stages must be a non-empty list"
    for forbidden in ("inner", "refine", "chunk_len"):
        assert forbidden not in en, f"legacy key {forbidden!r} reintroduced"
    # max_len at composite level (CHUNK_LEN) must propagate.
    assert en["max_len"] == demo_shared.CHUNK_LEN
    # Top-level Chunker.from_config requires only 'keep' / 'raise' here;
    # 'rule' is a stage-internal concept.
    assert cfg["on_failure"] in {"keep", "raise"}

    Chunker.from_config(cfg)


def test_make_chunk_config_with_engine_includes_llm_stage(demo_shared):
    from adapters.preprocess import Chunker

    class DummyEngine:  # minimal stand-in; from_config only needs identity
        pass

    engine = DummyEngine()
    cfg = demo_shared.make_chunk_config("zh", engine=engine)
    stages = cfg["backends"]["zh"]["stages"]
    libs = [s["library"] for s in stages]
    assert "spacy" in libs
    assert "llm" in libs
    assert "rule" in libs
    # llm stage must use ``engine``, ``max_len`` (NOT ``chunk_len``),
    # and stage-local ``on_failure='rule'`` for fallback.
    llm_stage = next(s for s in stages if s["library"] == "llm")
    assert llm_stage["engine"] is engine
    assert "max_len" in llm_stage and "chunk_len" not in llm_stage
    assert llm_stage.get("on_failure") == "rule"

    Chunker.from_config(cfg)


def test_make_engine_signature_no_max_retries(demo_shared):
    """make_engine() must NOT pass max_retries — that lives on create_context.

    Catches the exact bug shipped in 572f322 where ``create_engine(max_retries=…)``
    raised TypeError because the parameter doesn't exist on engines.
    """
    import inspect

    src = inspect.getsource(demo_shared.make_engine)
    assert "max_retries" not in src, "make_engine() must not pass max_retries to create_engine() — max_retries is a TranslationContext parameter, not an engine parameter"
