"""Tests for :class:`runtime.Usage` arithmetic (Stage 3.2a)."""

from __future__ import annotations

import pytest

from domain.model import Usage


def test_add_basic_counters() -> None:
    a = Usage(prompt_tokens=10, completion_tokens=3, cost_usd=0.1, model="m")
    b = Usage(prompt_tokens=20, completion_tokens=5, cost_usd=0.2, model="m")
    c = a + b
    assert c.prompt_tokens == 30
    assert c.completion_tokens == 8
    assert c.cost_usd == pytest.approx(0.3)
    assert c.requests == 2
    assert c.model == "m"


def test_add_cost_none_both_sides_stays_none() -> None:
    a = Usage(model="m")
    b = Usage(model="m")
    assert (a + b).cost_usd is None


def test_add_cost_none_plus_value_treats_as_zero() -> None:
    a = Usage(cost_usd=None, model="m")
    b = Usage(cost_usd=0.5, model="m")
    assert (a + b).cost_usd == 0.5
    assert (b + a).cost_usd == 0.5


def test_add_model_different_drops_to_empty() -> None:
    a = Usage(model="gpt-4")
    b = Usage(model="claude")
    assert (a + b).model == ""


def test_add_model_empty_absorbed() -> None:
    a = Usage(model="")
    b = Usage(model="gpt-4")
    assert (a + b).model == "gpt-4"
    assert (b + a).model == "gpt-4"


def test_add_extra_merges_later_wins() -> None:
    a = Usage(extra={"foo": 1, "bar": 2})
    b = Usage(extra={"bar": 99, "baz": 3})
    merged = (a + b).extra
    assert merged == {"foo": 1, "bar": 99, "baz": 3}


def test_zero_is_additive_identity() -> None:
    z = Usage.zero()
    assert z.requests == 0
    u = Usage(prompt_tokens=5, completion_tokens=3, cost_usd=0.1, model="m")
    assert (z + u) == u
    assert (u + z) == u


def test_sum_iterable_with_zero() -> None:
    usages = [Usage(prompt_tokens=10, cost_usd=0.1, model="m"), Usage(prompt_tokens=5, cost_usd=0.05, model="m"), Usage(prompt_tokens=2, cost_usd=0.02, model="m")]
    total = sum(usages, Usage.zero())
    assert total.prompt_tokens == 17
    assert total.cost_usd == pytest.approx(0.17)
    assert total.requests == 3
    assert total.model == "m"


def test_builtin_sum_works_via_radd() -> None:
    # Python's sum() starts from 0 by default.
    usages = [Usage(prompt_tokens=1), Usage(prompt_tokens=2)]
    total = sum(usages)  # type: ignore[arg-type]
    assert total.prompt_tokens == 3


def test_add_non_usage_returns_not_implemented() -> None:
    u = Usage()
    assert u.__add__("nope") is NotImplemented  # type: ignore[arg-type]
