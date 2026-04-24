"""Tests for the generic ``engine`` rule-pipeline library."""

from __future__ import annotations

from adapters.parsers.engine import NULL_TRACKER, NullTracker, Pipeline, RecordingTracker, Rule, RuleHit, TextItemRule


class _UpperRule(TextItemRule):
    def __init__(self):
        super().__init__("U1", "uppercase", lambda s: s.upper())


class _ExclaimRule(TextItemRule):
    def __init__(self):
        super().__init__("U2", "add exclamation", lambda s: s + "!" if s and not s.endswith("!") else s)


class _DropEmpty(TextItemRule):
    """Returns None for empty strings, triggering ItemRule drop."""

    def __init__(self):
        super().__init__("D1", "drop empties", lambda s: s)

    def apply_one(self, item, *, tracker, origin):
        return item or None


class _SplitSpacesRule(Rule[str]):
    id = "S1"
    reason = "split on space"

    def apply(self, items, origins, *, tracker):
        out, out_o = [], []
        for item, origin in zip(items, origins):
            parts = item.split() or [item]
            if len(parts) > 1:
                tracker.fire(self.id, self.reason, before=item, after=" | ".join(parts), origin=origin)
            for p in parts:
                out.append(p)
                out_o.append(origin)
        return out, out_o


# ── tracker basics ────────────────────────────────────────────────────


def test_null_tracker_fires_nothing():
    t = NullTracker()
    t.fire("X", "noop", before="a", after="b", origin=0)
    # No state, no exceptions.
    assert NULL_TRACKER is not None


def test_recording_tracker_records_by_origin():
    t = RecordingTracker()
    t.fire("X", "why", before="a", after="b", origin=0)
    t.fire("Y", "why2", before="b", after="c", origin=0)
    t.fire("X", "why", before="z", after="zz", origin=1)
    assert len(t.hits_by_origin[0]) == 2
    assert len(t.hits_by_origin[1]) == 1
    assert t.rule_counts == {"X": 2, "Y": 1}
    assert isinstance(t.hits_by_origin[0][0], RuleHit)


# ── Pipeline.run ──────────────────────────────────────────────────────


def test_pipeline_run_trivial_str_rules():
    pipe = Pipeline[str]([_UpperRule(), _ExclaimRule()])
    out, origins = pipe.run(["hello", "world"])
    assert out == ["HELLO!", "WORLD!"]
    assert origins == [0, 1]


def test_pipeline_run_with_tracker():
    pipe = Pipeline[str]([_UpperRule(), _ExclaimRule()])
    t = RecordingTracker()
    pipe.run(["abc"], tracker=t)
    # U1: abc→ABC; U2: ABC→ABC!
    assert t.rule_counts["U1"] == 1
    assert t.rule_counts["U2"] == 1
    hits = t.hits_by_origin[0]
    assert [h.rule_id for h in hits] == ["U1", "U2"]


# ── Idempotency ────────────────────────────────────────────────────────


def test_pipeline_is_idempotent():
    pipe = Pipeline[str]([_UpperRule(), _ExclaimRule()])
    first, _ = pipe.run(["hello"])
    second, _ = pipe.run(first)
    assert first == second


# ── Origin stability ──────────────────────────────────────────────────


def test_origin_stability_through_drop():
    pipe = Pipeline[str]([_DropEmpty()])
    items, origins = pipe.run(["a", "", "b", "", "c"])
    assert items == ["a", "b", "c"]
    assert origins == [0, 2, 4]


def test_origin_stability_through_split():
    pipe = Pipeline[str]([_SplitSpacesRule()])
    t = RecordingTracker()
    items, origins = pipe.run(["alpha beta", "gamma"], tracker=t)
    assert items == ["alpha", "beta", "gamma"]
    # both of "alpha" and "beta" share origin 0
    assert origins == [0, 0, 1]
    # Tracker recorded against origin 0 (the split item).
    assert t.rule_counts["S1"] == 1
    assert t.hits_by_origin[0][0].rule_id == "S1"


def test_max_lookahead_propagates():
    class _LA3(Rule[str]):
        id = "L3"
        reason = "3-ahead"
        lookahead = 3

        def apply(self, items, origins, *, tracker):
            return items, origins

    pipe = Pipeline[str]([_UpperRule(), _LA3()])
    assert pipe.max_lookahead == 3
