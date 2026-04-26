"""StageRegistry — name → factory + Pydantic params schema.

Stages register a *factory* (callable that takes a typed ``Params``
model and returns a configured Stage instance) along with the Params
schema. The runtime looks up factories by ``StageDef.name`` and feeds
each factory the parsed params plus any registered dependencies.

Phase 1 keeps registration in-process. Phase 6 will swap to
``importlib.metadata`` entry points without changing the registry's
public surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from ports.pipeline import StageDef
from ports.stage import RecordStage, SourceStage, SubtitleStage

__all__ = ["StageFactory", "StageEntry", "StageRegistry"]


Stage = SourceStage | SubtitleStage | RecordStage
StageFactory = Callable[[Mapping[str, Any]], Stage]


@dataclass(frozen=True, slots=True)
class StageEntry:
    name: str
    factory: StageFactory
    params_schema: type | None = None
    """Optional Pydantic model class. None = factory accepts raw mapping."""


class StageRegistry:
    """Mutable in-process registry.

    Not thread-safe by design — registration is expected at module
    import time (or test setup) before any pipeline runs.
    """

    __slots__ = ("_entries",)

    def __init__(self) -> None:
        self._entries: dict[str, StageEntry] = {}

    def register(
        self,
        name: str,
        factory: StageFactory,
        *,
        params_schema: type | None = None,
    ) -> None:
        if name in self._entries:
            raise ValueError(f"Stage {name!r} already registered")
        self._entries[name] = StageEntry(name=name, factory=factory, params_schema=params_schema)

    def unregister(self, name: str) -> None:
        self._entries.pop(name, None)

    def is_registered(self, name: str) -> bool:
        return name in self._entries

    def names(self) -> tuple[str, ...]:
        return tuple(self._entries.keys())

    def schema_of(self, name: str) -> type | None:
        entry = self._entries.get(name)
        return entry.params_schema if entry else None

    def build(self, defn: StageDef) -> Stage:
        try:
            entry = self._entries[defn.name]
        except KeyError as e:
            raise KeyError(f"Stage {defn.name!r} is not registered") from e
        params: Any = defn.params
        if entry.params_schema is not None:
            params = entry.params_schema(**dict(defn.params))
        return entry.factory(params)


DEFAULT_REGISTRY = StageRegistry()
"""Module-level default registry. Stage modules import and register
themselves on this instance. Tests can construct fresh registries."""
