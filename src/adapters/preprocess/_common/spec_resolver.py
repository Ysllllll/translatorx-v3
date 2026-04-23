"""Generic per-language backend-spec router with lazy resolution.

Shared between :class:`~adapters.preprocess.chunk.chunker.Chunker` and
:class:`~adapters.preprocess.punc.restorer.PuncRestorer`. Both store a
``{language → spec}`` mapping with a ``"*"`` wildcard fallback and resolve
each entry to a concrete backend the first time it is requested. The
resolver caches each resolved backend so subsequent calls for the same
language reuse the already-constructed instance.
"""

from __future__ import annotations

import threading
from typing import Callable, Generic, Hashable, Mapping, TypeVar

S = TypeVar("S")
B = TypeVar("B")

WILDCARD = "*"


class BackendSpecResolver(Generic[S, B]):
    """Lookup + cache helper for per-language backend specs.

    Parameters
    ----------
    specs:
        ``{language → spec}`` mapping. Use :data:`WILDCARD` as a catch-all.
    resolve_fn:
        Function that turns a spec into a concrete backend. Called at
        most once per language.
    """

    def __init__(
        self,
        specs: Mapping[str, S] | None,
        resolve_fn: Callable[[S], B],
    ) -> None:
        self._specs: dict[str, S] = dict(specs or {})
        self._resolve_fn = resolve_fn
        self._resolved: dict[str, B] = {}
        self._lock = threading.Lock()

    @property
    def specs(self) -> Mapping[str, S]:
        return self._specs

    def lookup(self, language: Hashable) -> S:
        if language in self._specs:
            return self._specs[language]  # type: ignore[index]
        if WILDCARD in self._specs:
            return self._specs[WILDCARD]
        raise KeyError(f"No backend configured for language {language!r} and no wildcard {WILDCARD!r} fallback provided")

    def resolve(self, language: str) -> B:
        cached = self._resolved.get(language)
        if cached is not None:
            return cached
        with self._lock:
            cached = self._resolved.get(language)
            if cached is not None:
                return cached
            backend = self._resolve_fn(self.lookup(language))
            self._resolved[language] = backend
            return backend


__all__ = ["BackendSpecResolver", "WILDCARD"]
