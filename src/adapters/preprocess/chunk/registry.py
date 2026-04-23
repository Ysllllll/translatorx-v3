"""Registry primitives for chunk backends.

A **backend** is a plain ``Callable[[list[str]], list[list[str]]]`` —
given a batch of raw texts, returns a list of chunk lists (one per
input). This matches :data:`~ports.apply_fn.ApplyFn` exactly, so a
backend *is* an ``ApplyFn``.

Libraries register a *factory* under a name; users reference that name
(in Python code or YAML config) to select it, and the orchestrator
:class:`~adapters.preprocess.chunk.chunker.Chunker` routes per-language
requests through it.

The registry class and ``resolve_backend_spec`` helper are thin wrappers
around the generic primitives in :mod:`adapters.preprocess._common.registry`.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Union

from adapters.preprocess._common.registry import BackendRegistry, resolve_spec

#: Pure backend contract — takes a batch of raw texts, returns one chunk
#: list per text (``len(output) == len(input)``). Order preserved.
#: Backends may raise on irrecoverable failure; :class:`Chunker` catches
#: and applies the configured failure policy.
Backend = Callable[[list[str]], list[list[str]]]

#: Factories build a :data:`Backend` from keyword configuration.
BackendFactory = Callable[..., Backend]

#: Shapes accepted in ``backends`` mappings:
#:
#: * ``Callable`` — used as-is (anything satisfying :data:`Backend`).
#: * ``Mapping``  — ``{"library": str, **kwargs}`` routed through the
#:   registry.
BackendSpec = Union[Backend, Mapping[str, Any]]


class ChunkBackendRegistry(BackendRegistry[Backend]):
    """Process-wide registry of named chunk-backend factories.

    Thread-safe for both registration and lookup. Mirrors
    :class:`~adapters.preprocess.punc.registry.PuncBackendRegistry`.
    """

    kind = "Chunk"


def resolve_backend_spec(spec: BackendSpec) -> Backend:
    """Normalize a :data:`BackendSpec` into a plain :data:`Backend`.

    * ``Mapping`` → ``ChunkBackendRegistry.create(mapping["library"], **rest)``
    * ``Callable`` → returned as-is
    """
    return resolve_spec(
        spec,
        ChunkBackendRegistry,
        expected_signature="Callable[[list[str]], list[list[str]]]",
    )


__all__ = [
    "Backend",
    "BackendFactory",
    "BackendSpec",
    "ChunkBackendRegistry",
    "resolve_backend_spec",
]
