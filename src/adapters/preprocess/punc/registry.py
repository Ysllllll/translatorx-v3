"""Registry primitives for punc backends.

A **backend** is a plain ``Callable[[list[str]], list[str]]`` — given a
batch of raw texts, returns a batch of punctuated texts (1:1). Libraries
register a *factory* under a name; users reference that name (in Python
code or YAML config) to select it.

The registry class and ``resolve_backend_spec`` helper are thin wrappers
around the generic primitives in :mod:`adapters.preprocess._common.registry`.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Union

from adapters.preprocess._common.registry import BackendRegistry, resolve_spec

#: Pure backend contract — takes a batch of raw texts, returns a batch
#: of punctuated texts (1:1, same length, same order). Backends may
#: raise on irrecoverable failure; :class:`PuncRestorer` catches and
#: applies the configured failure policy.
#:
#: The batch API lets LLM/remote backends parallelize requests
#: internally (e.g. ``asyncio.gather``) without imposing that concern
#: on the restorer itself.
Backend = Callable[[list[str]], list[str]]

#: Factories build a :data:`Backend` from keyword configuration.
BackendFactory = Callable[..., Backend]

#: Shapes accepted in ``backends`` mappings:
#:
#: * ``Callable`` — used as-is (anything satisfying :data:`Backend`).
#: * ``Mapping``  — ``{"library": str, **kwargs}`` routed through the
#:   registry.
BackendSpec = Union[Backend, Mapping[str, Any]]


class PuncBackendRegistry(BackendRegistry[Backend]):
    """Process-wide registry of named punc-backend factories.

    Thread-safe for both registration and lookup.
    """

    kind = "Punc"


def resolve_backend_spec(spec: BackendSpec) -> Backend:
    """Normalize a :data:`BackendSpec` into a plain :data:`Backend`.

    * ``Mapping`` → ``PuncBackendRegistry.create(mapping["library"], **rest)``
    * ``Callable`` → returned as-is

    Anything else raises :class:`TypeError`.
    """
    return resolve_spec(
        spec,
        PuncBackendRegistry,
        expected_signature="Callable[[list[str]], list[str]]",
    )


__all__ = [
    "Backend",
    "BackendFactory",
    "BackendSpec",
    "PuncBackendRegistry",
    "resolve_backend_spec",
]
