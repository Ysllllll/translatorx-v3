"""Shared internal helpers for ``chunk`` + ``punc`` adapter packages.

These are private implementation details — nothing here is exported at
the public ``adapters.preprocess`` surface. Extracted from code that was
duplicated verbatim between the two pipelines:

* :mod:`.async_bridge` — sync→async event-loop trampoline used by the
  LLM backends.
* :mod:`.spec_resolver` — generic language/wildcard routing with lazy,
  thread-safe backend resolution used by ``Chunker`` and ``PuncRestorer``.
* :mod:`.registry` — generic backend-factory registry base class used by
  ``ChunkBackendRegistry`` and ``PuncBackendRegistry``.
"""

from __future__ import annotations

from adapters.preprocess._common.async_bridge import run_async_in_sync
from adapters.preprocess._common.registry import BackendRegistry, resolve_spec
from adapters.preprocess._common.spec_resolver import BackendSpecResolver

__all__ = [
    "BackendRegistry",
    "BackendSpecResolver",
    "resolve_spec",
    "run_async_in_sync",
]
