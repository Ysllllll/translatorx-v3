"""Chunking / sentence-splitting adapters — registry + unified chunker.

Mirrors :mod:`adapters.preprocess.punc`: a single :class:`Chunker`
dispatches per language to a registered backend
(``Callable[[list[str]], list[list[str]]]``). Built-in backend factories
are registered at import time:

=============  ===================================================
``"rule"``     :func:`~adapters.preprocess.chunk.backends.rule.rule_backend`
``"spacy"``    :func:`~adapters.preprocess.chunk.backends.spacy.spacy_backend` (if ``spacy`` installed)
``"llm"``      :func:`~adapters.preprocess.chunk.backends.llm.llm_backend`
``"composite"``:func:`~adapters.preprocess.chunk.backends.composite.composite_backend`
=============  ===================================================

Adding a new splitter library means creating a new file under
:mod:`adapters.preprocess.chunk.backends` with a ``@register`` decorator —
no changes to :class:`Chunker` itself.
"""

from adapters.preprocess.chunk import backends as _backends  # noqa: F401  (registers factories)
from adapters.preprocess.chunk.chunker import Chunker
from adapters.preprocess.chunk.reconstruct import chunks_match_source
from adapters.preprocess.chunk.registry import (
    Backend,
    BackendFactory,
    BackendSpec,
    ChunkBackendRegistry,
    resolve_backend_spec,
)

__all__ = [
    "Backend",
    "BackendFactory",
    "BackendSpec",
    "ChunkBackendRegistry",
    "Chunker",
    "chunks_match_source",
    "resolve_backend_spec",
]
