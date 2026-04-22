"""Preprocessing package — punctuation restoration, sentence splitting, chunking.

Provides implementations conforming to the :data:`~ports.apply_fn.ApplyFn`
signature used by ``Subtitle.transform()``.

All heavy dependencies (``deepmultilingualpunctuation``, ``spacy``) are
optional — availability guards in :mod:`.availability` follow the same
pattern as :mod:`domain.lang._core._availability`.
"""

from adapters.preprocess.availability import (
    langdetect_is_available,
    punc_model_is_available,
    spacy_is_available,
)
from adapters.preprocess.chunk import ChunkBackendRegistry, Chunker
from adapters.preprocess.punc import (
    Backend,
    BackendFactory,
    BackendSpec,
    PuncBackendRegistry,
    PuncRestorer,
    resolve_backend_spec,
)
from ports.apply_fn import ApplyFn

__all__ = [
    "ApplyFn",
    "Backend",
    "BackendFactory",
    "BackendSpec",
    "ChunkBackendRegistry",
    "Chunker",
    "PuncBackendRegistry",
    "PuncRestorer",
    "langdetect_is_available",
    "punc_model_is_available",
    "resolve_backend_spec",
    "spacy_is_available",
]
