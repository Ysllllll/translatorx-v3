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
from adapters.preprocess.llm_chunk import LlmChunker
from adapters.preprocess.punc import (
    Backend,
    BackendFactory,
    BackendSpec,
    PuncBackendRegistry,
    PuncRestorer,
    resolve_backend_spec,
)
from ports.apply_fn import ApplyFn

_spacy_available = spacy_is_available()

__all__ = [
    "ApplyFn",
    "Backend",
    "BackendFactory",
    "BackendSpec",
    "LlmChunker",
    "PuncBackendRegistry",
    "PuncRestorer",
    "langdetect_is_available",
    "punc_model_is_available",
    "resolve_backend_spec",
    "spacy_is_available",
]

if _spacy_available:
    from adapters.preprocess.spacy_split import SpacySplitter
    from adapters.preprocess.spacy_llm_chunk import SpacyLlmChunker

    __all__ += ["SpacySplitter", "SpacyLlmChunker"]
