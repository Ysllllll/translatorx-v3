"""Preprocessing package — punctuation restoration, sentence splitting, chunking.

Provides implementations conforming to the ``ApplyFn`` signature used by
``Subtitle.transform()``.

All heavy dependencies (``deepmultilingualpunctuation``, ``spacy``) are
optional — availability guards in ``_availability.py`` follow the same
pattern as ``lang_ops._core._availability``.
"""

from adapters.preprocess._availability import (
    langdetect_is_available,
    punc_model_is_available,
    spacy_is_available,
)
from adapters.preprocess._chunk import LlmChunker
from adapters.preprocess._llm_punc import LlmPuncRestorer
from adapters.preprocess._protocol import ApplyFn
from adapters.preprocess._remote_punc import RemotePuncRestorer

# Conditional imports for heavy optional deps.
_ner_available = punc_model_is_available()
_spacy_available = spacy_is_available()

__all__ = [
    "ApplyFn",
    "LlmChunker",
    "LlmPuncRestorer",
    "RemotePuncRestorer",
    "langdetect_is_available",
    "punc_model_is_available",
    "spacy_is_available",
]

if _ner_available:
    from adapters.preprocess._ner_punc import NerPuncRestorer

    __all__.append("NerPuncRestorer")

if _spacy_available:
    from adapters.preprocess._spacy import SpacySplitter
    from adapters.preprocess._spacy_llm_chunk import SpacyLlmChunker

    __all__ += ["SpacySplitter", "SpacyLlmChunker"]
