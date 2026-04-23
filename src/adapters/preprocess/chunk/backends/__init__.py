"""Chunk backends package.

Importing this package registers all built-in backends with
:class:`~adapters.preprocess.chunk.registry.ChunkBackendRegistry`.

Registered names:

* ``"rule"`` — deterministic length-based splitter using
  :meth:`LangOps.split_by_length`. Always available.
* ``"spacy"`` — spaCy sentence splitter (requires ``spacy`` + a model).
* ``"llm"`` — LLM-driven recursive splitter (requires an engine).
* ``"remote"`` — HTTP-backed splitter calling a self-hosted service.
* ``"composite"`` — composite of two backends (coarse → refine).
"""

from __future__ import annotations

from adapters.preprocess.chunk.backends.composite import composite_backend  # noqa: F401
from adapters.preprocess.chunk.backends.llm import llm_backend  # noqa: F401
from adapters.preprocess.chunk.backends.remote import remote_backend  # noqa: F401
from adapters.preprocess.chunk.backends.rule import rule_backend  # noqa: F401
from adapters.preprocess.chunk.backends.connective import (  # noqa: F401
    pos_connective_backend,
    rule_connective_backend,
)

try:
    from adapters.preprocess.chunk.backends.spacy import spacy_backend  # noqa: F401
except ImportError:  # pragma: no cover - optional dependency
    pass


__all__ = [
    "composite_backend",
    "llm_backend",
    "remote_backend",
    "rule_backend",
    "rule_connective_backend",
    "pos_connective_backend",
]
