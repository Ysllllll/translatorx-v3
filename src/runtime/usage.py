"""Re-export of :class:`Usage` / :class:`CompletionResult` from the
:mod:`model` package (L0) to keep ``from runtime import Usage`` working.

Historically these types lived in ``runtime.usage``; they were promoted
to ``model.usage`` so :mod:`llm_ops` (L2) can import them without
depending on :mod:`runtime` (L3). See ``model/usage.py`` for the
implementation and D-048 design notes.
"""

from __future__ import annotations

from model.usage import CompletionResult, Usage

__all__ = ["CompletionResult", "Usage"]
