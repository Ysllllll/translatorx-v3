"""Punctuation restoration — registry + unified restorer.

A single :class:`PuncRestorer` class dispatches per language to a
registered backend (a plain ``(text) -> str`` callable). Each backend
library registers its own factory via :meth:`PuncBackendRegistry.register`;
adding a new library means creating a new file under
:mod:`~adapters.preprocess.punc.backends` with a ``@register`` decorator —
no changes to :class:`PuncRestorer` itself.

Shared cross-cutting concerns live in :class:`PuncRestorer`:

* threshold skip (uses :meth:`LangOps.length`, correct for mixed CJK/Latin)
* failure policy (``on_failure="keep"`` / ``"raise"``)
* content-change validation (:func:`punc_content_matches`)
* language-aware post-processing
  (:meth:`LangOps.protect_dotted_words`, :meth:`LangOps.preserve_trailing_punc`)

Backends only implement ``(text) -> str`` and may raise on failure; the
restorer catches and applies the configured policy.
"""

from adapters.preprocess.punc.registry import (
    Backend,
    BackendFactory,
    BackendSpec,
    PuncBackendRegistry,
    resolve_backend_spec,
)
from adapters.preprocess.punc.restorer import PuncRestorer

# Eagerly import built-in backend modules so their @register decorators run.
from adapters.preprocess.punc.backends import (  # noqa: F401  (side-effect import)
    deepmultilingualpunctuation as _deepml_backend,
    llm as _llm_backend,
    remote as _remote_backend,
)

__all__ = [
    "Backend",
    "BackendFactory",
    "BackendSpec",
    "PuncBackendRegistry",
    "PuncRestorer",
    "resolve_backend_spec",
]
