"""ProcessorBase — abstract base class for all runtime processors.

Design refs
-----------
* **D-001**: Processor is pure — no user/engine state beyond what the
  engine injected at construction time. All mutable run state lives in
  :class:`Store`.
* **D-040**: Missing-input handling. Subclasses declare their required
  inputs (e.g. a target-language translation for AlignProcessor); the
  :meth:`_missing_inputs` helper returns the list of missing fields and
  :meth:`_record_with_error` emits a ``degraded`` :class:`ErrorInfo`.
* **D-068**: ``output_is_stale`` default is ``return False``. Only
  :class:`TranslateProcessor` overrides it to react to the one-shot
  :class:`TermsProvider` ready transition. There is no integer
  ``ctx.version`` — stale detection uses the
  ``rec.extra["terms_ready_at_translate"]`` bool.

Concrete processors
-------------------
Subclasses must:

* set ``name: str`` (class attr)
* implement :meth:`fingerprint` (pure function over the processor's
  configuration — engine id + model + prompt template + relevant cfg)
* implement :meth:`process` as an ``async def`` generator function

Optional overrides:

* :meth:`output_is_stale` — only if the processor's output becomes stale
  due to a side-channel ctx update (currently only TranslateProcessor).
* :meth:`aclose` — release resources; framework awaits once in ``finally``.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import replace
from typing import TYPE_CHECKING, AsyncIterator, Generic, TypeVar

from model import SentenceRecord

from .errors import ErrorCategory, ErrorInfo

if TYPE_CHECKING:
    from llm_ops import TranslationContext

    from .protocol import VideoKey
    from .store import Store


In = TypeVar("In")
Out = TypeVar("Out")


class ProcessorBase(ABC, Generic[In, Out]):
    """Abstract base for runtime processors (D-001, D-040, D-068).

    Subclasses implement :meth:`process` as an async generator that
    transforms an upstream record iterator. The base class provides
    shared helpers for missing-input detection and structured error
    attachment to records.
    """

    name: str = ""
    """Short identifier used for ``store.meta._fingerprints`` keys,
    progress events, and error reporting. Subclass must set."""

    # ------------------------------------------------------------------
    # Required hooks
    # ------------------------------------------------------------------

    @abstractmethod
    def fingerprint(self) -> str:
        """Return a stable SHA-256 hex digest of the processor config.

        Includes at minimum ``(engine_id, model, prompt_template,
        relevant_config)``. Computed once at init time; reused for every
        cache-hit check (D-043 R4).
        """

    @abstractmethod
    def process(
        self,
        upstream: AsyncIterator[In],
        *,
        ctx: "TranslationContext",
        store: "Store",
        video_key: "VideoKey",
    ) -> AsyncIterator[Out]:
        """Transform ``upstream`` records, yielding enriched outputs.

        Declared as a plain method returning ``AsyncIterator`` so that
        subclasses can implement it as either an ``async def`` generator
        or a coroutine that returns an iterator. The contract (buffered
        flush, finally-shielded cleanup, fingerprint-gated hit path) is
        described in :mod:`runtime.protocol`.
        """

    # ------------------------------------------------------------------
    # Default implementations
    # ------------------------------------------------------------------

    def output_is_stale(self, rec: Out) -> bool:
        """Default: never stale.

        Only :class:`TranslateProcessor` overrides this to check the
        ``rec.extra["terms_ready_at_translate"]`` flag against the
        current ``TermsProvider.ready`` state (D-068).
        """
        return False

    async def aclose(self) -> None:
        """Release resources. Override if the processor holds sockets,
        temp files, etc. Must be idempotent (D-045)."""
        return None

    # ------------------------------------------------------------------
    # Shared helpers (D-040)
    # ------------------------------------------------------------------

    def _missing_inputs(
        self,
        rec: SentenceRecord,
        *,
        required_translations: tuple[str, ...] = (),
        required_extra: tuple[str, ...] = (),
    ) -> list[str]:
        """Return the list of missing input fields for ``rec``.

        Args:
            rec: The record to inspect.
            required_translations: Target-language keys that must be
                present in ``rec.translations``.
            required_extra: Keys that must be present in ``rec.extra``.

        Returns:
            A list of dotted-path labels for missing fields (empty if
            all present). Labels use the ``translations[xx]`` /
            ``extra[yy]`` style so they can be dropped into an
            :class:`ErrorInfo.message`.
        """
        missing: list[str] = []
        for lang in required_translations:
            if lang not in rec.translations or not rec.translations[lang]:
                missing.append(f"translations[{lang}]")
        for key in required_extra:
            if key not in rec.extra:
                missing.append(f"extra[{key}]")
        return missing

    def _record_with_error(
        self,
        rec: SentenceRecord,
        *,
        category: ErrorCategory,
        code: str,
        message: str,
        retryable: bool = False,
        attempts: int = 0,
        cause: BaseException | str | None = None,
    ) -> SentenceRecord:
        """Return a new record with an :class:`ErrorInfo` appended.

        Errors accumulate in ``rec.extra["errors"]`` as an immutable
        list. The reporter hook (D-038) is the caller's responsibility;
        this helper only updates the record.
        """
        cause_str: str | None
        if cause is None:
            cause_str = None
        elif isinstance(cause, BaseException):
            cause_str = f"{type(cause).__name__}: {cause}"
        else:
            cause_str = str(cause)

        err = ErrorInfo(
            processor=self.name,
            category=category,
            code=code,
            message=message,
            retryable=retryable,
            attempts=attempts,
            at=time.time(),
            cause=cause_str,
        )
        errors = list(rec.extra.get("errors", []))
        errors.append(err)
        new_extra = {**rec.extra, "errors": errors}
        return replace(rec, extra=new_extra)


__all__ = ["ProcessorBase"]
