"""Processor / Source Protocols — the runtime's pure-function contract.

Design refs
-----------
* **D-001**: :class:`Processor` is a pure async-generator function over
  ``AsyncIterator[In] -> AsyncIterator[Out]``. Its job is to compute an
  output for each input item. It does **not** manage stream state.
* **D-002**: Stream state (id allocation, stale tracking, reprocess) lives
  in :class:`api.app.stream.LiveStreamHandle` (live mode) and
  :class:`application.pipeline.runtime.PipelineRuntime` (batch mode),
  not here.
* **D-003**: Processors expose an optional ``output_is_stale(rec)`` hook so
  the stream layer can ask "is this output obsolete?" without knowing the
  processor's internals.
* **D-020**: Caching is a processor-internal responsibility. Each processor
  reads the relevant namespace field off its input record (e.g.
  ``record.translations[lang]``) plus the video-level fingerprint from
  ``Store.meta._fingerprints[self.name]``; on hit it yields the record
  unchanged, on miss it computes, writes via ``store.patch_video`` in
  buffered batches (D-044 L1 — 100 records or 60s), and finally shields
  the final flush on cancel (D-045).
* **D-060**: Streaming orchestration adds :class:`Priority` + ``seek(t)``
  semantics on top of the pure Processor contract.
* **D-067**: Final data-flow summary — see
  ``files/processor-architecture-memo.md`` §D-067.

Everything in this module is a signature-only skeleton; implementations
land in Stage 3.2 and later stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import AsyncIterator, Protocol, TypeVar, runtime_checkable

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from application.translate import TranslationContext
    from adapters.storage.store import Store


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class VideoKey:
    """Opaque addressing key for a single video within a course.

    Passed to every :meth:`Processor.process` call so processors can
    issue ``store.patch_video(...)`` without hard-coding routing.
    The ``course`` field may contain ``/`` for natural sub-namespacing
    (e.g. ``"2025-09/MIT-6.5940"``) per D-041.
    """

    course: str
    video: str


class Priority(IntEnum):
    """Queue priority for :class:`api.app.stream.LiveStreamHandle` (D-060).

    Lower numeric value = higher priority (IntEnum ordering). ``HIGH``
    is reserved for user-visible interactions such as *seek* targets in
    the browser-plugin scenario; ``NORMAL`` is the default for live
    streaming feed; ``LOW`` for speculative prefetch.
    """

    HIGH = 0
    NORMAL = 10
    LOW = 20


# ---------------------------------------------------------------------------
# Source Protocol
# ---------------------------------------------------------------------------


Out_co = TypeVar("Out_co", covariant=True)


@runtime_checkable
class Source(Protocol[Out_co]):
    """Yields raw items into the front of a processor chain.

    Typical implementations: SRT reader, WhisperX JSON reader, live
    subtitle feeder. For resume scenarios, the source is responsible
    for emitting the full record list (including already-translated
    ones) — downstream processors' hit-checks will skip them cheaply.

    C1 — :meth:`aclose` mirrors the Processor contract; orchestrators
    call it in ``finally`` so file handles / HTTP sessions / temp dirs
    can be released. Must be idempotent.
    """

    async def read(self) -> AsyncIterator[Out_co]:
        """Yield items in order. One-shot; not re-iterable."""
        ...

    async def aclose(self) -> None:
        """Release source-side resources. Idempotent."""
        ...


# ---------------------------------------------------------------------------
# Processor Protocol
# ---------------------------------------------------------------------------


In_contra = TypeVar("In_contra", contravariant=True)


@runtime_checkable
class Processor(Protocol[In_contra, Out_co]):
    """Pure async-generator transformer over records (D-001 / D-067).

    **Contract**

    * ``name`` is a short identifier used as the key in
      ``Store.meta._fingerprints[name]`` and in log/progress events.
    * ``fingerprint()`` returns a stable ``sha256`` hex digest derived
      from ``(engine_id, model, prompt_template, relevant_config)``
      per D-043 R4. It is computed once at construction time and reused
      for every hit check within the run.
    * ``process(upstream, *, ctx, store, video_key)`` is an async
      generator that consumes ``upstream`` and yields enriched records.
      The processor is expected to:

      1. On each record, check its own namespace field plus
         ``store.meta._fingerprints[self.name] == self.fingerprint()``.
         If both present and matching → hit (optionally update internal
         state such as :class:`ContextWindow`) → yield the record
         unchanged.
      2. Else → compute → return a *replaced* record with the new field
         filled (processor-specific markers such as
         ``extra["terms_ready_at_translate"]`` for TranslateProcessor) →
         append to an internal flush buffer → emit
         ``store.patch_video(...)`` when the buffer reaches 100 entries
         or 60s elapsed (D-044 L1) → yield the replaced record.
      3. In ``finally``, ``await asyncio.shield(self._flush())`` and
         ``await asyncio.shield(self.aclose())`` (D-045).

    * ``output_is_stale(rec)`` is the hook :class:`RecordStream` calls
      when building ``stale_ids``. Default semantics (Stage 3.2 base
      class) is ``return False`` — only processors that participate in
      the one-shot TermsProvider ready-transition override it. For
      :class:`TranslateProcessor` the canonical predicate is
      ``terms_provider.ready and not rec.extra.get(
      "terms_ready_at_translate", False)``. Per Phase 2.1 the terms
      state machine has exactly two states (False→True once), so no
      integer version counter is needed.
    * ``aclose()`` releases resources (HTTP sessions, temp files). Must
      be idempotent; framework invokes it once per run in ``finally``.
    """

    name: str

    def fingerprint(self) -> str:
        """Stable configuration digest; computed once at init time."""
        ...

    def process(
        self,
        upstream: AsyncIterator[In_contra],
        *,
        ctx: "TranslationContext",
        store: "Store",
        video_key: VideoKey,
    ) -> AsyncIterator[Out_co]:
        """Transform ``upstream`` records; see class docstring for contract."""
        ...

    def output_is_stale(self, rec: Out_co) -> bool:
        """Return ``True`` if ``rec`` should be reprocessed (D-003)."""
        ...

    async def aclose(self) -> None:
        """Release resources. Idempotent; called in ``finally``."""
        ...


__all__ = [
    "Priority",
    "Processor",
    "Source",
    "VideoKey",
]
