"""StreamAdapter — streaming translation with terms-stale tracking.

Wraps :func:`translate_node._translate_one` for a continuous, per-record
flow (as produced by :meth:`SubtitleStream.feed_records`).  Each fed
record is translated immediately and returned.  The adapter:

- Triggers :meth:`TermsProvider.request_generation` on every feed so
  OneShotTerms can accumulate text / fire at threshold.
- Tags records translated before terms were ready as *stale*, exposing
  them via :attr:`stale_record_ids` so the App layer can decide whether
  and when to retranslate.
- Exposes :meth:`retranslate` to re-run translation for any subset of
  records (e.g. after terms become ready).

Unlike :class:`Pipeline`, StreamAdapter is **stateful** — it owns a
context window + record store across feeds.  One adapter per streaming
session.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field, replace
from typing import Iterable

from llm_ops import (
    Checker,
    ContextWindow,
    LLMEngine,
    TranslateResult,
    TranslationContext,
)
from model import SentenceRecord

from .config import TranslateNodeConfig
from .nodes import _translate_one
from .prefix import PrefixHandler

logger = logging.getLogger(__name__)


__all__ = ["FeedResult", "StreamAdapter"]


# Key used to stamp the adapter-assigned id on record.extra
STREAM_ID_KEY = "stream_id"


@dataclass(frozen=True)
class FeedResult:
    """Outcome of one :meth:`StreamAdapter.feed` call.

    Attributes:
        record: The translated SentenceRecord.  Its ``extra[STREAM_ID_KEY]``
            holds the adapter-assigned id.
        result: Detailed translate result (attempts, report, etc.).
        terms_ready: Whether the terms provider was ready at translate time.
            ``False`` means the record is now in :attr:`stale_record_ids`.
    """

    record: SentenceRecord
    result: TranslateResult
    terms_ready: bool


class StreamAdapter:
    """Streaming translation adapter.

    Thread-safety: single-threaded async only.  Instances are not safe
    to share across tasks without external locking.
    """

    __slots__ = (
        "_engine",
        "_context",
        "_checker",
        "_config",
        "_window",
        "_prefix_handler",
        "_direct_map",
        "_records",
        "_stale_ids",
        "_next_id",
        "_lock",
    )

    def __init__(
        self,
        engine: LLMEngine,
        context: TranslationContext,
        checker: Checker,
        *,
        config: TranslateNodeConfig | None = None,
    ) -> None:
        self._engine = engine
        self._context = context
        self._checker = checker
        self._config = config or TranslateNodeConfig()
        self._window = ContextWindow(context.window_size)
        self._prefix_handler = (
            PrefixHandler(self._config.prefix_rules)
            if self._config.prefix_rules else None
        )
        self._direct_map = {
            k.lower(): v for k, v in self._config.direct_translate.items()
        } if self._config.direct_translate else {}
        self._records: dict[int, SentenceRecord] = {}
        self._stale_ids: set[int] = set()
        self._next_id: int = 0
        # Serializes feed/retranslate so the shared window stays coherent.
        self._lock = asyncio.Lock()

    # ---- public API --------------------------------------------------

    async def feed(self, record: SentenceRecord) -> FeedResult:
        """Translate one record and return the result.

        Side-effects:
          - triggers ``context.terms_provider.request_generation`` with this
            record's source text (non-blocking for OneShotTerms).
          - if provider is not ready, tags the record id as stale.
        """
        async with self._lock:
            # Fire-and-forget terms accumulation.  For OneShotTerms this may
            # schedule a background task; for PreloadableTerms this is a no-op
            # if preload hasn't been called.
            try:
                await self._context.terms_provider.request_generation([record.src_text])
            except Exception:  # noqa: BLE001
                logger.exception("terms_provider.request_generation failed")

            rec_id = self._next_id
            self._next_id += 1

            # Stamp id in extra so downstream can correlate
            stamped = replace(record, extra={**record.extra, STREAM_ID_KEY: rec_id})

            new_record, result = await _translate_one(
                stamped,
                self._engine,
                self._context,
                self._checker,
                self._window,
                self._context.target_lang,
                self._config,
                self._prefix_handler,
                self._direct_map,
            )

            terms_ready = self._context.terms_provider.ready
            if not terms_ready:
                self._stale_ids.add(rec_id)

            self._records[rec_id] = new_record
            return FeedResult(record=new_record, result=result, terms_ready=terms_ready)

    async def retranslate(
        self,
        record_ids: Iterable[int],
    ) -> list[SentenceRecord]:
        """Re-translate the given record ids.

        Uses current terms_provider state (so if terms are now ready,
        the new translations benefit from them).  Successfully
        re-translated ids are removed from :attr:`stale_record_ids`.

        Unknown ids are silently skipped.  Returns the newly translated
        records in the order requested.
        """
        ids = [rid for rid in record_ids if rid in self._records]
        if not ids:
            return []

        async with self._lock:
            out: list[SentenceRecord] = []
            for rid in ids:
                original = self._records[rid]
                # Clear existing translation so _translate_one re-runs LLM.
                # Keep src_text, timings, segments, extra.
                cleared = replace(original, translations={})
                new_record, _ = await _translate_one(
                    cleared,
                    self._engine,
                    self._context,
                    self._checker,
                    self._window,
                    self._context.target_lang,
                    self._config,
                    self._prefix_handler,
                    self._direct_map,
                )
                self._records[rid] = new_record
                self._stale_ids.discard(rid)
                out.append(new_record)
            return out

    async def flush(self) -> list[SentenceRecord]:
        """Return all translated records in order.

        In the immediate-translate design there is no internal buffer
        of untranslated records, so this is equivalent to ``records()``.
        Kept for API symmetry with :class:`SubtitleStream.flush`.
        """
        return self.records()

    # ---- queries -----------------------------------------------------

    def records(self) -> list[SentenceRecord]:
        """Snapshot of all translated records (in feed order)."""
        return [self._records[i] for i in sorted(self._records)]

    @property
    def stale_record_ids(self) -> tuple[int, ...]:
        """Ids of records translated before terms became ready.

        The App layer decides whether to call :meth:`retranslate` with
        these ids (possibly filtered by playback position / lookback).
        """
        return tuple(sorted(self._stale_ids))

    @property
    def terms_ready(self) -> bool:
        """Whether the underlying terms provider is ready."""
        return self._context.terms_provider.ready
