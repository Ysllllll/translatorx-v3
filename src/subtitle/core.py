"""Subtitle — chainable subtitle segment restructuring.

Thin wrapper that delegates text operations to
:class:`~lang_ops.chunk.ChunkPipeline` and word alignment to
:func:`~subtitle.align.align_segments`.

Typical usage::

    from subtitle import Subtitle

    sub = Subtitle(segments, language="zh")
    result = sub.sentences().split(40).build()
    records = sub.sentences().split(40).records()

With external text transforms (e.g. punctuation restoration)::

    records = (sub
        .sentences()
        .apply(restore_fn)       # fn: list[str] → list[list[str]]
        .sentences()             # re-split after text change
        .split(40)
        .records())
"""

from __future__ import annotations

from collections.abc import Callable, MutableMapping
from typing import TYPE_CHECKING

from model import Segment, SentenceRecord, Word
from .align import fill_words, align_segments, distribute_words

if TYPE_CHECKING:
    from lang_ops._core._base_ops import _BaseOps
    from lang_ops.chunk._pipeline import ApplyFn, ApplyCache


def _extract(
    segments: list[Segment],
    ops: _BaseOps,
) -> tuple[list[Word], str]:
    """Extract all words and build joined text from segments.

    Segments without words are auto-filled via :func:`fill_words`.
    """
    all_words: list[Word] = []
    texts: list[str] = []

    for seg in segments:
        filled = seg if seg.words else fill_words(seg, split_fn=ops.split)
        all_words.extend(filled.words)
        texts.append(seg.text)

    full_text = ops.join(texts)
    return all_words, full_text


def _speaker_chunks(all_words: list[Word], ops: _BaseOps) -> list[str]:
    """Group words by speaker runs, return text chunk per run."""
    if not all_words:
        return []

    chunks: list[str] = []
    run: list[Word] = [all_words[0]]

    for w in all_words[1:]:
        if w.speaker != run[0].speaker:
            chunks.append(ops.join([r.word for r in run]))
            run = [w]
        else:
            run.append(w)

    chunks.append(ops.join([r.word for r in run]))
    return chunks


class Subtitle:
    """Immutable chainable processor for restructuring subtitle segments.

    After ``sentences()``, the instance holds one pipeline per sentence
    with its corresponding words.  Subsequent operations (``clauses``,
    ``split``, ``apply``) are applied per-sentence, so they never
    cross sentence boundaries.

    All text operations are delegated to
    :class:`~lang_ops.chunk.ChunkPipeline` which tokenizes once and
    operates on the token array.
    """

    __slots__ = ("_ops", "_pipelines", "_words")

    # ---- construction ------------------------------------------------

    def __init__(
        self,
        segments: list[Segment],
        ops: _BaseOps | None = None,
        *,
        language: str | None = None,
        split_by_speaker: bool = False,
    ) -> None:
        from lang_ops import LangOps
        from lang_ops.chunk._pipeline import ChunkPipeline

        if ops is not None:
            self._ops = ops
        elif language is not None:
            self._ops = LangOps.for_language(language)
        else:
            raise TypeError("Subtitle requires either ops or language")

        all_words, full_text = _extract(segments, self._ops)

        if split_by_speaker:
            chunks = _speaker_chunks(all_words, self._ops)
            pipeline = ChunkPipeline.from_chunks(chunks, ops=self._ops)
        else:
            pipeline = ChunkPipeline(full_text, ops=self._ops)

        self._pipelines: list[ChunkPipeline] = [pipeline]
        self._words: list[list[Word]] = [all_words]

    @classmethod
    def from_words(
        cls,
        words: list[Word],
        ops: _BaseOps | None = None,
        *,
        language: str | None = None,
    ) -> Subtitle:
        """Create from a flat word list (e.g. WhisperX output)."""
        if not words:
            if ops is not None:
                return cls([], ops=ops)
            return cls([], language=language)
        text = "".join(w.word for w in words)
        seg = Segment(
            start=words[0].start, end=words[-1].end,
            text=text, words=words,
        )
        if ops is not None:
            return cls([seg], ops=ops)
        return cls([seg], language=language)

    def _with_pipelines(
        self,
        pipelines: list[object],
        words: list[list[Word]] | None = None,
    ) -> Subtitle:
        """Create a new Subtitle with updated pipelines (and optionally words)."""
        new = object.__new__(Subtitle)
        new._ops = self._ops
        new._pipelines = pipelines
        new._words = words if words is not None else self._words
        return new

    # ---- text operations (delegated to ChunkPipeline) ----------------

    def sentences(self) -> Subtitle:
        """Split each chunk into sentences.

        After this call, each pipeline holds exactly one sentence and
        words are distributed to their respective sentences (early
        alignment).  Subsequent operations (``clauses``, ``split``,
        ``merge``, ``apply``) are applied per-sentence — they never
        cross sentence boundaries.

        This is typically the first operation in a chain::

            sub.sentences().clauses(merge_under=60).split(40).build()
        """
        from lang_ops.chunk._pipeline import ChunkPipeline

        new_pipelines: list[ChunkPipeline] = []
        new_words: list[list[Word]] = []

        for pipeline, words in zip(self._pipelines, self._words):
            sent_pipeline = pipeline.sentences()
            sent_texts = sent_pipeline.result()

            if len(sent_texts) <= 1:
                # No actual split — keep original pipeline and words
                new_pipelines.append(sent_pipeline)
                new_words.append(words)
            else:
                word_groups = distribute_words(words, sent_texts)
                # Each sentence gets its own pipeline from pre-tokenized groups
                for group, wg in zip(sent_pipeline._groups, word_groups):  # noqa: SLF001
                    new_pipelines.append(
                        ChunkPipeline._from_groups([group], self._ops)  # noqa: SLF001
                    )
                    new_words.append(wg)

        return self._with_pipelines(new_pipelines, new_words)

    def clauses(self, merge_under: int | None = None) -> Subtitle:
        """Split each chunk into clauses (sentence-aware).

        Args:
            merge_under: If given, merge back clauses shorter than this.
        """
        return self._with_pipelines(
            [p.clauses(merge_under=merge_under) for p in self._pipelines]
        )

    def split(self, max_len: int) -> Subtitle:
        """Split each chunk by length.

        Args:
            max_len: Upper bound on chunk length.
        """
        return self._with_pipelines(
            [p.split(max_len) for p in self._pipelines]
        )

    def merge(self, max_len: int) -> Subtitle:
        """Greedily merge adjacent chunks within each pipeline.

        Merging is per-pipeline (i.e. per-sentence after ``sentences()``).
        """
        return self._with_pipelines(
            [p.merge(max_len) for p in self._pipelines]
        )

    def apply(
        self,
        fn: ApplyFn,
        cache: ApplyCache | None = None,
        batch_size: int = 1,
        workers: int = 1,
        skip_if: Callable[[str], bool] | None = None,
    ) -> Subtitle:
        """Apply an external function to each chunk.

        *fn* receives a batch of texts and returns one ``list[str]`` per
        input text:

        - ``["new text"]`` → 1:1 replacement (e.g. punctuation restoration)
        - ``["part1", "part2"]`` → 1:N splitting (e.g. NLP/LLM splitting)
        - ``[]`` → deletion

        Texts from all pipelines are collected and dispatched to *fn*
        together (respecting *batch_size*), then results are distributed
        back to their respective pipelines.

        See :meth:`ChunkPipeline.apply` for full parameter docs.
        """
        from lang_ops.chunk._pipeline import ChunkPipeline, _call_apply_fn

        # Collect all texts across pipelines
        pipeline_texts: list[list[str]] = [p.result() for p in self._pipelines]
        all_texts: list[str] = []
        for texts in pipeline_texts:
            all_texts.extend(texts)

        if not all_texts:
            return self

        # --- resolve from cache and skip_if ---
        all_results: list[list[str] | None] = [None] * len(all_texts)
        miss_indices: list[int] = []
        miss_texts: list[str] = []

        for idx, text in enumerate(all_texts):
            if skip_if is not None and skip_if(text):
                all_results[idx] = [text]
            elif cache is not None and text in cache:
                all_results[idx] = cache[text]
            else:
                miss_indices.append(idx)
                miss_texts.append(text)

        # --- call fn for cache misses ---
        if miss_texts:
            miss_results = _call_apply_fn(fn, miss_texts, batch_size, workers)
            for mi, result_list in zip(miss_indices, miss_results):
                all_results[mi] = result_list
                if cache is not None:
                    cache[all_texts[mi]] = result_list

        # --- distribute results back to per-pipeline groups ---
        new_pipelines: list[ChunkPipeline] = []
        offset = 0
        for i, texts in enumerate(pipeline_texts):
            n = len(texts)
            pipeline_results = all_results[offset:offset + n]
            offset += n

            new_groups: list[list[str]] = []
            ops = self._pipelines[i]._ops  # noqa: SLF001
            for parts in pipeline_results:
                assert parts is not None
                for part in parts:
                    tokens = ops.split(part)
                    if tokens:
                        new_groups.append(tokens)

            new_pipelines.append(ChunkPipeline._from_groups(new_groups, ops))  # noqa: SLF001

        return self._with_pipelines(new_pipelines)

    # ---- output ------------------------------------------------------

    def build(self) -> list[Segment]:
        """Align current chunks with words and return Segments."""
        result: list[Segment] = []
        for pipeline, words in zip(self._pipelines, self._words):
            chunks = pipeline.result()
            result.extend(align_segments(chunks, words))
        return result

    def records(self) -> list[SentenceRecord]:
        """Return SentenceRecords: one per sentence, with sub-segments.

        If ``sentences()`` has not been called, it is done automatically.

        Each record's ``src_text`` is a full sentence.  Use chained
        operations (``clauses``, ``split``) before calling ``records()``
        to control segment granularity::

            sub.sentences().clauses(merge_under=60).split(40).records()
        """
        # Ensure we're at sentence granularity
        sub = self if len(self._pipelines) > 1 or not self._pipelines else self.sentences()

        records: list[SentenceRecord] = []
        for pipeline, words in zip(sub._pipelines, sub._words):
            src_text = sub._ops.join(
                [sub._ops.join(g) for g in pipeline._groups]  # noqa: SLF001
            )
            if not src_text.strip():
                continue

            sub_chunks = pipeline.result()

            sub_segments = align_segments(sub_chunks, words)
            if sub_segments:
                records.append(SentenceRecord(
                    src_text=src_text,
                    start=sub_segments[0].start,
                    end=sub_segments[-1].end,
                    segments=sub_segments,
                ))

        return records

    # ---- stream mode -------------------------------------------------

    @classmethod
    def stream(
        cls,
        ops: _BaseOps | None = None,
        *,
        language: str | None = None,
        split_by_speaker: bool = False,
    ) -> SubtitleStream:
        """Create a streaming processor that accepts segments one at a time."""
        from lang_ops import LangOps
        if ops is not None:
            resolved_ops = ops
        elif language is not None:
            resolved_ops = LangOps.for_language(language)
        else:
            raise TypeError("stream() requires either ops or language")
        return SubtitleStream(resolved_ops, split_by_speaker=split_by_speaker)


class SubtitleStream:
    """Stateful streaming segment restructurer.

    Buffers only the last incomplete sentence.  On each :meth:`feed`,
    combines the incomplete sentence with the new segment, splits
    sentences, emits completed ones, and keeps only the trailing
    incomplete part.  This is O(incomplete + new) per call rather
    than O(total) — a significant improvement for long streams.

    :meth:`flush` emits everything remaining.
    """

    __slots__ = ("_ops", "_split_by_speaker", "_incomplete")

    def __init__(
        self,
        ops: _BaseOps,
        *,
        split_by_speaker: bool = False,
    ) -> None:
        self._ops = ops
        self._split_by_speaker = split_by_speaker
        self._incomplete: Segment | None = None

    def feed(self, segment: Segment) -> list[Segment]:
        """Feed one segment; return any completed sentence-segments."""
        if self._incomplete is not None:
            segs = [self._incomplete, segment]
        else:
            segs = [segment]

        sub = Subtitle(
            segs, self._ops,
            split_by_speaker=self._split_by_speaker,
        )
        all_sentences = sub.sentences().build()

        if len(all_sentences) <= 1:
            # No confirmed complete sentences yet
            self._incomplete = all_sentences[0] if all_sentences else None
            return []

        # All but the last sentence are confirmed complete
        done = all_sentences[:-1]
        last = all_sentences[-1]
        self._incomplete = last if last.words else None
        return done

    def flush(self) -> list[Segment]:
        """Flush remaining buffer as sentence-segments."""
        if self._incomplete is None:
            return []

        sub = Subtitle(
            [self._incomplete], self._ops,
            split_by_speaker=self._split_by_speaker,
        )
        result = sub.sentences().build()
        self._incomplete = None
        return result

    # ---- SentenceRecord streaming (for translation pipelines) --------

    def feed_records(self, segment: Segment) -> list[SentenceRecord]:
        """Feed one segment; return completed sentences as SentenceRecords.

        Each emitted record represents one complete sentence with a
        single sub-segment.  Designed to feed :class:`StreamAdapter`.
        """
        return [_segment_to_record(s) for s in self.feed(segment)]

    def flush_records(self) -> list[SentenceRecord]:
        """Flush remaining buffer as SentenceRecords."""
        return [_segment_to_record(s) for s in self.flush()]


def _segment_to_record(seg: Segment) -> SentenceRecord:
    """Wrap a completed sentence-segment as a SentenceRecord."""
    return SentenceRecord(
        src_text=seg.text,
        start=seg.start,
        end=seg.end,
        segments=[seg],
    )
