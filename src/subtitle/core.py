"""Subtitle — chainable subtitle segment restructuring.

Thin wrapper that delegates text structuring to
:class:`~lang_ops.chunk.TextPipeline` and word alignment to
:func:`~subtitle.align.align_segments`.

Typical usage::

    from subtitle import Subtitle

    sub = Subtitle(segments, language="zh")
    result = sub.sentences().split(40).build()
    records = sub.sentences().split(40).records()

With external text transforms (e.g. punctuation restoration)::

    punc_cache = {}
    chunk_cache = {}
    records = (sub
        .transform(restore_fn, cache=punc_cache, scope="pipeline")
        .sentences()
        .transform(chunker, cache=chunk_cache)
        .records())
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Literal

from model import Segment, SentenceRecord, Word
from .align import fill_words, align_segments, distribute_words

if TYPE_CHECKING:
    from lang_ops._core._base_ops import _BaseOps
    from preprocess._protocol import ApplyFn


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


def _call_apply_fn(
    fn: ApplyFn,
    texts: list[str],
    batch_size: int,
    workers: int,
) -> list[list[str]]:
    """Dispatch *texts* to *fn* in batches, optionally in parallel."""
    if batch_size == 0:
        batches = [texts]
    else:
        batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]

    if workers <= 1 or len(batches) <= 1:
        batch_results = [fn(b) for b in batches]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            batch_results = list(pool.map(fn, batches))

    # Flatten batch results into a single list aligned with *texts*.
    result: list[list[str]] = []
    for br, batch in zip(batch_results, batches):
        if len(br) != len(batch):
            raise ValueError(f"apply fn returned {len(br)} results for a batch of {len(batch)} texts")
        result.extend(br)
    return result


class Subtitle:
    """Immutable chainable processor for restructuring subtitle segments.

    After ``sentences()``, the instance holds one pipeline per sentence
    with its corresponding words.  Subsequent operations (``clauses``,
    ``split``, ``transform``) are applied per-sentence, so they never
    cross sentence boundaries.

    All text structuring is delegated to
    :class:`~lang_ops.chunk.TextPipeline` which tokenizes once and
    operates on the token array.
    """

    __slots__ = ("_ops", "_pipelines", "_words", "_sentence_split")

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
        from lang_ops.chunk._pipeline import TextPipeline

        if ops is not None:
            self._ops = ops
        elif language is not None:
            self._ops = LangOps.for_language(language)
        else:
            raise TypeError("Subtitle requires either ops or language")

        all_words, full_text = _extract(segments, self._ops)

        # NOTE: _extract joins segment texts then TextPipeline re-tokenizes
        # the full_text.  This join→split round trip could be eliminated by
        # tokenizing each segment separately in _extract and using
        # _from_groups().  Kept as-is for simplicity — the overhead is
        # negligible and join() guarantees correct inter-segment spacing.
        if split_by_speaker:
            chunks = _speaker_chunks(all_words, self._ops)
            pipeline = TextPipeline.from_chunks(chunks, ops=self._ops)
        else:
            pipeline = TextPipeline(full_text, ops=self._ops)

        self._pipelines: list[TextPipeline] = [pipeline]
        self._words: list[list[Word]] = [all_words]
        self._sentence_split: bool = False

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
            start=words[0].start,
            end=words[-1].end,
            text=text,
            words=words,
        )
        if ops is not None:
            return cls([seg], ops=ops)
        return cls([seg], language=language)

    def _with_pipelines(
        self,
        pipelines: list[object],
        words: list[list[Word]] | None = None,
        sentence_split: bool | None = None,
    ) -> Subtitle:
        """Create a new Subtitle with updated pipelines (and optionally words)."""
        new = object.__new__(Subtitle)
        new._ops = self._ops
        new._pipelines = pipelines
        new._words = words if words is not None else self._words
        new._sentence_split = sentence_split if sentence_split is not None else self._sentence_split
        return new

    # ---- text structuring (delegated to TextPipeline) ----------------

    def sentences(self) -> Subtitle:
        """Split each chunk into sentences.

        After this call, each pipeline holds exactly one sentence and
        words are distributed to their respective sentences (early
        alignment).  Subsequent operations (``clauses``, ``split``,
        ``merge``, ``transform``) are applied per-sentence — they never
        cross sentence boundaries.

        This is typically the first operation in a chain::

            sub.sentences().clauses(merge_under=60).split(40).build()
        """
        from lang_ops.chunk._pipeline import TextPipeline

        new_pipelines: list[TextPipeline] = []
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
                        TextPipeline._from_groups([group], self._ops)  # noqa: SLF001
                    )
                    new_words.append(wg)

        return self._with_pipelines(new_pipelines, new_words, sentence_split=True)

    def clauses(self, merge_under: int | None = None) -> Subtitle:
        """Split each chunk into clauses (sentence-aware).

        Args:
            merge_under: If given, merge back clauses shorter than this.
        """
        return self._with_pipelines([p.clauses(merge_under=merge_under) for p in self._pipelines])

    def split(self, max_len: int) -> Subtitle:
        """Split each chunk by length.

        Args:
            max_len: Upper bound on chunk length.
        """
        return self._with_pipelines([p.split(max_len) for p in self._pipelines])

    def merge(self, max_len: int) -> Subtitle:
        """Greedily merge adjacent chunks within each pipeline.

        Merging is per-pipeline (i.e. per-sentence after ``sentences()``).
        """
        return self._with_pipelines([p.merge(max_len) for p in self._pipelines])

    # ---- transform (unified content transform dispatch) --------------

    def transform(
        self,
        fn: ApplyFn,
        *,
        cache: dict[str, list[str]] | None = None,
        scope: Literal["chunk", "pipeline"] = "chunk",
        batch_size: int = 1,
        workers: int = 1,
        skip_if: Callable[[str], bool] | None = None,
    ) -> Subtitle:
        """Apply an external function to transform text content.

        *fn* receives a batch of texts and returns one ``list[str]`` per
        input text.  The return value determines the operation:

        - ``["new text"]`` — 1:1 replacement (e.g. punctuation restoration)
        - ``["part1", "part2"]`` — 1:N splitting (e.g. NLP/LLM splitting)
        - ``[]`` — deletion

        Args:
            fn: ``list[str] → list[list[str]]``.
            cache: Optional dict mapping ``text → list[str]``.
                Hits are reused; misses are computed by *fn* and stored.
                The cache is mutated in-place.
            scope: Granularity of text sent to *fn*.
                ``"chunk"`` (default): each chunk text is sent individually.
                ``"pipeline"``: all chunks within a pipeline are joined
                into one text before sending to *fn*.  Use this for
                operations like punctuation restoration that need full
                sentence context.
            batch_size: Number of texts per *fn* call.
                ``0`` means pass all uncached texts in one call.
                Default ``1`` (one text per call).
            workers: Number of threads for concurrent *fn* calls.
                Default ``1`` (sequential).
            skip_if: Optional predicate ``str → bool``.
                Chunks for which ``skip_if(text)`` returns ``True`` are
                left unchanged.

        Returns:
            A new Subtitle with transformed text content.
        """
        from lang_ops.chunk._pipeline import TextPipeline

        if scope == "pipeline":
            return self._transform_pipeline(fn, cache=cache, batch_size=batch_size, workers=workers, skip_if=skip_if)

        # scope == "chunk" — each chunk text sent to fn individually
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
            miss_results: list[list[str]] = _call_apply_fn(fn, miss_texts, batch_size, workers)
            for mi, result_list in zip(miss_indices, miss_results):
                all_results[mi] = result_list
                if cache is not None:
                    cache[all_texts[mi]] = result_list

        # --- distribute results back to per-pipeline groups ---
        new_pipelines: list[TextPipeline] = []
        offset = 0
        for i, texts in enumerate(pipeline_texts):
            n = len(texts)
            pipeline_results = all_results[offset : offset + n]
            offset += n

            new_groups: list[list[str]] = []
            ops = self._pipelines[i]._ops  # noqa: SLF001
            for parts in pipeline_results:
                assert parts is not None
                for part in parts:
                    tokens = ops.split(part)
                    if tokens:
                        new_groups.append(tokens)

            new_pipelines.append(TextPipeline._from_groups(new_groups, ops))  # noqa: SLF001

        return self._with_pipelines(new_pipelines)

    def _transform_pipeline(
        self,
        fn: ApplyFn,
        *,
        cache: dict[str, list[str]] | None = None,
        batch_size: int = 1,
        workers: int = 1,
        skip_if: Callable[[str], bool] | None = None,
    ) -> Subtitle:
        """Transform at pipeline granularity: join chunks, send to fn, re-split.

        Each pipeline's chunks are joined into one text before being sent
        to *fn*.  This is appropriate for operations like punctuation
        restoration that need full context rather than per-chunk input.
        """
        from lang_ops.chunk._pipeline import TextPipeline

        # Build one joined text per pipeline
        joined_texts: list[str] = []
        for pipeline in self._pipelines:
            chunks = pipeline.result()
            joined_texts.append(self._ops.join(chunks))

        if not joined_texts or all(not t for t in joined_texts):
            return self

        # --- resolve from cache and skip_if ---
        all_results: list[list[str] | None] = [None] * len(joined_texts)
        miss_indices: list[int] = []
        miss_texts: list[str] = []

        for idx, text in enumerate(joined_texts):
            if not text:
                all_results[idx] = [text]
            elif skip_if is not None and skip_if(text):
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
                    cache[joined_texts[mi]] = result_list

        # --- rebuild pipelines from results ---
        new_pipelines: list[TextPipeline] = []
        for i, result in enumerate(all_results):
            assert result is not None
            ops = self._pipelines[i]._ops  # noqa: SLF001
            new_groups: list[list[str]] = []
            for part in result:
                tokens = ops.split(part)
                if tokens:
                    new_groups.append(tokens)
            new_pipelines.append(TextPipeline._from_groups(new_groups, ops))  # noqa: SLF001

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
        sub = self if self._sentence_split else self.sentences()

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
                records.append(
                    SentenceRecord(
                        src_text=src_text,
                        start=sub_segments[0].start,
                        end=sub_segments[-1].end,
                        segments=sub_segments,
                    )
                )

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
            segs,
            self._ops,
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
            [self._incomplete],
            self._ops,
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
