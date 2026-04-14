"""SegmentProcessor — restructure subtitle segments via ChunkPipeline.

Thin wrapper that delegates all text operations to
:class:`~lang_ops.chunk.ChunkPipeline` and word alignment to
:func:`~subtitle.align.align_segments`.

Typical usage::

    from subtitle import SegmentProcessor

    processor = SegmentProcessor(segments, ops)
    result = processor.sentences().max_length(40).build()
    records = processor.records(max_length=40)

Stream usage::

    stream = SegmentProcessor.stream(ops)
    for seg in incoming:
        done = stream.feed(seg)
        process(done)
    process(stream.flush())
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .model import Segment, SentenceRecord, Word
from .align import fill_words, align_segments

if TYPE_CHECKING:
    from lang_ops._core._base_ops import _BaseOps


def _extract(
    segments: list[Segment],
    ops: _BaseOps,
) -> tuple[list[Word], str]:
    """Extract all words and build joined text from segments.

    Segments without words are auto-filled via :func:`fill_words`.

    Returns:
        (all_words, full_text)
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


class SegmentProcessor:
    """Immutable processor for restructuring subtitle segments.

    Holds ``(all_words, pipeline)`` — each method returns a **new**
    instance so calls can be chained without mutating prior state.

    All text operations are delegated to
    :class:`~lang_ops.chunk.ChunkPipeline` which tokenizes once and
    operates on the token array.  No redundant re-tokenization across
    chained calls.
    """

    __slots__ = ("_ops", "_all_words", "_pipeline")

    # ---- construction ------------------------------------------------

    def __init__(
        self,
        segments: list[Segment],
        ops: _BaseOps,
        *,
        split_by_speaker: bool = False,
    ) -> None:
        from lang_ops.chunk._pipeline import ChunkPipeline

        self._ops = ops
        all_words, full_text = _extract(segments, ops)
        self._all_words = all_words

        if split_by_speaker:
            chunks = _speaker_chunks(all_words, ops)
            self._pipeline = ChunkPipeline.from_chunks(chunks, ops=ops)
        else:
            self._pipeline = ChunkPipeline(full_text, ops=ops)

    def _with_pipeline(self, pipeline: object) -> SegmentProcessor:
        """Create a new processor sharing the same words."""
        new = object.__new__(SegmentProcessor)
        new._ops = self._ops
        new._all_words = self._all_words
        new._pipeline = pipeline
        return new

    # ---- text operations (delegated to ChunkPipeline) ----------------

    def sentences(self) -> SegmentProcessor:
        """Split each chunk into sentences."""
        return self._with_pipeline(self._pipeline.sentences())

    def clauses(self) -> SegmentProcessor:
        """Split each chunk into clauses (sentence-aware)."""
        return self._with_pipeline(self._pipeline.clauses())

    def max_length(self, max_length: int) -> SegmentProcessor:
        """Split each chunk by length."""
        return self._with_pipeline(self._pipeline.max_length(max_length))

    def merge(self, max_length: int) -> SegmentProcessor:
        """Greedily merge adjacent chunks whose combined length ≤ *max_length*."""
        return self._with_pipeline(self._pipeline.merge(max_length))

    # ---- output ------------------------------------------------------

    def build(self) -> list[Segment]:
        """Align current chunks with words and return Segments."""
        return align_segments(self._pipeline.result(), self._all_words)

    def records(self, max_length: int | None = None) -> list[SentenceRecord]:
        """Return SentenceRecords: one per sentence, with sub-segments.

        Each record's ``src_text`` is a full sentence.  If *max_length*
        is given, sentences are further split by ``clauses → max_length``
        to produce the record's ``segments``.
        """
        from lang_ops.chunk._pipeline import ChunkPipeline

        sentence_chunks = self._pipeline.sentences().result()
        sentence_segments = align_segments(sentence_chunks, self._all_words)

        records: list[SentenceRecord] = []
        for sent_seg in sentence_segments:
            if max_length is not None:
                sub_pipeline = ChunkPipeline(sent_seg.text, ops=self._ops)
                sub_chunks = (sub_pipeline
                              .clauses()
                              .max_length(max_length)
                              .result())
                sub_segments = align_segments(sub_chunks, sent_seg.words)
            else:
                sub_segments = [sent_seg]

            records.append(SentenceRecord(
                src_text=sent_seg.text,
                start=sent_seg.start,
                end=sent_seg.end,
                segments=sub_segments,
            ))

        return records

    # ---- stream mode -------------------------------------------------

    @classmethod
    def stream(
        cls,
        ops: _BaseOps,
        *,
        split_by_speaker: bool = False,
    ) -> _StreamProcessor:
        """Create a streaming processor that accepts segments one at a time."""
        return _StreamProcessor(ops, split_by_speaker=split_by_speaker)


class _StreamProcessor:
    """Stateful streaming segment restructurer.

    Buffers incoming segments.  On each :meth:`feed`, tries to emit
    completed sentences (those followed by more text, proving they are
    complete).  :meth:`flush` emits everything remaining.
    """

    __slots__ = ("_ops", "_split_by_speaker", "_buffer_segments")

    def __init__(
        self,
        ops: _BaseOps,
        *,
        split_by_speaker: bool = False,
    ) -> None:
        self._ops = ops
        self._split_by_speaker = split_by_speaker
        self._buffer_segments: list[Segment] = []

    def feed(self, segment: Segment) -> list[Segment]:
        """Feed one segment; return any completed sentence-segments."""
        self._buffer_segments.append(segment)

        processor = SegmentProcessor(
            self._buffer_segments, self._ops,
            split_by_speaker=self._split_by_speaker,
        )
        all_sentences = processor.sentences().build()

        if len(all_sentences) <= 1:
            return []

        # All but the last sentence are confirmed complete
        done = all_sentences[:-1]

        # Rebuild buffer from the last (incomplete) sentence's words
        last = all_sentences[-1]
        self._buffer_segments = [last] if last.words else []

        return done

    def flush(self) -> list[Segment]:
        """Flush remaining buffer as sentence-segments."""
        if not self._buffer_segments:
            return []

        processor = SegmentProcessor(
            self._buffer_segments, self._ops,
            split_by_speaker=self._split_by_speaker,
        )
        result = processor.sentences().build()
        self._buffer_segments = []
        return result
