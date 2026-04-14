"""Subtitle — chainable subtitle segment restructuring.

Thin wrapper that delegates text operations to
:class:`~lang_ops.chunk.ChunkPipeline` and word alignment to
:func:`~subtitle.align.align_segments`.

Typical usage::

    from subtitle import Subtitle

    sub = Subtitle(segments, language="zh")
    result = sub.sentences().max_length(40).build()
    records = sub.sentences().max_length(40).records()

With external text transforms (e.g. punctuation restoration)::

    records = (sub
        .sentences()
        .apply(restore_fn)       # fn: list[str] → list[list[str]]
        .sentences()             # re-split after text change
        .max_length(40)
        .records())
"""

from __future__ import annotations

from collections.abc import Callable, MutableMapping
from typing import TYPE_CHECKING

from .model import Segment, SentenceRecord, Word
from .align import fill_words, align_segments

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

    Holds ``(all_words, pipeline)`` — each method returns a **new**
    instance so calls can be chained without mutating prior state.

    All text operations are delegated to
    :class:`~lang_ops.chunk.ChunkPipeline` which tokenizes once and
    operates on the token array.
    """

    __slots__ = ("_ops", "_all_words", "_pipeline")

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
        self._all_words = all_words

        if split_by_speaker:
            chunks = _speaker_chunks(all_words, self._ops)
            self._pipeline = ChunkPipeline.from_chunks(chunks, ops=self._ops)
        else:
            self._pipeline = ChunkPipeline(full_text, ops=self._ops)

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

    def _with_pipeline(self, pipeline: object) -> Subtitle:
        """Create a new Subtitle sharing the same words."""
        new = object.__new__(Subtitle)
        new._ops = self._ops
        new._all_words = self._all_words
        new._pipeline = pipeline
        return new

    def _with_words_and_pipeline(
        self, words: list[Word], pipeline: object,
    ) -> Subtitle:
        """Create a new Subtitle with updated words and pipeline."""
        new = object.__new__(Subtitle)
        new._ops = self._ops
        new._all_words = words
        new._pipeline = pipeline
        return new

    # ---- text operations (delegated to ChunkPipeline) ----------------

    def sentences(self) -> Subtitle:
        """Split each chunk into sentences."""
        return self._with_pipeline(self._pipeline.sentences())

    def clauses(self) -> Subtitle:
        """Split each chunk into clauses (sentence-aware)."""
        return self._with_pipeline(self._pipeline.clauses())

    def max_length(self, max_length: int) -> Subtitle:
        """Split each chunk by length."""
        return self._with_pipeline(self._pipeline.max_length(max_length))

    def merge(self, max_length: int) -> Subtitle:
        """Greedily merge adjacent chunks whose combined length ≤ *max_length*."""
        return self._with_pipeline(self._pipeline.merge(max_length))

    def apply(
        self,
        fn: ApplyFn,
        cache: ApplyCache | None = None,
        batch_size: int = 1,
        workers: int = 1,
    ) -> Subtitle:
        """Apply an external function to each chunk.

        *fn* receives a batch of texts and returns one ``list[str]`` per
        input text:

        - ``["new text"]`` → 1:1 replacement (e.g. punctuation restoration)
        - ``["part1", "part2"]`` → 1:N splitting (e.g. NLP/LLM splitting)
        - ``[]`` → deletion

        See :meth:`ChunkPipeline.apply` for full parameter docs.
        """
        return self._with_pipeline(
            self._pipeline.apply(fn, cache=cache,
                                 batch_size=batch_size, workers=workers)
        )

    # ---- output ------------------------------------------------------

    def build(self) -> list[Segment]:
        """Align current chunks with words and return Segments."""
        return align_segments(self._pipeline.result(), self._all_words)

    def records(self, max_length: int | None = None) -> list[SentenceRecord]:
        """Return SentenceRecords: one per sentence, with sub-segments.

        Each record's ``src_text`` is a full sentence.  If *max_length*
        is given, sentences are further split by ``clauses → max_length``
        to produce the record's ``segments``.

        Efficient: operates on existing token groups without re-tokenizing.
        """
        sent_pipeline = self._pipeline.sentences()
        sentence_chunks = sent_pipeline.result()
        sentence_segments = align_segments(sentence_chunks, self._all_words)

        if max_length is None:
            return [
                SentenceRecord(
                    src_text=seg.text,
                    start=seg.start, end=seg.end,
                    segments=[seg],
                )
                for seg in sentence_segments
            ]

        # Operate on existing token groups — no re-tokenization
        sent_groups = sent_pipeline._groups  # noqa: SLF001
        records: list[SentenceRecord] = []
        for sent_seg, group in zip(sentence_segments, sent_groups):
            sub_pipeline = sent_pipeline._with_groups([group])  # noqa: SLF001
            sub_chunks = (sub_pipeline
                          .clauses()
                          .max_length(max_length)
                          .result())
            sub_segments = align_segments(sub_chunks, sent_seg.words)
            records.append(SentenceRecord(
                src_text=sent_seg.text,
                start=sent_seg.start, end=sent_seg.end,
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
