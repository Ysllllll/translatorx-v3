"""SegmentBuilder — restructure subtitle segments by sentence, clause, or length.

Typical usage::

    from subtitle import SegmentBuilder

    builder = SegmentBuilder(segments, ops)
    sentences = builder.sentences()                    # sentence-level
    chunks = builder.sentences().by_length(40)         # sentence → length
    merged = builder.sentences().clauses().merge(60)   # split then merge back
    records = builder.records(max_length=40)           # SentenceRecord wrappers

Stream usage::

    builder = SegmentBuilder.stream(ops)
    for seg in incoming:
        done = builder.feed(seg)
        process(done)
    process(builder.flush())
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from ._types import Segment, SentenceRecord, Word
from .words import fill_words, align_segments

if TYPE_CHECKING:
    from lang_ops._core._base_ops import _BaseOps


def _merge_segments(
    segments: list[Segment],
    ops: _BaseOps,
    split_by_speaker: bool,
) -> tuple[str, list[Word], list[tuple[int, int]]]:
    """Merge segments into full text + word list.

    When *split_by_speaker* is True, also records speaker-change
    positions (as char offsets) for downstream sentence splitting.

    Returns:
        (full_text, all_words, speaker_breaks)
        speaker_breaks is a list of (char_offset, char_offset) marking
        boundaries — only populated when split_by_speaker is True.
    """
    all_words: list[Word] = []
    texts: list[str] = []
    speaker_break_offsets: list[int] = []

    prev_speaker: str | None = None
    char_offset = 0

    for seg in segments:
        # Ensure words exist
        if seg.words:
            filled = seg
        else:
            filled = fill_words(seg, split_fn=ops.split)

        # Speaker boundary detection
        if split_by_speaker and filled.words:
            for w in filled.words:
                if prev_speaker is not None and w.speaker != prev_speaker:
                    # Mark the char offset where speaker changes
                    speaker_break_offsets.append(char_offset)
                prev_speaker = w.speaker
                char_offset += len(w.word)
            # Account for join separators between words
            # (approximate — exact offset not critical for break detection)
        else:
            char_offset += len(seg.text)

        all_words.extend(filled.words)
        texts.append(seg.text)

    # Join with space for space-delimited languages, empty for CJK
    # NOTE: if ops.join behaviour is preferred, swap the line below:
    #   full_text = ops.join(texts)
    sep = "" if ops.is_cjk else " "
    full_text = sep.join(texts)

    speaker_breaks = [(off, off) for off in speaker_break_offsets]
    return full_text, all_words, speaker_breaks


def _split_at_speaker_boundaries(
    text: str,
    speaker_breaks: list[int],
) -> list[str]:
    """Split text at speaker change offsets."""
    if not speaker_breaks:
        return [text]
    parts: list[str] = []
    prev = 0
    for offset in sorted(set(speaker_breaks)):
        if 0 < offset < len(text):
            parts.append(text[prev:offset])
            prev = offset
    parts.append(text[prev:])
    return [p for p in parts if p]


class SegmentBuilder:
    """Immutable builder for restructuring subtitle segments.

    Holds a merged ``(full_text, all_words)`` plus the current list of
    text chunks.  Each method returns a **new** SegmentBuilder so calls
    can be chained without mutating prior state.
    """

    __slots__ = ("_ops", "_full_text", "_all_words", "_chunks",
                 "_groups", "_split_by_speaker")

    # ---- construction ------------------------------------------------

    def __init__(
        self,
        segments: list[Segment],
        ops: _BaseOps,
        *,
        split_by_speaker: bool = False,
    ) -> None:
        self._ops = ops
        self._split_by_speaker = split_by_speaker

        full_text, all_words, speaker_breaks = _merge_segments(
            segments, ops, split_by_speaker,
        )
        self._full_text = full_text
        self._all_words = all_words

        # Initial chunks: if speaker splitting, break at speaker boundaries
        if split_by_speaker and speaker_breaks:
            offsets = [b for b, _ in speaker_breaks]
            self._chunks = _split_at_speaker_boundaries(full_text, offsets)
        else:
            self._chunks = [full_text] if full_text else []

        # Each chunk is its own group initially
        self._groups: list[int] = [1] * len(self._chunks)

    def _with_chunks(
        self,
        chunks: list[str],
        groups: list[int] | None = None,
    ) -> SegmentBuilder:
        """Create a new builder sharing the same words but different chunks."""
        new = object.__new__(SegmentBuilder)
        new._ops = self._ops
        new._full_text = self._full_text
        new._all_words = self._all_words
        new._split_by_speaker = self._split_by_speaker
        new._chunks = chunks
        new._groups = groups if groups is not None else [1] * len(chunks)
        return new

    # ---- splitting operations ----------------------------------------

    def sentences(self) -> SegmentBuilder:
        """Split each chunk into sentences.

        Resets group boundaries — each sentence becomes its own group,
        so subsequent ``merge()`` will not combine across sentences.
        """
        result: list[str] = []
        for chunk in self._chunks:
            result.extend(self._ops.split_sentences(chunk))
        # Each sentence is its own group
        return self._with_chunks(result, [1] * len(result))

    def clauses(self) -> SegmentBuilder:
        """Split each chunk into clauses (sentence-aware).

        Preserves group boundaries — clauses split within each group.
        """
        result: list[str] = []
        new_groups: list[int] = []
        idx = 0
        for g_size in self._groups:
            group_chunks = self._chunks[idx:idx + g_size]
            idx += g_size
            sub: list[str] = []
            for chunk in group_chunks:
                sub.extend(self._ops.split_clauses(chunk))
            result.extend(sub)
            new_groups.append(len(sub))
        return self._with_chunks(result, new_groups)

    def by_length(self, max_length: int) -> SegmentBuilder:
        """Split each chunk by length.

        Preserves group boundaries — length splits within each group.
        """
        result: list[str] = []
        new_groups: list[int] = []
        idx = 0
        for g_size in self._groups:
            group_chunks = self._chunks[idx:idx + g_size]
            idx += g_size
            sub: list[str] = []
            for chunk in group_chunks:
                sub.extend(self._ops.split_by_length(chunk, max_length))
            result.extend(sub)
            new_groups.append(len(sub))
        return self._with_chunks(result, new_groups)

    def merge(self, max_length: int) -> SegmentBuilder:
        """Greedily merge adjacent chunks **within each group**.

        Iterates through chunks left-to-right inside every group.
        Each chunk is appended to the current accumulator; if adding
        it would exceed *max_length*, the accumulator is flushed and
        a new one starts.  Uses ``ops.length()`` for measurement so
        CJK width rules apply.

        Group boundaries (set by ``sentences()``) are never crossed,
        so sentences stay separate even if they would fit together.
        """
        if not self._chunks:
            return self._with_chunks([])

        sep = "" if self._ops.is_cjk else " "
        merged: list[str] = []
        new_groups: list[int] = []
        idx = 0

        for g_size in self._groups:
            group_chunks = self._chunks[idx:idx + g_size]
            idx += g_size

            group_merged: list[str] = []
            current_parts: list[str] = []
            current_text = ""

            for chunk in group_chunks:
                if current_parts:
                    # Don't add separator if chunk already has leading space
                    if sep and chunk.startswith(sep):
                        candidate = current_text + chunk
                    else:
                        candidate = current_text + sep + chunk
                else:
                    candidate = chunk
                if current_parts and self._ops.length(candidate) > max_length:
                    group_merged.append(current_text)
                    current_parts = [chunk]
                    current_text = chunk
                else:
                    current_parts.append(chunk)
                    current_text = candidate

            if current_parts:
                group_merged.append(current_text)

            merged.extend(group_merged)
            new_groups.append(len(group_merged))

        return self._with_chunks(merged, new_groups)

    # ---- output ------------------------------------------------------

    def build(self) -> list[Segment]:
        """Align current chunks with words and return Segments."""
        return align_segments(self._chunks, self._all_words)

    def records(self, max_length: int | None = None) -> list[SentenceRecord]:
        """Return SentenceRecords: one per sentence, with sub-segments.

        Each record's ``src_text`` is a full sentence.  If *max_length*
        is given, sentences are further split by ``clauses → by_length``
        to produce the record's ``segments``.
        """
        sentence_chunks = self.sentences()._chunks
        sentence_segments = align_segments(sentence_chunks, self._all_words)

        records: list[SentenceRecord] = []
        for sent_seg in sentence_segments:
            if max_length is not None:
                # Sub-split: clauses → by_length
                sub_chunks: list[str] = []
                for clause in self._ops.split_clauses(sent_seg.text):
                    sub_chunks.extend(
                        self._ops.split_by_length(clause, max_length)
                    )
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
    ) -> _StreamBuilder:
        """Create a streaming builder that accepts segments one at a time."""
        return _StreamBuilder(ops, split_by_speaker=split_by_speaker)


class _StreamBuilder:
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

        builder = SegmentBuilder(
            self._buffer_segments, self._ops,
            split_by_speaker=self._split_by_speaker,
        )
        all_sentences = builder.sentences().build()

        if len(all_sentences) <= 1:
            # Not enough to confirm any sentence is complete
            return []

        # All but the last sentence are confirmed complete
        done = all_sentences[:-1]

        # Rebuild buffer from the last (incomplete) sentence's words
        last = all_sentences[-1]
        if last.words:
            self._buffer_segments = [last]
        else:
            self._buffer_segments = []

        return done

    def flush(self) -> list[Segment]:
        """Flush remaining buffer as sentence-segments."""
        if not self._buffer_segments:
            return []

        builder = SegmentBuilder(
            self._buffer_segments, self._ops,
            split_by_speaker=self._split_by_speaker,
        )
        result = builder.sentences().build()
        self._buffer_segments = []
        return result
