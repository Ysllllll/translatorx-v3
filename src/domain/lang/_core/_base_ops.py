"""Shared base class for all language operations."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod

from ._chars import STRIP_PUNCT, decompose_token


# Mode shorthand: "c" = "character", "w" = "word"
_MODE_SHORTHAND = {"c": "character", "w": "word"}
_VALID_MODES = {"character", "word"}


# Words with internal dots that should be treated as atoms (e.g. "Node.js").
_DOTTED_WORD_RE = re.compile(r"\b(\w+(?:\.\w+)+)\b")


def normalize_mode(mode: str) -> str:
    """Normalize mode shorthand to full name."""
    return _MODE_SHORTHAND.get(mode, mode)


class _BaseOps(ABC):
    """Abstract base for all language-specific text operations.

    Subclasses must implement:
        - split, join, length, normalize
        - sentence_terminators, clause_separators, abbreviations, is_cjk
    """

    # -- Abstract properties (override in subclass) -------------------------

    @property
    @abstractmethod
    def sentence_terminators(self) -> frozenset[str]: ...

    @property
    @abstractmethod
    def clause_separators(self) -> frozenset[str]: ...

    @property
    @abstractmethod
    def abbreviations(self) -> frozenset[str]: ...

    @property
    @abstractmethod
    def is_cjk(self) -> bool: ...

    @property
    def strip_spaces(self) -> bool:
        """Whether to strip inter-sentence/clause leading spaces. True for CJK except Korean."""
        return self.is_cjk

    # -- Abstract methods (override in subclass) ----------------------------

    @abstractmethod
    def split(self, text: str, mode: str = "word", attach_punctuation: bool = True) -> list[str]: ...

    @abstractmethod
    def join(self, tokens: list[str]) -> str: ...

    @abstractmethod
    def length(self, text: str, **kwargs: int) -> int: ...

    @abstractmethod
    def normalize(self, text: str) -> str: ...

    # -- Shared concrete methods --------------------------------------------

    def plength(self, text: str, font_path: str, font_size: int) -> int:
        from PIL import ImageFont

        left, _, right, _ = ImageFont.truetype(font_path, font_size).getbbox(text)
        return max(0, int(right - left))

    def strip(self, text: str, chars: str | None = None) -> str:
        return text.strip(chars)

    def lstrip(self, text: str, chars: str | None = None) -> str:
        return text.lstrip(chars)

    def rstrip(self, text: str, chars: str | None = None) -> str:
        return text.rstrip(chars)

    def strip_punc(self, text: str) -> str:
        return text.strip(STRIP_PUNCT)

    def lstrip_punc(self, text: str) -> str:
        return text.lstrip(STRIP_PUNCT)

    def rstrip_punc(self, text: str) -> str:
        return text.rstrip(STRIP_PUNCT)

    def transfer_punc(self, text_a: str, text_b: str) -> str:
        """Transfer punctuation from *text_b* onto words of *text_a*.

        Token-level punctuation transfer: takes the word content from
        *text_a* and wraps it with the leading/trailing punctuation
        found in *text_b*.

        Not to be confused with LLM-based punctuation *restoration*
        (see :class:`~adapters.preprocess.PuncRestorer`).
        """
        tokens_a = self.split(text_a)
        tokens_b = self.split(text_b)
        if len(tokens_a) != len(tokens_b):
            raise ValueError(f"Token count mismatch: text_a has {len(tokens_a)}, text_b has {len(tokens_b)}")
        result: list[str] = []
        for ta, tb in zip(tokens_a, tokens_b):
            _, content_a, _ = decompose_token(ta)
            lead_b, _, trail_b = decompose_token(tb)
            result.append(lead_b + content_a + trail_b)
        return self.join(result)

    # -- Punc-restoration post-processing hooks ----------------------------

    @property
    def _trailing_punct_chars(self) -> str:
        """Characters that can appear as trailing punctuation in this language.

        Default covers ASCII + CJK full-width terminators and clause separators
        so subclasses rarely need to override.
        """
        return ".!?,;:…。！？，；：、"

    def protect_dotted_words(self, source: str, restored: str) -> str:
        """Restore internal-dot words corrupted by a punc-restoration model.

        For example, the NER model may turn ``Node.js`` into ``Node. Js`` or
        ``e.g.`` into ``e. G.`` — this method detects such corruption and
        restores the original form. Language-agnostic: operates on the
        Latin-script substrings of *source* regardless of the surrounding
        script, so CJK callers get the same protection for embedded terms.
        """
        dotted_words = _DOTTED_WORD_RE.findall(source)
        if not dotted_words:
            return restored

        for original in dotted_words:
            parts = original.split(".")
            escaped = [re.escape(p) for p in parts]
            corrupted_pattern = r"[.\s,;:!?]*\s*".join(escaped)
            corrupted_re = re.compile(corrupted_pattern, re.IGNORECASE)
            restored = corrupted_re.sub(original, restored)

        return restored

    def preserve_trailing_punc(self, source: str, restored: str) -> str:
        """Preserve *source*'s trailing punctuation on *restored*.

        Punc-restoration models may drop or change trailing punctuation
        (including ``...``). If *source* ends with punctuation, ensure the
        restored text ends with the same sequence. The character set used is
        :attr:`_trailing_punct_chars`, which subclasses may override to add
        locale-specific marks.
        """
        chars = re.escape(self._trailing_punct_chars)
        trailing_re = re.compile(f"[{chars}]+$")

        src_trail = trailing_re.search(source.rstrip())
        if src_trail is None:
            return restored

        src_punc = src_trail.group()
        restored_stripped = trailing_re.sub("", restored.rstrip())
        if not restored_stripped:
            return restored
        return restored_stripped + src_punc

    # -- Segment-level shortcuts --------------------------------------------

    def split_sentences(self, text: str) -> list[str]:
        """Split text into sentences (token-based)."""
        from domain.lang.chunk._pipeline import TextPipeline

        return TextPipeline(text, ops=self).sentences().result()

    def split_clauses(self, text: str) -> list[str]:
        """Split text into clauses (sentence boundaries included, token-based)."""
        from domain.lang.chunk._pipeline import TextPipeline

        return TextPipeline(text, ops=self).clauses().result()

    def split_by_length(self, text: str, max_len: int) -> list[str]:
        """Split text into chunks whose length ≤ *max_len* (token-based)."""
        from domain.lang.chunk._pipeline import TextPipeline

        return TextPipeline(text, ops=self).split(max_len).result()

    def merge_by_length(self, chunks: list[str], max_len: int) -> list[str]:
        """Greedily merge adjacent chunks whose combined length ≤ *max_len*."""
        from domain.lang.chunk._merge import merge_chunks_by_length

        return merge_chunks_by_length(chunks, self, max_len)

    def chunk(self, text: str) -> "TextPipeline":
        """Create a TextPipeline for chainable text structuring."""
        from domain.lang.chunk._pipeline import TextPipeline

        return TextPipeline(text, ops=self)

    # -- Alignment support --------------------------------------------------

    def ends_with_clause_punct(self, text: str) -> bool:
        """True if *text* ends with a clause / sentence terminator for this language."""
        if not text:
            return False
        stripped = text.rstrip()
        if not stripped:
            return False
        return stripped[-1] in self.clause_separators or stripped[-1] in self.sentence_terminators

    def find_half_join_balance(self, texts: list[str]) -> list[int]:
        """Return candidate binary-split indices for a sequence of sibling texts.

        Given ``N >= 2`` source segments that must be split into two halves,
        return the boundary indices (1..N-1) ranked by a balance score.
        Boundaries that land on a clause / sentence terminator are strongly
        preferred; within equal preference, halves closer to equal total length
        rank higher. The returned list is ordered best-first so callers can
        try candidates in sequence until one succeeds.
        """
        n = len(texts)
        if n <= 2:
            return [1]

        # weight=1 when the text BEFORE the boundary ends on a clause/sentence
        # terminator; otherwise a large penalty so these are tried last.
        # Use a finite large value (not sys.maxsize) to avoid overflow in the
        # weighted product below.
        _PENALTY = 10**9
        weights = [1 if self.ends_with_clause_punct(t) else _PENALTY for t in texts]
        lengths = [self.length(t) for t in texts]
        total = sum(lengths)
        diffs: list[tuple[int, int]] = []  # (score, boundary_idx)
        for i in range(1, n):
            left = sum(lengths[:i])
            imbalance = abs(left - (total - left))
            score = weights[i - 1] * imbalance
            diffs.append((score, i))
        diffs.sort(key=lambda x: x[0])
        return [idx for _, idx in diffs]

    def length_ratio(self, a: str, b: str) -> float:
        """Cross-text length ratio ``length(a) / length(b)`` with zero-div guard."""
        la = self.length(a)
        lb = self.length(b)
        return (la + 1e-7) / (lb + 1e-7)

    def check_and_correct_split_sentence(
        self,
        split_sentence: list[str],
        sentence: str,
        can_reverse: bool = True,
    ) -> tuple[bool, list[str]]:
        """Verify that ``split_sentence`` (len=2) concatenates to ``sentence``.

        Mirrors the legacy ``LanguageHandler.check_and_correct_split_sentence``
        contract: accepts ``[a, b]``, tries to window-match each piece against
        the words of ``sentence``. If both match, returns ``(True, [a, b])``.
        When ``can_reverse`` is True and the reversed concatenation matches
        while the original does not, returns the swapped pieces (CJK-aware
        end/mid terminator swap for natural reading).

        Returns ``(False, pieces)`` on unrecoverable mismatch.
        """
        if len(split_sentence) != 2:
            raise ValueError(f"check_and_correct_split_sentence expects 2 pieces, got {len(split_sentence)}")

        words_list = list(self.split(sentence))
        fixed_texts: list[str] = []
        for piece in split_sentence:
            try:
                fixed = self._match_window(piece, words_list)
            except ValueError:
                fixed = ""
            fixed_texts.append(fixed)

        first, second = fixed_texts[0], fixed_texts[1]
        if not first and not second:
            return False, fixed_texts

        import re as _re

        # Recover the missing side from the full sentence by subtraction.
        if first and not second:
            second = _re.sub(_re.escape(first), "", sentence, count=1)
            fixed_texts = [first, second]
        elif not first and second:
            first = _re.sub(_re.escape(second), "", sentence, count=1)
            fixed_texts = [first, second]

        normalized_sent = _re.sub(r" +", "", sentence)
        concat_forward = _re.sub(r" +", "", self.join([first, second]))
        concat_reverse = _re.sub(r" +", "", self.join([second, first]))
        good = concat_forward == normalized_sent or concat_reverse == normalized_sent

        if not good:
            # Try one more time, recovering the missing side.
            if first:
                trial = [first, _re.sub(_re.escape(first), "", sentence, count=1)]
                if _re.sub(r" +", "", self.join(trial)) == normalized_sent:
                    fixed_texts = trial
                    good = True
            if not good and second:
                trial = [_re.sub(_re.escape(second), "", sentence, count=1), second]
                if _re.sub(r" +", "", self.join(trial)) == normalized_sent:
                    fixed_texts = trial
                    good = True

        if good and can_reverse and self.is_cjk:
            # CJK-specific: prefer the ordering that ends the first piece with a
            # sentence terminator (。？！) over a clause separator (，、).
            a, b = fixed_texts[0], fixed_texts[1]
            if (
                a
                and b
                and _re.sub(r" +", "", self.join([b, a])) == normalized_sent
                and a[-1] in self.sentence_terminators
                and b[-1] in self.clause_separators
            ):
                schar = b[-1]
                fchar = a[-1]
                fixed_texts = [a[:-1] + schar, b[:-1] + fchar]

        return good, [t.strip() for t in fixed_texts]

    def _match_window(self, phrase: str, words_list: list[str]) -> str:
        """Find a contiguous window in ``words_list`` whose stripped words match *phrase*.

        Returns the joined window string (unmodified, with punctuation) on match.
        Raises ``ValueError`` if no such window exists.
        """
        words = list(self.split(self.join([p.strip() for p in phrase.split("\n")])))
        n = len(words)
        if n == 0:
            return ""
        for i in range(len(words_list) - n + 1):
            window = words_list[i : i + n]
            if all(self.strip_punc(window[j]) == self.strip_punc(words[j]) for j in range(n)):
                return self.join(window)
        raise ValueError(f"phrase '{phrase}' does not match any window in word list")
