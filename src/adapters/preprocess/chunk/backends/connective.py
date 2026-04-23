"""Connective-word chunk backends — registered as ``"rule_connective"``
and ``"pos_connective"``.

Two independent splitters that cut long sentences at conjunction /
connective words (English ``because`` / ``but`` / ``when``,
Chinese ``因为`` / ``但是``, ...):

* ``rule_connective`` — deterministic keyword match against
  :attr:`LangOps.connectives`, with a min-words-per-side guard.
  No external dependencies, fast.

* ``pos_connective`` — spaCy POS + dependency-aware version. Matches
  the same connective lexicon but additionally inspects
  ``token.dep_`` / ``token.head.pos_`` so ambiguous words like English
  "that" can be included without producing false splits on determiner
  usage. Ported from the legacy ``split_by_nlp_model`` algorithm.

Both backends iterate until no further splits occur in a pass, so long
sentences with multiple connectives are fully decomposed.
"""

from __future__ import annotations

import logging
import string
import threading
from typing import ClassVar

from domain.lang import LangOps, normalize_language
from domain.lang._core._punctuation import STRIP_PUNCT

from adapters.preprocess.chunk.registry import Backend, ChunkBackendRegistry

logger = logging.getLogger(__name__)

_STRIP_CHARS = string.punctuation + STRIP_PUNCT


def _strip_punct(tok: str) -> str:
    return tok.strip(_STRIP_CHARS)


def _count_non_punct(tokens: list[str]) -> int:
    n = 0
    for t in tokens:
        if not t.strip():
            continue
        if _strip_punct(t) == "":
            continue
        n += 1
    return n


def _split_once(text: str, ops, connectives: frozenset[str], min_words: int) -> list[str]:
    """Split *text* at the first qualifying connective token.

    Returns a list of 1 (no split) or 2 (split) pieces. Tokens scan
    left-to-right; only the first split per pass is taken so that an
    outer loop can converge deterministically — matching the legacy
    ``split_by_nlp_model`` behaviour.
    """
    tokens = ops.split(text, mode="word")
    if not tokens:
        return [text]
    for i, tok in enumerate(tokens):
        normalized = _strip_punct(tok).lower()
        if not normalized or normalized not in connectives:
            continue
        # Skip if followed by an apostrophe-style contraction marker
        if i + 1 < len(tokens) and tokens[i + 1].startswith(("'", "’")):
            continue
        left = _count_non_punct(tokens[:i])
        right = _count_non_punct(tokens[i + 1 :])
        if left < min_words or right < min_words:
            continue
        left_text = ops.join(tokens[:i]).strip()
        right_text = ops.join(tokens[i:]).strip()
        if not left_text or not right_text:
            continue
        return [left_text, right_text]
    return [text]


def _split_by_connectives(text: str, ops, min_words: int) -> list[str]:
    """Iteratively split *text* at connectives until stable."""
    connectives = ops.connectives
    if not connectives:
        return [text]
    pieces = [text]
    # Safety cap: a sentence can't realistically produce more splits than
    # it has tokens; this guards pathological LLM / tokenizer output.
    max_passes = max(16, len(text))
    for _ in range(max_passes):
        changed = False
        new_pieces: list[str] = []
        for piece in pieces:
            sub = _split_once(piece, ops, connectives, min_words)
            if len(sub) > 1:
                changed = True
            new_pieces.extend(sub)
        pieces = new_pieces
        if not changed:
            break
    return [p for p in pieces if p.strip()] or [text]


@ChunkBackendRegistry.register("rule_connective")
def rule_connective_backend(*, language: str, min_words: int = 5) -> Backend:
    """Build a keyword-based connective splitter for *language*.

    Uses :attr:`LangOps.connectives`. Languages without a populated
    connective set degrade gracefully to passthrough.

    Parameters
    ----------
    language:
        BCP-47 / ISO code driving token segmentation and the connective
        lexicon.
    min_words:
        Minimum number of non-punctuation tokens that must exist on each
        side of the connective for a split to occur. Defaults to ``5``,
        matching the legacy implementation.
    """
    if min_words < 1:
        raise ValueError("min_words must be >= 1")
    ops = LangOps.for_language(normalize_language(language))

    def _backend(texts: list[str]) -> list[list[str]]:
        out: list[list[str]] = []
        for text in texts:
            if not text.strip():
                out.append([text])
                continue
            out.append(_split_by_connectives(text, ops, min_words))
        return out

    return _backend


# ---------------------------------------------------------------------------
# POS-aware connective backend
# ---------------------------------------------------------------------------


#: POS / dep rule table mirroring the legacy ``split_by_nlp_model``.
#: Each entry: (extra_conjunctions, verb_pos, mark_dep, noun_pos, det_pron_deps).
#: Extra conjunctions are added on top of the language's :attr:`ops.connectives`
#: — they are words the POS logic can safely disambiguate (e.g. English
#: "that" / "which") but which we do not list in the pure-lexicon set.
_POS_RULES: dict[str, tuple[frozenset[str], str, frozenset[str], frozenset[str], frozenset[str]]] = {
    "en": (
        frozenset({"that", "which", "where", "what"}),
        "VERB",
        frozenset({"mark", "nsubj", "dobj"}),
        frozenset({"NOUN", "PROPN"}),
        frozenset({"det", "pron"}),
    ),
    "zh": (
        frozenset(),
        "VERB",
        frozenset({"mark"}),
        frozenset({"NOUN", "PROPN"}),
        frozenset({"det", "pron"}),
    ),
    "ja": (
        frozenset(),
        "VERB",
        frozenset({"mark"}),
        frozenset({"NOUN", "PROPN"}),
        frozenset({"case"}),
    ),
    "ko": (
        frozenset({"만약", "비록", "왜냐하면", "그러나", "하지만", "때문에"}),
        "VERB",
        frozenset({"mark", "advcl"}),
        frozenset({"NOUN", "PROPN"}),
        frozenset({"det", "case"}),
    ),
    "fr": (
        frozenset({"que", "qui", "où"}),
        "VERB",
        frozenset({"mark"}),
        frozenset({"NOUN", "PROPN"}),
        frozenset({"det", "pron"}),
    ),
    "ru": (
        frozenset({"что", "который", "где"}),
        "VERB",
        frozenset({"mark"}),
        frozenset({"NOUN", "PROPN"}),
        frozenset({"det"}),
    ),
    "es": (
        frozenset({"que", "cual", "donde"}),
        "VERB",
        frozenset({"mark"}),
        frozenset({"NOUN", "PROPN"}),
        frozenset({"det", "pron"}),
    ),
    "de": (
        frozenset({"dass", "welche", "wo"}),
        "VERB",
        frozenset({"mark"}),
        frozenset({"NOUN", "PROPN"}),
        frozenset({"det", "pron"}),
    ),
    "pt": (
        frozenset({"que", "qual", "onde"}),
        "VERB",
        frozenset({"mark"}),
        frozenset({"NOUN", "PROPN"}),
        frozenset({"det", "pron"}),
    ),
}


class _PosConnectiveSplitter:
    """spaCy-backed POS-aware connective splitter.

    Instances are cached per model name so multiple configurations for
    the same pipeline share the loaded model (mirrors
    :class:`~adapters.preprocess.chunk.backends.spacy.SpacySplitter`).
    """

    _instances: ClassVar[dict[tuple[str, str, int], "_PosConnectiveSplitter"]] = {}
    _nlp_cache: ClassVar[dict[str, object]] = {}
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, model_name: str, language: str, min_words: int) -> None:
        nlp = self._nlp_cache.get(model_name)
        if nlp is None:
            import spacy

            logger.info("Loading spaCy model %r for pos_connective (%s) ...", model_name, language)
            nlp = spacy.load(model_name)
            self._nlp_cache[model_name] = nlp
        self._nlp = nlp
        self._language = language
        self._ops = LangOps.for_language(language)
        self._min_words = min_words
        self._infer_lock = threading.Lock()

    @classmethod
    def get(cls, model: str, language: str, min_words: int) -> "_PosConnectiveSplitter":
        key = (model, language, min_words)
        if key not in cls._instances:
            with cls._lock:
                if key not in cls._instances:
                    cls._instances[key] = cls(model, language, min_words)
        return cls._instances[key]

    # -- helpers --------------------------------------------------------

    def _conjunction_set(self) -> frozenset[str]:
        extra, *_ = _POS_RULES.get(self._language, _POS_RULES["en"])
        return self._ops.connectives | extra

    def _is_conjunction(self, token) -> bool:
        conj_set = self._conjunction_set()
        if token.text.lower() not in conj_set:
            return False
        extra, verb_pos, mark_dep, noun_pos, det_pron_deps = _POS_RULES.get(self._language, _POS_RULES["en"])
        # English "that" — only treat as connective if truly subordinating
        if self._language == "en" and token.text.lower() == "that":
            return token.dep_ in mark_dep and token.head.pos_ == verb_pos
        # Filter out determiner/pronoun attached to a noun (e.g. "that book")
        if token.dep_ in det_pron_deps and token.head.pos_ in noun_pos:
            return False
        return True

    def _split_iter(self, text: str) -> list[str]:
        sentences = [text]
        max_passes = 32
        for _ in range(max_passes):
            split_occurred = False
            new_sentences: list[str] = []
            for sent in sentences:
                with self._infer_lock:
                    doc = self._nlp(sent)
                start = 0
                did_split = False
                for token in doc:
                    if not self._is_conjunction(token):
                        continue
                    # Skip contractions like "isn't" → next token starts with '
                    if token.i + 1 < len(doc) and doc[token.i + 1].text.startswith(("'", "’")):
                        continue
                    left_words = [w.text for w in doc[max(0, token.i - self._min_words) : token.i] if not w.is_punct]
                    right_words = [w.text for w in doc[token.i + 1 : token.i + 1 + self._min_words] if not w.is_punct]
                    if len(left_words) >= self._min_words and len(right_words) >= self._min_words:
                        new_sentences.append(doc[start : token.i].text.strip())
                        start = token.i
                        split_occurred = True
                        did_split = True
                        break
                tail = doc[start:].text.strip()
                if tail:
                    new_sentences.append(tail)
            if not split_occurred:
                break
            sentences = [s for s in new_sentences if s]
        return [s for s in sentences if s] or [text]

    # -- ApplyFn interface ---------------------------------------------

    def __call__(self, texts: list[str]) -> list[list[str]]:
        out: list[list[str]] = []
        for text in texts:
            if not text.strip():
                out.append([text])
                continue
            out.append(self._split_iter(text))
        return out


@ChunkBackendRegistry.register("pos_connective")
def pos_connective_backend(*, language: str, min_words: int = 5, model: str | None = None) -> Backend:
    """Build a spaCy POS-aware connective splitter for *language*.

    Parameters
    ----------
    language:
        BCP-47 / ISO code (``"en"``, ``"zh"``, ...).
    min_words:
        Minimum non-punctuation tokens required on each side of the
        connective. Defaults to ``5``.
    model:
        Optional spaCy model override. Defaults to the per-language
        entry in
        :data:`~adapters.preprocess.chunk.backends.spacy.DEFAULT_MODELS_BY_LANG`
        (falling back to the multilingual ``xx_sent_ud_sm``).
    """
    if min_words < 1:
        raise ValueError("min_words must be >= 1")

    from adapters.preprocess.chunk.backends.spacy import (
        DEFAULT_MODELS_BY_LANG,
        FALLBACK_MODEL,
    )

    lang = normalize_language(language)
    chosen = model or DEFAULT_MODELS_BY_LANG.get(lang, FALLBACK_MODEL)
    splitter = _PosConnectiveSplitter.get(chosen, lang, min_words)

    def _backend(texts: list[str]) -> list[list[str]]:
        return splitter(texts)

    return _backend


__all__ = [
    "rule_connective_backend",
    "pos_connective_backend",
]
