"""Per-language punctuation configuration for text splitting."""

from __future__ import annotations

from typing import Mapping


# Sentence-terminal punctuation per language.
# Languages not listed fall back to "default".
SENTENCE_TERMINALS: Mapping[str, frozenset[str]] = {
    "default": frozenset({".", "!", "?"}),
    "zh": frozenset({"。", "！", "？"}),
    "ja": frozenset({"。", "！", "？"}),
    "ko": frozenset({".", "。", "!", "?"}),
}

# Clause-separator punctuation per language.
CLAUSE_SEPARATORS: Mapping[str, frozenset[str]] = {
    "default": frozenset({",", ";", ":", "\u2014"}),
    "zh": frozenset({"，", "、", "；", "："}),
    "ja": frozenset({"、", "；"}),
    "ko": frozenset({",", "；"}),
}

ABBREVIATIONS: frozenset[str] = frozenset({
    "Mr", "Mrs", "Ms", "Dr", "Prof", "Sr", "Jr", "St",
    "Inc", "Ltd", "Co", "Corp", "vs", "etc", "eg", "ie",
    "Jan", "Feb", "Mar", "Apr", "Jun", "Jul", "Aug", "Sep",
    "Oct", "Nov", "Dec",
})


def get_sentence_terminators(language: str) -> frozenset[str]:
    return SENTENCE_TERMINALS.get(language, SENTENCE_TERMINALS["default"])


def get_clause_separators(language: str) -> frozenset[str]:
    return CLAUSE_SEPARATORS.get(language, CLAUSE_SEPARATORS["default"])
