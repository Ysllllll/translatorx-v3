"""Tests for ``ops.connectives`` per-language data."""

from __future__ import annotations

import pytest

from domain.lang import LangOps


SUPPORTED = ["en", "zh", "ja", "ko", "es", "fr", "de", "pt", "ru", "vi"]


class TestConnectives:
    @pytest.mark.parametrize("lang", SUPPORTED)
    def test_connectives_is_frozenset(self, lang: str) -> None:
        ops = LangOps.for_language(lang)
        assert isinstance(ops.connectives, frozenset)

    @pytest.mark.parametrize("lang", SUPPORTED)
    def test_connectives_lowercase_ascii_languages(self, lang: str) -> None:
        if lang in ("zh", "ja", "ko"):
            return  # CJK not affected by case
        ops = LangOps.for_language(lang)
        for w in ops.connectives:
            assert w == w.lower(), f"{lang}: {w!r} should be lowercase"

    @pytest.mark.parametrize("lang", SUPPORTED)
    def test_connectives_non_empty(self, lang: str) -> None:
        ops = LangOps.for_language(lang)
        assert len(ops.connectives) > 0

    def test_en_contains_because(self) -> None:
        assert "because" in LangOps.for_language("en").connectives

    def test_en_excludes_ambiguous(self) -> None:
        """`that`, `and`, `or`, `which` are too ambiguous for lexical rule."""
        c = LangOps.for_language("en").connectives
        for w in ("that", "and", "or", "which", "who"):
            assert w not in c, f"{w!r} should not be in en connectives"

    def test_zh_contains_common(self) -> None:
        c = LangOps.for_language("zh").connectives
        for w in ("因为", "所以", "但是", "如果"):
            assert w in c

    def test_ja_contains_common(self) -> None:
        c = LangOps.for_language("ja").connectives
        for w in ("しかし", "だから"):
            assert w in c

    def test_ko_contains_common(self) -> None:
        c = LangOps.for_language("ko").connectives
        for w in ("하지만", "그래서"):
            assert w in c

    def test_unknown_language_defaults(self) -> None:
        # Direct EnTypeOps with unknown code falls through to en default.
        from domain.lang.en_type import EnTypeOps

        ops = EnTypeOps(language="unknown")
        assert "because" in ops.connectives
