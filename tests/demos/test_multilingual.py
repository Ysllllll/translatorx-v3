"""Multilingual coverage tests (no LLM required).

Parametrized across all 10 supported languages, these tests exercise the
core per-language text pipeline using the sample SRTs under
``demo_data/multilingual/``. The goal is a single line of defense against
a language regressing silently after a LangOps / Subtitle refactor.

LLM-driven end-to-end translation is covered by
``demos/multilingual/demo_translate.py`` (manual run against a local LLM).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from adapters.parsers import read_srt
from domain.lang import LangOps
from domain.subtitle import Subtitle


DATA_DIR = Path(__file__).resolve().parents[1].parent / "demo_data" / "multilingual"
ALL_LANGS = ("en", "zh", "ja", "ko", "de", "fr", "es", "pt", "ru", "vi")
CJK_LANGS = {"zh", "ja", "ko"}


@pytest.fixture(scope="module")
def demo_dir() -> Path:
    assert DATA_DIR.is_dir(), f"missing fixture dir: {DATA_DIR}"
    return DATA_DIR


@pytest.mark.parametrize("lang", ALL_LANGS)
def test_srt_fixture_exists(demo_dir: Path, lang: str) -> None:
    path = demo_dir / f"{lang}.srt"
    assert path.is_file(), f"missing fixture: {path}"
    segments = read_srt(path)
    assert len(segments) == 3, f"{lang}: expected 3 segments, got {len(segments)}"
    for seg in segments:
        assert seg.text.strip(), f"{lang}: empty segment text"


@pytest.mark.parametrize("lang", ALL_LANGS)
def test_langops_factory_and_flags(lang: str) -> None:
    ops = LangOps.for_language(lang)
    assert ops is not None
    assert ops.is_cjk is (lang in CJK_LANGS)


@pytest.mark.parametrize("lang", ALL_LANGS)
def test_tokenize_roundtrip(demo_dir: Path, lang: str) -> None:
    ops = LangOps.for_language(lang)
    segments = read_srt(demo_dir / f"{lang}.srt")
    for seg in segments:
        tokens = ops.split(seg.text)
        assert tokens, f"{lang}: empty token list for {seg.text!r}"
        rejoined = ops.join(tokens)
        assert rejoined == seg.text, f"{lang}: tokenize/join mismatch"


@pytest.mark.parametrize("lang", ALL_LANGS)
def test_sentence_split_covers_all_input(demo_dir: Path, lang: str) -> None:
    ops = LangOps.for_language(lang)
    segments = read_srt(demo_dir / f"{lang}.srt")
    joined = " ".join(s.text for s in segments)
    pipe = ops.chunk(joined)
    sents = pipe.sentences().result()
    assert len(sents) >= 1
    # Every input sentence should appear — our fixtures are 3 sentences joined by " ".
    # Allow the splitter to merge / punctuate, just check total length is preserved modulo whitespace.
    combined = "".join(sents).replace(" ", "")
    assert len(combined) >= len(joined.replace(" ", "")) * 0.9


@pytest.mark.parametrize("lang", ALL_LANGS)
def test_subtitle_pipeline_end_to_end(demo_dir: Path, lang: str) -> None:
    segments = read_srt(demo_dir / f"{lang}.srt")
    sub = Subtitle(segments, language=lang)
    built = sub.sentences().split(max_len=20).build()
    assert len(built) >= len(segments)
    for seg in built:
        assert seg.start <= seg.end
        assert seg.text.strip()


@pytest.mark.parametrize("lang", ALL_LANGS)
def test_length_is_nonzero(demo_dir: Path, lang: str) -> None:
    ops = LangOps.for_language(lang)
    segments = read_srt(demo_dir / f"{lang}.srt")
    for seg in segments:
        assert ops.length(seg.text) > 0
