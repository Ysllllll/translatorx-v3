"""Japanese (ja) splitter tests."""

from lang_ops import TextOps
from lang_ops._core._types import Span
from lang_ops.splitter._sentence import split_sentences
from lang_ops.splitter._clause import split_clauses
from ._base import SplitterTestBase


TEXT_SAMPLE: str = '日本の伝統文化と現代テクノロジーは、一見すると相反するもののように思える。しかし実際には、両者は深い関係にあり、多くの場面で融合が進んでいる。例えば京都の古い寺院では、AIを活用した観光ガイドが導入されている；また、書道の世界でもデジタルペンを使った新しい表現方法が注目を集めている。ある書道家は次のように語った。「技術は人間の感性を拡張するものである！」この言葉は、多くのクリエイターの共感を呼んだ……伝統工芸の分野でも、3Dプリンターを活用した新しい技法が開発されつつある。果たしてテクノロジーは伝統を破壊するのか、それとも新たな可能性を切り開くのか？専門家の意見は大きく分かれている。ある人類学者は「デジタルとアナログの融合は避けられない」と主張している！他方で、ある伝統工芸の職人は「手仕事の温もりは機械には出せない」と強く訴えている。しかし両者の対話を通じて、予想もしなかった革新的な作品が生まれることもある。例えば和紙とLED技術を融合した照明器具は、国内外で非常に高い評価を得ている。伝統を守りながら革新を取り入れる、この絶妙なバランスこそが日本文化の真髄ではないだろうか。'

PARAGRAPH_TEXT: str = '日本の伝統文化と現代テクノロジーは密接な関係にある。\n\n新しい表現方法が注目を集めている。\n\n伝統を守ることが大切だ。'

_ops = TextOps.for_language("ja")


def _s(text: str) -> list[str]:
    return Span.to_texts(split_sentences(
        text, _ops.sentence_terminators, _ops.abbreviations, is_cjk=True,
        strip_spaces=True,
    ))


def _c(text: str) -> list[str]:
    return Span.to_texts(split_clauses(text, _ops.clause_separators))


class TestJapaneseSplitter(SplitterTestBase):
    LANGUAGE = "ja"
    TEXT_SAMPLE = TEXT_SAMPLE
    PARAGRAPH_TEXT = PARAGRAPH_TEXT

    # ── split_sentences() ─────────────────────────────────────────────

    def test_split_sentences(self) -> None:
        assert _s("今日は。いい天気！") == ["今日は。", "いい天気！"]

    def test_split_sentences_consecutive_terminators(self) -> None:
        assert _s("すごい！！！本当？？？") == ["すごい！！！", "本当？？？"]
        assert _s("何！？どうして？！") == ["何！？", "どうして？！"]

    def test_split_sentences_halfwidth_terminators(self) -> None:
        assert _ops.split_sentences("Hello! テスト? はい。") == ["Hello!", "テスト?", "はい。"]

    def test_split_sentences_whitespace_stripping(self) -> None:
        assert _ops.split_sentences("  第一文。 第二文！ ") == ["第一文。", "第二文！"]

    def test_split_sentences_ops_shortcut(self) -> None:
        assert _ops.split_sentences("今日は。いい天気！") == ["今日は。", "いい天気！"]

    def test_split_sentences_long_text(self) -> None:
        assert self._split_sentences() == [
            '日本の伝統文化と現代テクノロジーは、一見すると相反するもののように思える。',
            'しかし実際には、両者は深い関係にあり、多くの場面で融合が進んでいる。',
            '例えば京都の古い寺院では、AIを活用した観光ガイドが導入されている；また、書道の世界でもデジタルペンを使った新しい表現方法が注目を集めている。',
            'ある書道家は次のように語った。',
            '「技術は人間の感性を拡張するものである！」',
            'この言葉は、多くのクリエイターの共感を呼んだ……伝統工芸の分野でも、3Dプリンターを活用した新しい技法が開発されつつある。',
            '果たしてテクノロジーは伝統を破壊するのか、それとも新たな可能性を切り開くのか？',
            '専門家の意見は大きく分かれている。',
            'ある人類学者は「デジタルとアナログの融合は避けられない」と主張している！',
            '他方で、ある伝統工芸の職人は「手仕事の温もりは機械には出せない」と強く訴えている。',
            'しかし両者の対話を通じて、予想もしなかった革新的な作品が生まれることもある。',
            '例えば和紙とLED技術を融合した照明器具は、国内外で非常に高い評価を得ている。',
            '伝統を守りながら革新を取り入れる、この絶妙なバランスこそが日本文化の真髄ではないだろうか。',
        ]

    # ── split_clauses() ──────────────────────────────────────────────

    def test_split_clauses(self) -> None:
        assert _c("今日は、いい天気ですね") == ["今日は、", "いい天気ですね"]

    def test_split_clauses_consecutive_separators(self) -> None:
        assert _c("、、、") == ["、、、"]
        assert _c("テスト、、次") == ["テスト、、", "次"]

    def test_split_clauses_halfwidth_comma(self) -> None:
        assert _ops.split_clauses("Hello, テスト, はい。") == ["Hello,", "テスト,", "はい。"]

    def test_split_clauses_ops_shortcut(self) -> None:
        assert _ops.split_clauses("今日は、いい天気ですね") == ["今日は、", "いい天気ですね"]

    def test_split_clauses_long_text(self) -> None:
        assert self._split_clauses() == [
            '日本の伝統文化と現代テクノロジーは、',
            '一見すると相反するもののように思える。',
            'しかし実際には、',
            '両者は深い関係にあり、',
            '多くの場面で融合が進んでいる。',
            '例えば京都の古い寺院では、',
            'AIを活用した観光ガイドが導入されている；',
            'また、',
            '書道の世界でもデジタルペンを使った新しい表現方法が注目を集めている。',
            'ある書道家は次のように語った。',
            '「技術は人間の感性を拡張するものである！」',
            'この言葉は、',
            '多くのクリエイターの共感を呼んだ……伝統工芸の分野でも、',
            '3Dプリンターを活用した新しい技法が開発されつつある。',
            '果たしてテクノロジーは伝統を破壊するのか、',
            'それとも新たな可能性を切り開くのか？',
            '専門家の意見は大きく分かれている。',
            'ある人類学者は「デジタルとアナログの融合は避けられない」と主張している！',
            '他方で、',
            'ある伝統工芸の職人は「手仕事の温もりは機械には出せない」と強く訴えている。',
            'しかし両者の対話を通じて、',
            '予想もしなかった革新的な作品が生まれることもある。',
            '例えば和紙とLED技術を融合した照明器具は、',
            '国内外で非常に高い評価を得ている。',
            '伝統を守りながら革新を取り入れる、',
            'この絶妙なバランスこそが日本文化の真髄ではないだろうか。',
        ]

    # ── ChunkPipeline ────────────────────────────────────────────────

    def test_pipeline_sentences_clauses(self) -> None:
        assert self._pipeline_sentences_clauses() == [
            '日本の伝統文化と現代テクノロジーは、',
            '一見すると相反するもののように思える。',
            'しかし実際には、',
            '両者は深い関係にあり、',
            '多くの場面で融合が進んでいる。',
            '例えば京都の古い寺院では、',
            'AIを活用した観光ガイドが導入されている；',
            'また、',
            '書道の世界でもデジタルペンを使った新しい表現方法が注目を集めている。',
            'ある書道家は次のように語った。',
            '「技術は人間の感性を拡張するものである！」',
            'この言葉は、',
            '多くのクリエイターの共感を呼んだ……伝統工芸の分野でも、',
            '3Dプリンターを活用した新しい技法が開発されつつある。',
            '果たしてテクノロジーは伝統を破壊するのか、',
            'それとも新たな可能性を切り開くのか？',
            '専門家の意見は大きく分かれている。',
            'ある人類学者は「デジタルとアナログの融合は避けられない」と主張している！',
            '他方で、',
            'ある伝統工芸の職人は「手仕事の温もりは機械には出せない」と強く訴えている。',
            'しかし両者の対話を通じて、',
            '予想もしなかった革新的な作品が生まれることもある。',
            '例えば和紙とLED技術を融合した照明器具は、',
            '国内外で非常に高い評価を得ている。',
            '伝統を守りながら革新を取り入れる、',
            'この絶妙なバランスこそが日本文化の真髄ではないだろうか。',
        ]

    def test_pipeline_paragraphs_sentences(self) -> None:
        assert self._pipeline_paragraphs_sentences() == [
            '日本の伝統文化と現代テクノロジーは密接な関係にある。',
            '新しい表現方法が注目を集めている。',
            '伝統を守ることが大切だ。',
        ]
