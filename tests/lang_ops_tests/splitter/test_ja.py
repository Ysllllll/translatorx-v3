"""Japanese (ja) splitter tests."""

from lang_ops import LangOps, ChunkPipeline
from ._base import SplitterTestBase


TEXT_SAMPLE: str = '日本の伝統文化と現代テクノロジーは、一見すると相反するもののように思える。しかし実際には、両者は深い関係にあり、多くの場面で融合が進んでいる。例えば京都の古い寺院では、AIを活用した観光ガイドが導入されている；また、書道の世界でもデジタルペンを使った新しい表現方法が注目を集めている。ある書道家は次のように語った。「技術は人間の感性を拡張するものである！」この言葉は、多くのクリエイターの共感を呼んだ……伝統工芸の分野でも、3Dプリンターを活用した新しい技法が開発されつつある。果たしてテクノロジーは伝統を破壊するのか、それとも新たな可能性を切り開くのか？専門家の意見は大きく分かれている。ある人類学者は「デジタルとアナログの融合は避けられない」と主張している！他方で、ある伝統工芸の職人は「手仕事の温もりは機械には出せない」と強く訴えている。しかし両者の対話を通じて、予想もしなかった革新的な作品が生まれることもある。例えば和紙とLED技術を融合した照明器具は、国内外で非常に高い評価を得ている。伝統を守りながら革新を取り入れる、この絶妙なバランスこそが日本文化の真髄ではないだろうか。'

PARAGRAPH_TEXT: str = '日本の伝統文化と現代テクノロジーは密接な関係にある。\n\n新しい表現方法が注目を集めている。\n\n伝統を守ることが大切だ。'

_ops = LangOps.for_language("ja")


class TestJapaneseSplitter(SplitterTestBase):
    LANGUAGE = "ja"
    TEXT_SAMPLE = TEXT_SAMPLE
    PARAGRAPH_TEXT = PARAGRAPH_TEXT

    def test_split_sentences(self) -> None:
        # 基本的な文分割
        assert _ops.split_sentences("今日は。いい天気！") == ["今日は。", "いい天気！"]
        assert _ops.split_sentences("すごい！本当？はい。") == ["すごい！", "本当？", "はい。"]

        # 連続する終端記号
        assert _ops.split_sentences("すごい！！！本当？？？") == ["すごい！！！", "本当？？？"]
        assert _ops.split_sentences("何！？どうして？！") == ["何！？", "どうして？！"]

        # 半角終端記号
        assert _ops.split_sentences("Hello! テスト? はい。") == ["Hello!", "テスト?", "はい。"]

        # 閉じ引用符
        assert _ops.split_sentences("「こんにちは！」彼は言った。") == ["「こんにちは！」", "彼は言った。"]

        # 絵文字
        assert _ops.split_sentences("すごい😊！本当👋？") == ["すごい😊！", "本当👋？"]

        # 空白の処理（日本語ではスペース除去）
        assert _ops.split_sentences("  第一文。 第二文！ ") == ["第一文。", "第二文！"]

        # 端のケース
        assert _ops.split_sentences("") == []
        assert _ops.split_sentences("ターミネータなし") == ["ターミネータなし"]

    def test_split_clauses(self) -> None:
        # 基本的な節分割
        assert _ops.split_clauses("今日は、いい天気ですね") == ["今日は、", "いい天気ですね"]
        assert _ops.split_clauses("最初、次、最後。") == ["最初、", "次、", "最後。"]

        # セミコロン
        assert _ops.split_clauses("まず；次に：最後に。") == ["まず；", "次に：最後に。"]

        # 半角カンマ
        assert _ops.split_clauses("Hello, テスト, はい。") == ["Hello,", "テスト,", "はい。"]

        # 連続する区切り文字
        assert _ops.split_clauses("、、、") == ["、、、"]
        assert _ops.split_clauses("テスト、、次") == ["テスト、、", "次"]

        # 端のケース
        assert _ops.split_clauses("") == []
        assert _ops.split_clauses("セパレータなし") == ["セパレータなし"]

    def test_split_by_length(self) -> None:
        # 文字数による分割
        assert _ops.split_by_length("これはテストです", max_length=4) == ["これは", "テスト", "です"]
        assert _ops.split_by_length("今日はいい天気ですね", max_length=5) == ["今日はいい", "天気ですね"]

        # フィット / 空 / 端のケース
        assert _ops.split_by_length("テスト", max_length=10) == ["テスト"]
        assert _ops.split_by_length("", max_length=10) == []

        # エラー
        import pytest
        with pytest.raises(ValueError):
            _ops.split_by_length("テスト", max_length=0)
        with pytest.raises(ValueError):
            _ops.split_by_length("テスト", max_length=-1)
        with pytest.raises(TypeError):
            _ops.split_by_length("テスト", max_length=5, unit="sentence")

        # チャンクチェーン
        assert _ops.chunk("これはテストです。次の文です。").sentences().by_length(8).result() == [
            "これはテスト", "です。", "次の文です。",
        ]
        assert _ops.chunk("最初、次、最後。").clauses().by_length(4).result() == [
            "最初、", "次、", "最後。",
        ]

    def test_split_long_text(self) -> None:
        # long text split_sentences()
        assert _ops.split_sentences(self.TEXT_SAMPLE) == [
            '日本の伝統文化と現代テクノロジーは、一見すると相反するもののように思える。',
            'しかし実際には、両者は深い関係にあり、多くの場面で融合が進んでいる。',
            '例えば京都の古い寺院では、AI を活用した観光ガイドが導入されている；また、書道の世界でもデジタルペンを使った新しい表現方法が注目を集めている。',
            'ある書道家は次のように語った。',
            '「技術は人間の感性を拡張するものである！」',
            'この言葉は、多くのクリエイターの共感を呼んだ……伝統工芸の分野でも、3 D プリンターを活用した新しい技法が開発されつつある。',
            '果たしてテクノロジーは伝統を破壊するのか、それとも新たな可能性を切り開くのか？',
            '専門家の意見は大きく分かれている。',
            'ある人類学者は「デジタルとアナログの融合は避けられない」と主張している！',
            '他方で、ある伝統工芸の職人は「手仕事の温もりは機械には出せない」と強く訴えている。',
            'しかし両者の対話を通じて、予想もしなかった革新的な作品が生まれることもある。',
            '例えば和紙と LED 技術を融合した照明器具は、国内外で非常に高い評価を得ている。',
            '伝統を守りながら革新を取り入れる、この絶妙なバランスこそが日本文化の真髄ではないだろうか。',
        ]

        # long text split_clauses()
        assert _ops.split_clauses(self.TEXT_SAMPLE) == [
            '日本の伝統文化と現代テクノロジーは、',
            '一見すると相反するもののように思える。',
            'しかし実際には、',
            '両者は深い関係にあり、',
            '多くの場面で融合が進んでいる。',
            '例えば京都の古い寺院では、',
            'AI を活用した観光ガイドが導入されている；',
            'また、',
            '書道の世界でもデジタルペンを使った新しい表現方法が注目を集めている。',
            'ある書道家は次のように語った。',
            '「技術は人間の感性を拡張するものである！」',
            'この言葉は、',
            '多くのクリエイターの共感を呼んだ……伝統工芸の分野でも、',
            '3 D プリンターを活用した新しい技法が開発されつつある。',
            '果たしてテクノロジーは伝統を破壊するのか、',
            'それとも新たな可能性を切り開くのか？',
            '専門家の意見は大きく分かれている。',
            'ある人類学者は「デジタルとアナログの融合は避けられない」と主張している！',
            '他方で、',
            'ある伝統工芸の職人は「手仕事の温もりは機械には出せない」と強く訴えている。',
            'しかし両者の対話を通じて、',
            '予想もしなかった革新的な作品が生まれることもある。',
            '例えば和紙と LED 技術を融合した照明器具は、',
            '国内外で非常に高い評価を得ている。',
            '伝統を守りながら革新を取り入れる、',
            'この絶妙なバランスこそが日本文化の真髄ではないだろうか。',
        ]

        # long text chunk chain equivalence
        assert ChunkPipeline(self.TEXT_SAMPLE, language=self.LANGUAGE).sentences().result() == _ops.split_sentences(self.TEXT_SAMPLE)
        assert ChunkPipeline(self.TEXT_SAMPLE, language=self.LANGUAGE).sentences().clauses().result() == _ops.split_clauses(self.TEXT_SAMPLE)
        assert ChunkPipeline(self.TEXT_SAMPLE, language=self.LANGUAGE).clauses().result() == _ops.split_clauses(self.TEXT_SAMPLE)

        # long text pipeline_paragraphs_sentences()
        assert ChunkPipeline(self.PARAGRAPH_TEXT, language=self.LANGUAGE).paragraphs().sentences().result() == [
            '日本の伝統文化と現代テクノロジーは密接な関係にある。',
            '新しい表現方法が注目を集めている。',
            '伝統を守ることが大切だ。',
        ]
