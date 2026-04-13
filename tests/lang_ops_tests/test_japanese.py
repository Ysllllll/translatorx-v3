from lang_ops import TextOps, mecab_is_available

from .conftest import TEST_FONT_PATH, expected_pixel_length
from tests.lang_ops_tests._base import TextOpsTestCase


class JapaneseTextTest(TextOpsTestCase):
    def setUp(self) -> None:
        self.assertTrue(mecab_is_available())
        self.ops = TextOps.for_language("ja")
        return super().setUp()

    def test_japanese(self) -> None:
        o = self.ops
        text0 = "米軍司令官 “1万か所超える標的を攻撃”司令官は25日、作戦の進捗（しんちょく）を説明ク・タこんにちは、世界！AIengine今日は AI 字幕エンジン 日本語    English Subtitle 自動 整列 します。"
        expected_join_text0 = "米軍司令官“1 万か所超える標的を攻撃”司令官は 25 日、作戦の進捗（しんちょく）を説明ク・タこんにちは、世界！AIengine 今日は AI 字幕エンジン日本語 English Subtitle 自動整列します。"
        expected_character_tokens = ["米", "軍", "司", "令", "官", "“1", "万", "か", "所", "超", "え", "る", "標", "的", "を", "攻", "撃”", "司", "令", "官", "は", "25", "日、", "作", "戦", "の", "進", "捗", "（し", "ん", "ち", "ょ", "く）", "を", "説", "明", "ク", "・", "タ", "こ", "ん", "に", "ち", "は、", "世", "界！", "AIengine", "今", "日", "は", "AI", "字", "幕", "エ", "ン", "ジ", "ン", "日", "本", "語", "English", "Subtitle", "自", "動", "整", "列", "し", "ま", "す。"]
        expected_character_tokens_without_punctuation = ["米", "軍", "司", "令", "官", "“", "1", "万", "か", "所", "超", "え", "る", "標", "的", "を", "攻", "撃", "”", "司", "令", "官", "は", "25", "日", "、", "作", "戦", "の", "進", "捗", "（", "し", "ん", "ち", "ょ", "く", "）", "を", "説", "明", "ク", "・", "タ", "こ", "ん", "に", "ち", "は", "、", "世", "界", "！", "AIengine", "今", "日", "は", "AI", "字", "幕", "エ", "ン", "ジ", "ン", "日", "本", "語", "English", "Subtitle", "自", "動", "整", "列", "し", "ま", "す", "。"]
        mecab_word_tokens = ["米軍", "司令", "官", "“1", "万", "か所", "超える", "標的", "を", "攻撃”", "司令", "官", "は", "25", "日、", "作戦", "の", "進捗", "（しん", "ちょく）", "を", "説明", "ク", "・", "タ", "こんにちは、", "世界！", "AIengine", "今日", "は", "AI", "字幕", "エンジン", "日本", "語", "English", "Subtitle", "自動", "整列", "し", "ます。"]
        mecab_word_tokens_without_punctuation = ["米軍", "司令", "官", "“", "1", "万", "か所", "超える", "標的", "を", "攻撃", "”", "司令", "官", "は", "25", "日", "、", "作戦", "の", "進捗", "（", "しん", "ちょく", "）", "を", "説明", "ク", "・", "タ", "こんにちは", "、", "世界", "！", "AIengine", "今日", "は", "AI", "字幕", "エンジン", "日本", "語", "English", "Subtitle", "自動", "整列", "し", "ます", "。"]
        actual_vs_expect = [
            [o.split(text0), mecab_word_tokens],
            [o.split(text0, mode="character"), expected_character_tokens],
            [o.split(text0, attach_punctuation=False), mecab_word_tokens_without_punctuation],
            [o.split(text0, mode="character", attach_punctuation=False), expected_character_tokens_without_punctuation],
            [o.split(text0, mode="word"), mecab_word_tokens],
            [o.split(text0, mode="word", attach_punctuation=False), mecab_word_tokens_without_punctuation],
            [o.length(text0), 99],
            [o.length(text0, cjk_width=2), 85],
            [o.plength(text0, TEST_FONT_PATH, 16), expected_pixel_length(text0, TEST_FONT_PATH, 16)],
        ]
        self.assert_actual_vs_expect(actual_vs_expect)
        self._assert_text_join_case(text0, expected_join_text0)

        mixed_text = "これはI'mの例で、deeplearning.aiとhttps://www.comを使う。"
        self._assert_preserved_fragments(
            mixed_text,
            ["I'm", "deeplearning.ai", "https://www.com"],
            modes=("word", "character"),
        )

        # split()
        self.assertEqual(o.split('こんにちは世界'), ['こんにちは', '世界'])

        # length()
        self.assertEqual(o.length('こんにちは世界'), 7)

        # join()
        self.assertEqual(o.join(['こんにちは', '世界']), 'こんにちは世界')

        # length() with cjk_width
        self.assertEqual(o.length("こんにちは世界"), 7)
        self.assertEqual(o.length("こんにちは世界", cjk_width=2), 7)
        self.assertEqual(o.length("hello", cjk_width=1), 5)
        self.assertEqual(o.length("hello", cjk_width=2), 3)
        self.assertEqual(o.length("こんにちはhello", cjk_width=1), 10)
        self.assertEqual(o.length("こんにちは hello", cjk_width=1), 10)
        self.assertEqual(o.length("こんにちはhello", cjk_width=2), 8)
        self.assertEqual(o.length("こんにちは hello", cjk_width=2), 8)
        self.assertEqual(o.length("AIエンジンOK", cjk_width=1), 8)
        self.assertEqual(o.length("AI エンジンOK", cjk_width=1), 8)
        self.assertEqual(o.length("AIエンジンOK", cjk_width=2), 6)
        self.assertEqual(o.length("AI エンジン OK", cjk_width=2), 6)

        # strip()
        self.assertEqual(o.strip('  こんにちは  '), 'こんにちは')
        self.assertEqual(o.strip(''), '')
        self.assertEqual(o.strip('。こんにちは！', '。！'), 'こんにちは')
        self.assertEqual(o.strip('...こんにちは...', '.'), 'こんにちは')
        self.assertEqual(o.strip('「日本」', '「」'), '日本')
        self.assertEqual(o.strip('こんにちは', '！'), 'こんにちは')
        self.assertEqual(o.strip('', '！'), '')

        # lstrip()
        self.assertEqual(o.lstrip('  こんにちは  '), 'こんにちは  ')
        self.assertEqual(o.lstrip(''), '')
        self.assertEqual(o.lstrip('「日本」', '「'), '日本」')
        self.assertEqual(o.lstrip('。こんにちは！', '。'), 'こんにちは！')

        # rstrip()
        self.assertEqual(o.rstrip('  こんにちは  '), '  こんにちは')
        self.assertEqual(o.rstrip(''), '')
        self.assertEqual(o.rstrip('世界。', '。'), '世界')
        self.assertEqual(o.rstrip('こんにちは！', '！'), 'こんにちは')

        # strip_punc()
        self.assertEqual(o.strip_punc('。こんにちは世界！'), 'こんにちは世界')
        self.assertEqual(o.strip_punc('「日本語」'), '日本語')
        self.assertEqual(o.strip_punc(''), '')
        self.assertEqual(o.strip_punc('。。！！'), '')

        # lstrip_punc()
        self.assertEqual(o.lstrip_punc('。こんにちは世界！'), 'こんにちは世界！')
        self.assertEqual(o.lstrip_punc('「日本語」'), '日本語」')
        self.assertEqual(o.lstrip_punc(''), '')
        self.assertEqual(o.lstrip_punc('こんにちは'), 'こんにちは')

        # rstrip_punc()
        self.assertEqual(o.rstrip_punc('。こんにちは世界！'), '。こんにちは世界')
        self.assertEqual(o.rstrip_punc('「日本語」'), '「日本語')
        self.assertEqual(o.rstrip_punc(''), '')
        self.assertEqual(o.rstrip_punc('こんにちは'), 'こんにちは')

        # restore_punc()
        self.assertEqual(o.restore_punc('こんにちは世界', 'こんにちは、世界！'), 'こんにちは、世界！')
        self.assertEqual(o.restore_punc('こんにちは世界', 'こんにちは世界'), 'こんにちは世界')
        self.assertEqual(o.restore_punc('', ''), '')

        # edge()
        self._assert_cjk_edge('あ')

        # mode()
        self._assert_cjk_mode('こんにちは世界')

        # normalize()
        self._assert_cjk_normalize('こんにちは', 'こんにちは、世界！「テスト」。')
