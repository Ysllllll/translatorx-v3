from lang_ops import TextOps, jieba_is_available

from .conftest import TEST_FONT_PATH, expected_pixel_length
from tests.lang_ops_tests._base import TextOpsTestCase


class ChineseTextTest(TextOpsTestCase):
    def setUp(self) -> None:
        self.assertTrue(jieba_is_available())
        self.ops = TextOps.for_language("zh")
        return super().setUp()

    def test_chinese(self) -> None:
        o = self.ops
        # complex text
        text1 = '今天AI Deep上线，你应该去体验一下最新的   Agent！叫“AI”什么？叫：“Deep”，是的...'
        self.assert_actual_vs_expect([
            [o.split(text1), ['今天', 'AI', 'Deep', '上线，', '你', '应该', '去', '体验', '一下', '最新', '的', 'Agent！', '叫', '“AI”', '什么？', '叫：', '“Deep”，', '是', '的', '...']],
            [o.split(text1, mode="character"), ['今', '天', 'AI', 'Deep', '上', '线，', '你', '应', '该', '去', '体', '验', '一', '下', '最', '新', '的', 'Agent！', '叫', '“AI”', '什', '么？', '叫：', '“Deep”，', '是', '的...']],
            [o.split(text1, attach_punctuation=False), ['今天', 'AI', 'Deep', '上线', '，', '你', '应该', '去', '体验', '一下', '最新', '的', 'Agent', '！', '叫', '“', 'AI', '”', '什么', '？', '叫', '：', '“', 'Deep', '”', '，', '是', '的', '...']],
            [o.split(text1, mode="character", attach_punctuation=False), ['今', '天', 'AI', 'Deep', '上', '线', '，', '你', '应', '该', '去', '体', '验', '一', '下', '最', '新', '的', 'Agent', '！', '叫', '“', 'AI', '”', '什', '么', '？', '叫', '：', '“', 'Deep', '”', '，', '是', '的', '...']],
            [o.split(text1, mode="word"), ['今天', 'AI', 'Deep', '上线，', '你', '应该', '去', '体验', '一下', '最新', '的', 'Agent！', '叫', '“AI”', '什么？', '叫：', '“Deep”，', '是', '的', '...']],
            [o.split(text1, mode="word", attach_punctuation=False), ['今天', 'AI', 'Deep', '上线', '，', '你', '应该', '去', '体验', '一下', '最新', '的', 'Agent', '！', '叫', '“', 'AI', '”', '什么', '？', '叫', '：', '“', 'Deep', '”', '，', '是', '的', '...']],
            [o.length(text1), 50],
            [o.length(text1, cjk_width=2), 39],
            [o.plength(text1, TEST_FONT_PATH, 16), expected_pixel_length(text1, TEST_FONT_PATH, 16)],
        ])
        expect_join_text = '今天 AI Deep 上线，你应该去体验一下最新的 Agent！叫“AI”什么？叫：“Deep”，是的...'
        self._assert_text_join_case(text1, expect_join_text)

        text2 = '[旁白]：突然天边飘来了一朵乌云，那乌云之中...他说：“突然天边飘来了一朵乌云，那乌云之中...”《三体》很好看。“AI”上线了！（测试）……开始这是   English subtitle，你知道吗？'
        self.assert_actual_vs_expect([
            [o.split(text2), ['[旁白]：', '突然', '天边', '飘来', '了', '一朵', '乌云，', '那', '乌云', '之中', '...', '他', '说：', '“突然', '天边', '飘来', '了', '一朵', '乌云，', '那', '乌云', '之中', '...”', '《三体》', '很', '好看。', '“AI”', '上线', '了！', '（测试）', '…', '…', '开始', '这是', 'English', 'subtitle，', '你', '知道', '吗？']],
            [o.split(text2, mode="character"), ['[旁', '白]：', '突', '然', '天', '边', '飘', '来', '了', '一', '朵', '乌', '云，', '那', '乌', '云', '之', '中...', '他', '说：', '“突', '然', '天', '边', '飘', '来', '了', '一', '朵', '乌', '云，', '那', '乌', '云', '之', '中...”', '《三', '体》', '很', '好', '看。', '“AI”', '上', '线', '了！', '（测', '试）', '…', '…', '开', '始', '这', '是', 'English', 'subtitle，', '你', '知', '道', '吗？']],
            [o.split(text2, attach_punctuation=False), ['[', '旁白', ']', '：', '突然', '天边', '飘来', '了', '一朵', '乌云', '，', '那', '乌云', '之中', '...', '他', '说', '：', '“', '突然', '天边', '飘来', '了', '一朵', '乌云', '，', '那', '乌云', '之中', '...', '”', '《', '三体', '》', '很', '好看', '。', '“', 'AI', '”', '上线', '了', '！', '（', '测试', '）', '…', '…', '开始', '这是', 'English', 'subtitle', '，', '你', '知道', '吗', '？']],
            [o.split(text2, mode="character", attach_punctuation=False), ['[', '旁', '白', ']', '：', '突', '然', '天', '边', '飘', '来', '了', '一', '朵', '乌', '云', '，', '那', '乌', '云', '之', '中', '...', '他', '说', '：', '“', '突', '然', '天', '边', '飘', '来', '了', '一', '朵', '乌', '云', '，', '那', '乌', '云', '之', '中', '...', '”', '《', '三', '体', '》', '很', '好', '看', '。', '“', 'AI', '”', '上', '线', '了', '！', '（', '测', '试', '）', '…', '…', '开', '始', '这', '是', 'English', 'subtitle', '，', '你', '知', '道', '吗', '？']],
            [o.split(text2, mode="word"), ['[旁白]：', '突然', '天边', '飘来', '了', '一朵', '乌云，', '那', '乌云', '之中', '...', '他', '说：', '“突然', '天边', '飘来', '了', '一朵', '乌云，', '那', '乌云', '之中', '...”', '《三体》', '很', '好看。', '“AI”', '上线', '了！', '（测试）', '…', '…', '开始', '这是', 'English', 'subtitle，', '你', '知道', '吗？']],
            [o.split(text2, mode="word", attach_punctuation=False), ['[', '旁白', ']', '：', '突然', '天边', '飘来', '了', '一朵', '乌云', '，', '那', '乌云', '之中', '...', '他', '说', '：', '“', '突然', '天边', '飘来', '了', '一朵', '乌云', '，', '那', '乌云', '之中', '...', '”', '《', '三体', '》', '很', '好看', '。', '“', 'AI', '”', '上线', '了', '！', '（', '测试', '）', '…', '…', '开始', '这是', 'English', 'subtitle', '，', '你', '知道', '吗', '？']],
            [o.length(text2), 97],
            [o.length(text2, cjk_width=2), 85],
            [o.plength(text2, TEST_FONT_PATH, 16), expected_pixel_length(text2, TEST_FONT_PATH, 16)],
        ])
        expect_text2 = '[旁白]：突然天边飘来了一朵乌云，那乌云之中...他说：“突然天边飘来了一朵乌云，那乌云之中...”《三体》很好看。“AI”上线了！（测试）……开始这是 English subtitle，你知道吗？'
        self._assert_text_join_case(text2, expect_text2)

        mixed_text = "这是I'm的例子，访问deeplearning.ai，地址是https://www.com。"
        self._assert_preserved_fragments(
            mixed_text,
            ["I'm", "deeplearning.ai", "https://www.com"],
            modes=("word", "character"),
        )

        # split()
        self.assertEqual(o.split('你好世界'), ['你好', '世界'])

        # length()
        self.assertEqual(o.length("你好世界"), 4)
        self.assertEqual(o.length("你好世界", cjk_width=2), 4)
        self.assertEqual(o.length("hello", cjk_width=1), 5)
        self.assertEqual(o.length("hello", cjk_width=2), 3)
        self.assertEqual(o.length("你好hello", cjk_width=1), 7)
        self.assertEqual(o.length("你好 hello", cjk_width=1), 7)
        self.assertEqual(o.length("你好hello", cjk_width=2), 5)
        self.assertEqual(o.length("你好 hello", cjk_width=2), 5)
        self.assertEqual(o.length("AI引擎OK", cjk_width=1), 6)
        self.assertEqual(o.length("AI 引擎OK", cjk_width=1), 6)
        self.assertEqual(o.length("AI引擎OK", cjk_width=2), 4)
        self.assertEqual(o.length("AI 引擎 OK", cjk_width=2), 4)

        # join()
        self.assertEqual(o.join(['你好', '世界']), '你好世界')

        # strip()
        self.assertEqual(o.strip('  你好  '), '你好')
        self.assertEqual(o.strip(''), '')
        self.assertEqual(o.strip('，你好！', '，！'), '你好')
        self.assertEqual(o.strip('...你好...', '.'), '你好')
        self.assertEqual(o.strip('《三体》', '《》'), '三体')
        self.assertEqual(o.strip('你好', '！'), '你好')
        self.assertEqual(o.strip('', '！'), '')

        # lstrip()
        self.assertEqual(o.lstrip('  你好  '), '你好  ')
        self.assertEqual(o.lstrip(''), '')
        self.assertEqual(o.lstrip('《三体》', '《'), '三体》')
        self.assertEqual(o.lstrip('，你好！', '，'), '你好！')

        # rstrip()
        self.assertEqual(o.rstrip('  你好  '), '  你好')
        self.assertEqual(o.rstrip(''), '')
        self.assertEqual(o.rstrip('你好。', '。'), '你好')
        self.assertEqual(o.rstrip('你好！', '！'), '你好')

        # strip_punc()
        self.assertEqual(o.strip_punc('。你好世界！'), '你好世界')
        self.assertEqual(o.strip_punc('「你好世界」'), '你好世界')
        self.assertEqual(o.strip_punc('《三体》很好看'), '三体》很好看')
        self.assertEqual(o.strip_punc(''), '')
        self.assertEqual(o.strip_punc('，。！'), '')

        # lstrip_punc()
        self.assertEqual(o.lstrip_punc('。你好世界！'), '你好世界！')
        self.assertEqual(o.lstrip_punc('「你好世界」'), '你好世界」')
        self.assertEqual(o.lstrip_punc(''), '')
        self.assertEqual(o.lstrip_punc('你好世界'), '你好世界')

        # rstrip_punc()
        self.assertEqual(o.rstrip_punc('。你好世界！'), '。你好世界')
        self.assertEqual(o.rstrip_punc('「你好世界」'), '「你好世界')
        self.assertEqual(o.rstrip_punc(''), '')
        self.assertEqual(o.rstrip_punc('你好世界'), '你好世界')

        # restore_punc()
        self.assertEqual(o.restore_punc('你好世界', '你好，世界！'), '你好，世界！')
        self.assertEqual(o.restore_punc('测试', '（测试）'), '（测试）')
        self.assertEqual(o.restore_punc('你好世界', '你好世界'), '你好世界')
        self.assertEqual(o.restore_punc('', ''), '')
        with self.assertRaises(ValueError):
            o.restore_punc('你好世界', '你好')

        # edge()
        self._assert_cjk_edge('你')

        # mode()
        self._assert_cjk_mode('你好世界')

        # normalize()
        self._assert_cjk_normalize('你好世界', '你好，世界！这是"测试"。')
