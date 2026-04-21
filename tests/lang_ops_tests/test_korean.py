from domain.lang import LangOps, kiwi_is_available

from .conftest import TEST_FONT_PATH, expected_pixel_length
from tests.lang_ops_tests._base import LangOpsTestCase


class KoreanTextTest(LangOpsTestCase):
    def setUp(self) -> None:
        self.assertTrue(kiwi_is_available())
        self.ops = LangOps.for_language("ko")
        return super().setUp()

    def test_korean(self) -> None:
        o = self.ops
        text0 = "질문이나 건의사항은 깃헙 이슈 트래커에 남겨주세요. English Subtitle, 오류보고는 실행환경, 에러메세지와함께 설명을 최대한상세히!"
        expected_join_text0 = "질문이나 건의사항은 깃헙 이슈 트래커에 남겨주세요. English Subtitle, 오류보고는 실행환경, 에러메세지와함께 설명을 최대한상세히!"
        expected_character_tokens = [
            "질",
            "문",
            "이",
            "나",
            " ",
            "건",
            "의",
            "사",
            "항",
            "은",
            " ",
            "깃",
            "헙",
            " ",
            "이",
            "슈",
            " ",
            "트",
            "래",
            "커",
            "에",
            " ",
            "남",
            "겨",
            "주",
            "세",
            "요.",
            " ",
            "English",
            " ",
            "Subtitle,",
            " ",
            "오",
            "류",
            "보",
            "고",
            "는",
            " ",
            "실",
            "행",
            "환",
            "경,",
            " ",
            "에",
            "러",
            "메",
            "세",
            "지",
            "와",
            "함",
            "께",
            " ",
            "설",
            "명",
            "을",
            " ",
            "최",
            "대",
            "한",
            "상",
            "세",
            "히!",
        ]
        expected_character_tokens_without_punctuation = [
            "질",
            "문",
            "이",
            "나",
            " ",
            "건",
            "의",
            "사",
            "항",
            "은",
            " ",
            "깃",
            "헙",
            " ",
            "이",
            "슈",
            " ",
            "트",
            "래",
            "커",
            "에",
            " ",
            "남",
            "겨",
            "주",
            "세",
            "요",
            ".",
            " ",
            "English",
            " ",
            "Subtitle",
            ",",
            " ",
            "오",
            "류",
            "보",
            "고",
            "는",
            " ",
            "실",
            "행",
            "환",
            "경",
            ",",
            " ",
            "에",
            "러",
            "메",
            "세",
            "지",
            "와",
            "함",
            "께",
            " ",
            "설",
            "명",
            "을",
            " ",
            "최",
            "대",
            "한",
            "상",
            "세",
            "히",
            "!",
        ]
        expected_word_tokens = [
            "질문",
            "이나",
            " ",
            "건의",
            "사항",
            "은",
            " ",
            "깃헙",
            " ",
            "이슈",
            " ",
            "트래커",
            "에",
            " ",
            "남기",
            "어",
            "주",
            "세요.",
            " ",
            "English",
            " ",
            "Subtitle,",
            " ",
            "오류",
            "보고",
            "는",
            " ",
            "실행",
            "환경,",
            " ",
            "에러",
            "메세지",
            "와",
            "함께",
            " ",
            "설명",
            "을",
            " ",
            "최대한",
            "상세히!",
        ]
        expected_word_tokens_without_punctuation = [
            "질문",
            "이나",
            " ",
            "건의",
            "사항",
            "은",
            " ",
            "깃헙",
            " ",
            "이슈",
            " ",
            "트래커",
            "에",
            " ",
            "남기",
            "어",
            "주",
            "세요",
            ".",
            " ",
            "English",
            " ",
            "Subtitle",
            ",",
            " ",
            "오류",
            "보고",
            "는",
            " ",
            "실행",
            "환경",
            ",",
            " ",
            "에러",
            "메세지",
            "와",
            "함께",
            " ",
            "설명",
            "을",
            " ",
            "최대한",
            "상세히",
            "!",
        ]
        actual_vs_expect = [
            [o.split(text0), expected_word_tokens],
            [o.split(text0, mode="character"), expected_character_tokens],
            [o.split(text0, attach_punctuation=False), expected_word_tokens_without_punctuation],
            [o.split(text0, mode="character", attach_punctuation=False), expected_character_tokens_without_punctuation],
            [o.split(text0, mode="word"), expected_word_tokens],
            [o.split(text0, mode="word", attach_punctuation=False), expected_word_tokens_without_punctuation],
            [o.length(text0), 67],
            [o.length(text0, cjk_width=2), 60],
            [o.plength(text0, TEST_FONT_PATH, 16), expected_pixel_length(text0, TEST_FONT_PATH, 16)],
        ]
        self.assert_actual_vs_expect(actual_vs_expect)

        self.assertEqual(o.join(o.split(text0, mode="character")), expected_join_text0)
        self.assertEqual(o.join(o.split(o.join(o.split(text0, mode="character")), mode="character")), expected_join_text0)
        self.assertEqual(o.join(o.split(text0, mode="character", attach_punctuation=False)), expected_join_text0)
        self.assertEqual(o.join(o.split(text0, mode="character")), expected_join_text0)
        self.assertEqual(o.join(o.split(text0, mode="character", attach_punctuation=False)), expected_join_text0)

        mixed_text = "이건 I'm 예시고 deeplearning.ai와 https://www.com 을 쓴다."
        self._assert_preserved_fragments(
            mixed_text,
            ["I'm", "deeplearning.ai", "https://www.com"],
            modes=("word", "character"),
        )

        # split()
        self.assertEqual(o.split("안녕하세요"), ["안녕", "하", "세요"])

        # length()
        self.assertEqual(o.length("안녕하세요"), 5)

        # join()
        self.assertEqual(o.join(["안녕", "하", "세요"]), "안녕하세요")

        # length() with cjk_width
        self.assertEqual(o.length("안녕하세요", cjk_width=2), 5)
        self.assertEqual(o.length("hello", cjk_width=1), 5)
        self.assertEqual(o.length("hello", cjk_width=2), 3)
        self.assertEqual(o.length("안녕하세요hello", cjk_width=1), 10)
        self.assertEqual(o.length("안녕하세요 hello", cjk_width=1), 10)
        self.assertEqual(o.length("안녕하세요hello", cjk_width=2), 8)
        self.assertEqual(o.length("안녕하세요 hello", cjk_width=2), 8)
        self.assertEqual(o.length("AI엔진OK", cjk_width=1), 6)
        self.assertEqual(o.length("AI 엔진OK", cjk_width=1), 6)
        self.assertEqual(o.length("AI엔진OK", cjk_width=2), 4)
        self.assertEqual(o.length("AI 엔진 OK", cjk_width=2), 4)

        # strip()
        self.assertEqual(o.strip("  안녕  "), "안녕")
        self.assertEqual(o.strip(""), "")
        self.assertEqual(o.strip("안녕하세요！", "！"), "안녕하세요")
        self.assertEqual(o.strip("...안녕...", "."), "안녕")
        self.assertEqual(o.strip("안녕", "!"), "안녕")
        self.assertEqual(o.strip("", "!"), "")

        # lstrip()
        self.assertEqual(o.lstrip("  안녕  "), "안녕  ")
        self.assertEqual(o.lstrip(""), "")
        self.assertEqual(o.lstrip("...안녕...", "."), "안녕...")
        self.assertEqual(o.lstrip("안녕하세요!", "!"), "안녕하세요!")

        # rstrip()
        self.assertEqual(o.rstrip("  안녕  "), "  안녕")
        self.assertEqual(o.rstrip(""), "")
        self.assertEqual(o.rstrip("반갑습니다。", "。"), "반갑습니다")
        self.assertEqual(o.rstrip("안녕하세요!", "!"), "안녕하세요")

        # strip_punc()
        self.assertEqual(o.strip_punc("。안녕하세요！"), "안녕하세요")
        self.assertEqual(o.strip_punc("《한국어》"), "한국어")
        self.assertEqual(o.strip_punc(""), "")
        self.assertEqual(o.strip_punc("！！。。"), "")

        # lstrip_punc()
        self.assertEqual(o.lstrip_punc("。안녕하세요！"), "안녕하세요！")
        self.assertEqual(o.lstrip_punc("《한국어》"), "한국어》")
        self.assertEqual(o.lstrip_punc(""), "")
        self.assertEqual(o.lstrip_punc("안녕하세요"), "안녕하세요")

        # rstrip_punc()
        self.assertEqual(o.rstrip_punc("。안녕하세요！"), "。안녕하세요")
        self.assertEqual(o.rstrip_punc("《한국어》"), "《한국어")
        self.assertEqual(o.rstrip_punc(""), "")
        self.assertEqual(o.rstrip_punc("안녕하세요"), "안녕하세요")

        # transfer_punc()
        self.assertEqual(o.transfer_punc("안녕하세요", "안녕하세요!"), "안녕하세요!")
        self.assertEqual(o.transfer_punc("테스트", "《테스트》"), "《테스트》")
        self.assertEqual(o.transfer_punc("안녕하세요", "안녕하세요"), "안녕하세요")
        self.assertEqual(o.transfer_punc("", ""), "")
        with self.assertRaises(ValueError):
            o.transfer_punc("안녕 하세요", "안녕")

        # edge()
        self._assert_cjk_edge("한")

        # mode()
        self._assert_cjk_mode("안녕하세요 세계")

        # normalize()
        self._assert_cjk_normalize("안녕하세요", "안녕하세요, 세계! 테스트.")
