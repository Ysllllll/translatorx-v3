from lang_ops import LangOps

from tests.lang_ops_tests._base import LangOpsTestCase


class EnglishTextTest(LangOpsTestCase):
    def setUp(self) -> None:
        self.ops = LangOps.for_language("en")
        return super().setUp()

    def test_english(self) -> None:
        o = self.ops
        text0 = "Hello, world!"
        expect_split_text = ["Hello,", "world!"]
        expect_join_text0 = "Hello, world!"
        self._assert_entype_text_case(text0, expect_split_text, expect_join_text0)

        text1 = "It's 2026-03-26."
        expect_split_text = ["It's", "2026-03-26."]
        expect_join_text1 = "It's 2026-03-26."
        self._assert_entype_text_case(text1, expect_split_text, expect_join_text1)

        text2 = "In 2026, the translation engine processed English, Russian, and Japanese subtitles in one pass."
        expect_split_text = ["In", "2026,", "the", "translation", "engine", "processed", "English,", "Russian,", "and", "Japanese", "subtitles", "in", "one", "pass."]
        expect_join_text2 = "In 2026, the translation engine processed English, Russian, and Japanese subtitles in one pass."
        self._assert_entype_text_case(text2, expect_split_text, expect_join_text2)

        text3 = "He said, \"It's AI.\" (Really?) [Yes.] {'OK.'} \"orphan 'solo tail) [loose {brace ,"
        expect_split_text = ["He", "said,", "\"It's", "AI.\"", "(Really?)", "[Yes.]", "{'OK.'}", "\"orphan", "'solo", "tail)", "[loose", "{brace,"]
        expect_split_text_raw = ["He", "said,", "\"It's", "AI.\"", "(Really?)", "[Yes.]", "{'OK.'}", "\"orphan", "'solo", "tail)", "[loose", "{brace", ","]
        expect_join_text3 = "He said, \"It's AI.\" (Really?) [Yes.] {'OK.'} \"orphan 'solo tail) [loose {brace,"
        self._assert_entype_text_case(
            text3,
            expect_split_text,
            expect_join_text3,
            expected_split_without_punctuation=expect_split_text_raw,
        )

        mixed_text = "Keep I'm deeplearning.ai and https://www.com intact."
        self._assert_preserved_fragments(
            mixed_text,
            ["I'm", "deeplearning.ai", "https://www.com"],
        )

        # split()
        self.assertEqual(o.split("Hello, world!"), ["Hello,", "world!"])
        self.assertEqual(o.split("It's 2026."), ["It's", "2026."])
        self.assertEqual(o.split("Hello, world !"), ["Hello,", "world!"])
        self.assertEqual(o.split("Hello, world      !"), ["Hello,", "world!"])
        self.assertEqual(o.split("Hello    , world      !"), ["Hello,", "world!"])
        self.assertEqual(o.split("Hello    ,            world      !"), ["Hello,", "world!"])
        self.assertEqual(o.split("Hello, world !", attach_punctuation=False), ["Hello,", "world", "!"])
        self.assertEqual(o.split("Hello    ,            world      !", attach_punctuation=False), ["Hello", ",", "world", "!"])

        # length()
        self.assertEqual(o.length("Hello"), 5)
        self.assertEqual(o.length("Hello, world!"), 13)

        # join()
        self.assertEqual(o.join(["Hello,", "world!"]), "Hello, world!")
        self.assertEqual(o.join(["It's", "AI."]), "It's AI.")
        self.assert_actual_vs_expect([
            [o.join(["\"", "Hello", ",", "world", "!", "\""]), "\"Hello, world!\""],
            [o.join(["'", "Hello", "'"]), "'Hello'"],
            [o.join(["It", "'", "s", "\"", "AI", "\""]), "It's \"AI\""],
        ])

        # strip()
        self.assertEqual(o.strip("  Hello  "), "Hello")
        self.assertEqual(o.strip(""), "")
        self.assertEqual(o.strip("Hello, world!", "!"), "Hello, world")
        self.assertEqual(o.strip("...Hello...", "."), "Hello")
        self.assertEqual(o.strip('!Hello!', "!"), "Hello")
        self.assertEqual(o.strip('"Hello"', '"'), "Hello")
        self.assertEqual(o.strip("'Hello'", "'"), "Hello")
        self.assertEqual(o.strip("Hello", "!"), "Hello")
        self.assertEqual(o.strip("", "!"), "")
        self.assertEqual(o.strip("!!!", "!"), "")

        # lstrip()
        self.assertEqual(o.lstrip("  Hello  "), "Hello  ")
        self.assertEqual(o.lstrip(""), "")
        self.assertEqual(o.lstrip("...Hello...", "."), "Hello...")
        self.assertEqual(o.lstrip('Hello!', "!"), "Hello!")
        self.assertEqual(o.lstrip("", "!"), "")

        # rstrip()
        self.assertEqual(o.rstrip("  Hello  "), "  Hello")
        self.assertEqual(o.rstrip(""), "")
        self.assertEqual(o.rstrip("...Hello...", "."), "...Hello")
        self.assertEqual(o.rstrip("Hello!", "!"), "Hello")
        self.assertEqual(o.rstrip("", "!"), "")

        # strip_punc()
        self.assertEqual(o.strip_punc("(Hello World!)"), "Hello World")
        self.assertEqual(o.strip_punc('"Hello"'), "Hello")
        self.assertEqual(o.strip_punc("'Hello'"), "Hello")
        self.assertEqual(o.strip_punc(""), "")
        self.assertEqual(o.strip_punc("!!!"), "")

        # lstrip_punc()
        self.assertEqual(o.lstrip_punc("(Hello World!)"), "Hello World!)")
        self.assertEqual(o.lstrip_punc('"Hello"'), "Hello\"")
        self.assertEqual(o.lstrip_punc(""), "")
        self.assertEqual(o.lstrip_punc("Hello!"), "Hello!")

        # rstrip_punc()
        self.assertEqual(o.rstrip_punc("(Hello World!)"), "(Hello World")
        self.assertEqual(o.rstrip_punc('"Hello"'), "\"Hello")
        self.assertEqual(o.rstrip_punc(""), "")
        self.assertEqual(o.rstrip_punc("!Hello"), "!Hello")

        # restore_punc()
        self.assertEqual(o.restore_punc("Hello world", "Hello, world!"), "Hello, world!")
        self.assertEqual(o.restore_punc("Hello", "Hello!"), "Hello!")
        self.assertEqual(o.restore_punc("Hello", "(Hello)"), "(Hello)")
        self.assertEqual(o.restore_punc("Hello world", '"Hello" world!'), '"Hello" world!')
        self.assertEqual(o.restore_punc("Hello world", "Hello world"), "Hello world")
        self.assertEqual(o.restore_punc("", ""), "")
        with self.assertRaises(ValueError):
            o.restore_punc("a b c", "x y")

        # edge()
        self._assert_entype_edge()

        # mode()
        self._assert_entype_mode("Hello, world!")

        # normalize()
        self._assert_entype_normalize()
