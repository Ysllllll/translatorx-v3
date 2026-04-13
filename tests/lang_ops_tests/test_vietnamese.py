from lang_ops import LangOps

from tests.lang_ops_tests._base import LangOpsTestCase


class VietnameseTextTest(LangOpsTestCase):
    def setUp(self) -> None:
        self.ops = LangOps.for_language("vi")
        return super().setUp()

    # -- split, length and join --
    def test_vietnamese(self) -> None:
        o = self.ops
        text0 = "Xin chào thế giới!"
        expect_split_text = ["Xin", "chào", "thế", "giới!"]
        expect_join_text0 = "Xin chào thế giới!"
        self._assert_entype_text_case(text0, expect_split_text, expect_join_text0)

        text1 = "Hệ thống phụ đề đa ngôn ngữ của chúng tôi xử lý tiếng Việt, tiếng Anh và tiếng Nhật trong một lần chạy."
        expect_split_text = ["Hệ", "thống", "phụ", "đề", "đa", "ngôn", "ngữ", "của", "chúng", "tôi", "xử", "lý", "tiếng", "Việt,", "tiếng", "Anh", "và", "tiếng", "Nhật", "trong", "một", "lần", "chạy."]
        expect_join_text1 = "Hệ thống phụ đề đa ngôn ngữ của chúng tôi xử lý tiếng Việt, tiếng Anh và tiếng Nhật trong một lần chạy."
        self._assert_entype_text_case(text1, expect_split_text, expect_join_text1)

        mixed_text = "Giữ I'm deeplearning.ai và https://www.com nguyên vẹn."
        self._assert_preserved_fragments(
            mixed_text,
            ["I'm", "deeplearning.ai", "https://www.com"],
        )

        # split()
        self.assertEqual(o.split("Xin chào thế giới!"), ["Xin", "chào", "thế", "giới!"])

        # length()
        self.assertEqual(o.length("Xin chào"), 8)

        # join()
        self.assertEqual(o.join(["Xin", "chào", "thế", "giới!"]), "Xin chào thế giới!")

        # strip()
        self.assertEqual(o.strip("  Xin chào  "), "Xin chào")
        self.assertEqual(o.strip(""), "")
        self.assertEqual(o.strip("Xin chào thế giới!", "!"), "Xin chào thế giới")
        self.assertEqual(o.strip("...Xin chào...", "."), "Xin chào")
        self.assertEqual(o.strip("Xin chào", "!"), "Xin chào")
        self.assertEqual(o.strip("", "!"), "")

        # lstrip()
        self.assertEqual(o.lstrip("  Xin chào  "), "Xin chào  ")
        self.assertEqual(o.lstrip(""), "")
        self.assertEqual(o.lstrip("...Xin chào...", "."), "Xin chào...")
        self.assertEqual(o.lstrip("Xin chào!", "!"), "Xin chào!")

        # rstrip()
        self.assertEqual(o.rstrip("  Xin chào  "), "  Xin chào")
        self.assertEqual(o.rstrip(""), "")
        self.assertEqual(o.rstrip("...Xin chào...", "."), "...Xin chào")
        self.assertEqual(o.rstrip("Xin chào thế giới!", "!"), "Xin chào thế giới")

        # edge()
        self._assert_entype_edge()

        # mode()
        self._assert_entype_mode("Xin chào thế giới!")

        # normalize()
        self._assert_entype_normalize()
        self.assertEqual(
            o.normalize("Xin chào , thế giới !"),
            "Xin chào, thế giới!",
        )

        # strip_punc()
        self.assertEqual(o.strip_punc("(Xin chào!)"), "Xin chào")
        self.assertEqual(o.strip_punc('"Xin chào"'), "Xin chào")
        self.assertEqual(o.strip_punc(""), "")
        self.assertEqual(o.strip_punc("!!!"), "")

        # lstrip_punc()
        self.assertEqual(o.lstrip_punc("(Xin chào!)"), "Xin chào!)")
        self.assertEqual(o.lstrip_punc(""), "")
        self.assertEqual(o.lstrip_punc("Xin chào"), "Xin chào")

        # rstrip_punc()
        self.assertEqual(o.rstrip_punc("(Xin chào!)"), "(Xin chào")
        self.assertEqual(o.rstrip_punc(""), "")
        self.assertEqual(o.rstrip_punc("Xin chào"), "Xin chào")

        # restore_punc()
        self.assertEqual(o.restore_punc("Xin chào", "Xin chào!"), "Xin chào!")
        self.assertEqual(o.restore_punc("Xin chào", "(Xin chào)"), "(Xin chào)")
        self.assertEqual(o.restore_punc("Xin chào", "Xin chào"), "Xin chào")
        self.assertEqual(o.restore_punc("", ""), "")
        with self.assertRaises(ValueError):
            o.restore_punc("a b c", "x y")
