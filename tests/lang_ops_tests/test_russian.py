from lang_ops import TextOps

from tests.lang_ops_tests._base import TextOpsTestCase


class RussianTextTest(TextOpsTestCase):
    def setUp(self) -> None:
        self.ops = TextOps.for_language("ru")
        return super().setUp()

    def test_russian(self) -> None:
        o = self.ops
        text0 = "Привет, мир!"
        expect_split_text = ["Привет,", "мир!"]
        expect_join_text0 = "Привет, мир!"
        self._assert_entype_text_case(text0, expect_split_text, expect_join_text0)

        text1 = "Сегодня система синхронизирует русские, английские и японские субтитры для большого проекта."
        expect_split_text = ["Сегодня", "система", "синхронизирует", "русские,", "английские", "и", "японские", "субтитры", "для", "большого", "проекта."]
        expect_join_text1 = "Сегодня система синхронизирует русские, английские и японские субтитры для большого проекта."
        self._assert_entype_text_case(text1, expect_split_text, expect_join_text1)

        # split()
        self.assertEqual(o.split("Привет, мир!"), ["Привет,", "мир!"])

        # length()
        self.assertEqual(o.length("Привет"), 6)

        # join()
        self.assertEqual(o.join(["Привет,", "мир!"]), "Привет, мир!")

        # strip()
        self.assertEqual(o.strip("  Привет  "), "Привет")
        self.assertEqual(o.strip(""), "")
        self.assertEqual(o.strip("Привет, мир!", "!"), "Привет, мир")
        self.assertEqual(o.strip("...Привет...", "."), "Привет")
        self.assertEqual(o.strip("Привет", "!"), "Привет")
        self.assertEqual(o.strip("", "!"), "")

        # lstrip()
        self.assertEqual(o.lstrip("  Привет  "), "Привет  ")
        self.assertEqual(o.lstrip(""), "")
        self.assertEqual(o.lstrip("...Привет...", "."), "Привет...")
        self.assertEqual(o.lstrip("Привет!", "!"), "Привет!")

        # rstrip()
        self.assertEqual(o.rstrip("  Привет  "), "  Привет")
        self.assertEqual(o.rstrip(""), "")
        self.assertEqual(o.rstrip("...Привет...", "."), "...Привет")
        self.assertEqual(o.rstrip("Привет!", "!"), "Привет")

        # edge()
        self._assert_entype_edge("А")

        # mode()
        self._assert_entype_mode("Привет, мир!")

        # normalize()
        self._assert_entype_normalize()
        self.assertEqual(
            o.normalize("Привет , мир !"),
            "Привет, мир!",
        )

        # strip_punc()
        self.assertEqual(o.strip_punc("(Привет, мир!)"), "Привет, мир")
        self.assertEqual(o.strip_punc('"Привет"'), "Привет")
        self.assertEqual(o.strip_punc(""), "")
        self.assertEqual(o.strip_punc("!!!"), "")

        # lstrip_punc()
        self.assertEqual(o.lstrip_punc("(Привет, мир!)"), "Привет, мир!)")
        self.assertEqual(o.lstrip_punc(""), "")
        self.assertEqual(o.lstrip_punc("Привет"), "Привет")

        # rstrip_punc()
        self.assertEqual(o.rstrip_punc("(Привет, мир!)"), "(Привет, мир")
        self.assertEqual(o.rstrip_punc(""), "")
        self.assertEqual(o.rstrip_punc("Привет"), "Привет")

        # restore_punc()
        self.assertEqual(o.restore_punc("Привет мир", "Привет, мир!"), "Привет, мир!")
        self.assertEqual(o.restore_punc("Привет", "(Привет)"), "(Привет)")
        self.assertEqual(o.restore_punc("Привет мир", "Привет мир"), "Привет мир")
        self.assertEqual(o.restore_punc("", ""), "")
        with self.assertRaises(ValueError):
            o.restore_punc("a b c", "x y")
