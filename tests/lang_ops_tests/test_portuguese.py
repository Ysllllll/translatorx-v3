from lang_ops import LangOps

from tests.lang_ops_tests._base import LangOpsTestCase


class PortugueseTextTest(LangOpsTestCase):
    def setUp(self) -> None:
        self.ops = LangOps.for_language("pt")
        return super().setUp()

    def test_portuguese(self) -> None:
        o = self.ops
        text0 = "Olá, mundo!"
        expect_split_text = ["Olá,", "mundo!"]
        expect_join_text0 = "Olá, mundo!"
        self._assert_entype_text_case(text0, expect_split_text, expect_join_text0)

        text1 = "Hoje o mecanismo organiza legendas em português, espanhol e inglês para uma série inteira."
        expect_split_text = ["Hoje", "o", "mecanismo", "organiza", "legendas", "em", "português,", "espanhol", "e", "inglês", "para", "uma", "série", "inteira."]
        expect_join_text1 = "Hoje o mecanismo organiza legendas em português, espanhol e inglês para uma série inteira."
        self._assert_entype_text_case(text1, expect_split_text, expect_join_text1)

        mixed_text = "Mantenha I'm deeplearning.ai e https://www.com juntos."
        self._assert_preserved_fragments(
            mixed_text,
            ["I'm", "deeplearning.ai", "https://www.com"],
        )

        # split()
        self.assertEqual(o.split("Olá, mundo!"), ["Olá,", "mundo!"])

        # length()
        self.assertEqual(o.length("Olá"), 3)

        # join()
        self.assertEqual(o.join(["Olá,", "mundo!"]), "Olá, mundo!")

        # strip()
        self.assertEqual(o.strip("  Olá  "), "Olá")
        self.assertEqual(o.strip(""), "")
        self.assertEqual(o.strip("Olá, mundo!", "!"), "Olá, mundo")
        self.assertEqual(o.strip("...Olá...", "."), "Olá")
        self.assertEqual(o.strip("Olá", "!"), "Olá")
        self.assertEqual(o.strip("", "!"), "")

        # lstrip()
        self.assertEqual(o.lstrip("  Olá  "), "Olá  ")
        self.assertEqual(o.lstrip(""), "")
        self.assertEqual(o.lstrip("...Olá...", "."), "Olá...")
        self.assertEqual(o.lstrip("Olá!", "!"), "Olá!")

        # rstrip()
        self.assertEqual(o.rstrip("  Olá  "), "  Olá")
        self.assertEqual(o.rstrip(""), "")
        self.assertEqual(o.rstrip("...Olá...", "."), "...Olá")
        self.assertEqual(o.rstrip("Olá!", "!"), "Olá")

        # edge()
        self._assert_entype_edge()

        # mode()
        self._assert_entype_mode("Olá, mundo!")

        # normalize()
        self._assert_entype_normalize()
        self.assertEqual(
            o.normalize("Olá , mundo !"),
            "Olá, mundo!",
        )

        # strip_punc()
        self.assertEqual(o.strip_punc("(Olá, mundo!)"), "Olá, mundo")
        self.assertEqual(o.strip_punc('"Olá"'), "Olá")
        self.assertEqual(o.strip_punc(""), "")
        self.assertEqual(o.strip_punc("!!!"), "")

        # lstrip_punc()
        self.assertEqual(o.lstrip_punc("(Olá, mundo!)"), "Olá, mundo!)")
        self.assertEqual(o.lstrip_punc(""), "")
        self.assertEqual(o.lstrip_punc("Olá"), "Olá")

                # rstrip_punc()
        self.assertEqual(o.rstrip_punc("(Olá, mundo!)"), "(Olá, mundo")
        self.assertEqual(o.rstrip_punc(""), "")
        self.assertEqual(o.rstrip_punc("Olá"), "Olá")

        # restore_punc()
        self.assertEqual(o.restore_punc("Olá mundo", "Olá, mundo!"), "Olá, mundo!")
        self.assertEqual(o.restore_punc("Olá", "(Olá)"), "(Olá)")
        self.assertEqual(o.restore_punc("Olá mundo", "Olá mundo"), "Olá mundo")
        self.assertEqual(o.restore_punc("", ""), "")
        with self.assertRaises(ValueError):
            o.restore_punc("a b c", "x y")
