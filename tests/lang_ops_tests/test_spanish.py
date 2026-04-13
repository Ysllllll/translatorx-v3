from lang_ops import LangOps

from tests.lang_ops_tests._base import LangOpsTestCase


class SpanishTextTest(LangOpsTestCase):
    def setUp(self) -> None:
        self.ops = LangOps.for_language("es")
        return super().setUp()

    # -- split, length and join --
    def test_spanish(self) -> None:
        o = self.ops
        text0 = "¡Hola mundo!"
        expect_split_text = ["¡Hola", "mundo!"]
        expect_join_text0 = "¡Hola mundo!"
        self._assert_entype_text_case(text0, expect_split_text, expect_join_text0)

        text1 = "¡Hoy nuestro sistema organiza subtítulos en español, inglés y japonés para una película completa!"
        expect_split_text = ["¡Hoy", "nuestro", "sistema", "organiza", "subtítulos", "en", "español,", "inglés", "y", "japonés", "para", "una", "película", "completa!"]
        expect_join_text1 = "¡Hoy nuestro sistema organiza subtítulos en español, inglés y japonés para una película completa!"
        self._assert_entype_text_case(text1, expect_split_text, expect_join_text1)

        mixed_text = "Mantén I'm deeplearning.ai y https://www.com juntos."
        self._assert_preserved_fragments(
            mixed_text,
            ["I'm", "deeplearning.ai", "https://www.com"],
        )

        # split()
        self.assertEqual(o.split("¡Hola mundo!"), ["¡Hola", "mundo!"])

        # length()
        self.assertEqual(o.length("Hola"), 4)

        # join()
        self.assertEqual(o.join(["¡Hola", "mundo!"]), "¡Hola mundo!")

        # strip()
        self.assertEqual(o.strip("  Hola  "), "Hola")
        self.assertEqual(o.strip(""), "")
        self.assertEqual(o.strip("¡Hola mundo!", "¡!"), "Hola mundo")
        self.assertEqual(o.strip("...Hola...", "."), "Hola")
        self.assertEqual(o.strip("Hola", "!"), "Hola")
        self.assertEqual(o.strip("", "!"), "")

        # lstrip()
        self.assertEqual(o.lstrip("  Hola  "), "Hola  ")
        self.assertEqual(o.lstrip(""), "")
        self.assertEqual(o.lstrip("¿Qué?", "¿"), "Qué?")
        self.assertEqual(o.lstrip("¡Hola!", "¡"), "Hola!")

        # rstrip()
        self.assertEqual(o.rstrip("  Hola  "), "  Hola")
        self.assertEqual(o.rstrip(""), "")
        self.assertEqual(o.rstrip("¡Hola!", "¡!"), "¡Hola")
        self.assertEqual(o.rstrip("¿Qué?", "?"), "¿Qué")

        # edge()
        self._assert_entype_edge()

        # mode()
        self._assert_entype_mode("¡Hola mundo!")

        # normalize()
        self._assert_entype_normalize()
        self.assertEqual(
            o.normalize("¡Hola mundo !"),
            "¡Hola mundo!",
        )

        # strip_punc()
        self.assertEqual(o.strip_punc("¡Hola mundo!"), "Hola mundo")
        self.assertEqual(o.strip_punc("(¿Qué?)"), "Qué")
        self.assertEqual(o.strip_punc(""), "")
        self.assertEqual(o.strip_punc("!!!"), "")

        # lstrip_punc()
        self.assertEqual(o.lstrip_punc("¡Hola mundo!"), "Hola mundo!")
        self.assertEqual(o.lstrip_punc(""), "")
        self.assertEqual(o.lstrip_punc("Hola"), "Hola")

        # rstrip_punc()
        self.assertEqual(o.rstrip_punc("¡Hola mundo!"), "¡Hola mundo")
        self.assertEqual(o.rstrip_punc(""), "")
        self.assertEqual(o.rstrip_punc("Hola"), "Hola")

        # restore_punc()
        self.assertEqual(o.restore_punc("Hola mundo", "¡Hola mundo!"), "¡Hola mundo!")
        self.assertEqual(o.restore_punc("Qué", "¿Qué?"), "¿Qué?")
        self.assertEqual(o.restore_punc("Hola mundo", "Hola mundo"), "Hola mundo")
        self.assertEqual(o.restore_punc("", ""), "")
        with self.assertRaises(ValueError):
            o.restore_punc("a b c", "x y")
