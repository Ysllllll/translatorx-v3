from lang_ops import TextOps

from tests.lang_ops_tests._base import TextOpsTestCase


class FrenchTextTest(TextOpsTestCase):
    def setUp(self) -> None:
        self.ops = TextOps.for_language("fr")
        return super().setUp()

    def test_french(self) -> None:
        o = self.ops
        text0 = "Bonjour le monde !"
        expect_split_text = ["Bonjour", "le", "monde", "!"]
        expect_join_text0 = "Bonjour le monde !"
        self._assert_entype_text_case(text0, expect_split_text, expect_join_text0)

        text1 = "Bonjour à tous : aujourd'hui, notre système traite les sous-titres français, allemands et portugais !"
        expect_split_text = ["Bonjour", "à", "tous", ":", "aujourd'hui,", "notre", "système", "traite", "les", "sous-titres", "français,", "allemands", "et", "portugais", "!"]
        expect_join_text1 = "Bonjour à tous : aujourd'hui, notre système traite les sous-titres français, allemands et portugais !"
        self._assert_entype_text_case(text1, expect_split_text, expect_join_text1)

        # split()
        self.assertEqual(o.split("Bonjour le monde !"), ["Bonjour", "le", "monde", "!"])

        # length()
        self.assertEqual(o.length("Bonjour"), 7)

        # join()
        self.assertEqual(o.join(["Bonjour", "le", "monde", "!"]), "Bonjour le monde !")

        # strip()
        self.assertEqual(o.strip("  Bonjour  "), "Bonjour")
        self.assertEqual(o.strip(""), "")
        self.assertEqual(o.strip("Bonjour le monde!", "!"), "Bonjour le monde")
        self.assertEqual(o.strip("...Bonjour...", "."), "Bonjour")
        self.assertEqual(o.strip("Bonjour", "!"), "Bonjour")
        self.assertEqual(o.strip("", "!"), "")

        # lstrip()
        self.assertEqual(o.lstrip("  Bonjour  "), "Bonjour  ")
        self.assertEqual(o.lstrip(""), "")
        self.assertEqual(o.lstrip("...Bonjour...", "."), "Bonjour...")
        self.assertEqual(o.lstrip("Bonjour!", "!"), "Bonjour!")

        # rstrip()
        self.assertEqual(o.rstrip("  Bonjour  "), "  Bonjour")
        self.assertEqual(o.rstrip(""), "")
        self.assertEqual(o.rstrip("Bonjour !", " !"), "Bonjour")
        self.assertEqual(o.rstrip("Question ?", " ?"), "Question")
        self.assertEqual(o.rstrip("...Bonjour...", "."), "...Bonjour")
        self.assertEqual(o.rstrip("Bonjour!", "!"), "Bonjour")

        # edge()
        self._assert_entype_edge()

        # mode()
        self._assert_entype_mode("Bonjour le monde !")

        # normalize() (French-specific: space before high punctuation) --
        # High punctuation: add space before ! ? ; :
        self.assertEqual(o.normalize("Bonjour le monde!"), "Bonjour le monde !")
        self.assertEqual(o.normalize("Question?"), "Question ?")
        self.assertEqual(o.normalize("Remarque:"), "Remarque :")
        self.assertEqual(o.normalize("Attention;"), "Attention ;")
        # Extra spaces collapsed
        self.assertEqual(o.normalize("Bonjour le monde  !"), "Bonjour le monde !")
        self.assertEqual(o.normalize("Question   ?"), "Question ?")
        # Low punctuation: space removed
        self.assertEqual(o.normalize("Bonjour le monde."), "Bonjour le monde.")
        self.assertEqual(o.normalize("Bonjour le monde ."), "Bonjour le monde.")
        self.assertEqual(o.normalize("Bonjour, le monde"), "Bonjour, le monde")
        self.assertEqual(o.normalize("Bonjour , le monde"), "Bonjour, le monde")
        # Already correct stays unchanged
        self.assertEqual(o.normalize("Bonjour le monde !"), "Bonjour le monde !")
        self.assertEqual(o.normalize("Question ?"), "Question ?")
        self.assertEqual(o.normalize("Bonjour à tous :"), "Bonjour à tous :")
        # Time notation not affected
        self.assertEqual(o.normalize("12:30"), "12:30")
        # Combined
        self.assertEqual(
            o.normalize("Bonjour à tous : aujourd'hui , notre système traite les sous-titres!"),
            "Bonjour à tous : aujourd'hui, notre système traite les sous-titres !",
        )
        # Edge cases
        self.assertEqual(o.normalize(""), "")
        self.assertEqual(o.normalize("Bonjour"), "Bonjour")

        # strip_punc()
        self.assertEqual(o.strip_punc("(Bonjour, monde!)"), "Bonjour, monde")
        self.assertEqual(o.strip_punc('"Bonjour"'), "Bonjour")
        self.assertEqual(o.strip_punc(""), "")
        self.assertEqual(o.strip_punc("!!!"), "")

        # lstrip_punc()
        self.assertEqual(o.lstrip_punc("(Bonjour, monde!)"), "Bonjour, monde!)")
        self.assertEqual(o.lstrip_punc(""), "")
        self.assertEqual(o.lstrip_punc("Bonjour"), "Bonjour")

        # rstrip_punc()
        self.assertEqual(o.rstrip_punc("(Bonjour, monde!)"), "(Bonjour, monde")
        self.assertEqual(o.rstrip_punc(""), "")
        self.assertEqual(o.rstrip_punc("Bonjour"), "Bonjour")

        # restore_punc()
        self.assertEqual(o.restore_punc("Bonjour monde", "Bonjour, monde!"), "Bonjour, monde!")
        self.assertEqual(o.restore_punc("Bonjour", "(Bonjour)"), "(Bonjour)")
        self.assertEqual(o.restore_punc("Bonjour monde", "Bonjour monde"), "Bonjour monde")
        self.assertEqual(o.restore_punc("", ""), "")
        with self.assertRaises(ValueError):
            o.restore_punc("a b c", "x y")
