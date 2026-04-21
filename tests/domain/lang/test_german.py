from domain.lang import LangOps

from tests.domain.lang._base import LangOpsTestCase


class GermanTextTest(LangOpsTestCase):
    def setUp(self) -> None:
        self.ops = LangOps.for_language("de")
        return super().setUp()

    def test_german(self) -> None:
        o = self.ops
        text0 = "Hallo, schöne Welt!"
        expect_split_text = ["Hallo,", "schöne", "Welt!"]
        expect_join_text0 = "Hallo, schöne Welt!"
        self._assert_entype_text_case(text0, expect_split_text, expect_join_text0)

        text1 = "Heute verarbeitet unser System deutsche, französische und portugiesische Untertitel in einem Durchgang."
        expect_split_text = [
            "Heute",
            "verarbeitet",
            "unser",
            "System",
            "deutsche,",
            "französische",
            "und",
            "portugiesische",
            "Untertitel",
            "in",
            "einem",
            "Durchgang.",
        ]
        expect_join_text1 = "Heute verarbeitet unser System deutsche, französische und portugiesische Untertitel in einem Durchgang."
        self._assert_entype_text_case(text1, expect_split_text, expect_join_text1)

        mixed_text = "Lasst I'm deeplearning.ai und https://www.com ganz."
        self._assert_preserved_fragments(
            mixed_text,
            ["I'm", "deeplearning.ai", "https://www.com"],
        )

        # split()
        self.assertEqual(o.split("Hallo, schöne Welt!"), ["Hallo,", "schöne", "Welt!"])

        # length()
        self.assertEqual(o.length("Hallo"), 5)

        # join()
        self.assertEqual(o.join(["Hallo,", "schöne", "Welt!"]), "Hallo, schöne Welt!")

        # strip()
        self.assertEqual(o.strip("  Hallo  "), "Hallo")
        self.assertEqual(o.strip(""), "")
        self.assertEqual(o.strip("Hallo, schöne Welt!", "!"), "Hallo, schöne Welt")
        self.assertEqual(o.strip("...Hallo...", "."), "Hallo")
        self.assertEqual(o.strip("Hallo", "!"), "Hallo")
        self.assertEqual(o.strip("", "!"), "")

        # lstrip()
        self.assertEqual(o.lstrip("  Hallo  "), "Hallo  ")
        self.assertEqual(o.lstrip(""), "")
        self.assertEqual(o.lstrip("...Hallo...", "."), "Hallo...")
        self.assertEqual(o.lstrip("Hallo!", "!"), "Hallo!")

        # rstrip()
        self.assertEqual(o.rstrip("  Hallo  "), "  Hallo")
        self.assertEqual(o.rstrip(""), "")
        self.assertEqual(o.rstrip("...Hallo...", "."), "...Hallo")
        self.assertEqual(o.rstrip("Hallo!", "!"), "Hallo")

        # edge()
        self._assert_entype_edge()

        # mode()
        self._assert_entype_mode("Hallo, schöne Welt!")

        # normalize()
        self._assert_entype_normalize()
        self.assertEqual(
            o.normalize("Hallo , schöne Welt !"),
            "Hallo, schöne Welt!",
        )

        # strip_punc()
        self.assertEqual(o.strip_punc("(Hallo, Welt!)"), "Hallo, Welt")
        self.assertEqual(o.strip_punc('"Hallo"'), "Hallo")
        self.assertEqual(o.strip_punc(""), "")
        self.assertEqual(o.strip_punc("!!!"), "")

        # lstrip_punc()
        self.assertEqual(o.lstrip_punc("(Hallo, Welt!)"), "Hallo, Welt!)")
        self.assertEqual(o.lstrip_punc(""), "")
        self.assertEqual(o.lstrip_punc("Hallo"), "Hallo")

        # rstrip_punc()
        self.assertEqual(o.rstrip_punc("(Hallo, Welt!)"), "(Hallo, Welt")
        self.assertEqual(o.rstrip_punc(""), "")
        self.assertEqual(o.rstrip_punc("Hallo"), "Hallo")

        # transfer_punc()
        self.assertEqual(o.transfer_punc("Hallo Welt", "Hallo, Welt!"), "Hallo, Welt!")
        self.assertEqual(o.transfer_punc("Hallo", "(Hallo)"), "(Hallo)")
        self.assertEqual(o.transfer_punc("Hallo Welt", "Hallo Welt"), "Hallo Welt")
        self.assertEqual(o.transfer_punc("", ""), "")
        with self.assertRaises(ValueError):
            o.transfer_punc("a b c", "x y")
