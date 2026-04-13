import unittest


class MechanismTest(unittest.TestCase):
    def test_unsupported_language(self) -> None:
        from lang_ops import LangOps
        for lang in ["it", "ar", "th", "xyz"]:
            with self.subTest(lang=lang):
                with self.assertRaises(ValueError):
                    LangOps.for_language(lang)
