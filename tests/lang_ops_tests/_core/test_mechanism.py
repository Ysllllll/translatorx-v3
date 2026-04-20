import unittest


class MechanismTest(unittest.TestCase):
    def test_span_not_exported(self) -> None:
        import lang_ops

        self.assertFalse(hasattr(lang_ops, "Span"))

    def test_unsupported_language(self) -> None:
        from lang_ops import LangOps

        for lang in ["it", "ar", "th", "xyz"]:
            with self.subTest(lang=lang):
                with self.assertRaises(ValueError):
                    LangOps.for_language(lang)
