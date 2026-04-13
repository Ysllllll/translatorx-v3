import unittest

from lang_ops import normalize_language

from .._base import LangOpsTestCase


class NormalizeLanguageTest(LangOpsTestCase):
    def test_normalize_aliases(self) -> None:
        cases = {
            "zh": ["zh", "ZH", " chinese ", "CN", "中文", "汉语"],
            "en": ["en", "EN", " english ", "English", "英语"],
            "ru": ["ru", "RU", " russian ", "Русский", "俄语"],
            "es": ["es", "ES", " spanish ", "Español", "西班牙语"],
            "ja": ["ja", "JA", " japanese ", "日本語", "日语"],
            "ko": ["ko", "KO", " korean ", "한국어", "韩语"],
            "fr": ["fr", "FR", " french ", "Français", "法语"],
            "de": ["de", "DE", " german ", "Deutsch", "德语"],
            "pt": ["pt", "PT", " portuguese ", "Português", "葡萄牙语"],
            "vi": ["vi", "VI", " vietnamese ", "Tiếng Việt", "越南语"],
        }

        actual_vs_expect: list[list] = []
        for language_code, values in cases.items():
            for value in values:
                actual_vs_expect.append([normalize_language(value), language_code])

        self.assert_actual_vs_expect(actual_vs_expect)

    def test_normalize_invalid_language(self) -> None:
        invalid_inputs = ["it", "Italian", "ไทย", "", "   "]

        for value in invalid_inputs:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    normalize_language(value)
