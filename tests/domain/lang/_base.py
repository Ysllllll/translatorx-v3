import unittest

from .conftest import TEST_FONT_PATH, expected_pixel_length


class LangOpsTestCase(unittest.TestCase):
    """Base class for per-language LangOps tests.

    Organization:
    - Roundtrip helpers: split/join/length/plength roundtrip testing
    - Composite helpers: language-group-level test bundles (EnType / CJK)
    - Atomic helpers: single-concept assertions used by composites
    """

    # ── Roundtrip helpers ──────────────────────────────────────────────────

    def assert_actual_vs_expect(self, actual_vs_expect: list[list]) -> None:
        for actual, expect in actual_vs_expect:
            self.assertEqual(actual, expect)

    def _assert_text_join_case(self, text: str, expected_join_text: str | None = None) -> None:
        """Verify split→join roundtrip produces expected text (all mode/attach combos)."""
        resolved = text if expected_join_text is None else expected_join_text
        actual_vs_expect = [
            [self.ops.join(self.ops.split(text)), resolved],
            [self.ops.join(self.ops.split(self.ops.join(self.ops.split(text)))), resolved],
            [self.ops.join(self.ops.split(text, attach_punctuation=False)), resolved],
            [self.ops.join(self.ops.split(self.ops.join(self.ops.split(text, attach_punctuation=False)))), resolved],
            [self.ops.join(self.ops.split(self.ops.join(self.ops.split(text, attach_punctuation=False)), attach_punctuation=False)), resolved],
            [self.ops.join(self.ops.split(self.ops.join(self.ops.split(text, attach_punctuation=False)), attach_punctuation=False, mode="word")), resolved],
            [self.ops.join(self.ops.split(text, mode="word")), resolved],
            [self.ops.join(self.ops.split(self.ops.join(self.ops.split(text, mode="word")))), resolved],
            [self.ops.join(self.ops.split(self.ops.join(self.ops.split(text, mode="word")), mode="word")), resolved],
            [self.ops.join(self.ops.split(self.ops.join(self.ops.split(text, mode="word")), mode="word", attach_punctuation=False)), resolved],
            [self.ops.join(self.ops.split(text, mode="word", attach_punctuation=False)), resolved],
            [self.ops.join(self.ops.split(self.ops.join(self.ops.split(text, mode="word", attach_punctuation=False)))), resolved],
            [self.ops.join(self.ops.split(self.ops.join(self.ops.split(text, mode="word", attach_punctuation=False)), mode="word")), resolved],
            [self.ops.join(self.ops.split(self.ops.join(self.ops.split(text, mode="word", attach_punctuation=False)), attach_punctuation=False)), resolved],
            [self.ops.join(self.ops.split(self.ops.join(self.ops.split(text, mode="word", attach_punctuation=False)), mode="word", attach_punctuation=False)), resolved],
        ]
        self.assert_actual_vs_expect(actual_vs_expect)

    def _assert_entype_text_case(self, text: str, expected_split: list[str], expected_join_text: str | None = None, expected_split_without_punctuation: list[str] | None = None) -> None:
        """EnType: verify split (all modes), length, plength, and join roundtrip."""
        expected_chars = [ch for ch in text if not ch.isspace()]
        expected_split_without_punctuation = expected_split if expected_split_without_punctuation is None else expected_split_without_punctuation
        actual_vs_expect = [
            [self.ops.split(text), expected_split],
            [self.ops.split(text, mode="word"), expected_split],
            [self.ops.split(text, mode="character"), expected_chars],
            [self.ops.split(text, attach_punctuation=False), expected_split_without_punctuation],
            [self.ops.split(text, mode="word", attach_punctuation=False), expected_split_without_punctuation],
            [self.ops.split(text, mode="character", attach_punctuation=False), expected_chars],
            [self.ops.length(text), len(text)],
            [self.ops.plength(text, TEST_FONT_PATH, 16), expected_pixel_length(text, TEST_FONT_PATH, 16)],
        ]
        self.assert_actual_vs_expect(actual_vs_expect)
        self._assert_text_join_case(text, expected_join_text)

    def _assert_preserved_fragments(self, text: str, fragments: list[str], *, modes: tuple[str, ...] = ("word",), attach_punctuation: bool = False) -> None:
        """Assert protected Latin fragments survive tokenization as single tokens."""
        for mode in modes:
            tokens = self.ops.split(text, mode=mode, attach_punctuation=attach_punctuation)
            for fragment in fragments:
                self.assertIn(fragment, tokens, msg=f"{fragment!r} should stay whole in {mode=} for {text!r}, got {tokens!r}")

    # ── Composite helpers: EnType (en, ru, es, fr, de, pt, vi) ────────────
    # Call from test_edge / test_mode / test_normalize in EnType test files.

    def _assert_entype_edge(self, single_char: str = "A") -> None:
        self._assert_edge_empty_string()
        self._assert_edge_single_character(single_char)
        self._assert_edge_join_boundary()
        self._assert_edge_pure_punctuation()

    def _assert_entype_mode(self, text: str) -> None:
        self._assert_mode_invalid(text)
        self._assert_mode_shorthand(text)

    def _assert_entype_normalize(self) -> None:
        self._assert_normalize_space_before_punct_removed()
        self._assert_normalize_bracket_spaces()
        self._assert_normalize_already_correct()
        self._assert_normalize_combined()

    # ── Composite helpers: CJK (zh, ja, ko) ───────────────────────────────
    # Call from test_edge / test_mode / test_normalize in CJK test files.

    def _assert_cjk_edge(self, single_char: str) -> None:
        self._assert_edge_empty_string()
        self._assert_edge_single_character(single_char)
        self._assert_edge_join_boundary()

    def _assert_cjk_mode(self, text: str) -> None:
        self._assert_mode_invalid(text)
        self._assert_mode_shorthand(text)

    def _assert_cjk_ascii_quotes(self) -> None:
        """CJK: verify ASCII-quoted Latin stays as one token (e.g. "AI" in CJK context)."""
        # Plain "AI" surrounded by CJK — stays as one token in mode=word
        self._assert_preserved_fragments('他说"AI"真好', ['"AI"'], modes=("word",))
        # 'single'-quoted Latin surrounded by CJK
        self._assert_preserved_fragments("他说'AI'真好", ["'AI'"], modes=("word",))
        # Contractions / dotted ids / URLs still protected
        self._assert_preserved_fragments("他说I'm是visit deeplearning.ai的https://example.com", ["I'm", "deeplearning.ai", "https://example.com"], modes=("word",))

    def _assert_cjk_normalize(self, text: str, text_with_punct: str) -> None:
        # CJK normalize applies pangu-style beautification at CJK↔Latin boundaries,
        # but leaves pure-CJK text and already-spaced text unchanged.

        # Full sentence roundtrip
        self.assertEqual(self.ops.normalize(text), text)

        # Punctuation-related: these are the core scenarios normalize is meant to address
        self.assertEqual(self.ops.normalize(text_with_punct), text_with_punct)

        # Pangu-style: ASCII-quoted Latin gets spaces between CJK and the quoted span
        self.assertEqual(self.ops.normalize('他说"AI"真好'), '他说 "AI" 真好')
        # Already-spaced stays unchanged
        self.assertEqual(self.ops.normalize('他说 "AI" 真好'), '他说 "AI" 真好')
        # Pure Latin inside quotes is untouched
        self.assertEqual(self.ops.normalize('"Deep Learning"'), '"Deep Learning"')
        # CJK↔Latin without whitespace gets a space
        self.assertEqual(self.ops.normalize("Hello世界"), "Hello 世界")

        # CJK punctuation stays in place
        self.assertEqual(self.ops.normalize("。"), "。")
        self.assertEqual(self.ops.normalize("！"), "！")
        self.assertEqual(self.ops.normalize("？"), "？")
        self.assertEqual(self.ops.normalize("，"), "，")
        self.assertEqual(self.ops.normalize("："), "：")
        self.assertEqual(self.ops.normalize("；"), "；")

        # CJK brackets stay in place
        self.assertEqual(self.ops.normalize("「」"), "「」")
        self.assertEqual(self.ops.normalize("（）"), "（）")
        self.assertEqual(self.ops.normalize("《》"), "《》")

        # Spaces around CJK punctuation are NOT removed
        self.assertEqual(self.ops.normalize("你好 。 世界"), "你好 。 世界")
        self.assertEqual(self.ops.normalize("你好 ！"), "你好 ！")

        # Mixed CJK + Latin punctuation
        self.assertEqual(self.ops.normalize("Hello, 世界!"), "Hello, 世界!")

        # Edge cases
        self.assertEqual(self.ops.normalize(""), "")
        self.assertEqual(self.ops.normalize("..."), "...")

        # Idempotency
        self.assertEqual(self.ops.normalize(self.ops.normalize(text_with_punct)), text_with_punct)

    # ── Atomic helpers: edge (shared by EnType and CJK) ───────────────────

    def _assert_edge_empty_string(self) -> None:
        self.assertEqual(self.ops.split(""), [])
        self.assertEqual(self.ops.join([]), "")
        self.assertEqual(self.ops.length(""), 0)

    def _assert_edge_single_character(self, ch: str) -> None:
        self.assertEqual(self.ops.split(ch), [ch])
        self.assertEqual(self.ops.join([ch]), ch)
        self.assertEqual(self.ops.length(ch), 1)

    def _assert_edge_join_boundary(self) -> None:
        self.assertEqual(self.ops.join([]), "")
        self.assertEqual(self.ops.join(["Hello"]), "Hello")
        self.assertEqual(self.ops.join([","]), ",")

    # ── Atomic helpers: mode (shared by EnType and CJK) ───────────────────

    def _assert_mode_invalid(self, text: str) -> None:
        with self.assertRaises(ValueError):
            self.ops.split(text, mode="sentence")

    def _assert_mode_shorthand(self, text: str) -> None:
        self.assertEqual(self.ops.split(text, mode="c"), self.ops.split(text, mode="character"))
        self.assertEqual(self.ops.split(text, mode="w"), self.ops.split(text, mode="word"))

    # ── Atomic helpers: EnType-only ────────────────────────────────────────
    # Called from _assert_entype_edge / _assert_entype_normalize composites.

    def _assert_edge_pure_punctuation(self) -> None:
        self.assertEqual(self.ops.split("..."), ["..."])
        self.assertEqual(self.ops.length("..."), 3)

    def _assert_normalize_space_before_punct_removed(self) -> None:
        self.assertEqual(self.ops.normalize("hello ."), "hello.")
        self.assertEqual(self.ops.normalize("hello ,"), "hello,")
        self.assertEqual(self.ops.normalize("hello !"), "hello!")
        self.assertEqual(self.ops.normalize("hello ?"), "hello?")
        self.assertEqual(self.ops.normalize("hello ;"), "hello;")
        self.assertEqual(self.ops.normalize("hello :"), "hello:")

    def _assert_normalize_bracket_spaces(self) -> None:
        self.assertEqual(self.ops.normalize("(hello )"), "(hello)")
        self.assertEqual(self.ops.normalize("( hello )"), "(hello)")
        self.assertEqual(self.ops.normalize("[hello ]"), "[hello]")
        self.assertEqual(self.ops.normalize("[ hello ]"), "[hello]")
        self.assertEqual(self.ops.normalize("{hello }"), "{hello}")
        self.assertEqual(self.ops.normalize("{ hello }"), "{hello}")

    def _assert_normalize_already_correct(self) -> None:
        self.assertEqual(self.ops.normalize("Hello, world!"), "Hello, world!")
        self.assertEqual(self.ops.normalize("(OK)"), "(OK)")

    def _assert_normalize_combined(self) -> None:
        self.assertEqual(self.ops.normalize('He said , "It\'s AI ." ( Really ? )'), 'He said, "It\'s AI." (Really?)')
