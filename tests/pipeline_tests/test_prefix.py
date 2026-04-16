"""Tests for PrefixHandler — strip and readd conversational prefixes."""

from __future__ import annotations

import pytest

from pipeline.config import PrefixRule, EN_ZH_PREFIX_RULES
from pipeline.prefix import PrefixHandler


@pytest.fixture
def handler() -> PrefixHandler:
    return PrefixHandler(EN_ZH_PREFIX_RULES)


class TestStripPrefix:
    def test_okay_comma(self, handler: PrefixHandler):
        text, prefix = handler.strip_prefix("Okay, let me explain.")
        assert text == "let me explain."
        assert prefix == "好的，"

    def test_okay_dot(self, handler: PrefixHandler):
        text, prefix = handler.strip_prefix("Okay. That's fine.")
        assert text == "That's fine."
        assert prefix == "好的。"

    def test_ok_comma(self, handler: PrefixHandler):
        text, prefix = handler.strip_prefix("Ok, sure thing.")
        assert text == "sure thing."
        assert prefix == "好的，"

    def test_ok_dot(self, handler: PrefixHandler):
        text, prefix = handler.strip_prefix("ok. Next step.")
        assert text == "Next step."
        assert prefix == "好的。"

    def test_ok_ellipsis(self, handler: PrefixHandler):
        text, prefix = handler.strip_prefix("OK... let me think.")
        assert text == "let me think."
        assert prefix == "好的。"

    def test_uh_comma(self, handler: PrefixHandler):
        text, prefix = handler.strip_prefix("Uh, what was that?")
        assert text == "what was that?"
        assert prefix == "呃，"

    def test_um_comma(self, handler: PrefixHandler):
        text, prefix = handler.strip_prefix("um, I think so.")
        assert text == "I think so."
        assert prefix == "嗯，"

    def test_case_insensitive(self, handler: PrefixHandler):
        text, prefix = handler.strip_prefix("OKAY, let's go.")
        assert text == "let's go."
        assert prefix == "好的，"

    def test_no_match(self, handler: PrefixHandler):
        text, prefix = handler.strip_prefix("Hello world.")
        assert text == "Hello world."
        assert prefix is None

    def test_prefix_only_no_remainder(self, handler: PrefixHandler):
        """If text is just the prefix with nothing after, don't strip."""
        text, prefix = handler.strip_prefix("Ok.")
        assert text == "Ok."
        assert prefix is None

    def test_whitespace_stripped(self, handler: PrefixHandler):
        text, prefix = handler.strip_prefix("  okay,   let me think.  ")
        assert text == "let me think."
        assert prefix == "好的，"

    def test_empty_string(self, handler: PrefixHandler):
        text, prefix = handler.strip_prefix("")
        assert text == ""
        assert prefix is None

    def test_empty_rules(self):
        handler = PrefixHandler(())
        text, prefix = handler.strip_prefix("Okay, test.")
        assert text == "Okay, test."
        assert prefix is None


class TestReaddPrefix:
    def test_readd(self, handler: PrefixHandler):
        result = handler.readd_prefix("让我想想。", "好的，")
        assert result == "好的，让我想想。"

    def test_readd_none(self, handler: PrefixHandler):
        result = handler.readd_prefix("你好世界。", None)
        assert result == "你好世界。"


class TestPrefixRule:
    def test_frozen(self):
        rule = PrefixRule("ok,", "好的，")
        with pytest.raises(AttributeError):
            rule.pattern = "test"  # type: ignore[misc]

    def test_custom_rules(self):
        rules = (
            PrefixRule("well,", "嗯，"),
            PrefixRule("yeah,", "是的，"),
        )
        handler = PrefixHandler(rules)
        text, prefix = handler.strip_prefix("Well, I think so.")
        assert text == "I think so."
        assert prefix == "嗯，"
