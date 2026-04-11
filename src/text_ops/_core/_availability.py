"""Availability checks for optional CJK dependencies."""

from __future__ import annotations


def jieba_is_available() -> bool:
    try:
        import jieba  # noqa: F401
        return True
    except ImportError:
        return False


def mecab_is_available() -> bool:
    try:
        import MeCab  # noqa: F401
        return True
    except ImportError:
        return False


def kiwi_is_available() -> bool:
    try:
        from kiwipiepy import Kiwi  # noqa: F401
        return True
    except ImportError:
        return False
