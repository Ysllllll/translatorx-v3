"""Availability guards for optional preprocessing dependencies.

Follows the same pattern as ``lang_ops._core._availability``.
"""


def punc_model_is_available() -> bool:
    """Check if ``deepmultilingualpunctuation`` is importable."""
    try:
        from deepmultilingualpunctuation import PunctuationModel  # noqa: F401

        return True
    except ImportError:
        return False


def spacy_is_available() -> bool:
    """Check if ``spacy`` is importable."""
    try:
        import spacy  # noqa: F401

        return True
    except ImportError:
        return False


def langdetect_is_available() -> bool:
    """Check if ``langdetect`` is importable."""
    try:
        import langdetect  # noqa: F401

        return True
    except ImportError:
        return False
