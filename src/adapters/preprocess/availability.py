"""Availability guards for optional preprocessing dependencies.

Each helper returns ``True`` iff the corresponding third-party package
is importable. Callers use them to skip features gracefully instead of
hard-failing at import time; pytest fixtures use them to emit
``pytest.skip`` when a backend's dependency is missing.

Mirrors the pattern in :mod:`domain.lang._core._availability`.
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


__all__ = [
    "langdetect_is_available",
    "punc_model_is_available",
    "spacy_is_available",
]
