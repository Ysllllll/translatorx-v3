"""Concrete :class:`ProcessorBase` implementations.

Each module exposes exactly one processor class. Stage 3.2c lands
:class:`TranslateProcessor`; Align / TTS land in Stage 6.
"""

from runtime.processors.summary import SummaryProcessor
from runtime.processors.translate import TranslateProcessor

__all__ = ["SummaryProcessor", "TranslateProcessor"]
