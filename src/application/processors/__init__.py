"""Concrete :class:`ProcessorBase` implementations.

Each module exposes exactly one processor class. Stage 3.2c lands
:class:`TranslateProcessor`; Align / TTS land in Stage 6.
"""

from application.processors.align import AlignProcessor
from application.processors.summary import SummaryProcessor
from application.processors.translate import TranslateProcessor
from application.processors.tts import TTSProcessor

__all__ = ["AlignProcessor", "SummaryProcessor", "TranslateProcessor", "TTSProcessor"]
