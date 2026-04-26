"""Stage adapters — wrap legacy components in Stage Protocol.

Phase 1 / Step 4: thin adapters for ``build`` and ``structure`` tiers.

* :mod:`application.stages.build` — :class:`FromSrtStage`,
  :class:`FromWhisperxStage`, :class:`FromPushStage`.
* :mod:`application.stages.structure` — :class:`PuncStage`,
  :class:`ChunkStage`, :class:`MergeStage`.

Adapters take a Pydantic ``Params`` model and a closure-bound legacy
component (e.g. an :class:`~adapters.preprocess.PuncRestorer`-derived
``ApplyFn``). Use :func:`make_default_registry` to assemble a populated
:class:`~application.pipeline.registry.StageRegistry` for an :class:`App`.
"""

from .build import (
    FromAudioStage,
    FromPushStage,
    FromSrtStage,
    FromWhisperxStage,
)
from .enrich import SummaryStage, TranslateStage
from .registry import make_default_registry
from .structure import ChunkStage, MergeStage, PuncStage

__all__ = [
    "ChunkStage",
    "FromAudioStage",
    "FromPushStage",
    "FromSrtStage",
    "FromWhisperxStage",
    "MergeStage",
    "PuncStage",
    "SummaryStage",
    "TranslateStage",
    "make_default_registry",
]
