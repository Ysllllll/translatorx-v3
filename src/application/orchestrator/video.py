"""VideoResult — aggregate outcome of one video's pipeline run.

The legacy ``VideoOrchestrator`` class has been retired. All
batch / live execution flows through
:class:`application.pipeline.runtime.PipelineRuntime` (see
:class:`api.app.video.VideoBuilder`,
:class:`api.app.course.CourseBuilder`,
:class:`api.app.stream.LiveStreamHandle`). This module now only
exposes :class:`VideoResult` — the public dataclass used by those
higher-level builders.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from domain.model import SentenceRecord
from ports.errors import ErrorInfo


@dataclass(frozen=True)
class VideoResult:
    """Outcome of a :meth:`VideoBuilder.run` call.

    Attributes
    ----------
    records:
        Final enriched :class:`SentenceRecord` list in source order.
    stale_ids:
        Record ids flagged by any processor's ``output_is_stale``.
        The caller (App layer) decides whether to schedule a rerun.
    failed:
        :class:`ErrorInfo` entries collected during the run. For
        permanent failures the record is still yielded downstream
        but flagged.
    elapsed_s:
        Wall-clock seconds spent inside the run.
    """

    records: list[SentenceRecord] = field(default_factory=list)
    stale_ids: tuple[int, ...] = ()
    failed: tuple[ErrorInfo, ...] = ()
    elapsed_s: float = 0.0


__all__ = ["VideoResult"]
