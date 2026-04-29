"""Per-video Unit-of-Work aggregate (:class:`VideoSession`) + result type.

This package was previously named ``application.session``. It was
renamed to make its scope explicit: it does **not** orchestrate a
pipeline run (that is :class:`application.pipeline.PipelineRuntime`).
It owns the persistent state of *one* video — load, hydrate, accumulate
patches, flush.

Processors receive a :class:`VideoSession` via
:class:`~application.pipeline.PipelineContext` and call:

1. :meth:`VideoSession.load` — read raw video state from the store
   (idempotent; once per run).
2. :meth:`VideoSession.hydrate` — populate each
   :class:`SentenceRecord` with persisted translations / alignment / TTS
   metadata.
3. ``rec.set_translation(...)`` (etc.) — accumulate patches in memory.
4. :meth:`VideoSession.flush` — durably persist patches to the store
   (typically wrapped in :func:`asyncio.shield` by the pipeline so
   in-flight work survives cancellation).

:class:`VideoResult` is the public dataclass returned by builders
(``VideoBuilder.run``, ``CourseBuilder.run`` per video).
"""

from .session import VideoSession
from .video import VideoResult

__all__ = ["VideoSession", "VideoResult"]
