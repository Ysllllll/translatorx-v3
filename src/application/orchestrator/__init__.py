"""Orchestration use cases — session + result types.

The legacy ``VideoOrchestrator`` / ``CourseOrchestrator`` /
``StreamingOrchestrator`` classes have been retired; batch and live
execution now live in :mod:`api.app.video` (VideoBuilder),
:mod:`api.app.course` (CourseBuilder), and :mod:`api.app.stream`
(LiveStreamHandle), all composed on top of
:class:`application.pipeline.runtime.PipelineRuntime`.
"""

from .session import VideoSession
from .video import VideoResult

__all__ = [
    "VideoSession",
    "VideoResult",
]
