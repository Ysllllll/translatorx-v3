"""Orchestration use cases — video/session primitives.

The legacy ``CourseOrchestrator`` / ``StreamingOrchestrator`` have been
retired; multi-video and live-stream execution now live in
:mod:`api.app.course` (CourseBuilder) and :mod:`api.app.stream`
(StreamBuilder + LiveStreamHandle), which compose
:class:`application.pipeline.runtime.PipelineRuntime`.
"""

from .session import VideoSession
from .video import VideoOrchestrator, VideoResult

__all__ = [
    "VideoSession",
    "VideoOrchestrator",
    "VideoResult",
]
